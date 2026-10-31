"""Reject soma-adjacent dead ends, with independent neurite evidence protection.

Coordinates and lengths are in the pipeline's isotropic 1-um image space.
This does not alter the soma, skeleton, or any path leading to a distant tip.
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from algorithms.tracing.signal_support import tree_paths, resample_line
from algorithms.tracing.repair_evidence import sample_mask, tube_support


def prune_soma_terminals(tree, evidence, soma, min_external_length_um=7.75):
    rows = [list(n) for n in tree]
    audit = dict(rule='soma_adjacent_dead_end', decisions=[], removed_ids=[],
                 before_nodes=rows,
                 capture_point='after other evidence pruning, before soma terminals and final length pruning',
                 parameters=dict(external_length_limit_um=min_external_length_um,
                                 support_fraction=.8, duplicate_distance_um=2.,
                                 signal_valley_ratio=.8, sampling_step_um=.25))
    if soma.shape != evidence.raw.shape:
        raise ValueError('Soma and raw must share 1-um ZYX coordinates')
    # Evaluate the topology that the existing final length rule will expose;
    # tiny side twigs must not hide an otherwise terminal soma spur. Apply
    # deletions to the full tree; the length rule still writes last.
    from algorithms.tracing.terminal_pruning import prune_final_short_leaves
    preview, _ = prune_final_short_leaves(rows)
    nodes, children, paths = tree_paths(preview)
    roots = [n for n in preview if int(n[6]) not in nodes]
    if len(roots) != 1 or not np.any(soma):
        return rows, set(), dict(audit, status='no_unique_target_soma')
    root = np.asarray(roots[0][2:5])[None, ::-1]
    if not sample_mask(soma, root)[0]:
        return rows, set(), dict(audit, status='root_outside_target_soma')

    soma_kd = cKDTree(np.argwhere(soma))
    candidates = []
    witnesses = []
    for path in paths:
        p = resample_line(np.array([nodes[i][2:5] for i in path])[:, ::-1], .25)
        ds = np.linalg.norm(np.diff(p, axis=0), axis=1)
        outside_edges = ~sample_mask(soma, (p[1:]+p[:-1])/2)
        exterior = float(ds[outside_edges].sum())
        terminal = not children[path[-1]]
        if (terminal and sample_mask(soma, p[:1])[0]
                and exterior < min_external_length_um
                and not any(int(nodes[i][1]) == 1 for i in path[1:])):
            candidates.append((path, p, exterior, float(ds.sum())))
        # A witness must be a path that this rule and final short-leaf pruning
        # will retain. Internal stems alone are not sufficient evidence.
        elif terminal and exterior >= min_external_length_um:
            chain = [path[-1]]
            while int(nodes[chain[-1]][6]) in nodes:
                chain.append(int(nodes[chain[-1]][6]))
            q = resample_line(np.array([nodes[i][2:5] for i in chain])[:, ::-1], .25)
            witnesses.extend(q[~sample_mask(soma, q)])
    witness_kd = cKDTree(witnesses) if len(witnesses) else None
    removed = set()
    for path, p, exterior, length in candidates:
        record = dict(tip_id=path[-1], attachment_id=path[0],
                      length_um=length, exterior_length_um=exterior,
                      node_ids=path[1:], decision='keep')
        outside = ~sample_mask(soma, p)
        if not outside.any():
            reason = 'terminal_entirely_inside_target_soma'
        else:
            # Exclude the boundary's interpolation band when enough samples
            # remain; do not infer a lack of support from an empty selection.
            selection = soma_kd.query(p)[0] > 1.
            if selection.sum() < 4:
                selection = outside
            safe = np.all((p[selection] >= 7) &
                          (p[selection] < np.array(soma.shape)-8))
            if not safe:
                record['reason'] = 'image_edge_uncertain'
                audit['decisions'].append(record)
                continue
            support = tube_support(evidence, p)[:, selection].mean(axis=1)
            record['tube_support_by_radius_2_3_4_6'] = support.tolist()
            reason = None
            if support.max() < .8:
                reason = 'short_exterior_without_independent_tube'
            elif witness_kd is not None and support[:2].max() < .8:
                q = p[selection]
                distance, nearest = witness_kd.query(q)
                record['max_distance_to_retained_path_um'] = float(distance.max())
                if np.all(distance <= 2.):
                    other = witness_kd.data[nearest]
                    samples = np.array([q*(1-t)+other*t for t in np.linspace(0, 1, 7)])
                    intensity = ndi.map_coordinates(evidence.smooth,
                        samples.reshape(-1, 3).T, order=1, mode='nearest').reshape(7, -1)
                    ends = np.minimum(intensity[0], intensity[-1])
                    same_ridge = (ends >= 10) & (intensity.min(0) >= .8*ends)
                    record['same_ridge_fraction'] = float(same_ridge.mean())
                    if same_ridge.mean() >= .8:
                        reason = 'short_duplicate_of_retained_signal_path'
            if reason is None:
                record['reason'] = 'independent_tube_or_uncertain_duplicate'
                audit['decisions'].append(record)
                continue
        removed.update(path[1:])
        record.update(decision='remove', reason=reason)
        audit['decisions'].append(record)
    # Include the tiny side twigs omitted by the preview, avoiding orphan nodes.
    original_nodes, original_children, _ = tree_paths(rows)
    stack = list(removed)
    while stack:
        for child in original_children[stack.pop()]:
            if child not in removed:
                removed.add(child)
                stack.append(child)
    if any(int(original_nodes[i][1]) == 1 or
           int(original_nodes[i][6]) not in original_nodes for i in removed):
        raise ValueError('Soma terminal rule would remove a protected root/soma')
    # One pass only: no cascading reinterpretation of internal branches.
    audit.update(removed_ids=sorted(removed), status='pruned' if removed else 'no_candidates')
    return [n for n in rows if int(n[0]) not in removed], removed, audit


def prune_with_context(tree, evidence, context):
    if not context or not context.get('measured_soma_path'):
        return tree, set(), dict(rule='soma_adjacent_dead_end', status='no_soma_context',
                                decisions=[], removed_ids=[])
    import tifffile
    soma = tifffile.imread(context['measured_soma_path']) > 0
    return prune_soma_terminals(tree, evidence, soma)
