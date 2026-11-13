"""Validate reviewed soma centers and audit their mapping onto a fixed APP2 tree."""
import numpy as np


def validate_context(context, shape_zyx, target_zyx):
    coordinates = np.asarray(context['centers_zyx'], dtype=float)
    radii = context['radii']
    ids = context['seed_ids']
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or not len(coordinates):
        raise ValueError('Expected nonempty ZYX centers')
    if len(radii) != len(coordinates) or len(ids) != len(coordinates):
        raise ValueError('Seed fields differ in length')
    if not all(isinstance(i, str) and i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Invalid or duplicate seed IDs')
    if not np.all(np.isfinite(coordinates) & (coordinates >= 0) & (coordinates < shape_zyx)):
        raise ValueError('Soma outside volume')
    if len(set(map(tuple, coordinates))) != len(coordinates):
        raise ValueError('Repeated soma coordinates')
    if ids.count('T') != 1 or not np.array_equal(coordinates[ids.index('T')], target_zyx):
        raise ValueError('Known target mismatch')
    if not np.array_equal(context['target_zyx'], target_zyx):
        raise ValueError('Context target mismatch')
    if any(r is not None and (not np.isfinite(r) or r <= 0) for r in radii):
        raise ValueError('Invalid measured radius')
    return coordinates.tolist(), {tuple(c):float('inf') if r is None else float(r)
                                for c,r in zip(coordinates,radii)}, list(ids)


def matching_audit(tree, coords, radii, ids=None, max_tolerance=12.):
    points = np.asarray([r[2:5] for r in tree], dtype=float)
    used = {}
    records = []
    for i, center in enumerate(coords):
        distances = np.linalg.norm(points - np.asarray(center)[::-1], axis=1)
        index = int(np.argmin(distances))
        node = int(tree[index][0])
        tolerance = min(max_tolerance, float(radii.get(tuple(center), max_tolerance))+2.)
        accepted = bool(distances[index] <= tolerance)
        duplicate = used.get(node) if accepted else None
        seed_id = ids[i] if ids is not None else f'legacy_{i}'
        status = 'same_app2_node' if duplicate is not None else ('matched' if accepted else 'unmatched')
        records.append(dict(seed_id=seed_id,center_zyx=list(map(float,center)),
                            nearest_node_id=node,nearest_xyz=points[index].tolist(),
                            distance_um=float(distances[index]),tolerance_um=tolerance,
                            status=status,shared_with=duplicate))
        if accepted and duplicate is None:
            used[node] = seed_id
    return dict(records=records,unique_matched_nodes=list(used),
                note='Existing APP2 nodes only; unmatched somas do not create new paths')
