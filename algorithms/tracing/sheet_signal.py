"""Conservative 3-D cross-section evidence against tracking a bright sheet.

Unlike a dark gap, a sheet is bright at the center but has no transverse
localization along one axis. No absolute Z plane or branch-angle cutoff is used.
All coordinates and lengths here assume the pipeline's 1-um image grid.
"""
import numpy as np
from scipy import ndimage as ndi

from algorithms.tracing.signal_support import resample_line, tree_paths


def completed_soma_protection(archive, shape, fallback):
    """Use shape-validated f somas when present, retaining legacy fallback.

    The older raw-brightness core can include the surface sheet itself. Do not
    change that protection for existing rules; only this new rule uses the
    measured completed soma regions with a 2-um safety margin.
    """
    from pathlib import Path
    if not Path(archive).is_file():
        return fallback, 'legacy_confirmed_soma_fallback'
    mask = np.zeros(shape, bool)
    with np.load(archive) as data:
        if not data.files:
            return fallback, 'legacy_confirmed_soma_fallback'
        for key in data.files:
            q = np.asarray(data[key])
            if (q.ndim != 2 or q.shape[1] != 3 or not np.isfinite(q).all()
                    or not np.equal(q, np.rint(q)).all() or np.any(q < 0)
                    or np.any(q >= np.asarray(shape))):
                raise ValueError('Invalid completed soma coordinates')
            mask[tuple(q.astype(int).T)] = True
    grid = np.indices((5, 5, 5))-2
    mask = ndi.binary_dilation(mask, structure=(grid*grid).sum(axis=0) <= 4)
    return mask, 'shape_validated_completed_somas_plus_2um'


def sheet_profile(evidence, points):
    """Require a broad bright axis AND a dark side on the perpendicular axis in 3-D.

    At +/-4,6,8 um the broad axis must retain 80% of the brightest center found
    within one voxel. At least one side of the perpendicular axis at 8 um must be below 40%
    (a surface sheet can have bright tissue immediately below it).
    Sixteen angular samples give eight opposite pairs. The same direction must
    meet all radii; a high ring median is not sufficient evidence for a sheet.
    """
    p = np.asarray(points, float)
    if len(p) < 2:
        return dict(sheet=np.zeros(len(p), bool), broad_ratio=np.zeros(len(p)),
                    normal_ratio=np.ones(len(p)), center=np.zeros(len(p)))
    t = np.gradient(p, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1)[:, None], 1e-8)
    ref = np.tile([1., 0., 0.], (len(p), 1)); ref[np.abs(t[:, 0]) > .9] = [0, 1, 0]
    u = np.cross(t, ref); u /= np.maximum(np.linalg.norm(u, axis=1)[:, None], 1e-8)
    v = np.cross(t, u)
    center = evidence.profile(p, center_search=True)['center']
    rings = []
    for radius in (4., 6., 8.):
        rings.append(np.array([
            ndi.map_coordinates(evidence.smooth,
                                (p+radius*(np.cos(a)*u+np.sin(a)*v)).T,
                                order=1, mode='nearest')
            for a in np.linspace(0, 2*np.pi, 16, endpoint=False)]))
    bright = np.minimum.reduce([np.minimum(r[:8], r[8:]) for r in rings])
    dim = np.minimum(rings[-1][:8], rings[-1][8:])[(np.arange(8)+4) % 8]
    broad = bright / np.maximum(center, 1.)
    normal = dim / np.maximum(center, 1.)
    valid_axis = (broad >= .8) & (normal <= .4)
    selected = np.argmax(np.where(valid_axis, broad, -1), axis=0)
    col = np.arange(len(p))
    inside = np.all((p >= 8) & (p <= np.array(evidence.raw.shape)-1-8), axis=1)
    return dict(sheet=valid_axis.any(axis=0) & (center >= 20) & inside,
                broad_ratio=broad[selected, col], normal_ratio=normal[selected, col], center=center)


def prune_sheet_supported(tree, evidence, soma_xyz, protect_radius=12., protected_mask=None):
    """Cut after >=6 um of sustained sheet evidence, outside protected somas.

    Short crossings, beading, dark signals and steep-but-tubular branches do not
    qualify. A downstream protected soma vetoes a cut. This is still heuristic:
    a genuine very wide, flattened process can resemble a sheet.
    """
    nodes, children, paths = tree_paths(tree)
    if not nodes:
        return list(tree), set(), []
    protected = set()
    for nid, node in nodes.items():
        xyz = np.asarray(node[2:5], float)
        q = np.rint(xyz[::-1]).astype(int)
        if (int(node[1]) == 1 or int(node[6]) not in nodes
                or np.linalg.norm(xyz-soma_xyz) <= protect_radius
                or (protected_mask is not None and np.all(q >= 0)
                    and np.all(q < protected_mask.shape) and protected_mask[tuple(q)])):
            protected.add(nid)
    # All ancestors of protected nodes are protected against subtree deletion.
    ancestors = set(protected)
    for nid in protected:
        parent = int(nodes[nid][6])
        while parent in nodes and parent not in ancestors:
            ancestors.add(parent); parent = int(nodes[parent][6])
    candidates = []
    for path in paths:
        xyz = np.asarray([nodes[i][2:5] for i in path], float)
        edge = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        if edge.sum() < 6:
            continue
        p = resample_line(xyz[:, ::-1], .75)
        stats = sheet_profile(evidence, p)
        good = stats['sheet'].copy()
        good &= np.linalg.norm(p-np.asarray(soma_xyz)[::-1], axis=1) > protect_radius
        if protected_mask is not None:
            q = np.rint(p).astype(int)
            valid = np.all(q >= 0, axis=1) & np.all(q < protected_mask.shape, axis=1)
            good[valid] &= ~protected_mask[tuple(q[valid].T)]
        distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        cumulative = np.r_[0., np.cumsum(edge)]
        start = None
        for j, hit in enumerate(good):
            if hit and start is None:
                start = j
            if start is None or (hit and j != len(good)-1):
                continue
            stop = j if not hit else j+1
            length = distance[stop-1]-distance[start]
            if length >= 6.:
                k = min(len(path)-1, max(1, int(np.searchsorted(cumulative, distance[start]+1.))))
                cut = path[k]
                if cut not in ancestors:
                    candidates.append(dict(node=int(cut), reason='sustained_sheet_like_signal',
                                           xyz=list(map(float, nodes[cut][2:5])),
                                           evidence_start_xyz=p[start, ::-1].tolist(),
                                           evidence_end_xyz=p[stop-1, ::-1].tolist(),
                                           sheet_length_um=float(length),
                                           median_center=float(np.median(stats['center'][start:stop])),
                                           median_broad_ratio=float(np.median(stats['broad_ratio'][start:stop])),
                                           median_normal_ratio=float(np.median(stats['normal_ratio'][start:stop]))))
                    break
            start = None
    # Report only the first selected cut on each root-to-tip route.
    by_cut = {c['node']: c for c in candidates}
    reports = []
    for cut, record in by_cut.items():
        parent = int(nodes[cut][6]); covered = False
        while parent in nodes:
            if parent in by_cut:
                covered = True; break
            parent = int(nodes[parent][6])
        if not covered:
            reports.append(record)
    removed = set(); stack = [r['node'] for r in reports]
    while stack:
        nid = stack.pop()
        if nid in removed:
            continue
        removed.add(nid); stack.extend(children[nid])
    assert not (removed & protected)
    return [n for n in tree if int(n[0]) not in removed], removed, reports
