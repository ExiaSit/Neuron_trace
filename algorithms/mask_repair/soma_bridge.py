"""Endpoint-to-thick-body candidates in TIFF voxel coordinates."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from utils import bootstrap as _bootstrap

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.graph import route_through_array


def candidates(raw, labels, tips, max_gap=40):
    # A thick interior is a soma proxy, not a biological soma classification.
    distance = ndi.distance_transform_edt(labels > 0)
    cores, count = ndi.label(distance >= 3)
    result, bodies = [], []
    for core_id, box in enumerate(ndi.find_objects(cores), 1):
        core = cores[box] == core_id
        if core.sum() < 80:
            continue
        points_core = np.argwhere(core).astype(float)
        eig = np.linalg.eigvalsh(np.cov(points_core, rowvar=False))
        aspect = float(np.sqrt(eig[-1] / max(eig[0], 1e-9)))
        if aspect > 3:
            continue  # Thick but strongly elongated cores are not soma-like.
        origin = np.array([s.start for s in box])
        center = np.argwhere(core).mean(0) + origin
        cid = int(labels[tuple((np.argwhere(core)[0] + origin).T)])
        lo = np.maximum(0, origin - 12)
        hi = np.minimum(labels.shape, np.array([s.stop for s in box]) + 12)
        sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        body = labels[sl] == cid
        local_core = cores[sl] == core_id
        near_core = ndi.distance_transform_edt(~local_core) <= 5
        surface = body & ~ndi.binary_erosion(body) & near_core
        points = np.argwhere(surface) + lo
        if not len(points):
            continue
        bodies.append(dict(component=cid, center_zyx=center.tolist(), core_voxels=int(core.sum()), core_axis_ratio=aspect))
        tree = cKDTree(points)
        best = {}
        for source, p, u in tips:
            if source == cid:
                continue
            gap, idx = tree.query(p)
            if gap < 1 or gap > max_gap:
                continue
            q = points[idx]
            angle = float(np.degrees(np.arccos(np.clip(u @ ((q-p)/gap), -1, 1))))
            if angle > 60:
                continue
            key = (source, tuple(p))
            if key in best:
                continue
            lower = np.maximum(0, np.minimum(p, q).astype(int)-8)
            upper = np.minimum(raw.shape, np.maximum(p, q).astype(int)+9)
            region = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
            patch = raw[region].astype(float)
            contrast = ndi.gaussian_filter(patch, .6) - ndi.gaussian_filter(patch, 4)
            start, end = np.rint(p-lower).astype(int), q-lower
            line = np.ones(patch.shape, bool)
            coords = np.rint(np.linspace(start, end, int(gap*3)+2)).astype(int)
            line[tuple(coords.T)] = False
            tube = ndi.distance_transform_edt(line) <= max(4, min(8, gap*.2))
            cost = 1 + 16/(1+np.maximum(contrast, 0))
            local_labels = labels[region]
            allowed = tube & ((local_labels == 0) | (local_labels == source) | (local_labels == cid))
            cost[~allowed] = np.inf
            try:
                path, total = route_through_array(cost, tuple(start), tuple(end), fully_connected=True)
            except ValueError:
                continue
            if not np.isfinite(total):
                continue
            path = np.asarray(path)
            # Assess the gap itself; bright soma/branch endpoints cannot inflate support.
            gap_points = local_labels[tuple(path.T)] == 0
            values = contrast[tuple(path.T)][gap_points]
            if not len(values):
                continue
            supported = values > 1.5
            run = dark = 0
            for good in supported:
                run = 0 if good else run+1
                dark = max(dark, run)
            length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
            if supported.mean() < .8 or dark > 3 or length > 1.8*gap:
                continue
            metrics = dict(gap=float(gap), source_angle=angle,
                           signal_fraction=float(supported.mean()), max_dark_samples=dark,
                           path_length=length, path_method='raw_guided_endpoint_to_soma_surface',
                           path_zyx=(path+lower).tolist(), soma_center_zyx=center.tolist())
            best[key] = dict(components=sorted([source, cid]), metrics=metrics,
                             voxels=path+lower, kind='soma_surface')
        result.extend(best.values())
    return result, bodies


def solid_bridge(path, shape):
    """One-voxel axial radius makes 26-neighbor paths face-connected.

    This is a minimum bridge thickness, not a measured neurite radius.
    """
    path = np.asarray(path, dtype=int)
    offsets = np.vstack((np.zeros((1, 3), int), np.eye(3, dtype=int), -np.eye(3, dtype=int)))
    voxels = (path[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    inside = np.all((voxels >= 0) & (voxels < np.asarray(shape)), axis=1)
    return np.unique(voxels[inside], axis=0)
