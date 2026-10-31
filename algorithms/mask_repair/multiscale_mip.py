"""Depth-preserving, overlapping slab projections in all three array axes.

Projections nominate measured 3-D seeds, never solid columns. Repeated windows
of one thickness count only once; agreement from independent views is required.
The returned guide is used for routing, not as the image or as a binary mask.
"""
import json
import os
import numpy as np
from utils.gpu_ndimage import ndi

DEFAULT_CONFIG = dict(mode='multiscale', thicknesses=[8, 16, 32],
                      overlap=.5, min_axes=2)


def configuration(value=None):
    if value is None:
        value = json.loads(os.environ.get('RECONNECT_MIP_CONFIG', '{}'))
    if not isinstance(value, dict):
        raise ValueError('mask_mip must be an object')
    unknown = set(value) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f'Unknown mask_mip settings: {sorted(unknown)}')
    cfg = dict(DEFAULT_CONFIG, **value)
    if cfg['mode'] not in ('legacy', 'multiscale'):
        raise ValueError('mask_mip.mode must be legacy or multiscale')
    widths = cfg['thicknesses']
    if (not isinstance(widths, (list, tuple)) or not widths or
            any(type(w) is not int or w < 2 for w in widths)):
        raise ValueError('mask_mip.thicknesses must contain integer widths >=2')
    cfg['thicknesses'] = sorted(set(widths))
    overlap = cfg['overlap']
    if isinstance(overlap, bool) or not isinstance(overlap, (int, float)) or not 0 < overlap < 1:
        raise ValueError('mask_mip.overlap must be between 0 and 1, exclusively')
    if type(cfg['min_axes']) is not int or not 2 <= cfg['min_axes'] <= 3:
        raise ValueError('mask_mip.min_axes must be 2 or 3')
    return cfg


def window_ranges(length, thickness, overlap=.5):
    """Fixed-width windows, with an end-aligned final window and no uncovered edge."""
    if length < 1 or thickness < 1 or not 0 < overlap < 1:
        raise ValueError('Invalid slab geometry')
    width = min(length, thickness)
    stride = max(1, int(np.floor(width * (1 - overlap))))
    starts = list(range(0, length-width+1, stride))
    if starts[-1] != length-width:
        starts.append(length-width)
    return [(s, s+width) for s in starts]


def supported_maxima(volume, contrast):
    """Maxima supported by >=2 neighboring columns at depth +/-2 voxels.

Each XY column gets at most one vote from each of its eight neighbors. Depth
is the measured argmax within this slab, including the slab's global offset
when scattered by the caller. This does not assume an XY projection direction.
"""
    depth = volume.argmax(axis=0)
    row, col = np.indices(depth.shape)
    valid = contrast[depth, row, col] > 2
    votes = np.zeros(depth.shape, np.uint8)
    height, width = depth.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == dx == 0:
                continue
            target = (slice(max(0, -dy), min(height, height-dy)),
                      slice(max(0, -dx), min(width, width-dx)))
            source = (slice(max(0, dy), min(height, height+dy)),
                      slice(max(0, dx), min(width, width+dx)))
            votes[target] += valid[source] & (np.abs(depth[target]-depth[source]) <= 2)
    return depth, valid & (votes >= 2)


def multiscale_seeds(raw, contrast, cfg):
    """Bounded memory: keep per-axis scale counts, never a stack of projections."""
    axis_support = np.zeros(raw.shape, np.uint8)
    records = []
    for axis, plane in enumerate(('XY', 'XZ', 'YZ')):
        volume = np.moveaxis(raw, axis, 0)
        local_contrast = np.moveaxis(contrast, axis, 0)
        # Full-volume projection is one additional scale; short dimensions do
        # not get repeated copies of an identical clipped thickness.
        widths = sorted({min(w, len(volume)) for w in cfg['thicknesses']} | {len(volume)})
        scale_count = np.zeros(raw.shape, np.uint8)
        scale_records = []
        for width in widths:
            flags = np.zeros(raw.shape, bool)
            local_flags = np.moveaxis(flags, axis, 0)
            windows = window_ranges(len(volume), width, cfg['overlap'])
            for start, stop in windows:
                depth, valid = supported_maxima(volume[start:stop], local_contrast[start:stop])
                row, col = np.nonzero(valid)
                local_flags[depth[row, col]+start, row, col] = True
            scale_count += flags
            scale_records.append(dict(thickness=width,
                                      stride=max(1, int(np.floor(width*(1-cfg['overlap'])))),
                                      windows=[[a, b] for a, b in windows],
                                      seed_voxels=int(flags.sum())))
        axis_support += scale_count > 0
        records.append(dict(axis_zyx=axis, plane=plane, scales=scale_records,
                            seed_voxels=int(np.count_nonzero(scale_count)),
                            multiple_scale_voxels=int(np.count_nonzero(scale_count >= 2))))
    seeds = axis_support >= cfg['min_axes']
    return seeds, dict(config=cfg, axes=records, consensus_seed_voxels=int(seeds.sum()),
                       axis_support_counts={str(i): int(np.count_nonzero(axis_support == i))
                                            for i in range(4)})


def reinforce(raw, contrast, legacy_guide, legacy_classes, cfg):
    seeds, report = multiscale_seeds(raw, contrast, cfg)
    values = np.where(seeds, np.maximum(contrast, 0), 0).astype(np.float32)
    peak = ndi.maximum_filter(values, size=3)
    near = ndi.maximum_filter(seeds.astype(np.uint8), size=3) > 0
    eligible = near & (contrast > .75)
    # Preserve the established one-voxel radius and maximum +3 contrast boost.
    boost = np.minimum(peak*.5, np.maximum(contrast, 0)+3)
    guide = legacy_guide.copy()
    guide[eligible] = np.maximum(guide[eligible], boost[eligible])
    classes = legacy_classes.copy()
    classes[(guide > legacy_guide) & (classes == 0)] = 3
    classes[seeds & (classes == 0)] = 4
    report.update(additional_reinforced_voxels=int(np.count_nonzero(guide > legacy_guide)),
                  measured_contrast_min=.75, neighborhood_radius=1, max_contrast_boost=3.,
                  original_guide_preserved=True)
    return guide, classes, report
