"""Persistent diagnostics; none of these functions modify reconstruction inputs."""
from pathlib import Path
import os
import tempfile
from functools import lru_cache
from typing import NamedTuple
import numpy as np

# Retain the old eighth title for explicitly replaying historical h snapshots.
MASK_TITLES = ['Skeleton and direction', 'Signal reconnect', 'Soma completion',
               'Soma-neurite reconnect', 'Connection thickness', 'Soma-root repair',
               'Guarded continuation', 'Root geodesic supplement']


class _ProjectedRaw(NamedTuple):
    shape: tuple
    projections: tuple

    def max(self, axis):
        return self.projections[axis]


@lru_cache(maxsize=1)
def _raw_projections(path, size, mtime_ns):
    """Retain only three small projections, not the full raw image per worker."""
    import tifffile
    raw=tifffile.imread(path)
    return _ProjectedRaw(raw.shape,tuple(raw.max(axis) for axis in range(3)))


def save_figure(fig, path):
    import matplotlib.pyplot as plt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.png', delete=False) as f:
        tmp = Path(f.name)
    try:
        fig.savefig(tmp, dpi=140)
        os.replace(tmp, path)
    finally:
        plt.close(fig)
        tmp.unlink(missing_ok=True)


def mask_delta(before, after):
    before, after = np.asarray(before, dtype=bool), np.asarray(after, dtype=bool)
    if before.shape != after.shape:
        raise ValueError('Mask snapshot shapes differ')
    return after & ~before, before & ~after


def mask_transition(raw, before, after, path, title, sources=None, provenance='recorded'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from utils.case_worker import save_json
    before, after = before > 0, after > 0
    if raw.shape != before.shape:
        raise ValueError('Raw and mask shapes differ')
    added, removed = mask_delta(before, after)
    counts = dict(before_voxels=int(before.sum()), after_voxels=int(after.sum()),
                  added_voxels=int(added.sum()), removed_voxels=int(removed.sum()))
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for axis, plane in enumerate(['XY', 'XZ', 'YZ']):
        bg = raw.max(axis)
        vmax = max(25, float(np.percentile(bg, 99.5)))
        projected_before=before.any(axis);projected_after=after.any(axis)
        projected_added=added.any(axis);projected_removed=removed.any(axis)
        for col, (projection, label) in enumerate([(projected_before, 'Before'), (projected_after, 'After'), (projected_before, 'Changes')]):
            ax = axes[axis, col]
            ax.imshow(bg, cmap='gray', vmin=0, vmax=vmax)
            rgba = np.zeros((*bg.shape, 4), dtype=np.float32)
            rgba[projection] = [0, 1, 1, .55]
            if col == 2:
                rgba[projected_added] = [1, .1, .05, .95]
                rgba[projected_removed] = [1, 1, 0, .95]
            ax.imshow(rgba)
            ax.set_title(f'{plane} | {label}')
            ax.axis('off')
    fig.suptitle(f'{title} | +{counts["added_voxels"]:,} / -{counts["removed_voxels"]:,} voxels\n'
                 f'Cyan: mask before/after | Red: added | Yellow: removed | {provenance}')
    fig.tight_layout(rect=(0, 0, 1, .95))
    save_figure(fig, path)
    save_json(Path(path).with_suffix('.json'), dict(title=title, provenance=provenance,
              sources=sources or {}, **counts))
    return counts


def record_mask_step(spec, index, before_path, after_path, provenance='recorded'):
    import tifffile
    root = Path(spec['config']['output_root'])
    path = root/'mask_step_mips'/f'{spec["id"]}_{index+1:02d}_{"abcdefgh"[index]}.png'
    raw_path=Path(spec['raw']).resolve();stamp=raw_path.stat()
    raw=_raw_projections(str(raw_path),stamp.st_size,stamp.st_mtime_ns)
    return mask_transition(raw, tifffile.imread(before_path),
                           tifffile.imread(after_path), path, f'{spec["id"]} | {index+1:02d} {MASK_TITLES[index]}',
                           dict(before=str(before_path), after=str(after_path)), provenance)


def short_leaf_figure(raw, audit, path, provenance='recorded'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    nodes = {n[0]: n for n in audit['before_nodes']}
    removed = set(audit['removed_ids'])
    from visualization.swc_view import SwcMipView
    view = SwcMipView(raw.shape, [n[2:5] for n in nodes.values()])
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    for ax, axis, plane, coords in zip(axes, range(3), ['XY', 'XZ', 'YZ'], [(2, 3), (2, 4), (3, 4)]):
        bg = view.project(raw, axis)
        ax.imshow(bg, cmap='gray', vmin=0, vmax=max(25, float(np.percentile(bg, 99.5))), extent=view.extent(axis))
        for deleted, color, width in [(False, 'cyan', 1), (True, 'red', 2)]:
            segments = [[[n[c] for c in coords], [nodes[n[6]][c] for c in coords]]
                        for n in nodes.values() if n[6] in nodes and (n[0] in removed) == deleted]
            ax.add_collection(LineCollection(segments, colors=color, linewidths=width))
        tips = [nodes[n] for n in sorted(removed)]
        if tips:
            ax.scatter([n[coords[0]] for n in tips], [n[coords[1]] for n in tips], c='red', marker='x', s=22)
        view.apply(ax, axis)
        ax.set_title(plane)
        ax.axis('off')
    threshold = (f'<{audit["min_leaf_branch_length_um"]:g} um path length'
                 if 'min_leaf_branch_length_um' in audit else
                 f'<{audit.get("min_leaf_branch_length", 2):g} nodes')
    fig.suptitle(f'G-Cut short-leaf removal: {len(nodes)} -> {len(audit["after_nodes"])} nodes; '
                 f'{len(removed)} removed\nCyan: retained | Red + crosses: removed | '
                 f'{threshold}, soma protected\nAfter soma contraction, before segmentation | {provenance}')
    fig.tight_layout(rect=(0, 0, 1, .88))
    save_figure(fig, path)


def write_short_leaf_audit(audit, raw, root, cid, provenance='recorded'):
    from utils.case_worker import save_json
    root = Path(root)/'gcut_audits'
    root.mkdir(parents=True, exist_ok=True)
    audit = dict(audit, provenance=provenance)
    save_json(root/f'{cid}_short_leaves.json', audit)
    for label in ['before', 'after']:
        (root/f'{cid}_short_leaves_{label}.swc').write_text(
            '# image coordinates; after soma contraction\n'+''.join(' '.join(map(str, n))+'\n' for n in audit[label+'_nodes']))
    short_leaf_figure(raw, audit, root/f'{cid}_short_leaves.png', provenance)
