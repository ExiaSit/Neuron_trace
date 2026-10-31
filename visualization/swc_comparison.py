"""Reusable version comparison with a common SWC XY box and full image Z."""
from pathlib import Path
import numpy as np
from visualization.swc_view import SwcMipView


def render_swc_comparison(raw, swcs, output, title='SWC comparison'):
    """swcs: sequence of (label, SWC path); coordinates are 1 um image XYZ."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from visualization.diagnostics import save_figure
    trees = [np.loadtxt(path, comments='#', ndmin=2) for _, path in swcs]
    view = SwcMipView(raw.shape, np.concatenate([a[:, 2:5] for a in trees]))
    fig, axes = plt.subplots(len(swcs), 3, figsize=(16, 4*len(swcs)),
                             layout='constrained', squeeze=False)
    colors = ['#ffcc55', '#44ddff', '#ff7799', '#88ee77']
    for row, ((label, _), nodes) in enumerate(zip(swcs, trees)):
        by = {int(n[0]): n[2:5] for n in nodes}
        edges = np.array([[n[2:5], by[int(n[6])]] for n in nodes
                          if int(n[6]) in by]).reshape(-1, 2, 3)
        for axis, (xy, plane) in enumerate([([0, 1], 'XY'), ([0, 2], 'XZ'), ([1, 2], 'YZ')]):
            ax = axes[row, axis]; bg = view.project(raw, axis)
            ax.imshow(bg, cmap='gray', vmin=0, vmax=max(25, float(np.percentile(bg, 99.5))),
                      extent=view.extent(axis))
            ax.add_collection(LineCollection(edges[:, :, xy], colors=colors[row % len(colors)], linewidths=.8))
            view.apply(ax, axis)
            ax.set_title(label+' | '+plane); ax.axis('off')
    fig.suptitle(title+'\nCommon SWC XY bounding box + 15 um | full original Z')
    save_figure(fig, Path(output))


if __name__ == '__main__':
    import argparse
    import tifffile
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', required=True)
    parser.add_argument('--swc', action='append', required=True)
    parser.add_argument('--label', action='append', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if len(args.swc) != len(args.label):
        parser.error('Each SWC needs one label')
    render_swc_comparison(tifffile.imread(args.raw), list(zip(args.label, args.swc)), args.output)
