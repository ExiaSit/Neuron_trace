from pathlib import Path
import numpy as np
import tifffile


def paths(spec):
    from utils.case_worker import paths as get
    return get(spec)


def check_swc(path):
    from utils.case_worker import check_swc as check
    return check(path)


def comparison(spec, failure=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from visualization.previous_app2 import reference_nodes
    from visualization.diagnostics import save_figure
    from utils.case_worker import save_json
    p = paths(spec)
    raw = tifffile.imread(p['raw'])
    before = tifffile.imread(spec['mask']) > 0
    after = tifffile.imread(p['mask']) > 0 if p['mask'].is_file() else None
    bg = raw.max(0)
    vmax = max(25, float(np.percentile(bg, 99.5)))
    old_app2, source = reference_nodes(spec, bg.shape[0])
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    titles = ['Raw XY MIP', 'Original mask', 'Updated mask: red additions', 'Previous SWC',
              'Previous APP2', 'Current APP2', 'APP2 overlay: cyan previous / orange current',
              spec['config'].get('method_version', 'v0.83')+' final SWC']
    for ax, title in zip(axes.flat, titles):
        ax.imshow(bg, cmap='gray', vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.axis('off')
    for ax, updated in [(axes[0, 1], False), (axes[0, 2], True)]:
        rgba = np.zeros((*bg.shape, 4))
        rgba[before.any(0)] = [0, 1, 1, .65]
        if updated and after is not None:
            rgba[(after & ~before).any(0)] = [1, .1, 0, .85]
            rgba[(before & ~after).any(0)] = [1, 1, 0, .85]
        if updated and after is None:
            ax.text(.5, .5, 'Updated mask unavailable', transform=ax.transAxes, ha='center', color='yellow')
        ax.imshow(rgba)
    for ax, files, color in [(axes[0, 3], spec.get('old_swcs', []), 'cyan'),
                             (axes[1, 1], [p['app2']], 'orange'),
                             (axes[1, 2], [p['app2']], 'orange'),
                             (axes[1, 3], [p['final']], 'lime')]:
        if not files:
            ax.text(.5, .5, 'Previous SWC not supplied', transform=ax.transAxes, ha='center', color='yellow')
        for file in files:
            if not Path(file).is_file():
                ax.text(.5, .5, 'No output: stage failed / not reached', transform=ax.transAxes, ha='center', color='yellow')
                continue
            nodes = check_swc(Path(file))
            edges = [[n.xy, nodes[n.parent].xy] for n in nodes.values() if n.parent in nodes]
            ax.add_collection(LineCollection(edges, colors=color, linewidths=.9))
    for ax in [axes[1, 0], axes[1, 2]]:
        if old_app2:
            edges = [[xyz[:2], old_app2[parent][0][:2]] for xyz, parent in old_app2.values() if parent in old_app2]
            ax.add_collection(LineCollection(edges, colors='cyan', linewidths=1, alpha=.85))
        else:
            ax.text(.5, .5, 'No matching previous APP2', transform=ax.transAxes, ha='center', color='yellow')
    metadata = spec['metadata']
    soma_x = float(metadata['soma_x'])*float(metadata['xy_resolution'])/1000
    soma_y = float(metadata['soma_y'])*float(metadata['xy_resolution'])/1000
    for ax in axes.flat:
        ax.scatter([soma_x], [soma_y], s=12, c='red', marker='o', edgecolors='none', zorder=20)
        ax.set_xlim(-.5, bg.shape[1]-.5)
        ax.set_ylim(bg.shape[0]-.5, -.5)
    from visualization.swc_view import crop_xy_axes
    previous_xy = [n.xy for file in spec.get('old_swcs', []) if Path(file).is_file()
                   for n in check_swc(Path(file)).values()]
    app2_xy = [xyz[:2] for xyz, parent in old_app2.values()]
    if p['app2'].is_file():
        app2_xy += [n.xy for n in check_swc(p['app2']).values()]
    crop_xy_axes(axes[0, 3], bg.shape, previous_xy)
    for ax in [axes[1, 0], axes[1, 1], axes[1, 2]]:
        crop_xy_axes(ax, bg.shape, app2_xy)
    final_xy = [n.xy for n in check_swc(p['final']).values()] if p['final'].is_file() else []
    crop_xy_axes(axes[1, 3], bg.shape, final_xy)
    fig.suptitle(f"{spec['id']} | {spec['selection_reason']} | red dot: target soma | " +
                 (f'INCOMPLETE: {failure}' if failure else 'SWCs in image coordinates'))
    source_text = ('Previous APP2: '+source['path']+' | '+source['transform'] if old_app2
                   else 'Previous APP2: no exact ID match in configured directories')
    fig.text(.02, .015, source_text, fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .96))
    save_figure(fig, p['comparison'])
    save_json(p['root']/'comparison_sources'/f'{spec["id"]}_app2.json', source)
