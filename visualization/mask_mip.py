from pathlib import Path
import os, tempfile
import numpy as np
import tifffile
def load_inputs(raw_path, mask_path):
    from utils.mask_batch import load_inputs as load
    return load(raw_path, mask_path)

def make_mip(pair, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    raw, mask = load_inputs(pair[0], pair[1])
    final = tifffile.imread(pair[2]) > 0
    bg = raw.max(axis=0)
    original = mask.any(axis=0)
    added = (final & ~mask).any(axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    for index, (ax, title) in enumerate(zip(axes, ['Raw XY MIP', 'Original input mask', 'Final: cyan original / red additions'])):
        ax.imshow(bg, cmap='gray', vmin=0, vmax=max(25, float(np.percentile(bg, 99.5))))
        if index:
            rgba = np.zeros((*bg.shape, 4))
            rgba[original] = [0, 1, 1, .6]
            if index == 2:
                rgba[added] = [1, .1, 0, .85]
            ax.imshow(rgba)
        ax.set_title(title)
        ax.axis('off')
    fig.suptitle(pair[1].stem)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.png', delete=False) as f:
        tmp = Path(f.name)
    try:
        fig.savefig(tmp, dpi=140)
        os.replace(tmp, path)
    finally:
        plt.close(fig)
        tmp.unlink(missing_ok=True)
