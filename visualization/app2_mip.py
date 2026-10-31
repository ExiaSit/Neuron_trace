import os
import numpy as np
import tifffile as tiff
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def _xy_mip(volume):
    if volume.ndim == 3:
        return np.max(volume, axis=0)
    if volume.ndim == 2:
        return volume
    return np.max(np.squeeze(volume), axis=0)


def _display_vmax(img_2d):
    if np.max(img_2d) <= 0:
        return 1
    vmax = np.percentile(img_2d, 99.5)
    return vmax if vmax > 0 else np.max(img_2d)


def _read_swc_xy_edges(swc_path):
    nodes = {}
    edges = []
    if not swc_path or not os.path.exists(swc_path):
        return nodes, edges
    with open(swc_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                node_id = int(float(parts[0]))
                x = float(parts[2])
                y = float(parts[3])
                parent_id = int(float(parts[6]))
            except ValueError:
                continue
            nodes[node_id] = (x, y)
            if parent_id != -1:
                edges.append((node_id, parent_id))
    return nodes, edges


def generate_xy_mip_visualization(raw_img_path, mask_path, swc_path, output_png_path, neuron_id):
    try:
        os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
        raw = tiff.imread(raw_img_path)
        mask = tiff.imread(mask_path)
        raw_xy = _xy_mip(raw)
        mask_xy = _xy_mip(mask) > 0
        nodes, edges = _read_swc_xy_edges(swc_path)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
        for ax in axes:
            ax.imshow(raw_xy, cmap="gray", vmin=0, vmax=_display_vmax(raw_xy))
            ax.axis("off")

        axes[0].set_title(f"Neuron {neuron_id} APP2 XY")
        for node_id, parent_id in edges:
            if node_id not in nodes or parent_id not in nodes:
                continue
            x0, y0 = nodes[node_id]
            x1, y1 = nodes[parent_id]
            axes[0].plot([x0, x1], [y0, y1], color="cyan", linewidth=0.7, alpha=0.85)
        if nodes:
            xs, ys = zip(*nodes.values())
            axes[0].scatter(xs, ys, s=2, c="yellow", alpha=0.7)

        axes[1].set_title(f"Neuron {neuron_id} Raw + Mask XY")
        overlay = np.zeros((*mask_xy.shape, 4), dtype=float)
        overlay[..., 0] = 1.0
        overlay[..., 3] = mask_xy.astype(float) * 0.35
        axes[1].imshow(overlay)

        from visualization.swc_view import crop_xy_axes
        for ax in axes:
            crop_xy_axes(ax, raw_xy.shape, nodes.values())
        fig.tight_layout()
        fig.savefig(output_png_path, bbox_inches="tight")
        plt.close(fig)
        return "OK"
    except Exception as e:
        plt.close("all")
        return f"VIS_FAIL: {e}"
