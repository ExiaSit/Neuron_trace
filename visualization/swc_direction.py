from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def node_type_color(node_type: int) -> str:
    colors = {
        1: "#d62728",
        2: "#4c78a8",
        3: "#59a14f",
        4: "#f28e2b",
        5: "#b07aa1",
        6: "#e15759",
        7: "#76b7b2",
    }
    return colors.get(node_type, "#666666")

def read_image_as_mip_xy(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        image = np.load(path)
    elif suffix in {".tif", ".tiff"}:
        import tifffile

        image = tifffile.imread(path)
    else:
        from PIL import Image

        image = np.asarray(Image.open(path))

    image = np.asarray(image)
    image = np.squeeze(image)

    if image.ndim == 2:
        return image

    if image.ndim == 3:
        if image.shape[-1] in (3, 4):
            return image[..., :3]
        return image.max(axis=0)

    if image.ndim == 4:
        if image.shape[-1] in (3, 4):
            return image.max(axis=0)[..., :3]
        if image.shape[1] in (3, 4):
            return image.max(axis=(0, 1))
        return image.max(axis=0)

    raise ValueError(f"Unsupported image shape {image.shape} for {path}")

def normalize_image_for_display(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=float)

    values = arr[finite]
    low, high = np.percentile(values, [1.0, 99.8])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(values.min())
        high = float(values.max())
    if high <= low:
        return np.zeros_like(arr, dtype=float)

    arr = (arr - low) / (high - low)
    return np.clip(arr, 0.0, 1.0)

def seg_mask_for_display(seg_mip_xy: np.ndarray) -> np.ndarray:
    arr = np.asarray(seg_mip_xy)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = arr[..., :3].max(axis=2)
    return np.asarray(arr > 0, dtype=bool)

def overlay_seg_mask(ax: plt.Axes, seg_mask: np.ndarray, alpha: float = 0.28) -> None:
    masked = np.ma.masked_where(~seg_mask, seg_mask)
    ax.imshow(masked, cmap="spring", alpha=alpha, origin="upper", interpolation="nearest")

def collect_edges(nodes: dict[int, SwcNode]) -> list[tuple[int, int]]:
    return [(node.parent, node.node_id) for node in nodes.values() if node.parent in nodes]

def set_view_limits(ax: plt.Axes, coords: np.ndarray, x_idx: int, y_idx: int) -> None:
    x_values = coords[:, x_idx]
    y_values = coords[:, y_idx]
    x_min, x_max = float(x_values.min()), float(x_values.max())
    y_min, y_max = float(y_values.min()), float(y_values.max())
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1.0)
    span = max(x_span, y_span)
    x_mid = (x_min + x_max) / 2.0
    y_mid = (y_min + y_max) / 2.0
    pad = span * 0.08
    ax.set_xlim(x_mid - span / 2.0 - pad, x_mid + span / 2.0 + pad)
    ax.set_ylim(y_mid - span / 2.0 - pad, y_mid + span / 2.0 + pad)

def draw_xy_direction_arrow(
    ax: plt.Axes,
    soma: SomaInfo,
    balance: SideBalance,
    span: float,
) -> None:
    if balance.major_vector is None:
        return

    arrow = balance.major_vector * span * 0.18
    ax.arrow(
        soma.center[0],
        soma.center[1],
        arrow[0],
        arrow[1],
        width=span * 0.004,
        head_width=span * 0.035,
        head_length=span * 0.045,
        length_includes_head=True,
        color="#ff7f0e",
        alpha=0.9,
        zorder=6,
    )

def plot_three_views(
    path: Path,
    nodes: dict[int, SwcNode],
    soma: SomaInfo,
    balance: SideBalance,
    output_png: Path,
    dpi: int,
) -> None:
    coords_by_id = {node_id: node.xyz for node_id, node in nodes.items()}
    all_coords = np.array(list(coords_by_id.values()), dtype=float)
    edges = collect_edges(nodes)

    views = [
        ("XY", 0, 1),
        ("XZ", 0, 2),
        ("YZ", 1, 2),
    ]
    axis_labels = ["X", "Y", "Z"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, (view_name, x_idx, y_idx) in zip(axes, views):
        for parent_id, child_id in edges:
            child = nodes[child_id]
            parent_xyz = coords_by_id[parent_id]
            child_xyz = coords_by_id[child_id]
            line_width = min(max(child.radius * 0.35, 0.25), 2.0)
            ax.plot(
                [parent_xyz[x_idx], child_xyz[x_idx]],
                [parent_xyz[y_idx], child_xyz[y_idx]],
                color=node_type_color(child.node_type),
                linewidth=line_width,
                alpha=0.82,
                solid_capstyle="round",
            )

        ax.scatter(
            [soma.center[x_idx]],
            [soma.center[y_idx]],
            s=72,
            c="#d62728",
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
        )

        set_view_limits(ax, all_coords, x_idx, y_idx)
        if view_name == "XY":
            span = max(ax.get_xlim()[1] - ax.get_xlim()[0], ax.get_ylim()[1] - ax.get_ylim()[0])
            draw_xy_direction_arrow(ax, soma, balance, span)

        ax.set_aspect("equal", adjustable="box")
        ax.set_title(view_name)
        ax.set_xlabel(axis_labels[x_idx])
        ax.set_ylabel(axis_labels[y_idx])
        ax.grid(color="#d9d9d9", linewidth=0.5, alpha=0.45)

    fig.suptitle(
        (
            f"{path.name} | XY score={balance.score:.3f}, "
            f"major={balance.major_fraction:.3f}, minor={balance.minor_fraction:.3f}"
        ),
        fontsize=11,
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi)
    plt.close(fig)

def draw_swc_xy_on_axis(
    ax: plt.Axes,
    nodes: dict[int, SwcNode],
    soma: SomaInfo,
    balance: SideBalance,
    line_alpha: float,
    line_width_scale: float,
    arrow_span: float,
) -> None:
    for parent_id, child_id in collect_edges(nodes):
        child = nodes[child_id]
        parent = nodes[parent_id]
        line_width = min(max(child.radius * line_width_scale, 0.45), 2.4)
        ax.plot(
            [parent.x, child.x],
            [parent.y, child.y],
            color="#00d5ff",
            linewidth=line_width,
            alpha=line_alpha,
            solid_capstyle="round",
            zorder=3,
        )

    ax.scatter(
        [soma.center[0]],
        [soma.center[1]],
        s=78,
        c="#ff2d2d",
        edgecolors="white",
        linewidths=0.9,
        zorder=5,
    )
    draw_xy_direction_arrow(ax, soma, balance, arrow_span)

def plot_mip_overlay(
    swc_path: Path,
    image_path: Path,
    image_mip_xy: np.ndarray,
    nodes: dict[int, SwcNode],
    soma: SomaInfo,
    balance: SideBalance,
    output_png: Path,
    dpi: int,
    zoom_padding: float,
    seg_path: Path | None = None,
    seg_mip_xy: np.ndarray | None = None,
) -> None:
    display_image = normalize_image_for_display(image_mip_xy)
    height, width = display_image.shape[:2]
    node_xy = np.array([node.xy for node in nodes.values()], dtype=float)
    has_seg = seg_mip_xy is not None
    seg_display = normalize_image_for_display(seg_mip_xy) if has_seg else None
    seg_mask = seg_mask_for_display(seg_mip_xy) if has_seg else None

    if has_seg:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
        titles = ["img MIP + SWC", "seg MIP + SWC", "img + seg + SWC zoom"]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
        titles = ["img MIP + SWC", "trace zoom"]

    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal", adjustable="box")

    axes[0].imshow(display_image, cmap="gray" if display_image.ndim == 2 else None, origin="upper")
    draw_swc_xy_on_axis(
        ax=axes[0],
        nodes=nodes,
        soma=soma,
        balance=balance,
        line_alpha=0.88,
        line_width_scale=0.35,
        arrow_span=float(max(width, height)),
    )
    axes[0].set_xlim(0, max(width - 1, 1))
    axes[0].set_ylim(max(height - 1, 1), 0)

    x_min, y_min = node_xy.min(axis=0)
    x_max, y_max = node_xy.max(axis=0)
    span = max(float(x_max - x_min), float(y_max - y_min), 1.0)
    pad = max(zoom_padding, span * 0.20)
    x_mid = float((x_min + x_max) / 2.0)
    y_mid = float((y_min + y_max) / 2.0)
    half = span / 2.0 + pad

    if has_seg and seg_display is not None and seg_mask is not None:
        seg_height, seg_width = seg_display.shape[:2]
        axes[1].imshow(seg_display, cmap="magma" if seg_display.ndim == 2 else None, origin="upper")
        draw_swc_xy_on_axis(
            ax=axes[1],
            nodes=nodes,
            soma=soma,
            balance=balance,
            line_alpha=0.88,
            line_width_scale=0.35,
            arrow_span=float(max(seg_width, seg_height)),
        )
        axes[1].set_xlim(0, max(seg_width - 1, 1))
        axes[1].set_ylim(max(seg_height - 1, 1), 0)

        axes[2].imshow(display_image, cmap="gray" if display_image.ndim == 2 else None, origin="upper")
        if seg_mask.shape[:2] == display_image.shape[:2]:
            overlay_seg_mask(axes[2], seg_mask)
        else:
            axes[2].set_title("img + SWC zoom (seg size differs)")
        draw_swc_xy_on_axis(
            ax=axes[2],
            nodes=nodes,
            soma=soma,
            balance=balance,
            line_alpha=0.92,
            line_width_scale=0.38,
            arrow_span=float(max(width, height)),
        )
        zoom_axis = axes[2]
    else:
        axes[1].imshow(display_image, cmap="gray" if display_image.ndim == 2 else None, origin="upper")
        draw_swc_xy_on_axis(
            ax=axes[1],
            nodes=nodes,
            soma=soma,
            balance=balance,
            line_alpha=0.88,
            line_width_scale=0.35,
            arrow_span=float(max(width, height)),
        )
        zoom_axis = axes[1]

    zoom_axis.set_xlim(max(0.0, x_mid - half), min(float(width - 1), x_mid + half))
    zoom_axis.set_ylim(min(float(height - 1), y_mid + half), max(0.0, y_mid - half))

    seg_name = f", seg={seg_path.name}" if seg_path is not None else ""
    fig.suptitle(
        (
            f"{swc_path.name} on {image_path.name}{seg_name} | "
            f"XY major={balance.major_fraction:.3f}, minor={balance.minor_fraction:.3f}"
        ),
        fontsize=11,
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi)
    plt.close(fig)
