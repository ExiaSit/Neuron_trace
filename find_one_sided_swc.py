#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE_EXTENSIONS = ".tif,.tiff,.png,.jpg,.jpeg,.npy"


@dataclass(frozen=True)
class SwcNode:
    node_id: int
    node_type: int
    x: float
    y: float
    z: float
    radius: float
    parent: int

    @property
    def xyz(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)


@dataclass
class SomaInfo:
    center: np.ndarray
    radius: float
    source: str

    @property
    def center_xy(self) -> np.ndarray:
        return self.center[:2]


@dataclass
class SideBalance:
    direction_name: str = ""
    direction: np.ndarray | None = None
    major_side: str = ""
    major_vector: np.ndarray | None = None
    pos_length: float = 0.0
    neg_length: float = 0.0
    neutral_length: float = 0.0
    total_length: float = 0.0
    counted_fraction: float = 0.0
    major_fraction: float = 0.0
    minor_fraction: float = 0.0
    score: float = 0.0


@dataclass
class SwcAnalysis:
    source_path: Path
    suspicious: bool
    reason: str
    n_nodes: int = 0
    n_samples: int = 0
    soma: SomaInfo | None = None
    balance: SideBalance | None = None
    copied_swc: Path | None = None
    view_png: Path | None = None
    image_path: Path | None = None
    seg_path: Path | None = None
    overlay_png: Path | None = None
    image_error: str = ""
    seg_error: str = ""
    error: str = ""


def parse_swc(path: Path) -> tuple[dict[int, SwcNode], list[int]]:
    nodes: dict[int, SwcNode] = {}
    order: list[int] = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            parts = stripped.split()
            if len(parts) < 7:
                raise ValueError(f"Malformed SWC line {line_number}: {stripped}")

            try:
                node = SwcNode(
                    node_id=int(float(parts[0])),
                    node_type=int(float(parts[1])),
                    x=float(parts[2]),
                    y=float(parts[3]),
                    z=float(parts[4]),
                    radius=float(parts[5]),
                    parent=int(float(parts[6])),
                )
            except ValueError as exc:
                raise ValueError(f"Cannot parse SWC line {line_number}: {stripped}") from exc

            if node.node_id in nodes:
                raise ValueError(f"Duplicate SWC node id {node.node_id}")

            nodes[node.node_id] = node
            order.append(node.node_id)

    if not nodes:
        raise ValueError("No SWC nodes found")

    return nodes, order


def find_soma(nodes: dict[int, SwcNode], order: Iterable[int]) -> SomaInfo:
    soma_nodes = [node for node in nodes.values() if node.node_type == 1]
    source = "type_1"

    if not soma_nodes:
        soma_nodes = [node for node in nodes.values() if node.parent < 0]
        source = "root"

    if not soma_nodes:
        soma_nodes = [nodes[next(iter(order))]]
        source = "first_node"

    coords = np.array([node.xyz for node in soma_nodes], dtype=float)
    weights = np.array([max(node.radius, 0.0) ** 2 for node in soma_nodes], dtype=float)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones(len(soma_nodes), dtype=float)

    center = np.average(coords, axis=0, weights=weights)
    radius_values = np.array([max(node.radius, 0.0) for node in soma_nodes], dtype=float)
    radius = float(np.average(radius_values, weights=weights))

    return SomaInfo(center=center, radius=radius, source=source)


def build_branch_samples_xy(
    nodes: dict[int, SwcNode],
    soma: SomaInfo,
    ignore_near_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    vectors: list[np.ndarray] = []
    weights: list[float] = []

    for child in nodes.values():
        parent = nodes.get(child.parent)
        if parent is None:
            continue

        if child.node_type == 1 and parent.node_type == 1:
            continue

        child_xy = child.xy
        parent_xy = parent.xy
        projected_length = float(np.linalg.norm(child_xy - parent_xy))
        if not np.isfinite(projected_length) or projected_length <= 0:
            continue

        midpoint_xy = (child_xy + parent_xy) / 2.0
        vector_xy = midpoint_xy - soma.center_xy
        if np.linalg.norm(vector_xy) <= ignore_near_distance:
            continue

        vectors.append(vector_xy)
        weights.append(projected_length)

    if vectors:
        return np.array(vectors, dtype=float), np.array(weights, dtype=float)

    for node in nodes.values():
        if node.node_type == 1:
            continue

        vector_xy = node.xy - soma.center_xy
        distance = float(np.linalg.norm(vector_xy))
        if not np.isfinite(distance) or distance <= ignore_near_distance:
            continue

        vectors.append(vector_xy)
        weights.append(1.0)

    return np.array(vectors, dtype=float), np.array(weights, dtype=float)


def add_direction(
    directions: list[tuple[str, np.ndarray]],
    name: str,
    direction: np.ndarray,
    duplicate_dot: float = 0.995,
) -> None:
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm == 0:
        return

    unit = direction / norm
    for _, existing in directions:
        if abs(float(np.dot(unit, existing))) >= duplicate_dot:
            return

    directions.append((name, unit))


def candidate_directions_xy(
    vectors: np.ndarray,
    weights: np.ndarray,
    max_farthest: int,
) -> list[tuple[str, np.ndarray]]:
    directions: list[tuple[str, np.ndarray]] = []
    add_direction(directions, "x_axis_xy", np.array([1.0, 0.0]))
    add_direction(directions, "y_axis_xy", np.array([0.0, 1.0]))

    weighted_mean = np.average(vectors, axis=0, weights=weights)
    add_direction(directions, "weighted_mean_xy", weighted_mean)

    covariance = (vectors.T * weights) @ vectors / max(float(weights.sum()), 1e-12)
    if np.isfinite(covariance).all():
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        for rank, idx in enumerate(np.argsort(eigenvalues)[::-1], start=1):
            add_direction(directions, f"pca_xy_{rank}", eigenvectors[:, idx])

    distances = np.linalg.norm(vectors, axis=1)
    farthest_indices = np.argsort(distances)[::-1][:max_farthest]
    for rank, idx in enumerate(farthest_indices, start=1):
        add_direction(directions, f"farthest_xy_{rank}", vectors[idx])

    return directions


def evaluate_side_balance_xy(
    vectors: np.ndarray,
    weights: np.ndarray,
    side_dead_zone: float,
    max_farthest_directions: int,
) -> SideBalance:
    best = SideBalance()

    total_length = float(weights.sum())
    for direction_name, direction in candidate_directions_xy(
        vectors, weights, max_farthest=max_farthest_directions
    ):
        projection = vectors @ direction
        pos_length = float(weights[projection > side_dead_zone].sum())
        neg_length = float(weights[projection < -side_dead_zone].sum())
        neutral_length = max(total_length - pos_length - neg_length, 0.0)
        counted_length = pos_length + neg_length

        if counted_length <= 0:
            continue

        major_length = max(pos_length, neg_length)
        minor_length = min(pos_length, neg_length)
        major_fraction = major_length / counted_length
        minor_fraction = minor_length / counted_length
        counted_fraction = counted_length / total_length if total_length > 0 else 0.0
        score = major_fraction * counted_fraction
        major_side = "positive" if pos_length >= neg_length else "negative"
        major_vector = direction if major_side == "positive" else -direction

        if score > best.score:
            best = SideBalance(
                direction_name=direction_name,
                direction=direction,
                major_side=major_side,
                major_vector=major_vector,
                pos_length=pos_length,
                neg_length=neg_length,
                neutral_length=neutral_length,
                total_length=total_length,
                counted_fraction=counted_fraction,
                major_fraction=major_fraction,
                minor_fraction=minor_fraction,
                score=score,
            )

    return best


def analyze_swc(
    path: Path,
    args: argparse.Namespace,
) -> tuple[SwcAnalysis, dict[int, SwcNode] | None, list[int] | None]:
    try:
        nodes, order = parse_swc(path)
        soma = find_soma(nodes, order)
        ignore_near_distance = max(
            args.min_ignore_distance,
            soma.radius * args.ignore_soma_radius_factor,
        )
        vectors, weights = build_branch_samples_xy(nodes, soma, ignore_near_distance)

        if len(weights) < args.min_samples and ignore_near_distance > 0:
            vectors, weights = build_branch_samples_xy(nodes, soma, 0.0)

        analysis = SwcAnalysis(
            source_path=path,
            suspicious=False,
            reason="not_one_sided_xy",
            n_nodes=len(nodes),
            n_samples=len(weights),
            soma=soma,
        )

        if len(weights) < args.min_samples:
            analysis.reason = "too_few_xy_branch_samples"
            return analysis, nodes, order

        total_length = float(weights.sum())
        if total_length < args.min_total_length:
            analysis.reason = "xy_branch_length_too_short"
            return analysis, nodes, order

        balance = evaluate_side_balance_xy(
            vectors=vectors,
            weights=weights,
            side_dead_zone=args.side_dead_zone,
            max_farthest_directions=args.max_farthest_directions,
        )
        analysis.balance = balance

        analysis.suspicious = (
            balance.major_fraction >= args.major_fraction
            and balance.minor_fraction <= args.minor_fraction
            and balance.counted_fraction >= args.min_counted_fraction
        )
        analysis.reason = "one_sided_xy" if analysis.suspicious else "not_one_sided_xy"
        return analysis, nodes, order
    except Exception as exc:
        return (
            SwcAnalysis(
                source_path=path,
                suspicious=False,
                reason="error",
                error=str(exc),
            ),
            None,
            None,
        )


def iter_swc_files(root: Path, pattern: str, recursive: bool) -> list[Path]:
    if recursive:
        paths = root.rglob(pattern)
    else:
        paths = root.glob(pattern)

    return sorted([path for path in paths if path.is_file()], key=lambda p: p.as_posix())


def parse_extensions(value: str) -> tuple[str, ...]:
    extensions: list[str] = []
    for item in value.split(","):
        extension = item.strip().lower()
        if not extension:
            continue
        if not extension.startswith("."):
            extension = f".{extension}"
        extensions.append(extension)
    return tuple(dict.fromkeys(extensions))


def build_image_index(img_dir: Path, extensions: tuple[str, ...]) -> dict[str, list[Path]]:
    image_index: dict[str, list[Path]] = {}
    for path in sorted(img_dir.rglob("*"), key=lambda p: p.as_posix()):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        image_index.setdefault(path.stem.lower(), []).append(path)
    return image_index


def image_stem_candidates(stem_path: Path, suffix: str) -> list[Path]:
    candidates = [stem_path]
    if suffix:
        candidates.append(stem_path.with_name(f"{stem_path.name}{suffix}"))
    return candidates


def find_matching_image(
    swc_path: Path,
    swc_root: Path,
    img_dir: Path,
    image_index: dict[str, list[Path]],
    extensions: tuple[str, ...],
    image_stem_suffix: str,
) -> Path | None:
    try:
        relative_stem = swc_path.relative_to(swc_root).with_suffix("")
    except ValueError:
        relative_stem = Path(swc_path.stem)

    for candidate_stem in image_stem_candidates(relative_stem, image_stem_suffix):
        for extension in extensions:
            candidate = (img_dir / candidate_stem).with_suffix(extension)
            if candidate.is_file():
                return candidate

    for stem in [swc_path.stem, f"{swc_path.stem}{image_stem_suffix}"]:
        matches = image_index.get(stem.lower(), [])
        if matches:
            return matches[0]

    return None


def safe_output_stem(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path.name

    stem = Path(relative).with_suffix("").as_posix()
    stem = stem.replace("/", "__")
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", stem).strip("_") or "swc"


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


def analysis_to_row(analysis: SwcAnalysis) -> dict[str, str | int | float]:
    soma = analysis.soma
    balance = analysis.balance or SideBalance()

    if balance.major_vector is None:
        major_vector = ""
    else:
        major_vector = ",".join(f"{value:.6g}" for value in balance.major_vector)

    return {
        "file_name": analysis.source_path.name,
        "source_path": str(analysis.source_path),
        "suspicious": int(analysis.suspicious),
        "reason": analysis.reason,
        "detection_space": "xy_projection",
        "n_nodes": analysis.n_nodes,
        "n_samples": analysis.n_samples,
        "soma_x": "" if soma is None else f"{soma.center[0]:.6f}",
        "soma_y": "" if soma is None else f"{soma.center[1]:.6f}",
        "soma_z": "" if soma is None else f"{soma.center[2]:.6f}",
        "soma_radius": "" if soma is None else f"{soma.radius:.6f}",
        "soma_source": "" if soma is None else soma.source,
        "direction_name": balance.direction_name,
        "major_side": balance.major_side,
        "major_vector_xy": major_vector,
        "pos_xy_length": f"{balance.pos_length:.6f}",
        "neg_xy_length": f"{balance.neg_length:.6f}",
        "neutral_xy_length": f"{balance.neutral_length:.6f}",
        "total_xy_length": f"{balance.total_length:.6f}",
        "counted_fraction": f"{balance.counted_fraction:.6f}",
        "major_fraction": f"{balance.major_fraction:.6f}",
        "minor_fraction": f"{balance.minor_fraction:.6f}",
        "score": f"{balance.score:.6f}",
        "image_path": "" if analysis.image_path is None else str(analysis.image_path),
        "seg_path": "" if analysis.seg_path is None else str(analysis.seg_path),
        "copied_swc": "" if analysis.copied_swc is None else str(analysis.copied_swc),
        "view_png": "" if analysis.view_png is None else str(analysis.view_png),
        "overlay_png": "" if analysis.overlay_png is None else str(analysis.overlay_png),
        "image_error": analysis.image_error,
        "seg_error": analysis.seg_error,
        "error": analysis.error,
    }


def write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    fieldnames = [
        "file_name",
        "source_path",
        "suspicious",
        "reason",
        "detection_space",
        "n_nodes",
        "n_samples",
        "soma_x",
        "soma_y",
        "soma_z",
        "soma_radius",
        "soma_source",
        "direction_name",
        "major_side",
        "major_vector_xy",
        "pos_xy_length",
        "neg_xy_length",
        "neutral_xy_length",
        "total_xy_length",
        "counted_fraction",
        "major_fraction",
        "minor_fraction",
        "score",
        "image_path",
        "seg_path",
        "copied_swc",
        "view_png",
        "overlay_png",
        "image_error",
        "seg_error",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_suspicious_txt(path: Path, analyses: list[SwcAnalysis]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for analysis in analyses:
            if analysis.suspicious:
                handle.write(f"{analysis.source_path}\n")


def make_output_dir(base_dir: Path | None) -> Path:
    if base_dir is not None:
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir

    run_name = datetime.now().strftime("one_sided_swc_xy_%Y%m%d_%H%M%S")
    output_dir = SCRIPT_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find SWC traces whose branches are concentrated on one side of the soma "
            "in the XY projection, then generate three-view PNGs and optional MIP overlays."
        )
    )
    parser.add_argument("swc_dir", type=Path, help="Folder containing cropped SWC files.")
    parser.add_argument(
        "img_dir",
        nargs="?",
        type=Path,
        help=(
            "Folder containing original images. Images are matched by the same stem or by "
            "adding --image-stem-suffix, for example image_60000.swc -> image_60000_0000.tif."
        ),
    )
    parser.add_argument(
        "seg_dir",
        nargs="?",
        type=Path,
        help=(
            "Folder containing segmentation results. Seg files are matched by the SWC stem by "
            "default, for example image_60000.swc -> image_60000.tif."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: a timestamped folder under this script directory.",
    )
    parser.add_argument(
        "--pattern",
        default="*.swc",
        help="SWC file glob pattern. Default: *.swc",
    )
    parser.add_argument(
        "--image-extensions",
        default=DEFAULT_IMAGE_EXTENSIONS,
        help=f"Comma-separated image extensions. Default: {DEFAULT_IMAGE_EXTENSIONS}",
    )
    parser.add_argument(
        "--image-stem-suffix",
        default="_0000",
        help="Extra suffix appended to image stems when matching SWC files. Default: _0000",
    )
    parser.add_argument(
        "--seg-extensions",
        default=DEFAULT_IMAGE_EXTENSIONS,
        help=f"Comma-separated segmentation extensions. Default: {DEFAULT_IMAGE_EXTENSIONS}",
    )
    parser.add_argument(
        "--seg-stem-suffix",
        default="",
        help="Extra suffix appended to segmentation stems when matching SWC files. Default: empty",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan files directly inside swc_dir.",
    )
    parser.add_argument(
        "--major-fraction",
        type=float,
        default=0.85,
        help="Minimum projected branch length fraction on the dominant XY side. Default: 0.85",
    )
    parser.add_argument(
        "--minor-fraction",
        type=float,
        default=0.15,
        help="Maximum projected branch length fraction on the opposite XY side. Default: 0.15",
    )
    parser.add_argument(
        "--min-counted-fraction",
        type=float,
        default=0.70,
        help="Minimum fraction outside the XY side dead zone. Default: 0.70",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="Minimum projected branch samples required for analysis. Default: 5",
    )
    parser.add_argument(
        "--min-total-length",
        type=float,
        default=0.0,
        help="Minimum projected XY branch length required for analysis. Default: 0",
    )
    parser.add_argument(
        "--ignore-soma-radius-factor",
        type=float,
        default=1.5,
        help="Ignore projected branch samples within this many soma radii. Default: 1.5",
    )
    parser.add_argument(
        "--min-ignore-distance",
        type=float,
        default=0.0,
        help="Additional minimum XY distance around soma to ignore. Default: 0",
    )
    parser.add_argument(
        "--side-dead-zone",
        type=float,
        default=0.0,
        help="Projected XY distance around the soma split line counted as neutral. Default: 0",
    )
    parser.add_argument(
        "--max-farthest-directions",
        type=int,
        default=32,
        help="Number of farthest XY directions used as candidate split lines. Default: 32",
    )
    parser.add_argument(
        "--overlay-zoom-padding",
        type=float,
        default=80.0,
        help="Minimum padding in pixels for the zoomed MIP overlay. Default: 80",
    )
    parser.add_argument(
        "--no-copy-swc",
        action="store_true",
        help="Do not copy suspicious SWC files into the output directory.",
    )
    parser.add_argument(
        "--plot-all",
        action="store_true",
        help="Generate three-view PNGs and MIP overlays for every SWC, not only suspicious ones.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG output DPI. Default: 180",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    swc_dir = args.swc_dir.expanduser().resolve()
    if not swc_dir.is_dir():
        raise SystemExit(f"SWC folder does not exist: {swc_dir}")

    img_dir = args.img_dir.expanduser().resolve() if args.img_dir else None
    if img_dir is not None and not img_dir.is_dir():
        raise SystemExit(f"Image folder does not exist: {img_dir}")

    seg_dir = args.seg_dir.expanduser().resolve() if args.seg_dir else None
    if seg_dir is not None and not seg_dir.is_dir():
        raise SystemExit(f"Segmentation folder does not exist: {seg_dir}")

    output_dir = make_output_dir(args.output_dir.expanduser().resolve() if args.output_dir else None)
    suspicious_dir = output_dir / "suspicious_swc"
    view_dir = output_dir / "three_views"
    overlay_dir = output_dir / "mip_overlays"
    all_swc_paths = iter_swc_files(
        root=swc_dir,
        pattern=args.pattern,
        recursive=not args.no_recursive,
    )

    if not all_swc_paths:
        raise SystemExit(f"No SWC files matched {args.pattern!r} under {swc_dir}")

    image_extensions = parse_extensions(args.image_extensions)
    image_index: dict[str, list[Path]] = {}
    if img_dir is not None:
        image_index = build_image_index(img_dir, image_extensions)
        print(f"Indexed images: {sum(len(paths) for paths in image_index.values())}")

    seg_extensions = parse_extensions(args.seg_extensions)
    seg_index: dict[str, list[Path]] = {}
    if seg_dir is not None:
        seg_index = build_image_index(seg_dir, seg_extensions)
        print(f"Indexed segmentations: {sum(len(paths) for paths in seg_index.values())}")

    analyses: list[SwcAnalysis] = []
    for index, swc_path in enumerate(all_swc_paths, start=1):
        print(f"[{index}/{len(all_swc_paths)}] {swc_path}")
        analysis, nodes, _ = analyze_swc(swc_path, args)

        if img_dir is not None:
            analysis.image_path = find_matching_image(
                swc_path=swc_path,
                swc_root=swc_dir,
                img_dir=img_dir,
                image_index=image_index,
                extensions=image_extensions,
                image_stem_suffix=args.image_stem_suffix,
            )
            if analysis.image_path is None:
                analysis.image_error = "matching_image_not_found"

        if seg_dir is not None:
            analysis.seg_path = find_matching_image(
                swc_path=swc_path,
                swc_root=swc_dir,
                img_dir=seg_dir,
                image_index=seg_index,
                extensions=seg_extensions,
                image_stem_suffix=args.seg_stem_suffix,
            )
            if analysis.seg_path is None:
                analysis.seg_error = "matching_seg_not_found"

        should_plot = analysis.suspicious or args.plot_all
        output_stem = safe_output_stem(swc_path, swc_dir)
        if analysis.suspicious and not args.no_copy_swc:
            copied_path = suspicious_dir / f"{output_stem}.swc"
            copied_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(swc_path, copied_path)
            analysis.copied_swc = copied_path

        if should_plot and nodes is not None and analysis.soma is not None:
            view_path = view_dir / f"{output_stem}_three_views.png"
            plot_three_views(
                path=swc_path,
                nodes=nodes,
                soma=analysis.soma,
                balance=analysis.balance or SideBalance(),
                output_png=view_path,
                dpi=args.dpi,
            )
            analysis.view_png = view_path

            if analysis.image_path is not None:
                overlay_path = overlay_dir / f"{output_stem}_mip_overlay.png"
                try:
                    image_mip_xy = read_image_as_mip_xy(analysis.image_path)
                    seg_mip_xy = None
                    if analysis.seg_path is not None:
                        try:
                            seg_mip_xy = read_image_as_mip_xy(analysis.seg_path)
                        except Exception as exc:
                            analysis.seg_error = f"seg_read_failed: {exc}"
                    plot_mip_overlay(
                        swc_path=swc_path,
                        image_path=analysis.image_path,
                        image_mip_xy=image_mip_xy,
                        nodes=nodes,
                        soma=analysis.soma,
                        balance=analysis.balance or SideBalance(),
                        output_png=overlay_path,
                        dpi=args.dpi,
                        zoom_padding=args.overlay_zoom_padding,
                        seg_path=analysis.seg_path if seg_mip_xy is not None else None,
                        seg_mip_xy=seg_mip_xy,
                    )
                    analysis.overlay_png = overlay_path
                except Exception as exc:
                    analysis.image_error = f"overlay_failed: {exc}"

        analyses.append(analysis)

    all_rows = [analysis_to_row(analysis) for analysis in analyses]
    suspicious_rows = [
        analysis_to_row(analysis)
        for analysis in analyses
        if analysis.suspicious
    ]
    write_csv(output_dir / "all_results.csv", all_rows)
    write_csv(output_dir / "suspicious_swc.csv", suspicious_rows)
    write_suspicious_txt(output_dir / "suspicious_swc.txt", analyses)

    overlay_count = sum(1 for analysis in analyses if analysis.overlay_png is not None)
    image_issue_count = sum(1 for analysis in analyses if analysis.image_error)
    seg_issue_count = sum(1 for analysis in analyses if analysis.seg_error)

    print()
    print(f"Scanned SWC files: {len(analyses)}")
    print(f"Suspicious one-sided XY SWC files: {len(suspicious_rows)}")
    print(f"MIP overlays generated: {overlay_count}")
    if img_dir is not None:
        print(f"Image match/read issues: {image_issue_count}")
    if seg_dir is not None:
        print(f"Segmentation match/read issues: {seg_issue_count}")
    print(f"Output directory: {output_dir}")
    print(f"All results CSV: {output_dir / 'all_results.csv'}")
    print(f"Suspicious CSV: {output_dir / 'suspicious_swc.csv'}")
    print(f"Three-view PNG folder: {view_dir}")
    print(f"MIP overlay PNG folder: {overlay_dir}")


if __name__ == "__main__":
    main()
