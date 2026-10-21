"""Historical comparison definitions, kept unchanged for numeric compatibility.

Precision/Recall are image-based heuristics, not accuracy against manual truth.
"""
from dataclasses import dataclass
import math
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class SwcData:
    nodes: dict[int, tuple[float, float, float, float]]
    parents: dict[int, int]

def read_swc(path: Path) -> SwcData:
    nodes: dict[int, tuple[float, float, float, float]] = {}
    parents: dict[int, int] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
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
                z = float(parts[4])
                radius = max(0.1, float(parts[5]))
                parent = int(float(parts[6]))
            except ValueError:
                continue
            nodes[node_id] = (x, y, z, radius)
            parents[node_id] = parent
    if not nodes:
        raise ValueError(f"no valid SWC nodes in {path}")
    return SwcData(nodes=nodes, parents=parents)

def swc_total_length(swc: SwcData) -> tuple[float, int, int, int]:
    total = 0.0
    edge_count = 0
    missing_parent_edges = 0
    for node_id, parent_id in swc.parents.items():
        if parent_id == -1:
            continue
        if parent_id not in swc.nodes:
            missing_parent_edges += 1
            continue
        x0, y0, z0, _ = swc.nodes[node_id]
        x1, y1, z1, _ = swc.nodes[parent_id]
        total += math.dist((x0, y0, z0), (x1, y1, z1))
        edge_count += 1
    return total, len(swc.nodes), edge_count, missing_parent_edges

def estimate_prec_rec_single(module, raw_img: np.ndarray, swc_path: Path) -> dict[str, object]:
    raw_img = np.squeeze(raw_img).astype(np.float32, copy=False)
    if raw_img.ndim != 3:
        raise ValueError(f"precision/recall expects 3D raw image, got shape={raw_img.shape}")

    coords_xyz, parent_indices, radii = module.read_and_resample_swc(str(swc_path))
    if len(coords_xyz) < 10:
        raise ValueError(f"too few resampled nodes for precision/recall: {len(coords_xyz)}")

    coords_zyx = coords_xyz[:, [2, 1, 0]]
    sampled = module.map_coordinates(raw_img, coords_zyx.T, order=1)
    mean_intensity = float(np.mean(sampled))
    mean_raw = float(np.mean(raw_img))
    coordinate_ok = mean_intensity >= mean_raw * 1.2

    _signals, bgs, stds, snrs = module.compute_robust_snr(raw_img, coords_zyx, parent_indices)
    linearities = module.compute_local_linearity(coords_zyx, parent_indices, window_steps=10)

    soma_idx = np.where(parent_indices == np.arange(len(coords_zyx)))[0]
    if len(soma_idx) > 0:
        soma_coord = coords_zyx[soma_idx[0]]
        soma_r = radii[soma_idx[0]]
    else:
        soma_coord = coords_zyx[0]
        soma_r = 5.0

    dists_to_soma = np.linalg.norm(coords_zyx - soma_coord, axis=1)
    outside_soma_mask = dists_to_soma > (soma_r + 2.0)
    valid_nodes = (snrs > 2.0) & (linearities > 0.6)
    valid_eval_count = int(np.sum(outside_soma_mask))
    precision_score = (
        float(np.sum(valid_nodes & outside_soma_mask) / valid_eval_count)
        if valid_eval_count > 0
        else 0.0
    )

    missed_branches, total_known_branches, _diag_passed, _diag_failed_angle, _diag_failed_gap = (
        module.detect_missed_branches(raw_img, coords_zyx, parent_indices, radii, bgs, stds)
    )
    recall_score = (
        1.0
        if total_known_branches + missed_branches == 0
        else float(total_known_branches / (total_known_branches + missed_branches))
    )

    return {
        "precision": precision_score,
        "recall": recall_score,
        "mean_snr": float(np.mean(snrs)),
        "missed_branches": int(missed_branches),
        "total_known_branches": int(total_known_branches),
        "total_nodes_prec_rec": int(len(coords_zyx)),
        "valid_eval_count": valid_eval_count,
        "mean_swc_intensity": mean_intensity,
        "mean_raw_intensity": mean_raw,
        "coordinate_ok": coordinate_ok,
    }
