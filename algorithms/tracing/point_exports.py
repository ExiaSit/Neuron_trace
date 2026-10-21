"""Candidate SWCs in original image XYZ coordinates, shared with 0.84_test."""
from pathlib import Path

def write_points(path, coords_zyx, target_zyx, shape_zyx, source, label):
    import numpy as np
    points = np.asarray(coords_zyx, dtype=float).reshape(-1, 3)
    if not np.isfinite(points).all() or not ((points >= 0) & (points < shape_zyx)).all():
        raise ValueError(f"Non-finite or out-of-image candidate coordinates: {label}")
    target_ids = [i+1 for i, p in enumerate(points) if np.allclose(p, target_zyx)]
    lines = [f"# {label}", f"# Source image: {source}",
             "# XYZ in original 1-um image; converted from ZYX; no Y flip or marker offset.",
             "# Independent points; radius=2 um for display, not a measured soma radius.",
             "# Target soma node IDs: "+",".join(map(str, target_ids)),
             "# n type x y z radius parent"]
    lines += [f"{i} 1 {x:.6f} {y:.6f} {z:.6f} 2.000000 -1"
              for i, (z, y, x) in enumerate(points, 1)]
    Path(path).write_text("\n".join(lines)+"\n")
    return dict(points=len(points), target_ids=target_ids)
