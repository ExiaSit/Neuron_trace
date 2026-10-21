"""Read-only diagnostics for distance NMS and soma candidate merging."""
import json
from pathlib import Path
import numpy as np


def write_audit(path, coords, scores, radius, after_nms, final, merge_audit):
    points = np.asarray(coords, dtype=float)
    kept = []
    decisions = []
    for idx in np.argsort(scores)[::-1]:
        blocker = next((j for j in kept if np.linalg.norm(points[idx]-points[j]) < radius), None)
        record = dict(candidate_index=int(idx), coordinate_zyx=points[idx].tolist(),
                      score=float(scores[idx]) if np.isfinite(scores[idx]) else None,
                      priority="target" if np.isinf(scores[idx]) else "distance/core radius",
                      kept=blocker is None)
        if blocker is None:
            kept.append(int(idx))
        else:
            delta = points[idx]-points[blocker]
            record.update(suppressed_by_index=int(blocker),
                          distance_3d_um=float(np.linalg.norm(delta)),
                          distance_xy_um=float(np.linalg.norm(delta[1:])),
                          delta_z_um=float(abs(delta[0])))
        decisions.append(record)
    reproduced = points[kept].tolist() if kept else []
    if reproduced != np.asarray(after_nms, dtype=float).tolist():
        raise ValueError("Diagnostic NMS disagrees with actual NMS")
    rejected = {}
    for record in merge_audit["rejected"]:
        rejected[frozenset((tuple(record["a_zyx"]), tuple(record["b_zyx"])))] = record["reason"]
    close_pairs = []
    for i, a in enumerate(final):
        for b in final[i+1:]:
            delta = np.asarray(a)-np.asarray(b)
            xy = float(np.linalg.norm(delta[1:]))
            if xy > 20:
                continue
            gap = float(np.linalg.norm(delta))
            close_pairs.append(dict(
                a_zyx=list(a), b_zyx=list(b), distance_xy_um=xy,
                delta_z_um=float(abs(delta[0])), distance_3d_um=gap,
                nms_reason="distance_at_least_6_um" if gap >= radius else "unexpected",
                signal_merge_reason=rejected.get(
                    frozenset((tuple(a), tuple(b))), "not_compared_by_distance_policy")))
    report = dict(
        coordinate_order="ZYX", voxel_size_um=[1, 1, 1], radius_um=radius,
        suppression_rule="3D Euclidean distance strictly less than radius",
        counts=dict(pre_nms=len(coords), after_distance_nms=len(after_nms),
                    after_signal_merge=len(final)),
        pre_nms_zyx=np.asarray(coords).tolist(), after_distance_nms_zyx=after_nms,
        after_signal_merge_zyx=final, nms_decisions=decisions,
        final_pairs_with_xy_distance_at_most_20_um=close_pairs)
    Path(path).write_text(json.dumps(report, indent=2, allow_nan=False,
                                    default=lambda x: x.item())+"\n")
    return report
