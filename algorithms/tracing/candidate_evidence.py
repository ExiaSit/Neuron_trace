"""Reject unsupported soma candidates without altering raw images or masks.

All geometry is measured in 3D, in the 1-um image frame. The supplied target
is retained. These conservative heuristics are experimental, not cell identity.
"""
import numpy as np
from scipy import ndimage as ndi

DEFAULTS = dict(enabled=True, roi_radius=16, background_radius=32,
                background_rescue_core_fraction=.5, search_radius=3,
                min_contrast=10., min_snr=3., min_core_radius=2.5,
                min_core_voxels=40, max_axis_ratio=4.,
                min_mask_fraction=.30, strong_contrast=25.,
                strong_snr=5., strong_core_radius=3.5)


def measure(raw, support_mask, point, config=None):
    """Keep local evidence; rescue background contamination only for large cores.

    A quiet distant shell must not turn a small, locally unsupported peak into
    a competing soma. The local high-threshold core must already span at least
    half the analysis radius before an exterior background can rescue it.
    """
    cfg = dict(DEFAULTS, **(config or {}))
    local = _measure_with_background(raw, support_mask, point, cfg, cfg['roi_radius'])
    local['background_policy'] = 'local'
    minimum_radius = max(cfg['min_core_radius'],
                         cfg['roi_radius'] * cfg['background_rescue_core_fraction'])
    eligible = (not local['accepted']
                and 'weak_local_contrast' in local['reasons']
                and set(local['reasons']) <= {'weak_local_contrast', 'insufficient_mask_support'}
                and local.get('core_radius_um', 0.) >= minimum_radius
                and cfg['background_radius'] > cfg['roi_radius'])
    local['exterior_rescue_eligible'] = bool(eligible)
    if not eligible:
        return local
    exterior = _measure_with_background(raw, support_mask, point, cfg, cfg['background_radius'])
    if exterior['accepted']:
        exterior.update(background_policy='large_core_exterior_rescue',
                        exterior_rescue_eligible=True, local_evidence=local)
        return exterior
    local['exterior_evidence'] = exterior
    return local


def _measure_with_background(raw, support_mask, point, cfg, background_radius):
    p = np.rint(point).astype(int)
    if np.any(p < 0) or np.any(p >= raw.shape):
        return dict(accepted=False, reasons=["outside_image"])
    radius = int(cfg["roi_radius"])
    lo = np.maximum(0, p-radius)
    hi = np.minimum(raw.shape, p+radius+1)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    smooth = ndi.gaussian_filter(raw[sl].astype(np.float32), .8)
    local = p-lo
    grid = np.indices(smooth.shape)
    distance2 = sum((grid[d]-local[d])**2 for d in range(3))
    background_radius = max(radius, int(background_radius))
    bg_lo = np.maximum(0, p-background_radius)
    bg_hi = np.minimum(raw.shape, p+background_radius+1)
    bg_sl = tuple(slice(int(a), int(b)) for a, b in zip(bg_lo, bg_hi))
    bg_smooth = ndi.gaussian_filter(raw[bg_sl].astype(np.float32), .8)
    bg_grid = np.ogrid[tuple(slice(0, n) for n in bg_smooth.shape)]
    bg_distance2 = sum((axis-center)**2 for axis, center in zip(bg_grid, p-bg_lo))
    shell = bg_smooth[(bg_distance2 >= (background_radius*.65)**2)
                      & (bg_distance2 <= background_radius**2)]
    if shell.size == 0:
        return dict(accepted=False, reasons=["insufficient_background"])
    bg = float(np.median(shell))
    noise = max(1., float(1.4826*np.median(np.abs(shell-bg))))
    central = distance2 <= float(cfg["search_radius"])**2
    peak = float(np.max(smooth[central]))
    contrast = peak-bg
    snr = contrast/noise
    threshold = bg+max(5., .5*max(0., contrast))
    bright = smooth >= threshold
    labels, _ = ndi.label(bright, np.ones((3, 3, 3)))
    # Find a core only within 3 um of the candidate. Never snap across a dark
    # gap to a distant bright body, even if it is in this diagnostic ROI.
    edt = ndi.distance_transform_edt(np.pad(bright, 1))[1:-1, 1:-1, 1:-1]
    core_point = np.unravel_index(np.argmax(np.where(central, edt, -1)), edt.shape)
    label = int(labels[core_point])
    component = (labels == label) if label else np.zeros_like(bright)
    core = component & (edt >= 1.5)
    pts = np.argwhere(core)
    axis_ratio = None
    if len(pts) >= 4:
        eig = np.linalg.eigvalsh(np.cov(pts.astype(float), rowvar=False))
        axis_ratio = float(np.sqrt(max(eig[-1], 0.)/max(eig[0], 1e-6)))
    support = float(np.mean(support_mask[sl][core])) if len(pts) else 0.
    core_radius = float(edt[core_point])
    strong = (contrast >= cfg["strong_contrast"] and snr >= cfg["strong_snr"]
              and core_radius >= cfg["strong_core_radius"])
    reasons = []
    if contrast < cfg["min_contrast"] or snr < cfg["min_snr"]:
        reasons.append("weak_local_contrast")
    if core_radius < cfg["min_core_radius"] or len(pts) < cfg["min_core_voxels"]:
        reasons.append("no_thick_3d_core")
    if axis_ratio is None or axis_ratio > cfg["max_axis_ratio"]:
        reasons.append("sheet_or_elongated_core")
    if support < cfg["min_mask_fraction"] and not strong:
        reasons.append("insufficient_mask_support")
    return dict(accepted=not reasons, reasons=reasons,
                raw_at_candidate=int(raw[tuple(p)]), local_peak=peak,
                background=bg, noise_mad=noise, contrast=contrast, snr=snr,
                background_radius_um=background_radius,
                high_threshold=threshold, core_radius_um=core_radius,
                core_voxels=len(pts), core_axis_ratio=axis_ratio,
                core_mask_fraction=support, strong_raw_exception=bool(strong),
                core_peak_zyx=(np.array(core_point)+lo).tolist(),
                image_boundary_truncated=bool(np.any(p-radius < 0) or
                                              np.any(p+radius+1 > raw.shape)))


def filter_candidates(raw, support_mask, coords, scores, target, config=None):
    cfg = dict(DEFAULTS, **(config or {}))
    if len(coords) != len(scores):
        raise ValueError("Candidate coordinates and scores differ in length")
    if raw.shape != support_mask.shape:
        raise ValueError("Image/mask shape mismatch")
    kept_coords, kept_scores, records, rejected = [], [], [], []
    for index, (point, score) in enumerate(zip(coords, scores)):
        explicit_target = target is not None and np.allclose(point, target)
        if explicit_target:
            rec = dict(accepted=True, reasons=[], policy="protected_metadata_target")
        elif not cfg["enabled"]:
            rec = dict(accepted=True, reasons=[], policy="filter_disabled")
        else:
            rec = measure(raw, support_mask, point, cfg)
        rec.update(candidate_index=index, coordinate_zyx=list(map(float, point)),
                   score=float(score) if np.isfinite(score) else None)
        records.append(rec)
        if rec["accepted"]:
            kept_coords.append(point)
            kept_scores.append(score)
        else:
            rejected.append(point)
    return kept_coords, kept_scores, dict(
        version="candidate-evidence-v3", parameters=cfg,
        before_filter_zyx=np.asarray(coords).tolist(),
        after_filter_zyx=np.asarray(kept_coords).tolist(),
        rejected_zyx=np.asarray(rejected).tolist(), records=records,
        counts=dict(before=len(coords), kept=len(kept_coords), rejected=len(rejected)),
        policy="target protected; local 3D evidence; exterior-background rescue only for large compact cores")
