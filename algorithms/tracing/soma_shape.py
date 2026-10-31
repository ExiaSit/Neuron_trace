"""Image-guided, smooth target soma for APP2; never modifies tracing masks."""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import ball


def signal_surface(raw, mask, measured, anchor, seed):
    """Blend measured image boundary and an oriented ellipsoid in distance space.

    The ellipsoid is a weak shape reference, not a replacement for image data.
    Work on a small ROI at the pipeline's isotropic 1 um resolution. The caller
    checks skeleton connectivity and falls back to the measured soma if needed.
    """
    report = dict(method='signal_ellipsoid_distance_surface', status='rejected',
                  signal_fraction=.4, image_weight=.5, smoothing_sigma_um=1.2)
    if raw.shape != mask.shape or measured.shape != mask.shape:
        raise ValueError('Raw, mask and measured soma must have matching shapes')
    points = np.argwhere(anchor)
    if len(points) < 32:
        return None, dict(report, reason='small_anchor')
    points_all = np.argwhere(measured)
    lo = np.maximum(0, points_all.min(0)-8)
    hi = np.minimum(mask.shape, points_all.max(0)+9)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    body = measured[sl]
    fixed = mask[sl]
    center_seed = tuple(np.asarray(seed)-lo)
    center = points.mean(0)-lo
    covariance = np.cov((points-lo).T, bias=True)
    eigenvalues, rotation = np.linalg.eigh(covariance)
    axes = np.sqrt(5*np.maximum(eigenvalues, 0))
    if axes.min() < 2 or axes.max()/axes.min() > 4:
        return None, dict(report, reason='uncertain_shape')
    coords = np.moveaxis(np.indices(body.shape), 0, -1)-center
    ellipsoid = np.sum((coords @ rotation / axes)**2, axis=-1) <= 1
    sig = ndi.gaussian_filter(np.asarray(raw[sl], dtype=np.float32), .8)
    ring = ndi.binary_dilation(body, ball(5)) & ~ndi.binary_dilation(body, ball(2))
    if ring.sum() < 32:
        return None, dict(report, reason='insufficient_background')
    background = float(np.median(sig[ring]))
    peak = float(np.percentile(sig[anchor[sl]], 80))
    amplitude = peak-background
    report.update(background=background, reference_intensity=peak,
                  center_zyx=(center+lo).tolist(), semiaxes_um=axes.tolist())
    if amplitude < 10:
        return None, dict(report, reason='low_contrast')
    threshold = background+.4*amplitude
    support = ndi.binary_dilation(ellipsoid, ball(3)) & fixed
    target = (sig >= threshold) & support
    structure = np.ones((3, 3, 3), bool)
    labels, _ = ndi.label(target, structure)
    component = labels[center_seed]
    if not component:
        return None, dict(report, reason='seed_lacks_image_support')
    target = labels == component

    def signed_distance(volume):
        return ndi.distance_transform_edt(volume)-ndi.distance_transform_edt(~volume)

    field = ndi.gaussian_filter(.5*signed_distance(target)+
                                .5*signed_distance(ellipsoid), 1.2)
    proposed = field > 0
    candidate = proposed & fixed
    labels, _ = ndi.label(candidate, structure)
    component = labels[center_seed]
    if not component:
        return None, dict(report, reason='seed_lost_after_smoothing')
    candidate = labels == component
    retained = float(candidate.sum()/body.sum())
    weak = float(np.mean(sig[candidate] < background+.15*amplitude))
    report.update(threshold=threshold, volume_ratio=retained,
                  weak_signal_fraction=weak,
                  clipped_outside_mask_voxels=int((proposed & ~fixed).sum()),
                  added_to_measured=int((candidate & ~body).sum()),
                  removed_from_measured=int((body & ~candidate).sum()))
    if not .8 <= retained <= 1.15 or weak > .05:
        return None, dict(report, reason='volume_or_signal_guard')
    result = np.zeros_like(measured)
    result[sl] = candidate
    return result, dict(report, status='proposed')
