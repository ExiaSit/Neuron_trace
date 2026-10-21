"""Local 3-D soma ownership, measured from raw signal in 1-um coordinates.

Watershed basins distinguish thick bodies even when a bright neurite connects
them. Ownership only supplements the target's merge decision; it never grows
through already merged candidates or changes the raw/repair mask.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import h_maxima
from skimage.segmentation import watershed


class TargetSomaOwnership:
    def __init__(self, evidence, target, auxiliary_labels=None, auxiliary_origin=None):
        self.target_coordinate = np.asarray(target, float)
        self.target = np.rint(target).astype(int)
        self.lo = np.maximum(0, self.target - 48)
        hi = np.minimum(evidence.raw.shape, self.target + 49)
        self.slices = tuple(slice(int(a), int(b)) for a, b in zip(self.lo, hi))
        v = evidence.smooth[self.slices]
        self.region = np.zeros(v.shape, bool)
        self.audit = dict(
            status="unmeasured", coordinate_order="ZYX", origin_zyx=self.lo.tolist(),
            shape_zyx=list(v.shape), target_zyx=self.target_coordinate.tolist(),
            parameters=dict(roi_half_width_um=48, smoothing_sigma_um=.65,
                            intensity_floor=35., intensity_fraction=.30,
                            minimum_interior_radius_um=2.5,
                            minimum_peak_radius_um=3.5, peak_prominence_um=1.,
                            maximum_peak_distance_in_radii=3.),
            candidate_decisions=[])
        q = self.target - self.lo
        if np.any(q < 0) or np.any(q >= v.shape):
            self.audit["status"] = "target_outside_image"
            return
        threshold = max(35., .30 * float(v[tuple(q)]))
        # Padding supplies real background outside the ROI/image; EDT on an
        # all-foreground crop otherwise has misleading boundary distances.
        d = ndi.distance_transform_edt(np.pad(v > threshold, 1))[1:-1, 1:-1, 1:-1]
        interior = d >= 2.5
        if not interior[tuple(q)]:
            self.audit["status"] = "target_outside_thick_interior"
            return
        maxima = h_maxima(d, 1.) & (d >= 3.5)
        markers, count = ndi.label(maxima, np.ones((3, 3, 3)))
        if not count:
            self.audit["status"] = "no_soma_peak"
            return
        basins = watershed(-d, markers, mask=interior, connectivity=np.ones((3, 3, 3)))
        own = int(basins[tuple(q)])
        if not own:
            self.audit["status"] = "target_without_basin"
            return
        basin = basins == own
        peak = np.array(np.unravel_index(np.where(basin, d, 0).argmax(), d.shape))
        radius = float(d[tuple(peak)])
        if np.linalg.norm(q - peak) > 3. * radius:
            self.audit["status"] = "target_too_far_from_soma_peak"
            return
        axes = np.ogrid[tuple(slice(0, n) for n in v.shape)]
        within = sum((axis - center) ** 2 for axis, center in zip(axes, peak)) <= (3. * radius) ** 2
        region = basin & within
        # Independent auxiliary components cannot be absorbed by this new
        # exception, even if the raw distance field has an ambiguous basin.
        protected = np.zeros(v.shape, bool)
        if auxiliary_labels is not None:
            origin = np.asarray(auxiliary_origin, int)
            local = self.target - origin
            aux_own = 0
            if np.all(local >= 0) and np.all(local < auxiliary_labels.shape):
                aux_own = int(auxiliary_labels[tuple(local)])
            lower = np.maximum(self.lo, origin)
            upper = np.minimum(hi, origin + np.array(auxiliary_labels.shape))
            if np.all(upper > lower):
                dst = tuple(slice(int(a), int(b)) for a, b in zip(lower-self.lo, upper-self.lo))
                src = tuple(slice(int(a), int(b)) for a, b in zip(lower-origin, upper-origin))
                labels = auxiliary_labels[src]
                protected[dst] = (labels > 0) & (labels != aux_own)
            self.audit["target_auxiliary_component"] = aux_own
        allowed = region & ~protected
        seed = np.zeros_like(allowed)
        seed[tuple(q)] = allowed[tuple(q)]
        self.region = ndi.binary_propagation(seed, np.ones((3, 3, 3)), mask=allowed)
        self.audit.update(status="measured_target_soma", threshold=threshold,
                          basin_count=int(count), target_basin=own,
                          peak_zyx=(peak + self.lo).tolist(), peak_radius_um=radius,
                          region_voxels=int(self.region.sum()),
                          protected_auxiliary_voxels=int(protected.sum()))

    def contains_pair(self, a, b):
        """Only a pair anchored at the original target can use this evidence."""
        a, b = np.asarray(a), np.asarray(b)
        if np.linalg.norm(a-self.target_coordinate) > 1e-6:
            return False
        p = np.rint(b).astype(int) - self.lo
        inside = bool(np.all(p >= 0) and np.all(p < self.region.shape)
                      and self.region[tuple(p)])
        self.audit["candidate_decisions"].append(dict(candidate_zyx=b.tolist(),
                                                      same_target_region=inside))
        return inside

    def save(self, path, shape):
        """Full image coordinates for direct overlay in Vaa3D."""
        import tifffile
        volume = np.zeros(shape, np.uint8)
        volume[self.slices] = self.region.astype(np.uint8) * 255
        tifffile.imwrite(path, volume, compression="zlib", metadata={"axes": "ZYX"})
