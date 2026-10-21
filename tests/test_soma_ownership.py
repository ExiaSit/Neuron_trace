from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tempfile
import unittest
import numpy as np
import tifffile

from algorithms.tracing.signal_support import SignalEvidence
from algorithms.tracing.soma_ownership import TargetSomaOwnership
from algorithms.tracing.soma_signal_groups import filter_seeds


class OwnershipTests(unittest.TestCase):
    def ellipsoid(self):
        z, y, x = np.indices((64, 64, 72))
        body = ((z-32)/12.)**2 + ((y-32)/12.)**2 + ((x-36)/22.)**2 <= 1
        return np.where(body, 220, 3).astype(np.uint8)

    def test_off_center_points_share_body_and_target_stays_fixed(self):
        raw = self.ellipsoid()
        target, other = [32, 32, 23], [32, 32, 49]
        owner = TargetSomaOwnership(SignalEvidence(raw), target)
        self.assertTrue(owner.contains_pair(target, other))
        kept, audit = filter_seeds(raw, [other, target], target)
        self.assertEqual(kept, [target])
        self.assertTrue(audit["merged"][0]["evidence"]["same_target_region"])

    def test_close_bulbs_with_thick_bright_neurite_have_separate_basins(self):
        z, y, x = np.indices((48, 48, 72))
        a, b = [24, 24, 24], [24, 24, 44]
        body = ((z-24)**2+(y-24)**2+(x-24)**2 <= 7**2)
        body |= ((z-24)**2+(y-24)**2+(x-44)**2 <= 7**2)
        tube = ((z-24)**2+(y-24)**2 <= 3**2) & (x >= 24) & (x <= 44)
        raw = np.where(body | tube, 220, 3).astype(np.uint8)
        owner = TargetSomaOwnership(SignalEvidence(raw), a)
        self.assertFalse(owner.contains_pair(a, b))
        kept, _ = filter_seeds(raw, [a, b], a)
        self.assertEqual(len(kept), 2)

    def test_projected_overlap_at_different_depths_is_not_ownership(self):
        z, y, x = np.indices((64, 48, 48))
        a, b = [20, 24, 24], [40, 24, 24]
        body = ((z-20)**2+(y-24)**2+(x-24)**2 <= 7**2)
        body |= ((z-40)**2+(y-24)**2+(x-24)**2 <= 7**2)
        raw = np.where(body, 220, 3).astype(np.uint8)
        owner = TargetSomaOwnership(SignalEvidence(raw), a)
        self.assertFalse(owner.contains_pair(a, b))
        self.assertEqual(len(filter_seeds(raw, [a, b], a)[0]), 2)

    def test_other_auxiliary_component_cannot_use_ownership_exception(self):
        raw = self.ellipsoid()
        a, b = [32, 32, 23], [32, 32, 49]
        # Labels in a cropped auxiliary image exercise XYZ/ZYX origin handling.
        labels = np.zeros((30, 30, 50), np.int32)
        origin = np.array([17, 17, 10])
        labels[13:18, 13:18, 11:16] = 1
        labels[13:18, 13:18, 37:42] = 2
        owner = TargetSomaOwnership(SignalEvidence(raw), a, labels, origin)
        self.assertFalse(owner.contains_pair(a, b))
        self.assertEqual(owner.audit["target_auxiliary_component"], 1)

    def test_dark_gap_is_not_bridged_by_region_growth(self):
        raw = self.ellipsoid()
        raw[:, :, 34:39] = 3
        a, b = [32, 32, 23], [32, 32, 49]
        owner = TargetSomaOwnership(SignalEvidence(raw), a)
        self.assertFalse(owner.contains_pair(a, b))
        self.assertEqual(len(filter_seeds(raw, [a, b], a)[0]), 2)

    def test_no_transitive_use_by_absorbed_candidate(self):
        raw = self.ellipsoid()
        a, b, c = [32, 32, 23], [32, 32, 36], [32, 32, 49]
        owner = TargetSomaOwnership(SignalEvidence(raw), a)
        self.assertTrue(owner.contains_pair(a, b))
        self.assertFalse(owner.contains_pair(b, c))

    def test_fractional_metadata_keeps_original_coordinate(self):
        raw = self.ellipsoid()
        a, b = [32.2, 32.1, 23.3], [32, 32, 49]
        owner = TargetSomaOwnership(SignalEvidence(raw), a)
        self.assertTrue(owner.contains_pair(a, b))
        self.assertEqual(filter_seeds(raw, [b, a], a)[0], [a])

    def test_edge_crop_export_preserves_original_image_coordinates(self):
        raw = self.ellipsoid()[25:, 20:, 15:]
        target, other = [7, 12, 8], [7, 12, 34]
        before = raw.copy()
        owner = TargetSomaOwnership(SignalEvidence(raw), target)
        self.assertTrue(owner.contains_pair(target, other))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"region.tif"
            owner.save(path, raw.shape)
            region = tifffile.imread(path)
            self.assertEqual(region.shape, raw.shape)
            self.assertEqual(region[tuple(target)], 255)
            self.assertEqual(region[tuple(other)], 255)
        np.testing.assert_array_equal(raw, before)

    def test_diffuse_signal_does_not_merge_even_with_one_basin(self):
        raw = np.full((48, 48, 64), 220, np.uint8)
        a, b = [24, 24, 24], [24, 24, 40]
        self.assertEqual(len(filter_seeds(raw, [a, b], a)[0]), 2)


if __name__ == "__main__":
    unittest.main()
