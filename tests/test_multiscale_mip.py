import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import ndimage as ndi

from algorithms.mask_repair.mip_backmap import backmap, legacy_backmap
from algorithms.mask_repair.multiscale_mip import (
    configuration, window_ranges, supported_maxima, multiscale_seeds)


class MultiscaleMipTests(unittest.TestCase):
    def test_windows_overlap_and_cover_end_without_short_last_slab(self):
        windows = window_ranges(37, 12, .5)
        self.assertEqual(windows[-1], (25, 37))
        count = np.zeros(37, int)
        for start, stop in windows:
            self.assertEqual(stop-start, 12)
            count[start:stop] += 1
        self.assertTrue(np.all(count > 0))
        self.assertTrue(all(b > c for (_, b), (c, _) in zip(windows, windows[1:])))

    def test_short_dimension_has_one_window(self):
        self.assertEqual(window_ranges(3, 16), [(0, 3)])
        self.assertEqual(window_ranges(1, 16), [(0, 1)])

    def test_depth_is_not_filled_as_a_column(self):
        raw = np.zeros((48, 24, 30), np.uint8)
        raw[8:11, 10:13, 3:27] = 100
        contrast = raw.astype(np.float32)*.2
        guide, classes, _, _, stats = backmap(raw, contrast)
        self.assertFalse(guide[20:40].any())
        self.assertFalse(classes[20:40].any())
        self.assertEqual(stats['mode'], 'multiscale')

    def test_hidden_depths_recovered_in_overlapping_slabs(self):
        raw = np.zeros((48, 28, 36), np.uint8)
        raw[8:11, 12:15, 4:32] = 180
        raw[30:33, 12:15, 4:32] = 90
        contrast = raw.astype(np.float32)*.2
        seeds, audit = multiscale_seeds(raw, contrast, configuration())
        self.assertTrue(seeds[8:11].any())
        self.assertTrue(seeds[30:33].any())
        self.assertEqual([x['plane'] for x in audit['axes']], ['XY', 'XZ', 'YZ'])
        self.assertGreater(len(audit['axes'][0]['scales']), 1)

    def test_union_is_equivariant_under_axis_permutations(self):
        rng = np.random.default_rng(30)
        raw = ndi.gaussian_filter(rng.random((19, 23, 29))*100, .8)
        contrast = raw-ndi.gaussian_filter(raw, 3)
        cfg = configuration(dict(thicknesses=[6, 12]))
        expected, _ = multiscale_seeds(raw, contrast, cfg)
        for axes in [(2, 1, 0), (1, 2, 0)]:
            actual, _ = multiscale_seeds(raw.transpose(axes), contrast.transpose(axes), cfg)
            np.testing.assert_array_equal(actual, expected.transpose(axes))

    def test_legacy_mode_preserves_original_arrays_exactly(self):
        rng = np.random.default_rng(2)
        raw = rng.integers(0, 255, (13, 17, 21), dtype=np.uint8)
        contrast = raw.astype(np.float32)-120
        expected = legacy_backmap(raw, contrast)
        actual = backmap(raw, contrast, dict(mode='legacy'))
        for a, b in zip(expected[:4], actual[:4]):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(actual[4]['mode'], 'legacy')

    def test_no_new_evidence_in_dark_or_negative_contrast(self):
        raw = np.full((17, 23, 29), 4, np.uint8)
        contrast = np.zeros(raw.shape, np.float32)
        guide, classes, *_ = backmap(raw, contrast)
        self.assertFalse(guide.any())
        self.assertFalse(classes.any())

    def test_boost_bounded_and_inputs_unchanged(self):
        rng = np.random.default_rng(5)
        raw = rng.integers(0, 255, (17, 23, 29), dtype=np.uint8)
        contrast = raw.astype(np.float32)/15-4
        before_raw, before_contrast = raw.copy(), contrast.copy()
        guide, *_ = backmap(raw, contrast)
        legacy, *_ = legacy_backmap(raw, contrast)
        self.assertTrue(np.all(guide >= legacy))
        self.assertTrue(np.all(guide-contrast <= 3.00001))
        np.testing.assert_array_equal(raw, before_raw)
        np.testing.assert_array_equal(contrast, before_contrast)
        self.assertTrue(np.all(guide[contrast <= .75] == contrast[contrast <= .75]))

    def test_isolated_column_cannot_vote_for_itself(self):
        raw = np.zeros((16, 9, 11), np.uint8)
        raw[5, 4, 5] = 100
        _, valid = supported_maxima(raw, raw.astype(float))
        self.assertFalse(valid.any())

    def test_config_propagates_to_child_environment(self):
        cfg = dict(mode='legacy', thicknesses=[6, 12], overlap=.75)
        with patch.dict(os.environ, RECONNECT_MIP_CONFIG=json.dumps(cfg)):
            self.assertEqual(configuration()['mode'], 'legacy')
            self.assertEqual(configuration()['overlap'], .75)

    def test_reject_invalid_settings_and_shapes(self):
        for cfg in [dict(overlap=0), dict(overlap=1), dict(thicknesses=[0]),
                    dict(thicknesses=[3.5]), dict(min_axes=1), dict(typo=True)]:
            with self.assertRaises(ValueError):
                configuration(cfg)
        with self.assertRaises(ValueError):
            backmap(np.zeros((2, 3, 4)), np.zeros((2, 3, 5)))


if __name__ == '__main__':
    unittest.main()
