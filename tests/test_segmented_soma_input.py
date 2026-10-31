import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile
from algorithms.tracing.skeleton_input import (
    load_segmented_soma, skeleton_with_segmented_soma, prepare_input,
)


class SegmentedSomaInputTests(unittest.TestCase):
    def fixture(self, root):
        crop = np.zeros((9, 11, 13), np.uint8)
        crop[2:6, 3:7, 4:8] = 255
        crop[7:9, 8:11, 10:13] = 255  # Independent neighboring soma.
        mask = root/'soma.tif'; meta = root/'soma.json'
        tifffile.imwrite(mask, crop)
        # Deliberately retain historical nominal end bounds one voxel larger.
        meta.write_text(json.dumps(dict(bounds_1um_zxy=[5, 15, 17, 31, 11, 23])))
        return mask, meta

    def test_zxy_metadata_places_zyx_crop_and_excludes_neighbor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); mask, meta = self.fixture(root)
            soma, audit = load_segmented_soma((30, 40, 50), [22, 15, 8], meta, mask)
            expected = np.zeros_like(soma); expected[7:11, 14:18, 21:25] = True
            np.testing.assert_array_equal(soma, expected)
            self.assertEqual(audit['voxels'], 64)
            self.assertEqual(audit['component_count'], 2)

    def test_empty_seed_never_snaps_to_nearby_component(self):
        with tempfile.TemporaryDirectory() as tmp:
            mask, meta = self.fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError, 'no estimated-soma fallback'):
                load_segmented_soma((30, 40, 50), [20, 15, 8], meta, mask)

    def test_addback_is_exact_even_outside_repaired_mask(self):
        mask = np.zeros((21, 21, 21), bool); mask[10, 10, 2:19] = True
        soma = np.zeros_like(mask); soma[7:14, 7:14, 7:14] = True
        soma[10, 14, 10] = True  # Segmented protrusion must not be smoothed away.
        before = mask.copy()
        sk, body, merged, a = skeleton_with_segmented_soma(mask, soma, [10, 10, 10])
        np.testing.assert_array_equal(mask, before)
        np.testing.assert_array_equal(body, soma)
        np.testing.assert_array_equal(merged, sk | soma)
        self.assertGreater(a['outside_repaired_mask'], 0)
        self.assertEqual(a['target_skeleton_unreachable'], 0)

    def test_disconnection_is_reported_without_inventing_soma_bridge(self):
        mask = np.zeros((21, 21, 21), bool); mask[10, 10, 2:19] = True
        soma = np.zeros_like(mask); soma[9:12, 9:12, 9:12] = True
        sk = mask.copy(); sk[10, 10, 12:15] = False
        with patch('algorithms.tracing.skeleton_input.skeletonize', return_value=sk):
            _, body, merged, a = skeleton_with_segmented_soma(mask, soma, [10, 10, 10])
        self.assertGreater(a['target_skeleton_unreachable'], 0)
        np.testing.assert_array_equal(body, soma)
        self.assertFalse(merged[10, 10, 13])

    def test_default_preparation_uses_segmentation_without_fitting_or_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); seg, meta = self.fixture(root)
            mask = np.zeros((30, 40, 50), np.uint8); mask[8, 15, 5:40] = 255
            repaired = root/'mask.tif'; tifffile.imwrite(repaired, mask)
            paths = dict(root=root, mask=repaired, soma_mask=seg, soma_json=meta,
                         raw=root/'unneeded_raw.tif')
            with patch('algorithms.mask_repair.guarded_continuation.target_body',
                       side_effect=AssertionError('must not estimate soma')):
                dest = prepare_input(dict(id='1', config={}), paths, [22, 15, 8])
            measured = tifffile.imread(root/'app2_input/1_soma_measured.tif')
            body = tifffile.imread(root/'app2_input/1_soma.tif')
            sk = tifffile.imread(root/'app2_input/1_skeleton.tif')
            np.testing.assert_array_equal(measured, body)
            np.testing.assert_array_equal(tifffile.imread(dest), sk | body)
            audit = json.loads((root/'logs/1/app2-input.json').read_text())
            self.assertEqual(audit['soma_source'], 'segmentation')

    def test_crop_clipping_does_not_wrap_negative_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); seg = root/'soma.tif'; meta = root/'soma.json'
            tifffile.imwrite(seg, np.ones((5, 5, 5), np.uint8))
            meta.write_text(json.dumps(dict(bounds_1um_zxy=[-2, 3, 2, 7, 3, 8])))
            soma, a = load_segmented_soma((10, 10, 10), [4, 5, 1], meta, seg)
            self.assertEqual(soma.sum(), 75)
            self.assertEqual(a['clipped_outside_image_voxels'], 50)
            self.assertFalse(soma[-1].any())
