"""Geometric regressions for signal-following soma completion (1 um voxels)."""
import os
import unittest
from unittest.mock import patch

import numpy as np
from scipy import ndimage

from algorithms.mask_repair.soma_root_repair import restore_roots, surface_evidence


def ellipsoid():
    z, y, x = np.indices((65, 65, 65)) - 32
    radius = np.sqrt((z / 1.4)**2 + y*y + x*x)
    raw = (10 + 210 / (1 + np.exp(np.clip((radius - 9)/.6, -50, 50)))).astype('uint8')
    mask = z*z + y*y + x*x <= 25
    return raw, mask


class SomaRootShapeTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, RECONNECT_COMPUTE_BACKEND='cpu')
        env.start()
        self.addCleanup(env.stop)

    def test_round_caps_follow_signal_beyond_seed_band(self):
        raw, mask = ellipsoid()
        bodies=[]
        result, added, report = restore_roots(raw, mask, completed_somas=bodies)
        self.assertGreater(added.sum(), 3000)
        # A physical rounded boundary should taper at both poles, not end in
        # a large planar cap at seed +/-4 slices (the former implementation).
        area = result.sum((1, 2))
        active = np.flatnonzero(area)
        self.assertGreater(active[-1] - active[0], 22)
        self.assertLess(area[active[0]], .2 * area.max())
        self.assertLess(area[active[-1]], .2 * area.max())
        # The known ellipsoid surface is the sigmoid half-height. Allow its
        # sub-voxel transition band; the old truncated cylinder fails this.
        expected = raw >= 115
        self.assertGreater((result & expected).sum() / (result | expected).sum(), .85)
        self.assertTrue(result[mask].all())
        self.assertEqual(report['components_before'], report['components_after'])
        anchors=[a for a in report['anchors'] if 'completed_soma_top_z' in a]
        self.assertTrue(anchors)
        self.assertEqual(min(a['completed_soma_top_z'] for a in anchors),int(active[0]))
        self.assertTrue(bodies)
        for body in bodies:
            points=body['points_zyx']
            self.assertTrue(result[tuple(points.T)].all())
            self.assertEqual(body['top_z'],int(points[:,0].min()))

    def test_original_neurite_attachment_is_preserved(self):
        raw, mask = ellipsoid()
        mask[32, 32, 32:60] = True
        raw[32, 32, 32:60] = 210
        result, added, _ = restore_roots(raw, mask)
        self.assertGreater(added.sum(), 1000)
        self.assertTrue(result[mask].all())
        self.assertEqual(ndimage.label(result, np.ones((3,3,3)))[1], 1)

    def test_adjacent_somas_remain_separate(self):
        raw, mask = ellipsoid()
        raw = np.maximum(ndimage.shift(raw, (0,0,-10), order=0, cval=10),
                         ndimage.shift(raw, (0,0,10), order=0, cval=10))
        mask = (ndimage.shift(mask, (0,0,-10), order=0) |
                ndimage.shift(mask, (0,0,10), order=0))
        result, added, report = restore_roots(raw, mask)
        self.assertGreater(added.sum(), 1000)
        self.assertTrue(result[mask].all())
        self.assertEqual(ndimage.label(result, np.ones((3,3,3)))[1], 2)
        self.assertEqual(report['components_before'], report['components_after'])

    def test_uniform_bright_volume_is_not_a_soma(self):
        _, mask = ellipsoid()
        result, added, _ = restore_roots(np.full(mask.shape, 180, np.uint8), mask)
        self.assertFalse(added.any())
        np.testing.assert_array_equal(result, mask)

    def test_axial_sheet_does_not_pass_on_caps_alone(self):
        z, y, x = np.indices((49,49,49)) - 24
        region = (np.abs(z) <= 3) & (x*x+y*y <= 18**2)
        signal = np.where(np.abs(z) <= 3, 200., 10.)
        evidence = surface_evidence(signal, region, np.full(49,10.), np.zeros(49))
        self.assertLess(evidence['lateral_supported_fraction'], .55)


if __name__ == '__main__':
    unittest.main()
