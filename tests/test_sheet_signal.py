import unittest
import tempfile
from pathlib import Path

import numpy as np

from algorithms.tracing.signal_support import SignalEvidence
from algorithms.tracing.sheet_signal import prune_sheet_supported, completed_soma_protection
from tests.test_signal_support import tube, swc


class SheetSignalTests(unittest.TestCase):
    def setUp(self):
        self.shape = (48, 64, 96)
        self.z, self.y, self.x = np.indices(self.shape)
        self.points = np.array([[24, 32, x] for x in range(10, 81)])

    def sheet(self):
        return 3+130*np.exp(-((self.z-24)/1.8)**2/2)

    def cut(self, raw, points=None, protected=None):
        p = self.points if points is None else points
        return prune_sheet_supported(swc(p), SignalEvidence(np.clip(raw, 0, 255).astype('uint8')),
                                     p[0, ::-1], protected_mask=protected)

    def test_long_bright_sheet_is_cut(self):
        kept, removed, report = self.cut(self.sheet())
        self.assertTrue(removed); self.assertLess(len(kept), len(self.points))
        self.assertGreaterEqual(report[0]['sheet_length_um'], 6)

    def test_bright_tissue_below_surface_does_not_hide_sheet(self):
        raw = np.maximum(self.sheet(), np.where(self.z > 24, 125, 3))
        self.assertTrue(self.cut(raw)[1])

    def test_true_narrow_and_thick_round_tubes_survive(self):
        for width in [.8, 3., 6., 12.]:
            raw = 3+130*np.exp(-((self.z-24)**2+(self.y-32)**2)/(2*width**2))
            with self.subTest(width=width):
                self.assertFalse(self.cut(raw)[1])

    def test_localized_tube_on_sheet_survives(self):
        raw = self.sheet()+110*np.exp(-((self.z-24)**2+(self.y-32)**2)/2)
        self.assertFalse(self.cut(raw)[1])

    def test_short_sheet_crossing_survives(self):
        raw = tube(self.points, shape=self.shape).astype(float)
        raw = np.maximum(raw, np.where((self.x >= 40) & (self.x <= 44), self.sheet(), 3))
        self.assertFalse(self.cut(raw)[1])

    def test_dark_and_uniform_volume_are_not_sheet_evidence(self):
        for value in [3, 130]:
            with self.subTest(value=value):
                self.assertFalse(self.cut(np.full(self.shape, value))[1])

    def test_protected_downstream_soma_vetoes_cut(self):
        protected = np.zeros(self.shape, bool); protected[24, 32, 80] = True
        self.assertFalse(self.cut(self.sheet(), protected=protected)[1])

    def test_tilted_sheet_uses_3d_orientation(self):
        raw = 3+130*np.exp(-((self.z-24+self.y-32)/np.sqrt(2)/1.8)**2/2)
        self.assertTrue(self.cut(raw)[1])

    def test_soma_and_image_boundary_are_protected(self):
        self.assertFalse(self.cut(self.sheet(), protected=np.ones(self.shape, bool))[1])
        p = self.points.copy(); p[:, 1] = 3
        self.assertFalse(self.cut(self.sheet(), points=p)[1])

    def test_true_curved_branch_survives(self):
        a = np.linspace(-1.2, 1.2, 70)
        p = np.column_stack([np.full(70,24), 32+15*np.sin(a), 40+15*np.cos(a)])
        self.assertFalse(self.cut(tube(p, shape=self.shape), points=p)[1])

    def test_completed_soma_protection_margin_and_fallback(self):
        fallback = np.ones(self.shape, bool)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'somas.npz'
            mask, source = completed_soma_protection(path, self.shape, fallback)
            self.assertIs(mask, fallback)
            np.savez(path, body_0=np.array([[24,32,40]]))
            mask, source = completed_soma_protection(path, self.shape, fallback)
            self.assertTrue(mask[24,32,42]); self.assertFalse(mask[24,32,43])
            self.assertEqual(int(mask.sum()),33)
            self.assertIn('shape_validated',source)

    def test_invalid_completed_soma_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'somas.npz'
            np.savez(path, body_0=np.array([[-1,32,40]]))
            with self.assertRaises(ValueError):
                completed_soma_protection(path,self.shape,None)


if __name__ == '__main__':
    unittest.main()
