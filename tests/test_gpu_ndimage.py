import os
import unittest
from unittest.mock import patch

import numpy as np
from scipy import ndimage as cpu

from utils.gpu_ndimage import MaskNdimage, candidate_backend


class CandidateScopeTests(unittest.TestCase):
    def test_candidate_cuda_scope_restores_cpu(self):
        with patch.dict(os.environ, RECONNECT_COMPUTE_BACKEND="cpu"):
            with candidate_backend("cuda", "test"):
                self.assertEqual(os.environ["RECONNECT_COMPUTE_BACKEND"], "cuda")
            self.assertEqual(os.environ["RECONNECT_COMPUTE_BACKEND"], "cpu")

    def test_exception_restores_missing_environment(self):
        with patch.dict(os.environ):
            os.environ.pop("RECONNECT_COMPUTE_BACKEND", None)
            with self.assertRaises(RuntimeError):
                with candidate_backend("cuda", "test_failure"):
                    raise RuntimeError("test")
            self.assertNotIn("RECONNECT_COMPUTE_BACKEND", os.environ)


class GpuNdimageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy
            if not cupy.cuda.runtime.getDeviceCount():
                raise unittest.SkipTest("CUDA device unavailable")
        except ImportError:
            raise unittest.SkipTest("CuPy unavailable")
        cls.environment = patch.dict(os.environ, RECONNECT_COMPUTE_BACKEND="cuda",
                                     RECONNECT_GPU_MIN_VOXELS="0")
        cls.environment.start()
        cls.backend = MaskNdimage()

    @classmethod
    def tearDownClass(cls):
        cls.environment.stop()

    def test_gaussian_matches_cpu_for_actual_sigmas_and_boundaries(self):
        raw = np.random.default_rng(51).uniform(0, 255, (17, 29, 31)).astype(np.float32)
        for sigma in [1., 6., (0, .6, .6), (0, 4, 4)]:
            expected = cpu.gaussian_filter(raw, sigma)
            actual = self.backend.gaussian_filter(raw, sigma)
            self.assertEqual(actual.dtype, expected.dtype)
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=5e-5)
        np.testing.assert_allclose(
            self.backend.gaussian_filter1d(raw, .6, axis=0),
            cpu.gaussian_filter1d(raw, .6, axis=0), rtol=1e-6, atol=5e-5)
        self.assertGreater(self.backend.stats["gpu_calls"]["gaussian_filter"], 0)

    def test_exact_distance_and_threshold_decisions(self):
        rng = np.random.default_rng(21)
        for shape in [(27, 31), (13, 29, 31)]:
            mask = rng.random(shape) > .15
            for sampling in [None, [2.0]*len(shape)]:
                actual = self.backend.distance_transform_edt(mask, sampling=sampling)
                expected = cpu.distance_transform_edt(mask, sampling=sampling)
                self.assertEqual(actual.dtype, expected.dtype)
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
                for threshold in [2., 2.5, 3.5, 5.]:
                    np.testing.assert_array_equal(actual >= threshold, expected >= threshold)

    def test_nearest_owner_indices_keep_scipy_tie_breaking(self):
        mask = np.ones((11, 13, 15), bool)
        mask[5, 6, 3] = mask[5, 6, 11] = False
        count = self.backend.stats["gpu_calls"].get("distance_transform_edt", 0)
        actual = self.backend.distance_transform_edt(mask, return_indices=True)
        expected = cpu.distance_transform_edt(mask, return_indices=True)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(count, self.backend.stats["gpu_calls"].get("distance_transform_edt", 0))

    def test_all_foreground_background_and_cpu_switch(self):
        for fill in [0, 1]:
            mask = np.full((7, 9, 11), fill, np.uint8)
            np.testing.assert_array_equal(self.backend.distance_transform_edt(mask),
                                          cpu.distance_transform_edt(mask))
        with patch.dict(os.environ, RECONNECT_COMPUTE_BACKEND="cpu"):
            raw = np.arange(125, dtype=np.float32).reshape(5, 5, 5)
            np.testing.assert_array_equal(self.backend.gaussian_filter(raw, 1),
                                          cpu.gaussian_filter(raw, 1))


if __name__ == "__main__":
    unittest.main()
