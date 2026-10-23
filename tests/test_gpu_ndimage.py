import os
import multiprocessing
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from scipy import ndimage as cpu

from utils.gpu_ndimage import MaskNdimage, candidate_backend


def probe_gpu_slot(path, barrier, events):
    os.environ['RECONNECT_GPU_LOCK_PATH'] = path
    backend = MaskNdimage()
    events.put(('cpu', os.getpid(), time.monotonic()))
    barrier.wait(timeout=5)
    with backend.gpu_slot():
        events.put(('start', os.getpid(), time.monotonic()))
        time.sleep(.05)
        events.put(('end', os.getpid(), time.monotonic()))


class GpuSerializationTests(unittest.TestCase):
    def test_processes_share_one_gpu_slot_and_cpu_work_remains_parallel(self):
        context = multiprocessing.get_context('fork')
        barrier = context.Barrier(3); events = context.Queue()
        with tempfile.TemporaryDirectory() as directory:
            processes = [context.Process(target=probe_gpu_slot, args=(
                str(Path(directory)/'gpu.lock'), barrier, events)) for _ in range(3)]
            for process in processes:process.start()
            recorded = [events.get(timeout=8) for _ in range(9)]
            for process in processes:
                process.join(timeout=5)
                self.assertEqual(process.exitcode, 0)
            active = maximum = 0
            for kind, _, _ in sorted(recorded, key=lambda event:event[2]):
                if kind == 'start':active += 1; maximum = max(maximum, active)
                elif kind == 'end':active -= 1
            self.assertEqual(maximum, 1)
            self.assertEqual(active, 0)
            self.assertLess(max(t for kind,_,t in recorded if kind=='cpu'),
                            min(t for kind,_,t in recorded if kind=='start'))

    def test_exception_releases_slot(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                os.environ, RECONNECT_GPU_LOCK_PATH=str(Path(directory)/'gpu.lock')):
            backend = MaskNdimage()
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                with backend.gpu_slot():raise RuntimeError('injected')
            with backend.gpu_slot():pass
            self.assertEqual(backend.stats['gpu_serialization']['acquisitions'], 2)


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


class CpuFallbackTests(unittest.TestCase):
    def test_oom_returns_cpu_result_and_later_scopes_do_not_retry_gpu(self):
        class AllocationFailed(MemoryError):
            pass
        for operation in ("gaussian_filter", "distance_transform_edt"):
            with self.subTest(operation=operation), patch.dict(
                    os.environ, RECONNECT_COMPUTE_BACKEND="cuda",
                    RECONNECT_GPU_MIN_VOXELS="0"):
                backend = MaskNdimage()
                pool = SimpleNamespace(free_all_blocks=Mock())
                upload = Mock(side_effect=AllocationFailed("injected allocation failure"))
                backend.cp = SimpleNamespace(
                    asarray=upload,
                    cuda=SimpleNamespace(memory=SimpleNamespace(OutOfMemoryError=AllocationFailed)),
                    get_default_memory_pool=lambda: pool)
                backend.gpu = SimpleNamespace(**{operation: Mock()})
                raw = np.arange(125, dtype=np.float32).reshape(5, 5, 5)
                array = raw if operation == "gaussian_filter" else raw > 10
                args = (1.,) if operation == "gaussian_filter" else ()
                before = array.copy()
                actual = getattr(backend, operation)(array, *args)
                np.testing.assert_array_equal(actual, getattr(cpu, operation)(array, *args))
                np.testing.assert_array_equal(array, before)
                self.assertTrue(backend.gpu_disabled)
                # Re-entering a CUDA scope does not retry GPU allocation.
                with candidate_backend("cuda", "after_oom"):
                    again = backend.gaussian_filter(raw, 1.)
                np.testing.assert_array_equal(again, cpu.gaussian_filter(raw, 1.))
                upload.assert_called_once()
                pool.free_all_blocks.assert_called_once()
                self.assertEqual(len(backend.stats["cpu_fallbacks"]), 1)
                self.assertEqual(backend.stats["gpu_calls"], {})

    def test_non_oom_errors_are_not_hidden(self):
        class AllocationFailed(MemoryError):
            pass
        with patch.dict(os.environ, RECONNECT_COMPUTE_BACKEND="cuda", RECONNECT_GPU_MIN_VOXELS="0"):
            backend = MaskNdimage()
            backend.cp = SimpleNamespace(
                asarray=Mock(side_effect=ValueError("invalid GPU argument")),
                get_default_memory_pool=lambda: SimpleNamespace(free_all_blocks=lambda:None),
                cuda=SimpleNamespace(memory=SimpleNamespace(OutOfMemoryError=AllocationFailed)))
            backend.gpu = SimpleNamespace(gaussian_filter=Mock())
            with self.assertRaisesRegex(ValueError, "invalid GPU argument"):
                backend.gaussian_filter(np.zeros((3, 4, 5), np.float32), 1.)
            self.assertFalse(backend.gpu_disabled)


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
