"""Selective CuPy acceleration with NumPy inputs/outputs and SciPy semantics.

Mask workers and scoped candidate generation opt in. Small ROIs and EDT nearest-index queries
stay on CPU: tie-breaking of equidistant owners affects reconstruction.
"""
import atexit
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from scipy import ndimage as cpu


class MaskNdimage:
    def __init__(self):
        self.cp = None
        self.gpu = None
        self.gpu_disabled = False
        self.stats = dict(backend=os.environ.get("RECONNECT_COMPUTE_BACKEND", "cpu"),
                          gpu_calls={}, gpu_seconds_including_transfers={},
                          cpu_calls={}, minimum_gpu_voxels=int(os.environ.get(
                              "RECONNECT_GPU_MIN_VOXELS", "262144")))
        atexit.register(self.report)

    def initialize(self):
        if self.cp is None:
            started = time.perf_counter()
            import cupy
            from cupyx.scipy import ndimage
            # Fail visibly if CUDA was explicitly requested but cannot run.
            device = cupy.cuda.runtime.getDeviceProperties(cupy.cuda.Device().id)
            self.cp, self.gpu = cupy, ndimage
            self.stats.update(cupy_version=cupy.__version__,
                              device=device["name"].decode(),
                              initialize_seconds=time.perf_counter()-started)
            print(f"[GPU] {self.stats['device']}，CuPy {cupy.__version__}；"
                  f"≥{self.stats['minimum_gpu_voxels']} 体素的滤波/EDT 使用 CUDA", flush=True)

    def __getattr__(self, name):
        return getattr(cpu, name)

    @contextmanager
    def input_slot(self):
        """Optionally serialize volume reads while leaving computation parallel."""
        configured = os.environ.get("RECONNECT_INPUT_READ_LOCK_PATH")
        if not configured:
            yield
            return
        path = Path(configured)
        path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        with path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            stats = self.stats.setdefault("input_serialization", dict(
                lock_path=str(path), acquisitions=0, wait_seconds=0., max_concurrent=1))
            stats["acquisitions"] += 1
            stats["wait_seconds"] += time.perf_counter()-started
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    @contextmanager
    def gpu_slot(self):
        """Serialize CUDA calls across workers, leaving their CPU work parallel."""
        path = Path(os.environ.get("RECONNECT_GPU_LOCK_PATH", str(
            Path(tempfile.gettempdir()) / f"neuron-tracing-cuda-{os.getuid()}.lock")))
        path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        with path.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            waited = time.perf_counter()-started
            stats = self.stats.setdefault("gpu_serialization", dict(
                lock_path=str(path), acquisitions=0, wait_seconds=0., max_concurrent=1))
            stats["acquisitions"] += 1
            stats["wait_seconds"] += waited
            lock.seek(0); lock.truncate()
            json.dump(dict(pid=os.getpid(), acquired_at=time.time()), lock)
            lock.flush()
            try:
                yield
            finally:
                # Arrays have already been transferred back to NumPy. Release
                # cached blocks before another process takes the CUDA slot.
                try:
                    if self.cp is not None:
                        self.cp.get_default_memory_pool().free_all_blocks()
                finally:
                    lock.seek(0); lock.truncate()
                    json.dump(dict(pid=None, released_at=time.time()), lock)
                    lock.flush()
                    fcntl.flock(lock, fcntl.LOCK_UN)

    def call(self, name, array, args, kwargs, eligible=True):
        backend = os.environ.get("RECONNECT_COMPUTE_BACKEND", "cpu")
        if backend not in ("cpu", "cuda"):
            raise ValueError(f"Unknown compute backend: {backend}")
        self.stats["backend"] = backend
        use_gpu = (backend == "cuda" and not self.gpu_disabled and eligible and isinstance(array, np.ndarray)
                   and array.size >= self.stats["minimum_gpu_voxels"])
        if not use_gpu:
            self.stats["cpu_calls"][name] = self.stats["cpu_calls"].get(name, 0)+1
            return getattr(cpu, name)(array, *args, **kwargs)
        try:
            with self.gpu_slot():
                self.initialize()
                started = time.perf_counter()
                gpu_args=tuple(self.cp.asarray(x) if isinstance(x,np.ndarray) else x for x in args)
                gpu_kwargs={k:self.cp.asarray(v) if isinstance(v,np.ndarray) else v for k,v in kwargs.items()}
                if name in ('binary_dilation','binary_erosion'):
                    # CuPy only implements repeated iterations in brute-force mode.
                    gpu_kwargs['brute_force']=True
                result = getattr(self.gpu, name)(self.cp.asarray(array), *gpu_args, **gpu_kwargs)
                result = self.cp.asnumpy(result)  # Synchronizes, includes transfer time.
        except Exception as exc:
            if self.cp is None or not isinstance(exc, self.cp.cuda.memory.OutOfMemoryError):
                raise
            # Keep this isolated worker on CPU after its first allocation
            # failure. A later candidate scope must not retry the same GPU.
            self.gpu_disabled = True
            self.stats.setdefault("cpu_fallbacks", []).append(dict(
                operation=name, reason="cuda_out_of_memory", error=str(exc)))
            print(f"[GPU] {name} 显存不足；本样本后续滤波/EDT 自动改用 CPU，继续处理。", flush=True)
            self.stats["cpu_calls"][name] = self.stats["cpu_calls"].get(name, 0)+1
            return getattr(cpu, name)(array, *args, **kwargs)
        self.stats["gpu_calls"][name] = self.stats["gpu_calls"].get(name, 0)+1
        times = self.stats["gpu_seconds_including_transfers"]
        times[name] = times.get(name, 0.)+time.perf_counter()-started
        return result

    def gaussian_filter(self, array, sigma, **kwargs):
        eligible = (getattr(array, "dtype", None) in (np.dtype("float32"), np.dtype("float64"))
                    and kwargs.get("output") is None
                    and not ({"radius", "axes"} & kwargs.keys()))
        return self.call("gaussian_filter", array, (sigma,), kwargs, eligible)

    def gaussian_filter1d(self, array, sigma, **kwargs):
        eligible = (getattr(array, "dtype", None) in (np.dtype("float32"), np.dtype("float64"))
                    and kwargs.get("output") is None and "radius" not in kwargs)
        return self.call("gaussian_filter1d", array, (sigma,), kwargs, eligible)

    def distance_transform_edt(self, array, **kwargs):
        eligible = (getattr(array, "ndim", 0) in (2, 3)
                    and not kwargs.get("return_indices", False)
                    and kwargs.get("return_distances", True)
                    and kwargs.get("distances") is None and kwargs.get("indices") is None)
        # SciPy's implicit background for all-foreground arrays is special.
        if (eligible and os.environ.get("RECONNECT_COMPUTE_BACKEND") == "cuda"
                and array.size >= self.stats["minimum_gpu_voxels"]):
            eligible = bool(np.any(array == 0))
        return self.call("distance_transform_edt", array, (), kwargs, eligible)

    def maximum_filter(self, array, **kwargs):
        return self.call('maximum_filter',array,(),kwargs,kwargs.get('output') is None)

    def minimum_filter(self, array, **kwargs):
        return self.call('minimum_filter',array,(),kwargs,kwargs.get('output') is None)

    def maximum_filter1d(self, array, size, **kwargs):
        return self.call('maximum_filter1d',array,(size,),kwargs,kwargs.get('output') is None)

    def convolve(self, array, weights, **kwargs):
        return self.call('convolve',array,(weights,),kwargs,kwargs.get('output') is None)

    def binary_dilation(self, array, structure=None, iterations=1, **kwargs):
        return self.call('binary_dilation',array,(structure,iterations),kwargs,
                         iterations>0 and kwargs.get('output') is None)

    def binary_erosion(self, array, structure=None, iterations=1, **kwargs):
        return self.call('binary_erosion',array,(structure,iterations),kwargs,
                         iterations>0 and kwargs.get('output') is None)

    def report(self):
        path = os.environ.get("RECONNECT_GPU_STATS")
        if path:
            Path(path).write_text(json.dumps(self.stats, indent=2)+"\n")
        if self.stats["gpu_calls"]:
            total = sum(self.stats["gpu_seconds_including_transfers"].values())
            print(f"[GPU] 完成 {sum(self.stats['gpu_calls'].values())} 次计算，"
                  f"运算及传输 {total:.2f} 秒；详情见阶段 gpu.json", flush=True)


ndi = MaskNdimage()
