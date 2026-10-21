"""Call APP2's unmodified full-3D fastmarching_dt, retaining float output.

GSDT values are intensity-weighted path costs, never geometric radii. This is
the 3D transform, not APP2's separate XY-only automatic-root initialization.
"""
import ctypes
import fcntl
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

SOURCE = Path(__file__).resolve().parents[2] / "vendor/app2_gsdt"


@lru_cache(maxsize=1)
def _library():
    sources = sorted(SOURCE.glob("*.h")) + [SOURCE / "wrapper.cpp"]
    digest = hashlib.sha256(b"".join(p.read_bytes() for p in sources)).hexdigest()
    cache = SOURCE / "build"
    cache.mkdir(exist_ok=True)
    library = cache / ("gsdt-" + digest[:20] + ".so")
    with (cache / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.exists():
            temporary = library.with_suffix(f".{os.getpid()}.tmp.so")
            command = ["g++", "-O3", "-std=c++11", "-shared", "-fPIC", "-pthread",
                       str(SOURCE / "wrapper.cpp"), "-o", str(temporary)]
            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
                temporary.replace(library)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f"APP2 GSDT build failed: {exc.stderr}") from exc
            finally:
                temporary.unlink(missing_ok=True)
    lib = ctypes.CDLL(str(library))
    lib.app2_gsdt.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int] * 6
    lib.app2_gsdt.restype = ctypes.c_int
    return lib, digest


def distance_transform(raw, background_threshold, connectivity=2):
    """Return (float32 ZYX field, audit). connectivity=2 means 18 neighbors.

    Preserve native intensity units and APP2's integer background threshold;
    do not min/max normalize the field to 8 bit as a display plugin might.
    """
    raw = np.asarray(raw)
    if raw.ndim != 3 or not raw.size or raw.size >= 2**31:
        raise ValueError("APP2 GSDT requires a nonempty 3D volume smaller than 2^31 voxels")
    if connectivity not in (1, 2, 3):
        raise ValueError("APP2 connectivity must be 1 (6), 2 (18), or 3 (26)")
    if raw.dtype not in (np.dtype('uint8'), np.dtype('uint16'), np.dtype('float32')):
        raise ValueError("APP2 GSDT supports uint8, uint16 and float32 images")
    if not np.isfinite(raw).all() or np.any(raw < 0):
        raise ValueError("APP2 GSDT requires finite nonnegative intensities")
    if not np.isfinite(background_threshold) or not 0 <= background_threshold < 2**31:
        raise ValueError("APP2 background threshold must fit a nonnegative int32")
    threshold = int(background_threshold)
    if not np.any(raw <= threshold):
        raise ValueError("APP2 GSDT has no background seeds; refusing an infinite field")
    data = np.ascontiguousarray(raw)
    result = np.empty(raw.shape, np.float32)
    lib, digest = _library()
    status = lib.app2_gsdt(data.ctypes.data, result.ctypes.data, *raw.shape[::-1],
                           raw.dtype.itemsize, connectivity, threshold)
    if status or not np.isfinite(result).all() or np.any(result >= 1e20):
        raise RuntimeError(f"APP2 GSDT failed or returned unreachable voxels (status={status})")
    return result, dict(
        function="APP2 fastmarching_dt (3D)", backend="cpu_native",
        connectivity=connectivity, neighbors={1: 6, 2: 18, 3: 26}[connectivity],
        requested_background_threshold=float(background_threshold),
        effective_background_threshold=threshold, input_dtype=str(raw.dtype),
        units="intensity-weighted path cost; NOT um", source_sha256=digest,
        source_provenance=json.loads((SOURCE / "provenance.json").read_text()))
