import os
import time
import json as json_lib

import numpy as np
import tifffile
from tqdm import tqdm
from joblib import Parallel, delayed
from skimage.transform import resize

from pipeline_config import BASE_DIR, PATHS, NEURON_IDS
from neuroutils.meta.neuron import get_source_v3d_img_file, get_neuron_meta
from neuroutils.image.io import load_image


# ============================================================
# Configuration
# ============================================================

test_dir = str(PATHS.get("image_1um_dir", BASE_DIR))
os.makedirs(test_dir, exist_ok=True)

# 是否写 image_{id}.json。
# 旧代码即使 GENERATE_JSON=False 也会写 JSON；
# 这版会严格遵循该开关。
GENERATE_JSON = False

skipped_log_path = str(
    PATHS.get(
        "pipeline_skip_log",
        BASE_DIR / "pipeline_skipped_neurons.log",
    )
)

# ------------------------------------------------------------
# Resize settings for grayscale microscopy images.
# ------------------------------------------------------------

# 原图是灰度图，不是 mask，因此使用线性插值。
RESIZE_ORDER = 1

# 图像边界外的取值方式，保持 reflect。
RESIZE_MODE = "reflect"

# 下采样时保留抗锯齿，减少 aliasing。
RESIZE_ANTI_ALIASING = True

# ------------------------------------------------------------
# Parallel settings.
#
# Cell 50000 的 PBD.load() 原始 float32 图像约 2.4 GB，
# 不要直接上 10 个 worker。
#
# 默认单进程。确认稳定后，先试：
#   RESCALE_N_JOBS=2 python prepare_nnunet_1um.py
# ------------------------------------------------------------

CPU_N_JOBS = int(os.environ.get("RESCALE_N_JOBS", "1"))

# 设置为 1 时打印每个样本各阶段耗时：
#   RESCALE_PRINT_TIMING=1 python prepare_nnunet_1um.py
PRINT_TIMING = os.environ.get("RESCALE_PRINT_TIMING", "0") == "1"


# ============================================================
# Logging and metadata helpers
# ============================================================

def log_skip(neuron_id, reason):
    """Log a skipped or failed neuron to terminal and persistent log."""
    os.makedirs(os.path.dirname(skipped_log_path) or ".", exist_ok=True)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{timestamp}] [image_{neuron_id}] SKIP: rescale: {reason}"

    print(msg, flush=True)

    with open(skipped_log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def validate_resolution(xy_resolution, z_resolution):
    """Validate and convert metadata resolutions to finite positive floats."""
    try:
        xy_resolution = float(xy_resolution)
        z_resolution = float(z_resolution)

    except (TypeError, ValueError) as exc:
        raise ValueError(
            "invalid resolution in metadata: "
            f"xy_resolution={xy_resolution!r}, "
            f"z_resolution={z_resolution!r}"
        ) from exc

    if not np.isfinite(xy_resolution) or not np.isfinite(z_resolution):
        raise ValueError(
            "non-finite resolution in metadata: "
            f"xy_resolution={xy_resolution!r}, "
            f"z_resolution={z_resolution!r}"
        )

    if xy_resolution <= 0 or z_resolution <= 0:
        raise ValueError(
            "non-positive resolution in metadata: "
            f"xy_resolution={xy_resolution!r}, "
            f"z_resolution={z_resolution!r}"
        )

    return xy_resolution, z_resolution


def get_resolution_for_rescale(neuron_id):
    """
    Read xy/z resolution from metadata.

    Supports both English and Chinese metadata column names currently used
    in your project.
    """
    meta = get_neuron_meta(int(neuron_id))

    if meta is None or len(meta) == 0:
        raise ValueError("empty metadata result")

    row = meta.iloc[0]

    if "xy_resolution" in meta.columns and "z_resolution" in meta.columns:
        return validate_resolution(
            row["xy_resolution"],
            row["z_resolution"],
        )

    if (
        "xy拍摄分辨率(*10e-3μm/px)" in meta.columns
        and "z拍摄分辨率(*10e-3μm/px)" in meta.columns
    ):
        return validate_resolution(
            row["xy拍摄分辨率(*10e-3μm/px)"],
            row["z拍摄分辨率(*10e-3μm/px)"],
        )

    raise ValueError("resolution columns not found in metadata")


# ============================================================
# Image processing helpers
# ============================================================

def get_target_shape(img_shape, xy_resolution, z_resolution):
    """
    Compute nearest integer image dimensions for 1 µm isotropic spacing.

    img_shape is assumed to be (Z, Y, X).
    Metadata unit is assumed to be 10^-3 µm / pixel, matching your
    original /1000 conversion.
    """
    scale = np.array(
        [z_resolution, xy_resolution, xy_resolution],
        dtype=np.float64,
    ) / 1000.0

    target_shape = np.rint(
        np.asarray(img_shape, dtype=np.float64) * scale
    ).astype(np.int64)

    target_shape = np.maximum(target_shape, 1)

    return tuple(int(x) for x in target_shape)


def to_uint8_after_resize(image):
    """
    Convert resized grayscale image to uint8 without another min-max stretch.

    Input image has already been normalized to approximately 0~255 by
    load_image(normalize=True). Resize may introduce float values such as
    43.17 through interpolation, so we:

      1. verify finite values;
      2. clip to [0, 255];
      3. round to nearest integer;
      4. cast to uint8.

    This intentionally does NOT re-normalize according to the resized image's
    new min/max. Therefore, resize-induced smoothing does not cause each image
    to be re-stretched to full contrast again.
    """
    image = np.asarray(image)

    img_min = float(np.min(image))
    img_max = float(np.max(image))

    if not np.isfinite(img_min) or not np.isfinite(img_max):
        raise ValueError(
            f"invalid resized image range: min={img_min}, max={img_max}"
        )

    # resize() should return a writable float array. This fallback keeps the
    # function robust if that assumption changes.
    if (
        not np.issubdtype(image.dtype, np.floating)
        or not image.flags.writeable
    ):
        image = image.astype(np.float32, copy=True)

    np.clip(image, 0.0, 255.0, out=image)

    # rint is preferred for newly generated data:
    # 127.6 -> 128, rather than astype(uint8) truncating it to 127.
    np.rint(image, out=image)

    return image.astype(np.uint8, copy=False)


def write_tiff_atomic(output_path, image):
    """
    Write TIFF to a temporary file first, then atomically rename it.

    This prevents a partially written TIFF from being mistaken as a finished
    output when the process is interrupted.
    """
    tmp_path = f"{output_path}.tmp.{os.getpid()}.tif"

    try:
        tifffile.imwrite(tmp_path, image)
        os.replace(tmp_path, output_path)

    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def write_json_atomic(output_path, data):
    """Atomically write JSON metadata."""
    tmp_path = f"{output_path}.tmp.{os.getpid()}"

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json_lib.dump(data, f, indent=4)

        os.replace(tmp_path, output_path)

    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# ============================================================
# Per-neuron processing
# ============================================================

def prepare_nnunet_file(neuron_id):
    """Load one source image, rescale it to 1 µm, and save nnUNet TIFF."""
    neuron_id = int(neuron_id)

    nnunet_file_name = f"image_{neuron_id}_0000.tif"
    json_file_name = f"image_{neuron_id}.json"

    nnunet_file_path = os.path.join(test_dir, nnunet_file_name)
    json_file_path = os.path.join(test_dir, json_file_name)

    # Fast path: do no metadata lookup or image loading if output is ready.
    if os.path.exists(nnunet_file_path):
        if not GENERATE_JSON or os.path.exists(json_file_path):
            print(
                f"[image_{neuron_id}] already exists, skipping.",
                flush=True,
            )
            return True

    total_start = time.perf_counter()

    # --------------------------------------------------------
    # 1. Read metadata first.
    # Invalid metadata should not trigger expensive image loading.
    # --------------------------------------------------------
    try:
        xy_resolution, z_resolution = get_resolution_for_rescale(neuron_id)

    except Exception as exc:
        log_skip(
            neuron_id,
            f"invalid metadata resolution: {type(exc).__name__}: {exc}",
        )
        return False

    # --------------------------------------------------------
    # 2. Read image.
    # --------------------------------------------------------
    try:
        load_start = time.perf_counter()

        v3d_img_file = get_source_v3d_img_file(neuron_id)

        # Optimized load_image behavior:
        # - ordinary 0~255 integer-valued PBD -> direct uint8;
        # - unusual range, e.g. 2~360 -> min-max normalized uint8.
        img = np.asarray(
            load_image(v3d_img_file, normalize=True)
        )

        load_seconds = time.perf_counter() - load_start

        if img.ndim != 3:
            raise ValueError(
                f"expected 3D image shape (Z, Y, X), got {img.shape}"
            )

        if img.dtype != np.uint8:
            raise ValueError(
                f"expected uint8 image after normalization, got {img.dtype}"
            )

    except Exception as exc:
        log_skip(
            neuron_id,
            f"get v3d img file failed: {type(exc).__name__}: {exc}",
        )
        return False

    # --------------------------------------------------------
    # 3. Resize to target 1 µm shape.
    # --------------------------------------------------------
    try:
        img_size = img.shape

        target_shape = get_target_shape(
            img_shape=img_size,
            xy_resolution=xy_resolution,
            z_resolution=z_resolution,
        )

        resize_start = time.perf_counter()

        rescaled_1um_img = resize(
            img,
            target_shape,
            order=RESIZE_ORDER,
            mode=RESIZE_MODE,
            anti_aliasing=RESIZE_ANTI_ALIASING,
            preserve_range=True,
        )

        resize_seconds = time.perf_counter() - resize_start

        output_convert_start = time.perf_counter()

        # Do not apply another min-max normalization here.
        rescaled_1um_img = to_uint8_after_resize(
            rescaled_1um_img
        )

        output_convert_seconds = (
            time.perf_counter() - output_convert_start
        )

    except Exception as exc:
        log_skip(
            neuron_id,
            f"process failed: {type(exc).__name__}: {exc}",
        )
        return False

    # --------------------------------------------------------
    # 4. Write outputs.
    # --------------------------------------------------------
    try:
        write_start = time.perf_counter()

        write_tiff_atomic(
            nnunet_file_path,
            rescaled_1um_img,
        )

        if GENERATE_JSON:
            json_data = {
                "original_shape": list(img_size),
                "spacing": [1.0, 1.0, 1.0],
            }

            write_json_atomic(
                json_file_path,
                json_data,
            )

        write_seconds = time.perf_counter() - write_start

    except Exception as exc:
        log_skip(
            neuron_id,
            f"write output failed: {type(exc).__name__}: {exc}",
        )
        return False

    if PRINT_TIMING:
        total_seconds = time.perf_counter() - total_start

        print(
            f"[image_{neuron_id}] done | "
            f"input={img_size} | "
            f"target={target_shape} | "
            f"load={load_seconds:.2f}s | "
            f"resize={resize_seconds:.2f}s | "
            f"to_uint8={output_convert_seconds:.2f}s | "
            f"write={write_seconds:.2f}s | "
            f"total={total_seconds:.2f}s",
            flush=True,
        )

    return True


def try_prepare_nnunet_file(neuron_id):
    """Prevent one failed sample from stopping the entire batch."""
    try:
        return prepare_nnunet_file(neuron_id)

    except Exception as exc:
        log_skip(
            neuron_id,
            f"unexpected wrapper error: {type(exc).__name__}: {exc}",
        )
        return False


# ============================================================
# Entrypoint
# ============================================================

if __name__ == "__main__":
    todo_neuron_ids = list(NEURON_IDS)

    if CPU_N_JOBS < 1:
        raise ValueError(
            f"RESCALE_N_JOBS must be >= 1, got {CPU_N_JOBS}"
        )

    print(
        f"Starting {len(todo_neuron_ids)} neurons with "
        f"RESCALE_N_JOBS={CPU_N_JOBS}",
        flush=True,
    )

    if CPU_N_JOBS == 1:
        results = []

        for neuron_id in tqdm(
            todo_neuron_ids,
            desc="Preparing 1um images",
        ):
            results.append(
                try_prepare_nnunet_file(neuron_id)
            )

    else:
        # 先从 RESCALE_N_JOBS=2 开始测试。
        # 不建议直接开 10 个，因为 PBD 解码阶段内存很高。
        results = Parallel(
            n_jobs=CPU_N_JOBS,
            backend="loky",
            pre_dispatch=CPU_N_JOBS,
        )(
            delayed(try_prepare_nnunet_file)(neuron_id)
            for neuron_id in todo_neuron_ids
        )

    succeeded = sum(bool(x) for x in results)
    failed_or_skipped = len(results) - succeeded

    print(
        f"Finished. successful={succeeded}, "
        f"failed_or_skipped={failed_or_skipped}",
        flush=True,
    )