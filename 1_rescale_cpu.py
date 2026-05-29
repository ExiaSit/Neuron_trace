import os
import time
import numpy as np
import tifffile
import json as json_lib
from tqdm import tqdm
from joblib import Parallel, delayed
from skimage.transform import resize

from scipy.ndimage import zoom 
from pipeline_config import BASE_DIR, PATHS, NEURON_IDS

# 保持你原有的库引用
from neuroutils.meta.neuron import get_source_v3d_img_file, get_neuron_meta
from neuroutils.image.io import load_image

# 路径配置
# Direct script execution should use the same output directory as run_pipeline.py.
# Previously this defaulted to BASE_DIR, so existing files under PATHS["image_1um_dir"]
# were not detected and the script reprocessed neurons unnecessarily.
test_dir = str(PATHS.get("image_1um_dir", BASE_DIR))
os.makedirs(test_dir, exist_ok=True)
GENERATE_JSON = False
skipped_log_path = str(PATHS.get("pipeline_skip_log", BASE_DIR / "pipeline_skipped_neurons.log"))

def log_skip(neuron_id, reason):
    os.makedirs(os.path.dirname(skipped_log_path) or ".", exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{timestamp}] [image_{neuron_id}] SKIP: rescale: {reason}"
    print(msg)
    with open(skipped_log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def validate_resolution(xy_resolution, z_resolution):
    try:
        xy_resolution = float(xy_resolution)
        z_resolution = float(z_resolution)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid resolution in metadata: xy_resolution={xy_resolution!r}, "
            f"z_resolution={z_resolution!r}"
        ) from exc
    if not np.isfinite(xy_resolution) or not np.isfinite(z_resolution):
        raise ValueError(
            f"non-finite resolution in metadata: xy_resolution={xy_resolution!r}, "
            f"z_resolution={z_resolution!r}"
        )
    if xy_resolution <= 0 or z_resolution <= 0:
        raise ValueError(
            f"non-positive resolution in metadata: xy_resolution={xy_resolution!r}, "
            f"z_resolution={z_resolution!r}"
        )
    return xy_resolution, z_resolution

def get_resolution_for_rescale(neuron_id):
    meta = get_neuron_meta(int(neuron_id))
    if "xy_resolution" in meta.columns and "z_resolution" in meta.columns:
        return validate_resolution(meta["xy_resolution"].values[0], meta["z_resolution"].values[0])
    if "xy拍摄分辨率(*10e-3μm/px)" in meta.columns and "z拍摄分辨率(*10e-3μm/px)" in meta.columns:
        return validate_resolution(
            meta["xy拍摄分辨率(*10e-3μm/px)"].values[0],
            meta["z拍摄分辨率(*10e-3μm/px)"].values[0],
        )
    raise ValueError("resolution columns not found in metadata")

def prepare_nnunet_file(neuron_id):
    nnunet_file_name = f"image_{neuron_id}_0000.tif"
    json_file_name = f"image_{neuron_id}.json"
    nnunet_file_path = os.path.join(test_dir, nnunet_file_name)
    json_file_path =  os.path.join(test_dir, json_file_name)

    # Fast path: do this before any metadata lookup or image import work.
    if os.path.exists(nnunet_file_path):
        if not GENERATE_JSON or os.path.exists(json_file_path):
            print(nnunet_file_path + " already exists, skipping.")
            return

    print(f"Processing neuron {neuron_id}...")

    # --- 1. 先检查 metadata。无效分辨率不需要读取大图。 ---
    try:
        xy_resolution, z_resolution = get_resolution_for_rescale(neuron_id)
    except Exception as e:
        log_skip(neuron_id, f"invalid metadata resolution: {type(e).__name__}: {e}")
        return

    # --- 2. 读取图像 ---
    try:
        v3d_img_file = get_source_v3d_img_file(neuron_id)
        img = load_image(v3d_img_file)
    except Exception as e:
        log_skip(neuron_id, f"get v3d img file failed: {type(e).__name__}: {e}")
        return

    # --- 3. 计算目标尺寸 ---
    try:
        img_size = img.shape # (Z, Y, X)
        
        # 计算 1um 分辨率下的目标形状
        rescaled_1um_shape = [
            int(img_size[0] * z_resolution / 1000), 
            int(img_size[1] * xy_resolution / 1000),
            int(img_size[2] * xy_resolution / 1000)
        ]
        
        rescaled_1um_shape = [int(i) for i in rescaled_1um_shape]
        rescaled_1um_img = resize(img, rescaled_1um_shape, order=0, mode='reflect', anti_aliasing=True)
        
    except Exception as e:
        log_skip(neuron_id, f"process failed: {type(e).__name__}: {e}")
        return

    rescaled_1um_img = (rescaled_1um_img - np.min(rescaled_1um_img)) / (np.max(rescaled_1um_img) - np.min(rescaled_1um_img)) * 255
    tifffile.imwrite(nnunet_file_path, rescaled_1um_img.astype("uint8"))


    json_data = {
        "original_shape": list(img_size),
        "spacing": [1.0, 1.0, 1.0]
    }

    with open(json_file_path, 'w') as f:
        json_lib.dump(json_data, f, indent=4)

def try_repare_nnunet_file(neuron_id):
    try:
        prepare_nnunet_file(neuron_id)
    except Exception as e:
        print(f"Error wrapper {neuron_id}: {e}")

if __name__ == "__main__":
    todo_neuron_ids = NEURON_IDS
    cpu_n_jobs = 1
    
    print(f"Starting processing with {cpu_n_jobs} parallel jobs on CPU...")
    for neuron_id in tqdm(todo_neuron_ids):
        try_repare_nnunet_file(neuron_id) 
    # Parallel(n_jobs=cpu_n_jobs)(
    #     delayed(try_repare_nnunet_file)(neuron_id) for neuron_id in tqdm(todo_neuron_ids)
    # )