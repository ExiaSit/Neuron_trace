import os
import numpy as np
import tifffile
import json as json_lib
from tqdm import tqdm
from joblib import Parallel, delayed
from skimage.transform import resize

from scipy.ndimage import zoom 
from pipeline_config import BASE_DIR, PATHS

# 保持你原有的库引用
from neuroutils.meta.neuron import get_source_v3d_img_file, get_xy_z_resolution
from neuroutils.image.io import load_image

# 路径配置
# Direct script execution should use the same output directory as run_pipeline.py.
# Previously this defaulted to BASE_DIR, so existing files under PATHS["image_1um_dir"]
# were not detected and the script reprocessed neurons unnecessarily.
test_dir = str(PATHS.get("image_1um_dir", BASE_DIR))
os.makedirs(test_dir, exist_ok=True)
GENERATE_JSON = False

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

    # --- 1. 读取 ---
    try:
        v3d_img_file = get_source_v3d_img_file(neuron_id)
        img = load_image(v3d_img_file)
    except Exception as e:
        print(f"neuron {neuron_id} get v3d img file failed: {e}")
        return

    # --- 2. 计算目标尺寸 ---
    try:
        xy_resolution, z_resolution = get_xy_z_resolution(neuron_id)
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
        print(f"neuron {neuron_id} process failed: {e}")
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
    todo_neuron_ids = [i for i in range(130500,136278)]  
    cpu_n_jobs = 1
    
    print(f"Starting processing with {cpu_n_jobs} parallel jobs on CPU...")
    for neuron_id in tqdm(todo_neuron_ids):
        try_repare_nnunet_file(neuron_id) 
    # Parallel(n_jobs=cpu_n_jobs)(
    #     delayed(try_repare_nnunet_file)(neuron_id) for neuron_id in tqdm(todo_neuron_ids)
    # )