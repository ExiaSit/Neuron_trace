import os
import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as patches 
import glob
from neuroutils.meta.neuron import get_neuron_meta, get_soma_pos, get_xy_z_resolution
from neuroutils.image.io import load_image 
from neuroutils.image.preprocessor import calculate_bounds_3d
import json as json_lib
from tqdm import tqdm
from joblib import Parallel, delayed
import subprocess
import shutil
import math
from multiprocessing import Process
from pipeline_config import BASE_DIR,NNUNET,PATHS


# ================= Configure =================
DATA_TYEP = "CELL_BLOCK" 
TARGET_BLOCK_SIZE = (128, 128, 128)
SAVE_MIP_VISUALIZATION = False

base_path = BASE_DIR
img_1um_dir = PATHS["image_1um_dir"]
soma_crop_dir = PATHS["soma_crop_dir"]
soma_seg_dir = PATHS["soma_seg_dir"]
os.makedirs(soma_crop_dir, exist_ok=True)
os.makedirs(soma_seg_dir, exist_ok=True)

base_temp_dir = PATHS["soma_temp_dir"]
skipped_neurons_log_path = PATHS["soma_skipped_log"]

os.environ['nnUNet_raw'] = "Wait_No_Need"
os.environ['nnUNet_preprocessed'] = "Wait_No_Need"
os.environ['nnUNet_results'] = NNUNET["soma_results"]

AVAILABLE_GPUS = [0] 
NUM_INFER_WORKERS = 10 
INFER_BATCH_SIZE = 200 
# =========================================

def save_crop_context_plot(full_img, bounds, neuron_id, save_dir):
    """
    输入:
        full_img: 原始大的 1um 图像 (Z, Y, X)
        bounds: (z_start, z_end, y_start, y_end, x_start, x_end)
    输出:
        保存带红框的三视图
    """
    z_start, z_end, y_start, y_end, x_start, x_end = bounds
    
    # 图像形状
    # img[z, y, x]
    # Axis 0: Z, Axis 1: Y, Axis 2: X
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # --- 1. XY View (Top View) ---
    # 投影方向: Z轴 (axis 0) -> 得到 (Y, X) 图像
    # imshow 显示: Y轴为垂直方向(Rows), X轴为水平方向(Cols)
    # 框的坐标: (x, y), 宽: dx, 高: dy
    mip_xy = np.max(full_img, axis=0)
    axes[0].imshow(mip_xy, cmap='gray', aspect='auto', vmin=0, vmax=255) # 假设是uint8, 或者去掉vmin/vmax自动缩放
    rect_xy = patches.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start, 
                                linewidth=2, edgecolor='red', facecolor='none')
    axes[0].add_patch(rect_xy)
    axes[0].set_title(f"XY View (Full)\nCrop X[{x_start}:{x_end}] Y[{y_start}:{y_end}]")
    axes[0].axis('off')

    # --- 2. XZ View (Side View) ---
    # 投影方向: Y轴 (axis 1) -> 得到 (Z, X) 图像
    # imshow 显示: Z轴为垂直方向(Rows), X轴为水平方向(Cols)
    # 框的坐标: (x, z), 宽: dx, 高: dz
    mip_xz = np.max(full_img, axis=1)
    axes[1].imshow(mip_xz, cmap='gray', aspect='auto', vmin=0, vmax=255)
    rect_xz = patches.Rectangle((x_start, z_start), x_end - x_start, z_end - z_start, 
                                linewidth=2, edgecolor='red', facecolor='none')
    axes[1].add_patch(rect_xz)
    axes[1].set_title(f"XZ View (Full)\nCrop X[{x_start}:{x_end}] Z[{z_start}:{z_end}]")
    axes[1].axis('off')

    # --- 3. YZ View (Front View) ---
    # 投影方向: X轴 (axis 2) -> 得到 (Z, Y) 图像
    # imshow 显示: Z轴为垂直方向(Rows), Y轴为水平方向(Cols)
    # 框的坐标: (y, z), 宽: dy, 高: dz
    mip_yz = np.max(full_img, axis=2)
    axes[2].imshow(mip_yz, cmap='gray', aspect='auto', vmin=0, vmax=255)
    rect_yz = patches.Rectangle((y_start, z_start), y_end - y_start, z_end - z_start, 
                                linewidth=2, edgecolor='red', facecolor='none')
    axes[2].add_patch(rect_yz)
    axes[2].set_title(f"YZ View (Full)\nCrop Y[{y_start}:{y_end}] Z[{z_start}:{z_end}]")
    axes[2].axis('off')

    plt.suptitle(f"Neuron ID: {neuron_id} - Crop Context Visualization", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"vis_context_{neuron_id}.png"), dpi=100)
    plt.close(fig)

def prepare_nnunet_file(neuron_id, target_size, save_vis=SAVE_MIP_VISUALIZATION):
    nnunet_file_name = f"image_{neuron_id}_0000.tif"
    json_file_name = f"image_{neuron_id}.json"
    nnunet_file_path = os.path.join(soma_crop_dir, nnunet_file_name)
    json_file_path = os.path.join(soma_crop_dir, json_file_name)
    
    if os.path.exists(nnunet_file_path) and os.path.exists(json_file_path):
        return

    image_path = os.path.join(img_1um_dir, f"image_{neuron_id}_0000.tif") 
    if not os.path.exists(image_path):
        print(f"Skipping {neuron_id}: File not found at {image_path}")
        return

    # 读取完整的 1um 原图
    img = tifffile.imread(image_path)
    
    soma_x_raw, soma_y_raw, soma_z_raw = get_soma_pos(int(neuron_id))
    try:
        xy_resolution, z_resolution = get_xy_z_resolution(int(neuron_id))
    except Exception as e:
        log_file = skipped_neurons_log_path
        try:
            msg = f"Neuron {neuron_id} SKIPPED: Failed to get resolution: {e}"
            with open(log_file, "a") as f:
                f.write(msg + "\n")
        except Exception as e:
            print(f"Failed to write log: {e}")
        return 
    center_z = int(round(soma_z_raw * (z_resolution / 1000)))
    center_y = int(round(soma_y_raw * (xy_resolution / 1000)))
    center_x = int(round(soma_x_raw * (xy_resolution / 1000)))
    z_start, z_end, y_start, y_end, x_start, x_end = calculate_bounds_3d(
        center_pos=(center_z, center_y, center_x),
        block_size=target_size, 
        img_shape=img.shape
    )

    soma_block = img[z_start:z_end, y_start:y_end, x_start:x_end]
    if 0 in soma_block.shape:
        msg = (f"Neuron {neuron_id} SKIPPED: Empty block detected. "
               f"Shape: {soma_block.shape}. "
               f"Global Soma: ({soma_z_raw}, {soma_y_raw}, {soma_x_raw}), "
               f"Img Shape: {img.shape}, "
               f"Bounds: Z[{z_start}:{z_end}] Y[{y_start}:{y_end}] X[{x_start}:{x_end}]")
        
        # 记录到 txt 文件中
        log_file = skipped_neurons_log_path
        try:
            with open(log_file, "a") as f:
                f.write(msg + "\n")
        except Exception as e:
            print(f"Failed to write log: {e}")

        return 
    # 传入: 原图 img, 计算好的边界, ID, 保存路径
    if save_vis:
        save_crop_context_plot(
            img, 
            (z_start, z_end, y_start, y_end, x_start, x_end), 
            neuron_id, 
            soma_crop_dir
        )
    # Padding
    if soma_block.shape != target_size:
        pad_z = target_size[0] - soma_block.shape[0]
        pad_y = target_size[1] - soma_block.shape[1]
        pad_x = target_size[2] - soma_block.shape[2]
        soma_block = np.pad(soma_block, ((0, pad_z), (0, pad_y), (0, pad_x)), 'constant')
    
    if(np.max(soma_block) - np.min(soma_block)) == 0:
        print(soma_block.shape)
        print(f"Warning: Image {neuron_id} is constant/empty.")
        soma_block[:] = 0
    else:
        # Normalize
        soma_block = (soma_block - np.min(soma_block)) / (np.max(soma_block) - np.min(soma_block)) * 255
    
    tifffile.imwrite(nnunet_file_path, soma_block.astype("uint8"))

    with open(json_file_path, 'w') as f:
            json_lib.dump({
                "original_1um_shape": list(soma_block.shape),
                "bounds_1um_zxy": [int(z_start), int(z_end), int(y_start), int(y_end), int(x_start), int(x_end)],
                "spacing": [1.0, 1.0, 1.0],
                "source_file": image_path
            }, f, indent=4)

def preprocess_wrapper(neuron_id):
    try:
        prepare_nnunet_file(neuron_id, TARGET_BLOCK_SIZE)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"neuron {neuron_id} prepare nnunet file failed: {e}")
############ Seg ###########
def get_processed_ids(output_dir):
    files = glob.glob(os.path.join(output_dir, "*.tif")) # 假设输出也是 tif 或 nii.gz
    # 如果输出是 nii.gz, 改成 *.nii.gz
    done_ids = set()
    for f in files:
        fname = os.path.basename(f)
        # nnUNet 输出通常不带 _0000
        fid = fname.replace('.tif', '')
        done_ids.add(fid)
    return done_ids

def run_nnunet_cmd(input_folder, output_folder, worker_id, gpu_id):
    """调用 nnUNet 命令行"""
    cmd = [
        "nnUNetv2_predict",
        "-i", input_folder,
        "-o", output_folder,
        "-d", "206", # 使用全局配置的 Dataset ID (206)
        "-c", "3d_fullres",
        "-f", "0",
        "-device", "cuda"
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        # subprocess.DEVNULL 隐藏海量输出，只看报错
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env)
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode('utf-8', errors='ignore')
        print(f"❌ [Worker-{worker_id}|GPU-{gpu_id}] Error: {err_msg}")

def inference_worker(worker_id, tasks, output_dir):
    gpu_index = worker_id % len(AVAILABLE_GPUS)
    target_gpu = AVAILABLE_GPUS[gpu_index]
    
    worker_temp_dir = f"{base_temp_dir}_{worker_id}"
    
    try:
        if os.path.exists(worker_temp_dir): shutil.rmtree(worker_temp_dir)
        os.makedirs(worker_temp_dir, exist_ok=True)

        total = len(tasks)

        for i in range(0, total, INFER_BATCH_SIZE):
            current_batch = tasks[i : i + INFER_BATCH_SIZE]
            
            for f in os.listdir(worker_temp_dir):
                os.remove(os.path.join(worker_temp_dir, f))
            
            for task in current_batch:
                src_path = task['path']
                fname = os.path.basename(src_path)
                dst_path = os.path.join(worker_temp_dir, fname) 
                os.symlink(src_path, dst_path)
            
            print(f"🚀 [Worker-{worker_id}] GPU:{target_gpu} Processing batch {i}/{total}...")
            run_nnunet_cmd(worker_temp_dir, output_dir, worker_id, target_gpu)

    finally:
        if os.path.exists(worker_temp_dir): shutil.rmtree(worker_temp_dir)

############### Execution ###############
if __name__ == "__main__":
    print(f"====== PHASE 0: Task Initialization ======")
    
    print(f"Scanning input directory: {img_1um_dir} ...")
    
    input_files = glob.glob(os.path.join(img_1um_dir, "*.tif"))

    todo_neuron_ids = []
    for fpath in input_files:
        fname = os.path.basename(fpath)
        if fname.startswith('.'):
            continue
        temp_name = os.path.splitext(fname)[0] # -> "image_80000_0000"
        
        nid = temp_name.replace("image_", "").replace("_0000", "")
        todo_neuron_ids.append(nid)
    try:
        todo_neuron_ids.sort(key=lambda x: int(x))
    except ValueError:
        todo_neuron_ids.sort()

    total_input_count = len(todo_neuron_ids)
    if total_input_count == 0:
        print(f"❌ Error: No .tif/.tiff files found in {img_1um_dir}")
        print("Please check your 'img_1um_dir' path.")
        sys.exit(1)
        
    print(f"✅ Found {total_input_count} neurons to process (from {todo_neuron_ids[0]} to {todo_neuron_ids[-1]}).")
    print(f"\n====== PHASE 1: Data Preprocessing ======")
    Parallel(n_jobs=12)(delayed(preprocess_wrapper)(nid) for nid in tqdm(todo_neuron_ids, desc="Preprocessing"))
    
    print(f"\n====== PHASE 2: nnUNet Inference ======")
    
    # 1. 扫描刚刚生成的预处理文件 (位于 soma_crop_dir)
    all_ready_files = glob.glob(os.path.join(soma_crop_dir, "*_0000.tif"))
    
    # 2. 过滤掉已经预测过的 (位于 soma_seg_dir)
    done_ids = get_processed_ids(soma_seg_dir)
    valid_tasks = []
    
    for fpath in all_ready_files:
        fname = os.path.basename(fpath)
        fid_base = fname.replace('image_', '').replace('_0000.tif', '') 
        
        # 检查是否已完成
        # 注意：结果文件通常叫 {ID}.tif 或 {ID}.nii.gz，没有 image_ 前缀
        if fid_base not in done_ids and f"image_{fid_base}" not in done_ids:
            valid_tasks.append({
                "path": fpath,
                "fid": fid_base  # 这里的 fid 用于日志显示
            })
            
    num_tasks = len(valid_tasks)
    print(f"Input Ready: {len(all_ready_files)} files.")
    print(f"Already Done: {len(done_ids)} files.")
    print(f"Remaining Tasks: {num_tasks} files.")
    print(f"Inference Workers: {NUM_INFER_WORKERS} | GPUs: {AVAILABLE_GPUS}")

    if num_tasks > 0:
        # 3. 分发任务给 GPU Worker
        chunk_size = math.ceil(num_tasks / NUM_INFER_WORKERS)
        process_list = []

        for i in range(NUM_INFER_WORKERS):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size
            worker_batch = valid_tasks[start_idx : end_idx]

            if not worker_batch: continue

            # 启动多进程
            p = Process(target=inference_worker, args=(i, worker_batch, soma_seg_dir))
            p.start()
            process_list.append(p)

        # 等待所有进程结束
        for p in process_list:
            p.join()
    else:
        print("✨ No new tasks to process. All done!")
            
    print("\n✅ All jobs (Preprocessing + Inference) completed!")