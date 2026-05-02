import os
import numpy as np
import tifffile
import json
import traceback # 用于获取详细报错信息
from tqdm import tqdm # 进度条
from joblib import Parallel, delayed # 多进程并行库
from neuroutils.meta.neuron import get_xy_z_resolution, get_source_v3d_img_file, get_soma_pos
from neuroutils.image.io import load_image


# ================= 2. 合并核心函数 (底层逻辑) =================
def merge_logic(neuron_id, seg_path, full_1um_dir, output_dir, json_dir, soma_block_size_um=(128, 128, 128)):
    # --- A. 准备工作 ---
    full_img_path = os.path.join(full_1um_dir, f"image_{neuron_id}.tif") # 注意：这里文件名格式需确认是否带_0000
    if not os.path.exists(full_img_path):
        # 尝试兼容带 _0000 的情况
        full_img_path = os.path.join(full_1um_dir, f"image_{neuron_id}_0000.tif")
        if not os.path.exists(full_img_path):
            raise FileNotFoundError(f"Full 1um image missing: {full_img_path}")
    
    if not os.path.exists(seg_path):
        raise FileNotFoundError(f"Seg result missing: {seg_path}")

    # 读取 1um 全图
    # 建议使用 memmap=True 如果内存吃紧，但在多进程下可能会有文件锁问题，读取到内存更稳
    full_img_1um = tifffile.imread(full_img_path)
    
    # 确定最大值 (uint8 -> 255, uint16 -> 65535)
    max_val = 255 # 默认值
    if np.issubdtype(full_img_1um.dtype, np.integer):
        max_val = np.iinfo(full_img_1um.dtype).max
    
    # 【关键修改】：处理大图 (Neurite)
    # 逻辑：只要像素值大于0，就强制赋值为 max_val。
    # 使用 full_img_1um[condition] = val 是原地修改 (In-place)，
    # 比 full_img_1um = (full_img_1um > 0) * 255 更省内存。
    full_img_1um[full_img_1um > 0] = max_val
    
    # 将其作为画布
    merged_mask = full_img_1um 
    full_shape = merged_mask.shape

    # -----------------------------------------------------------
    # 2. 读取 Soma 分割块 (seg_block) 并处理亮度
    # -----------------------------------------------------------
    seg_block = tifffile.imread(seg_path)
    xy_res, z_res = get_xy_z_resolution(int(neuron_id))
    
    # 【关键修改】：处理小块 (Soma)
    # 逻辑：先生成 0/1 掩膜，转换类型与大图一致，然后乘以最大值
    seg_block = (seg_block > 0).astype(merged_mask.dtype) * max_val
    
    # 读取 JSON 获取原始裁切坐标
    json_path = os.path.join(json_dir, f"image_{neuron_id}.json")
    if not os.path.exists(json_path):
         raise FileNotFoundError(f"JSON meta missing: {json_path}")
         
    with open(json_path, 'r') as f:
        meta = json.load(f)
        z_start_raw = meta["bounds_zxy"][0] 
        y_start_raw = meta["bounds_zxy"][2]
        x_start_raw = meta["bounds_zxy"][4]

    # --- D. 坐标映射 (Raw -> 1um) ---
    z_start_1um = int(round(z_start_raw * (z_res / 1000)))
    y_start_1um = int(round(y_start_raw * (xy_res / 1000)))
    x_start_1um = int(round(x_start_raw * (xy_res / 1000)))

    # --- E. 粘贴 (Paste) ---
    z_len, y_len, x_len = seg_block.shape
    
    z_end_1um = min(z_start_1um + z_len, full_shape[0])
    y_end_1um = min(y_start_1um + y_len, full_shape[1])
    x_end_1um = min(x_start_1um + x_len, full_shape[2])
    
    actual_z_len = z_end_1um - z_start_1um
    actual_y_len = y_end_1um - y_start_1um
    actual_x_len = x_end_1um - x_start_1um

    if actual_z_len <= 0 or actual_y_len <= 0 or actual_x_len <= 0:
        raise ValueError(f"Calculated dimensions are zero or negative: {actual_z_len, actual_y_len, actual_x_len}")

    # 【核心修改】：使用 |= (OR) 而不是 = (赋值)
    # 赋值(=)会将 mask 里的 0 (背景) 覆盖到原图上，导致原图出现黑框。
    # OR(|=) 会保留原图背景，只叠加高亮部分。
    merged_mask[z_start_1um:z_end_1um, y_start_1um:y_end_1um, x_start_1um:x_end_1um] |= \
        seg_block[:actual_z_len, :actual_y_len, :actual_x_len]

    # --- F. 保存 ---
    save_path = os.path.join(output_dir, f"image_{neuron_id}.tif")
    tifffile.imwrite(save_path, merged_mask)
    
    return f"Success: {neuron_id}"

# ================= 3. 异常处理封装 wrapper =================
def process_one_neuron(neuron_id, seg_file_path, full_dir, out_dir, json_dir, log_file):
    """
    单个神经元的处理流程，包含 Try-Except
    """
    try:
        merge_logic(neuron_id, seg_file_path, full_dir, out_dir, json_dir)
        return None # 成功返回 None
    except Exception as e:
        # 捕捉所有异常
        error_msg = f"{neuron_id}: {str(e)}"
        print(f"❌ Error processing {neuron_id}")
        
        # 写入错误日志 (追加模式)
        # 注意：多进程同时写一个文件偶尔会乱，但对于 simple text log 通常问题不大
        # 更严格的做法是返回错误信息，在主进程统一写，但这里直接写最简单
        with open(log_file, "a") as f:
            f.write(error_msg + "\n")
            # 如果想看详细堆栈，可以取消下面注释
            # f.write(traceback.format_exc() + "\n")
            
        return error_msg

# ================= 4. 运行主程序 =================
if __name__ == "__main__":
    print("注意，这个concat适用于B4.5数据，C6.0的请使用另一个方法")
    # --- 配置路径 ---
    Neurite_Seg_DIR = "/data/disk/C6.0/app_test/1um_neurite_seg_0424"
    Soma_SEG_DIR = "/data/disk/C6.0/app_test/soma_seg"
    OUT_DIR = "/data/disk/C6.0/app_test/mask_new_version_0424"
    JSON_DIR = "/data/disk/C6.0/app_test/soma_img" # 原 script A 输出 json 的目录
    ERROR_LOG = "/data/disk/C6.0/app_test/merge_failed_log.txt"
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 清空或创建日志文件
    with open(ERROR_LOG, "w") as f:
        f.write("=== Merge Error Log ===\n")

    # --- 获取任务列表 ---
    # 自动扫描文件夹获取 ID，而不是手动写 list
    # 假设 seg 文件名格式为 image_{ID}.tif
    import glob
    seg_files = glob.glob(os.path.join(Soma_SEG_DIR, "*.tif"))
    todo_tasks = []
    print(len(seg_files))
    for fpath in seg_files:
        fname = os.path.basename(fpath)
        # 解析 ID: image_20000.tif -> 20000
        # 你的代码里有些可能是 image_20000.tif，有些可能是 20000.tif，做个兼容
        if(os.path.exists(os.path.join(OUT_DIR,fname))):
            
            # print(os.path.join(OUT_DIR,fname))
            continue
        try:
            nid_str = fname.replace("image_", "").replace(".tif", "")
            nid = int(nid_str)
            # 过滤：如果你只想跑 20000 到 60000
            # if 20000 <= nid < 20001:
            todo_tasks.append((nid, fpath))
        except ValueError:
            continue
    
    # 按 ID 排序
    todo_tasks.sort(key=lambda x: x[0])
    
    print(f"Found {len(todo_tasks)} neurons to process.")

    # --- 多进程并行执行 ---
    # n_jobs=-1 使用所有 CPU 核心，n_jobs=8 使用 8 个核
    # 建议设置为核心数的一半或 8-12，防止磁盘 I/O 拥堵
    NUM_CORES = 12
    
    print(f"Starting parallel processing with {NUM_CORES} cores...")
    
    Parallel(n_jobs=NUM_CORES)(
        delayed(process_one_neuron)(
            nid, 
            fpath, 
            Neurite_Seg_DIR, 
            OUT_DIR, 
            JSON_DIR, # 传入 JSON 目录
            ERROR_LOG
        ) 
        for nid, fpath in tqdm(todo_tasks)
    )
    
    print(f"\nDone! Check error log at: {ERROR_LOG}")
