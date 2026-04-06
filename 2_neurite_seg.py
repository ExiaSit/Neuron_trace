import os
import subprocess
import glob
import shutil
import math
from multiprocessing import Process
from tqdm import tqdm

# ================= 配置区 =================
basic_path = "/data/disk/C6.0/app_test/test"
# basic_path = "/data/disk3/zll/origin"
raw_input_dir = os.path.join(basic_path, 'img')
final_output_dir = os.path.join(basic_path, '1um_neurite_seg')
base_temp_dir = os.path.join(basic_path, 'temp_batch_input') 

# [修改点 1]：定义可用的物理 GPU 列表
# 这里定义了你要使用的物理显卡ID，如果有4张卡可以是 [0, 1, 2, 3]
AVAILABLE_GPUS = [0] 

# 模型路径
os.environ['nnUNet_raw'] = "Wait_No_Need"
os.environ['nnUNet_preprocessed'] = "Wait_No_Need"
os.environ['nnUNet_results'] = "/data/disk/nnUNet_local/nnUNet_results"

# 并发数
# 注意：10个并发意味着每张显卡会同时跑 5 个模型 (10/2=5)。
# 如果显存不够（OOM），请适当降低这个数字（例如改为 2 或 4）
NUM_WORKERS = 1
BATCH_SIZE = 10 
# =========================================

def get_processed_ids(output_dir):
    files = glob.glob(os.path.join(output_dir, "*.tif"))
    print(f"在 {output_dir} 中找到 {len(files)} 个已处理文件。")
    done_ids = set()
    for f in files:
        fname = os.path.basename(f)
        # 结果通常没有 _0000，直接去掉后缀
        fid = fname.replace('.tif', '') 
        done_ids.add(fid)
    return done_ids

def run_nnunet(input_folder, output_folder, worker_id, gpu_id):
    """
    运行 nnUNet
    [修改点 2]：接收 gpu_id 参数，并为子进程设置独立的环境变量
    """
    cmd = [
        "nnUNetv2_predict",
        "-i", input_folder,
        "-o", output_folder,
        "-d", "169",
        "-c", "3d_fullres",
        "-f", "0",
        # "--save_probabilities",
        # 显式指定 device 为 cuda (配合环境变量使用)
        "-device", "cuda" 
    ]
    
    # 复制当前环境变量，并强制指定该进程只能看到分配给它的那张卡
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        # 使用 env 参数传入修改后的环境变量
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode('utf-8', errors='ignore')
        print(f"❌ [Worker-{worker_id} | GPU-{gpu_id}] 预测出错: {err_msg}")

def worker_task(worker_id, tasks, output_dir):
    """
    子进程的工作函数
    """
    # [修改点 3]：根据 worker_id 计算该进程应该使用哪张显卡
    # 逻辑：worker_0 用 GPU 0, worker_1 用 GPU 1, worker_2 用 GPU 0 ...
    gpu_index = worker_id % len(AVAILABLE_GPUS)
    target_gpu = AVAILABLE_GPUS[gpu_index]

    worker_temp_dir = f"{base_temp_dir}_{worker_id}"
    print(f"🔧 [Worker-{worker_id}] 启动 (绑定 GPU: {target_gpu})，处理 {len(tasks)} 张图片...")
    
    try:
        if os.path.exists(worker_temp_dir): shutil.rmtree(worker_temp_dir)
        os.makedirs(worker_temp_dir, exist_ok=True)

        total = len(tasks)
        for i in range(0, total, BATCH_SIZE):
            current_batch = tasks[i : i + BATCH_SIZE]
            
            # 清空临时目录
            for f in os.listdir(worker_temp_dir):
                os.remove(os.path.join(worker_temp_dir, f))
            
            # 建立软链接
            for task in current_batch:
                src_tif = task["tif"]
                fid = task["fid_clean"] 
                dst_tif_name = f"{fid}_0000.tif"
                os.symlink(src_tif, os.path.join(worker_temp_dir, dst_tif_name))
                
            # 运行预测，传入 target_gpu
            run_nnunet(worker_temp_dir, output_dir, worker_id, target_gpu)

    finally:
        if os.path.exists(worker_temp_dir): shutil.rmtree(worker_temp_dir)
        print(f"✅ [Worker-{worker_id}] 任务完成！")

# === 主程序 ===
if __name__ == "__main__":
    # [修改点 4]：移除全局的 CUDA_VISIBLE_DEVICES 设置
    # 我们将在每个子进程内部单独设置，避免全局冲突
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]

    os.makedirs(final_output_dir, exist_ok=True)

    print(f"启动 (并发数: {NUM_WORKERS}, 可用GPU: {AVAILABLE_GPUS})")

    # 1. 扫描与筛选
    all_tif_files = glob.glob(os.path.join(raw_input_dir, "*.tif"))
    all_tif_files = [f for f in all_tif_files if not os.path.basename(f).startswith('.')]
    
    done_ids = get_processed_ids(final_output_dir)
    valid_tasks = [] 
    
    print("正在扫描并分配任务...")
    for tif_path in tqdm(all_tif_files, desc="数据质检"):
        fname = os.path.basename(tif_path)
        fid_raw = os.path.splitext(fname)[0] 
        
        if fid_raw.endswith("_0000"):
            fid_clean = fid_raw[:-5] 
        else:
            fid_clean = fid_raw
        
        if fid_clean not in done_ids:
            valid_tasks.append({
                "tif": tif_path,
                "fid_clean": fid_clean
            })

    total_tasks = len(valid_tasks)
    if total_tasks == 0:
        print("🎉 没有剩余任务。")
        exit()

    print(f"🔥 总有效任务: {total_tasks} 张")
    
    # 2. 任务分发
    chunk_size = math.ceil(total_tasks / NUM_WORKERS)
    process_list = []

    for i in range(NUM_WORKERS):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        worker_batch = valid_tasks[start_idx : end_idx]
        
        if not worker_batch:
            continue
            
        p = Process(target=worker_task, args=(i, worker_batch, final_output_dir))
        p.start()
        process_list.append(p)

    print(f"⚡️ {len(process_list)} 个进程已启动...")
    
    for p in process_list:
        p.join()

    print("\n🎉🎉🎉 所有并行任务圆满结束！")
