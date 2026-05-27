import tifffile as tiff
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
import time
import random
import gc
import psutil
import subprocess
import traceback
import re
import glob
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from pipeline_config import BASE_DIR, PATHS

# ================= 配置区域 =================
# 1. Mask 路径
base_dir = BASE_DIR
seg_dir = os.path.join(base_dir, "mask")

# 2. 原图路径
raw_img_dir = os.path.join(base_dir, "img")

# 3. 输出路径
output_dir = os.path.join(base_dir, "trace_app2")

swc_dir = os.path.join(output_dir, "down_sampled_swcs_app2") 
marker_dir = os.path.join(output_dir, "markers")

# 日志文件
log_file = os.path.join(output_dir, "process_log.txt")
timeout_record = os.path.join(output_dir, "record_timeout_5min.txt")
error_record = os.path.join(output_dir, "record_failed_5min.txt")
cost_time_record = os.path.join(output_dir, "record_cost_time_5min.txt")

# Vaa3D 程序路径
V3D_PATH = PATHS["vaa3d_path"]
# ===========================================

os.makedirs(swc_dir, exist_ok=True)
os.makedirs(marker_dir, exist_ok=True)

class TimeoutException(Exception): pass

# === 辅助函数：强力杀进程 ===
def kill_child_processes():
    try:
        parent = psutil.Process(os.getpid())
        children = parent.children(recursive=True)
        for child in children:
            try:
                if "resource_tracker" in child.name() or "resource_tracker" in " ".join(child.cmdline()):
                    continue 
                child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except psutil.NoSuchProcess:
        pass

# === 可视化函数：生成 2x3 对比图 (修复 Y 轴翻转) ===
# === 可视化函数：生成 2x3 对比图 (修复 YZ 视图翻转) ===
def generate_center_marker(neuron_id, shape_zyx, output_dir):
    try:
        try:
            from neuroutils.meta.neuron import get_soma_pos, get_xy_z_resolution
            soma_x_raw, soma_y_raw, soma_z_raw = get_soma_pos(int(neuron_id))
            xy_resolution, z_resolution = get_xy_z_resolution(int(neuron_id))
            center_z = int(round(soma_z_raw * (z_resolution / 1000)))
            center_y = int(round(soma_y_raw * (xy_resolution / 1000)))
            center_x = int(round(soma_x_raw * (xy_resolution / 1000)))
            print( f"【Info】Using Geometric Center: ({center_x}, {center_y}, {center_z}) for neuron {neuron_id}")
            print( f"【Info】Original Soma Position (raw): ({soma_x_raw}, {soma_y_raw}, {soma_z_raw}), Resolutions: (XY: {xy_resolution} nm, Z: {z_resolution} nm)" )
            # raise ImportError("Prefer Geometric Center") 
        except:
            d, h, w = shape_zyx
            center_x = w // 2
            center_y = h // 2
            center_z = d // 2
        
        marker_header = "##x,y,z,radius,shape,name,comment, color_r,color_g,color_b"
        marker_content = f"{center_x}, {center_y}, {center_z}, 1, 1, soma_{neuron_id}, , 255, 0, 0"
        filename = f"marker_{neuron_id}.marker"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(marker_header + "\n")
            f.write(marker_content)
            
        return filepath
    except Exception as e:
        raise Exception(f"Marker Gen Failed: {e}")

# === APP2 运行逻辑 ===
def run_app2_locally(img_file, marker_file, out_swc_file, timeout_sec):
    if os.path.exists(out_swc_file): return

    expected_result_base = img_file + "_app2.swc"
    
    cmd = (
        f'xvfb-run -a -s "-screen 0 640x480x16" {V3D_PATH} -x vn2 -f app2 -i {img_file} -p {marker_file} 0 10 0 1 1 1 1 10'
    )
    env = os.environ.copy()
    ld_path = os.path.dirname(os.path.dirname(V3D_PATH)) + "/lib"
    
    try:
        subprocess.run(
            cmd, shell=True, check=True, env=env, timeout=timeout_sec, capture_output=True
        )
        
        if os.path.exists(expected_result_base):
            os.rename(expected_result_base, out_swc_file)
            return

        ini_file = img_file + "_ini.swc"
        if os.path.exists(ini_file):
            os.rename(ini_file, out_swc_file)
            return

        potential_files = glob.glob(img_file + "*app2*.swc")
        if potential_files:
            os.rename(potential_files[0], out_swc_file)
            return

        raise Exception("Trace failed: No output SWC found")

    except subprocess.TimeoutExpired:
        raise TimeoutException("Subprocess timed out")
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode('utf-8', errors='ignore') if e.stderr else "No stderr"
        if "Xlib:  extension" in err_msg and not os.path.exists(out_swc_file): pass 
        raise Exception(f"Vaa3D Crash: {err_msg[:100]}")

# === Worker 函数 ===
def process_single_file(mask_file, swc_file, marker_dir, vis_dir, raw_img_dir):
    fname = os.path.basename(mask_file)
    start_time = time.time()
    
    match = re.search(r'(\d+)', fname)
    if match:
        neuron_id_str = match.group(1)
    else:
        return (fname, "FAIL", "No numeric ID", 0)

    raw_filename = f"image_{neuron_id_str}_0000.tif"
    raw_file_path = os.path.join(raw_img_dir, raw_filename)
    
    if not os.path.exists(raw_file_path):
        # 移除底下的 glob.glob 逻辑，直接判缺
        print(f"【错误】原图缺失: {raw_file_path}")
        return (fname, "FAIL", f"Raw image missing: {raw_filename}", 0)

    if not os.path.exists(mask_file):
        print(f"【错误】Mask 文件缺失: {mask_file}")
        return (fname, "MISSING", "", 0)
    
    if os.path.exists(swc_file):
        return (fname, "SKIP", "", 0)

    CURRENT_TIMEOUT = 300 
    max_retries = 2
    
    for attempt in range(max_retries):
        print(f"开始第一次{swc_file}")
        generated_marker_path = None
        try:
            sleep_time = random.uniform(0.1, 1.0) if attempt == 0 else random.uniform(2.0, 4.0)
            time.sleep(sleep_time)

            with tiff.TiffFile(mask_file) as tif:
                img_shape = tif.pages[0].shape 
            
            generated_marker_path = generate_center_marker(neuron_id_str, img_shape, marker_dir)
            
            run_app2_locally(mask_file, generated_marker_path, swc_file, timeout_sec=CURRENT_TIMEOUT)
            
            # if generated_marker_path and os.path.exists(generated_marker_path):
            #     try: os.remove(generated_marker_path)
            #     except: pass

            elapsed_min = (time.time() - start_time) / 60.0
            return (fname, "SUCCESS", "", elapsed_min)

        except TimeoutException:
            kill_child_processes()
            if attempt == max_retries - 1: 
                return (fname, "TIMEOUT", f"> {CURRENT_TIMEOUT}s", (time.time()-start_time)/60.0)
        except Exception as e:
            kill_child_processes()
            err_str = str(e)
            if ("Exit 139" in err_str or "Crash" in err_str) and attempt < max_retries - 1:
                continue 
            return (fname, "FAIL", err_str, (time.time()-start_time)/60.0)
        finally:
             gc.collect()
    
    return (fname, "FAIL", "Unknown Error", (time.time()-start_time)/60.0)

# === 主程序 ===
if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Scanning files in {seg_dir} ...")
    all_files = []
    # 使用 os.scandir 避免一次性加载海量目录结构
    with os.scandir(seg_dir) as entries:
        for entry in entries:
            if entry.name.endswith(".tif") and entry.is_file():
                all_files.append(entry.name)
    
    def task_generator():
        for f in all_files:
            yield (
                os.path.join(seg_dir, f), 
                os.path.join(swc_dir, f.replace(".tif", ".swc")),
                marker_dir,
                None,
                raw_img_dir   
            )
            
    total_tasks = len(all_files)
    print(f"Total files: {total_tasks}")
    
    results_gen = Parallel(n_jobs=4, return_as="generator", pre_dispatch="2*n_jobs")(
        delayed(process_single_file)(*args) for args in task_generator()
    )

    stats = {"SUCCESS": 0, "SKIP": 0, "TIMEOUT": 0, "FAIL": 0, "MISSING": 0}
    
    try:
        with open(timeout_record, "a") as f_timeout, \
             open(error_record, "a") as f_error, \
             open(cost_time_record, "a") as f_cost:
            
            for result_tuple in tqdm(results_gen, total=total_tasks, unit="file"):
                fname, status, msg, cost_min = result_tuple

                stats[status] = stats.get(status, 0) + 1

                try:
                    f_cost.write(f"{fname},{cost_min:.2f}\n")
                    f_cost.flush()
                except: pass
                
                if status == "TIMEOUT":
                    f_timeout.write(f"{fname}\n")
                    f_timeout.flush()
                elif status == "FAIL":
                    f_error.write(f"{fname},{msg}\n")
                    f_error.flush() 

        print("\n=== Processing Finished ===")
        print(f"Success: {stats['SUCCESS']}")
        print(f"Failed : {stats['FAIL']}")

    except KeyboardInterrupt:
        print("\n[User stopped]")
    finally:
        kill_child_processes()
