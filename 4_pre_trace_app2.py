import tifffile as tiff
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
from pipeline_config import BASE_DIR, PATHS

# ================= 配置区域 =================
# 1. Mask 路径
base_dir = BASE_DIR
seg_dir = PATHS["merged_mask_dir"]

# 2. 原图路径
raw_img_dir = PATHS["image_1um_dir"]

# 3. 输出路径
output_dir = PATHS["trace_output_dir"]

swc_dir =  PATHS["trace_swc_dir"]
marker_dir =  PATHS["trace_marker_dir"]
vis_dir =  PATHS["trace_vis_dir"]

# 日志文件
log_file =  PATHS["pipeline_skip_log"]
timeout_record =  PATHS["trace_timeout_log"]
error_record =  PATHS["trace_error_log"]
cost_time_record =  PATHS["trace_cost_log"]

# Vaa3D 程序路径
V3D_PATH = str(PATHS["vaa3d_path"])
# ===========================================

os.makedirs(swc_dir, exist_ok=True)
os.makedirs(marker_dir, exist_ok=True)
os.makedirs(vis_dir, exist_ok=True)

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

def _xy_mip(volume):
    if volume.ndim == 3:
        return np.max(volume, axis=0)
    if volume.ndim == 2:
        return volume
    return np.max(np.squeeze(volume), axis=0)


def _display_vmax(img_2d):
    if np.max(img_2d) <= 0:
        return 1
    vmax = np.percentile(img_2d, 99.5)
    return vmax if vmax > 0 else np.max(img_2d)


def _read_swc_xy_edges(swc_path):
    nodes = {}
    edges = []
    if not swc_path or not os.path.exists(swc_path):
        return nodes, edges
    with open(swc_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                node_id = int(float(parts[0]))
                x = float(parts[2])
                y = float(parts[3])
                parent_id = int(float(parts[6]))
            except ValueError:
                continue
            nodes[node_id] = (x, y)
            if parent_id != -1:
                edges.append((node_id, parent_id))
    return nodes, edges


LEGACY_SWC_Y_FLIP_MARKER = "# y_axis_flipped_to_image_coordinates"
SWC_Y_FLIP_MARKER = "# app2_native_to_image_zero_based_v2"


def image_to_app2_marker(xyz, shape_zyx):
    """Image voxel -> marker for the calibrated Vaa3D 3.601 vn2 plugin.

    TIFF loading reverses Y. This plugin subtracts TWO from marker-file
    coordinates before tracing (verified against its initial SWC roots).
    Native SWCs themselves are zero based. Do not apply this offset to SWCs.
    """
    xyz = np.asarray(xyz, dtype=float)
    bounds = np.asarray(shape_zyx[::-1], dtype=int)
    if xyz.shape != (3,) or not np.isfinite(xyz).all() or np.any(xyz < 0) or np.any(xyz >= bounds):
        raise ValueError(f'Soma {xyz} outside image {shape_zyx}')
    x, y, z = np.rint(xyz).astype(int)
    if np.any(np.array([x, y, z]) >= bounds):
        raise ValueError('Rounded soma outside image')
    return [int(x + 2), int(shape_zyx[-2] - 1 - y + 2), int(z + 2)]


def flip_swc_y_to_image_coordinates(swc_path, image_height):
    """Flip APP2's Y coordinate once so SWC coordinates match image coordinates."""
    with open(swc_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    if any(line.strip() == SWC_Y_FLIP_MARKER for line in lines):
        return False
    if any(line.strip() == LEGACY_SWC_Y_FLIP_MARKER for line in lines):
        raise ValueError('Legacy H-y coordinates: migrate Y by -1 or regenerate from native SWC; do not flip twice')

    output_lines = [SWC_Y_FLIP_MARKER + "\n"]
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            output_lines.append(line)
            continue

        parts = stripped.split()
        if len(parts) < 7:
            output_lines.append(line)
            continue
        try:
            parts[3] = f"{float(image_height) - 1 - float(parts[3]):.5f}"
        except ValueError:
            output_lines.append(line)
            continue
        output_lines.append(" ".join(parts) + "\n")

    temp_path = swc_path + ".yflip.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        f.writelines(output_lines)
    os.replace(temp_path, swc_path)
    return True


def generate_xy_mip_visualization(raw_img_path, mask_path, swc_path, output_png_path, neuron_id):
    try:
        os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
        raw = tiff.imread(raw_img_path)
        mask = tiff.imread(mask_path)
        raw_xy = _xy_mip(raw)
        mask_xy = _xy_mip(mask) > 0
        nodes, edges = _read_swc_xy_edges(swc_path)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
        for ax in axes:
            ax.imshow(raw_xy, cmap="gray", vmin=0, vmax=_display_vmax(raw_xy))
            ax.axis("off")

        axes[0].set_title(f"Neuron {neuron_id} APP2 XY")
        for node_id, parent_id in edges:
            if node_id not in nodes or parent_id not in nodes:
                continue
            x0, y0 = nodes[node_id]
            x1, y1 = nodes[parent_id]
            axes[0].plot([x0, x1], [y0, y1], color="cyan", linewidth=0.7, alpha=0.85)
        if nodes:
            xs, ys = zip(*nodes.values())
            axes[0].scatter(xs, ys, s=2, c="yellow", alpha=0.7)

        axes[1].set_title(f"Neuron {neuron_id} Raw + Mask XY")
        overlay = np.zeros((*mask_xy.shape, 4), dtype=float)
        overlay[..., 0] = 1.0
        overlay[..., 3] = mask_xy.astype(float) * 0.35
        axes[1].imshow(overlay)

        fig.tight_layout()
        fig.savefig(output_png_path, bbox_inches="tight")
        plt.close(fig)
        return "OK"
    except Exception as e:
        plt.close("all")
        return f"VIS_FAIL: {e}"


# === Marker 生成函数 ===
def generate_center_marker(neuron_id, shape_zyx, output_dir):
    try:
        try:
            from neuroutils.meta.neuron import get_soma_pos, get_xy_z_resolution
            soma_x_raw, soma_y_raw, soma_z_raw = get_soma_pos(int(neuron_id))
            xy_resolution, z_resolution = get_xy_z_resolution(int(neuron_id))
            center_z = int(round(soma_z_raw * (z_resolution / 1000)))
            center_y = int(round(soma_y_raw * (xy_resolution / 1000)))
            center_x = int(round(soma_x_raw * (xy_resolution / 1000)))
            # print( f"【Info】Using Geometric Center: ({center_x}, {center_y}, {center_z}) for neuron {neuron_id}")
            # print( f"【Info】Original Soma Position (raw): ({soma_x_raw}, {soma_y_raw}, {soma_z_raw}), Resolutions: (XY: {xy_resolution} nm, Z: {z_resolution} nm)" )
            # raise ImportError("Prefer Geometric Center") 
        # except:
        #     d, h, w = shape_zyx
        #     center_x = w // 2
        #     center_y = h // 2
        #     center_z = d // 2
        except Exception as e:
            raise Exception(f"Failed to compute geometric center: {e},for neuron {neuron_id}")

        
        marker_header = "##x,y,z,radius,shape,name,comment, color_r,color_g,color_b"
        marker_xyz = image_to_app2_marker([center_x, center_y, center_z], shape_zyx)
        marker_content = ", ".join(map(str, marker_xyz)) + f", 1, 1, soma_{neuron_id}, , 255, 0, 0"
        filename = f"marker_{neuron_id}.marker"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(marker_header + "\n")
            f.write(marker_content)
            
        return filepath
    except Exception as e:
        raise Exception(f"Marker Gen Failed: {e}")

# === APP2 运行逻辑 ===
def _build_vaa3d_env(v3d_path):
    env = os.environ.copy()
    v3d_bin_dir = os.path.dirname(os.path.abspath(v3d_path))
    current_ld_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = (
        f"{v3d_bin_dir}:{current_ld_path}" if current_ld_path else v3d_bin_dir
    )
    return env


def run_app2_locally(img_file, marker_file, out_swc_file, timeout_sec):
    if os.path.exists(out_swc_file): return

    expected_result_base = img_file + "_app2.swc"
    app2_candidates = lambda: sorted(
        glob.glob(img_file + "*_app2.swc"),
        key=lambda path: os.path.getmtime(path),
        reverse=True,
    )
    ini_candidates = lambda: sorted(glob.glob(img_file + "*_ini.swc"))

    def cleanup_ini_outputs():
        for ini_path in ini_candidates():
            try:
                os.remove(ini_path)
            except OSError:
                pass

    cmd = [
        "xvfb-run", "-a", "-s", "-screen 0 640x480x16",
        V3D_PATH,
        "-x", "vn2",
        "-f", "app2",
        "-i", img_file,
        "-p", marker_file,
        "0", "10", "0", "1", "1", "1", "1", "10",
    ]
    env = _build_vaa3d_env(V3D_PATH)
    
    try:
        subprocess.run(
            cmd, check=True, env=env, timeout=timeout_sec, capture_output=True
        )
        
        if os.path.exists(expected_result_base):
            os.rename(expected_result_base, out_swc_file)
            cleanup_ini_outputs()
            return

        potential_files = app2_candidates()
        if potential_files:
            os.rename(potential_files[0], out_swc_file)
            cleanup_ini_outputs()
            return

        cleanup_ini_outputs()
        raise Exception("Trace failed: No APP2 output SWC found")

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
    
    with tiff.TiffFile(mask_file) as tif:
        mask_shape = tif.series[0].shape
    image_height = mask_shape[-2]

    if os.path.exists(swc_file):
        swc_was_flipped = flip_swc_y_to_image_coordinates(swc_file, image_height)
        if vis_dir:
            vis_png = os.path.join(vis_dir, f"{neuron_id_str}_xy_mip.png")
            if swc_was_flipped or not os.path.exists(vis_png):
                generate_xy_mip_visualization(raw_file_path, mask_file, swc_file, vis_png, neuron_id_str)
        return (fname, "SKIP", "", 0)

    CURRENT_TIMEOUT = 300 
    max_retries = 2
    
    for attempt in range(max_retries):
        generated_marker_path = None
        try:
            sleep_time = random.uniform(0.1, 1.0) if attempt == 0 else random.uniform(2.0, 4.0)
            time.sleep(sleep_time)

            generated_marker_path = generate_center_marker(neuron_id_str, mask_shape, marker_dir)
            
            run_app2_locally(mask_file, generated_marker_path, swc_file, timeout_sec=CURRENT_TIMEOUT)

            # APP2 uses a bottom-origin Y axis. Convert it before saving downstream
            # products so both the SWC and its MIP use image (top-origin) coordinates.
            flip_swc_y_to_image_coordinates(swc_file, image_height)

            if vis_dir and os.path.exists(swc_file):
                vis_png = os.path.join(vis_dir, f"{neuron_id_str}_xy_mip.png")
                generate_xy_mip_visualization(raw_file_path, mask_file, swc_file, vis_png, neuron_id_str)

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
                vis_dir,
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
        print(f"XY visualization saved in: {vis_dir}")

    except KeyboardInterrupt:
        print("\n[User stopped]")
    finally:
        kill_child_processes()
