import tifffile as tiff
import os
import sys
import time
import random
import gc
import psutil
import subprocess
import traceback
from joblib import Parallel, delayed

# === 引入 Rich 组件 ===
from rich.live import Live
from rich.table import Table
from rich.console import Console, Group
from rich import box
from rich.text import Text
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn, 
    TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn, ProgressColumn
)

# ================= 配置区域 =================
# 1. 原始数据路径 (保持不变)
seg_dir = "/data/disk3/B4.5/concat"

output_dir = "/data/disk3/B4.5/retrace"

skel_dir = os.path.join(output_dir, "down_sampled_uint8")
swc_dir = os.path.join(output_dir, "down_sampled_swcs_pzy") # 独立的 SWC 文件夹

# 新的日志文件
log_file = os.path.join(output_dir, "process_log.txt")
timeout_record = os.path.join(output_dir, "record_timeout_5min.txt")
error_record = os.path.join(output_dir, "record_failed_5min.txt")
cost_time_record = os.path.join(output_dir, "record_cost_time_5min.txt")

# Vaa3D 程序路径
V3D_PATH = "/home/pzy/Vaa3D_CentOS_64bit_v3.601/bin/vaa3d" 
# ===========================================

os.makedirs(skel_dir, exist_ok=True)
os.makedirs(swc_dir, exist_ok=True)

class TimeoutException(Exception): pass

# === 自定义：安全的速度显示列 ===
class SafeSpeedColumn(ProgressColumn):
    def render(self, task) -> Text:
        if task.speed is None:
            return Text("Init...", style="yellow")
        return Text(f"{task.speed:.1f} it/s", style="bold green")

# === 辅助函数：强力杀进程 ===
def kill_child_processes():
    try:
        parent = psutil.Process(os.getpid())
        children = parent.children(recursive=True)
        for child in children:
            try:
                # 保护 resource_tracker 不被误杀
                if "resource_tracker" in child.name() or "resource_tracker" in " ".join(child.cmdline()):
                    continue 
                child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except psutil.NoSuchProcess:
        pass

# === 核心追踪逻辑 ===
def run_neutube_locally(img_file, out_swc_file, timeout_sec):
    if os.path.exists(out_swc_file): return

    result_name = img_file + "_neutube.swc"
    cmd = f'xvfb-run -a -s "-screen 0 640x480x16" {V3D_PATH} -x neuTube -f neutube_trace -i {img_file} -p 1 1'
    
    env = os.environ.copy()
    ld_path = os.path.dirname(V3D_PATH)
    env['LD_LIBRARY_PATH'] = f"{ld_path}:{env.get('LD_LIBRARY_PATH', '')}"

    try:
        subprocess.run(
            cmd, shell=True, check=True, env=env, timeout=timeout_sec, capture_output=True
        )
        if os.path.exists(result_name):
            os.rename(result_name, out_swc_file)
        else:
            raise Exception("Vaa3D ran but output file missing")
    except subprocess.TimeoutExpired:
        raise TimeoutException("Subprocess timed out")
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode('utf-8', errors='ignore') if e.stderr else "No stderr"
        raise Exception(f"Vaa3D Crash (Exit {e.returncode}): {err_msg[:50]}")

# === Worker 函数 ===
def process_single_file(seg_file, skel_file, swc_file):
    fname = os.path.basename(seg_file)
    start_time = time.time()
    
    if not os.path.exists(seg_file):
        elapsed_min = (time.time() - start_time) / 60.0
        return (fname, "MISSING", "", elapsed_min)
    if os.path.exists(swc_file):
        elapsed_min = (time.time() - start_time) / 60.0
        return (fname, "SKIP", "", elapsed_min)

    # === 【关键修改】设置超时时间为 300秒 (5分钟) ===
    CURRENT_TIMEOUT = 300 

    max_retries = 2
    for attempt in range(max_retries):
        try:
            # 错峰启动
            sleep_time = random.uniform(0.1, 1.0) if attempt == 0 else random.uniform(2.0, 4.0)
            time.sleep(sleep_time)

            seg = tiff.imread(seg_file)
            skel = (seg * 255).astype("uint8")
            tiff.imwrite(skel_file, skel)
            del seg

            run_neutube_locally(skel_file, swc_file, timeout_sec=CURRENT_TIMEOUT)
            elapsed_min = (time.time() - start_time) / 60.0
            return (fname, "SUCCESS", "", elapsed_min)

        except TimeoutException:
            kill_child_processes()
            if attempt == max_retries - 1: 
                elapsed_min = (time.time() - start_time) / 60.0
                return (fname, "TIMEOUT", f"Exceeded {CURRENT_TIMEOUT}s", elapsed_min)
        except Exception as e:
            kill_child_processes()
            err_str = str(e)
            if ("Exit 139" in err_str or "Crash" in err_str) and attempt < max_retries - 1:
                continue 
            elapsed_min = (time.time() - start_time) / 60.0
            return (fname, "FAIL", err_str, elapsed_min)
        finally:
            gc.collect()
    
    elapsed_min = (time.time() - start_time) / 60.0
    return (fname, "FAIL", "Unknown Error", elapsed_min)

# === 仪表盘生成器 ===
def generate_dashboard(progress_obj, stats, total, last_error, start_time):
    current_time = time.time()
    elapsed = current_time - start_time
    processed = sum(stats.values())
    
    if elapsed > 0.5: 
        speed_per_min = (processed / elapsed) * 60
    else:
        speed_per_min = 0.0

    table = Table(title="Retry Dashboard (Timeout=300s)", box=box.ROUNDED, width=80)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Note", justify="left")

    pct = (processed / total) * 100 if total > 0 else 0
    fail_count = stats.get("FAIL", 0)
    timeout_count = stats.get("TIMEOUT", 0)

    table.add_row("Progress", f"{processed}/{total}", f"{pct:.1f}%")
    table.add_row("Speed", f"{speed_per_min:.1f} files/min", "Global Avg")
    table.add_row("[green]Success[/green]", str(stats.get("SUCCESS", 0)), "Recovered")
    table.add_row("[dim]Skipped[/dim]", str(stats.get("SKIP", 0)), "Exists")
    table.add_row("[yellow]Timeout[/yellow]", str(timeout_count), "Still > 5min")
    table.add_row("[red]Errors[/red]", str(fail_count), "Still Crashing")
    
    if last_error:
        display_err = last_error if len(last_error) < 40 else last_error[:37] + "..."
        table.add_row("Last Error", "See Log", f"[red]{display_err}[/red]")

    return Group(progress_obj, table)

# === 主程序 ===
if __name__ == "__main__":
    # 初始化本次的日志文件
    os.makedirs(output_dir, exist_ok=True)
    with open(log_file, "w") as f: f.write(f"=== Retry Process (120s) started at {time.ctime()} ===\n")
    with open(timeout_record, "w") as f: f.write("Filename\n")
    with open(error_record, "w") as f: f.write("Filename,Error_Message\n")
    with open(cost_time_record, "w") as f: f.write("Filename,cost_time\n")
    # === 1. 收集需要重跑的文件列表 ===
    target_files = set()
    
    # === 2. 获取文件列表并过滤 ===
    all_files = [f for f in os.listdir(seg_dir) if f.endswith(".tif")]
    
    # === 【关键修改】倒序排列 ===
    #all_files.reverse() 
    #print("File list reversed. Processing from the end.")

    tasks = []
    skipped_count = 0
    
    for f in all_files:
        
        tasks.append((
            os.path.join(seg_dir, f), 
            os.path.join(skel_dir, f), 
            os.path.join(swc_dir, f.replace(".tif", ".swc"))
        ))

    total_tasks = len(tasks)
    print(f"Total files: {len(all_files)}")
    
    print(f"Total files to retry: {total_tasks}")
    print(f"Output Directory: {output_dir}")

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        TextColumn("•"),
        SafeSpeedColumn(),
    )
    task_id = progress.add_task("Retrying...", total=total_tasks)

    # 启动任务
    results_gen = Parallel(n_jobs=8, return_as="generator", pre_dispatch="2*n_jobs")(
        delayed(process_single_file)(s, k, w) for s, k, w in tasks
    )

    stats = {"SUCCESS": 0, "SKIP": 0, "TIMEOUT": 0, "FAIL": 0}
    last_err_msg = ""
    start_time = time.time()
    
    try:
        with open(timeout_record, "a") as f_timeout, open(error_record, "a") as f_error, open(cost_time_record, "a") as f_cost:
            with Live(generate_dashboard(progress, stats, total_tasks, last_err_msg, start_time), refresh_per_second=4) as live:
                for result_tuple in results_gen:
                    # unpack with cost (minutes)
                    fname, status, msg, cost_min = result_tuple

                    stats[status] = stats.get(status, 0) + 1

                    # write per-file cost_time as: Filename,cost_time
                    try:
                        f_cost.write(f"{fname},{cost_min:.2f}\n")
                        f_cost.flush()
                    except Exception:
                        pass
                    
                    if status == "TIMEOUT":
                        f_timeout.write(f"{fname}\n")
                        f_timeout.flush()
                    elif status == "FAIL":
                        last_err_msg = msg
                        f_error.write(f"{fname},{msg}\n")
                        f_error.flush() 

                    progress.update(task_id, advance=1)
                    live.update(generate_dashboard(progress, stats, total_tasks, last_err_msg, start_time))

    except KeyboardInterrupt:
        print("\n[yellow]User stopped the process (Ctrl+C).[/yellow]")
    except Exception as e:
        print("\n[red]CRITICAL ERROR IN MAIN LOOP:[/red]")
        traceback.print_exc()
    finally:
        print("\nCleaning up remaining processes...")
        kill_child_processes()
        print(f"Retry finished. Check results in: {output_dir}")
