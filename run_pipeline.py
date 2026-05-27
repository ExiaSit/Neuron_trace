"""Sequential one-neuron-at-a-time pipeline orchestrator.

This file intentionally calls the existing processing functions and command-line
models instead of changing image-processing logic in the stage scripts.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import os
import time
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, Union

import numpy as np
import pandas as pd

from pipeline_config import ENABLED_STAGES, GCUT, NEURON_IDS, NNUNET, PATHS, PIPELINE, PRUNING, SOMA

ROOT = Path(__file__).resolve().parent
StageFunc = Callable[[Union[int, str]], None]
StageEntry = tuple[str, StageFunc]


def load_stage_module(module_name: str, filename: str) -> Any:
    """Load an existing stage script whose filename is not a valid module name."""
    module_path = ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_dirs() -> None:
    for key, path in PATHS.items():
        if key.endswith(("_dir", "output_dir")):
            Path(path).mkdir(parents=True, exist_ok=True)


def set_nnunet_env(results_dir: str | Path) -> None:
    os.environ["nnUNet_raw"] = str(NNUNET["raw"])
    os.environ["nnUNet_preprocessed"] = str(NNUNET["preprocessed"])
    os.environ["nnUNet_results"] = str(results_dir)


def image_stem(neuron_id: int | str) -> str:
    return f"image_{neuron_id}"


def find_gcut_target_swc(neuron_id: int | str) -> Path:
    """Return the G-Cut SWC that should be passed to pruning."""
    stem = image_stem(neuron_id)
    gcut_dir = Path(PATHS["gcut_output_dir"])
    candidates = sorted(gcut_dir.glob(f"{stem}_0000_gcut_soma_*_TARGET.swc"))
    if not candidates:
        candidates = sorted(gcut_dir.glob(f"{stem}_gcut_soma_*_TARGET.swc"))
    if not candidates:
        candidates = sorted(gcut_dir.glob(f"{stem}_0000_gcut_soma_*.swc"))
    if not candidates:
        candidates = sorted(gcut_dir.glob(f"{stem}_gcut_soma_*.swc"))
    if not candidates:
        raise FileNotFoundError(f"No G-Cut SWC found for {stem} in {gcut_dir}")
    return candidates[0]


def prepare_gcut_swc_for_pruning(neuron_id: int | str) -> Path:
    """Copy the selected G-Cut SWC to a pruning-compatible name.

    6_gcut_pruning_copy60228.py searches for image_<id>.swc or
    image_<id>_0000.swc. G-Cut writes names like
    image_<id>_0000_gcut_soma_1_TARGET.swc, so the orchestrator creates a
    compatibility copy without changing G-Cut output naming.
    """
    source_swc = find_gcut_target_swc(neuron_id)
    selected_dir = Path(PATHS["gcut_selected_swc_dir"])
    selected_dir.mkdir(parents=True, exist_ok=True)
    compat_swc = selected_dir / f"{image_stem(neuron_id)}.swc"
    shutil.copy2(source_swc, compat_swc)
    print(f"[gcut] selected for pruning: {source_swc} -> {compat_swc}")
    return compat_swc


def run_nnunet_single(
    *,
    source_tif: Path,
    output_dir: Path,
    temp_dir: Path,
    dataset_id: str,
    results_dir: str | Path,
    output_stem: str,
) -> None:
    """Run nnUNet for exactly one tif by reusing the existing file naming scheme."""
    output_path = output_dir / f"{output_stem}.tif"
    if output_path.exists():
        print(f"[skip] nnUNet output exists: {output_path}")
        return
    if not source_tif.exists():
        raise FileNotFoundError(f"nnUNet input missing: {source_tif}")

    output_dir.mkdir(parents=True, exist_ok=True)
    job_temp_dir = temp_dir / output_stem
    if job_temp_dir.exists():
        shutil.rmtree(job_temp_dir)
    job_temp_dir.mkdir(parents=True, exist_ok=True)

    temp_input = job_temp_dir / f"{output_stem}_0000.tif"
    try:
        os.symlink(source_tif, temp_input)
    except OSError:
        shutil.copy2(source_tif, temp_input)

    set_nnunet_env(results_dir)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(NNUNET["gpu_id"])
    cmd = [
        "nnUNetv2_predict",
        "-i", str(job_temp_dir),
        "-o", str(output_dir),
        "-d", str(dataset_id),
        "-c", str(NNUNET["configuration"]),
        "-f", str(NNUNET["fold"]),
        "-device", str(NNUNET["device"]),
    ]
    print("[run] " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, env=env)
    finally:
        shutil.rmtree(job_temp_dir, ignore_errors=True)


def batched(items: Sequence[int | str], batch_size: int) -> Iterable[list[int | str]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start:start + batch_size])


def is_file_stable(path: Path, stable_seconds: float = 2.0, interval_seconds: float = 0.5) -> bool:
    if not path.exists():
        return False
    first = path.stat()
    if first.st_size <= 0:
        return False
    time.sleep(interval_seconds)
    if not path.exists():
        return False
    second = path.stat()
    return (
        first.st_size == second.st_size
        and first.st_mtime_ns == second.st_mtime_ns
        and time.time() - second.st_mtime >= stable_seconds
    )


def prepare_nnunet_batch_input(
    *,
    source_tifs: dict[int | str, Path],
    output_dir: Path,
    temp_dir: Path,
    temp_name: str,
) -> tuple[Path, list[int | str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    job_temp_dir = temp_dir / temp_name
    if job_temp_dir.exists():
        shutil.rmtree(job_temp_dir)
    job_temp_dir.mkdir(parents=True, exist_ok=True)

    pending: list[int | str] = []
    for neuron_id, source_tif in source_tifs.items():
        stem = image_stem(neuron_id)
        output_path = output_dir / f"{stem}.tif"
        if output_path.exists():
            print(f"[skip] nnUNet output exists: {output_path}")
            continue
        if not source_tif.exists():
            raise FileNotFoundError(f"nnUNet input missing: {source_tif}")
        temp_input = job_temp_dir / f"{stem}_0000.tif"
        try:
            os.symlink(source_tif, temp_input)
        except OSError:
            shutil.copy2(source_tif, temp_input)
        pending.append(neuron_id)

    return job_temp_dir, pending


def build_nnunet_cmd(input_dir: Path, output_dir: Path, dataset_id: str) -> list[str]:
    return [
        "nnUNetv2_predict",
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-d", str(dataset_id),
        "-c", str(NNUNET["configuration"]),
        "-f", str(NNUNET["fold"]),
        "-device", str(NNUNET["device"]),
    ]


def run_nnunet_batch(
    *,
    source_tifs: dict[int | str, Path],
    output_dir: Path,
    temp_dir: Path,
    dataset_id: str,
    results_dir: str | Path,
    temp_name: str,
) -> list[int | str]:
    job_temp_dir, pending = prepare_nnunet_batch_input(
        source_tifs=source_tifs,
        output_dir=output_dir,
        temp_dir=temp_dir,
        temp_name=temp_name,
    )
    if not pending:
        shutil.rmtree(job_temp_dir, ignore_errors=True)
        return []

    set_nnunet_env(results_dir)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(NNUNET["gpu_id"])
    cmd = build_nnunet_cmd(job_temp_dir, output_dir, str(dataset_id))
    print("[run] " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, env=env)
    finally:
        shutil.rmtree(job_temp_dir, ignore_errors=True)
    return pending


def run_nnunet_batch_with_output_watch(
    *,
    source_tifs: dict[int | str, Path],
    output_dir: Path,
    temp_dir: Path,
    dataset_id: str,
    results_dir: str | Path,
    temp_name: str,
    on_output_ready: Callable[[int | str], None],
    poll_seconds: float = 2.0,
) -> list[int | str]:
    job_temp_dir, pending = prepare_nnunet_batch_input(
        source_tifs=source_tifs,
        output_dir=output_dir,
        temp_dir=temp_dir,
        temp_name=temp_name,
    )

    completed: list[int | str] = []
    for neuron_id in source_tifs:
        output_path = output_dir / f"{image_stem(neuron_id)}.tif"
        if neuron_id not in pending and is_file_stable(output_path):
            on_output_ready(neuron_id)
            completed.append(neuron_id)

    if not pending:
        shutil.rmtree(job_temp_dir, ignore_errors=True)
        return completed

    set_nnunet_env(results_dir)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(NNUNET["gpu_id"])
    cmd = build_nnunet_cmd(job_temp_dir, output_dir, str(dataset_id))
    print("[run] " + " ".join(cmd))
    process = subprocess.Popen(cmd, env=env)
    remaining = set(pending)

    try:
        while remaining:
            ready_now = []
            for neuron_id in list(remaining):
                output_path = output_dir / f"{image_stem(neuron_id)}.tif"
                if is_file_stable(output_path):
                    ready_now.append(neuron_id)
            for neuron_id in ready_now:
                remaining.remove(neuron_id)
                completed.append(neuron_id)
                on_output_ready(neuron_id)

            return_code = process.poll()
            if return_code is not None:
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, cmd)
                break
            time.sleep(poll_seconds)

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)

        for neuron_id in list(remaining):
            output_path = output_dir / f"{image_stem(neuron_id)}.tif"
            if not is_file_stable(output_path, stable_seconds=0.0):
                raise FileNotFoundError(f"nnUNet output missing or still changing: {output_path}")
            remaining.remove(neuron_id)
            completed.append(neuron_id)
            on_output_ready(neuron_id)
    finally:
        if process.poll() is None:
            process.terminate()
        shutil.rmtree(job_temp_dir, ignore_errors=True)

    return completed


def run_neurite_infer_batch_with_watch(
    neuron_ids: Sequence[int | str],
    batch_size: int,
    on_output_ready: Callable[[int | str], None],
) -> None:
    for batch_index, batch in enumerate(batched(neuron_ids, batch_size)):
        source_tifs = {neuron_id: Path(PATHS["image_1um_dir"]) / f"{image_stem(neuron_id)}_0000.tif" for neuron_id in batch}

        def handle_ready(neuron_id: int | str) -> None:
            on_output_ready(neuron_id)

        run_nnunet_batch_with_output_watch(
            source_tifs=source_tifs,
            output_dir=Path(PATHS["neurite_seg_dir"]),
            temp_dir=Path(PATHS["neurite_temp_dir"]),
            dataset_id=str(NNUNET["neurite_dataset_id"]),
            results_dir=NNUNET["neurite_results"],
            temp_name=f"neurite_batch_{batch_index}_{image_stem(batch[0])}_{image_stem(batch[-1])}",
            on_output_ready=handle_ready,
        )


def run_rescale(neuron_id: int | str) -> None:
    mod = load_stage_module("stage_rescale", "1_rescale_cpu.py")
    mod.test_dir = str(PATHS["image_1um_dir"])
    Path(mod.test_dir).mkdir(parents=True, exist_ok=True)
    mod.prepare_nnunet_file(neuron_id)


def run_soma_crop(neuron_id: int | str) -> None:
    stem = image_stem(neuron_id)
    crop_tif = Path(PATHS["soma_crop_dir"]) / f"{stem}_0000.tif"
    crop_json = Path(PATHS["soma_crop_dir"]) / f"{stem}.json"
    if crop_tif.exists() and crop_json.exists():
        print(f"[skip] soma crop output exists: {crop_tif}, {crop_json}")
        return

    mod = load_stage_module("stage_soma", "2_soma_crop_seg.py")
    mod.img_1um_dir = str(PATHS["image_1um_dir"])
    mod.soma_crop_dir = str(PATHS["soma_crop_dir"])
    mod.soma_seg_dir = str(PATHS["soma_seg_dir"])
    mod.base_temp_dir = str(PATHS["soma_temp_dir"])
    mod.skipped_neurons_log_path = str(PATHS["soma_skipped_log"])
    mod.SAVE_MIP_VISUALIZATION = bool(SOMA["save_mip_visualization"])
    Path(mod.soma_crop_dir).mkdir(parents=True, exist_ok=True)
    Path(mod.soma_seg_dir).mkdir(parents=True, exist_ok=True)
    mod.prepare_nnunet_file(neuron_id, SOMA["target_block_size"], save_vis=mod.SAVE_MIP_VISUALIZATION)


def run_soma_infer(neuron_id: int | str) -> None:
    stem = image_stem(neuron_id)
    run_nnunet_single(
        source_tif=Path(PATHS["soma_crop_dir"]) / f"{stem}_0000.tif",
        output_dir=Path(PATHS["soma_seg_dir"]),
        temp_dir=Path(PATHS["soma_temp_dir"]),
        dataset_id=str(NNUNET["soma_dataset_id"]),
        results_dir=NNUNET["soma_results"],
        output_stem=stem,
    )


def run_neurite_infer(neuron_id: int | str) -> None:
    stem = image_stem(neuron_id)
    run_nnunet_single(
        source_tif=Path(PATHS["image_1um_dir"]) / f"{stem}_0000.tif",
        output_dir=Path(PATHS["neurite_seg_dir"]),
        temp_dir=Path(PATHS["neurite_temp_dir"]),
        dataset_id=str(NNUNET["neurite_dataset_id"]),
        results_dir=NNUNET["neurite_results"],
        output_stem=stem,
    )


def run_merge(neuron_id: int | str) -> None:
    mod = load_stage_module("stage_concat", "3.concat_neurite_soma.py")
    stem = image_stem(neuron_id)
    tmp_json_dir = Path(PATHS["merge_temp_json_dir"]) / str(neuron_id)
    if tmp_json_dir.exists():
        shutil.rmtree(tmp_json_dir)
    tmp_json_dir.mkdir(parents=True, exist_ok=True)
    source_json = Path(PATHS["soma_crop_dir"]) / f"{stem}.json"
    if not source_json.exists():
        raise FileNotFoundError(f"Soma crop json missing: {source_json}")
    shutil.copy2(source_json, tmp_json_dir / source_json.name)
    Path(PATHS["merged_mask_dir"]).mkdir(parents=True, exist_ok=True)
    mod.restore_crops_to_original(
        str(PATHS["neurite_seg_dir"]),
        str(tmp_json_dir),
        str(PATHS["soma_seg_dir"]),
        str(PATHS["merged_mask_dir"]),
        mod.name_parser,
        save_preview=False,
    )


def run_app2_trace(neuron_id: int | str) -> None:
    mod = load_stage_module("stage_app2", "4_pre_trace_app2.py")
    mod.V3D_PATH = str(PATHS["vaa3d_path"])
    for path_key in ("trace_output_dir", "trace_swc_dir", "trace_marker_dir"):
        Path(PATHS[path_key]).mkdir(parents=True, exist_ok=True)
    stem = image_stem(neuron_id)
    result = mod.process_single_file(
        str(Path(PATHS["merged_mask_dir"]) / f"{stem}.tif"),
        str(Path(PATHS["trace_swc_dir"]) / f"{stem}.swc"),
        str(PATHS["trace_marker_dir"]),
        None,
        str(PATHS["image_1um_dir"]),
    )
    fname, status, msg, cost_min = result
    with open(PATHS["trace_cost_log"], "a", encoding="utf-8") as f_cost:
        f_cost.write(f"{fname},{cost_min:.2f}\n")
    if status == "TIMEOUT":
        with open(PATHS["trace_timeout_log"], "a", encoding="utf-8") as f_timeout:
            f_timeout.write(f"{fname}\n")
    elif status == "FAIL":
        with open(PATHS["trace_error_log"], "a", encoding="utf-8") as f_error:
            f_error.write(f"{fname},{msg}\n")
    if status not in {"SUCCESS", "SKIP"}:
        raise RuntimeError(f"APP2 failed for {fname}: {status} {msg}")



def run_gcut(neuron_id: int | str, meta_df: pd.DataFrame) -> bool:
    mod = load_stage_module("stage_gcut", "5.gcut_pipeline.py")
    Path(PATHS["gcut_output_dir"]).mkdir(parents=True, exist_ok=True)
    stem = image_stem(neuron_id)
    swc_path = Path(PATHS["trace_swc_dir"]) / f"{stem}.swc"
    if not swc_has_nodes(swc_path):
        reason = f"APP2 SWC is empty or has no valid nodes: {swc_path}"
        print(f"[skip] {stem}: {reason}")
        log_pipeline_skip(Path(PATHS["gcut_error_log"]), neuron_id, reason)
        return False

    args = (
        Path(PATHS["image_1um_dir"]) / f"{stem}_0000.tif",
        Path(PATHS["merged_mask_dir"]) / f"{stem}.tif",
        swc_path,
        Path(PATHS["gcut_output_dir"]),
        GCUT["target_percentile"],
        GCUT["gsdt_threshold_x"],
        Path(PATHS["gcut_error_log"]),
        Path(PATHS["soma_crop_dir"]) / f"{stem}.json",
        Path(PATHS["soma_seg_dir"]) / f"{stem}.tif",
        meta_df,
    )
    name, _result, log_msg = mod.worker_task(args)
    if log_msg and any(token in log_msg for token in ("失败", "崩溃", "Traceback", "Error", "不存在")):
        raise RuntimeError(f"G-Cut reported an issue for {name}: {log_msg}")
    try:
        prepare_gcut_swc_for_pruning(neuron_id)
    except FileNotFoundError:
        reason = f"G-Cut did not produce a target SWC for {stem}: {log_msg}"
        print(f"[skip] {stem}: {reason}")
        log_pipeline_skip(Path(PATHS["gcut_error_log"]), neuron_id, reason)
        return False
    return True


def run_prune(neuron_id: int | str, meta_df: pd.DataFrame) -> None:
    prepare_gcut_swc_for_pruning(neuron_id)
    mod = load_stage_module("stage_prune", "6_gcut_pruning_copy60228.py")
    Path(PATHS["prune_mip_dir"]).mkdir(parents=True, exist_ok=True)
    Path(PATHS["prune_swc_dir"]).mkdir(parents=True, exist_ok=True)
    config = {
        **PRUNING,
        "downsample_scale": np.array(PRUNING["downsample_scale"]),
        "meta_file": str(PATHS["meta_file"]),
        "meta": meta_df,
        "concat_dir": str(PATHS["merged_mask_dir"]),
        "traced_dir": str(PATHS["gcut_selected_swc_dir"]),
        "raw_image_dir": str(PATHS["image_1um_dir"]),
        "mip_dir": str(PATHS["prune_mip_dir"]),
        "out_swc_dir": str(PATHS["prune_swc_dir"]),
        "soma_img_dir": str(PATHS["soma_crop_dir"]),
        "soma_mask_dir": str(PATHS["soma_seg_dir"]),
    }
    result = mod.worker_task(str(Path(PATHS["merged_mask_dir"]) / f"{image_stem(neuron_id)}.tif"), config)
    if result["status"] not in {"success", "skipped"}:
        with open(PATHS["prune_error_log"], "a", encoding="utf-8") as f:
            f.write(f"\n[{result['status'].upper()}] File: {result['file']}\n{result.get('msg', '')}\n")
        # raise RuntimeError(f"Pruning failed for {result['file']}: {result.get('msg', '')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the neuron tracing pipeline.")
    parser.add_argument(
        "--mode",
        choices=("neuron", "stage-batch"),
        default=PIPELINE.get("mode", "stage-batch"),
        help="neuron: keep the original one-neuron-at-a-time order; stage-batch: pipeline batches across stages.",
    )
    parser.add_argument(
        "--stage-max-tasks",
        type=int,
        default=int(PIPELINE.get("stage_max_tasks", 1)),
        help="Maximum CPU/downstream tasks to run concurrently in stage-batch mode.",
    )
    parser.add_argument(
        "--gpu-max-tasks",
        type=int,
        default=int(PIPELINE.get("gpu_max_tasks", 1)),
        help="Maximum nnUNet GPU batch processes to run concurrently in stage-batch mode.",
    )
    parser.add_argument(
        "--infer-batch-size",
        type=int,
        default=int(PIPELINE.get("infer_batch_size", PIPELINE.get("stage_max_tasks", 1))),
        help="Number of neurons placed into one nnUNet batch input directory.",
    )
    return parser.parse_args()


def load_meta_if_needed() -> pd.DataFrame | None:
    if ENABLED_STAGES.get("gcut", True) or ENABLED_STAGES.get("prune", True):
        print(PATHS["meta_file"])
        return pd.read_csv(
            PATHS["meta_file"],
            index_col="cell_id",
            low_memory=False,
            encoding="latin1",
        )
    return None


def build_stages(meta_df: pd.DataFrame | None) -> list[StageEntry]:
    stages: list[StageEntry] = [
        ("rescale", run_rescale),
        ("soma_crop", run_soma_crop),
        ("soma_infer", run_soma_infer),
        ("neurite_infer", run_neurite_infer),
        ("merge", run_merge),
        ("app2_trace", run_app2_trace),
        ("gcut", lambda neuron_id: run_gcut(neuron_id, meta_df)),
        ("prune", lambda neuron_id: run_prune(neuron_id, meta_df)),
    ]
    return [(stage_name, stage_func) for stage_name, stage_func in stages if ENABLED_STAGES.get(stage_name, True)]


def run_stage_for_neurons(stage_name: str, stage_func: StageFunc, neuron_ids: Sequence[int | str], max_tasks: int) -> None:
    if not neuron_ids:
        return
    workers = min(max_tasks, len(neuron_ids))
    print(f"\n--- {stage_name}: {len(neuron_ids)} neurons, {workers} workers ---")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_neuron = {executor.submit(stage_func, neuron_id): neuron_id for neuron_id in neuron_ids}
        for future in concurrent.futures.as_completed(future_to_neuron):
            neuron_id = future_to_neuron[future]
            try:
                future.result()
            except Exception as exc:
                for pending in future_to_neuron:
                    pending.cancel()
                raise RuntimeError(f"Stage {stage_name} failed for neuron {neuron_id}") from exc


def run_neuron_mode(stages: Sequence[StageEntry], neuron_ids: Sequence[int | str]) -> None:
    for neuron_id in neuron_ids:
        print(f"\n========== Processing neuron {neuron_id} ==========")
        for stage_name, stage_func in stages:
            print(f"\n--- {stage_name} ---")
            result = stage_func(neuron_id)
            if stage_name == "gcut" and result is False:
                print(f"[skip] neuron {neuron_id}: G-Cut skipped, pruning will not run.")
                break
    print("\nAll requested neurons have been processed sequentially.")


SOMA_PREFILL_STAGES = ("rescale", "soma_crop", "soma_infer")
GPU_STAGES = {"soma_infer", "neurite_infer"}
DOWNSTREAM_AFTER_NEURITE_STAGES = ("merge", "app2_trace", "gcut", "prune")


def soma_outputs_ready(neuron_id: int | str) -> bool:
    stem = image_stem(neuron_id)
    return (
        (Path(PATHS["soma_crop_dir"]) / f"{stem}.json").exists()
        and (Path(PATHS["soma_seg_dir"]) / f"{stem}.tif").exists()
    )


def neurite_output_ready(neuron_id: int | str) -> bool:
    return (Path(PATHS["neurite_seg_dir"]) / f"{image_stem(neuron_id)}.tif").exists()


def swc_has_nodes(swc_path: Path) -> bool:
    if not swc_path.exists() or swc_path.stat().st_size == 0:
        return False
    with open(swc_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if len(line.split()) >= 7:
                return True
    return False


def log_pipeline_skip(log_path: Path, neuron_id: int | str, reason: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{image_stem(neuron_id)}] SKIP: {reason}\n")


def run_downstream_after_neurite(stage_funcs: dict[str, StageFunc], neuron_id: int | str) -> None:
    if "merge" in stage_funcs:
        if not soma_outputs_ready(neuron_id):
            raise FileNotFoundError(f"Soma crop json or soma mask is missing for neuron {neuron_id}; merge is not safe to run.")
        if "neurite_infer" not in stage_funcs and not neurite_output_ready(neuron_id):
            raise FileNotFoundError(f"Neurite segmentation is missing for neuron {neuron_id}; merge is not safe to run.")

    for stage_name in DOWNSTREAM_AFTER_NEURITE_STAGES:
        stage_func = stage_funcs.get(stage_name)
        if stage_func is None:
            continue
        print(f"\n--- {stage_name}: neuron {neuron_id} ---")
        result = stage_func(neuron_id)
        if stage_name == "gcut" and result is False:
            print(f"[skip] neuron {neuron_id}: G-Cut skipped, pruning will not run.")
            break


def run_neurite_and_downstream(
    stage_funcs: dict[str, StageFunc],
    neuron_ids: Sequence[int | str],
    gpu_max_tasks: int,
    downstream_max_tasks: int,
    infer_batch_size: int,
) -> None:
    downstream_enabled = any(stage_name in stage_funcs for stage_name in DOWNSTREAM_AFTER_NEURITE_STAGES)
    downstream_futures: list[concurrent.futures.Future[None]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=downstream_max_tasks) as downstream_executor:
        def submit_downstream(neuron_id: int | str) -> None:
            if downstream_enabled:
                downstream_futures.append(downstream_executor.submit(run_downstream_after_neurite, stage_funcs, neuron_id))

        if "neurite_infer" in stage_funcs:
            print(
                f"\n--- neurite_infer: {len(neuron_ids)} neurons, batch size {infer_batch_size}, "
                f"{min(gpu_max_tasks, len(neuron_ids))} GPU batch workers ---"
            )
            batches = list(enumerate(batched(neuron_ids, infer_batch_size)))
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(gpu_max_tasks, len(batches))) as gpu_executor:
                future_to_batch = {
                    gpu_executor.submit(run_neurite_infer_batch_with_watch, batch, infer_batch_size, submit_downstream): batch
                    for _batch_index, batch in batches
                }
                for future in concurrent.futures.as_completed(future_to_batch):
                    batch = future_to_batch[future]
                    try:
                        future.result()
                    except Exception as exc:
                        for pending in future_to_batch:
                            pending.cancel()
                        raise RuntimeError(f"Stage neurite_infer failed for batch {batch}") from exc
        elif downstream_enabled:
            for neuron_id in neuron_ids:
                submit_downstream(neuron_id)

        for future in concurrent.futures.as_completed(downstream_futures):
            future.result()


def run_stage_batch_mode(
    stages: Sequence[StageEntry],
    neuron_ids: Sequence[int | str],
    stage_max_tasks: int,
    gpu_max_tasks: int,
    infer_batch_size: int,
) -> None:
    if stage_max_tasks < 1:
        raise ValueError("--stage-max-tasks must be >= 1")
    if gpu_max_tasks < 1:
        raise ValueError("--gpu-max-tasks must be >= 1")
    if infer_batch_size < 1:
        raise ValueError("--infer-batch-size must be >= 1")
    if not stages:
        print("\nNo enabled stages to run.")
        return
    if not neuron_ids:
        print("\nNo neurons requested.")
        return

    stage_funcs = dict(stages)
    print("\nStage-batch resource plan:")
    print(f"  CPU/downstream max tasks: {stage_max_tasks}")
    print(f"  GPU batch processes: {gpu_max_tasks}")
    print(f"  nnUNet batch size: {infer_batch_size}")
    print("  order: soma prefill first with per-neuron soma inference, then neurite batch inference; each stable neurite output triggers downstream")

    for stage_name in ("rescale", "soma_crop"):
        stage_func = stage_funcs.get(stage_name)
        if stage_func is None:
            continue
        run_stage_for_neurons(stage_name, stage_func, neuron_ids, stage_max_tasks)

    if "soma_infer" in stage_funcs:
        run_stage_for_neurons("soma_infer", stage_funcs["soma_infer"], neuron_ids, gpu_max_tasks)

    run_neurite_and_downstream(stage_funcs, neuron_ids, gpu_max_tasks, stage_max_tasks, infer_batch_size)

    print("\nAll requested neurons have been processed in stage-batch mode.")


def main() -> None:
    args = parse_args()
    ensure_dirs()
    meta_df = load_meta_if_needed()
    stages = build_stages(meta_df)
    if args.mode == "neuron":
        run_neuron_mode(stages, NEURON_IDS)
    else:
        run_stage_batch_mode(stages, NEURON_IDS, args.stage_max_tasks, args.gpu_max_tasks, args.infer_batch_size)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted by user")
