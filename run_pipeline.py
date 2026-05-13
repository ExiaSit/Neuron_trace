"""Sequential one-neuron-at-a-time pipeline orchestrator.

This file intentionally calls the existing processing functions and command-line
models instead of changing image-processing logic in the stage scripts.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

from pipeline_config import ENABLED_STAGES, GCUT, NEURON_IDS, NNUNET, PATHS, PRUNING, SOMA, VISUALIZATION

ROOT = Path(__file__).resolve().parent


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


def save_volume_mip(volume_path: Path, stage_name: str, neuron_id: int | str) -> None:
    """Save a report-quality XYZ MIP summary for a stage output volume."""
    if not VISUALIZATION.get("save_stage_mips", True):
        return
    if not volume_path.exists():
        print(f"[mip skip] missing volume for {stage_name}: {volume_path}")
        return

    mip_dir = Path(PATHS["stage_mip_dir"]) / str(neuron_id)
    mip_dir.mkdir(parents=True, exist_ok=True)
    img = tifffile.imread(volume_path)
    if img.ndim == 2:
        projections = [("XY", img)]
    elif img.ndim == 3:
        projections = [
            ("XY", np.max(img, axis=0)),
            ("XZ", np.max(img, axis=1)),
            ("YZ", np.max(img, axis=2)),
        ]
    else:
        print(f"[mip skip] unsupported ndim={img.ndim}: {volume_path}")
        return

    fig, axes = plt.subplots(1, len(projections), figsize=(6 * len(projections), 6), dpi=VISUALIZATION.get("dpi", 200))
    if len(projections) == 1:
        axes = [axes]
    for ax, (title, mip) in zip(axes, projections):
        vmax = np.percentile(mip, 99.5) if np.max(mip) > 0 else 1
        if vmax <= 0:
            vmax = np.max(mip) if np.max(mip) > 0 else 1
        ax.imshow(mip, cmap="gray", vmin=0, vmax=vmax)
        ax.set_title(f"{stage_name} {title}")
        ax.axis("off")
    fig.suptitle(f"Neuron {neuron_id} - {stage_name}")
    fig.tight_layout()
    fig.savefig(mip_dir / f"{stage_name}.png")
    plt.close(fig)


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
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    temp_input = temp_dir / f"{output_stem}_0000.tif"
    try:
        os.symlink(source_tif, temp_input)
    except OSError:
        shutil.copy2(source_tif, temp_input)

    set_nnunet_env(results_dir)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(NNUNET["gpu_id"])
    cmd = [
        "nnUNetv2_predict",
        "-i", str(temp_dir),
        "-o", str(output_dir),
        "-d", str(dataset_id),
        "-c", str(NNUNET["configuration"]),
        "-f", str(NNUNET["fold"]),
        "-device", str(NNUNET["device"]),
    ]
    print("[run] " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    shutil.rmtree(temp_dir, ignore_errors=True)


def run_rescale(neuron_id: int | str) -> None:
    mod = load_stage_module("stage_rescale", "1_rescale_cpu.py")
    mod.test_dir = str(PATHS["image_1um_dir"])
    Path(mod.test_dir).mkdir(parents=True, exist_ok=True)
    mod.prepare_nnunet_file(neuron_id)
    save_volume_mip(Path(PATHS["image_1um_dir"]) / f"{image_stem(neuron_id)}_0000.tif", "01_rescale_1um", neuron_id)


def run_soma_crop(neuron_id: int | str) -> None:
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
    save_volume_mip(Path(PATHS["soma_crop_dir"]) / f"{image_stem(neuron_id)}_0000.tif", "02_soma_crop", neuron_id)


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
    save_volume_mip(Path(PATHS["soma_seg_dir"]) / f"{stem}.tif", "03_soma_seg", neuron_id)


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
    save_volume_mip(Path(PATHS["neurite_seg_dir"]) / f"{stem}.tif", "04_neurite_seg", neuron_id)


def run_merge(neuron_id: int | str) -> None:
    mod = load_stage_module("stage_concat", "3.concat_neurite_soma.py")
    stem = image_stem(neuron_id)
    tmp_json_dir = Path(PATHS["merge_temp_json_dir"])
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
    save_volume_mip(Path(PATHS["merged_mask_dir"]) / f"{stem}.tif", "05_merged_mask", neuron_id)


def run_app2_trace(neuron_id: int | str) -> None:
    mod = load_stage_module("stage_app2", "4_pre_trace_app2.py")
    mod.V3D_PATH = str(PATHS["vaa3d_path"])
    for path_key in ("trace_output_dir", "trace_swc_dir", "trace_marker_dir", "trace_vis_dir"):
        Path(PATHS[path_key]).mkdir(parents=True, exist_ok=True)
    stem = image_stem(neuron_id)
    result = mod.process_single_file(
        str(Path(PATHS["merged_mask_dir"]) / f"{stem}.tif"),
        str(Path(PATHS["trace_swc_dir"]) / f"{stem}.swc"),
        str(PATHS["trace_marker_dir"]),
        str(PATHS["trace_vis_dir"]),
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

    app2_mip = Path(PATHS["trace_vis_dir"]) / f"{neuron_id}_mip.png"
    if app2_mip.exists() and VISUALIZATION.get("save_stage_mips", True):
        neuron_mip_dir = Path(PATHS["stage_mip_dir"]) / str(neuron_id)
        neuron_mip_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(app2_mip, neuron_mip_dir / "06_app2_trace.png")


def run_gcut(neuron_id: int | str, meta_df: pd.DataFrame) -> None:
    mod = load_stage_module("stage_gcut", "5.gcut_pipeline.py")
    Path(PATHS["gcut_output_dir"]).mkdir(parents=True, exist_ok=True)
    stem = image_stem(neuron_id)
    args = (
        Path(PATHS["image_1um_dir"]) / f"{stem}_0000.tif",
        Path(PATHS["merged_mask_dir"]) / f"{stem}.tif",
        Path(PATHS["trace_swc_dir"]) / f"{stem}.swc",
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
    prepare_gcut_swc_for_pruning(neuron_id)


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
        raise RuntimeError(f"Pruning failed for {result['file']}: {result.get('msg', '')}")


def main() -> None:
    ensure_dirs()
    meta_df = None
    if ENABLED_STAGES.get("gcut") or ENABLED_STAGES.get("prune"):
        meta_df = pd.read_csv(PATHS["meta_file"], index_col="cell_id", low_memory=False)

    stages = [
        ("rescale", run_rescale),
        ("soma_crop", run_soma_crop),
        ("soma_infer", run_soma_infer),
        ("neurite_infer", run_neurite_infer),
        ("merge", run_merge),
        ("app2_trace", run_app2_trace),
    ]

    for neuron_id in NEURON_IDS:
        print(f"\n========== Processing neuron {neuron_id} ==========")
        for stage_name, stage_func in stages:
            if ENABLED_STAGES.get(stage_name, True):
                print(f"\n--- {stage_name} ---")
                stage_func(neuron_id)
        if ENABLED_STAGES.get("gcut", True):
            print("\n--- gcut ---")
            run_gcut(neuron_id, meta_df)
        if ENABLED_STAGES.get("prune", True):
            print("\n--- prune ---")
            run_prune(neuron_id, meta_df)

    print("\nAll requested neurons have been processed sequentially.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted by user")
