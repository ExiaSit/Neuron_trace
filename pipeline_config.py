"""
Central path and runtime configuration for the sequential neuron tracing pipeline.

Edit this file only when moving data, models, logs, or external tools. The
orchestrator in ``run_pipeline.py`` reads all input/output locations from here.
"""
import os
import json
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
CONFIG_FILE = Path(os.environ.get("NEURON_V081_CONFIG", PACKAGE_ROOT / "config.json")).resolve()
with CONFIG_FILE.open(encoding="utf-8") as _f:
    CONFIG = json.load(_f)

def config_path(key, default):
    value = Path(CONFIG.get(key, default)).expanduser()
    return value if value.is_absolute() else (CONFIG_FILE.parent / value).resolve()

# Neurons are processed in this order. Use integers or strings.
# NEURON_IDS = [60160]
NEURON_IDS = [int(n) for n in CONFIG.get("neuron_ids", [])]

PIPELINE = {
    # Default keeps the stage-batch behavior. Override from the command line with:
    # --mode stage-batch --stage-max-tasks 20 --gpu-max-tasks 1 --infer-batch-size 20
    "mode": "stage-batch",
    "stage_max_tasks": 5,
    "gpu_max_tasks": 1,
    "infer_batch_size": 20,
}

# Base dataset directory. The paths below may be absolute or relative to this.
# DATASET_ROOT = Path("/data/disk3/120_130k/")
# DATASET_ROOT = Path("/data/disk2/B4.5/")
# DATASET_ROOT = Path("/data/disk3/recovered_potential_neurons")
DATASET_ROOT = config_path("dataset_root", "/mnt/hndb")

BASE_DIR = config_path("output_root", "./outputs")
# META_FILE = Path("/home/pzy/Neuron_Trace/meta_260526.csv")
META_FILE = config_path("metadata_file", "/data/disk/Neuron_trace/metadata/meta_260804.csv")

# Keep neuroutils metadata and image lookup aligned with this deployment.
os.environ["NEURON_META_INFO_PATH"] = str(META_FILE)
os.environ["NEURON_META_ENCODING"] = "auto"
os.environ["NEURON_META_DIR"] = str(BASE_DIR / "neuron_metas")
os.environ["V3D_IMAGE_ROOT"] = str(config_path("v3d_root", DATASET_ROOT / "Cell_Image"))

PATHS = {
    # 1_rescale_cpu.py output and downstream 1um image input.
    "image_1um_dir": BASE_DIR / "img",

    # 2_soma_crop_seg.py crop output and soma nnUNet output.
    "soma_crop_dir": BASE_DIR / "soma_img",
    "soma_reader_json_dir": BASE_DIR / "soma_img_v081_zxy",
    "soma_seg_dir": BASE_DIR / "soma_seg",
    "soma_temp_dir": BASE_DIR / "soma_temp_single_input",
    "soma_skipped_log": BASE_DIR / "skipped_neurons.txt",

    # 2_neurite_seg.py nnUNet output.
    "neurite_seg_dir": BASE_DIR / "1um_neurite_seg",
    "neurite_temp_dir": BASE_DIR / "neurite_temp_single_input",

    # 3.concat_neurite_soma.py output.
    "merged_mask_dir": BASE_DIR / "mask",
    "merge_temp_json_dir": BASE_DIR / ".pipeline_tmp" / "merge_json",
    "merge_error_log": BASE_DIR / "merge_failed_log.txt",
    "pipeline_skip_log": BASE_DIR / "pipeline_skipped_neurons.log",

    # 4_pre_trace_app2.py output.
    "trace_output_dir": BASE_DIR / "trace_app2",
    "trace_swc_dir": BASE_DIR / "trace_app2" / "down_sampled_swcs_app2",
    # Legacy pre-G-Cut resample output, retained for earlier experiments.
    "trace_resampled_swc_dir": BASE_DIR / "trace_app2" / "resampled_swcs_app2_step1",
    "trace_resample_error_log": BASE_DIR / "trace_app2" / "resample_step1_errors.log",
    "trace_marker_dir": BASE_DIR / "trace_app2" / "markers",
    "trace_vis_dir": BASE_DIR / "trace_app2" / "vis_mip",

    "trace_timeout_log": BASE_DIR / "trace_app2" / "record_timeout_5min.txt",
    "trace_error_log": BASE_DIR / "trace_app2" / "record_failed_5min.txt",
    "trace_cost_log": BASE_DIR / "trace_app2" / "record_cost_time_5min.txt",

    # 5.gcut_pipeline.py output.
    "gcut_output_dir": BASE_DIR / "gcut_output",
    "gcut_error_log": BASE_DIR / "gcut_output" / "error_log.txt",
    "gcut_selected_swc_dir": BASE_DIR / "gcut_output" / "selected_for_pruning",
    # 4.5_resample_app2_swc.py now runs here, after G-Cut and before pruning.
    "gcut_resampled_swc_dir": BASE_DIR / "gcut_output" / "selected_resampled_step1",
    "gcut_resample_error_log": BASE_DIR / "gcut_output" / "resample_step1_errors.log",

    # 6_gcut_pruning_no_radius_jump_no_long_edge_copy.py output.
    "prune_output_dir": BASE_DIR / "gcut_pruned",
    "prune_mip_dir": BASE_DIR / "gcut_pruned" / "pruning_mips",
    "prune_swc_dir": BASE_DIR / "gcut_pruned" / "pruning_swcs",
    "prune_error_log": BASE_DIR / "gcut_pruned" / "error_neurons.log",

    # External metadata / binaries.
    "meta_file": META_FILE,
    "vaa3d_path": config_path("vaa3d_app2", "/home/pzy/Vaa3D_CentOS_64bit_v3.601/bin/vaa3d"),
    "vaa3d_resample_path": config_path("vaa3d_resample", "/home/pzy/pzy/Vaa3D-x.1.1.4_Ubuntu/Vaa3D-x"),
    "pylib": str(PACKAGE_ROOT / "vendor" / "pylib"),
    "TraceFlow":str(PACKAGE_ROOT / "vendor" / "TraceFlow"),
    "gcut": str(PACKAGE_ROOT / "vendor" / "gcut")
}

NNUNET = {
    "soma_dataset_id": "206",
    "neurite_dataset_id": "169",
    "configuration": "3d_fullres",
    "fold": "0",
    "device": "cuda",
    "gpu_id": 0,
    "raw": "Wait_No_Need",
    "preprocessed": "Wait_No_Need",
    "soma_results": "/data/disk/nnUNet_local/nnUNet_results",
    "neurite_results": "/data/disk/nnUNet_local/nnUNet_results",
}

SOMA = {
    "target_block_size": (128, 128, 128),
    "save_mip_visualization": False,
}

GCUT = {
    "target_percentile": 55.0,
    "gsdt_threshold_x": 5.0,
}

PRUNING = {
    "binary_input": True,
    "tgamma": True,
    "debug": True,
    "sphere_zradius": 10,
    "downsample_scale": (1, 1, 1),
    "pre_traced": True,
    "verbose": True,
}

# Toggle whole stages without changing code.
ENABLED_STAGES = {
    "rescale": True,
    "soma_crop": True,
    "soma_infer": True,
    "neurite_infer": True,
    "merge": True,
    "app2_trace": True,
    "gcut": True,
    "gcut_resample": True,
    "prune": True,
}

PIPELINE.update(CONFIG.get("pipeline", {}))
NNUNET.update(CONFIG.get("nnunet", {}))
SOMA.update(CONFIG.get("soma", {}))
GCUT.update(CONFIG.get("gcut", {}))
PRUNING.update(CONFIG.get("pruning", {}))
ENABLED_STAGES.update(CONFIG.get("enabled_stages", {}))

for _key in ("soma_results", "neurite_results"):
    _value = Path(NNUNET[_key]).expanduser()
    NNUNET[_key] = str(_value if _value.is_absolute() else (CONFIG_FILE.parent / _value).resolve())
