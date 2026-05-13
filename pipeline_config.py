"""
Central path and runtime configuration for the sequential neuron tracing pipeline.

Edit this file only when moving data, models, logs, or external tools. The
orchestrator in ``run_pipeline.py`` reads all input/output locations from here.
"""
from pathlib import Path

# Neurons are processed one-by-one in this order. Use integers or strings.
NEURON_IDS = [20000]

# Base dataset directory. The paths below may be absolute or relative to this.
BASE_DIR = Path("/data/disk/C6.0/app_test")

PATHS = {
    # 1_rescale_cpu.py output and downstream 1um image input.
    "image_1um_dir": BASE_DIR / "img",

    # 2_soma_crop_seg.py crop output and soma nnUNet output.
    "soma_crop_dir": BASE_DIR / "soma_img",
    "soma_seg_dir": BASE_DIR / "soma_seg",
    "soma_temp_dir": BASE_DIR / "soma_temp_single_input",
    "soma_skipped_log": BASE_DIR / "skipped_neurons.txt",

    # 2_neurite_seg.py nnUNet output.
    "neurite_seg_dir": BASE_DIR / "1um_neurite_seg_0424",
    "neurite_temp_dir": BASE_DIR / "neurite_temp_single_input",

    # 3.concat_neurite_soma.py output.
    "merged_mask_dir": BASE_DIR / "mask",
    "merge_temp_json_dir": BASE_DIR / ".pipeline_tmp" / "merge_json",
    "merge_error_log": BASE_DIR / "merge_failed_log.txt",

    # 4_pre_trace_app2.py output.
    "trace_output_dir": BASE_DIR / "trace_app2",
    "trace_swc_dir": BASE_DIR / "trace_app2" / "down_sampled_swcs_app2",
    "trace_marker_dir": BASE_DIR / "trace_app2" / "markers",
    "trace_vis_dir": BASE_DIR / "trace_app2" / "vis_mip",
    "trace_timeout_log": BASE_DIR / "trace_app2" / "record_timeout_5min.txt",
    "trace_error_log": BASE_DIR / "trace_app2" / "record_failed_5min.txt",
    "trace_cost_log": BASE_DIR / "trace_app2" / "record_cost_time_5min.txt",

    # 5.gcut_pipeline.py output.
    "gcut_output_dir": BASE_DIR / "gcut_output",
    "gcut_error_log": BASE_DIR / "gcut_output" / "error_log.txt",

    # 6_gcut_pruning_copy60228.py output.
    "prune_output_dir": BASE_DIR / "gcut_pruned4",
    "prune_mip_dir": BASE_DIR / "gcut_pruned4" / "pruning_mips",
    "prune_swc_dir": BASE_DIR / "gcut_pruned4" / "pruning_swcs",
    "prune_error_log": BASE_DIR / "gcut_pruned4" / "error_neurons.log",

    # External metadata / binaries.
    "meta_file": Path("/home/pzy/Neuron_Trace/meta_260205.csv"),
    "vaa3d_path": Path("/home/pzy/pzy/Vaa3D-x.1.1.4_Ubuntu/Vaa3D-x"),
}

NNUNET = {
    "soma_dataset_id": "206",
    "neurite_dataset_id": "170",
    "configuration": "3d_fullres",
    "fold": "0",
    "device": "cuda",
    "gpu_id": 0,
    "raw": "Wait_No_Need",
    "preprocessed": "Wait_No_Need",
    "soma_results": "/data/disk/nnUNet_local/nnUNet_results",
    "neurite_results": "/data/disk3/nnUNet_base/nnUNet_results",
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

# Toggle whole stages without changing code. Stages still run sequentially.
ENABLED_STAGES = {
    "rescale": True,
    "soma_crop": True,
    "soma_infer": True,
    "neurite_infer": True,
    "merge": True,
    "app2_trace": True,
    "gcut": True,
    "prune": True,
}
