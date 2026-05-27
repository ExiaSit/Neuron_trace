# Neuron_trace

Neuron tracing pipeline orchestrated by `run_pipeline.py`. Runtime paths, model IDs, external binaries, and concurrency settings live in `pipeline_config.py`.

## Pipeline Flow

Default mode is `stage-batch`:

```bash
python run_pipeline.py --mode stage-batch --stage-max-tasks 20 --gpu-max-tasks 1 --infer-batch-size 20
```

The stage-batch flow is:

1. `rescale` prepares the 1um image input when enabled.
2. `soma_crop` crops the soma block and writes the crop JSON.
3. `soma_infer` runs nnUNet once with the full `soma_crop_dir` as input, after all requested soma crops are prepared. It skips the command when all requested soma masks already exist.
4. `neurite_infer` runs nnUNet in batches over the full 1um images.
5. A watcher polls neurite segmentation outputs; each stable `image_<id>.tif` immediately triggers downstream work for that neuron.
6. `merge` runs after the matching soma crop JSON, soma mask, and neurite mask are all present.
7. `app2_trace` traces the merged mask with Vaa3D APP2.
8. `gcut` selects the target neuron SWC. Empty APP2 SWC files, or SWC files with no valid nodes, are logged and skipped so the pipeline can continue.
9. `prune` writes the final pruned SWC.

Single-neuron sequential mode is still available:

```bash
python run_pipeline.py --mode neuron
```

## Deployment Config

When deploying on a new machine, update `pipeline_config.py` first:

- `NEURON_IDS`: neuron IDs to process.
- `BASE_DIR`: root working directory for generated inputs, masks, SWCs, logs, and temp files.
- `PATHS["meta_file"]`: metadata CSV path.
- `PATHS["vaa3d_path"]`: Vaa3D executable path.
- `PATHS["pylib"]` and `PATHS["TraceFlow"]`: external library paths used by downstream tracing/G-Cut code.
- `NNUNET["soma_dataset_id"]` and `NNUNET["neurite_dataset_id"]`: nnUNet dataset IDs.
- `NNUNET["soma_results"]` and `NNUNET["neurite_results"]`: nnUNet results/model directories.
- `NNUNET["gpu_id"]`, `NNUNET["device"]`, `NNUNET["configuration"]`, and `NNUNET["fold"]`: GPU/model runtime settings.
- `PIPELINE["stage_max_tasks"]`: CPU/downstream concurrency.
- `PIPELINE["gpu_max_tasks"]`: concurrent nnUNet batch processes, usually `1` for one GPU.
- `PIPELINE["infer_batch_size"]`: number of neurons per neurite nnUNet batch input directory; soma inference uses the full soma crop folder directly.
- `ENABLED_STAGES`: turn stages on/off without editing code.

Optional algorithm settings are also in `pipeline_config.py`: `SOMA`, `GCUT`, and `PRUNING`.

The pipeline no longer writes `pipeline_mips`, `trace_app2/vis_mip`, or per-neuron pruning `*_process_panels` folders.
