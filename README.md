# Neuron Trace v0.82

从原始三维神经元图像生成神经元重建 SWC 的处理流程。项目按神经元 ID 读取图像与胞体元数据，依次进行 1 µm 重采样、胞体与神经突分割、mask 合并、APP2 追踪、G-Cut 分离、SWC 重采样和剪枝，同时保存各阶段结果与可视化图。

## 本版修改

以本地 `v0.81` 和 Git 的 `main`（`4862a2f`）为基础，移植 `0.84_test` 当前修改后的 G-Cut。
分支名为 `v0.82`。分割、mask 合并、APP2、SWC 重采样和剪枝继续使用 0.81 的算法。
流水线直接将合并后的 mask 交给 APP2/G-Cut，不包含 mask 补全或重连阶段。

G-Cut 默认使用 APP2 原生三维 GSDT 选择厚核心内的候选峰值，保留 EDT 的 5 µm
厚度门槛与半径评分，并包含胞体证据过滤、6 µm 距离 NMS、三维信号合并、
目标胞体区域归属和保留稀疏 APP2 末梢的预处理修改。
G-Cut 需要的胞体几何测量单独放在 `algorithms/tracing/soma_geometry.py`，只用于分析候选，
不修改或保存输入 mask。其函数从当前版本原样提取。

每个 G-Cut 样本使用独立子进程，以隔离批处理中的候选参数和 GPU 环境设置。
本版的配置环境变量为 `NEURON_V082_CONFIG`，命令行 `--config` 会自动设置它。
新版 G-Cut、GSDT 原生源码和辅助模块都位于本版目录，运行时不依赖 `0.84_test` 目录。

## 环境与依赖

- Python 3.9，推荐使用已配置的 `hb_seg` 环境。
- Python 库：NumPy、pandas、SciPy、scikit-image、tifffile、matplotlib、OpenCV、joblib、tqdm、NetworkX、connected-components-3d、edt、v3dpy。
- nnUNet v2 及其 PyTorch 环境，默认使用 CUDA 推理。
- nnUNet 模型：Dataset206 胞体分割、Dataset169 神经突分割，默认 `3d_fullres`、fold `0`。
- G++（首次编译 APP2 GSDT）；CuPy/CUDA（候选生成配置为 `cuda` 时需要）。
- Vaa3D 3.601 的 APP2 插件，以及用于 SWC 重采样、排序的 Vaa3D-x。
- `xvfb-run`，用于运行 Vaa3D 命令行任务。

运行所需的本地依赖源码按下面的布局准备。前三项外部依赖由 `.gitignore` 排除，单独部署；`app2_gsdt` 源码随本版保存：

```text
vendor/
├── pylib/          # swc_handler、morph_topo 等形态学工具
├── TraceFlow/      # core/io、core/visualization 等模块
├── gcut/           # neuron_segmentation 及 G-Cut 实现和资源
└── app2_gsdt/      # 本版跟踪的 APP2 GSDT 原始头文件与 C ABI 包装
```

程序相对于项目目录定位这前三个依赖目录，可从任意工作目录启动。

## 输入数据

输入包括原始 V3DPBD/V3DRAW 图像与 CSV 元数据表。元数据使用以下字段：

| 字段 | 内容 |
| --- | --- |
| `cell_id` | 神经元数字 ID |
| `v3dpbd_file` | 以 `Cell_Image/` 开头的图像相对路径 |
| `soma_x`、`soma_y`、`soma_z` | 原图中的胞体坐标 |
| `xy_resolution`、`z_resolution` | 原图分辨率，单位 nm/像素 |

例如 `v3dpbd_file` 为 `Cell_Image/47000_47999/47300_47399/47312.v3dpbd`，程序会在配置的 `v3d_root` 下查找 `47000_47999/47300_47399/47312.v3dpbd`。

## 快速开始

在项目目录中准备本机配置：

```bash
conda activate hb_seg
cp config.example.json config.json
```

编辑 `config.json`，填写处理 ID 和资源路径：

| 配置项 | 用途 |
| --- | --- |
| `neuron_ids` | 要处理的 ID，例如 `[47312, 47313]` |
| `v3d_root` | 原始图像的 `Cell_Image` 目录 |
| `metadata_file` | CSV 元数据表 |
| `output_root` | 输出根目录 |
| `vaa3d_app2` | Vaa3D 3.601 可执行文件 |
| `vaa3d_resample` | 用于重采样和排序的 Vaa3D-x 可执行文件 |
| `nnunet.soma_results` | 胞体模型的 nnUNet results 根目录 |
| `nnunet.neurite_results` | 神经突模型的 nnUNet results 根目录 |
| `nnunet.gpu_id` | 使用的 GPU 编号 |
| `nnunet.device` | 推理设备，默认 `cuda` |

相对的数据、模型和输出路径以配置 JSON 所在目录为基准。`config.json` 保存本机路径，不进入 Git；共享配置使用 `config.example.json`。

启动完整流程：

```bash
python run_pipeline.py --config config.json --mode neuron
```

`neuron` 模式逐个 ID 完成所有阶段。批量任务可以使用阶段并行模式：

```bash
python run_pipeline.py --config config.json --mode stage-batch \
  --stage-max-tasks 4 --gpu-max-tasks 1 --infer-batch-size 4
```

`--stage-max-tasks` 控制 CPU 与下游阶段并行数，`--gpu-max-tasks` 控制 GPU 推理进程数，`--infer-batch-size` 控制每批输入的神经元数量。也可以在 JSON 的 `pipeline` 中设置对应的 `stage_max_tasks`、`gpu_max_tasks`、`infer_batch_size` 和 `mode`。

## 各部分功能

| 文件或目录 | 功能与调用方式 |
| --- | --- |
| `run_pipeline.py` | 统一入口，协调全流程、单 ID 与阶段批处理 |
| `pipeline_config.py` | 解析 JSON 配置，组织路径、阶段开关和参数 |
| `1_rescale_cpu.py` | 读取原始 V3D 图像，归一化并生成 1 µm TIFF |
| `2_soma_crop_seg.py` | 根据胞体坐标裁剪区域，准备胞体分割输入 |
| `2_neurite_seg.py` | 神经突分割批处理脚本；统一入口通过 `run_neurite_infer` 调用同一模型 |
| `3.concat_neurite_soma.py` | 将胞体分割放回全图，与神经突 mask 合并 |
| `4_pre_trace_app2.py` | 生成胞体 marker、调用 APP2、将 SWC 转回图像坐标并生成 MIP |
| `5.gcut_pipeline.py` | 生成胞体候选点、执行 G-Cut、选择目标神经元并生成分离过程图 |
| `5.5_resample_app2_swc.py` | 对 G-Cut 目标 SWC 重采样和排序 |
| `6_gcut_pruning_no_radius_jump_no_long_edge_copy.py` | 对目标 SWC 剪枝，输出最终结构及剪枝过程总图 |
| `neuroutils/config/` | 图像与元数据工具的配置 |
| `neuroutils/image/` | V3D/TIFF 读写、裁剪边界和图像预处理 |
| `neuroutils/meta/` | 元数据读取、ID 与文件路径映射 |
| `tests/` | 运行接口、坐标转换、路径解析的回归检查 |

日常使用通过统一入口控制阶段，配置和路径由入口传递给各阶段函数。

## 选择阶段与参数

所有阶段默认开启。通过 JSON 的 `enabled_stages` 可以控制以下阶段：

```json
{
  "enabled_stages": {
    "rescale": true,
    "soma_crop": true,
    "soma_infer": true,
    "neurite_infer": true,
    "merge": true,
    "app2_trace": true,
    "gcut": true,
    "gcut_resample": true,
    "prune": true
  }
}
```

只运行部分阶段时，先在同一输出目录准备好它们需要的上游产物。

JSON 的 `soma` 可配置 `target_block_size` 和 `save_mip_visualization`；`pruning` 可配置剪枝参数。
`gcut` 配置如下，默认使用 GSDT，且只在大体积 EDT/滤波中启用 GPU：

```json
{
  "gcut": {
    "target_percentile": 55.0,
    "gsdt_threshold_x": 5.0,
    "candidate_method": "app2_gsdt",
    "candidate_backend": "cuda",
    "candidate_evidence": {"enabled": true}
  }
}
```

GSDT 本身在原生 CPU 中运行；无 CUDA 时可将 `candidate_backend` 改为 `cpu`。
`candidate_method: "edt"` 只用于显式对照。`gsdt_threshold_x` 是兼容旧配置的键名，
实际仍表示几何 EDT 的厚度门槛，单位 µm。各项默认值见 `pipeline_config.py`。

## 输出目录

```text
output_root/
├── img/                                 # 1 µm 原图
├── neuron_metas/                         # 按 ID 缓存的元数据
├── soma_img/                            # 胞体裁剪图及边界 JSON
├── soma_seg/                            # 胞体预测 mask
├── soma_img_v081_zxy/                    # G-Cut/剪枝使用的裁剪坐标记录
├── 1um_neurite_seg/                      # 神经突预测 mask
├── mask/                                # 合并后的 mask
├── trace_app2/
│   ├── down_sampled_swcs_app2/           # APP2 SWC
│   ├── markers/                         # 胞体 marker
│   └── vis_mip/                         # APP2 叠加图
├── gcut_output/
│   ├── selected_for_pruning/            # G-Cut 目标 SWC
│   └── selected_resampled_step1/        # 重采样、排序后的目标 SWC
└── gcut_pruned/
    ├── pruning_swcs/                    # 最终 SWC
    └── pruning_mips/                    # 剪枝过程总图与原因记录
```

G-Cut 分离过程总图和候选审计保存在 `gcut_output/<id>/`：

- `overview.png`、`gcut_target.swc`：总览图及目标 SWC。
- `before_evidence_points.swc`、`rejected_candidates.swc`：证据过滤前及被拒绝的点。
- `pre_nms_points.swc`、`distance_nms_only_points.swc`、`post_nms_points.swc`：三个候选阶段。
- `candidate-generation.json`、`candidate-evidence.json`、`nms-audit.json`、`soma-merge-audit.json`：候选依据。
- `target-soma-region.tif`：用于候选合并的目标胞体内部区域，与原图同坐标。
- `input_mask.tif`、`original_image.tif`：输入快捷链接。
- `worker.log`、`worker-spec.json`、`worker-result.json`、`gcut-timing.json`：运行记录。

候选 SWC 使用原图 XYZ 坐标，独立点的显示半径为 2 µm。控制台输出 G-Cut 各子阶段耗时。阶段之间使用 `image_<id>` 文件名对应同一神经元，图像通道输入带 `_0000` 后缀。

## 坐标与续跑

图像数组使用 Z/Y/X 顺序，SWC 使用 X/Y/Z 坐标。APP2 marker 与输出 SWC 的坐标转换由追踪模块处理；裁剪边界分别按合并阶段和 G-Cut/剪枝阶段需要的顺序提供。可视化采用图像坐标。

再次使用同一配置运行时，流程按阶段检查已有产物并跳过已完成部分。输入、模型或处理参数发生变化时，使用新的 `output_root` 生成对应结果。

## 回归检查

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

检查内容包括裁剪坐标顺序、APP2 坐标转换、剪枝函数传参、失败状态传播及与当前工作目录无关的路径解析。


42 项相关检查已通过；15214、107046、105612、16963 的移植对照中，候选坐标/分数、
过滤/NMS/合并审计、所有 G-Cut 分离 SWC 和胞体区域均与相同输入下的当前版本一致。
另以 15214 的原始 mask 实跑 APP2 → G-Cut → 重采样 → 0.81 剪枝，完成全部阶段。
详细范围见 `VALIDATION.md`。移植来源和哈希见 `PORT_PROVENANCE.json`。

## 固定 200 例的历史版本比较

比较与统计入口为 `tools/compare_reconstructions.py`，配置为 `comparison/original200.json`。
评价方法、固定样本清单、参数、验证检查及使用说明都随 Git 保存，见 [comparison/README.md](comparison/README.md)。

```bash
python tools/compare_reconstructions.py --config comparison/original200.json
```

2026-10-22 的 200 例已全部尝试，197 例完成重建；共同有效样本 197 例。
完整记录与失败原因见 [comparison/reports/original200-20261022.md](comparison/reports/original200-20261022.md)。
