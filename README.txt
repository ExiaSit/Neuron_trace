Neuron reconstruction v0.83a — 三方向、多厚度、重叠 MIP 引导

本分支基于 v0.83 d48644f。mask 修复的 b、d 两步新增 XY/XZ/YZ 的
全图投影，以及 8/16/32/64 体素厚度、50% 重叠的滑动投影。
三维种子至少获两个方向支持；保留原 XY 引导，并取有限增强后的并集。
增强仅用于寻路，三维原图信号验收、防胞体误连、碰撞及 ring guard 保持原规则。
没有把 MIP 柱体直接写入 mask，也没有修改 G-Cut 候选算法或后续剪枝。

config.example.json / configs/from_existing_mask.json 中可配置：
  "mask_mip": {"mode":"multiscale", "thicknesses":[8,16,32,64],
               "overlap":0.5, "min_axes":2}
尺寸单位为输入的体素（标准输入为 1 µm）；stride=floor(厚度×(1-overlap))，
至少为 1。最后一个窗口对齐图像末端；超过轴长的厚度去重为全图投影。
mode="legacy" 可对照原 XY 引导。改变设置后使用新 output_root，不能混用检查点。
logs/<ID>/mask-b-mip.json 和 mask-d-mip.json 保留各方向、尺度、窗口及连接统计。

300 例对比工具：tools/compare_mip_repair.py；支持断点继续、--full 全流程对比，
保存逐例 mask 增删、三视图、阶段节点数/长度/平均分支长度，以及新增独立胞体
核心接触检查。没有人工真值，长度增加不代表质量提高。结果与本机配置不进入 Git。
本次验证状态见 docs/V083A_MIP_VALIDATION.txt。
本分支的示例配置默认 validated_baseline_registry=null，直接展示新候选，
不从已缺失的旧基线目录恢复结果。300例实验也使用相同设置。
tools/audit_mip_pairs.py 另行计算基于原图信号的双向分支覆盖率，
将旧分支覆盖不足98%的案例标记供复核；不会用旧SWC替换本次输出。
300例已完成：300对mask、296组有效成对SWC。78组旧分支覆盖不足98%，
85231有新增mask核心连通风险；不能据此认定整体质量提升。
总体数据、失败原因、重复一致性与方法限制见上面的验证文档。

以下为继承的 v0.83 使用说明和历史验证，不能视作 v0.83a 的质量验收。
------------------------------------------------------------------------
Neuron reconstruction v0.83 — 完整流程整理版
整理日期：2026-10-20，Asia/Shanghai

总入口：run_pipeline.py
统一配置：config.json
推荐环境：/home/pzy/anaconda3/envs/hb_seg/bin/python

本目录覆盖 V3D 到最终 SWC 的完整流程。后半段以
v0.83-ring-guard-20261016 为基础；2026-10-23 移植当前 v0.82 的 G-Cut
短末梢处理。当前算法修订为 ring-guard-20261016-short-leaves-20261023。
前处理采用 /data/disk/Neuron_trace 当前的线性重采样、
Dataset206 胞体分割、Dataset169 neurite 分割和二值并集流程。
/home/pzy/Neuron_Trace/Neuron_Trace_pipeline 是流程命名的参考。

2026-10-23 更新
-----------------
G-Cut 胞体收缩后、分离和 1 µm SWC 重采样之前，调用原有
prune_short_leaves(min_leaf_branch_length=2, stable_nodes=soma_ids)。
按节点数迭代删除不足 2 个节点的末梢分支（不计起始分叉点），保护已知胞体。
该门槛不是 2 µm；删除顺序沿用原 G-Cut 实现，每次删除后更新子节点数，
剩余兄弟分支可能与原分叉点组成两节点路径而保留。
日志记录处理前后的节点数。mask 修复、EDT 胞体候选、最终剪枝及历史回归
保护保持现有逻辑；历史回归保护仍可能恢复旧分支或选择旧基线。
算法变更后请使用新的 output_root。2026-10-20 的一致性结论仅对应原基线。
本次验证范围见 docs/SHORT_LEAVES_VALIDATION.txt。

文件按执行顺序命名
------------------
01_v3d_to_tif.py       读取 .v3dpbd/.v3draw，按原规则归一化为 uint8 TIFF
02_resample_1um.py     按 metadata 的 XY/Z 分辨率重采样到 1 µm
03_crop_soma.py        裁剪胞体区域，补齐尺寸并保存原图中的位置
04_segment_soma.py     Dataset206 胞体分割
05_segment_neurite.py  Dataset169 neurite 分割
06_merge_masks.py     将胞体预测放回原图，与 neurite mask 合并
07_reconnect_mask.py   骨架/方向分析、逐层信号验证、胞体及根部补全、
                      受约束延伸、ring guard 和弯曲根部补全
08_trace_app2.py       APP2；局部临时目录、正确胞体 marker 和坐标转换
09_separate_gcut.py    候选胞体筛选、G-Cut 分离、保存目标 neuron
10_resample_swc.py     G-Cut 后按 1 µm 步长重采样和排序
11_prune_swc.py        信号/方向/形态剪枝、胞体保护及根部重连
12_guard_regression.py 经输入内容核验的历史回归保护
13_compare_results.py  原图/初始 mask/新 mask/v0.81/APP2/最终 SWC 比较

目录职责
--------
algorithms/preprocessing.py  重采样与裁剪的基础算法
algorithms/mask_repair/      第 07 步使用的 mask 修复算法
algorithms/fragment_geometry/ 骨架端点、方向与片段几何工具
algorithms/tracing/          APP2、G-Cut、剪枝和三维信号判断的实现
utils/                      调度、输入检查、断点记录、I/O、SWC 工具
visualization/              mask MIP、APP2 MIP、G-Cut 总图、
                            剪枝总图和最终比较图
vendor/                     所需 G-Cut 和 pylib 代码的本地副本
tests/                      原有回归测试及新增前处理测试
tests/baselines/             已验证历史样本登记和原版本验证报告
configs/                    已有 img/mask 的配置示例
docs/                       来源映射、算法说明和本次验证记录

从 V3D 开始运行
---------------
1. 编辑 config.json：
   - metadata_file：元数据 CSV；
   - v3d_root：元数据 v3dpbd_file 对应的 Cell_Image 根目录；
   - neuron_ids：本次 ID，例如 [17254, 16047]；
   - prepared_root：前处理结果目录；
   - output_root：重建结果目录；
   - models：两种模型的位置、dataset、trainer、fold 和 checkpoint；
   - workers / memory_budget_gb / memory_reserve_gb：后半段并行调度；
   - preprocess_workers / inference_batch_size / gpu_id：前处理与分割。

   或者设置 v3d_dir 指向一个具体 V3D 文件目录，按数字 ID 自动匹配；
   它支持递归扫描和 image_ 前缀，重复 ID 会报错。neuron_ids 为空时，
   v3d_dir 中的 ID 可全部被选择。未设置 v3d_dir 时必须指定 neuron_ids
   或 --ids，避免无意启动整个元数据表。

2. 检查：
   /data/disk/Neuron_tracing_method/v0.83/run.sh --ids 17254 --check

3. 运行：
   /data/disk/Neuron_tracing_method/v0.83/run.sh --ids 17254

   若已在配置中填写 neuron_ids：
   /data/disk/Neuron_tracing_method/v0.83/run.sh

   使用另一份配置：
   /data/disk/Neuron_tracing_method/v0.83/run.sh --config /absolute/path/config.json

可从任意当前工作目录执行。配置中的相对路径始终相对于配置文件。
run.sh 使用 hb_seg Python；如果移动环境，更新 run.sh 与配置中的 Python。

网页实验工作台
--------------
访问 http://10.133.59.18:8766/ ，可在前端输入 ID，通过“浏览服务器”选择输入
文件或文件夹（也可手填路径），选择 01–13
任意连续步骤，检查输入、启动实验、暂停/停止/续跑，并查看日志、图片和 SWC。
这是重建/推理控制台，不包含 nnUNet 模型训练。原 8765 只读图库仍可使用。
首次连接的控制口令保存在：
/data/disk/Neuron_tracing_method/results/v083-console/control-token

网页为每次实验建立独立输出，固定展示本次算法候选，不加载历史基线替换。
从已有 APP2 或 SWC 开始时，按 v0.83 布局导入必需中间结果，不修改源结果。
网页使用独立区间执行器；下面 --start/--stop 的范围限制只针对旧命令行入口。
操作、输入布局和服务管理详见 web_console/README.md。

如何从中间步骤开始执行
----------------------
本节以 17254 为例；多个样本可写成 --ids 17254,16047。
“从已有 1 µm 图像开始”与“开始做 1 µm 重采样”是两个不同入口。
跳过的上游步骤不会自动补做，必须先准备对应的文件。

1. 当前入口支持范围

统一入口 run.sh / run_pipeline.py：
  --start：支持 1、2、3、4、5、6、7，默认 1。
  --stop：支持 1、2、3、4、5、6、13，默认 13。
  --start 7：跳过前处理，执行 mask 修复到最终比较的全部重建步骤。
  --stop 13：执行到完整重建结束。

目前不支持 --start 9 --stop 11，也不支持 --stop 7。
第 08–13 步需要通过单阶段脚本调试，见第 5 小节。
input_mode="prepared" 会跳过全部前处理，直接进入重建；不能用它执行 01–06。

2. 建立独立测试配置

以下变量可在同一个 Bash 终端中复用：

  V083_DIR=/data/disk/Neuron_tracing_method/v0.83
  PY083=/home/pzy/anaconda3/envs/hb_seg/bin/python
  cp "$V083_DIR/config.example.json" "$V083_DIR/config.test.local.json"
  CFG083="$V083_DIR/config.test.local.json"

编辑 config.test.local.json，填写 metadata_file、v3d_root、prepared_root、
output_root、models 和 gpu_id。前处理入口保留 input_mode="v3d"。
配置中的相对路径以配置文件所在目录为基准，移动配置文件时要重新核对。
下文 /你的数据/ 和 /你的结果/ 都是占位路径，使用前必须替换。

同一次任务中断后，重新运行相同命令可续跑，已完成阶段会跳过。
--start/--stop 选择执行区间，不代表强制重算已有结果。
更换重建代码或参数时使用新的 output_root；前处理输入、模型或参数变化时，
也使用新的 prepared_root。不要把旧 status/manifests 直接复制进新实验冒充检查点。

3. 从重采样、已有 1 µm 图像或前处理区间开始

提前准备的文件如下，路径均相对于配置中的 prepared_root：

  开始步骤     必需的直接上游文件
  02 重采样    source_tif/17254.tif：原始分辨率、已归一化的 uint8 灰度图
  03 胞体裁剪  img/17254_0000.tif：已重采样到 1 µm 的灰度图
  04 胞体分割  soma_img/17254_0000.tif 和 soma_img/17254.json
  05 神经突分割 img/17254_0000.tif
  06 合并      neurite_seg/17254.tif、soma_seg/17254.tif、soma_img/17254.json

如果继续执行后续步骤，还要满足那些步骤的输入要求。例如从 05 开始并执行
到 06，除了原图，还必须已有胞体分割和位置 JSON；完整重建始终需要原图。
胞体位置 JSON 应由 03_crop_soma.py 生成，或使用经核对兼容的已有记录。

元数据需包含 cell_id、v3dpbd_file、soma_x/y/z、xy_resolution、z_resolution。
当前代码即使从 02/03 开始，也会读取 v3dpbd_file 来构造来源路径。
没有原始 V3D、只复用 TIFF 时，保留该字段并将 v3d_dir 设为 null；
从 02 起不会要求原始 V3D 文件存在。胞体坐标仍按原图坐标及原始分辨率换算，
不要只把分辨率改成 1000 而不处理坐标。

  # 开始做 1 µm 重采样，只执行这一步
  "$V083_DIR/run.sh" --config "$CFG083" --ids 17254 --start 2 --stop 2

  # 从原始分辨率 TIFF 开始，重采样并完成全部流程
  "$V083_DIR/run.sh" --config "$CFG083" --ids 17254 --start 2

  # 已有 1 µm 图像，从胞体裁剪开始，只做到初始 mask
  "$V083_DIR/run.sh" --config "$CFG083" --ids 17254 --start 3 --stop 6

  # 初始 mask 和辅助输入齐全后，继续全部重建
  "$V083_DIR/run.sh" --config "$CFG083" --ids 17254 --start 7

01–06 也支持单脚本入口，例如只做胞体裁剪：
  "$PY083" "$V083_DIR/03_crop_soma.py" --config "$CFG083" --ids 17254

--check 可检查依赖和部分输入，但不是完整的逐阶段输入验证。
特别是从 02–06 开始时，仍需按上表核对 TIFF/JSON；v3d 模式的 --check
还会检查分割模型，即使本次只准备执行重采样。

4. 从外部已有 img/mask 开始修复

  cp "$V083_DIR/configs/from_existing_mask.json" \
     "$V083_DIR/configs/test-mask.local.json"
  MASK_CFG083="$V083_DIR/configs/test-mask.local.json"

编辑新配置中的以下字段；这里只展示需要修改的部分，其他字段保留模板内容：

  {
    "input_mode": "prepared",
    "raw_dir": "/你的数据/1um_img",
    "mask_dir": "/你的数据/initial_mask",
    "soma_crop_dir": "/你的数据/soma_img",
    "soma_seg_dir": "/你的数据/soma_seg",
    "metadata_file": "/你的数据/metadata.csv",
    "output_root": "/你的结果/v083_mask_test"
  }

每个 ID 必须有 1 µm 灰度原图、同坐标同尺寸的初始 mask、胞体裁剪位置 JSON、
胞体分割 mask 和元数据。soma_crop_dir 在这个入口用于查找位置 JSON。
只有一张 mask 不足以完成修复，算法还需要原图信号和胞体辅助信息。
output_root 必须与上述四个输入目录分开，不能互相包含。

  # 检查匹配和依赖
  "$V083_DIR/run.sh" --config "$MASK_CFG083" --ids 17254 --start 7 --check

  # mask 修复 → APP2 → G-Cut → SWC 重采样 → 剪枝 → 回归保护 → 比较图
  "$V083_DIR/run.sh" --config "$MASK_CFG083" --ids 17254 --start 7

初始 mask 不被覆盖，修复结果写入 output_root/mask/17254.tif。
省略 --ids 且配置未指定 neuron_ids 时，处理所有匹配且稳定的输入。
缺失、重复或仍在迁移中的文件记录到 output_root/skipped-inputs.json。
minimum_stable_seconds 默认 120；刚写完的文件可能暂时被跳过，稳定后重跑。
此入口不复制或软链接输入图像到输出目录。

5. 单独执行 mask 修复、APP2、G-Cut 或剪枝

前提：已有与当前实验匹配的 output_root/manifests/17254.json，以及所需上游结果。
manifest 由正式重建任务生成；--check 不生成 manifest。
manifest 内保存输入、元数据和输出配置，修改外部配置文件不会自动更新它。
请在独立调试结果目录操作，不要与同一目录中的批处理同时运行。

  SPEC083=/你的结果/v083_mask_test/manifests/17254.json

  # 仅修复 mask：包含六轮基础修复、受约束延伸和弯曲根部补全
  "$PY083" "$V083_DIR/07_reconnect_mask.py" --spec "$SPEC083"

  # APP2：需要 output_root/mask/17254.tif 及原图/胞体信息
  "$PY083" "$V083_DIR/08_trace_app2.py" --spec "$SPEC083"

  # G-Cut：还需要 trace_app2/down_sampled_swcs_app2/17254.swc
  "$PY083" "$V083_DIR/09_separate_gcut.py" --spec "$SPEC083"

  # 重采样：需要 gcut_output/selected_for_pruning/17254.swc
  "$PY083" "$V083_DIR/10_resample_swc.py" --spec "$SPEC083"

  # 剪枝：还需要 gcut_output/selected_resampled_step1/17254.swc
  "$PY083" "$V083_DIR/11_prune_swc.py" --spec "$SPEC083"

  # 可选的后续步骤：历史回归保护、最终比较图
  "$PY083" "$V083_DIR/12_guard_regression.py" --spec "$SPEC083"
  "$PY083" "$V083_DIR/13_compare_results.py" --spec "$SPEC083"

上面列出的 SWC 相对路径均位于 manifest 配置的 output_root 下。
07 使用阶段检查点，已完成的修复可能跳过；08–13 单独执行不会更新完整流水线
的阶段完成记录，也不会自动使下游结果失效。重新运行 09 后，应依次运行 10、11，
按实验需要再执行 12、13，避免查看旧的剪枝结果或比较图。
完整批处理和自动断点记录请使用 run_pipeline.py。

当前没有“只生成 manifest”的命令，也没有统一导入外部中间 SWC 的入口。
因此，全新外部数据若只想修复 mask 就停止，或直接从外部 G-Cut/SWC 结果开始，
仍需另行准备正确的 manifest 和上述目录布局，不能只传 --start/--stop 完成。

6. 测试算法效果与查看输出

默认历史回归保护可能恢复受支持的旧分支或选择旧基线。若要在独立实验中
观察当前算法本身的结果，可在启动新任务前将测试配置设为：

  {
    "baseline_root": null,
    "validated_baseline_registry": null
  }

这是配置片段。baseline_root 始终必须为 null；关闭登记表只用于明确选择的
独立实验，不代表该实验经过历史回归保护。已经生成的 manifest 可能仍含
guard_baseline_root，不能只改外部配置就认为它失效，应使用新的 output_root。
保留登记表时，查看 regression_guard/17254.json 的 decision，确认最终结果
是否经过恢复或基线替换。修改登记表设置也应使用新的 output_root。

主要查看 output_root 下的 mask_mips/、gcut_mips/、pruning_mips/ 和 comparisons/；
最终 SWC 在 pruning_swcs/。阶段失败查看 logs/ 和 status/。
前处理失败则查看 prepared_root/preparation-summary.json 和 prepared_root/logs/。

输出和命名
----------
prepared_root/
  source_tif/   V3D 解码后的 TIFF
  img/          1 µm 灰度图
  soma_img/     胞体 crop 和位置 JSON
  soma_seg/     胞体预测
  neurite_seg/  neurite 预测
  initial_mask/ 合并后的初始 mask
  status/ logs/ preparation-summary.json

output_root/
  mask/                 v0.83 新 mask
  trace_app2/            APP2 SWC 和 marker
  gcut_output/           G-Cut 目标 SWC、重采样 SWC
  pruning_swcs/         最终 SWC
  mask_mips/            mask 前后比较
  gcut_mips/            每 ID 一张 G-Cut 总图
  pruning_mips/         每 ID 一张剪枝过程总图
  comparisons/          每 ID 一张最终比较图，目标 soma 用小红点标记
  status/ logs/ manifests/ regression_guard/
  progress.json run-summary.json run-config.json runtime.json

文件使用纯数字 ID，不加 image_ 前缀。nnUNet 输入图像仍使用必需的
ID_0000.tif 通道后缀；mask 和最终 SWC 使用 ID.tif、ID.swc。
推理时的临时软链接只存在于 scratch，退出后清理。

断点续跑与版本保护
------------------
相同命令可以续跑。前处理和重建都有阶段记录，已经完成的结果会跳过。
输入、关键配置或算法变动会拒绝混用旧结果；请使用新的结果目录。
单个输出根目录只能运行一个批任务。前处理与重建分别有文件锁。
改变 worker 数量不等于同步启动所有 worker，内存预算仍约束实际并行数。
baseline_root 必须为 null，禁止复用修复前的环状延伸 mask/APP2。

外部依赖与历史回归数据
--------------------
算法 Python 源码和所需 G-Cut/pylib 辅助源码已复制到此目录；运行时不再
导入 /data/disk/Neuron_trace/method 或 /home/pzy/Neuron_Trace 的源码。

图像数据、metadata、nnUNet 模型权重、Vaa3D 二进制及 hb_seg 环境仍通过
配置引用。requirements.txt 记录本机环境版本；V3D 读取还需要 hb_seg
中已安装的 v3dpy Cython 扩展（Raw/PBD），不适合直接搬到其他 Python ABI。
未复制模型权重、历史图像数据或整个 Python 环境。

历史回归保护有 52 个已验证 ID。它必须同时匹配 metadata 和四项输入
SHA-256，才会使用登记中的基线结果。登记表在：
tests/baselines/validated-registry.json。
新输入不匹配这些记录时，作为新样本处理。

2026-10-20 整理时，登记表指向的原目录
/media/pzy/U-Disk2/human_reconstruction/v0.83-ring-review-20261016-v2/selected
已不存在。本包保留原登记，不把其他结果冒充该基线。
若输入命中该历史记录且基线缺失，会明确报错；应恢复基线或在配置中用
baseline_path_overrides 将旧路径映射到经核实的基线新位置。
本次 17254 整理一致性测试显式使用已部署版本的对应结果进行对比，
其 mask、APP2、G-Cut、重采样及最终 SWC 均与整理前一致。

验证和交接
----------
测试命令：
  cd /data/disk/Neuron_tracing_method/v0.83
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /home/pzy/anaconda3/envs/hb_seg/bin/python -m unittest discover -s tests -v

docs/VALIDATION.txt 说明本次实际执行的检查。
docs/ALGORITHM_RELEASE_NOTES.txt 是原 ring-guard 算法说明。
docs/SOURCE_MAP.json 记录旧文件到新文件的对应关系。

可以直接告诉另一个会话：
“使用 /data/disk/Neuron_tracing_method/v0.83 的完整流程，从 V3D 转 TIFF
开始。先读 README.txt 和 docs/VALIDATION.txt；入口是 run_pipeline.py，
使用 hb_seg，复制 config.json 配置独立输入输出目录。保留现有算法
和回归保护，修改算法时运行 tests/ 并使用独立输出做回归比较。”

逐步诊断图（2026-10-23）
-----------------------
网页结果区可选择“Mask 每步变化”，查看八步修复各自的前后与增删 MIP（XY/XZ/YZ）。
红色是本步新增，黄色是本步删除；每张图带三维体素数与历史快照/复算说明。
最终比较用 previous_app2_dirs 独立查找旧 APP2，并展示来源路径和新旧 APP2 叠加。
G-Cut 短末梢删除以红线和叉号标记，提供删除 ID、前后 SWC 和三视图。
详细使用方法、坐标约定和已有任务补图入口见 web_console/README.md。
