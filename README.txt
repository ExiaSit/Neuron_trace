Neuron reconstruction v0.83 — 完整流程整理版
整理日期：2026-10-20，Asia/Shanghai

总入口：run_pipeline.py
统一配置：config.json
推荐环境：/home/pzy/anaconda3/envs/hb_seg/bin/python

本目录覆盖 V3D 到最终 SWC 的完整流程。后半段固定为
v0.83-ring-guard-20261016，不引入后续实验算法。
前处理采用 /data/disk/Neuron_trace 当前的线性重采样、
Dataset206 胞体分割、Dataset169 neurite 分割和二值并集流程。
/home/pzy/Neuron_Trace/Neuron_Trace_pipeline 是流程命名的参考。

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

分段运行
--------
只做到初始 mask：run.sh --ids 17254 --stop 6
准备完成后重建：run.sh --ids 17254 --start 7
只重跑前处理区间：run.sh --ids 17254 --start 3 --stop 6

01–06 也支持 python 对应脚本 --config ... --ids ...。
07–13 的单样本调试入口使用 --spec output_root/manifests/ID.json。
07 单独运行时包含六轮基础修复和受约束延伸。
完整批处理和自动断点记录请统一使用 run_pipeline.py。

从已有 img/mask 开始
--------------------
编辑 configs/from_existing_mask.json 的 raw_dir、mask_dir、soma_crop_dir、
soma_seg_dir、metadata_file 和 output_root，然后运行：

  /data/disk/Neuron_tracing_method/v0.83/run.sh \
    --config /data/disk/Neuron_tracing_method/v0.83/configs/from_existing_mask.json \
    --ids 17254

省略 --ids 时，处理所有匹配且稳定的输入。缺失、重复或仍在迁移中的
文件会记录到 skipped-inputs.json。此入口不复制或软链接输入图像到输出目录。

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
