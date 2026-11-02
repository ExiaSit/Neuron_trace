v0.83b 的主要目标（2026-11-01）：在 v0.82 基础上改善 mask 分支断裂，
包括胞体—分支、分支—分支之间的连接。胞体分割作为位置/归属参考，
不以胞体体积补全、形状拟合或整体扩张为优化目标。

短断口补充规则：骨架折回使常规端点缺失时，仅在距已有胞体表面不超过18µm、
局部骨架朝间隙的一侧为空且存在可靠向内支撑时，补充候选（方向上限60度）。
该候选仍须通过完整路径及实际连接前缀的原始三维信号检验：信号支持≥95%，
横截面脊线支持≥95%，连续弱信号不超过1点，无MIP推断信号替代；保留胞体
竞争、第三连通域/桥碰撞、逐胞体上沿及最终连通性检查。只增加窄连接路径。
原有严格端点规则及70µm搜索范围保留；不会将上述短断口条件推广至长间隙。

2026-11-02 双尺度断口补充：粗分支骨架折回时，额外允许≤32µm的候选，
但12/24µm两个邻域均需朝间隙单侧分布：前方骨架≤5%，后方支撑≥35%，
质心方向夹角≤30°、质心距离≥邻域半径1/3、向内深度≥半径2/3。
仍要求完整路径与截短路径≥95%原始信号及横截面支持、不依赖MIP增强。
若分割交叉已使该分支与远处胞体同属一个连通域，仅在满足上述双尺度条件，
且断口距该域所有胞体表面均>max(70µm,3倍缺口长度)时允许补充；近邻胞体
仍禁止合并。此例外只证明局部连接证据，不把连通域视为单个神经元；后续
G-Cut仍负责归属。审计字段remote_component_somas记录各距离，且保留胞体
竞争、第三区域碰撞、无环及最终逐胞体上沿限制。此修改不增加胞体体积。

v0.83b 基于 v0.83a 工作区：禁用mask胞体体积补全。
c：保留胞体证据检测，参考区域与已有前景求交；输出mask逐体素等于b。
f：跳过胞体根部形状生长；输出mask逐体素等于e；从已有c胞体前景导出归属参考。
g：只用分割中目标坐标所在连通域与现有mask的交集作为胞体锚点；不扩张、填洞或写回胞体。
保留a/b/d/e的神经突连接与路径加粗、g的方向延伸和逐胞体上沿约束。
APP2仍可独立使用骨架OR分割目标胞体，该输入不是mask胞体补全。
其余GSDT、剪枝和最终7.75µm规则沿用v0.83a；近期胞体毛刺规则保持撤回。
下面是继承的v0.83a历史说明，以本段及当前实现为准。

2026-10-30 SWC/原图叠加 MIP 的显示范围：
XY 使用所显示 SWC 的外接盒，每侧向外扩 15 µm 并截到图像边界；Z 使用原图
全部层。当前流程原图与 SWC 均为 1 µm 图像坐标，显示裁剪不平移 SWC。
多版本、剪枝前后对照用相关 SWC 的联合外接盒，包含被删除分支；空 SWC 使用
全图。XZ/YZ 必须先裁剪 XY 再投影，避免盒外信号进入侧视 MIP。
统一实现 visualization/swc_view.py；新版本三视图对照使用
visualization/swc_comparison.py（python -m visualization.swc_comparison）。
原始 mask/补全过程等不叠加 SWC 的独立图保留原显示范围。

2026-10-30 d 步补充局部环/分叉遮蔽的末端：
已有胞体连接完成后，检查仍独立片段朝分割胞体的最近骨架端帽；不替换已提取
的端点。局部骨架需单侧延伸至少 8 体素、朝向胞体夹角 <=30°，到紧凑胞体
表面距离仍 <=70 µm。到胞体的完整路径及实际新增前缀都需 >=95% 原始三维
信号支持、>=80% 横截面脊线支持，且不能依赖 MIP 推断增益。路径第一次接触
已归属该胞体的分支时停止；保留已有桥，禁止第三方片段/其他补充桥碰撞，
多个胞体候选仍需 15% 代价优势。每个胞体的上沿限制及后续剪枝保持原规则。
d/report.json 的 hidden_caps 及 accepted 中的完整/截短路径保留验收依据；
对应自动回归 tests/test_hidden_soma_caps.py。案例实验文件保存在仓库外。

Neuron reconstruction v0.83a — 三方向、多厚度、重叠 MIP 引导

2026-10-29 逐胞体上沿限制（第 07 步最终写入前，当前默认）：
  "mask_growth_direction": "per_soma_top"
使用 f 通过三维形状检查并实际保留的完整胞体区域，各自的最小 Z 为上沿。
区域坐标保存在 f/completed-somas.npz；g 的已知目标胞体补全也参与。仅当
目标与一个 f 胞体中心距 <=8 体素、较小区域重叠比例 >=50% 时合并重复记录；
不合并不同 f 核心。未通过胞体证据检查的亮斑不作为归属种子。
在补全候选的 26 邻域前景图上，从每个胞体计算路径距离，禁止穿过背景，
不按空间直线距离分配。最近与次近路径距离差须 >=max(2体素,最近距离×15%)；
否则归属不明确。无法连到已确认胞体或归属不明的新增体素暂不接受。
明确归属的新增体素仅在 Z>=所属胞体上沿时保留，同层横向补全允许；所有原始
体素保持。裁剪后再次核对新增体素是否仍能连接所属胞体，清除失去连接的新增，
迭代到稳定。g/proposal-soma-ownership.tif 中 0 表示背景或归属不明，正数为
报告中的胞体 ID；这是候选前景的归属诊断，不是最终 mask 或生物学真值。
logs/<ID>/continuity.json 的 soma_surface_guard 记录各胞体上沿、各自新增数、
上沿拒绝数、无归属/不明确拒绝数、失效桥接以及路径距离判据。
当前只有可靠识别出的胞体参与；漏检、原始误连都可能影响归属，需要查看对照。
测试：tests/test_per_soma_surface_guard.py；历史单目标和 unrestricted 对照保留。

历史单目标上沿限制（保留用于比较，不再默认）：
  "mask_growth_direction": "below_soma_top"
先按现有 a–g 规则生成补全候选，同时记录 f 中通过三维形状检查的胞体区域，
以及 g 中由目标胞体坐标/辅助胞体分割定位并补全的胞体。f 的匹配核心需距
目标坐标 <=8 体素且范围覆盖该坐标；不使用整个神经突连通区域的最小 Z。
取目标补全胞体的最小 Z 为上沿。最终 mask = 原始 mask + 上沿及更深处的
补全新增部分；上沿以上与原始 mask 完全一致，同层横向补全仍允许。
裁剪后去除仅靠被裁部分连接的新孤岛，但保留全部原始体素，包括原有表面噪声。
若没有可信目标胞体，只保留原始 mask 并记录 no_supported_target_soma，
不猜测分界面。这是每张图以目标胞体为基准的限制，不是组织表面识别，也不
给图内每个其他胞体分配独立分界面。g 阶段图可能显示撤回 a–f 的部分新增。
logs/<ID>/continuity.json 的 soma_surface_guard 记录上沿、来源、撤回数量，
以及在最终 mask 中失效的 b/d 桥接中心线；前面各步的报告描述裁剪前的阶段结果。
设为 "unrestricted" 可做同输入对照。改变规则须使用新的 output_root；
原始 mask_original、修复 mask 和历史结果分别保留。
回归测试：tests/test_soma_surface_guard.py、tests/test_soma_root_shape.py。

本分支基于 v0.83 d48644f。mask 修复的 b、d 两步新增 XY/XZ/YZ 的
全图投影，以及 8/16/32 体素厚度、50% 重叠的滑动投影。
三维种子至少获两个方向支持；保留原 XY 引导，并取有限增强后的并集。
增强仅用于寻路，三维原图信号验收、防胞体误连、碰撞及 ring guard 保持原规则。
没有把 MIP 柱体直接写入 mask，也没有修改 G-Cut 候选算法或后续剪枝。

config.example.json / configs/from_existing_mask.json 中可配置：
  "mask_mip": {"mode":"multiscale", "thicknesses":[8,16,32],
               "overlap":0.5, "min_axes":2}
尺寸单位为输入的体素（标准输入为 1 µm）；stride=floor(厚度×(1-overlap))，
至少为 1。最后一个窗口对齐图像末端；超过轴长的厚度去重为全图投影。
mode="legacy" 可对照原 XY 引导。改变设置后使用新 output_root，不能混用检查点。
logs/<ID>/mask-b-mip.json 和 mask-d-mip.json 保留各方向、尺度、窗口及连接统计。

2026-10-29 胞体根部形状修正（第 07 步的 f 子步骤）：
取消核心上下各 4 层的硬截断和逐层接受/拒绝形成的平面切口。
逐层背景/噪声仍独立测量，但强度差以核心所在层的 80 分位数为下限，
强种子阈值使用该强度差的 50%，外缘使用 35%（原先为逐层 18%）。
沿三维表面法线测量内外亮度下降，整体表面和侧向表面的支持比例均须 >=55%；
逐层 XY 检查仅记录诊断，不再切割结果。核心附近的纹理检查、邻近连通区域
隔离和增长体积上限继续生效。已有 e 步前景完整保留，不对整幅 mask 腐蚀。
f/report.json 的 shape_rule 为 core-referenced-3d-boundary-v1。
旧结果需从原始输入写入新的 output_root 重跑，不能对已补全 mask 叠加此修改。
形状、相邻胞体、细分支和背景的回归测试：tests/test_soma_root_shape.py。

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

2026-10-30 当前短末梢规则
-----------------------
G-Cut 只做胞体收缩和分离，不再执行 7.75 µm 短末梢删除。
顺序为 G-Cut → 1 µm 重采样 → 原有全部剪枝 → 短末梢删除 → 保存结果。
最后一步按三维折线路径长度（包含与连接点之间的边）迭代删除严格小于
7.75 µm 的末梢；保留根、胞体和连接点。恰好 7.75 µm 不删除。
pruning_reasons.json 和 logs/ID/pruning-audit.json 的 final_short_leaf_audit
记录前后节点、删除分支及长度；网页原因筛选为“最终末梢三维路径长度小于 7.75 µm”。
历史结果保持不变；下文 G-Cut 短末梢描述是历史记录。

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

初始 mask 不被覆盖。修复前先独立复制到 output_root/mask_original/17254.tif，
修复结果写入 output_root/mask/17254.tif；清理中间文件时两个版本都保留。
原始副本按文件内容核验，续跑时不会覆盖不同内容；输入与修复输出指向同一文件时拒绝运行。
网页“07 · Mask 修复”分别提供原始 Mask 和补全后 Mask 的 3D TIFF 下载。
省略 --ids 且配置未指定 neuron_ids 时，处理所有匹配且稳定的输入。
缺失、重复或仍在迁移中的文件记录到 output_root/skipped-inputs.json。
minimum_stable_seconds 默认 120；刚写完的文件可能暂时被跳过，稳定后重跑。
此入口仅备份原始 mask，不复制或软链接灰度原图到输出目录。

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

局部端点三方向 MIP 连接（v0.83a）
--------------------------------
在 mask b 步中增加补充候选。对两个独立连通块的邻近端点，沿各自骨架向内
截取最多 12 体素的局部段，取两段和端点共同 bounding box，外扩 6 体素。
默认搜索距离 70 体素，两端都须朝向缺口（夹角不超过 60 度）。
只对尚未有既有连接提案的连通块对应用此补充规则，原有提案优先。

在同一 crop 的 XY/XZ/YZ MIP 中分别寻路，通常要求每个视图均有 >=85% 的信号支持、
最长暗段 <=2 个像素、绕行 <=1.6 倍。投影路径各扩 3 体素，反投影交集作为
三维寻路通道。最终仍用未增强的三维对比度、横截面、暗段、层内支持和拓扑
规则验证。投影通过不等于三维连接通过；沿某轴投影端点重合时会单独标记。

配置 local_pair_mip.enabled 可关闭新增规则以进行同输入对照，其他参数见
config.example.json 的 local_pair_mip。该规则不改变 a 步现有搜索范围或
c-g 的规则。相关源文件：algorithms/mask_repair/local_pair_mip.py。
审计：mask_steps/<id>/b/local-pair-mip.json；保留的网页图片和 JSON 位于
mask_step_mips/<id>_02_local_pairs.*。红线表示通过三维检查的连接路径，
白虚线为投影候选，青/橙线为两侧局部段。图片用于人工判断，不代表生物学真值。

连接加粗 e 步的胞体例外（2026-10-28）
------------------------------------
加粗恢复原规则：距连接路径不超过 6 体素，原图及平滑图都达到
背景 + max(3, 0.45 × (中心信号 - 背景))；保留原来的两体素邻块隔离。
撤回后续新增的最低亮度、信噪比、边界锐度限制，以及胞体内生长放宽。
仅在末尾防误合并检查增加例外：合并组的全部新增跨区域接触都位于
同一个 c 步识别的胞体域内时保留；涉及不同胞体或胞体外接触仍整组回退
本步新增体素。胞体域以 c 步 soma-only 和测量阈值为依据，最多扩 3 体素，
不能只凭 bounding box 放行；缺少胞体证据时保持原有回退行为。
e/report.json 记录 proposed_voxels、rollback_voxels 和 mergers 中的
allowed_same_soma / rolled_back，保留参与连通块及接触位置以供检查。

连接搜索距离（2026-10-29）
------------------------
a 步端点连接、b 步局部三方向 MIP 补充连接、d 步神经突到胞体的端点连接，
搜索距离由上一轮的 100 缩小至 70 体素；在当前 1 µm 等距图像中对应 70 µm。
方向、信号、绕行和拓扑检查保持原规则，无方向小碎片的 18 体素限制不变。
局部 MIP 的 max_gap 配置允许 1..200；当前 config.json 和示例默认均为 70；200 仅作为显式局部 MIP 对照配置保留。

Mask GPU 后端（2026-10-28）
-------------------------
正式入口的 mask_compute_backend 默认 cuda，可设 cpu 做对照；
mask_gpu_min_voxels 默认 262144，小 ROI 保持 CPU，避免传输开销。
a-g 及局部 MIP 共用 utils/gpu_ndimage.py。大数组的高斯滤波、纯距离 EDT、
最大/最小值滤波、卷积和有限轮二值膨胀/腐蚀可用 CuPy。
连通域编号、骨架化、三维寻路、拓扑检查以及返回最近点索引的 EDT 保持 CPU，
避免最近点等距归属改变结果。CuPy 高斯滤波可能有微小浮点差异。
多个样本共用 /tmp/neuron-tracing-cuda-<uid>.lock，GPU 运算串行，CPU 可并行；
每次调用结束归还缓存。显存不足时，该阶段后续计算回退 CPU，并在日志说明。
logs/<id>/<stage>-gpu.json 记录调用次数、运算及传输耗时、锁等待和回退原因。
此后端仅启用于 mask 阶段；不改变下游 APP2/G-Cut/剪枝后端。
15616 固定 b 输入的 c 步 CPU/GPU 对照：三个 mask TIFF 均逐体素一致。
测试：python -m unittest tests.test_mask_gpu（CUDA 不可用时跳过 GPU 测试）。

局部连接的弯曲端点与微小碎片链（2026-10-28）
------------------------------------------
b 步局部 MIP 补充候选采用最长 12 体素骨架段，并检查该段及 8、6 体素前缀。
至少一个尺度的直线度 >=0.65、有效长度 >=4、端点位移 >=3，且方向朝向缺口
（<=60 度），才提名候选；较远的弯折不再直接淘汰端点。
三个投影通常仍各要求 >=85% 支持。仅允许一个视图比所需支持像素数少 1，
且支持率 >=80%、最长暗段 <=2、绕行仍 <=1.6；另外两个必须正常通过。
此例外必须再满足原始三维信号支持 >=90%、横截面脊线支持 >=75%。

仅对通过原始三维检查的局部 MIP 候选，允许路径接触的小碎片共同参与合并：
每块 <=8 体素、空间跨度 <=4，最多 4 块且总量 <=24，整块距路径 <=2.5，
不能落在路径端点之外，且 >=85% 碎片体素有原图对比度和信噪比支持。
胞体、较大侧支、已经连接到无关区域的碎片仍拒绝。只有这些小碎片自身
此前的连接桥可以重叠；其他旧桥保持禁止碰撞。合并仍禁止连接独立胞体。
report.json 保存 chain_fragment_ids、component_reduction、projection_tolerance；
联合合并按实际减少的连通块数更新图和校验，不再把一次连接固定算作减少一块。

排序时优先处理仍符合旧方向/投影/拓扑条件的局部连接，再处理依赖新例外的
候选，避免新短候选提前占用连通块并替换原有有效路径。

剪枝中的三维片状亮信号检查（2026-10-29）
--------------------------------------
原有横截面暗段检查仍保留。新增 sustained_sheet_like_signal 用于中心很亮、
但实际沿宽片状或界面亮信号延伸的路径，不按统一 Z 平面、分支方向或长度
直接删枝。实现位于 algorithms/tracing/sheet_signal.py，在原有信号断裂和
串线剪枝之后运行；输入为 1 µm 图像坐标下的 SWC 与三维原图。

沿非分叉段约每 0.75 µm 采样，在垂直于路径的截面上取 16 个方向。
同一对相反方向上，距中心 4、6、8 µm 的六个点都必须达到中心亮度的
80%；与该方向垂直的轴在距中心 8 µm 处至少一侧低于中心亮度的40%。
中心使用原图 sigma=0.65 平滑值，在六邻域 ±1 µm 内取最大值以容忍偏心，
要求 >=20。这个形态须连续维持 >=6 µm，且采样窗口完全在图像内部。
选择满足上述条件的区段起点后约1 µm对应的SWC节点，删除它及全部后代。
短暂穿过亮区、有局部管状对比度的路径、不满足宽度的细纤维均不因此删除。

胞体中心12 µm内不触发。仅这条新增规则优先用f步骤导出的
completed-somas.npz真实胞体区域加2 µm余量作为保护；该档案缺失时保守回退
原有confirmed soma保护，不推测边界。下游包含受保护胞体节点时也禁止切除。
原有暗段、串线等规则继续使用原来的保护区。来源写入sheet-soma-protection.json，
逐次判定写入pruning-audit.json和pruning_reasons.json的sheet_signal_decisions。
网页可筛选“连续三维路径沿宽片状亮信号延伸，缺少横向局限”。

该证据不是生物学真值，真实的宽扁突起也可能相似。43199已验证指定分支
之外节点及父子关系完全不变；其他案例的变动需结合原图复核，不能把未改变
或与v0.81接近当作准确率证明。诊断结果保存在仓库外，不提交到Git。
测试：python -m unittest tests.test_sheet_signal tests.test_signal_support。

APP2 骨架化实验（默认关闭）
  app2_input_mode 默认 mask；设置 skeleton_soma 可仅对 APP2 输入执行三维 Lee
  骨架化，再加回胞体分割文件中目标坐标所在的连通区域。参考作者代码
  https://github.com/SEU-ALLEN-codebase/neuron_seg_human/blob/main/tracer.py
  这是预处理思路的适配，胞体识别、APP2 参数与后续 G-Cut/剪枝仍使用本流程。
  mask_original 和 mask 均不修改；独立 app2_input 保存 skeleton、soma、
  skeleton_soma 及 soma_measured TIFF，logs/<ID>/app2-input.json 记录体素、连通性、源 hash。
  app2_soma_source 默认 segmentation：按 soma_json 的 bounds_1um_zxy 起点
  将1um胞体分割TIFF（ZYX数组）放回全图，使用实际TIFF尺寸，不再次缩放或翻转。
  仅保留目标坐标所在的26连通区域；目标坐标不在分割胞体内时明确报错，
  不自动选择邻近胞体或退回原图估计。胞体不做扩张、开运算、椭圆拟合，
  也不与补全mask求交：APP2输入严格等于补全mask的骨架 OR 分割的目标胞体。
  因此APP2输入允许包含分割胞体中位于补全mask外的体素，原始与补全mask仍独立保留。
  soma_measured和soma在此模式下完全相同；日志记录来源文件hash、坐标及连通性。
  如有骨架未连接到分割胞体，记录target_skeleton_unreachable，不自动膨胀补桥。
  下述形状处理仅在显式设置 app2_soma_source=measured 时用于历史方法对照。
  该历史模式采用三维球形开运算（app2_soma_opening_radius_um=3，1um体素）。
  从3um尝试，若胞体中心丢失、保留体积不足80%或原来可达的骨架变得不可达，
  依次退回2um、1um、原胞体；设为0可关闭。骨架不改变，不强制填充理想椭球。
  soma_measured保存原测量胞体，soma保存实际加回的平滑胞体；补全mask不变。
  app2_soma_shape默认signal_surface：开运算形成参考核心后，用原图局部背景与
  核心亮度确定40%对比度等值区域，与定向椭球参考的有符号距离场等权融合，
  以1.2um尺度平滑。最终仍限制在补全mask内，并保留中心、80%~115%原胞体
  体积及骨架连通关系；低对比度或检查失败则退回开运算。opening可复现仅
  开运算的方式；app2_soma_opening_radius_um=0关闭全部胞体形状处理。
  骨架化不保证去除已有错误连接或表面毛刺；应检查远端分支是否丢失后再采用。

胞体附近短末端检查（2026-10-30；2026-10-31 已撤回）
  当前流程不再调用 algorithms/tracing/soma_terminals.py。以下为历史实验规则说明，
  仅保留供旧结果审计；soma_terminal_audit.status=disabled，不产生删除。
  原有信号证据、误接检测及最终7.75um长度剪枝继续保留。
  使用实际加回APP2的目标胞体mask；旧任务没有该mask上下文时不启用。
  仅检查从该胞体内部出发、胞体外路径总长严格小于7.75um的末端。
  不将胞体内部连线计入候选长度；按0.25um采样、边中点是否在胞体内计长。
  原图在2/3/4/6um横截面尺度均缺少持续双侧管状支持（支持率均<80%），
  才删除；任何尺度有支持时通常保留。另一个可删除情形为：2/3um均无
  独立细管支持，且外部采样全部距通向远处的保留路径<=2um，连接两路径的
  亮度没有暗谷（至少80%采样的沿线最小亮度>=两端较暗值的80%，两端>=10）。
  完全位于胞体内的无后代末端可直接删除；根、类型1胞体、长外部末端、
  图像边缘证据不足的候选保留，不整体膨胀胞体清除邻域。
  用最终短枝规则的虚拟预览识别被微小侧枝遮蔽的末端；实际长度剪枝仍最后执行。
  决策写入soma_terminal_audit；网页剪枝原因码soma_adjacent_dead_end。
  测试：python -m unittest tests.test_soma_terminals tests.test_final_short_leaves。
