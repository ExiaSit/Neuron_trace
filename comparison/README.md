# 固定样本的重建版本比较

本工具按参考会话 `019f1b48-6d93-7441-89fe-d19646a47d2d` 的原始 200 例比较方式，
重新运行当前 Git 工作目录中的 v0.82，并比较历史 v0.7、v0.8、v0.81、旧 v0.82
和当前 APP2/最终 SWC。原始抽样清单保存在 `cohorts/original200.csv`，不补抽或替换失败编号。

当前 v0.82 指提交 `3fc3341` 移植的 GSDT 流程；旧 v0.82 指参考会话中
2026-10-13 修正 Y 坐标后的结果。它们在图表中使用不同标签。
这是整套流程的历史版本比较，包含原始 mask 和 APP2/剪枝流程的差别，**不能把全部变化归因于 GSDT**。

## 固定已生成的历史 APP2

用户于 2026-10-22 确认的历史 APP2 来源为：

- `/data/disk2/B4.5/trace_app2/down_sampled_swcs_app2`：原始样本中的 75 例。
- `/data/disk/C6.0/trace_app2/down_sampled_swcs_app2`：原始样本中的 104 例。
- `/data/disk3/120_130k/trace_app2/down_sampled_swcs_app2`：用户随后要求补入的其余 21 例。

`cohorts/original200_archived_app2.csv` 固定完整 200 个原始编号、输入路径和 SWC SHA-256，
不补抽或替换编号。首次只确认两个目录时的 179 例清单和当时剩余的 21 例清单分别保存在
`cohorts/original200_confirmed_archived_app2.csv` 和 `cohorts/original200_archived_app2_excluded.csv`，
仅用于记录初批范围；最终 200 例比较没有排除这 21 例。
`archived_app2_provenance.json` 保存源文件哈希、根节点、树结构检查及 v0.7 汇总来源记录。

```bash
/home/pzy/anaconda3/envs/hb_seg/bin/python tools/compare_reconstructions.py \
  --config comparison/original200_archived_app2.json
```

此配置设置 `app2_input_mode=archived_native`，执行历史 SWC 导入 → G-Cut → 重采样 → 剪枝，
**不执行 APP2 重建或 mask 补全**，使用当前工作目录代码。其他配置默认仍重跑 APP2。
导入只将原始 Vaa3D SWC 的 Y 转为 `H-1-Y`，在独立输出副本中操作；不调整 X/Z、
半径、起点、节点、父子关系或多个根节点，也不对现成 SWC 使用 marker 的 `+2` 补偿。
每次运行核对源文件哈希，独立验证会逐节点确认导入关系与总长度一致；续跑拒绝混入其他输入。

完整 200 例配置的 `reuse_from` 指向已完成的 179 例运行目录。程序只在流水线代码、
处理参数及逐例输入哈希、胞体元数据、图像尺寸、裁剪记录一致时复用终态结果。
核心阶段文件独立复制并记录哈希，诊断目录链接到原运行；已有失败作为终态保留，不重试或替换。
`reuse_manifest.json` 保存来源、阶段文件哈希与案例状态。最终验证还将复用案例的全部指标
与旧运行逐项比较，保证扩充样本不会改变原 179 例的数值。
如果从头运行完整 200 例，删除配置中的可选 `reuse_from` 项；初批 179 例配置另存为
`original179_archived_app2.json`。

历史 v0.7 汇总包含多个来源。这里按用户确认的历史 APP2 固定当前 v0.82 的输入，
但未逐例证明它们就是每份旧版最终 SWC 的直接输入，因而不能宣称这是所有历史算法
均使用相同输入的严格消融实验；它可用于观察当前后续流程处理这批旧 APP2 的效果。
表格同时列出 v0.7、v0.8、历史 v0.81、历史 v0.82、固定 APP2 和当前 v0.82 的输出。
主图、配对差值和平均分支长度继续使用所有列共同有效的编号，失败和 PR 不可用案例单列。

## 运行

先准备项目本机 `config.json`（环境配置见项目 README），再编辑本目录的
`original200.json` 中的输入、历史结果和输出路径。
所有版本复用固定 1 µm 原图；本次直接使用原始输入 mask，不做 mask 补全。
胞体 JSON 适配为 0.81 的坐标约定，保留辅助分割的实际数组，不缩放或补齐输入 mask。

```bash
/home/pzy/anaconda3/envs/hb_seg/bin/python tools/compare_reconstructions.py \
  --config comparison/original200.json
```

默认依次执行输入核对、APP2 → G-Cut → 重采样 → 剪枝、统计、绘图及完整性验证。
使用当前工作目录代码运行，记录代码/输入 SHA-256；重建不从独立代码快照启动。
200 例从头尝试，既有原图、原始 mask 和辅助胞体分割作为只读输入，不重新进行 nnUNet 推理。

也可以分阶段运行：

```bash
# 只核对输入和三维尺寸，不运行重建
python tools/compare_reconstructions.py --config comparison/original200.json --check

# 重建；相同代码、配置和输入可从阶段检查点继续
python tools/compare_reconstructions.py --config comparison/original200.json --run

# 统计和绘图；也可与上述重建并行启动，等待各案例结束后逐例分析
python tools/compare_reconstructions.py --config comparison/original200.json --report

# 仅重新绘图，复用完成的数值统计
python tools/compare_reconstructions.py --config comparison/original200.json --render-only

# 从未改变的 SWC 补算结构指标（例如平均分支长度），复用 PR 和逐例图
python tools/compare_reconstructions.py --config comparison/original200.json --refresh-structure

# 独立核对保存的结果
python tools/compare_reconstructions.py --config comparison/original200.json --verify
```

重建默认 6 路调度，同时最多 2 个 G-Cut 进程使用 GPU；统计默认 8 个 CPU 进程。
可通过 `workers`、`gcut_workers`、`analysis_workers` 调整。
算法代码、输入或处理配置改变时使用新的 `output_root`；程序会拒绝混用不匹配的检查点。
不使用旧版 SWC 替代失败重建。失败编号和原因属于比较结果。

## 保留的计算方法

- `tools/reconstruction_comparison/legacy_prec_rec.py`：参考会话实际使用的评价实现，原样保存。
- `tools/reconstruction_comparison/metrics.py`：原脚本的 SWC 读取、长度统计、单例 PR 调用函数。
- `tools/reconstruction_comparison/report.py`：统一评价所有版本、汇总、配对差值、MIP 和网页。
- `tools/reconstruction_comparison/audit.py`：样本、输入、坐标、树结构、图像、链接及历史数值的独立核对。
- `metric_provenance.json`：原评价代码及抽样清单来源和 SHA-256。
- `original200.json`：本次对比的完整路径和并行参数。

修改评价算法时应保存新的算法标识/哈希，不能把不同定义的指标直接放入同一比较表。
历史算法哈希核对可以防止无意修改旧的 PR 定义。

## 统计定义

长度单位为 µm，坐标均在 1 µm 原图中；SWC 为 XYZ，图像数组为 ZYX。
长度为原始最终 SWC 所有有效父子边的三维欧氏距离之和；节点数直接统计文件。
`branches` 沿用旧代码：**末梢点数＋子节点数大于 1 的分叉点数**，根节点符合条件时也计入。
它不是 SWC 边数，也不能单独当作重建质量指标。

**平均分支长度**（`mean_branch_length`，µm）：将根节点或分叉点至下一个分叉点或
末梢点之间的完整路径作为一个分支段，沿三维父子边累加长度，再求段长的算术平均。
每条边恰好归属一个分支段；根节点发出的各段分别计算，多根森林按各棵树统计。
不额外排除胞体范围，不把相邻采样节点之间的一条边当作完整分支。
无边的单节点树段数与均长均记为 0；包含零长度边的分支仍计入段数。
逐例 CSV 同时保存实际分支段数 `branch_segments`，它与历史 `branches` 的定义不同。

汇总表中的 `mean_branch_length` 为**先求每个细胞的平均分支长度，再对有效细胞等权平均**，
不是把不同细胞的所有分支混在一起平均，也不是各版本平均总长度除以平均历史分枝数。
主图、配对差值及 bootstrap 使用与其他指标相同的共同有效样本；原有指标定义保持不变。
`--refresh-structure` 在确认 SWC 的 SHA-256 与缓存一致后补算结构指标，
不重跑重建或 PR；程序会重新生成 CSV、统计图、网页汇总、配对统计并执行独立核对。

PR 先按旧实现以 1 µm 重采样。Precision 为胞体范围之外、同时满足 `SNR > 2`
和局部线性度 `> 0.6` 的节点比例；胞体排除半径沿用根节点半径加 2 µm。
Recall 沿用端点附近方向探测：`known / (known + missed)`，分母为零时取 1。
SNR 对重采样节点求平均。完整数值细节由已保存的原实现确定。

这里的 PR 是**基于原图的历史启发式指标**，不是对人工真值的准确率/召回率。
端点位置改变会改变 Recall 探测位置；有信号也不保证属于目标神经元。
长度、分枝、节点数描述结构规模，不能据此单独宣称准确性提升。

同时输出两套样本口径：

1. `available`：各版本所有指标可计算的案例，样本量可能不同。
2. `paired_all_versions`：所有版本都有有效指标的同一批编号；主图和主要对比使用这套口径。

失败不填零、不替换，也不在保留旧版本完整样本量的同时只删新版失败案例来比较均值。
`paired_deltas.csv` 在共同样本上逐细胞相减，报告均值、中位数、增减例数；
另以细胞为抽样单位做 2000 次 bootstrap，固定种子 `20261022`，给出描述性的 95% 区间。
它不替代供体分层分析或独立人工评估。

## 输出

`output_root` 包含：

- `index.html`：200 例 MIP 图库，依次显示原图和六组 SWC，可下载对应 SWC。
- `summary_metrics.png`：共同样本的平均/总长度、分枝、节点、历史 PR 和 SNR 图。
- `per_cell_metrics.csv`、`summary_metrics.csv`、`paired_deltas.csv`：逐例、汇总和配对变化。
- `missing_or_failed.csv`、`reconstruction_failures.csv`：缺失与失败的完整记录。
- `run_manifest.json`、`manifests/`、`status/`、`logs/`：输入、代码、参数、状态和各阶段日志。
- `validation.json`：完整性及历史指标兼容性验证。
- `reconstruction/`：当前 v0.82 的 APP2、G-Cut、重采样和最终 SWC。

大体积图像、SWC 和运行目录在本机保存。Git 保留方法、固定样本清单、配置、测试和
`reports/` 下的小体积逐例数据、汇总表、配对统计与报告。

## 检查

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m unittest tests.test_reconstruction_comparison -v
```
