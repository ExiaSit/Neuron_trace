# v0.83 / v0.83a 重建实验工作台

本机 Web 控制台，支持输入数据路径、选择 01–13 任意连续区间、后台执行、
暂停/立即停止/续跑，以及按 neuron 查看阶段日志、图片和 SWC。
这是重建与模型推理入口，不包含 nnUNet 模型训练。

访问：<http://10.133.59.18:8766/>。原只读图库仍在 8765 端口。

## 首次使用

1. 在服务器读取控制口令：

   ```bash
   cat /data/disk/Neuron_tracing_method/results/v083-console/control-token
   ```

2. 打开网页，粘贴口令并连接。口令仅保存在服务器的 0600 文件和浏览器
   HttpOnly / SameSite=Strict 会话 Cookie 中，不写入 URL 或页面源码。
3. 输入实验名称、数字 ID、开始和结束阶段。快捷按钮可选择原始 V3D、
   重采样、已有 1 µm 图像、已有 mask、已有 APP2 或已有重采样 SWC。
4. 点击各输入位置旁的“浏览服务器”，选择输入文件夹或元数据 CSV；也可手填
   绝对路径。路径属于服务器，不是浏览器所在电脑。点击“检查输入”，通过后启动。
5. 左侧选择实验，切换 neuron 或日志阶段。每 3 秒刷新，点击结果图片放大；
   SWC 可下载。也可用“历史结果浏览”查看原图库，如 16984。

## 选择服务器文件或文件夹

- 每个输入路径旁都有“浏览服务器”按钮；先连接工作台再打开。
- 元数据入口选择 CSV 文件，其他入口选择文件夹。文件夹单击进入，CSV 单击选中，
  再点击底部确认按钮填入原表单；取消不会改动原输入。
- 可从 `/data`、`/mnt`、`/media`、`/home/pzy` 开始，通过目录层级、上级按钮或
  地址栏跳转。打开已有 CSV 路径时自动进入它的目录并选中该文件。
- 搜索按当前目录中的名称过滤，不递归搜索整个磁盘；每页最多 100 项。
  目录模式只列文件夹，即使没有子目录也可选择当前目录。
- 列表需要与任务控制相同的认证，仅提供目录/文件名信息，不上传或读取文件内容。
  隐藏项不在列表中展示；无权限、目录不存在和越出数据目录的链接会提示或跳过。
  符号链接显示 ↗，选中后填入真实路径。

每个实验使用独立目录：

```text
/data/disk/Neuron_tracing_method/results/v083-console/jobs/<job-id>/
  job.json             # 不变的请求、输入清单和代码指纹
  config.json          # 本次运行配置
  progress.json        # 当前阶段、完成记录与失败原因
  worker.log           # 总日志
  logs/                # 按 ID、阶段保存的子进程日志
  prepared/            # 新生成的前处理产物
  reconstruction/      # 新生成或导入的重建结果、manifest、图片
```

## 起点与依赖

| 起点 | 主要外部输入 |
| --- | --- |
| 01 | V3D 与元数据 |
| 02 | 原始分辨率、已归一化的 uint8 TIFF 目录 |
| 03 | 1 µm uint8 原图目录 |
| 04 | 胞体 crop TIFF 和位置 JSON |
| 05 | 1 µm 原图 |
| 06 | 神经突分割、胞体分割、位置 JSON |
| 07 | 1 µm 原图、初始 mask、胞体位置 JSON、胞体分割 |
| 08–09 | 上述四项，以及已有重建目录中的修复 mask / APP2 |
| 10 | 已有重建目录中的 G-Cut 目标 SWC |
| 11 | 原图、初始 mask、胞体辅助信息，以及修复 mask、重采样 SWC |
| 12–13 | 已有最终 SWC、原图、mask、胞体辅助信息；比较图还需要 APP2 |

实际必需输入按整个执行区间推导，而不是只看起点。例如 05–06 还需已有
胞体分割和位置 JSON；10–13 除 G-Cut SWC 外还需原图、mask、APP2 等。
页面仅显示本次需要的输入项，后端重新检查，不依赖前端校验。
输入文件支持数字 ID、image_ 前缀、_0000 通道后缀；同目录同类型重复 ID 会拒绝。

已有重建目录按 v0.83 结构读取：

```text
mask_original/<id>.tif  # 原始 mask 的独立副本（新版 v0.83a 保存）
mask/<id>.tif           # 补全后的 mask，下游使用此文件
trace_app2/down_sampled_swcs_app2/<id>.swc
gcut_output/selected_for_pruning/<id>.swc
gcut_output/selected_resampled_step1/<id>.swc
pruning_swcs/<id>.swc
```

这些可写中间产物复制到新实验，不通过硬链接修改源结果。灰度原图、初始
mask 和胞体辅助输入原位读取；v0.83a 修复前另将原始 mask 独立复制至
`reconstruction/mask_original/<id>.tif`，与补全后 mask 同时保留并提供下载。
原始副本与输入内容不一致、或输入和修复输出指向同一文件时拒绝覆盖。
输入的大小、时间戳及元数据变化会使续跑拒绝。
输入在任务执行期间应保持稳定。代码改变后需“复制配置新建”，不能续跑旧代码任务。
目前元数据胞体坐标按原始分辨率换算，不继承模板里的单样本坐标 override。

## 执行和结果含义

- 运行时一个实验、一个样本、一个阶段依次执行，避免同时占用 GPU。
- “完成本步后暂停”不打断当前计算，已完成步骤保留；随后可续跑。
- “立即停止”终止当前任务及子进程，保留检查点；续跑会重做未完成阶段。
- 关闭网页不停止后台进程。服务重启会将未完成任务标记为中断，不自动重启任务。
- 前处理步骤生成 XY MIP 预览，APP2 生成叠加图；07、09、11、13 使用现有
  mask/G-Cut/剪枝/比较图。只生成 SWC 的步骤提供下载链接。
- 网页实验固定不加载历史基线替换。12 记录无基线；最终结果表示本次候选，
  不表示已经通过历史回归验收。旧命令行入口的历史保护逻辑不变。
- 网页执行器保留调试中间文件，磁盘使用量会随实验增加。没有自动删除旧实验。
- 网页的新区间执行器独立于 run_pipeline.py；旧 CLI 仍保留原 --start/--stop 范围。
  不要在同一网页实验目录并行运行旧批处理入口。
- 分割模型与外部程序使用 config.example.json 的受控配置；网页不能传入任意命令
  或替换可执行程序。GPU 编号可选。模型部署路径变化由服务器管理员修改模板。

## 服务管理

```bash
systemctl --user status neuron-v083-console.service
journalctl --user -u neuron-v083-console.service -n 50
systemctl --user restart neuron-v083-console.service
systemctl --user stop neuron-v083-console.service
```

停止或重启服务会终止其计算子进程。安装文件位于
web_console/neuron-v083-console.service，部署到 ~/.config/systemd/user/。
服务已启用用户会话自启动，是否在无登录时启动取决于本机用户 linger 设置。
绑定 0.0.0.0:8766 供内网使用；这是 HTTP 内网服务，不应直接暴露到公网。
默认 MemoryHigh=96G、MemoryMax=128G，超大图像需结合机器资源调整服务限制。

独立启动（不要与同一状态目录的现有服务同时运行）：

```bash
cd /data/disk/Neuron_tracing_method/v0.83a
/home/pzy/anaconda3/envs/hb_seg/bin/python -m web_console.server \
  --host 127.0.0.1 --port 8766 \
  --state-root /data/disk/Neuron_tracing_method/results/v083-console
```

## 验证

```bash
cd /data/disk/Neuron_tracing_method/v0.83a
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/pzy/anaconda3/envs/hb_seg/bin/python -m unittest discover -s tests -v
node --check web_console/static/app.js
```

测试覆盖真实 TIFF 重采样与裁剪、预览生成、区间依赖、输入重复/缺失、
已完成结果跳过、元数据变化拒绝、外部 SWC 复制、立即停止子进程、API 认证、
跨站请求和目录越界拒绝。真实样本与本次验证范围见 VALIDATION.txt。

## 追查每次 mask 修复与短末梢删除

在结果区的“查看结果”选择“Mask 每步变化”“最终比较”或“G-Cut 短末梢删除”。
当前 v0.83a 的 Mask 按 a–g 顺序执行七步：骨架及方向分析、信号重连、胞体补全、
胞体与分支重连、连接厚度恢复、胞体根部补全、受约束延伸。
根部测地线补全 h 已停用，最终 mask 直接使用 g 的输出。f 胞体根部补全仍然保留。
已有历史实验的 h 图片与记录仍可查看，不能视为当前新运行的步骤。
每张图三列为本步前、本步后、本步增删，三行为 XY/XZ/YZ MIP；同一投影使用相同
灰度范围。青色为 mask，红色为本步新增，黄色为本步删除。先在三维计算增删，再
投影；统计单位是体素，不能把 MIP 面积当作体素数。一次重连也可能删除少量体素。
图片逐步出现，PNG 和 JSON 记录保存在 mask_step_mips/，完整 CLI 清理临时 mask 后仍保留。

最终比较增加旧版 APP2、当前 APP2 和两者叠加。配置 previous_app2_dirs 按目录顺序
精确匹配 ID.swc 或 image_ID.swc，不会把 116984 当作 16984；第一份匹配结果优先。
网页可通过“旧版 APP2 目录”选择服务器目录覆盖默认列表，留空使用 config.example.json。
默认列表从 C6.0、B4.5 等 trace_app2/down_sampled_swcs_app2 目录查找。
previous_app2_coordinates 默认为 native，即旧 Vaa3D 原生零起点坐标，仅在绘图内存中
执行 y=height-1-y；已带 app2_native_to_image_zero_based_v2 或
archived_app2_native_to_image_zero_based_v1 标记的文件直接使用图像坐标。
无标记但已是图像坐标的自定义参考文件应在 CLI 配置中设为 image。
comparison_sources/ 记录实际来源、SHA256 和转换方式。未匹配到时明确显示缺失；
旧 APP2 仅供比较，不替换本次输出，与 previous_swc_dirs 独立。

当前 v0.83a 在全部剪枝完成后，按三维路径长度删除严格小于 7.75 µm 的末梢。
长度累加到分叉点或受保护胞体的全部连接边；不删除该连接点。等于阈值保留，
胞体节点与树根受保护。逐条删除后立即更新分叉关系，反复检查。
新结果在“剪枝原因”中筛选“最终末梢三维路径长度小于 7.75 µm”。
pruning_reasons.json 和 logs/ID/pruning-audit.json 的 final_short_leaf_audit
包含前后节点、逐分支长度、删除 ID 和阈值。
历史 G-Cut 红线、叉号及 gcut_audits/ 下载仍展示当时的实际预处理，
不代表新运行的阶段顺序，也不会把旧结果改标为新规则。
默认阈值位于 algorithms/tracing/gcut_adapter.py 的 SHORT_LEAF_MIN_LENGTH_UM。


已有任务补图需要保留 a–f 中间 mask 和 G-Cut injected SWC/soma ID：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/pzy/anaconda3/envs/hb_seg/bin/python -m utils.backfill_diagnostics \
  --spec /absolute/job/reconstruction/manifests/16984.json \
  --previous-app2-dir /data/disk/C6.0/trace_app2/down_sampled_swcs_app2 \
  --previous-app2-dir /data/disk2/B4.5/trace_app2/down_sampled_swcs_app2 \
  --replay-continuation
```

仅在任务停止或完成后使用。--replay-continuation 根据原 continuity.json 判断步骤：新任务仅复算 g，
原记录明确包含根部测地线补全的历史任务才复算 g/h。
并要求最终 mask 与已有结果逐体素一致，否则拒绝发布分步图；图和页面明确标注复算。
不使用该选项则只展示最后两步的合并变化。G-Cut 删除审计通过保留输入复算预处理，
不重跑追踪或分离。补图核对原 mask、APP2、G-Cut、重采样和最终 SWC 的 SHA256；
另存 ID_diagnostics.png，页面优先显示它，原比较图和完成检查点保留。

## 输入路径默认值

打开页面时自动填写 web_console/input_defaults.json 中的常用路径，目前使用
16984 实验所在的原图、mask、胞体位置及分割目录，另提供 V3D 与神经突分割默认目录。
所有默认路径都可编辑或通过“浏览服务器”修改；切换阶段不会重置已填写的值。
“恢复默认路径”仅重置输入路径，不改变 ID、执行区间和 GPU。具体 V3D 子目录、
原始分辨率 TIFF、已有重建结果根目录仍按实验选择；旧 APP2 留空继续按配置目录查找。
管理员可编辑 input_defaults.json，下次刷新网页即生效，无需重启服务。

## v0.83 / v0.83a 与 300 例对照

控制台由 v0.83a 工作树启动，新建实验默认选择 v0.83a，也可选择 v0.81、v0.82、v0.82a 或 v0.83。
v0.83a 的 Mask 修复使用三方向、重叠、多厚度 MIP；该选择不额外改变 G-Cut 算法。
输入检查和执行都使用所选版本自己的代码，通过独立进程隔离 Python 模块。
旧实验未记录版本时按 v0.83 处理；续跑仍校验代码指纹，不会默默切换算法。

已完成的配对实验可由管理员注册为两个只读实验：

```bash
/home/pzy/anaconda3/envs/hb_seg/bin/python -m web_console.import_comparison \
  --state-root /data/disk/Neuron_tracing_method/results/v083-console \
  --cohort-root /home/pzy/neuron_validation/v083a_mip_300_20261025
```

该命令幂等，不复制或修改原结果；reconstruction/ 链接至已有输出。
页面提供三方向 Mask 对照、总体统计、逐例 CSV、节点数、总长度、平均分支长度、
分支覆盖风险和两版最终 SWC。这里的分支覆盖是原图信号启发式指标，不等于人工真值。
原实验失败与严格 SWC 结构检查失败会保留，不能把“只读导入”当作全部通过。
只读实验不能续跑，点击“用当前编号新建实验”仅复制当前样本的输入与版本配置。
注册 300 例的浏览不受新建任务最多 200 例限制；不开放 HTTP 任意路径导入接口。

代码、测试、文档可提交 Git；实际对照结果、实验记录、控制口令与本机服务备份不提交。

## 剪枝原因筛选

打开实验和 neuron，在结果区勾选“剪枝过程 / 原因筛选”，再选择删除原因。
有逐节点记录时，XY / XZ / YZ 的 SWC 投影突出显示该原因的删除部分，并列出节点数及原始父边累计长度（包含连接到保留节点的边，单位 µm）。这些投影不叠加原图，阶段图片仍在下方显示。
筛选仅查看第 11 步已有结果，不改变算法；G-Cut 前的短末梢删除在单独结果组中。

v0.83a 的结果清理保留 `pruning_mips/<id>_pruning_reasons.json`。历史结果若只有
`logs/<id>/pruning-audit.json`，仅支持按原因查看节点计数，不能从汇总推断被删分支的位置或长度；逐节点记录中缺失的原因明确标为未知。
原因可能继承自上游切断点，几何简化删除节点也包含在统计中；累计长度不代表剪枝前后总长度之差。
前端仅加载当前结果列表中的文件，不读取原因 JSON 中记载的绝对路径。

前端筛选检查：`node --test tests/test_web_pruning.js`。

### 按规则筛选 neuron

实验页“当前 neuron”上方提供“按剪枝规则筛选 neuron · 当前实验”。选择规则后，列出在第 11 步记录中该原因删除节点数大于零的样本；可设置最少删除节点数，列表按命中节点数降序排列。点击编号会打开该样本并选中对应的原因展示。
索引只读取该实验登记的 ID，历史只读导入实验也支持；无记录、异常记录与有效零删除记录分别处理。运行中的实验可点击“刷新筛选数据”更新；切换实验后重新加载，不混合版本。
这表示实际记录过的删除原因，不能据此推断被保护或更早删除的结构是否满足某规则的理论条件。原因仍可能由上游切断点继承。

“查看结果”使用多选勾选框控制各组 panel（含对照统计与剪枝原因面板），支持全选、清空；自动刷新和切换 neuron 保留当前勾选。点击规则匹配的 neuron 时会勾选剪枝面板，同时保留其他已勾选项。

## 一键打开本地 Vaa3D

样本结果区的“在 Vaa3D 中打开 3D”按钮调用本地 `vaa3d://open` 协议。列表提供全选、取消全部叠加和恢复默认。
默认组合为一份原始三维图像、最终剪枝 SWC 和当前 Mask；直接在“勾选要加载的文件”列表中
勾选原始 Mask、各阶段 SWC 和登记的版本对照。尚未生成最终 SWC 时，优先选
已有的重采样 / G-Cut / APP2 结果；诊断 SWC 不自动选中。同一样本自动刷新保留
手动选择，切换样本会重新选择默认组合。

协议按 local-040 接口传入单个 `image` 和重复的 `swc`、`mask` 参数；由 Vaa3D
打开同一个三维视窗并显示 Object Manager。路径来自当前样本的保存清单和登记产物，
解析软链接并去重，不下载、不改写、不转换坐标，也不在服务器启动桌面程序。
原图优先使用 `manifests/<ID>.json` 中的 `raw`，否则查找本实验 `prepared/img` / `img`。
缺失原图时按钮禁用，不拿 MIP 或胞体裁剪图代替。文件选择不受下方 panel 显隐影响。

浏览器所在电脑须安装该协议处理器，并能访问所列绝对路径。首次使用时浏览器可能
询问是否打开 Vaa3D。传出链接不等于确认应用加载成功；坐标方向可在 Object Manager
检查、分别设置 Y flip。当前控制台不自动变换历史 SWC 或重采样尺寸不符的 Mask。

验证：`python -m unittest discover -s tests -p 'test_web*.py'` 与
`node --test tests/test_web*.js`，包括文件角色、原图定位、去重、编码、鉴权和切换样本。

## 按大步骤浏览下载资源

下载文件按前处理、Mask 修复、APP2、G-Cut、SWC 重采样、后续剪枝、回归保护、
最终比较分组；当前 neuron 的版本对照与整批统计单独分组。文件旁标明用途，
例如同名 SWC 分别标为 APP2 追踪、G-Cut 目标、1 µm 重采样、最终剪枝。
悬浮下载链接或结果图片时显示该资源的实际绝对路径，导入结果的根目录软链接会解析到原存储目录。
显示名和下载文件名、原有鉴权下载接口保持一致；路径仍受实验/登记对照目录边界约束。
“查看结果”的 panel 勾选继续控制图片；下载资源收在默认折叠的“下载结果与中间文件”面板中，点击展开后仍按步骤分组。自动刷新保留面板的展开状态，Vaa3D 文件勾选区独立显示。

## 选择算法流程版本

新建实验的“算法流程版本”支持 v0.81、v0.82、v0.82a、v0.83、v0.83a。
默认仍为 v0.83a。下方说明与阶段名称会随版本更新；实验监控与复制配置保留所选版本。
这是整套算法流程的选择，不是 nnUNet 模型权重选择。

- v0.81：对应工作树的原始 G-Cut 和后续剪枝，不执行 Mask 修复。
- v0.82：对应工作树的 APP2 GSDT、候选证据过滤、三维合并和剪枝，不执行 Mask 修复。
- v0.82a：保留该版本的通道零 / uint8、gamma=0.5 与 GPU 1 µm 重采样；后续调用该工作树代码。
- v0.83 / v0.83a：继续运行各自的 web_console/runner.py。

前三个版本由 web_console/legacy_stage.py 在独立进程中调用各自工作树的原生阶段函数，
不复制可执行算法快照，不改动旧版本源码或已有批处理配置。原始 image_<ID> 产物位于
本实验 reconstruction/native/，同时复制到控制台统一目录供查看/下载。输入软链接只读，
可写 mask/SWC 使用本实验副本，裁剪 JSON 仅在副本中适配 ZYX/ZXY。
第 7 步明确为原始合并 mask 直通，第 12 步仅记录未使用历史替换，不能伪装为执行过修复或回归保护。
第 1–6 步分别使用所选版本的解码/重采样、胞体裁剪、分割和合并规则；选择从已有 1 µm 图像
或 mask 开始时不会倒回执行该版本的原图前处理。

新实验记录所选版本的数值源码指纹；断点续跑及每个原生阶段启动前重新核对，源码变化后需新建实验。
已有重建目录同时接受 <ID>.swc 和 image_<ID>.swc；最终 SWC 也支持历史 gcut_pruned/pruning_swcs 布局。

2026-10-30：新运行已将 7.75 µm 短末梢删除移至剪枝最后一步。
历史 G-Cut 短末梢下载仍保留；新结果在剪枝原因中筛选 final_short_terminal，
审计位于 pruning_reasons.json / pruning-audit.json 的 final_short_leaf_audit。

2026-10-31：任务页增加整批样本状态面板，按完成、运行、等待、失败/超时、暂停/中断筛选，
支持编号搜索和点击跳转，运行中的并行样本分别显示当前阶段。外部批次读取每例检查点；
运行状态的总进度超过 90 秒未更新时显示中断提示，不再保留过期的运行中标记。
