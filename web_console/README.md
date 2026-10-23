# v0.83 重建实验工作台

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
mask/<id>.tif
trace_app2/down_sampled_swcs_app2/<id>.swc
gcut_output/selected_for_pruning/<id>.swc
gcut_output/selected_resampled_step1/<id>.swc
pruning_swcs/<id>.swc
```

这些可写中间产物复制到新实验，不通过硬链接修改源结果。灰度原图、初始
mask 和胞体辅助输入原位读取；其大小、时间戳及元数据变化会使续跑拒绝。
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
cd /data/disk/Neuron_tracing_method/v0.83
/home/pzy/anaconda3/envs/hb_seg/bin/python -m web_console.server \
  --host 127.0.0.1 --port 8766 \
  --state-root /data/disk/Neuron_tracing_method/results/v083-console
```

## 验证

```bash
cd /data/disk/Neuron_tracing_method/v0.83
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/pzy/anaconda3/envs/hb_seg/bin/python -m unittest discover -s tests -v
node --check web_console/static/app.js
```

测试覆盖真实 TIFF 重采样与裁剪、预览生成、区间依赖、输入重复/缺失、
已完成结果跳过、元数据变化拒绝、外部 SWC 复制、立即停止子进程、API 认证、
跨站请求和目录越界拒绝。真实样本与本次验证范围见 VALIDATION.txt。

## 追查每次 mask 修复与短末梢删除

在结果区的“查看结果”选择“Mask 每步变化”“最终比较”或“G-Cut 短末梢删除”。
Mask 按 a–h 顺序显示八步：骨架及方向分析、信号重连、胞体补全、胞体与分支重连、
连接厚度恢复、胞体根部补全、受约束延伸、根部测地线补全。
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

G-Cut 红线和叉号标记短末梢删除，青色表示保留部分。记录从胞体收缩之后开始、在
G-Cut 分离之前结束，规则是“不计起始分叉点，不足两个节点”，不是两微米。
gcut_audits/ 提供三视图、删除节点 ID、前后 SWC 和阈值说明；后续任务的 G-Cut 总图
也会叠加删除标记。未删除任何节点时会显示 0。

已有任务补图需要保留 a–f 中间 mask 和 G-Cut injected SWC/soma ID：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/pzy/anaconda3/envs/hb_seg/bin/python -m utils.backfill_diagnostics \
  --spec /absolute/job/reconstruction/manifests/16984.json \
  --previous-app2-dir /data/disk/C6.0/trace_app2/down_sampled_swcs_app2 \
  --previous-app2-dir /data/disk2/B4.5/trace_app2/down_sampled_swcs_app2 \
  --replay-continuation
```

仅在任务停止或完成后使用。旧任务缺少 g/h 快照时，--replay-continuation 单独复算，
并要求最终 mask 与已有结果逐体素一致，否则拒绝发布分步图；图和页面明确标注复算。
不使用该选项则只展示最后两步的合并变化。G-Cut 删除审计通过保留输入复算预处理，
不重跑追踪或分离。补图核对原 mask、APP2、G-Cut、重采样和最终 SWC 的 SHA256；
另存 ID_diagnostics.png，页面优先显示它，原比较图和完成检查点保留。
