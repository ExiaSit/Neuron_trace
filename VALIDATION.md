# v0.82 移植验证（2026-10-21）

本版以 0.81 为基础，只移植当前 0.84_test 的 G-Cut 及其只读候选计算依赖。
没有加入 mask 补全、mask 重连或修改输入 mask 的阶段。

- 42 项单元/接口检查通过：原生 GSDT、当前胞体证据过滤、目标胞体区域归属、GPU 后端、
  0.81 原有运行接口，以及本版的原始 mask 传递、进程隔离、阶段顺序和胞体几何只读检查。
  初次运行缺少一个测试 fixture；补入同版本的 fixture 后该测试通过。
- 4 例移植对照：15214、107046、105612、16963。两版使用相同原图、原始输入 mask、
  既有 APP2 SWC、辅助胞体分割和元数据。候选生成、证据过滤、NMS、信号合并、
  所有分离 SWC、目标树几何及胞体区域均一致；运行前后输入文件未改变。
  这些对照复用 APP2 输入，用于验证 G-Cut 移植等价性。
- 15214 另用原始 mask 重新运行 0.81 APP2，再执行移植后的 G-Cut、0.81 重采样和剪枝，
  确认阶段接口、文件名、裁剪坐标转换及后续处理能实际衔接。

本次没有重新运行 nnUNet 模型推理。分割、mask 合并、APP2、重采样与剪枝的算法
经 AST 比较与本地 0.81 一致；已有 0.81 接口检查继续通过。

本机详细记录位于 `validation/gcut-port-20261021/`（运行产物不进入 Git）：

- `unit-tests.log`、`fixture-retry.log`：检查记录。
- `comparison.json`、`cases/<id>/comparison.json`：逐例移植对照结果。
- `cases/<id>/v082/gcut_output/<id>/`：本版总览图、候选点 SWC 和目标 SWC。
- `pipeline-smoke.log`、`pipeline-smoke-15214/`：15214 的 APP2 至剪枝实跑。

复跑接口检查：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m unittest discover -s tests -v
```
