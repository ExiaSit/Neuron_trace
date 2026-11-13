本项目用户指定的实验输出位置：
- v0.84 方法实验结果统一保存在 /data/disk/Neuron_tracing_method/test/0.84/。
- data 保存输入与历史对照；新运行写 runs/ 下的独立目录，不覆盖源任务。
- 胞体识别入口 tools/run_gemma_soma_seeds.py：仅测试 G-Cut 前的胞体候选识别/判重，不执行 G-Cut 分配或后续剪枝。
- 第 11 步已恢复之前的基础剪枝。soma branch walk 是已停止的独立实验，不是默认剪枝。
- 2026-11-13 用户已授权将已审核胞体接入后续流程：tools/run_gemma_seed_downstream.py 从冻结 APP2 开始，成对执行原种子 / Gemma 种子的 G-Cut、重采样、基础剪枝和校验，独立目录保存对照；不把 Vaa3D 显示半径用作计算半径。
