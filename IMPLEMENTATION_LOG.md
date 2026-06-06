# 实施日志

本文档用于记录 `header-ai-train` 的实施进展、负责人、变更内容和后续待办，方便多人同时协作。

## 1. 使用规则

1. 每次开始一个功能版本时，在日志中新增一条记录。
2. 每次完成、阻塞或修复问题时，更新对应记录。
3. 不在日志中记录密码、token、私有数据路径或训练数据敏感内容。
4. 多人同时开发时，先查看“当前进行中”和“阻塞项”，避免重复工作。
5. 代码提交时，建议在提交信息中带上对应版本号，例如 `v0.2.0 dataset windowing`。

## 2. 状态定义

| 状态 | 含义 |
|---|---|
| `TODO` | 尚未开始 |
| `IN_PROGRESS` | 正在进行 |
| `DONE` | 已完成 |
| `BLOCKED` | 被外部条件阻塞 |
| `FIXING` | 正在修复问题 |

## 3. 日志模板

```text
## YYYY-MM-DD - v0.x.y - 标题

状态：
负责人：

变更内容：
1.
2.
3.

验证结果：
1.
2.

阻塞项：
1.

下一步：
1.
```

## 4. 当前进行中

## 2026-06-05 - v0.1.0 - 训练工程基础初始化

状态：DONE
负责人：Copilot / SESA855007

变更内容：
1. 初始化 `header-ai-train` 仓库文档。
2. 新增 `README.md`。
3. 新增 `VERSION_PLAN.md`。
4. 新增 Python 工程骨架。
5. 新增 `pyproject.toml`、`requirements.txt`、`.python-version`。
6. 新增 `configs/default.yaml`。
7. 新增 `scripts/setup_env.ps1` 和 `scripts/verify_env.ps1`。
8. 新增参考项目说明 `REFERENCES.md`。
9. 新增实施日志 `IMPLEMENTATION_LOG.md`。

验证结果：
1. 当前机器检测到 Python 3.14、Python 3.12、Python 3.11 和 Python 3.9。
2. 已安装 Python 3.12.10。
3. 已创建 `.venv`，虚拟环境使用 Python 3.12.10。
4. 已完成依赖安装，关键依赖包括 torch、onnx、onnxruntime、numpy、pandas、scikit-learn 和 PyYAML。
5. `.\scripts\verify_env.ps1` 验证通过。

阻塞项：
1. 无。

下一步：
1. 开始 `v0.2.0` 数据加载与滑动窗口实现。

## 2026-06-06 - v0.2.0 - 数据加载与滑动窗口

状态：DONE
负责人：Codex / SESA855007

变更内容：
1. 实现 TXT 单变量时间序列加载。
2. 实现 CSV 指定数值列加载。
3. 实现滑动窗口切片，输出 shape 为 `[num_windows, input_dim]`。
4. 增加输入格式、路径、窗口长度、步长、空数据、非数值和长度不足校验。
5. 将项目版本推进到 `0.2.0`。
6. 更新 README 数据输入说明和默认配置。

验证结果：
1. 长度 100 的序列，`window_size=60`、`stride=1`，生成窗口 shape 为 `(41, 60)`。
2. TXT 加载验证通过。
3. CSV 加载验证通过。
4. CLI 版本输出为 `header-ai-train 0.2.0`。

阻塞项：
1. 无。

下一步：
1. 开始 `v0.3.0` 归一化与数据集划分实现。

## 2026-06-06 - v0.3.0 - 归一化与数据集划分

状态：DONE
负责人：Codex / SESA855007

变更内容：
1. 实现 `split_train_validation`，支持固定随机种子的确定性训练/验证划分。
2. 实现 `fit_standard_scaler`，只基于训练窗口计算单变量 `mean/std`。
3. 实现 `transform_standard`，使用同一组 `mean/std` 归一化训练集和验证集。
4. 实现 `prepare_train_validation`，串联划分、拟合归一化参数和转换流程。
5. 增加 `std == 0`、非法验证比例、空窗口和非有限值校验。
6. 将项目版本推进到 `0.3.0`。

验证结果：
1. 相同输入和随机种子下，训练/验证划分一致。
2. `mean/std` 为 JSON 可序列化数组。
3. 归一化输出 dtype 为 `float32`。
4. `std == 0` 时明确报错。
5. CLI 版本输出为 `header-ai-train 0.3.0`。

阻塞项：
1. 无。

下一步：
1. 开始 `v0.4.0` PyTorch AutoEncoder 基线训练实现。

## 2026-06-06 - v0.4.0 - PyTorch AutoEncoder 基线训练

状态：DONE
负责人：Codex / SESA855007

变更内容：
1. 实现 `AutoEncoder(input_dim, hidden_dim, latent_dim)` MLP 基线模型。
2. 实现模型输入维度校验，输入输出 shape 均为 `[batch_size, input_dim]`。
3. 实现 `train_autoencoder`，使用 DataLoader、MSELoss 和 Adam 训练。
4. 实现训练 loss 和验证 loss 日志输出。
5. 实现 `save_model_checkpoint` 和 `load_model_checkpoint`。
6. 实现 `python -m header_ai_train.train --config configs/default.yaml` 阶段性训练入口。
7. 将项目版本推进到 `0.4.0`。

验证结果：
1. 模型输入输出 shape 验证通过。
2. 使用合成正常窗口完成至少 1 个 epoch 训练。
3. 训练 loss 为有限值。
4. 能保存并重新加载 `model.pt`。
5. CLI 版本输出为 `header-ai-train 0.4.0`。

阻塞项：
1. 无。

下一步：
1. 在当前阶段分支继续推进 `v0.5.0` 重构误差与异常阈值。

## 2026-06-06 - v0.5.0 - 重构误差与异常阈值

状态：DONE
负责人：Codex / SESA855007

变更内容：
1. 实现 `compute_reconstruction_errors`，输出每个训练窗口的 MSE。
2. 实现 `compute_threshold`，按配置百分位数计算异常阈值。
3. 实现 `build_metrics` 和 `write_metrics`，生成 `artifacts/metrics.json`。
4. 在训练流程中保存 `model.pt` 后计算训练窗口误差、阈值和指标。
5. 将项目版本推进到 `0.5.0`。

验证结果：
1. 误差数组长度等于训练窗口数量。
2. `threshold >= 0`。
3. `metrics.json` 可 JSON 解析。
4. `metrics.json` 包含 min、max、mean、p95、p99、threshold 和 threshold_percentile。
5. CLI 版本输出为 `header-ai-train 0.5.0`。

阻塞项：
1. 无。

下一步：
1. 在当前阶段分支继续推进 `v0.6.0` meta.json 生成。

## 5. 待办列表

| 版本 | 任务 | 状态 | 备注 |
|---|---|---|---|
| v0.1.0 | 工程骨架与环境配置 | DONE | 环境验收通过 |
| v0.2.0 | 数据加载与滑动窗口 | DONE | 支持 TXT/CSV 单变量 |
| v0.3.0 | 归一化与数据集划分 | DONE | StandardScaler |
| v0.4.0 | PyTorch AutoEncoder 基线训练 | DONE | MLP AutoEncoder |
| v0.5.0 | 重构误差与异常阈值 | DONE | 默认 P99 |
| v0.6.0 | meta.json 生成 | TODO | runtime 合同 |
| v0.7.0 | ONNX 导出 | TODO | checker 校验 |
| v0.8.0 | ONNX Runtime 验证 | TODO | PyTorch/ONNX 对齐 |
| v0.9.0 | 端到端训练命令 | TODO | 一条命令生成全部产物 |
| v0.10.0 | 单变量时间序列可跑通版 | TODO | 交付 runtime |
