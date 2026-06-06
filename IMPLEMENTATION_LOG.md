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

## 2026-06-06 - v0.6.0 - meta.json 生成

状态：DONE
负责人：Codex / SESA855007

变更内容：
1. 实现 `build_meta`，生成 runtime 可读取的部署合同字段。
2. 实现 `validate_meta`，校验必填字段、输入输出名、维度关系和归一化参数。
3. 实现 `write_meta`，输出 `artifacts/meta.json`。
4. 在训练流程中生成 `model.pt`、`metrics.json` 和 `meta.json`。
5. 将项目版本推进到 `0.6.0`。

验证结果：
1. `input_dim == window_size * feature_dim`。
2. `normalization.std` 不含 0。
3. `input_name/output_name` 非空。
4. `meta.json` 可 JSON 解析。
5. CLI 版本输出为 `header-ai-train 0.6.0`。

阻塞项：
1. 无。

下一步：
1. 代码审查后推送 `feature/train-pipeline-v0.4-v0.6`。

## 2026-06-06 - v0.7.0 - ONNX 导出

状态：DONE
负责人：Codex / SESA855007

变更内容：
1. 实现 `load_meta`，读取并校验 `artifacts/meta.json`。
2. 实现 `export_model_to_onnx`，从 `model.pt` 和 `meta.json` 导出 `model.onnx`。
3. 设置 ONNX 输入输出名与 `meta.json` 一致。
4. 设置 batch 维度为动态轴。
5. 使用 ONNX checker 校验模型。
6. 增加 ONNX 输入输出 shape 和名称校验。
7. 使用 `dynamo=False` 的传统 ONNX exporter，稳定支持当前 opset 17 和 dynamic_axes 配置。
8. 将项目版本推进到 `0.7.0`。

验证结果：
1. 能生成 `artifacts/model.onnx`。
2. ONNX checker 通过。
3. ONNX 输入名等于 `meta.input_name`。
4. ONNX 输出名等于 `meta.output_name`。
5. 输入输出 shape 为 `[batch_size, input_dim]`。

问题与处理：
1. Torch 2.12 默认 `dynamo=True` 导出路径需要 `onnxscript`，且在 Windows GBK 控制台下可能因导出日志包含非 GBK 字符失败。
2. 当前模型是简单 MLP AutoEncoder，因此改用传统 exporter `dynamo=False`，避免新增不必要依赖。
3. 为避免 ONNX tracing 记录 Python shape check，导出时使用 `_AutoEncoderOnnxWrapper` 包装模型。
4. 已对已知 legacy exporter deprecation warning 做精确屏蔽，保留 ONNX checker 和输入输出校验作为有效验证。

阻塞项：
1. 无。

下一步：
1. 在当前阶段分支继续推进 `v0.8.0` ONNX Runtime 验证。

## 2026-06-06 - v0.8.0 - ONNX Runtime 验证

状态：DONE
负责人：Codex / SESA855007

变更内容：
1. 实现 `validate_onnx_outputs`，比较 PyTorch 和 ONNX Runtime 输出。
2. 实现 ONNX Runtime session 输入输出名称和 shape 校验。
3. 计算 `max_abs_diff`、PyTorch MSE、ONNX MSE 和 MSE 差异。
4. 实现 `write_validation_report`，生成 `artifacts/validation_report.json`。
5. 实现 `python -m header_ai_train.validate_onnx --artifacts-dir artifacts` 命令。
6. 支持可选 `--config` 加载真实窗口，并使用 `meta.json` 归一化参数验证。
7. 将项目版本推进到 `0.8.0`。

验证结果：
1. `max_abs_diff < 1e-4`。
2. PyTorch MSE 与 ONNX MSE 接近。
3. 能生成并解析 `validation_report.json`。
4. 超过容差时命令失败。
5. CLI 版本输出为 `header-ai-train 0.8.0`。

阻塞项：
1. 无。

下一步：
1. 代码审查后推送 `feature/onnx-runtime-v0.7-v0.8`。

## 2026-06-06 - v0.9.0 - 端到端训练命令

状态：DONE
负责人：Codex / SESA855007

变更内容：
1. 将 `header-ai-train --config configs/default.yaml` 实现为端到端流水线入口。
2. 串联配置加载、数据加载、滑动窗口、划分、归一化、训练、阈值计算、`metrics.json`、`meta.json`、ONNX 导出和 ONNX Runtime 验证。
3. 增加阶段日志，失败时输出明确失败阶段。
4. 成功时输出 `model.pt`、`model.onnx`、`meta.json`、`metrics.json` 和 `validation_report.json` 路径。
5. 将项目版本推进到 `0.9.0`。

验证结果：
1. 一条命令能完成全部流程。
2. 能生成 `model.pt`、`model.onnx`、`meta.json`、`metrics.json` 和 `validation_report.json`。
3. ONNX Runtime 验证通过。
4. 失败阶段错误信息包含阶段名。
5. CLI 版本输出为 `header-ai-train 0.9.0`。

阻塞项：
1. 无。

下一步：
1. 在当前阶段分支继续推进 `v0.10.0` 单变量时间序列可跑通版验收。

## 2026-06-06 - v0.10.0 - 单变量时间序列可跑通版

状态：DONE
负责人：Codex / SESA855007

变更内容：
1. 将第一阶段能力收口为单变量时间序列可交付版本。
2. 在端到端流水线末尾增加 runtime 交付产物校验。
3. 校验 `model.onnx`、`meta.json` 和 `validation_report.json` 存在。
4. 重新校验 `meta.json` 合同，并确认 `validation_report.json` 状态为 `passed`。
5. 更新 README，明确 runtime 只交付 `model.onnx` 和 `meta.json`。
6. 将项目版本推进到 `0.10.0`。

验证结果：
1. 单变量 TXT 数据端到端训练通过。
2. 生成 `artifacts/model.onnx` 和 `artifacts/meta.json`。
3. 生成 `artifacts/model.pt`、`artifacts/metrics.json` 和 `artifacts/validation_report.json`。
4. `validation_report.json` 验证通过。
5. `meta.json` 合同校验通过。
6. CLI 版本输出为 `header-ai-train 0.10.0`。

阻塞项：
1. 无。

下一步：
1. 代码审查后推送 `feature/e2e-v0.9-v0.10`。

## 5. 待办列表

| 版本 | 任务 | 状态 | 备注 |
|---|---|---|---|
| v0.1.0 | 工程骨架与环境配置 | DONE | 环境验收通过 |
| v0.2.0 | 数据加载与滑动窗口 | DONE | 支持 TXT/CSV 单变量 |
| v0.3.0 | 归一化与数据集划分 | DONE | StandardScaler |
| v0.4.0 | PyTorch AutoEncoder 基线训练 | DONE | MLP AutoEncoder |
| v0.5.0 | 重构误差与异常阈值 | DONE | 默认 P99 |
| v0.6.0 | meta.json 生成 | DONE | runtime 合同 |
| v0.7.0 | ONNX 导出 | DONE | checker 校验 |
| v0.8.0 | ONNX Runtime 验证 | DONE | PyTorch/ONNX 对齐 |
| v0.9.0 | 端到端训练命令 | DONE | 一条命令生成全部产物 |
| v0.10.0 | 单变量时间序列可跑通版 | DONE | 交付 runtime |
