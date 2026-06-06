# header-ai-train

## 工程用途

`header-ai-train` 是时间序列异常检测的训练工程，用于在电脑或训练服务器上训练 PyTorch AutoEncoder，并导出 Linux runtime 可部署的模型产物。

本工程最终必须产出：

```text
artifacts/model.onnx
artifacts/meta.json
```

## 工程边界

本工程负责：

1. 读取训练数据。
2. 构造滑动窗口。
3. 训练 PyTorch AutoEncoder。
4. 计算归一化参数。
5. 计算异常阈值。
6. 导出 ONNX 模型。
7. 使用 ONNX Runtime 验证模型。

本工程不负责：

1. Linux 嵌入式部署。
2. C++ 实时推理。
3. 现场报警输出。
4. 串口、CAN、GPIO、MQTT 等设备接口。

职责边界：

1. 本工程的职责终点是生成并验证 `model.onnx` 和 `meta.json`。
2. 本工程不包含 Linux C++ runtime、实时采集和现场报警实现。
3. 本工程修改 `meta.json` 合同时，必须同步更新 runtime 工程的解析规则。

## 与 runtime 工程的接口

`header-ai-train` 和 `header-ai-runtime` 之间只通过以下文件交付：

```text
model.onnx
meta.json
```

runtime 工程不应依赖 Python、PyTorch、pickle、训练数据或训练脚本。

## 项目文档

| 文档 | 说明 |
|---|---|
| `VERSION_PLAN.md` | 版本号规则、阶段拆分、验收标准 |
| `VERSION_IMPLEMENTATION_DETAILS.md` | 每个版本的期望效果、实现方式、输入输出、风险和验收细则 |
| `REFERENCES.md` | 参考项目、借鉴内容、采用/不采用的设计 |
| `IMPLEMENTATION_LOG.md` | 实施进展、阻塞项、多人协作记录 |
| `configs/default.yaml` | 默认训练配置 |

## 环境要求

推荐使用：

```text
Python 3.12
```

允许范围：

```text
Python >= 3.10, < 3.13
```

当前不建议使用 Python 3.13 或 Python 3.14，因为 PyTorch、ONNX Runtime 等训练依赖可能尚未完整支持。

## 本地环境初始化

Windows PowerShell：

```powershell
cd D:\header-ai-train
.\scripts\setup_env.ps1
.\scripts\verify_env.ps1
```

Linux/macOS：

```bash
cd header-ai-train
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
header-ai-train --version
```

## PyCharm 配置

PyCharm 不需要 Anaconda。推荐使用本项目的 `.venv`：

1. 先安装 Python 3.12。
2. 在 PowerShell 中运行 `.\scripts\setup_env.ps1`。
3. PyCharm 打开本项目根目录。
4. 进入 `Settings -> Project -> Python Interpreter`。
5. 选择 `Existing environment`。
6. 解释器路径选择：

```text
D:\header-ai-train\.venv\Scripts\python.exe
```

如果还没有 `.venv`，也可以在 PyCharm 中新建 Virtualenv，Base interpreter 选择 Python 3.12。

## 数据输入

`v0.2.0` 支持单变量 TXT 和 CSV 数据加载，并根据 `window_size` 和 `stride` 生成滑动窗口。

TXT 示例：

```text
98.1
98.2
98.3
```

CSV 示例：

```text
value
98.1
98.2
98.3
```

默认配置从以下路径读取 TXT 数据：

```text
data/train.txt
```

也可以在 `configs/default.yaml` 中修改：

```yaml
data:
  input_format: txt
  input_path: data/train.txt
  value_column: value
  window_size: 60
  stride: 1
```

## 训练前预处理

`v0.3.0` 支持训练集/验证集划分和 StandardScaler 归一化。

默认配置：

```yaml
training:
  validation_split: 0.2
  random_seed: 42
```

处理顺序：

1. 先对滑动窗口按 `random_seed` 做确定性划分。
2. 只使用训练窗口计算 `mean/std`。
3. 使用同一组 `mean/std` 归一化训练集和验证集。
4. `mean/std` 使用 JSON 数组保存，供后续 `meta.json` 写入。

## AutoEncoder 训练

`v0.9.0` 支持一条命令完成训练、ONNX 导出和 ONNX Runtime 验证：

```powershell
header-ai-train --config configs/default.yaml
```

完整流程：

```text
load config
  -> load data
  -> make windows
  -> split
  -> normalize
  -> train
  -> compute threshold
  -> write metrics.json
  -> write meta.json
  -> export model.onnx
  -> validate onnx
```

如需分阶段调试，也可以单独训练 MLP AutoEncoder 基线模型：

```powershell
python -m header_ai_train.train --config configs/default.yaml
```

当前阶段输出：

```text
artifacts/model.pt
artifacts/metrics.json
artifacts/meta.json
artifacts/model.onnx
artifacts/validation_report.json
```

模型结构：

```text
Input(input_dim)
  -> Linear(input_dim, hidden_dim)
  -> ReLU
  -> Linear(hidden_dim, latent_dim)
  -> ReLU
  -> Linear(latent_dim, hidden_dim)
  -> ReLU
  -> Linear(hidden_dim, input_dim)
```

`model.pt` 仅供训练工程内部使用，不作为 runtime 交付产物。

`v0.5.0` 会在正常训练窗口上计算每个窗口的重构 MSE，并按 `threshold.percentile` 计算异常阈值。`metrics.json` 包含误差分布统计和最终阈值：

```json
{
  "error_min": 0.0,
  "error_max": 0.0,
  "error_mean": 0.0,
  "error_p95": 0.0,
  "error_p99": 0.0,
  "threshold": 0.0,
  "threshold_percentile": 99.0
}
```

`v0.6.0` 会生成 runtime 合同文件 `meta.json`，包含输入输出名、窗口参数、归一化参数、异常阈值、报警默认参数和 ONNX opset。runtime 工程后续只应依赖 `model.onnx` 和 `meta.json`。

`v0.7.0` 支持根据 `model.pt` 和 `meta.json` 导出 ONNX：

```powershell
python -m header_ai_train.export_onnx --artifacts-dir artifacts
```

导出的 ONNX 输入输出名来自 `meta.json`，输入输出 shape 为：

```text
input:          [batch_size, input_dim]
reconstruction: [batch_size, input_dim]
```

`v0.8.0` 支持使用 ONNX Runtime 验证 ONNX 输出与 PyTorch 输出一致：

```powershell
python -m header_ai_train.validate_onnx --artifacts-dir artifacts
```

也可以用配置文件加载真实窗口进行验证：

```powershell
python -m header_ai_train.validate_onnx --artifacts-dir artifacts --config configs/default.yaml
```

验证通过后生成：

```text
artifacts/validation_report.json
```

## runtime 交付

`v0.10.0` 是第一阶段单变量时间序列可跑通版。完整流水线通过后，交付给 `header-ai-runtime` 的文件只有：

```text
artifacts/model.onnx
artifacts/meta.json
```

训练工程内部可保留：

```text
artifacts/model.pt
artifacts/metrics.json
artifacts/validation_report.json
```

交付前必须确认：

1. `validation_report.json` 中 `status` 为 `passed`。
2. `meta.json` 中 `input_dim == window_size * feature_dim`。
3. `meta.json` 中 `normalization.std` 不包含 `0`。
4. `model.onnx` 的输入输出名与 `meta.json` 一致。

## 版本推进顺序

详细版本拆分见：

```text
VERSION_PLAN.md
```

版本号统一采用：

```text
vA.B.C
```

规则：

| 字段 | 含义 |
|---|---|
| `A` | 大版本号，只有大规模更新、正式大版本发布或重大不兼容变更时修改 |
| `B` | 功能版本号，每添加一个新功能递增 |
| `C` | 修复版本号，用于 debug、缺陷修复和小范围功能修正 |

当前阶段固定在 `v0.B.C`，先把训练工程和 runtime 工程整体跑通，不进入 `v1`。

推荐按以下顺序推进：

```text
v0.1.0 -> v0.2.0 -> v0.3.0 -> v0.4.0 -> v0.5.0 -> v0.6.0 -> v0.7.0 -> v0.8.0 -> v0.9.0 -> v0.10.0
```

第一阶段目标是完成 `v0.10.0`：单变量时间序列 AutoEncoder 训练、ONNX 导出、`meta.json` 生成和 ONNX Runtime 验证。

## v0.10.0 最小交付目标

`v0.10.0` 完成时，本工程应稳定支持：

1. 单变量时间序列数据加载。
2. 滑动窗口切片。
3. StandardScaler 归一化。
4. MLP AutoEncoder 训练。
5. P99 异常阈值计算。
6. `model.onnx` 导出。
7. `meta.json` 生成。
8. PyTorch 与 ONNX Runtime 输出一致性验证。

## 后续扩展

`v0.10.0` 稳定后，再扩展：

1. 多变量时间序列。
2. 模型结构调优。
3. 阈值策略调优。
4. 有标签异常数据评估。
5. 自动生成训练评估报告。
