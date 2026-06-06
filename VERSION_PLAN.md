# header-ai-train 版本规划与功能拆分

## 0. 版本号规则

本工程统一使用版本号格式：

```text
vA.B.C
```

版本号含义：

| 字段 | 含义 | 修改规则 |
|---|---|---|
| `A` | 大版本号 | 只有发生大规模架构升级、工程方向变化、接口合同重大不兼容时才修改 |
| `B` | 功能版本号 | 每添加一个明确的新功能或完成一个阶段性功能模块时递增 |
| `C` | 修复版本号 | debug、缺陷修复、兼容性修复、小范围功能修正时递增 |

当前阶段统一固定在：

```text
v0.B.C
```

也就是说，在整个训练工程和 runtime 工程跑通之前，统一保持在 `v0` 阶段，不进入 `v1`。当前所有新增功能都通过递增 `B` 表示；同一功能阶段内的问题修复通过递增 `C` 表示。

示例：

```text
v0.3.0  新增归一化与数据集划分功能
v0.3.1  修复归一化 std 为 0 时的报错问题
v0.3.2  修复训练/验证划分随机种子不稳定问题
v1.0.0  只有在整体工程跑通并进入正式大版本，或发生大规模架构升级时才允许出现
```

## 1. 工程定位

`header-ai-train` 是异常检测模型训练工程，运行环境为开发电脑或训练服务器。

本工程只负责：

1. 读取时间序列训练数据。
2. 构造滑动窗口样本。
3. 训练 PyTorch AutoEncoder 异常检测模型。
4. 计算归一化参数和异常阈值。
5. 导出 `model.onnx`。
6. 生成 `meta.json`。
7. 使用 ONNX Runtime 在电脑端验证导出模型。

本工程不负责：

1. 嵌入式实时数据采集。
2. Linux C++ 部署。
3. 现场报警输出。
4. GPIO、串口、CAN、MQTT 等硬件或通信逻辑。

职责边界规定：

1. `header-ai-train` 的职责终点是生成并验证 `model.onnx` 和 `meta.json`。
2. `header-ai-train` 不允许引入任何 Linux 现场部署、C++ runtime、硬件报警相关实现。
3. `header-ai-train` 可以提供用于验证 ONNX 的 Python 推理脚本，但该脚本不能成为 runtime 的依赖。
4. `header-ai-train` 修改 `meta.json` 合同时，必须同步通知并更新 `header-ai-runtime` 的解析规则。

## 2. 与 runtime 工程的交付边界

`header-ai-train` 向 `header-ai-runtime` 交付以下文件：

```text
model.onnx
meta.json
```

两个工程之间禁止通过 Python 代码、pickle 文件、PyTorch 权重或训练脚本耦合。

`header-ai-runtime` 只能依赖 `model.onnx` 和 `meta.json` 完成推理。

## 3. 产物目录约定

训练工程最终产物建议放在：

```text
artifacts/
├── model.pt              # PyTorch 训练权重，仅训练工程内部使用
├── model.onnx            # 交付给 runtime 工程
├── meta.json             # 交付给 runtime 工程
├── validation_report.json
└── metrics.json
```

其中必须交付：

```text
model.onnx
meta.json
```

## 4. meta.json 合同

`meta.json` 是训练工程和运行时工程之间最重要的接口合同。

建议格式：

```json
{
  "schema_version": "1.0",
  "model_type": "autoencoder",
  "input_name": "input",
  "output_name": "reconstruction",
  "window_size": 60,
  "feature_dim": 1,
  "input_dim": 60,
  "flatten_order": "time_major",
  "threshold": 0.0123,
  "threshold_percentile": 99.0,
  "normalization": {
    "type": "standard",
    "mean": [0.0],
    "std": [1.0]
  },
  "alarm": {
    "consecutive_count": 3,
    "clear_count": 5
  },
  "onnx": {
    "opset": 17
  }
}
```

字段要求：

| 字段 | 要求 |
|---|---|
| `schema_version` | 用于保证 runtime 能识别当前元数据格式 |
| `input_name` | 必须与 ONNX 输入名一致 |
| `output_name` | 必须与 ONNX 输出名一致 |
| `window_size` | 滑动窗口长度 |
| `feature_dim` | 单个时间点的特征数量 |
| `input_dim` | 必须等于 `window_size * feature_dim` |
| `flatten_order` | 默认 `time_major`，表示按时间优先展平 |
| `threshold` | runtime 用于异常判断的最终阈值 |
| `normalization.mean` | 每个特征的均值 |
| `normalization.std` | 每个特征的标准差，不能为 0 |
| `alarm` | runtime 默认报警参数 |
| `onnx.opset` | ONNX 导出 opset |

## 5. 版本路线图

每个版本的详细期望效果、实施方式、输入输出、风险和验收细则见：

```text
VERSION_IMPLEMENTATION_DETAILS.md
```

### v0.1.0 - 训练工程基础初始化

目标：建立 Python 训练工程最小结构。

需要实现：

1. Python 项目目录结构。
2. 依赖管理文件，例如 `requirements.txt` 或 `pyproject.toml`。
3. 基础命令入口。
4. `artifacts/` 输出目录约定。
5. README 中说明工程用途和运行环境。

建议目录：

```text
header-ai-train/
├── README.md
├── requirements.txt
├── src/
│   └── header_ai_train/
│       ├── __init__.py
│       ├── dataset.py
│       ├── model.py
│       ├── train.py
│       ├── export_onnx.py
│       ├── validate_onnx.py
│       └── cli.py
└── artifacts/
```

验收标准：

1. 能创建 Python 虚拟环境并安装依赖。
2. 工程结构清晰。
3. 暂不要求训练模型。

### v0.2.0 - 数据加载与滑动窗口

目标：完成时间序列训练数据输入。

需要实现：

1. 支持从 CSV 或 TXT 加载单变量时间序列。
2. 支持配置 `window_size`。
3. 支持配置 `stride`。
4. 将原始序列切成固定长度滑动窗口。
5. 输出训练样本形状 `[num_windows, input_dim]`。
6. 过滤空值、非法值和长度不足的序列。

单变量输入示例：

```text
value
98.1
98.2
98.3
```

窗口输出示例：

```text
[x1, x2, ..., x60]
[x2, x3, ..., x61]
```

验收标准：

1. 给定一段长度为 100 的单变量序列，`window_size=60`、`stride=1` 时，应生成 41 个窗口。
2. 每个窗口展平后的长度等于 `input_dim`。
3. 数据加载错误需要明确报错，不能静默忽略。

### v0.3.0 - 归一化与数据集划分

目标：完成训练前预处理。

需要实现：

1. StandardScaler 归一化。
2. 只基于正常训练数据计算 `mean/std`。
3. 保存 `mean/std` 到内存结构，供后续写入 `meta.json`。
4. 支持训练集、验证集划分。
5. 支持固定随机种子，保证结果可复现。

归一化公式：

```text
x_norm = (x - mean) / std
```

验收标准：

1. `std` 不允许为 0。
2. 相同输入和随机种子下，训练/验证划分一致。
3. 输出归一化后的训练窗口可直接输入模型。

### v0.4.0 - PyTorch AutoEncoder 基线训练

目标：训练第一个可用的 AutoEncoder 模型。

需要实现：

1. MLP AutoEncoder 模型定义。
2. 训练循环。
3. MSELoss 重构损失。
4. Adam 优化器。
5. 训练日志输出。
6. 保存 `model.pt`。

推荐模型：

```text
Input(input_dim)
  -> Linear(input_dim, 128)
  -> ReLU
  -> Linear(128, 32)
  -> ReLU
  -> Linear(32, 128)
  -> ReLU
  -> Linear(128, input_dim)
```

验收标准：

1. 能使用正常数据完成训练。
2. 训练 loss 能正常下降。
3. 能生成 `artifacts/model.pt`。

### v0.5.0 - 重构误差与异常阈值

目标：计算 runtime 使用的异常判断阈值。

需要实现：

1. 在正常训练窗口上运行模型重构。
2. 计算每个窗口的 MSE。
3. 按百分位数计算阈值，默认 P99。
4. 输出误差分布统计。
5. 保存 `metrics.json`。

重构误差：

```text
mse = mean((reconstruction - input) ^ 2)
```

验收标准：

1. 能输出 `threshold`。
2. 能输出 min、max、mean、p95、p99 等误差统计。
3. 阈值必须写入后续 `meta.json`。

### v0.6.0 - meta.json 生成

目标：生成 runtime 可直接读取的元数据文件。

需要实现：

1. 写出 `schema_version`。
2. 写出模型输入输出名。
3. 写出 `window_size`、`feature_dim`、`input_dim`。
4. 写出 `flatten_order`。
5. 写出 `threshold`。
6. 写出归一化参数。
7. 写出默认报警参数。
8. 写出 ONNX opset 信息。

验收标准：

1. 能生成 `artifacts/meta.json`。
2. JSON 字段完整。
3. `input_dim == window_size * feature_dim`。
4. `normalization.std` 中不能出现 0。

### v0.7.0 - ONNX 导出

目标：将 PyTorch 模型导出为 ONNX。

需要实现：

1. 加载 `model.pt`。
2. 读取 `meta.json` 中的 `input_dim`、输入名、输出名和 opset。
3. 根据 `input_dim` 构造 dummy input。
4. 导出 `artifacts/model.onnx`。
5. batch 维度设置为动态。
6. 使用 ONNX checker 校验模型。

ONNX 输入输出要求：

```text
input:          [batch_size, input_dim]
reconstruction: [batch_size, input_dim]
```

1. 能生成 `artifacts/model.onnx`。
2. ONNX checker 通过。
3. ONNX 模型输入输出名称与 `meta.json` 一致。

### v0.8.0 - ONNX Runtime 验证

目标：确认 ONNX 模型与 PyTorch 模型推理结果一致。

需要实现：

1. 使用相同测试窗口分别输入 PyTorch 和 ONNX Runtime。
2. 读取 `meta.json` 中的输入名、输出名和 `input_dim`。
3. 比较两个模型输出。
4. 比较两个模型计算出的 MSE。
5. 生成 `validation_report.json`。

建议验收阈值：

```text
max_abs_diff < 1e-4
```

验收标准：

1. 验证报告包含 `max_abs_diff`。
2. 验证报告包含 PyTorch MSE 和 ONNX MSE。
3. 差异超过阈值时命令必须失败。

### v0.9.0 - 端到端训练命令

目标：提供一条命令完成训练、导出和验证。

需要实现：

```text
load data
  -> windowing
  -> normalization
  -> train
  -> threshold
  -> write meta.json
  -> export onnx using meta.json
  -> validate onnx using meta.json
  -> write reports
```

建议命令：

```bash
python -m header_ai_train.train --config configs/default.yaml
```

验收标准：

1. 一条命令能生成 `model.pt`、`model.onnx`、`meta.json`、`metrics.json`、`validation_report.json`。
2. 失败时能明确说明失败阶段。
3. 成功时输出产物路径。

### v0.10.0 - 单变量时间序列可跑通版

目标：形成可交付给 runtime 的第一个稳定版本。

功能范围：

1. 单变量时间序列。
2. MLP AutoEncoder。
3. StandardScaler。
4. P99 阈值。
5. ONNX Runtime 验证。
6. `model.onnx + meta.json` 稳定交付。

验收标准：

1. 训练工程能稳定生成 runtime 所需产物。
2. `meta.json` 合同稳定。
3. runtime 工程能够加载本版本产物。

### v0.11.0 - 多变量时间序列支持

目标：支持多个传感器或多个特征。

需要实现：

1. CSV 多列输入。
2. 配置 feature 列顺序。
3. `feature_dim > 1`。
4. `time_major` 展平顺序固定。
5. 每个特征独立计算 `mean/std`。

验收标准：

1. `input_dim == window_size * feature_dim`。
2. 多变量数据训练、导出、验证流程全部通过。
3. `meta.json` 中明确记录特征名和顺序。

### v0.12.0 - 模型与阈值调优

目标：提高检测效果。

可选实现：

1. 支持不同 hidden dimension。
2. 支持不同 latent dimension。
3. 支持 P95、P99、P99.5、P99.9 阈值。
4. 支持验证集评估。
5. 支持已标注异常数据的 precision、recall、F1 评估。

验收标准：

1. 可通过配置切换模型参数。
2. 可通过配置切换阈值百分位。
3. 有标注数据时能输出分类指标。

## 6. 推荐实施顺序

严格按以下顺序推进：

```text
v0.1.0 -> v0.2.0 -> v0.3.0 -> v0.4.0 -> v0.5.0 -> v0.6.0 -> v0.7.0 -> v0.8.0 -> v0.9.0 -> v0.10.0
```

`v0.10.0` 完成前，不建议提前做多变量、复杂模型或平台集成。

## 7. 第一阶段最小可行目标

最小可行版本到 `v0.10.0` 为止，必须能稳定产出：

```text
artifacts/model.onnx
artifacts/meta.json
```

并保证 `header-ai-runtime` 可以只依赖这两个文件完成 Linux C++ 推理。
