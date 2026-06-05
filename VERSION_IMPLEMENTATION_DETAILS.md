# 版本实施细则

本文档对 `header-ai-train` 每个版本的期望效果、实现方式、输入输出、验收标准、风险点和协作注意事项进行详细说明。

版本号遵循：

```text
vA.B.C
```

当前阶段固定在 `v0.B.C`：

| 字段 | 含义 |
|---|---|
| `A` | 大版本号，当前固定为 0，整体跑通或重大升级后才进入 v1 |
| `B` | 功能版本号，每新增一个明确功能递增 |
| `C` | 修复版本号，用于 debug、缺陷修复、小范围修正 |

## v0.1.0 - 训练工程基础初始化

### 目标

建立可协作、可扩展、可被 PyCharm 打开的 Python 训练工程骨架。

### 期望效果

1. 新成员拉取仓库后，可以快速理解工程用途。
2. 项目具备标准 Python 包结构。
3. 后续数据加载、训练、导出、验证代码有明确放置位置。
4. 环境配置方式被固化为脚本和文档。

### 实现方式

1. 创建 `src/header_ai_train/` 包目录。
2. 创建基础模块文件：
   - `dataset.py`
   - `model.py`
   - `train.py`
   - `export_onnx.py`
   - `validate_onnx.py`
   - `cli.py`
3. 创建 `pyproject.toml` 管理项目依赖。
4. 创建 `requirements.txt` 作为 pip 安装入口。
5. 创建 `.python-version` 固定推荐 Python 版本。
6. 创建 `configs/default.yaml` 保存默认配置。
7. 创建 `scripts/setup_env.ps1` 和 `scripts/verify_env.ps1`。
8. 创建 `artifacts/` 和 `data/` 目录。

### 输入

无业务输入。

### 输出

```text
项目基础目录结构
pyproject.toml
requirements.txt
configs/default.yaml
scripts/setup_env.ps1
scripts/verify_env.ps1
```

### 验收标准

1. 能打开 PyCharm 项目。
2. 安装 Python 3.12 后能运行 `.\scripts\setup_env.ps1`。
3. 能运行 `.\scripts\verify_env.ps1`。
4. CLI 能输出版本号。

### 风险点

| 风险 | 处理方式 |
|---|---|
| Python 版本过新导致依赖不可安装 | 固定推荐 Python 3.12，限制 `<3.13` |
| PyCharm 生成 `.idea/` 被误提交 | `.gitignore` 忽略 `.idea/` |
| 模型产物误提交 | `.gitignore` 忽略 `.onnx`、`.pt`、`artifacts/*` |

### 协作注意事项

1. 不在 v0.1.0 中实现训练逻辑。
2. 不提交本地 `.venv`。
3. 不提交真实训练数据。

## v0.2.0 - 数据加载与滑动窗口

### 目标

实现单变量时间序列数据读取，并切分为 AutoEncoder 可训练的滑动窗口样本。

### 期望效果

1. 支持读取 TXT 单列数据。
2. 支持读取 CSV 中指定数值列。
3. 能根据 `window_size` 和 `stride` 生成窗口。
4. 输出形状满足 `[num_windows, input_dim]`。

### 实现方式

在 `dataset.py` 中实现：

1. `load_txt_series(path: Path) -> np.ndarray`
2. `load_csv_series(path: Path, value_column: str) -> np.ndarray`
3. `make_windows(series: np.ndarray, window_size: int, stride: int) -> np.ndarray`
4. `load_dataset(config) -> np.ndarray`

窗口规则：

```text
series = [x1, x2, x3, ..., xn]
window_size = 60
stride = 1

windows:
[x1, x2, ..., x60]
[x2, x3, ..., x61]
[x3, x4, ..., x62]
```

窗口数量：

```text
num_windows = floor((len(series) - window_size) / stride) + 1
```

### 输入

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

配置项：

```yaml
data:
  input_format: txt
  value_column: value
  window_size: 60
  stride: 1
```

### 输出

```text
numpy.ndarray, shape = [num_windows, window_size]
```

### 验收标准

1. 长度 100 的序列，`window_size=60`、`stride=1`，生成 41 个窗口。
2. 窗口 dtype 为 `float32`。
3. 输入长度小于 `window_size` 时明确报错。
4. 非数值数据明确报错或按规则过滤，并记录数量。

### 风险点

| 风险 | 处理方式 |
|---|---|
| 数据存在空行 | 明确过滤并记录 |
| 数据存在非数值 | 报错或过滤策略必须固定 |
| 训练端和 runtime 端窗口定义不一致 | 文档固定窗口与展平规则 |

### 协作注意事项

1. 第一阶段只做单变量。
2. 多变量输入放到 `v0.11.0`。
3. 不在该版本引入 PyTorch 训练。

## v0.3.0 - 归一化与数据集划分

### 目标

实现训练数据标准化，并支持训练集/验证集划分。

### 期望效果

1. 使用正常训练窗口计算 `mean/std`。
2. 使用同一组 `mean/std` 归一化训练集和验证集。
3. 归一化参数后续可写入 `meta.json`。

### 实现方式

新增或在 `dataset.py` 中实现：

1. `fit_standard_scaler(windows) -> mean, std`
2. `transform_standard(windows, mean, std) -> normalized_windows`
3. `split_train_validation(windows, validation_split, random_seed)`

归一化公式：

```text
x_norm = (x - mean) / std
```

单变量第一阶段：

```text
mean = [global_mean]
std = [global_std]
```

### 输入

```text
windows: [num_windows, input_dim]
validation_split: 0.2
random_seed: 42
```

### 输出

```text
train_windows_norm
val_windows_norm
normalization.mean
normalization.std
```

### 验收标准

1. `std` 不为 0。
2. 相同随机种子下划分结果一致。
3. 输出 dtype 为 `float32`。
4. `mean/std` 可 JSON 序列化。

### 风险点

| 风险 | 处理方式 |
|---|---|
| `std == 0` | 明确报错，提示数据无变化 |
| 数据泄漏 | 只用训练集 fit scaler |
| runtime 归一化不一致 | `mean/std` 必须写入 `meta.json` |

### 协作注意事项

1. 不使用 pickle 保存 scaler。
2. 归一化参数必须是 JSON 数组。

## v0.4.0 - PyTorch AutoEncoder 基线训练

### 目标

实现第一个可训练的 MLP AutoEncoder 基线模型。

### 期望效果

1. 能用正常窗口训练模型。
2. loss 正常下降。
3. 生成 `artifacts/model.pt`。

### 实现方式

在 `model.py` 中实现：

```text
AutoEncoder(input_dim, hidden_dim, latent_dim)
```

推荐结构：

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

在 `train.py` 中实现：

1. DataLoader。
2. MSELoss。
3. Adam。
4. epoch 训练循环。
5. 训练日志。
6. 保存 `model.pt`。

### 输入

```text
train_windows_norm
input_dim
hidden_dim
latent_dim
epochs
batch_size
learning_rate
```

### 输出

```text
artifacts/model.pt
training loss log
```

### 验收标准

1. 模型输入输出 shape 均为 `[batch_size, input_dim]`。
2. 能完成至少 1 个 epoch。
3. 能保存并重新加载 `model.pt`。
4. 训练 loss 不为 NaN。

### 风险点

| 风险 | 处理方式 |
|---|---|
| 输入维度不匹配 | 根据配置计算 `input_dim` 并断言 |
| loss 为 NaN | 检查归一化、学习率和输入数据 |
| 模型过大 | 第一阶段保持轻量 MLP |

### 协作注意事项

1. 不在该版本导出 ONNX。
2. 不引入复杂模型。

## v0.5.0 - 重构误差与异常阈值

### 目标

计算每个窗口的重构误差，并根据正常数据误差分布得到异常阈值。

### 期望效果

1. 能输出所有训练窗口的 MSE。
2. 能计算 P99 阈值。
3. 能生成 `metrics.json`。

### 实现方式

在 `train.py` 或独立模块中实现：

1. `compute_reconstruction_errors(model, windows)`
2. `compute_threshold(errors, percentile)`
3. `write_metrics(errors, threshold)`

MSE：

```text
mse = mean((reconstruction - input) ^ 2)
```

默认阈值：

```text
threshold = percentile(errors, 99)
```

### 输入

```text
trained_model
train_windows_norm
threshold.percentile
```

### 输出

```text
threshold
artifacts/metrics.json
```

`metrics.json` 建议包含：

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

### 验收标准

1. `threshold >= 0`。
2. 误差数组长度等于训练窗口数量。
3. `metrics.json` 可 JSON 解析。
4. 阈值后续可写入 `meta.json`。

### 风险点

| 风险 | 处理方式 |
|---|---|
| 阈值过低导致误报 | 后续通过百分位数调整 |
| 训练数据混入异常 | 先保证训练数据尽量为正常数据 |
| MSE 计算空间不一致 | 训练端和 runtime 都在归一化空间计算 MSE |

## v0.6.0 - meta.json 生成

### 目标

生成 runtime 可读取的部署合同文件 `meta.json`。

### 期望效果

1. runtime 不需要 Python 代码即可理解模型输入、归一化和阈值。
2. `model.onnx` 导出和验证也读取同一份 `meta.json`。

### 实现方式

新增模块或函数：

```text
write_meta(config, normalization, threshold, metrics)
```

输出字段至少包括：

```text
schema_version
model_type
input_name
output_name
window_size
feature_dim
input_dim
flatten_order
threshold
threshold_percentile
normalization
alarm
onnx
```

### 输入

```text
config
normalization.mean
normalization.std
threshold
metrics
```

### 输出

```text
artifacts/meta.json
```

### 验收标准

1. `input_dim == window_size * feature_dim`。
2. `normalization.std` 不含 0。
3. `input_name/output_name` 非空。
4. JSON 可被 runtime 解析。

### 风险点

| 风险 | 处理方式 |
|---|---|
| meta 合同频繁变化 | 修改时同步更新 runtime 文档 |
| 字段缺失 | 增加 meta 校验函数 |
| 数字类型不可序列化 | numpy 类型转换为 Python float/int |

### 协作注意事项

1. `meta.json` 是两个工程的接口合同。
2. 修改字段必须同步通知 runtime 开发者。

## v0.7.0 - ONNX 导出

### 目标

将 PyTorch AutoEncoder 导出为 `model.onnx`。

### 期望效果

1. 生成可被 ONNX Runtime 加载的 ONNX 模型。
2. ONNX 输入输出名与 `meta.json` 一致。
3. batch 维度可动态变化。

### 实现方式

在 `export_onnx.py` 中实现：

1. 读取 `meta.json`。
2. 加载 `model.pt`。
3. 重建 AutoEncoder。
4. 创建 dummy input `[1, input_dim]`。
5. 调用 `torch.onnx.export`。
6. 使用 `onnx.checker.check_model`。
7. 可选执行 shape inference。

### 输入

```text
artifacts/model.pt
artifacts/meta.json
```

### 输出

```text
artifacts/model.onnx
```

### 验收标准

1. `model.onnx` 文件存在。
2. ONNX checker 通过。
3. 输入名等于 `meta.input_name`。
4. 输出名等于 `meta.output_name`。
5. 输入输出 shape 为 `[batch_size, input_dim]`。

### 风险点

| 风险 | 处理方式 |
|---|---|
| PyTorch 与 ONNX opset 不兼容 | 固定 opset 17 |
| 输入输出名不一致 | 从 `meta.json` 读取名称 |
| runtime 加载失败 | 导出后必须进入 v0.8.0 验证 |

## v0.8.0 - ONNX Runtime 验证

### 目标

验证 ONNX 模型和 PyTorch 模型在相同输入下输出一致。

### 期望效果

1. 发现导出错误。
2. 确认 runtime 使用 ONNX 模型时结果可信。
3. 生成验证报告。

### 实现方式

在 `validate_onnx.py` 中实现：

1. 读取 `meta.json`。
2. 加载 PyTorch 模型。
3. 加载 ONNX Runtime session。
4. 使用同一批测试窗口推理。
5. 比较输出最大绝对误差。
6. 比较 MSE。
7. 写出 `validation_report.json`。

建议阈值：

```text
max_abs_diff < 1e-4
```

### 输入

```text
artifacts/model.pt
artifacts/model.onnx
artifacts/meta.json
test_windows_norm
```

### 输出

```text
artifacts/validation_report.json
```

### 验收标准

1. `max_abs_diff < 1e-4`。
2. PyTorch MSE 与 ONNX MSE 接近。
3. 超过阈值时命令失败。

### 风险点

| 风险 | 处理方式 |
|---|---|
| ONNX Runtime 不支持当前 Python | 使用 Python 3.12 |
| 数值误差略有差异 | 使用合理容差 |
| 输入数据未归一化 | 验证输入必须与训练一致 |

## v0.9.0 - 端到端训练命令

### 目标

提供一条命令完成从数据到产物的完整训练流程。

### 期望效果

用户只需执行一条命令：

```powershell
header-ai-train --config configs/default.yaml
```

即可生成：

```text
model.pt
model.onnx
meta.json
metrics.json
validation_report.json
```

### 实现方式

在 `cli.py` 或 `train.py` 中串联：

```text
load config
  -> load data
  -> make windows
  -> split
  -> normalize
  -> train
  -> compute threshold
  -> write metrics
  -> write meta.json
  -> export onnx
  -> validate onnx
```

### 输入

```text
configs/default.yaml
data/
```

### 输出

```text
artifacts/model.pt
artifacts/model.onnx
artifacts/meta.json
artifacts/metrics.json
artifacts/validation_report.json
```

### 验收标准

1. 一条命令能完成全部流程。
2. 任一阶段失败时明确提示阶段和原因。
3. 成功后输出产物路径。
4. 日志中能看到训练 loss、threshold 和 ONNX 验证结果。

### 风险点

| 风险 | 处理方式 |
|---|---|
| CLI 参数混乱 | 第一阶段只保留 `--config` |
| 阶段失败不明显 | 每个阶段打印明确标题 |
| 重复运行覆盖产物 | 默认覆盖，后续可增加 run_id |

## v0.10.0 - 单变量时间序列可跑通版

### 目标

形成第一个可以交付给 `header-ai-runtime` 的单变量模型训练版本。

### 期望效果

1. 从单变量数据训练模型。
2. 导出 `model.onnx`。
3. 生成 `meta.json`。
4. ONNX Runtime 验证通过。
5. runtime 工程可以开始加载该版本产物。

### 实现方式

整合 `v0.1.0` 到 `v0.9.0` 的能力，并进行完整回归验证。

### 输入

```text
正常单变量时间序列数据
configs/default.yaml
```

### 输出

必须交付：

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

### 验收标准

1. `model.onnx` 和 `meta.json` 可以交给 runtime。
2. `validation_report.json` 验证通过。
3. `meta.json` 合同稳定。
4. README 中有完整运行步骤。

### 风险点

| 风险 | 处理方式 |
|---|---|
| runtime 无法解析 meta | 与 runtime 联调 |
| 阈值现场效果不佳 | 后续 v0.12.0 调优 |
| 数据格式不统一 | 当前只承诺单变量 TXT/CSV |

## v0.11.0 - 多变量时间序列支持

### 目标

支持多个传感器或多个特征列。

### 期望效果

1. CSV 多列输入。
2. 每个时间点包含多个 feature。
3. 滑动窗口按 `time_major` 展平。
4. 每个 feature 独立归一化。

### 实现方式

扩展 `dataset.py`：

1. 支持 `feature_columns`。
2. 输入 shape 从 `[time]` 扩展为 `[time, feature_dim]`。
3. 窗口 shape 为 `[num_windows, window_size, feature_dim]`。
4. 展平为 `[num_windows, window_size * feature_dim]`。

展平顺序：

```text
[t1_f1, t1_f2, ..., t1_fn, t2_f1, ..., t60_fn]
```

### 输入

CSV 示例：

```text
temperature,current,voltage
36.1,1.2,220.0
36.2,1.3,219.8
```

### 输出

```text
input_dim = window_size * feature_dim
meta.json 中记录 feature_columns
```

### 验收标准

1. 多变量训练、导出、验证全流程通过。
2. `meta.json` 中明确记录 feature 顺序。
3. runtime 端可按相同顺序展平。

## v0.12.0 - 模型与阈值调优

### 目标

提升异常检测效果，支持不同模型参数和阈值策略。

### 期望效果

1. 可配置 hidden_dim。
2. 可配置 latent_dim。
3. 可配置阈值百分位。
4. 有标签数据时可输出 precision、recall、F1。

### 实现方式

扩展配置：

```yaml
model:
  hidden_dim: 128
  latent_dim: 32

threshold:
  percentile: 99.0
```

如果提供 label：

```text
label = 0 正常
label = 1 异常
```

则额外计算分类指标。

### 输入

```text
训练数据
可选标签数据
配置参数
```

### 输出

```text
metrics.json
validation_report.json
可选 evaluation_report.json
```

### 验收标准

1. 配置改动能影响模型结构或阈值。
2. 无标签数据仍可正常训练。
3. 有标签数据时能输出分类指标。

### 风险点

| 风险 | 处理方式 |
|---|---|
| 过度调参导致过拟合 | 保留简单基线 |
| 指标依赖异常标签 | 标签评估作为可选能力 |
| 阈值变化影响现场报警 | 与 runtime 配置一起评估 |
