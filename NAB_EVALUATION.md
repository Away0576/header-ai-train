# NAB 数据集训练与异常评估记录

本文档记录使用 NAB 公开时间序列样本验证 `header-ai-train v0.10.2` 的过程、命令、产物和结果。

## 1. 数据来源

使用 Numenta Anomaly Benchmark（NAB）公开数据：

| 用途 | 文件 | 来源 |
|---|---|---|
| 正常训练/正常评估 | `art_daily_no_noise.csv` | `numenta/NAB/data/artificialNoAnomaly/art_daily_no_noise.csv` |
| 异常评估 | `art_daily_jumpsup.csv` | `numenta/NAB/data/artificialWithAnomaly/art_daily_jumpsup.csv` |

本地保存路径：

```text
data/nab/art_daily_no_noise.csv
data/nab/art_daily_jumpsup.csv
```

CSV 格式：

```text
timestamp,value
2014-04-01 00:00:00,20.0
2014-04-01 00:05:00,20.0
...
```

当前工程只读取 `value` 列作为单变量时间序列。

## 2. 训练配置

本次实验使用临时配置：

```text
artifacts/nab_config.yaml
```

关键配置：

```yaml
data:
  input_format: csv
  input_path: data/nab/art_daily_no_noise.csv
  value_column: value
  window_size: 60
  stride: 1
  feature_dim: 1
  flatten_order: time_major

model:
  hidden_dim: 128
  latent_dim: 32

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  validation_split: 0.2
  random_seed: 42

threshold:
  percentile: 99.9
```

## 3. 训练命令

```powershell
cd C:\Users\SESA855007\header-ai-train
.\.venv\Scripts\header-ai-train.exe --config artifacts\nab_config.yaml
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
  -> verify runtime delivery artifacts
```

## 4. 训练产物

本次 NAB 实验产物保存到：

```text
artifacts/nab/
```

文件：

| 文件 | 说明 |
|---|---|
| `model.pt` | PyTorch checkpoint，仅训练工程内部使用 |
| `model.onnx` | runtime 交付模型 |
| `meta.json` | runtime 交付元数据 |
| `metrics.json` | 训练窗口重构误差统计 |
| `validation_report.json` | PyTorch 与 ONNX Runtime 一致性验证报告 |
| `evaluation_no_noise.csv` | 正常样本逐窗口评估结果 |
| `evaluation_jumpsup.csv` | 异常样本逐窗口评估结果 |

## 5. 本次训练结果

训练 loss 从：

```text
Epoch 1 train_loss = 0.251689
```

下降到：

```text
Epoch 100 train_loss = 0.001656
Epoch 100 validation_loss = 0.001808
```

这说明 AutoEncoder 已经学会较好重构 NAB 正常样本窗口。

## 6. metrics.json

```json
{
  "error_min": 0.000036836936487816274,
  "error_max": 0.009978564456105232,
  "error_mean": 0.0015918738496682517,
  "error_p95": 0.005454277154058218,
  "error_p99": 0.00712855439633131,
  "threshold": 0.009978564456105232,
  "threshold_percentile": 99.9
}
```

最终异常阈值：

```text
threshold = 0.009978564456105232
```

## 7. meta.json 关键字段

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
  "threshold": 0.009978564456105232,
  "threshold_percentile": 99.9,
  "normalization": {
    "type": "standard",
    "mean": [43.097919775672935],
    "std": [28.07703083103901]
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

## 8. ONNX Runtime 验证

```json
{
  "status": "passed",
  "max_abs_diff": 0.0,
  "max_abs_diff_tolerance": 0.0001,
  "pytorch_mse": 0.000045272723663221845,
  "onnx_mse": 0.000045272723663221845,
  "mse_abs_diff": 0.0,
  "num_windows": 16,
  "input_name": "input",
  "output_name": "reconstruction",
  "input_dim": 60
}
```

结论：

```text
PyTorch 输出与 ONNX Runtime 输出一致，ONNX 导出可用于 runtime 对接。
```

## 9. 评估命令

正常样本评估：

```powershell
python -m header_ai_train.evaluate `
  --artifacts-dir artifacts\nab `
  --input-path data\nab\art_daily_no_noise.csv `
  --input-format csv `
  --value-column value `
  --output-csv artifacts\nab\evaluation_no_noise.csv
```

异常样本评估：

```powershell
python -m header_ai_train.evaluate `
  --artifacts-dir artifacts\nab `
  --input-path data\nab\art_daily_jumpsup.csv `
  --input-format csv `
  --value-column value `
  --output-csv artifacts\nab\evaluation_jumpsup.csv
```

## 10. 评估结果

| 数据 | 窗口数 | 异常窗口数 | 异常率 | 最大 MSE | 平均 MSE |
|---|---:|---:|---:|---:|---:|
| `art_daily_no_noise.csv` | 3973 | 0 | 0.00% | 0.0099785626 | 0.0016350439 |
| `art_daily_jumpsup.csv` | 3973 | 2013 | 50.67% | 1.0713998079 | 0.0277908941 |

判断规则：

```text
mse > threshold => anomaly
```

## 11. 结果解读

本次实验说明：

1. 训练工程的端到端流程可运行。
2. AutoEncoder 能学习 NAB 正常样本的窗口模式。
3. ONNX 导出与 ONNX Runtime 验证通过，模型可交付 runtime。
4. 正常样本评估中没有窗口超过阈值。
5. 异常样本 `art_daily_jumpsup.csv` 中约一半窗口超过阈值，说明该方案能识别明显跳变异常。

注意事项：

1. NAB 是公开样例数据，不代表最终现场效果。
2. 当前版本只输出逐窗口异常判断，不计算 precision、recall、F1。
3. 当前模型只支持单变量时间序列。
4. 现场部署前需要使用真实正常数据训练，并使用真实或人工注入异常数据评估。
