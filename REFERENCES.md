# 参考项目说明

本文档记录 `header-ai-train` 参考的开源项目、可借鉴内容，以及本工程采用/不采用的设计边界，方便新成员快速理解项目来源和实现方向。

## 1. 参考项目列表

| 项目 | 地址 | 主要参考点 |
|---|---|---|
| SpO2 Anomaly Detection | https://github.com/fairy1234556/spo2-anomaly-detection | 时间序列滑动窗口、AutoEncoder 输入形状 `[batch, window_size]`、重构误差 MSE、百分位阈值、ONNX 导出 |
| BoxFile Audit Anomaly Detection | https://github.com/Ronakdeora/boxfile-audit-anomaly | 工程化训练链路、只用正常数据训练、保存 `meta.json`、ONNX checker 校验、ONNX Runtime 验证、归一化参数随模型发布 |

## 2. 本工程采用的关键设计

### 2.1 时间序列滑动窗口

参考 `spo2-anomaly-detection`，本工程将连续时间序列切成固定长度窗口：

```text
[x1, x2, ..., x60]
[x2, x3, ..., x61]
[x3, x4, ..., x62]
```

模型输入：

```text
[batch_size, input_dim]
```

单变量第一阶段：

```text
window_size = 60
feature_dim = 1
input_dim = 60
```

### 2.2 AutoEncoder 异常检测

参考两个项目的共同思路：

1. 使用正常数据训练 AutoEncoder。
2. 模型学习正常模式的重构能力。
3. 异常数据重构误差更高。
4. 使用 MSE 作为异常分数。

重构误差：

```text
mse = mean((reconstruction - input) ^ 2)
```

### 2.3 百分位阈值

参考两个项目的阈值策略，本工程第一阶段默认使用正常训练数据重构误差的 P99：

```text
threshold = percentile(normal_train_errors, 99)
```

后续可根据现场误报/漏报情况调整为 P95、P99.5 或 P99.9。

### 2.4 meta.json 作为部署合同

参考 `boxfile-audit-anomaly`，本工程不使用 pickle 作为部署配置，不让 runtime 依赖 Python 对象。

训练工程必须输出：

```text
artifacts/model.onnx
artifacts/meta.json
```

runtime 工程只能依赖这两个文件。

`meta.json` 至少包含：

```text
window_size
feature_dim
input_dim
input_name
output_name
threshold
normalization.mean
normalization.std
onnx.opset
alarm.consecutive_count
alarm.clear_count
```

### 2.5 ONNX Runtime 验证

参考 `boxfile-audit-anomaly` 的工程化方式，导出 ONNX 后必须进行验证：

1. PyTorch 模型推理。
2. ONNX Runtime 推理。
3. 比较输出差异。
4. 比较 MSE 差异。
5. 差异超过阈值时失败。

建议阈值：

```text
max_abs_diff < 1e-4
```

## 3. 本工程暂不采用的内容

| 内容 | 原因 |
|---|---|
| 图像可视化 | 第一阶段目标是生成 runtime 可部署产物，图像报告后续再做 |
| 复杂模型，如 LSTM/Transformer | 第一阶段优先跑通 MLP AutoEncoder + ONNX + C++ runtime |
| 多变量输入 | 放到 `v0.11.0` 后再支持 |
| Python runtime 部署 | 嵌入式端必须使用 C++ + ONNX Runtime |
| pickle 元数据 | 不利于跨语言部署，不适合作为 runtime 合同 |

## 4. 快速上手阅读顺序

建议新成员按以下顺序阅读：

1. `README.md`
2. `VERSION_PLAN.md`
3. `REFERENCES.md`
4. `IMPLEMENTATION_LOG.md`
5. `configs/default.yaml`
6. `src/header_ai_train/`

## 5. 当前实现优先级

当前优先完成 `v0.1.0 -> v0.10.0`：

```text
工程骨架
  -> 数据加载
  -> 滑动窗口
  -> 归一化
  -> AutoEncoder 训练
  -> 阈值计算
  -> meta.json
  -> ONNX 导出
  -> ONNX Runtime 验证
  -> 端到端跑通
```
