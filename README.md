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
