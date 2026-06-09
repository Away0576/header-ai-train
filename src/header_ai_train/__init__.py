"""header-ai-train package.

这个包只负责“训练侧”工作：
1. 读取正常时间序列数据；
2. 切成固定长度滑动窗口；
3. 训练 PyTorch AutoEncoder；
4. 导出 runtime 需要的 `model.onnx` 和 `meta.json`。

注意：Linux/C++ 实时推理、报警、串口/CAN/GPIO 等部署逻辑不属于本包。
"""

__version__ = "0.10.2"
