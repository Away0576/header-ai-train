"""PyTorch AutoEncoder model definitions.

AutoEncoder 的目标不是预测未来，而是“复原输入窗口”：
输入窗口 x -> 编码到低维 latent -> 解码得到 x'

训练时只看正常数据，因此模型会擅长重构正常模式；异常窗口由于不符合正常模式，
通常重构误差 MSE 会变大。
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class AutoEncoder(nn.Module):
    """MLP AutoEncoder baseline for flattened time-series windows.

    结构：
        input_dim -> hidden_dim -> latent_dim -> hidden_dim -> input_dim

    latent_dim 是瓶颈层，迫使模型学习正常窗口的压缩表示，而不是简单记忆输入。
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        for name, value in {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        # Encoder 把窗口压缩到 latent space。
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        # Decoder 尝试从 latent space 复原原始窗口。
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2:
            raise ValueError(f"Expected input tensor with shape [batch_size, input_dim], got {tuple(inputs.shape)}")
        if inputs.shape[1] != self.input_dim:
            raise ValueError(f"Expected input_dim {self.input_dim}, got {inputs.shape[1]}")
        return self.decoder(self.encoder(inputs))


def build_autoencoder(config: dict[str, Any], input_dim: int) -> AutoEncoder:
    """Build the configured AutoEncoder model from YAML config."""
    model_config = config.get("model", {})
    return AutoEncoder(
        input_dim=input_dim,
        hidden_dim=int(model_config.get("hidden_dim", 128)),
        latent_dim=int(model_config.get("latent_dim", 32)),
    )
