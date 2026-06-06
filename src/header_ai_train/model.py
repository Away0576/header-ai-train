"""PyTorch AutoEncoder model definitions."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class AutoEncoder(nn.Module):
    """MLP AutoEncoder baseline for flattened time-series windows."""

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
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
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
    """Build the configured AutoEncoder model."""
    model_config = config.get("model", {})
    return AutoEncoder(
        input_dim=input_dim,
        hidden_dim=int(model_config.get("hidden_dim", 128)),
        latent_dim=int(model_config.get("latent_dim", 32)),
    )
