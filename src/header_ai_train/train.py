"""Training utilities for the AutoEncoder baseline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from header_ai_train import __version__
from header_ai_train.dataset import load_dataset, prepare_train_validation
from header_ai_train.model import AutoEncoder, build_autoencoder


@dataclass(frozen=True)
class TrainingResult:
    model: AutoEncoder
    train_losses: list[float]
    validation_losses: list[float]
    model_path: Path
    normalization: dict[str, list[float] | str]


def train_autoencoder(
    train_windows_norm: np.ndarray,
    *,
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    random_seed: int,
    validation_windows_norm: np.ndarray | None = None,
) -> tuple[AutoEncoder, list[float], list[float]]:
    """Train an AutoEncoder with MSE reconstruction loss."""
    if input_dim <= 0:
        raise ValueError(f"input_dim must be positive, got {input_dim}")
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")

    _set_random_seed(random_seed)
    train_windows_norm = _as_training_windows(train_windows_norm, input_dim, "train_windows_norm")
    if validation_windows_norm is not None and len(validation_windows_norm):
        validation_windows_norm = _as_training_windows(validation_windows_norm, input_dim, "validation_windows_norm")
    else:
        validation_windows_norm = None

    model = AutoEncoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(random_seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_windows_norm)),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    train_losses: list[float] = []
    validation_losses: list[float] = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        sample_count = 0
        for (batch,) in loader:
            optimizer.zero_grad()
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            if not torch.isfinite(loss):
                raise ValueError(f"Training loss is not finite at epoch {epoch}")
            loss.backward()
            optimizer.step()

            batch_size_actual = len(batch)
            epoch_loss += float(loss.detach().cpu()) * batch_size_actual
            sample_count += batch_size_actual

        train_loss = epoch_loss / sample_count
        train_losses.append(train_loss)
        if validation_windows_norm is not None:
            validation_losses.append(evaluate_reconstruction_loss(model, validation_windows_norm))
        print(_format_epoch_log(epoch, epochs, train_loss, validation_losses[-1] if validation_losses else None))

    return model, train_losses, validation_losses


def evaluate_reconstruction_loss(model: AutoEncoder, windows_norm: np.ndarray) -> float:
    """Compute mean reconstruction MSE for normalized windows."""
    windows_norm = _as_training_windows(windows_norm, model.input_dim, "windows_norm")
    loss_fn = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(windows_norm)
        loss = loss_fn(model(inputs), inputs)
    loss_value = float(loss.cpu())
    if not np.isfinite(loss_value):
        raise ValueError("Reconstruction loss is not finite")
    return loss_value


def save_model_checkpoint(
    model: AutoEncoder,
    model_path: Path | str,
    *,
    train_losses: list[float],
    validation_losses: list[float],
) -> Path:
    """Save a reloadable PyTorch model checkpoint."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "project_version": __version__,
        "model_type": "autoencoder",
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "latent_dim": model.latent_dim,
        "model_state_dict": model.state_dict(),
        "train_losses": train_losses,
        "validation_losses": validation_losses,
    }
    torch.save(checkpoint, model_path)
    return model_path


def load_model_checkpoint(model_path: Path | str) -> tuple[AutoEncoder, dict[str, Any]]:
    """Load a model saved by save_model_checkpoint."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = AutoEncoder(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        latent_dim=int(checkpoint["latent_dim"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def run_training(config: dict[str, Any], project_root: Path | str | None = None) -> TrainingResult:
    """Run v0.4 training from config and save artifacts/model.pt."""
    project_root = Path.cwd() if project_root is None else Path(project_root)
    windows = load_dataset(config, project_root=project_root)

    training_config = config.get("training", {})
    train_windows_norm, validation_windows_norm, normalization = prepare_train_validation(
        windows,
        validation_split=float(training_config.get("validation_split", 0.2)),
        random_seed=int(training_config.get("random_seed", 42)),
    )
    input_dim = train_windows_norm.shape[1]
    model_config = config.get("model", {})
    model, train_losses, validation_losses = train_autoencoder(
        train_windows_norm,
        input_dim=input_dim,
        hidden_dim=int(model_config.get("hidden_dim", 128)),
        latent_dim=int(model_config.get("latent_dim", 32)),
        epochs=int(training_config.get("epochs", 100)),
        batch_size=int(training_config.get("batch_size", 32)),
        learning_rate=float(training_config.get("learning_rate", 0.001)),
        random_seed=int(training_config.get("random_seed", 42)),
        validation_windows_norm=validation_windows_norm,
    )

    artifacts_dir = Path(config.get("paths", {}).get("artifacts_dir", "artifacts"))
    if not artifacts_dir.is_absolute():
        artifacts_dir = project_root / artifacts_dir
    model_path = save_model_checkpoint(
        model,
        artifacts_dir / "model.pt",
        train_losses=train_losses,
        validation_losses=validation_losses,
    )
    return TrainingResult(
        model=model,
        train_losses=train_losses,
        validation_losses=validation_losses,
        model_path=model_path,
        normalization=normalization,
    )


def load_config(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m header_ai_train.train",
        description="Train the v0.4 PyTorch AutoEncoder baseline.",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to the training config file.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config_path = Path(args.config)
    result = run_training(load_config(config_path), project_root=Path.cwd())
    print(f"Saved model checkpoint: {result.model_path}")


def _as_training_windows(windows: np.ndarray, input_dim: int, name: str) -> np.ndarray:
    windows = np.asarray(windows, dtype=np.float32)
    if windows.ndim != 2:
        raise ValueError(f"{name} must have shape [num_windows, input_dim], got {windows.shape}")
    if windows.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one window")
    if windows.shape[1] != input_dim:
        raise ValueError(f"{name} input_dim mismatch: expected {input_dim}, got {windows.shape[1]}")
    if not np.isfinite(windows).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return np.ascontiguousarray(windows, dtype=np.float32)


def _set_random_seed(random_seed: int) -> None:
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)


def _format_epoch_log(epoch: int, epochs: int, train_loss: float, validation_loss: float | None) -> str:
    if validation_loss is None:
        return f"Epoch {epoch}/{epochs} train_loss={train_loss:.6f}"
    return f"Epoch {epoch}/{epochs} train_loss={train_loss:.6f} validation_loss={validation_loss:.6f}"


if __name__ == "__main__":
    main()
