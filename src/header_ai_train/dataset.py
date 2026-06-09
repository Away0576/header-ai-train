"""Dataset loading, sliding-window generation, and normalization utilities.

本模块对应训练流程的前半段：

raw TXT/CSV time series
  -> one-dimensional float array
  -> sliding windows [num_windows, window_size]
  -> train/validation split
  -> StandardScaler normalization

当前 v0.10.x 只支持“单变量时间序列”。多变量传感器输入会在后续版本扩展，
因此这里的 mean/std 都只有 1 个值。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SUPPORTED_INPUT_FORMATS = {"txt", "csv"}


def load_txt_series(path: Path | str) -> np.ndarray:
    """Load a single-variable time series from a TXT file.

    TXT 文件格式要求每行一个数值，例如：
        98.1
        98.2
        98.3

    空行会被跳过；非数值内容会直接报错，避免训练数据被静默污染。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"TXT data file not found: {path}")

    values: list[float] = []
    skipped_empty = 0
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            skipped_empty += 1
            continue
        try:
            values.append(float(line))
        except ValueError as exc:
            raise ValueError(f"Invalid numeric value in {path} at line {line_number}: {line!r}") from exc

    if not values:
        raise ValueError(f"No numeric values found in {path}; skipped {skipped_empty} empty lines")

    return _as_valid_series(values, source=path)


def load_csv_series(path: Path | str, value_column: str) -> np.ndarray:
    """Load a single-variable time series from one CSV column.

    CSV 可以包含时间戳列，但模型只使用 `value_column` 指定的数值列。
    例如 NAB 数据集格式是 `timestamp,value`，这里读取的是 `value`。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV data file not found: {path}")

    frame = pd.read_csv(path)
    if value_column not in frame.columns:
        available = ", ".join(str(column) for column in frame.columns)
        raise ValueError(f"Column {value_column!r} not found in {path}. Available columns: {available}")

    column = pd.to_numeric(frame[value_column], errors="coerce")
    invalid_count = int(column.isna().sum())
    if invalid_count:
        raise ValueError(f"Column {value_column!r} in {path} contains {invalid_count} empty or non-numeric values")

    return _as_valid_series(column.to_numpy(), source=path)


def make_windows(series: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """Convert a one-dimensional series into flattened sliding windows.

    AutoEncoder 不能直接处理无限长数据流，所以训练时先切固定长度窗口：

        [x1, x2, ..., x60]
        [x2, x3, ..., x61]

    单变量情况下，窗口天然就是扁平向量，shape 为 [num_windows, window_size]。
    runtime 的 SlidingWindow 必须使用同样的窗口长度和步长语义。
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    series = _as_valid_series(series, source="series")
    if len(series) < window_size:
        raise ValueError(f"Series length {len(series)} is shorter than window_size {window_size}")

    # sliding_window_view 不复制底层数据；ascontiguousarray 会把结果整理成
    # PyTorch/ONNX Runtime 更友好的连续 float32 矩阵。
    windows = np.lib.stride_tricks.sliding_window_view(series, window_shape=window_size)[::stride]
    return np.ascontiguousarray(windows, dtype=np.float32)


def load_dataset(config: dict[str, Any], project_root: Path | str | None = None) -> np.ndarray:
    """Load configured data and return windows with shape [num_windows, input_dim].

    这是训练入口使用的高层函数。它根据 `configs/default.yaml` 中的 data 配置
    决定读取 TXT 还是 CSV，并负责调用 `make_windows`。
    """
    project_root = Path.cwd() if project_root is None else Path(project_root)
    data_config = config.get("data", {})

    input_format = str(data_config.get("input_format", "txt")).lower()
    if input_format not in SUPPORTED_INPUT_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_INPUT_FORMATS))
        raise ValueError(f"Unsupported input_format {input_format!r}. Supported formats: {supported}")

    data_path = _resolve_data_path(config, project_root, input_format)
    if input_format == "txt":
        series = load_txt_series(data_path)
    else:
        series = load_csv_series(data_path, str(data_config.get("value_column", "value")))

    window_size = int(data_config.get("window_size", 60))
    stride = int(data_config.get("stride", 1))
    return make_windows(series, window_size=window_size, stride=stride)


def split_train_validation(
    windows: np.ndarray,
    validation_split: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split windows into deterministic train and validation subsets.

    这里使用随机种子打乱窗口，保证每次训练/验证划分可复现。
    validation_split=0 时允许没有验证集；否则必须至少产生 1 个验证窗口。
    """
    windows = _as_valid_windows(windows, source="windows")
    if not 0.0 <= validation_split < 1.0:
        raise ValueError(f"validation_split must be in [0.0, 1.0), got {validation_split}")

    num_windows = len(windows)
    validation_count = int(num_windows * validation_split)
    if validation_split > 0.0 and validation_count == 0:
        raise ValueError(
            f"validation_split {validation_split} creates no validation samples from {num_windows} windows"
        )
    train_count = num_windows - validation_count
    if train_count <= 0:
        raise ValueError("Training split is empty; reduce validation_split or provide more data")

    rng = np.random.default_rng(random_seed)
    indices = rng.permutation(num_windows)
    validation_indices = indices[:validation_count]
    train_indices = indices[validation_count:]

    train_windows = np.ascontiguousarray(windows[train_indices], dtype=np.float32)
    validation_windows = np.ascontiguousarray(windows[validation_indices], dtype=np.float32)
    return train_windows, validation_windows


def fit_standard_scaler(windows: np.ndarray) -> tuple[list[float], list[float]]:
    """Fit single-variable standard normalization parameters.

    只在训练窗口上计算 mean/std，避免验证数据或异常数据泄漏进归一化参数。
    返回 list 而不是 numpy 类型，是为了后续可以直接写入 `meta.json`。
    """
    windows = _as_valid_windows(windows, source="windows")
    mean = float(np.mean(windows, dtype=np.float64))
    std = float(np.std(windows, dtype=np.float64))
    if std == 0.0:
        raise ValueError("Standard normalization failed because std is 0; training data has no variation")
    if not np.isfinite(mean) or not np.isfinite(std):
        raise ValueError("Standard normalization failed because mean or std is not finite")
    return [mean], [std]


def transform_standard(windows: np.ndarray, mean: list[float] | np.ndarray, std: list[float] | np.ndarray) -> np.ndarray:
    """Apply single-variable standard normalization to windows.

    训练端和 runtime 端都必须使用同一公式：
        x_norm = (x - mean) / std

    AutoEncoder 的 MSE 阈值是在归一化空间中计算的，所以 runtime 也必须先归一化。
    """
    windows = _as_valid_windows(windows, source="windows")
    mean_array = np.asarray(mean, dtype=np.float32)
    std_array = np.asarray(std, dtype=np.float32)
    if mean_array.shape != (1,):
        raise ValueError(f"mean must contain one value for single-variable data, got shape {mean_array.shape}")
    if std_array.shape != (1,):
        raise ValueError(f"std must contain one value for single-variable data, got shape {std_array.shape}")
    if not np.isfinite(mean_array).all() or not np.isfinite(std_array).all():
        raise ValueError("mean and std must contain finite values")
    if std_array[0] == 0.0:
        raise ValueError("std must not be 0")

    normalized = (windows - mean_array[0]) / std_array[0]
    return np.ascontiguousarray(normalized, dtype=np.float32)


def prepare_train_validation(
    windows: np.ndarray,
    validation_split: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[float] | str]]:
    """Split windows and normalize train/validation data with train-fitted parameters.

    这是训练前预处理的组合函数：
    1. 切分 train/validation；
    2. 只用 train fit scaler；
    3. 用同一组 mean/std transform train 和 validation；
    4. 返回 normalization 字典，后续写入 `meta.json`。
    """
    train_windows, validation_windows = split_train_validation(windows, validation_split, random_seed)
    mean, std = fit_standard_scaler(train_windows)
    train_windows_norm = transform_standard(train_windows, mean, std)
    validation_windows_norm = transform_standard(validation_windows, mean, std) if len(validation_windows) else validation_windows
    normalization = {
        "type": "standard",
        "mean": mean,
        "std": std,
    }
    return train_windows_norm, validation_windows_norm, normalization


def _resolve_data_path(config: dict[str, Any], project_root: Path, input_format: str) -> Path:
    """Resolve the input data file from config.

    如果配置了 data.input_path，则优先使用它；否则在 paths.data_dir 下查找唯一
    一个匹配后缀的文件。发现多个文件时要求显式配置，避免误用数据集。
    """
    paths_config = config.get("paths", {})
    data_config = config.get("data", {})
    configured_path = data_config.get("input_path")

    if configured_path:
        path = Path(configured_path)
        return path if path.is_absolute() else project_root / path

    data_dir = Path(paths_config.get("data_dir", "data"))
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    candidates = sorted(path for path in data_dir.glob(f"*.{input_format}") if path.is_file())
    if not candidates:
        raise FileNotFoundError(
            f"No .{input_format} data file found in {data_dir}. "
            "Set data.input_path in the config or add a data file."
        )
    if len(candidates) > 1:
        candidate_list = ", ".join(str(path) for path in candidates)
        raise ValueError(f"Multiple .{input_format} data files found. Set data.input_path explicitly: {candidate_list}")
    return candidates[0]


def _as_valid_series(values: Any, source: Path | str) -> np.ndarray:
    """Convert input values to a valid one-dimensional float32 series."""
    series = np.asarray(values, dtype=np.float32)
    if series.ndim != 1:
        raise ValueError(f"Expected one-dimensional series from {source}, got shape {series.shape}")
    if series.size == 0:
        raise ValueError(f"Series from {source} is empty")
    if not np.isfinite(series).all():
        raise ValueError(f"Series from {source} contains NaN or infinite values")
    return series


def _as_valid_windows(values: Any, source: Path | str) -> np.ndarray:
    """Convert input values to a valid two-dimensional float32 window matrix."""
    windows = np.asarray(values, dtype=np.float32)
    if windows.ndim != 2:
        raise ValueError(f"Expected two-dimensional windows from {source}, got shape {windows.shape}")
    if windows.shape[0] == 0:
        raise ValueError(f"Windows from {source} are empty")
    if windows.shape[1] == 0:
        raise ValueError(f"Windows from {source} have empty input dimension")
    if not np.isfinite(windows).all():
        raise ValueError(f"Windows from {source} contain NaN or infinite values")
    return windows
