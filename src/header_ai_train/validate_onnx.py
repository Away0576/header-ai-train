"""ONNX Runtime validation utilities.

ONNX 导出成功不等于数值结果一定正确。本模块负责把同一批归一化窗口分别
输入 PyTorch 模型和 ONNX Runtime，比较输出差异和 MSE 差异。

只有验证通过的 `model.onnx` 才应交付给 runtime 工程。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch

from header_ai_train.dataset import load_dataset, transform_standard
from header_ai_train.export_onnx import load_meta
from header_ai_train.train import load_config, load_model_checkpoint


DEFAULT_MAX_ABS_DIFF_TOLERANCE = 1e-4


@dataclass(frozen=True)
class OnnxValidationResult:
    """Summary of PyTorch-vs-ONNX validation."""
    report_path: Path
    max_abs_diff: float
    pytorch_mse: float
    onnx_mse: float
    mse_abs_diff: float


def validate_onnx_outputs(
    model_path: Path | str,
    onnx_path: Path | str,
    meta_path: Path | str,
    test_windows_norm: np.ndarray,
    report_path: Path | str,
    *,
    max_abs_diff_tolerance: float = DEFAULT_MAX_ABS_DIFF_TOLERANCE,
) -> OnnxValidationResult:
    """Compare PyTorch and ONNX Runtime outputs for the same normalized windows.

    test_windows_norm 必须已经完成 StandardScaler 归一化，否则 MSE 不具备
    与训练阈值比较的意义。
    """
    meta = load_meta(meta_path)
    input_dim = int(meta["input_dim"])
    test_windows_norm = _as_validation_windows(test_windows_norm, input_dim)

    # PyTorch 路径：加载训练 checkpoint，用同一批窗口得到基准输出。
    pytorch_model, _ = load_model_checkpoint(model_path)
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_input = torch.from_numpy(test_windows_norm)
        pytorch_output = pytorch_model(pytorch_input).cpu().numpy()

    # ONNX Runtime 路径：使用 CPUExecutionProvider，模拟嵌入式 CPU 推理。
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    _validate_session_io(session, input_name=str(meta["input_name"]), output_name=str(meta["output_name"]), input_dim=input_dim)
    onnx_output = session.run([str(meta["output_name"])], {str(meta["input_name"]): test_windows_norm})[0]

    max_abs_diff = float(np.max(np.abs(pytorch_output - onnx_output)))
    pytorch_mse = float(np.mean((pytorch_output - test_windows_norm) ** 2, dtype=np.float64))
    onnx_mse = float(np.mean((onnx_output - test_windows_norm) ** 2, dtype=np.float64))
    mse_abs_diff = float(abs(pytorch_mse - onnx_mse))
    # validation_report.json 是交付前的质量门禁；CLI 会要求 status=passed。
    report = {
        "status": "passed" if max_abs_diff < max_abs_diff_tolerance else "failed",
        "max_abs_diff": max_abs_diff,
        "max_abs_diff_tolerance": float(max_abs_diff_tolerance),
        "pytorch_mse": pytorch_mse,
        "onnx_mse": onnx_mse,
        "mse_abs_diff": mse_abs_diff,
        "num_windows": int(len(test_windows_norm)),
        "input_name": str(meta["input_name"]),
        "output_name": str(meta["output_name"]),
        "input_dim": input_dim,
    }
    report_path = write_validation_report(report, report_path)
    if max_abs_diff >= max_abs_diff_tolerance:
        raise ValueError(
            f"ONNX validation failed: max_abs_diff {max_abs_diff} >= tolerance {max_abs_diff_tolerance}"
        )
    return OnnxValidationResult(
        report_path=report_path,
        max_abs_diff=max_abs_diff,
        pytorch_mse=pytorch_mse,
        onnx_mse=onnx_mse,
        mse_abs_diff=mse_abs_diff,
    )


def validate_from_artifacts(
    artifacts_dir: Path | str,
    *,
    config: dict[str, Any] | None = None,
    project_root: Path | str | None = None,
    max_windows: int = 16,
    max_abs_diff_tolerance: float = DEFAULT_MAX_ABS_DIFF_TOLERANCE,
) -> OnnxValidationResult:
    """Validate artifacts/model.onnx against artifacts/model.pt.

    如果传入 config，则用真实数据窗口做验证；否则生成确定性随机窗口，主要用于
    检查 ONNX 导出是否数值一致。
    """
    artifacts_dir = Path(artifacts_dir)
    meta = load_meta(artifacts_dir / "meta.json")
    if config is None:
        test_windows_norm = _make_deterministic_probe_windows(input_dim=int(meta["input_dim"]), max_windows=max_windows)
    else:
        project_root = Path.cwd() if project_root is None else Path(project_root)
        windows = load_dataset(config, project_root=project_root)
        normalization = meta["normalization"]
        test_windows_norm = transform_standard(windows, normalization["mean"], normalization["std"])
        test_windows_norm = test_windows_norm[:max_windows]
    return validate_onnx_outputs(
        model_path=artifacts_dir / "model.pt",
        onnx_path=artifacts_dir / "model.onnx",
        meta_path=artifacts_dir / "meta.json",
        test_windows_norm=test_windows_norm,
        report_path=artifacts_dir / "validation_report.json",
        max_abs_diff_tolerance=max_abs_diff_tolerance,
    )


def write_validation_report(report: dict[str, Any], report_path: Path | str) -> Path:
    """Write validation report as JSON for human review and CLI gatekeeping."""
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m header_ai_train.validate_onnx",
        description="Validate ONNX Runtime output against the PyTorch model.",
    )
    parser.add_argument("--artifacts-dir", default="artifacts", help="Directory containing model.pt, model.onnx, and meta.json.")
    parser.add_argument("--config", default=None, help="Optional config file for loading real validation windows.")
    parser.add_argument("--max-windows", type=int, default=16, help="Maximum windows used for validation.")
    parser.add_argument(
        "--max-abs-diff",
        type=float,
        default=DEFAULT_MAX_ABS_DIFF_TOLERANCE,
        help="Maximum allowed absolute output difference.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config) if args.config else None
    result = validate_from_artifacts(
        args.artifacts_dir,
        config=config,
        project_root=Path.cwd(),
        max_windows=args.max_windows,
        max_abs_diff_tolerance=args.max_abs_diff,
    )
    print(f"Saved validation report: {result.report_path}")
    print(f"max_abs_diff: {result.max_abs_diff:.8f}")
    print(f"pytorch_mse: {result.pytorch_mse:.8f}")
    print(f"onnx_mse: {result.onnx_mse:.8f}")


def _validate_session_io(session: ort.InferenceSession, *, input_name: str, output_name: str, input_dim: int) -> None:
    """Ensure ONNX Runtime session IO matches `meta.json`."""
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1:
        raise ValueError(f"Expected one ONNX Runtime input, got {len(inputs)}")
    if len(outputs) != 1:
        raise ValueError(f"Expected one ONNX Runtime output, got {len(outputs)}")
    if inputs[0].name != input_name:
        raise ValueError(f"ONNX Runtime input name {inputs[0].name!r} does not match meta input_name {input_name!r}")
    if outputs[0].name != output_name:
        raise ValueError(f"ONNX Runtime output name {outputs[0].name!r} does not match meta output_name {output_name!r}")
    if len(inputs[0].shape) != 2 or inputs[0].shape[1] != input_dim:
        raise ValueError(f"ONNX Runtime input shape must be [batch_size, {input_dim}], got {inputs[0].shape}")
    if len(outputs[0].shape) != 2 or outputs[0].shape[1] != input_dim:
        raise ValueError(f"ONNX Runtime output shape must be [batch_size, {input_dim}], got {outputs[0].shape}")


def _as_validation_windows(windows: np.ndarray, input_dim: int) -> np.ndarray:
    """Validate normalized windows before feeding PyTorch/ONNX Runtime."""
    windows = np.asarray(windows, dtype=np.float32)
    if windows.ndim != 2:
        raise ValueError(f"test_windows_norm must have shape [num_windows, input_dim], got {windows.shape}")
    if windows.shape[0] == 0:
        raise ValueError("test_windows_norm must contain at least one window")
    if windows.shape[1] != input_dim:
        raise ValueError(f"test_windows_norm input_dim mismatch: expected {input_dim}, got {windows.shape[1]}")
    if not np.isfinite(windows).all():
        raise ValueError("test_windows_norm contains NaN or infinite values")
    return np.ascontiguousarray(windows, dtype=np.float32)


def _make_deterministic_probe_windows(input_dim: int, max_windows: int) -> np.ndarray:
    """Create deterministic normalized probe windows when no dataset is provided."""
    if max_windows <= 0:
        raise ValueError(f"max_windows must be positive, got {max_windows}")
    rng = np.random.default_rng(42)
    return rng.normal(loc=0.0, scale=1.0, size=(max_windows, input_dim)).astype(np.float32)


if __name__ == "__main__":
    main()
