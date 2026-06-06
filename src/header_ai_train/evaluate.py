"""Evaluate trained ONNX artifacts on a test time series."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from header_ai_train.dataset import load_csv_series, load_txt_series, make_windows, transform_standard
from header_ai_train.export_onnx import load_meta


@dataclass(frozen=True)
class EvaluationResult:
    report_path: Path
    num_windows: int
    anomaly_count: int
    threshold: float
    max_score: float
    mean_score: float


def evaluate_series_file(
    *,
    artifacts_dir: Path | str,
    input_path: Path | str,
    input_format: str,
    value_column: str,
    report_path: Path | str,
) -> EvaluationResult:
    """Evaluate a TXT/CSV series and write per-window anomaly scores."""
    artifacts_dir = Path(artifacts_dir)
    meta = load_meta(artifacts_dir / "meta.json")
    input_format = input_format.lower()
    if input_format == "txt":
        series = load_txt_series(input_path)
    elif input_format == "csv":
        series = load_csv_series(input_path, value_column=value_column)
    else:
        raise ValueError(f"Unsupported input_format {input_format!r}; expected txt or csv")

    windows = make_windows(series, window_size=int(meta["window_size"]), stride=1)
    windows_norm = transform_standard(windows, meta["normalization"]["mean"], meta["normalization"]["std"])
    scores = compute_onnx_reconstruction_errors(
        onnx_path=artifacts_dir / "model.onnx",
        meta=meta,
        windows_norm=windows_norm,
    )
    threshold = float(meta["threshold"])
    is_anomaly = scores > threshold
    report_path = write_evaluation_report(
        report_path=report_path,
        scores=scores,
        threshold=threshold,
        window_size=int(meta["window_size"]),
        stride=1,
    )
    return EvaluationResult(
        report_path=report_path,
        num_windows=int(len(scores)),
        anomaly_count=int(np.count_nonzero(is_anomaly)),
        threshold=threshold,
        max_score=float(np.max(scores)),
        mean_score=float(np.mean(scores, dtype=np.float64)),
    )


def compute_onnx_reconstruction_errors(
    *,
    onnx_path: Path | str,
    meta: dict[str, Any],
    windows_norm: np.ndarray,
) -> np.ndarray:
    """Compute per-window reconstruction MSE using ONNX Runtime."""
    windows_norm = np.asarray(windows_norm, dtype=np.float32)
    if windows_norm.ndim != 2:
        raise ValueError(f"windows_norm must have shape [num_windows, input_dim], got {windows_norm.shape}")
    if windows_norm.shape[1] != int(meta["input_dim"]):
        raise ValueError(f"windows_norm input_dim mismatch: expected {meta['input_dim']}, got {windows_norm.shape[1]}")
    if len(windows_norm) == 0:
        raise ValueError("windows_norm must contain at least one window")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = str(meta["input_name"])
    output_name = str(meta["output_name"])
    reconstruction = session.run([output_name], {input_name: windows_norm})[0]
    scores = np.mean((reconstruction - windows_norm) ** 2, axis=1).astype(np.float32)
    if not np.isfinite(scores).all():
        raise ValueError("Evaluation scores contain NaN or infinite values")
    return scores


def write_evaluation_report(
    *,
    report_path: Path | str,
    scores: np.ndarray,
    threshold: float,
    window_size: int,
    stride: int,
) -> Path:
    """Write per-window scores and anomaly decisions to CSV."""
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["window_index", "start_index", "end_index", "mse", "threshold", "is_anomaly"],
        )
        writer.writeheader()
        for index, score in enumerate(scores):
            start_index = index * stride
            writer.writerow(
                {
                    "window_index": index,
                    "start_index": start_index,
                    "end_index": start_index + window_size - 1,
                    "mse": float(score),
                    "threshold": float(threshold),
                    "is_anomaly": bool(score > threshold),
                }
            )
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m header_ai_train.evaluate",
        description="Evaluate trained ONNX artifacts on a single-variable test series.",
    )
    parser.add_argument("--artifacts-dir", default="artifacts", help="Directory containing model.onnx and meta.json.")
    parser.add_argument("--input-path", required=True, help="Path to test TXT/CSV data.")
    parser.add_argument("--input-format", choices=["txt", "csv"], default="csv", help="Test data format.")
    parser.add_argument("--value-column", default="value", help="CSV value column name.")
    parser.add_argument("--output-csv", default="artifacts/evaluation_windows.csv", help="Output per-window CSV report.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = evaluate_series_file(
        artifacts_dir=args.artifacts_dir,
        input_path=args.input_path,
        input_format=args.input_format,
        value_column=args.value_column,
        report_path=args.output_csv,
    )
    print(f"Saved evaluation report: {result.report_path}")
    print(f"windows: {result.num_windows}")
    print(f"anomaly_windows: {result.anomaly_count}")
    print(f"threshold: {result.threshold:.8f}")
    print(f"max_score: {result.max_score:.8f}")
    print(f"mean_score: {result.mean_score:.8f}")


if __name__ == "__main__":
    main()
