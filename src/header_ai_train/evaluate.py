"""Evaluate trained ONNX artifacts on a test time series.

训练完成后，本模块用于“拿另一份时间序列试跑模型”：
1. 读取 TXT/CSV 测试序列；
2. 按 meta.json 的 window_size 切窗口；
3. 用 meta.json 的 mean/std 归一化；
4. 用 ONNX Runtime 推理；
5. 计算每个窗口的 MSE；
6. 输出逐窗口 CSV，标记 `mse > threshold` 的窗口为异常。

当前版本不计算 precision/recall/F1，因为输入数据没有统一标签格式。
"""

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
    """Summary of one evaluation run."""
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
    """Evaluate a TXT/CSV series and write per-window anomaly scores.

    这个函数不重新训练模型，只消费 runtime 交付物 `model.onnx + meta.json`。
    因此它可以提前模拟 runtime 侧的核心推理逻辑。
    """
    artifacts_dir = Path(artifacts_dir)
    meta = load_meta(artifacts_dir / "meta.json")
    input_format = input_format.lower()
    if input_format == "txt":
        series = load_txt_series(input_path)
    elif input_format == "csv":
        series = load_csv_series(input_path, value_column=value_column)
    else:
        raise ValueError(f"Unsupported input_format {input_format!r}; expected txt or csv")

    # 评估时使用训练时写入 meta.json 的窗口长度，避免人工配置不一致。
    windows = make_windows(series, window_size=int(meta["window_size"]), stride=1)
    # 必须使用训练阶段保存的 mean/std；不能重新 fit 测试数据。
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

    # 这里固定 CPU provider，和后续嵌入式 Linux runtime 的默认路径保持一致。
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
    """Write per-window scores and anomaly decisions to CSV.

    start_index/end_index 表示该窗口在原始序列中的覆盖范围，便于后续把异常窗口
    映射回原始时间轴。
    """
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
