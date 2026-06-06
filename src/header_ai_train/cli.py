"""Command line entry point for header-ai-train."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from header_ai_train import __version__
from header_ai_train.export_onnx import export_from_artifacts, load_meta
from header_ai_train.train import load_config, run_training
from header_ai_train.validate_onnx import validate_from_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="header-ai-train",
        description="Train PyTorch AutoEncoder models and export ONNX artifacts.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"header-ai-train {__version__}",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the training configuration file.",
    )
    parser.add_argument(
        "--max-validation-windows",
        type=int,
        default=16,
        help="Maximum windows used for ONNX validation.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config_path = Path(args.config)
    project_root = Path.cwd()
    try:
        print(f"header-ai-train {__version__}")
        print(f"Config: {config_path}")
        config = _run_stage("load config", lambda: load_config(config_path))
        training_result = _run_stage("train model and write training artifacts", lambda: run_training(config, project_root=project_root))
        artifacts_dir = training_result.model_path.parent
        onnx_result = _run_stage("export onnx", lambda: export_from_artifacts(artifacts_dir))
        validation_result = _run_stage(
            "validate onnx",
            lambda: validate_from_artifacts(
                artifacts_dir,
                config=config,
                project_root=project_root,
                max_windows=args.max_validation_windows,
            ),
        )
        delivery_artifacts = _run_stage(
            "verify runtime delivery artifacts",
            lambda: _verify_runtime_delivery_artifacts(
                model_onnx_path=onnx_result.onnx_path,
                meta_path=training_result.meta_path,
                validation_report_path=validation_result.report_path,
            ),
        )
    except Exception as exc:
        raise SystemExit(f"header-ai-train failed: {exc}") from exc

    print("Pipeline completed.")
    print(f"model.pt: {training_result.model_path}")
    print(f"model.onnx: {delivery_artifacts['model.onnx']}")
    print(f"meta.json: {delivery_artifacts['meta.json']}")
    print(f"metrics.json: {training_result.metrics_path}")
    print(f"validation_report.json: {validation_result.report_path}")


def _run_stage(stage: str, action):
    print(f"[{stage}] start")
    try:
        result = action()
    except Exception as exc:
        raise RuntimeError(f"{stage}: {exc}") from exc
    print(f"[{stage}] done")
    return result


def _verify_runtime_delivery_artifacts(
    *,
    model_onnx_path: Path,
    meta_path: Path,
    validation_report_path: Path,
) -> dict[str, Path]:
    if not model_onnx_path.is_file():
        raise FileNotFoundError(f"runtime delivery artifact missing: {model_onnx_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"runtime delivery artifact missing: {meta_path}")
    if not validation_report_path.is_file():
        raise FileNotFoundError(f"validation report missing: {validation_report_path}")

    load_meta(meta_path)
    with validation_report_path.open("r", encoding="utf-8") as file:
        report = json.load(file)
    if report.get("status") != "passed":
        raise ValueError(f"validation_report.json status must be passed, got {report.get('status')!r}")

    return {
        "model.onnx": model_onnx_path,
        "meta.json": meta_path,
    }


if __name__ == "__main__":
    main()
