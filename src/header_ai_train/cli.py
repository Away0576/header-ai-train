"""Command line entry point for the end-to-end training pipeline.

`header-ai-train --config configs/default.yaml` 会串起整个训练侧流程：
load config -> train -> write meta -> export onnx -> validate onnx -> verify artifacts

这个文件只做流程编排，具体训练、导出、验证逻辑分别在 train/export_onnx/
validate_onnx 模块中实现。
"""

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
        # 训练阶段会生成 model.pt、metrics.json 和 meta.json。
        training_result = _run_stage("train model and write training artifacts", lambda: run_training(config, project_root=project_root))
        artifacts_dir = training_result.model_path.parent
        # ONNX 导出阶段读取 model.pt + meta.json，确保导出模型的输入输出名
        # 与 runtime 合同一致。
        onnx_result = _run_stage("export onnx", lambda: export_from_artifacts(artifacts_dir))
        # ONNX Runtime 验证阶段比较 PyTorch 输出和 ONNX 输出，防止导出后模型失真。
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
    """Run one pipeline stage and wrap failures with the stage name.

    这样用户看到错误时能知道是“加载配置”“训练”“导出”还是“验证”失败。
    """
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
    """Verify the two files that runtime really needs are usable.

    runtime 只需要 model.onnx + meta.json。这里额外要求 validation_report.json
    状态为 passed，避免把未验证的 ONNX 模型交付给 C++ 工程。
    """
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
