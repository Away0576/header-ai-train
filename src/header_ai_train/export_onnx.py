"""ONNX export utilities."""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnx
import torch
from torch import nn

from header_ai_train.model import AutoEncoder
from header_ai_train.train import load_model_checkpoint, validate_meta


@dataclass(frozen=True)
class OnnxExportResult:
    onnx_path: Path
    input_name: str
    output_name: str
    input_dim: int
    opset: int


def load_meta(meta_path: Path | str) -> dict[str, Any]:
    meta_path = Path(meta_path)
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.json not found: {meta_path}")
    with meta_path.open("r", encoding="utf-8") as file:
        meta = json.load(file)
    if not isinstance(meta, dict):
        raise ValueError(f"meta.json must contain a JSON object: {meta_path}")
    validate_meta(meta)
    return meta


def export_model_to_onnx(
    model_path: Path | str,
    meta_path: Path | str,
    onnx_path: Path | str,
) -> OnnxExportResult:
    """Export a PyTorch AutoEncoder checkpoint to ONNX."""
    meta = load_meta(meta_path)
    model, checkpoint = load_model_checkpoint(model_path)
    input_dim = int(meta["input_dim"])
    if model.input_dim != input_dim:
        raise ValueError(f"Model input_dim {model.input_dim} does not match meta input_dim {input_dim}")
    if int(checkpoint["input_dim"]) != input_dim:
        raise ValueError("Checkpoint input_dim does not match meta input_dim")

    input_name = str(meta["input_name"])
    output_name = str(meta["output_name"])
    opset = int(meta["onnx"]["opset"])
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    export_model = _AutoEncoderOnnxWrapper(model)
    dummy_input = torch.zeros(1, input_dim, dtype=torch.float32)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="You are using the legacy TorchScript-based ONNX export.*",
            category=DeprecationWarning,
        )
        torch.onnx.export(
            export_model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=opset,
            dynamo=False,
            do_constant_folding=True,
            input_names=[input_name],
            output_names=[output_name],
            dynamic_axes={
                input_name: {0: "batch_size"},
                output_name: {0: "batch_size"},
            },
        )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    _validate_onnx_io(onnx_model, input_name=input_name, output_name=output_name, input_dim=input_dim)
    return OnnxExportResult(
        onnx_path=onnx_path,
        input_name=input_name,
        output_name=output_name,
        input_dim=input_dim,
        opset=opset,
    )


def export_from_artifacts(artifacts_dir: Path | str) -> OnnxExportResult:
    artifacts_dir = Path(artifacts_dir)
    return export_model_to_onnx(
        model_path=artifacts_dir / "model.pt",
        meta_path=artifacts_dir / "meta.json",
        onnx_path=artifacts_dir / "model.onnx",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m header_ai_train.export_onnx",
        description="Export artifacts/model.pt to artifacts/model.onnx using artifacts/meta.json.",
    )
    parser.add_argument("--artifacts-dir", default="artifacts", help="Directory containing model.pt and meta.json.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = export_from_artifacts(args.artifacts_dir)
    print(f"Saved ONNX model: {result.onnx_path}")
    print(f"Input: {result.input_name} [batch_size, {result.input_dim}]")
    print(f"Output: {result.output_name} [batch_size, {result.input_dim}]")
    print(f"Opset: {result.opset}")


def _validate_onnx_io(onnx_model: onnx.ModelProto, *, input_name: str, output_name: str, input_dim: int) -> None:
    graph = onnx_model.graph
    if len(graph.input) != 1:
        raise ValueError(f"Expected one ONNX input, got {len(graph.input)}")
    if len(graph.output) != 1:
        raise ValueError(f"Expected one ONNX output, got {len(graph.output)}")

    actual_input = graph.input[0]
    actual_output = graph.output[0]
    if actual_input.name != input_name:
        raise ValueError(f"ONNX input name {actual_input.name!r} does not match meta input_name {input_name!r}")
    if actual_output.name != output_name:
        raise ValueError(f"ONNX output name {actual_output.name!r} does not match meta output_name {output_name!r}")

    input_shape = _value_info_shape(actual_input)
    output_shape = _value_info_shape(actual_output)
    if len(input_shape) != 2 or input_shape[1] != input_dim:
        raise ValueError(f"ONNX input shape must be [batch_size, {input_dim}], got {input_shape}")
    if len(output_shape) != 2 or output_shape[1] != input_dim:
        raise ValueError(f"ONNX output shape must be [batch_size, {input_dim}], got {output_shape}")


def _value_info_shape(value_info: onnx.ValueInfoProto) -> list[int | str]:
    shape = value_info.type.tensor_type.shape
    dims: list[int | str] = []
    for dim in shape.dim:
        if dim.dim_param:
            dims.append(dim.dim_param)
        elif dim.dim_value:
            dims.append(int(dim.dim_value))
        else:
            dims.append("?")
    return dims


class _AutoEncoderOnnxWrapper(nn.Module):
    """Export wrapper that avoids Python shape checks during ONNX tracing."""

    def __init__(self, model: AutoEncoder) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model.decoder(self.model.encoder(inputs))


if __name__ == "__main__":
    main()
