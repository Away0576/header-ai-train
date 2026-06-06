"""Command line entry point for header-ai-train."""

from __future__ import annotations

import argparse

from header_ai_train import __version__


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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    print(f"header-ai-train {__version__}")
    print(f"Config: {args.config}")
    print("v0.2.0 supports single-variable TXT/CSV loading and sliding-window dataset creation.")


if __name__ == "__main__":
    main()
