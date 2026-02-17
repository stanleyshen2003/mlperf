#!/usr/bin/env python3
"""
Export BERT (bert-large-uncased, 336M params) to ONNX for Triton Inference Server.
Uses Hugging Face Optimum to download the model and export to ONNX.
Output: serving/model-repo2/bert_onnx/1/model.onnx
"""
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export BERT-large to ONNX for Triton")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "model-repo2" / "bert_onnx" / "1",
        help="Directory to write model.onnx (default: serving/model-repo2/bert_onnx/1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google-bert/bert-large-uncased",
        help="Hugging Face model id (default: bert-large-uncased, 336M params)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="default",
        help="Export task: default (encoder only), fill-mask, etc.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=21,
        help="ONNX opset version (default: 21)",
    )
    args = parser.parse_args()

    try:
        from optimum.exporters.onnx import main_export
    except ImportError:
        print(
            "Install optimum and deps: uv pip install 'optimum[onnx]' transformers torch",
            file=sys.stderr,
        )
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "model.onnx"

    print(f"Exporting {args.model} to ONNX (task={args.task}, opset={args.opset})...")
    main_export(
        model_name_or_path=args.model,
        output=args.output_dir,
        task=args.task,
        opset=args.opset,
    )

    # Optimum writes model.onnx into output dir
    if (args.output_dir / "model.onnx").exists():
        print(f"Saved to {args.output_dir / 'model.onnx'}")
    else:
        print("Export finished; check output dir for .onnx files.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
