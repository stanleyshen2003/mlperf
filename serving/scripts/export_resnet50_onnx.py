#!/usr/bin/env python3
"""
Export ResNet-50 to ONNX for Triton Inference Server.
Uses torchvision ResNet-50 (ImageNet, ~25M params).
Output: serving/model-repo4/resnet50_onnx/1/model.onnx
"""
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export ResNet-50 to ONNX for Triton")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "model-repo4" / "resnet50_onnx" / "1",
        help="Directory to write model.onnx (default: serving/model-repo4/resnet50_onnx/1)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=21,
        help="ONNX opset version (default: 21)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="IMAGENET1K_V2",
        choices=["IMAGENET1K_V1", "IMAGENET1K_V2", "none"],
        help="Pretrained weights: IMAGENET1K_V1, IMAGENET1K_V2, or none",
    )
    args = parser.parse_args()

    try:
        import torch
        import torchvision.models as models
    except ImportError:
        print("Install torch and torchvision: uv pip install torch torchvision", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "model.onnx"

    print(f"Loading ResNet-50 (weights={args.weights})...")
    if args.weights == "none":
        model = models.resnet50(weights=None)
    else:
        weights_enum = getattr(models.ResNet50_Weights, args.weights, models.ResNet50_Weights.IMAGENET1K_V2)
        model = models.resnet50(weights=weights_enum)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}
    print(f"Exporting to ONNX (opset={args.opset})...")
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
    )
    print(f"Saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
