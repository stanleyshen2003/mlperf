#!/usr/bin/env python3
"""
Export Vision Transformer (ViT) to ONNX for Triton Inference Server.
Target size 3-8GB: uses a ViT-G-style config (~1.2B params, ~4.9GB) with random
initialization (no pretrained weights). Optionally export a pretrained ViT (e.g. vit-huge).
Output: serving/model-repo3/vit_onnx/1/model.onnx
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export ViT to ONNX for Triton")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "model-repo3" / "vit_onnx" / "1",
        help="Directory to write model.onnx (default: serving/model-repo3/vit_onnx/1)",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="large",
        choices=["large", "huge", "giant"],
        help="large=~4.9GB/1.2B params (ViT-G-like), huge=pretrained ~2.5GB, giant=~7GB/1.8B",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=21,
        help="ONNX opset version (default: 21)",
    )
    args = parser.parse_args()

    try:
        from transformers import ViTConfig, ViTForImageClassification
    except ImportError:
        print("Install transformers: uv pip install transformers torch", file=sys.stderr)
        sys.exit(1)

    try:
        from optimum.exporters.onnx import main_export
    except ImportError:
        print("Install optimum: uv pip install 'optimum[onnx]' onnx onnxruntime", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.size == "huge":
        # Pretrained ViT-huge ~2.5GB (largest standard pretrained ViT on HF)
        model_id = "google/vit-huge-patch14-224-in21k"
        print(f"Exporting pretrained {model_id} to ONNX...")
        main_export(
            model_name_or_path=model_id,
            output=args.output_dir,
            task="image-classification",
            opset=args.opset,
        )
    else:
        # Custom ViT-G-style config: 3-8GB with random weights
        if args.size == "giant":
            config = ViTConfig(
                image_size=224,
                patch_size=14,
                num_hidden_layers=56,
                hidden_size=1408,
                num_attention_heads=16,
                intermediate_size=6144,
                num_labels=1000,
            )
        else:
            # large: ~1.2B params, ~4.9GB
            config = ViTConfig(
                image_size=224,
                patch_size=14,
                num_hidden_layers=48,
                hidden_size=1408,
                num_attention_heads=16,
                intermediate_size=6144,
                num_labels=1000,
            )
        model = ViTForImageClassification._from_config(config)
        n = sum(p.numel() for p in model.parameters())
        print(f"Built ViT with {n / 1e9:.2f}B params (~{n * 4 / 1e9:.2f} GB fp32)")

        with tempfile.TemporaryDirectory(prefix="vit_export_") as tmp:
            tmp_path = Path(tmp)
            model.save_pretrained(tmp_path)
            print(f"Exporting to ONNX (task=image-classification, opset={args.opset})...")
            main_export(
                model_name_or_path=str(tmp_path),
                output=args.output_dir,
                task="image-classification",
                opset=args.opset,
            )

    if (args.output_dir / "model.onnx").exists():
        print(f"Saved to {args.output_dir / 'model.onnx'}")
    else:
        print("Export finished; check output dir for .onnx files.", file=sys.stderr)
        sys.exit(1)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
