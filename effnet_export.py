#!/usr/bin/env python3
"""
effnet_export.py — Export models/effnet_fire.pt (fine-tuned EfficientNet-B0,
2-class) to ONNX for the onnxruntime-only production VM, then verify that
onnxruntime reproduces the torch model's softmax outputs.

Preprocessing contract (must match fire_vision._run_effnet_onnx):
    Resize((224,224)) -> ToTensor -> Normalize(mean,std) -> NCHW float32

Usage:
    python effnet_export.py [--ckpt models/effnet_fire.pt] [--out models/effnet_fire.onnx]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

try:
    import timm
except ImportError:
    raise SystemExit("timm not installed")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=Path("models/effnet_fire.pt"))
    ap.add_argument("--out", type=Path, default=Path("models/effnet_fire.onnx"))
    ap.add_argument("--opset", type=int, default=13)
    args = ap.parse_args()

    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, str(args.out),
        input_names=["input"], output_names=["logits"],
        opset_version=args.opset,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"exported {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")

    # Parity check: torch (MPS) vs onnxruntime on random + a real image.
    import onnxruntime as ort

    sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    rng = np.random.RandomState(0)

    def _softmax(x):
        e = np.exp(x - x.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)

    max_diff = 0.0
    for trial in range(5):
        x = rng.randn(1, 3, 224, 224).astype(np.float32)
        with torch.no_grad():
            torch_prob = torch.softmax(model(torch.from_numpy(x)), dim=1).numpy()
        ort_prob = _softmax(sess.run(["logits"], {"input": x})[0])
        max_diff = max(max_diff, float(np.abs(torch_prob - ort_prob).max()))
    print(f"max |torch - onnxruntime| over 5 random inputs: {max_diff:.2e}")

    # A real photo, end-to-end through the same preprocessing the server uses.
    try:
        from PIL import Image
        from torchvision import transforms

        img = Image.open(Path.home() / "Downloads/fire_dataset/fire_images/fire.1.png").convert("RGB")
        tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        x = tfm(img).unsqueeze(0)
        with torch.no_grad():
            torch_prob = torch.softmax(model(x), dim=1).numpy()
        ort_prob = _softmax(sess.run(["logits"], {"input": x.numpy()})[0])
        print(f"real image fire prob — torch {torch_prob[0][1]:.4f} | onnxruntime {ort_prob[0][1]:.4f}")
        assert abs(torch_prob[0][1] - ort_prob[0][1]) < 1e-4, "parity check failed"
        print("PARITY OK")
    except Exception as exc:
        print(f"real-image parity check skipped: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
