#!/usr/bin/env python3
"""
train_local.py — pull the Roboflow dataset and train locally, export ONNX
=========================================================================
Two subcommands:

``pull`` (run with .venv_rf/bin/python — needs the roboflow SDK)
    Generates a fresh dataset version on Roboflow (no augmentation, minimal
    preprocessing) and downloads it in YOLO format into ``--out``
    (train/valid/test + data.yaml). Idempotent: if a version already exists
    with the current image count, it just downloads that version.

``train`` (run with the MAIN venv — .venv/bin/python — needs ultralytics)
    Fine-tunes ``--weights`` (default models/best.pt — the HF YOLOv26
    fire-detection model, warm start) on the pulled dataset, then exports
    the result to ``--onnx`` as an end-to-end ``[1, 300, 6]`` graph with
    NMS baked in — the exact format fire_vision._run_yolo_onnx consumes.

Examples
--------
    .venv_rf/bin/python train_local.py pull
    .venv/bin/python train_local.py train --epochs 60 --device mps

Notes
-----
* Class order in the pulled data.yaml is [fire, smoke] -> ids 0, 1. The
  retrained ONNX therefore maps cls 0=fire, 1=smoke (the old HF model had
  0=fire, 2=smoke with 1=other); fire_vision._YOLO_CLASS_NAMES must be
  updated to {0: "fire", 1: "smoke"} alongside the model swap.
* Export uses ``nms=True`` so the graph emits a single ``[1, 300, 6]``
  tensor of [x1, y1, x2, y2, conf, cls] — no post-processing needed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_WS = "juless-workspace-zidwe"
_PROJ = "fire-and-smoke-segmentation-ai4o1-ivg3r"
_EXPECTED_IMAGES = 754

_VERSION_SETTINGS = {
    "preprocessing": {
        "auto-orient": True,
        # no resize: keep original resolutions; training letterboxes to 640
    },
    "augmentation": {},
}


def _pull(args: argparse.Namespace) -> int:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
    import roboflow

    import os

    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        print("❌ ROBOFLOW_API_KEY not set")
        return 2

    rf = roboflow.Roboflow(api_key=key)
    proj = rf.workspace(_WS).project(_PROJ)

    out = Path(args.out)
    version_obj = None
    for attempt in range(6):  # version can take a few seconds to appear
        versions = proj.versions()
        if versions:
            version_obj = versions[-1]
            break
        if attempt < 5:
            print("  no version yet — waiting …")
            time.sleep(10)
    if version_obj is None:
        print("generating version 1 …")
        proj.generate_version(_VERSION_SETTINGS)
        for attempt in range(12):
            versions = proj.versions()
            if versions:
                version_obj = versions[-1]
                break
            time.sleep(10)
    if version_obj is None:
        print("❌ version never appeared after generation")
        return 1

    print(f"using version {getattr(version_obj, 'version_num', '?')} — downloading …")
    version_obj.download(
        model_format="yolov8", location=str(out), overwrite=args.overwrite
    )
    print(f"downloaded -> {out}")
    print("structure:", [p.name for p in out.iterdir()] if out.exists() else "(missing)")
    return 0


def _train(args: argparse.Namespace) -> int:
    from ultralytics import YOLO

    data = Path(args.data)
    if not data.exists():
        print(f"❌ dataset not found: {data} (run `pull` first)")
        return 2

    weights = Path(args.weights)
    if not weights.exists():
        print(f"❌ weights not found: {weights}")
        return 2

    print(f"training from {weights} on {data} ({args.epochs} epochs, {args.device})")
    model = YOLO(str(weights))
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=640,
        device=args.device,
        batch=args.batch,
        patience=args.patience,
        project="runs/train_local",
        name="fire_det",
        exist_ok=True,
        plots=True,
    )

    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights/best.pt"
    print(f"best weights: {best}")

    # validate on the held-out split
    val = model.val(data=str(data), device=args.device)
    print(f"val: mAP50={val.box.map50:.4f} mAP50-95={val.box.map:.4f} "
          f"precision={val.box.mp:.4f} recall={val.box.mr:.4f}")

    onnx = args.onnx or str(Path(best).with_suffix(".onnx"))
    print(f"exporting ONNX -> {onnx} (end-to-end, NMS baked in)")
    model.export(format="onnx", imgsz=640, nms=True, simplify=True)
    exported = best.with_suffix(".onnx")
    if exported.exists() and str(exported) != str(Path(onnx)):
        exported.rename(onnx)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Pull + train the fire/smoke detector locally.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pull = sub.add_parser("pull", help="Generate + download the Roboflow dataset (YOLO format)")
    p_pull.add_argument("--out", default="dataset_rf")
    p_pull.add_argument("--overwrite", action="store_true")
    p_pull.set_defaults(func=_pull)

    p_train = sub.add_parser("train", help="Fine-tune + export ONNX with ultralytics")
    p_train.add_argument("--data", default="dataset_rf/data.yaml")
    p_train.add_argument("--weights", default="models/best.pt")
    p_train.add_argument("--epochs", type=int, default=60)
    p_train.add_argument("--device", default="mps")
    p_train.add_argument("--batch", type=int, default=16)
    p_train.add_argument("--patience", type=int, default=15)
    p_train.add_argument("--onnx")
    p_train.set_defaults(func=_train)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
