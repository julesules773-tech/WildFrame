"""
Quick test script for the MobileNetV3 fire/smoke pre-filter.

Usage:
    python test_mobilenet_prefilter.py --weights model.pt --images path/to/folder
    python test_mobilenet_prefilter.py --weights model.pt --images single_image.jpg
    python test_mobilenet_prefilter.py --images path/to/folder   # no weights = sanity test w/ ImageNet head only

Notes:
- If you haven't fine-tuned yet, pass no --weights and the script will still run
  using raw ImageNet-pretrained MobileNetV3 with a randomly-initialized 2-class head.
  This is only useful to confirm the pipeline runs end-to-end (timing, image loading,
  etc.) — predictions will be meaningless until you fine-tune.
- Once you have a checkpoint, pass --weights to load your fine-tuned model.
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_NAMES = ["no_fire", "fire_smoke"]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_model(weights_path: str | None) -> nn.Module:
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(CLASS_NAMES))

    if weights_path:
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"Loaded fine-tuned weights from {weights_path}")
    else:
        print("No --weights given — running with an untrained head (sanity check only).")

    model.eval()
    return model


def gather_images(path_str: str) -> list[Path]:
    path = Path(path_str)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    raise FileNotFoundError(f"No such file or directory: {path_str}")


@torch.no_grad()
def predict(model: nn.Module, image_path: Path, threshold: float) -> dict:
    img = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0)

    start = time.perf_counter()
    logits = model(tensor)
    elapsed_ms = (time.perf_counter() - start) * 1000

    probs = torch.softmax(logits, dim=1)[0]
    fire_prob = probs[1].item()

    return {
        "file": image_path.name,
        "fire_prob": fire_prob,
        "pass_filter": fire_prob >= threshold,
        "inference_ms": elapsed_ms,
    }


def main():
    parser = argparse.ArgumentParser(description="Test the MobileNetV3 fire/smoke pre-filter")
    parser.add_argument("--weights", type=str, default=None, help="Path to fine-tuned .pt state_dict")
    parser.add_argument("--images", type=str, required=True, help="Image file or folder of images")
    parser.add_argument("--threshold", type=float, default=0.3, help="Pass threshold (default 0.3)")
    args = parser.parse_args()

    model = build_model(args.weights)
    images = gather_images(args.images)

    if not images:
        print("No images found.")
        return

    print(f"\nRunning inference on {len(images)} image(s), threshold={args.threshold}\n")
    print(f"{'file':<30} {'fire_prob':>10} {'pass?':>8} {'ms':>8}")
    print("-" * 60)

    total_ms = 0.0
    passed = 0
    for image_path in images:
        try:
            result = predict(model, image_path, args.threshold)
        except Exception as e:
            print(f"{image_path.name:<30}  ERROR: {e}")
            continue

        total_ms += result["inference_ms"]
        passed += result["pass_filter"]
        print(f"{result['file']:<30} {result['fire_prob']:>10.3f} "
              f"{str(result['pass_filter']):>8} {result['inference_ms']:>7.1f}")

    print("-" * 60)
    print(f"Passed filter: {passed}/{len(images)}")
    print(f"Avg inference time: {total_ms / len(images):.1f} ms/image")


if __name__ == "__main__":
    main()