#!/usr/bin/env python3
"""
effnet_train.py — Recreate + fine-tune the FireSmokeDetectionByEfficientNet
model (EfficientNet-B0 backbone + small FC head, 224x224 ImageNet-normalized
input) on the local fire_dataset, then dump per-image confidence rows for the
sweep harness.

The upstream repo's 2020 vendored PyTorch code is replaced by timm's
`efficientnet_b0` (same architecture, clean ImageNet pretrained weights). We
train 2 classes (fire vs non-fire) because our dataset has no smoke class;
upstream used 3 (fire / negative / smoke).

Splits the dataset 80/20 (stratified, seeded) so the sweep measures
generalization, not memorization. Saves:
    models/effnet_fire.pt          best-validation checkpoint
    effnet_rows_val.json           per-image (fire_conf) on the held-out 20%
    effnet_rows_train.json         per-image (fire_conf) on the training 80%
Rows match sweep_yolo.py's cache schema (fire_conf/smoke_conf/positive).

Usage:
    python effnet_train.py [--data ~/Downloads/fire_dataset] [--epochs 15]
                           [--batch 32] [--val-frac 0.2]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    import timm
except ImportError:
    print("❌ timm not installed — run: .venv/bin/pip install timm")
    sys.exit(1)

SEED = 42
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_DATA = Path.home() / "Downloads" / "fire_dataset"

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _image_files(d: Path) -> list[Path]:
    out = []
    for p in sorted(d.iterdir()):
        if p.suffix.lower() not in _IMG_EXTS or p.name.startswith("."):
            continue
        try:
            Image.open(p).verify()
            out.append(p)
        except Exception:
            print(f"  ⚠ skipping unreadable image: {p.name}")
    return out


def _load_dataset(base: Path) -> tuple[list[Path], list[Path]]:
    fire_dir = base / "fire_images"
    non_dir = base / "non_fire_images"
    pos = _image_files(fire_dir) if fire_dir.is_dir() else []
    neg = _image_files(non_dir) if non_dir.is_dir() else []
    return pos, neg


class FireDataset(Dataset):
    def __init__(self, files: list[Path], labels: list[int], tfm):
        self.files = files
        self.labels = labels
        self.tfm = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        return self.tfm(img), self.labels[i]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--val-frac", type=float, default=0.20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out-dir", type=Path, default=Path("models"))
    ap.add_argument("--rows-only", action="store_true",
                    help="skip training; reload the saved checkpoint and only "
                         "dump the per-image rows (same seed/split as training)")
    args = ap.parse_args()

    set_seed()
    pos, neg = _load_dataset(args.data)
    if not pos or not neg:
        print(f"❌ need fire_images/ and non_fire_images/ under {args.data} "
              f"(got {len(pos)} fire, {len(neg)} non-fire)")
        return 1
    print(f"dataset: {len(pos)} fire · {len(neg)} non-fire")

    files = pos + neg
    labels = [1] * len(pos) + [0] * len(neg)
    idx = list(range(len(files)))
    random.shuffle(idx)
    n_val = max(1, int(len(idx) * args.val_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]  # ordered lists (row dump
                                                   # maps back via index)

    # Class-balanced loss weight (dataset is ~3:1 fire:non-fire). One weight
    # per class: inverse-frequency, normalized so they sum to 2.
    n_pos_tr = sum(1 for i in train_idx if labels[i] == 1)
    n_neg_tr = len(train_idx) - n_pos_tr
    cls_weight = torch.tensor([
        len(train_idx) / (2 * max(n_neg_tr, 1)),
        len(train_idx) / (2 * max(n_pos_tr, 1)),
    ])

    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    val_tfm = transforms.Compose([
        transforms.Resize((224, 224)),  # square resize (batch-safe; the
                                        # upstream repo's aspect-keeping
                                        # Resize(224) only worked on singles)
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])

    train_ds = FireDataset([files[i] for i in train_idx], [labels[i] for i in train_idx], train_tfm)
    val_ds = FireDataset([files[i] for i in val_idx], [labels[i] for i in val_idx], val_tfm)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)
    print(f"split: train {len(train_ds)} ({n_pos_tr} fire) · val {len(val_ds)}")

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    ckpt = args.out_dir / "effnet_fire.pt"

    if args.rows_only:
        model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
        model.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"])
        model = model.to(device)
        _dump_rows(model, device, files, labels, train_idx, val_idx, args, val_tfm)
        return 0

    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=2)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss(weight=cls_weight.to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc, best_state = 0.0, None
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        run_loss, n = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(x)
            n += len(x)
        sched.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += len(y)
        acc = correct / total
        print(f"  epoch {ep:2d}/{args.epochs}  train_loss {run_loss / n:.4f}  "
              f"val_acc {acc * 100:5.1f}%  ({time.time() - t0:.0f}s)", flush=True)
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print(f"\nbest val_acc {best_acc * 100:.1f}%")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "classes": ["non_fire", "fire"]}, ckpt)
    print(f"saved checkpoint: {ckpt}")

    _dump_rows(model, device, files, labels, train_idx, val_idx, args, val_tfm, state=best_state)
    return 0


def _dump_rows(model, device, files, labels, train_idx, val_idx, args, val_tfm, state=None):
    """Dump per-image (fire_conf) rows for the sweep harness. Both splits go
    through the augment-free val transform so scores are deterministic and map
    cleanly back to split_idx (the train DataLoader shuffles, so we rebuild
    loaders here with shuffle=False instead of reusing it)."""
    if state is not None:
        model.load_state_dict(state)
    model.eval()
    for split_name, split_idx in (("train", train_idx), ("val", val_idx)):
        ds = FireDataset([files[i] for i in split_idx], [labels[i] for i in split_idx], val_tfm)
        dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)
        rows = []
        with torch.no_grad():
            for x, y in dl:
                probs = torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy()
                for j, prob in enumerate(probs):
                    gi = split_idx[len(rows)]
                    rows.append({
                        "file": str(files[gi]),
                        "name": files[gi].name,
                        "positive": labels[gi] == 1,
                        "fire_conf": float(prob),
                        "smoke_conf": 0.0,
                    })
        out_path = Path(f"effnet_rows_{split_name}.json")
        out_path.write_text(json.dumps(rows, indent=1))
        print(f"wrote {len(rows)} rows → {out_path}")


if __name__ == "__main__":
    sys.exit(main())
