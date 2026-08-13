#!/usr/bin/env python3
"""
auto_label_roboflow.py — YOLO pre-annotation + upload to the Roboflow project
================================================================================
Two phases, because they need different interpreters:

Phase 1 — ``scan`` (run with the MAIN venv: ``.venv/bin/python``)
    Runs the local YOLOv26 engine over image folders, writes a YOLO-format
    .txt per image (classes: 0=fire, 1=smoke) into ``--out``, and records a
    ``manifest.json`` of (image, annotation) pairs. Images with no boxes at
    ``--conf`` are skipped (they need human annotation, or are negatives).

Phase 2 — ``upload`` (run with the RF venv: ``.venv_rf/bin/python``)
    Uploads each manifest pair to the Roboflow project using the official
    ``roboflow`` SDK (needs Python >=3.10). The annotation labelmap is pinned
    to ``["fire", "smoke"]`` so YOLO class ids map onto the project classes —
    WITHOUT this the boxes land as a phantom class named "0".

Examples
--------
    .venv/bin/python auto_label_roboflow.py scan \\
        --dir ~/Downloads/fire_dataset --conf 0.40 --out auto_labels

    .venv_rf/bin/python auto_label_roboflow.py upload \\
        --manifest auto_labels/manifest.json --batch yolo-v26-auto --split train

    .venv_rf/bin/python auto_label_roboflow.py upload ... --limit 50  # dry-run-ish batch

    .venv_rf/bin/python auto_label_roboflow.py audit   # verify the project

Phase 3 -- ``audit`` (.venv_rf)
    Fetches every stored annotation and flags anything broken: labels other
    than fire/smoke (e.g. the phantom "object" class), boxes out of image
    bounds, or records that can't be fetched. Run this after uploads to
    confirm the dataset is clean before training.

Repair notes (learned the hard way)
-----------------------------------
* Out-of-range boxes (w/h > 1.0 after un-letterboxing roundoff) are mangled
  by Roboflow into a phantom "object" class instead of being rejected.
  scan() now clamps boxes to [0,1]; re-uploading a corrupted image must use
  ``annotation_overwrite=True`` (duplicate uploads only ADD missing
  annotations, they never replace bad ones), and if the record was deleted
  the content-hash dedup traps re-uploads against a dead record — re-encode
  the image (new bytes = new hash = fresh record).

Notes
-----
* The YOLO weights are gitignored; uploads land in the project's existing
  "Fire-and-Smoke-Segmentation" project (object-detection, fire/smoke).
* Roboflow renames uploaded files (dots -> dashes); duplicates are skipped
  server-side and counted as such in the summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".bmp"}

# YOLOv26 class ids -> Roboflow project classes (labelmap order matters!).
_LABELMAP = ["fire", "smoke"]
_YOLO_CLASS_TO_ID = {"fire": 0, "smoke": 1}

_WS = "juless-workspace-zidwe"
_PROJ = "fire-and-smoke-segmentation-ai4o1-ivg3r"


# ---------------------------------------------------------------------------
# Phase 1: scan (main venv)
# ---------------------------------------------------------------------------


def _image_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(
        p for p in root.iterdir()
        if p.suffix.lower() in _IMG_EXTS and not p.name.startswith(".")
    )


def _scan_cmd(args: argparse.Namespace) -> int:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
    from fire_vision import _resolve_engine, _yolo_model_path, scan_photo

    if _resolve_engine() != "yolo" or _yolo_model_path() is None:
        print("❌ Local YOLO engine not active — put models/best.onnx or best.pt "
              "in models/ (or set WILDFRAME_VISION_ENGINE=yolo).")
        return 2
    print(f"engine: YOLO — {_yolo_model_path()} | conf {args.conf:.2f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    roots = [Path(d).expanduser() for d in args.dir]
    images = [f for r in roots for f in _image_files(r)]
    print(f"scanning {len(images)} images…", flush=True)

    manifest: list[dict] = []
    skipped: list[str] = []
    n_boxes = {"fire": 0, "smoke": 0}
    t0 = time.time()
    done = 0

    def _run(p: Path):
        r = scan_photo(p, confidence_threshold=args.conf)
        if r.get("error"):
            return p, None, r["error"]
        dets = [
            d for d in r.get("detections", [])
            if d.get("class") in _YOLO_CLASS_TO_ID and d.get("confidence", 0) >= args.conf
        ]
        return p, dets, None

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_run, p) for p in images]
        for fut in as_completed(futs):
            p, dets, err = fut.result()
            done += 1
            if err or not dets:
                skipped.append(str(p))
                continue
            from PIL import Image, ImageOps

            # fire_vision applies exif_transpose before inference, so boxes are
            # in the ORIENTED frame — normalize against the oriented size or
            # any EXIF-rotated photo (phone uploads) gets misaligned boxes.
            with Image.open(p) as im:
                w, h = ImageOps.exif_transpose(im).size
            lines = []
            for d in dets:
                cls_id = _YOLO_CLASS_TO_ID[d["class"]]
                n_boxes[d["class"]] += 1
                x1, y1, x2, y2 = d["bbox"]
                # Un-letterboxing roundoff can push boxes a hair past the edge
                # (Roboflow rejects w/h > 1.0) — clamp the pixel box first.
                x1 = max(0.0, min(float(x1), w))
                y1 = max(0.0, min(float(y1), h))
                x2 = max(0.0, min(float(x2), w))
                y2 = max(0.0, min(float(y2), h))
                if x2 <= x1 or y2 <= y1:
                    continue  # degenerate after clamping — skip this box
                xc = (x1 + x2) / 2 / w
                yc = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            txt_path = out_dir / (p.stem + ".txt")
            txt_path.write_text("\n".join(lines) + "\n")
            manifest.append({"image": str(p), "txt": str(txt_path), "name": p.name})
            if done % 50 == 0 or done == len(images):
                rate = done / max(time.time() - t0, 0.01)
                print(f"  [{done}/{len(images)}] {rate:.1f} img/s "
                      f"(ETA {(len(images) - done) / rate:.0f}s)", flush=True)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f"\nannotated: {len(manifest)} images "
          f"(boxes: fire={n_boxes['fire']}, smoke={n_boxes['smoke']})")
    print(f"skipped (no boxes @ {args.conf:.2f}): {len(skipped)}")
    print(f"manifest: {manifest_path}")
    return 0


# ---------------------------------------------------------------------------
# Phase 2: upload (.venv_rf — needs Python >=3.10 + the roboflow package)
# ---------------------------------------------------------------------------


def _upload_cmd(args: argparse.Namespace) -> int:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
    import roboflow

    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        print("❌ ROBOFLOW_API_KEY not set")
        return 2

    manifest = json.loads(Path(args.manifest).read_text())
    if args.limit > 0:
        manifest = manifest[: args.limit]
    print(f"uploading {len(manifest)} images to {_WS}/{_PROJ} "
          f"(batch={args.batch}, split={args.split}, labelmap={_LABELMAP})")

    if args.dry_run:
        print("dry-run: nothing uploaded")
        return 0

    rf = roboflow.Roboflow(api_key=key)
    ws = rf.workspace(_WS)
    proj = ws.project(_PROJ)

    uploaded = duplicated = errored = 0
    errors: list[str] = []
    t0 = time.time()
    for i, item in enumerate(manifest, 1):
        try:
            res = proj.single_upload(
                image_path=item["image"],
                annotation_path=item["txt"],
                annotation_labelmap=_LABELMAP,
                split=args.split,
                batch_name=args.batch,
                annotation_overwrite=args.overwrite,
                num_retry_uploads=args.retries,
            )
            img_resp = (res or {}).get("image") or {}
            if img_resp.get("duplicate"):
                duplicated += 1
            # The SDK returns annotation: None when the labelmap/annotation
            # failed to attach — count it as an error so it can't silently
            # upload boxes under a phantom class (seen with the old SDK path).
            # NOTE: this field is best-effort; the `audit` subcommand is the
            # source of truth for what actually landed on the server.
            if not (res or {}).get("annotation"):
                errored += 1
                errors.append(f"{item['name']}: annotation did not attach")
            elif not img_resp.get("duplicate"):
                uploaded += 1
        except Exception as exc:  # one bad image must not kill the batch
            errored += 1
            errors.append(f"{item['name']}: {str(exc)[:120]}")
        if i % 25 == 0 or i == len(manifest):
            rate = i / max(time.time() - t0, 0.01)
            print(f"  [{i}/{len(manifest)}] {rate:.1f}/s · "
                  f"up {uploaded} dup {duplicated} err {errored}", flush=True)

    print(f"\nsummary: uploaded={uploaded} duplicate={duplicated} "
          f"errored={errored} in {time.time() - t0:.0f}s")
    for e in errors[:10]:
        print("  ERR", e)
    if errors:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Phase 3: audit (.venv_rf) — verify annotation integrity of the project
# ---------------------------------------------------------------------------


def _audit_cmd(args: argparse.Namespace) -> int:
    """Fetch every image's stored annotation and flag anything broken.

    Boxes arrive in PIXEL coordinates relative to the annotation's image
    dims — a box is bad if its label isn't fire/smoke or it extends past
    the image bounds (or the record can't be fetched at all).
    """
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
    import roboflow

    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        print("❌ ROBOFLOW_API_KEY not set")
        return 2

    rf = roboflow.Roboflow(api_key=key)
    ws = rf.workspace(_WS)
    proj = ws.project(_PROJ)

    def _f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return -1.0

    # 1) all images -> name -> id (paginated)
    ids: dict[str, str] = {}
    token = None
    while True:
        kwargs = dict(query="*", page_size=500, fields=["id", "name"])
        if token:
            kwargs["continuation_token"] = token
        res = ws.search(**kwargs)
        for r in res.get("results") or []:
            if r.get("name") and r.get("id"):
                ids[r["name"]] = r["id"]
        token = res.get("continuationToken")
        if not token:
            break
    print(f"total images in index: {len(ids)}")

    broken: list[tuple] = []
    clean_fire = clean_smoke = 0
    for name, img_id in sorted(ids.items()):
        try:
            img = proj.image(img_id)
        except Exception as exc:
            # Roboflow's API is flaky under load — retry once before flagging.
            try:
                img = proj.image(img_id)
            except Exception as exc2:
                if args.ignore_fetch_errs:
                    continue
                broken.append((name, img_id, f"fetch-err: {str(exc2)[:60]}"))
                continue
        time.sleep(0.02)  # gentle pacing to dodge rate limits
        ann = img.get("annotation")
        if not isinstance(ann, dict) or not ann.get("boxes"):
            broken.append((name, img_id, "no-annotation"))
            continue
        iw = _f(ann.get("width")) or 1.0
        ih = _f(ann.get("height")) or 1.0
        bad_label = [b for b in ann["boxes"] if b.get("label") not in ("fire", "smoke")]
        bad_range = [
            b for b in ann["boxes"]
            if not (0 <= _f(b.get("x")) < iw and 0 <= _f(b.get("y")) < ih
                    and 0 < _f(b.get("width")) <= iw
                    and 0 < _f(b.get("height")) <= ih)
        ]
        if bad_label:
            labels = sorted({b.get("label") for b in ann["boxes"]})
            broken.append((name, img_id, f"labels={labels}"))
        elif bad_range:
            broken.append((name, img_id, "out-of-bounds-box"))
        else:
            for b in ann["boxes"]:
                if b.get("label") == "fire":
                    clean_fire += 1
                else:
                    clean_smoke += 1

    print(f"checked annotations: {len(ids) - len(broken)}")
    print(f"clean boxes: fire={clean_fire} smoke={clean_smoke}")
    print(f"BROKEN images: {len(broken)}")
    for name, img_id, why in broken:
        print(f"  {name} | {img_id} | {why}")
    return 1 if broken else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="YOLO auto-label + upload to Roboflow (two-phase).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_scan = sub.add_parser("scan", help="Phase 1: YOLO -> YOLO txt + manifest (main venv)")
    p_scan.add_argument("--dir", action="append", default=["~/Downloads/fire_dataset"],
                        help="Image folder(s); parent folder naming is not required.")
    p_scan.add_argument("--conf", type=float, default=0.40, help="Min box confidence")
    p_scan.add_argument("--out", default="auto_labels", help="Output dir for txt + manifest")
    p_scan.add_argument("--workers", type=int, default=4)
    p_scan.set_defaults(func=_scan_cmd)

    p_up = sub.add_parser("upload", help="Phase 2: upload to Roboflow (.venv_rf)")
    p_up.add_argument("--manifest", default="auto_labels/manifest.json")
    p_up.add_argument("--split", default="train")
    p_up.add_argument("--batch", default="yolo-v26-auto")
    p_up.add_argument("--limit", type=int, default=0, help="0 = all")
    p_up.add_argument("--retries", type=int, default=2)
    p_up.add_argument("--overwrite", action="store_true",
                      help="Pass annotation_overwrite=True to replace existing "
                           "annotations (needed to fix corrupt ones)")
    p_up.add_argument("--dry-run", action="store_true")
    p_up.set_defaults(func=_upload_cmd)

    p_audit = sub.add_parser(
        "audit",
        help="Phase 3: verify every stored annotation (fire/smoke, in bounds)",
    )
    p_audit.add_argument("--ignore-fetch-errs", action="store_true",
                         help="Don't fail on records that can't be fetched "
                              "(e.g. a stale deleted index tombstone)")
    p_audit.set_defaults(func=_audit_cmd)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
