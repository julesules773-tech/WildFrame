#!/usr/bin/env python3
"""
threshold_sweep.py — Find the best fire/smoke detection confidence threshold
============================================================================
Scans a folder of images ONCE with the model at a zero confidence threshold
(keeping every raw detection), then simulates the model's verdict at many
confidence thresholds locally. The full sweep therefore costs the same number
of API calls as a single backtest run.

Ground truth comes from the folder layout:
    fire_images/*.png      → positive (contains fire/smoke)
    non_fire_images/*.png  → negative (clean)

Usage
-----
    # Quick run: 100 balanced samples per class from the default dataset
    python threshold_sweep.py

    # Full run over both folders (755 fire + 246 non-fire)
    python threshold_sweep.py --limit 0

    # Point it at any folder(s) — each image is labeled by its parent dir name
    python threshold_sweep.py ../Downloads/fire_dataset/fire_images \
                              ../Downloads/fire_dataset/non_fire_images

    # Custom threshold grid + JSON export
    python threshold_sweep.py --thresholds "0.5,0.6,0.7,0.8,0.85,0.9,0.95" \
                              --json sweep.json

Notes
-----
* "Positive" verdict = flame, smoke, or both (matches how the server treats
  AI evidence when deciding to auto-approve).
* Images whose parent folder isn't recognizably "fire" or "non-fire" are
  scanned but excluded from the metrics (printed as "unlabeled").
* The API key is read from ROBOFLOW_API_KEY (e.g. your .env file) or --api-key.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Load .env so ROBOFLOW_API_KEY is picked up without re-exporting it.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    from fire_vision import scan_photo
except ImportError:
    print("❌ Could not import fire_vision. Run from the WildFrame project root.")
    sys.exit(1)

_DEFAULT_DIRS = [Path("../Downloads/fire_dataset")]
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
_POSITIVE_VERDICTS = {"flame", "smoke", "both"}
_DEFAULT_THRESHOLDS = [round(0.50 + 0.05 * i, 2) for i in range(10)]  # 0.50..0.95
_WORKERS = 3  # conservative: hosted API is happy with a few concurrent calls


class _C:  # minimal ANSI colors
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    GREY = "\033[90m"


# ---------------------------------------------------------------------------
# Discovery & labeling
# ---------------------------------------------------------------------------


def _image_files(d: Path) -> list[Path]:
    return sorted(
        p for p in d.iterdir()
        if p.suffix.lower() in _IMG_EXTS and not p.name.startswith(".")
    )


def _is_positive(p: Path) -> bool | None:
    """Label an image by its parent folder name. None = unlabeled.

    Negative checks come FIRST so 'non_fire_images' isn't caught by the
    'fire' substring ('fire' is a substring of 'non_fire_images').
    """
    parent = p.parent.name.lower()
    if parent.startswith("non") or parent.startswith("no"):
        return False
    if "fire" in parent:
        return True
    return None


def _discover(dirs: list[Path], limit: int) -> tuple[list[Path], list[Path], list[Path]]:
    """Collect (positive, negative, unlabeled) image files from the given dirs.

    A directory holding class subdirectories is treated as a dataset root;
    a directory holding images directly is treated as a single class.
    """
    pos, neg, unk = [], [], []
    for base in dirs:
        if not base.is_dir():
            print(f"{_C.YELLOW}⚠ Skipping (not a directory): {base}{_C.RESET}")
            continue
        class_dirs = [d for d in base.iterdir() if d.is_dir() and not d.name.startswith(".")]
        targets = class_dirs if class_dirs else [base]
        for target in targets:
            for f in _image_files(target):
                label = _is_positive(f)
                bucket = pos if label is True else neg if label is False else unk
                bucket.append(f)
    if limit > 0:
        pos, neg = pos[:limit], neg[:limit]
    return pos, neg, unk


# ---------------------------------------------------------------------------
# Scanning (one model call per image, at threshold 0.0)
# ---------------------------------------------------------------------------


def _scan(paths: list[Path], api_key: str | None, quiet: bool, workers: int = _WORKERS):
    rows: list[dict] = []
    total = len(paths)
    t0 = time.time()
    done = 0

    def _run(p: Path) -> dict:
        try:
            r = scan_photo(p, api_key=api_key, confidence_threshold=0.0)
            return {"fire_conf": float(r.get("fire_confidence", 0.0) or 0.0),
                    "smoke_conf": float(r.get("smoke_confidence", 0.0) or 0.0),
                    "error": r.get("error")}
        except Exception as exc:  # network/parse errors should never kill the sweep
            return {"fire_conf": 0.0, "smoke_conf": 0.0, "error": str(exc)}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run, p): p for p in paths}
        for fut in as_completed(futs):
            p = futs[fut]
            data = fut.result()
            rows.append({
                "file": str(p),
                "name": p.name,
                "positive": _is_positive(p),
                **data,
            })
            done += 1
            if not quiet and (done % 25 == 0 or done == total):
                elapsed = max(time.time() - t0, 0.01)
                rate = done / elapsed
                eta = (total - done) / max(rate, 0.01)
                print(f"    [{done}/{total}] {rate:.1f} img/s · ETA {eta:.0f}s", flush=True)
    rows.sort(key=lambda r: r["name"])  # deterministic output
    return rows, time.time() - t0


# ---------------------------------------------------------------------------
# Threshold sweep (pure local simulation, no extra API calls)
# ---------------------------------------------------------------------------


def _verdict(fire_conf: float, smoke_conf: float, t: float) -> str:
    fire = fire_conf if fire_conf >= t else 0.0
    smoke = smoke_conf if smoke_conf >= t else 0.0
    if fire and smoke:
        return "both"
    if fire:
        return "flame"
    if smoke:
        return "smoke"
    return "nothing"


def _sweep(rows: list[dict], thresholds: list[float]) -> list[dict]:
    labeled = [r for r in rows if not r["error"] and r["positive"] is not None]
    table = []
    for t in thresholds:
        tp = fp = tn = fn = 0
        for r in labeled:
            positive = _verdict(r["fire_conf"], r["smoke_conf"], t) in _POSITIVE_VERDICTS
            if r["positive"]:
                tp += positive
                fn += not positive
            else:
                fp += positive
                tn += not positive
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        f1 = None
        if prec is not None and rec is not None:
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else None
        table.append({
            "threshold": t,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(prec, 4) if prec is not None else None,
            "recall": round(rec, 4) if rec is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
            "accuracy": round(acc, 4) if acc is not None else None,
        })
    return table


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _pct(x) -> str:
    return f"{x * 100:6.1f}%" if x is not None else "    n/a"


def _best_row(table: list[dict]) -> dict | None:
    """Best threshold by F1, tie-broken by precision. None if no TP anywhere."""
    cands = [r for r in table if r["f1"] is not None]
    return max(cands, key=lambda r: (r["f1"], r["precision"])) if cands else None


def _print_sweep(table: list[dict], n_pos: int, n_neg: int) -> None:
    best = _best_row(table)
    zero_fp = [r for r in table if r["fp"] == 0]

    print(f"\n{_C.BOLD}{'═' * 78}{_C.RESET}")
    print(f"{_C.BOLD}  THRESHOLD SWEEP — {n_pos} positives · {n_neg} negatives{_C.RESET}")
    print(f"{_C.BOLD}{'═' * 78}{_C.RESET}")
    header = f"  {'T':>6}  {'PRECISION':>9}  {'RECALL':>7}  {'F1':>6}  {'ACC':>6}  {'FP':>3}/{n_neg:<3}  {'FN':>3}/{n_pos:<3}"
    print(f"{_C.GREY}{header}{_C.RESET}")
    for r in table:
        star = "★" if best is not None and r is best else " "
        line = (f"  {star}{r['threshold']:5.2f}  {_pct(r['precision'])}  {_pct(r['recall'])}"
                f"  {_pct(r['f1'])}  {_pct(r['accuracy'])}  {r['fp']:3d}/{n_neg:<3}  {r['fn']:3d}/{n_pos:<3}")
        if best is not None and r is best:
            print(f"{_C.GREEN}{line}{_C.RESET}  ← best F1")
        else:
            print(line)
    if zero_fp:
        safest = max(zero_fp, key=lambda r: r["recall"])
        print(f"\n  {_C.BLUE}Highest threshold with ZERO false positives: {safest['threshold']:.2f} "
              f"(recall {_pct(safest['recall'])})\n  With auto-approval in mind, this is the safe line "
              f"if false fires on the map are worse than missed fires.{_C.RESET}")


def _print_worst(rows: list[dict], t: float, n: int = 6) -> None:
    """Show the highest-confidence false positives and missed fires at threshold t."""
    fpos = [r for r in rows if not r["error"] and r["positive"] is False
            and _verdict(r["fire_conf"], r["smoke_conf"], t) in _POSITIVE_VERDICTS]
    misses = [r for r in rows if not r["error"] and r["positive"] is True
              and _verdict(r["fire_conf"], r["smoke_conf"], t) not in _POSITIVE_VERDICTS]
    fpos.sort(key=lambda r: max(r["fire_conf"], r["smoke_conf"]), reverse=True)
    misses.sort(key=lambda r: max(r["fire_conf"], r["smoke_conf"]), reverse=True)

    if fpos:
        print(f"\n  {_C.RED}Highest-confidence FALSE POSITIVES @ {t:.2f} (clean images the model called fire):{_C.RESET}")
        for r in fpos[:n]:
            print(f"    🔴 {r['name']:42s} fire={r['fire_conf']:.3f} smoke={r['smoke_conf']:.3f}")
    if misses:
        print(f"\n  {_C.YELLOW}Highest-confidence MISSED FIRES @ {t:.2f} (real fires the model called clean):{_C.RESET}")
        for r in misses[:n]:
            print(f"    ⚠️ {r['name']:42s} fire={r['fire_conf']:.3f} smoke={r['smoke_conf']:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep fire/smoke detection confidence thresholds to find the best one.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python threshold_sweep.py                     # quick balanced run (100/class)
  python threshold_sweep.py --limit 0           # full dataset
  python threshold_sweep.py PATH_A PATH_B       # custom folders
        """,
    )
    parser.add_argument("dirs", nargs="*", metavar="DIR",
                        help="Dataset folder(s). Default: ../Downloads/fire_dataset "
                             "(fire_images + non_fire_images, labeled by folder name)")
    parser.add_argument("--limit", type=int, default=100, metavar="N",
                        help="Max images per class to scan (0 = all). Default: 100")
    parser.add_argument("--thresholds", metavar="CSV", default=None,
                        help="Comma-separated threshold grid. Default: 0.50..0.95 step 0.05")
    parser.add_argument("--api-key", metavar="KEY", default=None,
                        help="Roboflow API key (defaults to ROBOFLOW_API_KEY env var)")
    parser.add_argument("--workers", type=int, default=3, metavar="N",
                        help="Concurrent scans (default 3)")
    parser.add_argument("--json", metavar="FILE", default=None, help="Export results as JSON")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-image progress")
    args = parser.parse_args()

    dirs = [Path(d) for d in args.dirs] if args.dirs else _DEFAULT_DIRS
    if args.thresholds:
        try:
            thresholds = [float(x.strip()) for x in args.thresholds.split(",")]
        except ValueError:
            parser.error(f"--thresholds must be a comma-separated list of floats, got: {args.thresholds}")
        if not thresholds or any(not 0.0 <= t <= 1.0 for t in thresholds):
            parser.error("--thresholds values must be between 0.0 and 1.0")
    else:
        thresholds = _DEFAULT_THRESHOLDS

    pos, neg, unk = _discover(dirs, args.limit)
    if not pos and not neg:
        print(f"{_C.RED}❌ No labeled images found in {', '.join(str(d) for d in dirs)}.{_C.RESET}")
        return 1
    paths = pos + neg + unk
    print(f"{_C.BOLD}🔍 Scanning {len(paths)} images "
          f"({len(pos)} fire, {len(neg)} non-fire"
          + (f", {len(unk)} unlabeled" if unk else "")
          + f") once, then sweeping {len(thresholds)} thresholds locally…{_C.RESET}")

    rows, elapsed = _scan(paths, args.api_key, args.quiet, workers=max(1, args.workers))
    errors = [r for r in rows if r["error"]]
    if errors:
        kinds: dict[str, int] = {}
        for e in errors:
            key = e["error"].split(":")[0][:60]
            kinds[key] = kinds.get(key, 0) + 1
        detail = ", ".join(f"{k} ×{v}" for k, v in list(kinds.items())[:3])
        names = ", ".join(r["name"] for r in errors[:5])
        suffix = "…" if len(errors) > 5 else ""
        print(f"{_C.RED}⚠ {len(errors)}/{len(rows)} scans failed ({detail}). "
              f"Files: {names}{suffix}{_C.RESET}")

    table = _sweep(rows, thresholds)
    n_pos = sum(1 for r in rows if not r["error"] and r["positive"] is True)
    n_neg = sum(1 for r in rows if not r["error"] and r["positive"] is False)
    if n_pos + n_neg == 0:
        print(f"{_C.RED}❌ No usable (labeled, non-error) results.{_C.RESET}")
        return 1

    _print_sweep(table, n_pos, n_neg)
    best = _best_row(table)
    if best:
        _print_worst(rows, best["threshold"])
    else:
        print(f"\n{_C.YELLOW}⚠ No threshold produced a true positive — the model found no fire/smoke "
              f"in any positive image at any threshold ≥ 0.50. Consider a wider grid:\n"
              f"   python threshold_sweep.py --thresholds \"0.1,0.2,0.3,0.4,0.5\"{_C.RESET}")
    print(f"\n{_C.GREY}  Scan took {elapsed:.0f}s ({elapsed / max(len(rows), 1):.1f}s/image avg). "
          f"API calls = {len(rows)} (one per image).{_C.RESET}")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "meta": {"images": len(rows), "api_calls": len(rows),
                     "elapsed_s": round(elapsed, 1)},
            "rows": rows,
            "sweep": table,
            "recommendation": {"best_f1_threshold": best["threshold"] if best else None,
                               "zero_fp_threshold": max((r["threshold"] for r in table if r["fp"] == 0),
                                                        default=None)},
        }, indent=2))
        print(f"{_C.GREEN}📄 Results exported to {args.json}{_C.RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
