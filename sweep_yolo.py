#!/usr/bin/env python3
"""
sweep_yolo.py — Re-tune the auto-approval confidence floors for the local
YOLOv26 engine (fire >= T_fire OR smoke >= T_smoke) against ground truth.
================================================================================
Scans the fire_dataset ONCE with the local YOLO model (the default
fire_vision engine), caches the per-image (fire_conf, smoke_conf) scores,
then evaluates every (T_fire, T_smoke) pair offline — no re-inference.

Ground truth comes from the folder layout (same convention as
threshold_sweep.py):
    fire_images/*.png      → positive (contains fire/smoke)
    non_fire_images/*.png  → negative (clean)

The simulated gate matches server._auto_approval_decision's model side:
    pass = (fire_conf >= T_fire) OR (smoke_conf >= T_smoke)
(the corroboration requirement — cluster/satellite — is a separate gate,
applied afterwards in the server; this sweep measures the model side alone).

Usage
-----
    python sweep_yolo.py                # full dataset, default grid
    python sweep_yolo.py --json sweep_yolo.json --reuse   # re-analyze cache
    python sweep_yolo.py --limit 100    # quick balanced subset

Notes
-----
* The exported ONNX graph bakes in its own NMS confidence filter (~0.25),
  so per-image scores are either 0 or >= ~0.25 — thresholds below that are
  equivalent to "any detection".
* Deterministic: rows are cached to the --json file; the sweep itself is
  pure arithmetic.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp"}
_DEFAULT_DATASET = Path.home() / "Downloads" / "fire_dataset"

# Default floor grid for auto-approval pairs (fire, smoke).
_FIRE_GRID = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90]
_SMOKE_GRID = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]

# The floors currently in production, read live from server.py's defaults so
# this stays correct after re-tuning (no import of the heavy server module).
def _current_floors() -> tuple[float, float]:
    src = (Path(__file__).resolve().parent / "server.py").read_text()
    tree = ast.parse(src)
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if not any(t in ("AUTO_APPROVE_FLAME_MIN_CONF", "AUTO_APPROVE_SMOKE_MIN_CONF") for t in targets):
                continue
            call = node.value
            # os.environ.get("WILDFRAME_...", "0.50") -> default literal in args[1]
            if (isinstance(call.func, ast.Attribute) and call.func.attr == "get"
                    and len(call.args) == 2
                    and isinstance(call.args[1], ast.Constant)):
                for t in targets:
                    found[t] = float(call.args[1].value)
    return (found.get("AUTO_APPROVE_FLAME_MIN_CONF", 0.50),
            found.get("AUTO_APPROVE_SMOKE_MIN_CONF", 0.70))

_C = {"BOLD": "\033[1m", "RED": "\033[31m", "GREEN": "\033[32m",
      "YELLOW": "\033[33m", "BLUE": "\033[34m", "GREY": "\033[90m",
      "RESET": "\033[0m"}


# ---------------------------------------------------------------------------
# Dataset discovery + scanning
# ---------------------------------------------------------------------------


def _image_files(d: Path) -> list[Path]:
    return sorted(
        p for p in d.iterdir()
        if p.suffix.lower() in _IMG_EXTS and not p.name.startswith(".")
    )


def _is_positive(p: Path) -> bool | None:
    """Label by parent folder name (negative first — 'non_fire' contains 'fire')."""
    parent = p.parent.name.lower()
    if parent.startswith("non") or parent.startswith("no"):
        return False
    if "fire" in parent:
        return True
    return None


def _discover(dirs: list[Path], limit: int) -> tuple[list[Path], list[Path]]:
    pos, neg = [], []
    for base in dirs:
        if not base.is_dir():
            print(f"{_C['YELLOW']}⚠ Skipping (not a directory): {base}{_C['RESET']}")
            continue
        class_dirs = [d for d in base.iterdir() if d.is_dir() and not d.name.startswith(".")]
        targets = class_dirs if class_dirs else [base]
        for target in targets:
            for f in _image_files(target):
                label = _is_positive(f)
                if label is True:
                    pos.append(f)
                elif label is False:
                    neg.append(f)
                # unlabeled folders (not fire/non-fire) are skipped by convention
    if limit > 0:
        pos, neg = pos[:limit], neg[:limit]
    return pos, neg


def _scan(paths: list[Path], workers: int, quiet: bool) -> list[dict]:
    rows: list[dict] = []
    total = len(paths)
    t0 = time.time()
    done = 0

    def _run(p: Path) -> dict:
        try:
            r = scan_photo(p, confidence_threshold=0.0)
            return {
                "fire_conf": float(r.get("fire_confidence", 0.0) or 0.0),
                "smoke_conf": float(r.get("smoke_confidence", 0.0) or 0.0),
                "error": r.get("error"),
            }
        except Exception as exc:
            return {"fire_conf": 0.0, "smoke_conf": 0.0, "error": str(exc)}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run, p): p for p in paths}
        for fut in as_completed(futs):
            p = futs[fut]
            data = fut.result()
            rows.append({"file": str(p), "name": p.name, "positive": _is_positive(p), **data})
            done += 1
            if not quiet and (done % 25 == 0 or done == total):
                elapsed = max(time.time() - t0, 0.01)
                rate = done / elapsed
                print(f"    [{done}/{total}] {rate:.1f} img/s · ETA {(total - done) / max(rate, 0.01):.0f}s", flush=True)
    rows.sort(key=lambda r: r["name"])
    return rows


# ---------------------------------------------------------------------------
# Pair sweep (offline)
# ---------------------------------------------------------------------------


def _evaluate_pair(rows: list[dict], t_fire: float, t_smoke: float) -> dict:
    tp = fp = tn = fn = 0
    for r in rows:
        if r["error"] or r["positive"] is None:
            continue
        passed = r["fire_conf"] >= t_fire or r["smoke_conf"] >= t_smoke
        if r["positive"]:
            tp += passed
            fn += not passed
        else:
            fp += passed
            tn += not passed
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec is not None and rec is not None and prec + rec) else 0.0
    return {
        "t_fire": t_fire, "t_smoke": t_smoke,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": prec, "recall": rec, "f1": f1,
    }


def _sweep(rows: list[dict], fire_grid: list[float], smoke_grid: list[float]) -> list[dict]:
    return [_evaluate_pair(rows, tf, ts) for tf in fire_grid for ts in smoke_grid]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _pct(x) -> str:
    return f"{x * 100:6.1f}%" if x is not None else "    n/a"


def _print_pair_table(pairs: list[dict], n_pos: int, n_neg: int) -> None:
    """Best F1 with zero FPs first, then zero-FP rows by recall, then best F1 overall."""
    best_zero_fp = max((p for p in pairs if p["fp"] == 0), key=lambda p: (p["recall"] or 0.0, p["f1"]), default=None)
    best_f1 = max(pairs, key=lambda p: (p["f1"], p["precision"] if p["precision"] is not None else 0.0))

    print(f"\n{_C['BOLD']}{'═' * 82}{_C['RESET']}")
    print(f"{_C['BOLD']}  AUTO-APPROVAL FLOOR SWEEP — YOLOv26 · {n_pos} positives · {n_neg} negatives{_C['RESET']}")
    print(f"  rule: pass = fire ≥ T_fire  OR  smoke ≥ T_smoke   (corroboration gate is separate)")
    print(f"{_C['BOLD']}{'═' * 82}{_C['RESET']}")
    header = (f"  {'T_fire':>6}  {'T_smoke':>7}  {'PRECISION':>9}  {'RECALL':>7}  {'F1':>6}  "
              f"{'FP':>3}/{n_neg:<3}  {'FN':>3}/{n_pos:<3}")
    print(f"{_C['GREY']}{header}{_C['RESET']}")
    for p in pairs:
        star = "★" if p is best_zero_fp or p is best_f1 else " "
        line = (f"  {star}{p['t_fire']:5.2f}  {p['t_smoke']:5.2f}  {_pct(p['precision'])}  {_pct(p['recall'])}  "
                f"{_pct(p['f1'])}  {p['fp']:3d}/{n_neg:<3}  {p['fn']:3d}/{n_pos:<3}")
        if p is best_zero_fp:
            print(f"{_C['GREEN']}{line}{_C['RESET']}  ← safest (0 FP, max recall)")
        elif p is best_f1 and p is not best_zero_fp:
            print(f"{_C['BLUE']}{line}{_C['RESET']}  ← best F1")
        else:
            print(line)


def _print_current(rows: list[dict], t_fire: float, t_smoke: float, n_pos: int, n_neg: int) -> None:
    p = _evaluate_pair(rows, t_fire, t_smoke)
    print(f"\n  Current production floors ({t_fire:.2f}, {t_smoke:.2f}): "
          f"recall {_pct(p['recall'])} · precision {_pct(p['precision'])} · "
          f"F1 {_pct(p['f1'])} · FP {p['fp']}/{n_neg} · FN {p['fn']}/{n_pos}")


def _print_gate_split(rows: list[dict], t_fire: float, t_smoke: float, label: str) -> None:
    """Which gate catches each true fire — fire-only, smoke-only, or both."""
    fire_only = smoke_only = both = 0
    for r in rows:
        if r["error"] or r["positive"] is not True:
            continue
        f = r["fire_conf"] >= t_fire
        s = r["smoke_conf"] >= t_smoke
        if f and s:
            both += 1
        elif f:
            fire_only += 1
        elif s:
            smoke_only += 1
    print(f"  [{label}] gates on true fires: fire-only {fire_only} · "
          f"smoke-only {smoke_only} · both {both}")


def _print_worst(rows: list[dict], t_fire: float, t_smoke: float, n: int = 8) -> None:
    def _passes(r):
        return r["fire_conf"] >= t_fire or r["smoke_conf"] >= t_smoke

    fpos = sorted(
        (r for r in rows if not r["error"] and r["positive"] is False and _passes(r)),
        key=lambda r: max(r["fire_conf"], r["smoke_conf"]), reverse=True,
    )
    misses = sorted(
        (r for r in rows if not r["error"] and r["positive"] is True and not _passes(r)),
        key=lambda r: max(r["fire_conf"], r["smoke_conf"]), reverse=True,
    )
    if fpos:
        print(f"\n  {_C['RED']}Highest-confidence FALSE POSITIVES @ ({t_fire:.2f},{t_smoke:.2f}) "
              f"(clean images that would pass the floor):{_C['RESET']}")
        for r in fpos[:n]:
            print(f"    🔴 {r['name']:44s} fire={r['fire_conf']:.3f} smoke={r['smoke_conf']:.3f}")
    if misses:
        print(f"\n  {_C['YELLOW']}Highest-confidence MISSED FIRES @ ({t_fire:.2f},{t_smoke:.2f}) "
              f"(real fires that would NOT pass the floor):{_C['RESET']}")
        for r in misses[:n]:
            print(f"    ⚠️ {r['name']:44s} fire={r['fire_conf']:.3f} smoke={r['smoke_conf']:.3f}")


def _print_distributions(rows: list[dict]) -> None:
    pos = [r for r in rows if not r["error"] and r["positive"] is True]
    neg = [r for r in rows if not r["error"] and r["positive"] is False]

    def _stats(rs, key):
        vals = sorted((r[key] for r in rs if r[key] > 0), reverse=True)
        n_any = len(vals)
        return n_any, (vals[0] if vals else 0.0), (sum(vals) / len(vals) if vals else 0.0)

    print(f"\n  {_C['GREY']}Per-class score distributions (images with any detection > 0):{_C['RESET']}")
    for key, label in (("fire_conf", "fire "), ("smoke_conf", "smoke")):
        pn, p_max, p_avg = _stats(pos, key)
        nn, n_max, n_avg = _stats(neg, key)
        print(f"    {label}:  positives {pn:4d}/{len(pos)} (max {p_max:.3f}, avg {p_avg:.3f})   |   "
              f"negatives {nn:4d}/{len(neg)} (max {n_max:.3f}, avg {n_avg:.3f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep auto-approval (fire, smoke) floor pairs against the fire_dataset.",
    )
    parser.add_argument("dirs", nargs="*", metavar="DIR",
                        help="Dataset folder(s). Default: ~/Downloads/fire_dataset")
    parser.add_argument("--limit", type=int, default=0, metavar="N",
                        help="Max images per class to scan (0 = all). Default: 0 (all)")
    parser.add_argument("--fire-grid", metavar="CSV", default=",".join(f"{t:.2f}" for t in _FIRE_GRID))
    parser.add_argument("--smoke-grid", metavar="CSV", default=",".join(f"{t:.2f}" for t in _SMOKE_GRID))
    parser.add_argument("--workers", type=int, default=4, metavar="N", help="Concurrent scans")
    parser.add_argument("--json", metavar="FILE", default="sweep_yolo.json", help="Cache/export file")
    parser.add_argument("--reuse", action="store_true", help="Skip scanning, re-analyze the cache file")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cache = Path(args.json)
    rows: list[dict] = []
    if args.reuse and cache.is_file():
        rows = json.loads(cache.read_text())
        print(f"Reused {len(rows)} cached scans from {cache}")
    else:
        # Guard: this harness measures the YOLO engine specifically — if auto
        # resolves elsewhere (e.g. no models/best.onnx in a fresh checkout),
        # scan_photo would silently sweep the hosted Roboflow model instead.
        from fire_vision import _resolve_engine, _yolo_model_path

        if _resolve_engine() != "yolo" or _yolo_model_path() is None:
            print("❌ Local YOLO engine not active. Put models/best.onnx (or best.pt) "
                  f"in models/ or set WILDFRAME_VISION_ENGINE=yolo (resolved: "
                  f"{_resolve_engine()}).")
            sys.exit(2)
        print(f"engine: YOLO — {_yolo_model_path()}")
        dirs = [Path(d) for d in args.dirs] if args.dirs else [_DEFAULT_DATASET]
        pos, neg = _discover(dirs, args.limit)
        print(f"Scanning {len(pos)} positives + {len(neg)} negatives…", flush=True)
        rows = _scan(pos + neg, args.workers, args.quiet)
        cache.write_text(json.dumps(rows, indent=1))
        print(f"Saved {len(rows)} rows to {cache}")

    labeled = [r for r in rows if not r["error"] and r["positive"] is not None]
    n_pos = sum(1 for r in labeled if r["positive"])
    n_neg = sum(1 for r in labeled if not r["positive"])
    errs = sum(1 for r in rows if r["error"])
    print(f"labeled: {n_pos} pos · {n_neg} neg · {errs} errors")

    fire_grid = [float(x) for x in args.fire_grid.split(",")]
    smoke_grid = [float(x) for x in args.smoke_grid.split(",")]
    pairs = _sweep(labeled, fire_grid, smoke_grid)

    _print_pair_table(pairs, n_pos, n_neg)
    current = _current_floors()
    _print_current(labeled, *current, n_pos, n_neg)
    _print_gate_split(labeled, *current, "current")

    # Highlight the two candidate settings: safest (0 FP) and balanced.
    safest = max((p for p in pairs if p["fp"] == 0), key=lambda p: (p["recall"] or 0.0, p["f1"]), default=None)
    best_f1 = max(pairs, key=lambda p: (p["f1"], p["precision"] if p["precision"] is not None else 0.0))
    if safest:
        print(f"\n  {_C['BLUE']}SAFEST with ZERO false positives: (fire ≥ {safest['t_fire']:.2f}, "
              f"smoke ≥ {safest['t_smoke']:.2f}) — recall {_pct(safest['recall'])}{_C['RESET']}")
        _print_worst(labeled, safest["t_fire"], safest["t_smoke"])
    if best_f1 and best_f1 is not safest:
        print(f"\n  {_C['GREEN']}BALANCED (best F1): (fire ≥ {best_f1['t_fire']:.2f}, "
              f"smoke ≥ {best_f1['t_smoke']:.2f}) — recall {_pct(best_f1['recall'])}, "
              f"FP {best_f1['fp']}/{n_neg}{_C['RESET']}")
        _print_worst(labeled, best_f1["t_fire"], best_f1["t_smoke"])

    _print_distributions(labeled)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
