#!/usr/bin/env python3
"""
sweep_effnet.py — Sweep the auto-approval fire floor against the fine-tuned
EfficientNet-B0 classifier (trained by effnet_train.py) on the held-out split.

EfficientNet is a classifier: it produces ONE fire probability per image (no
bounding boxes, no separate smoke channel on our 2-class model), so the gate
here is the single rule `pass = fire >= T` — the smoke leg is dropped.

Rows come from effnet_train.py (effnet_rows_val.json / effnet_rows_train.json).

Usage:
    python sweep_effnet.py                       # held-out val split
    python sweep_effnet.py --rows effnet_rows_train.json
    python sweep_effnet.py --grid 0.50,0.60,0.70,0.80,0.90
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

_C = {"BOLD": "\033[1m", "RED": "\033[31m", "GREEN": "\033[32m",
      "YELLOW": "\033[33m", "BLUE": "\033[34m", "GREY": "\033[90m",
      "RESET": "\033[0m"}

_DEFAULT_ROWS = Path("effnet_rows_val.json")
_FIRE_GRID = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]


def _pct(x) -> str:
    return f"{x * 100:6.1f}%" if x is not None else "    n/a"


def _eval(rows, t) -> dict:
    tp = fp = tn = fn = 0
    for r in rows:
        if r.get("error") or r.get("positive") is None:
            continue
        passed = r["fire_conf"] >= t
        if r["positive"]:
            tp += passed
            fn += not passed
        else:
            fp += passed
            tn += not passed
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec is not None and rec is not None and prec + rec) else 0.0
    return {"t": t, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1}


def _print_table(rows, label: str) -> None:
    labeled = [r for r in rows if not r.get("error") and r.get("positive") is not None]
    n_pos = sum(1 for r in labeled if r["positive"])
    n_neg = len(labeled) - n_pos
    print(f"\n{_C['BOLD']}{'═' * 78}{_C['RESET']}")
    print(f"{_C['BOLD']}  EFFICIENTNET-B0 FIRE FLOOR SWEEP — {label} · "
          f"{n_pos} positives · {n_neg} negatives{_C['RESET']}")
    print(f"  rule: pass = fire ≥ T   (classifier → one fire probability/image, no smoke leg)")
    print(f"{_C['BOLD']}{'═' * 78}{_C['RESET']}")
    print(f"{_C['GREY']}  {'T_fire':>6}  {'PRECISION':>9}  {'RECALL':>7}  {'F1':>6}  "
          f"{'FP':>3}/{n_neg:<3}  {'FN':>3}/{n_pos:<3}{_C['RESET']}")

    results = [_eval(labeled, t) for t in _FIRE_GRID]
    best_zero_fp = max((p for p in results if p["fp"] == 0), key=lambda p: (p["recall"] or 0.0, p["f1"]), default=None)
    best_f1 = max(results, key=lambda p: (p["f1"], p["precision"] if p["precision"] is not None else 0.0))
    for p in results:
        star = "★" if p is best_zero_fp or p is best_f1 else " "
        line = (f"  {star}{p['t']:5.2f}  {_pct(p['precision'])}  {_pct(p['recall'])}  "
                f"{_pct(p['f1'])}  {p['fp']:3d}/{n_neg:<3}  {p['fn']:3d}/{n_pos:<3}")
        if p is best_zero_fp:
            print(f"{_C['GREEN']}{line}{_C['RESET']}  ← safest (0 FP, max recall)")
        elif p is best_f1 and p is not best_zero_fp:
            print(f"{_C['BLUE']}{line}{_C['RESET']}  ← best F1")
        else:
            print(line)

    if best_zero_fp:
        print(f"\n  {_C['BLUE']}SAFEST with ZERO false positives: fire ≥ {best_zero_fp['t']:.2f} "
              f"— recall {_pct(best_zero_fp['recall'])}{_C['RESET']}")
        fpos = sorted((r for r in labeled if r["positive"] is False and r["fire_conf"] >= best_zero_fp["t"]),
                      key=lambda r: r["fire_conf"], reverse=True)
        misses = sorted((r for r in labeled if r["positive"] is True and r["fire_conf"] < best_zero_fp["t"]),
                        key=lambda r: r["fire_conf"], reverse=True)
        if fpos:
            print(f"  {_C['RED']}False positives (clean images above the floor):{_C['RESET']}")
            for r in fpos[:8]:
                print(f"    🔴 {r['name']:44s} fire={r['fire_conf']:.3f}")
        if misses:
            print(f"  {_C['YELLOW']}Missed fires (below the floor):{_C['RESET']}")
            for r in misses[:10]:
                print(f"    ⚠️ {r['name']:44s} fire={r['fire_conf']:.3f}")
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=Path, default=_DEFAULT_ROWS)
    ap.add_argument("--also-train", action="store_true",
                    help="Also sweep the training split rows (memorization check)")
    ap.add_argument("--grid", default=None, help="CSV of thresholds (default: %(default)s)")
    args = ap.parse_args()
    if args.grid:
        _FIRE_GRID[:] = [float(x) for x in args.grid.split(",")]

    rows = json.loads(args.rows.read_text())
    print(f"loaded {len(rows)} rows from {args.rows}")
    _print_table(rows, args.rows.stem.replace("effnet_rows_", ""))
    if args.also_train:
        tr = Path("effnet_rows_train.json")
        if tr.is_file():
            _print_table(json.loads(tr.read_text()), "train")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
