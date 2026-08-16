#!/usr/bin/env python3
"""Tests for the static-source mask downweight (Step 2 of the filtering plan).

Covers:
  * Evidence.satellite_hotspot(weight=) — the reduced-LR path used to
    downweight real-but-not-wildfire heat (refinery flares, steel mills)
  * db.static_source_hits_batch — the batched PostGIS lookup against the
    `static_sources` mask (built by build_static_mask.py), fail-open when
    the table is missing
  * server._fuse_firms_hotspots — a hotspot tagged `_static_source` is
    injected at reduced weight and cannot push a grid as high as a
    full-strength detection

Run from the project root:  .venv/bin/python test_static_source.py
Exit code is non-zero on failure.
"""

from __future__ import annotations

import math
import sys

import numpy as np

from bayesian_filter import BayesianFireGrid, Evidence
from nasa_firms import FIRMSHotspot
from server import STATIC_SOURCE_EVIDENCE_WEIGHT, _fuse_firms_hotspots


def _hs(lat: float, lon: float, hhmm: str = "1200", date: str = "2026-08-15") -> FIRMSHotspot:
    return FIRMSHotspot(
        latitude=lat, longitude=lon,
        brightness=350.0, scan=1.0, track=1.0,
        acq_date=date, acq_time=hhmm,
        satellite="N", instrument="VIIRS",
        confidence="nominal", version="2.0", frp=42.0, daynight="D",
    )


def _grid() -> BayesianFireGrid:
    return BayesianFireGrid(center_lat=51.0, center_lon=19.0, cell_size_m=100, nx=60, ny=60)


def main() -> int:
    failures = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal failures
        if cond:
            print(f"  PASS  {name}")
        else:
            failures += 1
            print(f"  FAIL  {name}  {detail}")

    print("== Evidence weight scaling ==")
    full = Evidence.satellite_hotspot(51.0, 19.0)
    low = Evidence.satellite_hotspot(51.0, 19.0, weight=STATIC_SOURCE_EVIDENCE_WEIGHT)
    check("full-weight LR is ln(50) ≈ 3.91", abs(full.log_likelihood_ratio - math.log(50.0)) < 1e-9)
    check(
        "downweighted LR is ln(50*weight)",
        abs(low.log_likelihood_ratio - math.log(50.0 * STATIC_SOURCE_EVIDENCE_WEIGHT)) < 1e-9,
        f"{low.log_likelihood_ratio}",
    )
    check(
        "downweighted LR < full LR",
        low.log_likelihood_ratio < full.log_likelihood_ratio,
    )

    print("== _fuse_firms_hotspots downweight ==")
    # Identical single-detection grids, one full-strength, one tagged static.
    g_full = _grid()
    g_static = _grid()
    hs_full = _hs(51.0, 19.0)
    hs_static = _hs(51.0, 19.0)
    hs_static._static_source = True
    _fuse_firms_hotspots(g_full, [hs_full])
    _fuse_firms_hotspots(g_static, [hs_static])

    p_full = float(g_full.probabilities.max())
    p_static = float(g_static.probabilities.max())
    # Grid prior is p=0.01 (logit −4.60); a single full-strength ln(50)
    # detection lands at logit −0.69 → p ≈ 0.335. Assert it clearly lifts
    # the cell above the prior and stays below the auto-approval floor.
    check("full-strength hotspot lifts max_p above prior", p_full > 0.2, f"{p_full:.4f}")
    check(
        "static-source hotspot stays well below full-strength",
        p_static < p_full * 0.5,
        f"static {p_static:.4f} vs full {p_full:.4f}",
    )
    check(
        "static-source hotspot cannot reach auto-approval floor (0.85)",
        p_static < 0.85,
        f"{p_static:.4f}",
    )

    # Untagged hotspots keep working exactly as before (no behaviour change).
    g3 = _grid()
    window = [_hs(51.0, 19.0 + 0.004 * i, "1200") for i in range(5)]
    n = _fuse_firms_hotspots(g3, window)
    check("untagged window still injects all 5", n == 5, f"got {n}")
    check("dedup still holds for tagged hotspots", _fuse_firms_hotspots(g_static, [hs_static]) == 0)

    print("== db.static_source_hits_batch (needs built mask) ==")
    import db

    try:
        with db._conn() as conn:
            conn.execute("SELECT 1 FROM static_sources LIMIT 1").fetchone()
    except Exception:
        print("  SKIP static_sources not built yet (run build_static_mask.py first)")
    else:
        # Dąbrowa Górnicza steel belt — the top flagged cell in the mask
        # (167 detections / 15 months, centroid from the loaded table).
        hits = db.static_source_hits_batch([(50.33287807428219, 19.28924863499593)])
        check("Dąbrowa Górnicza steel is flagged static", hits.get(0) is True, f"{hits}")

        # Białowieża forest — no industrial cell, must NOT be flagged.
        hits2 = db.static_source_hits_batch([(52.7419, 23.8590)])
        check("Białowieża forest is NOT flagged static", 0 not in hits2, f"{hits2}")

        # Mixed batch returns only the flagged index.
        hits3 = db.static_source_hits_batch([
            (52.7419, 23.8590),
            (50.33287807428219, 19.28924863499593),
        ])
        check("mixed batch flags only the industrial point", 0 not in hits3 and hits3.get(1) is True)

    print()
    if failures:
        print(f"FAILED ({failures} check(s) failed)")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
