#!/usr/bin/env python3
"""
test_firms_dedup.py — Regression test for FIRMS re-injection saturation
========================================================================

Locks down the dedup in server._fuse_firms_hotspots: the FIRMS poller
fetches the FULL past-24h window on every pass, and before this fix it
re-injected every hotspot each time — each `grid.update()` added +log(50)
to logits AND re-stamped `last_updated = now`, so probabilities ratcheted
to the 0.9999 clamp and evidence never aged past the 12h shelf life
(decay never engaged). 99% of production grids ended up pinned at
max_p >= 0.85 (the all-crimson wall).

Run from the project root:  .venv/bin/python test_firms_dedup.py
Exit code is non-zero on failure.
"""

from __future__ import annotations

import sys

import numpy as np

from bayesian_filter import BayesianFireGrid
from nasa_firms import FIRMSHotspot
from server import _fuse_firms_hotspots


def _hs(lat: float, lon: float, hhmm: str, date: str = "2026-08-15") -> FIRMSHotspot:
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

    print("== scenario 1: same 24h window fetched twice (the saturation bug) ==")
    g = _grid()
    # A fire seen by one VIIRS overpass: 5 pixels at 375m+ spacing (real
    # pixel pitch — anything denser would be the same detection twice).
    window = [_hs(51.0, 19.0 + 0.004 * i, "1200") for i in range(5)]
    n1 = _fuse_firms_hotspots(g, window)
    check("pass 1 injects all 5 hotspots", n1 == 5, f"got {n1}")
    p_after_pass1 = float(g.probabilities.max())
    logits_after_pass1 = g.logits.copy()

    n2 = _fuse_firms_hotspots(g, window)  # identical window, 10 min later
    check("pass 2 injects 0 (all already fused)", n2 == 0, f"got {n2}")
    check(
        "max_p unchanged after re-injection",
        float(g.probabilities.max()) == p_after_pass1,
        f"{p_after_pass1:.6f} -> {float(g.probabilities.max()):.6f}",
    )
    check(
        "logits bit-identical after re-injection",
        np.array_equal(g.logits, logits_after_pass1),
    )

    print("== scenario 2: next overpass adds a NEW detection ==")
    new_detection = [_hs(51.0005, 19.0005, "1800")]  # newer acq_time
    n3 = _fuse_firms_hotspots(g, new_detection)
    check("new detection IS injected", n3 == 1, f"got {n3}")
    check(
        "max_p rises with new evidence",
        float(g.probabilities.max()) > p_after_pass1,
    )

    print("== scenario 3: duplicate pixels, same cell, same minute ==")
    g2 = _grid()
    dup = [_hs(51.0, 19.0, "0900"), _hs(51.0, 19.0, "0900")]  # exact duplicates
    n4 = _fuse_firms_hotspots(g2, dup)
    check("exact duplicate fused once", n4 == 1, f"got {n4}")

    print("== scenario 4: pre-deploy grid stamped with recent wall-clock time ==")
    # Before the fix, last_updated held fetch (wall-clock) times. A grid
    # loaded from the old DB has fresh stamps on its evidence cells, so
    # hotspots from the same cells in the 24h window must be skipped on
    # the first post-deploy pass — no mass re-boost.
    g3 = _grid()
    recent = 2_000_000_000.0  # a recent wall-clock stamp (2033 — newer than any probe)
    for lat, lon in [(51.0, 19.0), (51.001, 19.0), (51.0, 19.001)]:
        g3.update(
            __import__("bayesian_filter").Evidence.satellite_hotspot(lat, lon),
            at=recent,
        )
    old_window = [
        _hs(51.0, 19.0, "0800"),       # same cell, older detection
        _hs(51.001, 19.0, "0830"),     # same cell, older detection
        _hs(51.0, 19.001, "0900"),     # same cell, older detection
    ]
    n5 = _fuse_firms_hotspots(g3, old_window)
    check("old-window detections skipped vs recent stamps", n5 == 0, f"got {n5}")

    print("== scenario 5: first-time evidence for a never-seen cell still injects ==")
    g4 = _grid()
    recent = 2_000_000_000.0
    g4.update(
        __import__("bayesian_filter").Evidence.satellite_hotspot(51.0, 19.0),
        at=recent,
    )
    n6 = _fuse_firms_hotspots(g4, [_hs(51.002, 19.002, "0800")])  # >100m away = new cell
    check("new-cell evidence injected once", n6 == 1, f"got {n6}")

    print()
    if failures:
        print(f"FAILED ({failures} check(s) failed)")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
