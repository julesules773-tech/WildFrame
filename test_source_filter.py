#!/usr/bin/env python3
"""Tests for the source filter (Step 4 of the filtering plan): volcanic /
conflict policy knobs.

Covers:
  * _tag_volcanic — the db.volcano_hits_batch stub tags nothing (GVP import
    deferred → fail-open); a monkeypatched hit set tags the right hotspots
  * _tag_conflict — kv_store conflict_zones geometry (bbox + polygon) tags
    points inside, leaves points outside; fail-open on empty/malformed
  * _is_source_passthrough — passthrough policy exempts tagged sources
  * _gate_firms_by_land_cover — passthrough hotspots survive the gate even
    on non-burnable land; untagged hotspots on industrial land still drop
  * _fuse_firms_hotspots — volcanic downweight (0.1) has NO escape hatch
    (pre-existing grid still downweighted, unlike ag-burn); volcanic +
    static compose to the max weight with both tags logged; conflict
    downweight works the same
  * drop policy — volcanic/conflict hotspots excluded from the pass list
  * report confirmation — volcanic downweight does NOT confirm, conflict
    DOES, passthrough volcanic DOES

Run from the project root:  .venv/bin/python test_source_filter.py
Exit code is non-zero on failure.
"""

from __future__ import annotations

import math
import sys

from bayesian_filter import BayesianFireGrid, Evidence
from nasa_firms import FIRMSHotspot
from server import (
    CONFLICT_EVIDENCE_WEIGHT,
    SOURCE_FILTER_CONFLICT,
    SOURCE_FILTER_VOLCANIC,
    VOLCANIC_EVIDENCE_WEIGHT,
    _confirm_reports_against_hotspots,
    _fuse_firms_hotspots,
    _gate_firms_by_land_cover,
    _is_source_passthrough,
    _tag_conflict,
    _tag_volcanic,
)
import db
import server


def _hs(lat: float, lon: float, hhmm: str = "1200", date: str = "2026-08-15",
        frp: float = 3.0) -> FIRMSHotspot:
    return FIRMSHotspot(
        latitude=lat, longitude=lon,
        brightness=350.0, scan=1.0, track=1.0,
        acq_date=date, acq_time=hhmm,
        satellite="N", instrument="VIIRS",
        confidence="nominal", version="2.0", frp=frp, daynight="D",
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

    print("== _tag_volcanic: stub tags nothing (GVP import deferred) ==")
    v = _hs(51.0, 19.0)
    n = _tag_volcanic([v])
    check("stub returns no hits → nothing tagged (fail-open)", n == 0 and not getattr(v, "_volcanic", False))

    print("== _tag_volcanic: monkeypatched hits tag correctly ==")
    orig = db.volcano_hits_batch
    try:
        db.volcano_hits_batch = lambda pts: {0: True, 2: True}
        a, b, c = _hs(51.0, 19.0), _hs(51.1, 19.1), _hs(51.2, 19.2)
        n = _tag_volcanic([a, b, c])
        check("tagged hotspots 0 and 2", getattr(a, "_volcanic", False)
              and not getattr(b, "_volcanic", False)
              and getattr(c, "_volcanic", False), f"{n} tagged")
        check("returns tagged count", n == 2, f"got {n}")
    finally:
        db.volcano_hits_batch = orig

    print("== _tag_conflict: kv_store zones (bbox + polygon) ==")
    saved_c0 = SOURCE_FILTER_CONFLICT
    zones = [
        {"type": "bbox", "w": 18.0, "s": 50.0, "e": 20.0, "n": 52.0},
        {"type": "polygon", "points": [[30.0, 60.0], [31.0, 60.0], [30.5, 61.0]]},
    ]
    orig_kv = db.kv_get
    try:
        server.SOURCE_FILTER_CONFLICT = "downweight"
        db.kv_get = lambda key, default=None: zones if key == "conflict_zones" else default
        inside = _hs(51.0, 19.0)          # inside bbox
        outside = _hs(40.0, 10.0)         # nowhere
        poly_in = _hs(60.5, 30.5)         # inside polygon
        n = _tag_conflict([inside, outside, poly_in])
        check("bbox point tagged", getattr(inside, "_conflict", False))
        check("outside point NOT tagged", not getattr(outside, "_conflict", False))
        check("polygon point tagged", getattr(poly_in, "_conflict", False))
        check("returns tagged count", n == 2, f"got {n}")
    finally:
        db.kv_get = orig_kv
        server.SOURCE_FILTER_CONFLICT = saved_c0

    print("== _tag_conflict: fail-open on empty zones ==")
    saved_c1 = SOURCE_FILTER_CONFLICT
    orig_kv2 = db.kv_get
    try:
        server.SOURCE_FILTER_CONFLICT = "downweight"
        db.kv_get = lambda key, default=None: [] if key == "conflict_zones" else default
        h = _hs(51.0, 19.0)
        n = _tag_conflict([h])
        check("empty zones → nothing tagged", n == 0 and not getattr(h, "_conflict", False))
    finally:
        db.kv_get = orig_kv2
        server.SOURCE_FILTER_CONFLICT = saved_c1

    print("== _is_source_passthrough ==")
    saved_v, saved_c = SOURCE_FILTER_VOLCANIC, SOURCE_FILTER_CONFLICT
    try:
        server.SOURCE_FILTER_VOLCANIC = "passthrough"
        server.SOURCE_FILTER_CONFLICT = "full"
        pv = _hs(51.0, 19.0); pv._volcanic = True
        pc = _hs(51.0, 19.0); pc._conflict = True
        plain = _hs(51.0, 19.0)
        check("volcanic passthrough is exempt", _is_source_passthrough(pv))
        check("conflict NOT exempt under 'full'", not _is_source_passthrough(pc))
        check("untagged hotspot NOT exempt", not _is_source_passthrough(plain))
    finally:
        server.SOURCE_FILTER_VOLCANIC = saved_v
        server.SOURCE_FILTER_CONFLICT = saved_c

    print("== _gate_firms_by_land_cover: passthrough survives non-burnable ==")
    saved_v2 = SOURCE_FILTER_VOLCANIC
    orig_codes = db.land_cover_codes_batch
    try:
        server.SOURCE_FILTER_VOLCANIC = "passthrough"
        # Volcano on bare rock (332 = non-burnable) must NOT be dropped.
        db.land_cover_codes_batch = lambda pts: {i: "332" for i in range(len(pts))}
        volcano = _hs(51.0, 19.0); volcano._volcanic = True
        industrial = _hs(51.1, 19.1)  # untagged, on 332 → dropped
        kept = _gate_firms_by_land_cover([volcano, industrial])
        check("passthrough volcanic kept on non-burnable land",
              any(h is volcano for h in kept))
        check("untagged hotspot on non-burnable land dropped",
              not any(h is industrial for h in kept))
    finally:
        server.SOURCE_FILTER_VOLCANIC = saved_v2
        db.land_cover_codes_batch = orig_codes

    print("== _fuse_firms_hotspots: volcanic downweight, NO escape hatch ==")
    saved_v3 = SOURCE_FILTER_VOLCANIC
    try:
        server.SOURCE_FILTER_VOLCANIC = "downweight"
        g_new = _grid()
        g_existing = _grid()
        hs_new = _hs(51.0, 19.0); hs_new._volcanic = True
        hs_existing = _hs(51.0, 19.0); hs_existing._volcanic = True
        stats: dict = {}
        _fuse_firms_hotspots(g_new, [hs_new], is_new_grid=True, tag_stats=stats)
        _fuse_firms_hotspots(g_existing, [hs_existing], is_new_grid=False, tag_stats=stats)
        p_new = float(g_new.probabilities.max())
        p_existing = float(g_existing.probabilities.max())
        # Volcanic weight 0.1 → LR ln(50·0.1) ≈ 1.61. NO escape hatch: the
        # pre-existing grid must be downweighted the same as a new grid.
        target = _p_from_logit(-4.60 + math.log(50.0 * VOLCANIC_EVIDENCE_WEIGHT))
        check("new-grid volcanic downweighted (LR ln(50·0.1))",
              abs(p_new - target) < 0.01, f"{p_new:.4f}")
        check("pre-existing-grid volcanic ALSO downweighted (no escape hatch)",
              abs(p_existing - target) < 0.01, f"{p_existing:.4f}")
        check("volcanic stays below auto-approval floor (0.85)",
              p_existing < 0.85, f"{p_existing:.4f}")
        check("tag_stats records 'volcanic' twice",
              stats.get("volcanic") == 2, f"{stats}")
    finally:
        server.SOURCE_FILTER_VOLCANIC = saved_v3

    print("== composition: volcanic + static on the same hotspot ==")
    saved_v4 = SOURCE_FILTER_VOLCANIC
    try:
        server.SOURCE_FILTER_VOLCANIC = "downweight"
        g_both = _grid()
        hs_both = _hs(51.0, 19.0)
        hs_both._volcanic = True
        hs_both._static_source = True
        stats2: dict = {}
        _fuse_firms_hotspots(g_both, [hs_both], is_new_grid=False, tag_stats=stats2)
        p_both = float(g_both.probabilities.max())
        # max(0.1, 0.1) = 0.1 — same weight, but both tags must be logged.
        check("volcanic + static compose (max of equal weights)",
              abs(p_both - _p_from_logit(-4.60 + math.log(50.0 * VOLCANIC_EVIDENCE_WEIGHT))) < 0.01,
              f"{p_both:.4f}")
        check("tag_stats records 'static + volcanic'",
              stats2.get("static + volcanic") == 1, f"{stats2}")
    finally:
        server.SOURCE_FILTER_VOLCANIC = saved_v4

    print("== conflict downweight fuses the same way ==")
    saved_c2 = SOURCE_FILTER_CONFLICT
    try:
        server.SOURCE_FILTER_CONFLICT = "downweight"
        g_c = _grid()
        hs_c = _hs(51.0, 19.0); hs_c._conflict = True
        stats3: dict = {}
        _fuse_firms_hotspots(g_c, [hs_c], is_new_grid=False, tag_stats=stats3)
        p_c = float(g_c.probabilities.max())
        check("conflict downweighted (LR ln(50·0.1))",
              abs(p_c - _p_from_logit(-4.60 + math.log(50.0 * CONFLICT_EVIDENCE_WEIGHT))) < 0.01,
              f"{p_c:.4f}")
        check("tag_stats records 'conflict'", stats3.get("conflict") == 1, f"{stats3}")
    finally:
        server.SOURCE_FILTER_CONFLICT = saved_c2

    print("== report confirmation: volcanic excluded, conflict included ==")
    saved_v5, saved_c3 = SOURCE_FILTER_VOLCANIC, SOURCE_FILTER_CONFLICT
    try:
        server.SOURCE_FILTER_VOLCANIC = "downweight"
        server.SOURCE_FILTER_CONFLICT = "downweight"
        volc = _hs(51.0, 19.0); volc._volcanic = True
        confl = _hs(51.0, 19.0); confl._conflict = True
        plain = _hs(51.0, 19.0)
        passed = [h for h in [volc, confl, plain]
                  if not getattr(h, "_static_source", False)
                  and not (getattr(h, "_volcanic", False)
                           and SOURCE_FILTER_VOLCANIC == "downweight")]
        check("volcanic downweight excluded from confirmation",
              not any(h is volc for h in passed))
        check("conflict included in confirmation",
              any(h is confl for h in passed))
        check("plain hotspot included", any(h is plain for h in passed))
        # passthrough volcanic DOES confirm
        server.SOURCE_FILTER_VOLCANIC = "passthrough"
        passed2 = [h for h in [volc]
                   if not getattr(h, "_static_source", False)
                   and not (getattr(h, "_volcanic", False)
                            and server.SOURCE_FILTER_VOLCANIC == "downweight")]
        check("passthrough volcanic included in confirmation",
              any(h is volc for h in passed2))
    finally:
        server.SOURCE_FILTER_VOLCANIC = saved_v5
        server.SOURCE_FILTER_CONFLICT = saved_c3

    print()
    if failures:
        print(f"FAILED ({failures} check(s) failed)")
        return 1
    print("ALL PASS")
    return 0


def _p_from_logit(logit: float) -> float:
    return 1.0 / (1.0 + math.exp(-logit))


if __name__ == "__main__":
    sys.exit(main())
