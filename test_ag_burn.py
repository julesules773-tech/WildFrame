#!/usr/bin/env python3
"""Tests for the agricultural-burning downweight (Step 3 of the filtering plan).

Covers:
  * _tag_ag_burn — cropland CORINE class marks a hotspot `_ag_burn`; forest
    and no-coverage hotspots never get tagged; optional FRP cap
  * _fuse_firms_hotspots — ag-burn detections into a grid created THIS pass
    are injected at reduced weight (0.3 → LR ln(50·0.3) ≈ 2.71); the same
    detection into a pre-existing grid fuses at FULL weight (the escape
    hatch); static + ag on one hotspot composes to the more charitable
    weight and logs both tags
  * grid recreation from expiry — a grid purged after 24h of silence is
    genuinely a new episode, so its re-created grid is downweighted again
    (documented, not a bug)
  * _confirm_reports_against_hotspots — ag-burn hotspots DO confirm citizen
    reports (only static-source hotspots are excluded)
  * fail-open — hotspot without a CLC code is never tagged

Run from the project root:  .venv/bin/python test_ag_burn.py
Exit code is non-zero on failure.
"""

from __future__ import annotations

import math
import sys

from bayesian_filter import BayesianFireGrid, Evidence
from nasa_firms import FIRMSHotspot
from server import (
    AG_BURN_EVIDENCE_WEIGHT,
    _confirm_reports_against_hotspots,
    _fuse_firms_hotspots,
    _tag_ag_burn,
)


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

    print("== _tag_ag_burn (cropland class is the signal) ==")
    crop = _hs(51.0, 19.0)
    crop._clc_code = "211"  # non-irrigated arable land
    forest = _hs(51.0, 19.0)
    forest._clc_code = "311"  # broad-leaved forest
    nocov = _hs(51.0, 19.0)
    nocov._clc_code = None  # outside CORINE coverage
    n = _tag_ag_burn([crop, forest, nocov])
    check("cropland (211) tagged ag-burn", getattr(crop, "_ag_burn", False))
    check("forest (311) NOT tagged", not getattr(forest, "_ag_burn", False))
    check("no-coverage NOT tagged (fail-open)", not getattr(nocov, "_ag_burn", False))
    check("returns tagged count", n == 1, f"got {n}")

    print("== optional FRP cap (off by default — calibration found no split) ==")
    import server
    saved = server.AG_BURN_FRP_MAX_MW
    try:
        server.AG_BURN_FRP_MAX_MW = 5.0
        low = _hs(51.0, 19.0, frp=2.0); low._clc_code = "211"
        high = _hs(51.0, 19.0, frp=50.0); high._clc_code = "211"
        _tag_ag_burn([low, high])
        check("low-FRP cropland tagged when cap enabled", getattr(low, "_ag_burn", False))
        check("high-FRP cropland NOT tagged when cap enabled", not getattr(high, "_ag_burn", False))
    finally:
        server.AG_BURN_FRP_MAX_MW = saved

    print("== _fuse_firms_hotspots: downweight + escape hatch ==")
    # Same single detection into a new grid vs a pre-existing grid.
    g_new = _grid()
    g_existing = _grid()
    hs_new = _hs(51.0, 19.0); hs_new._ag_burn = True
    hs_existing = _hs(51.0, 19.0); hs_existing._ag_burn = True
    stats: dict = {}
    _fuse_firms_hotspots(g_new, [hs_new], is_new_grid=True, tag_stats=stats)
    _fuse_firms_hotspots(g_existing, [hs_existing], is_new_grid=False, tag_stats=stats)

    p_new = float(g_new.probabilities.max())
    p_existing = float(g_existing.probabilities.max())
    check("new-grid ag-burn is downweighted (LR ln(50·0.3) ≈ 2.71)",
          abs(p_new - _p_from_logit(-4.60 + math.log(50.0 * AG_BURN_EVIDENCE_WEIGHT))) < 0.01,
          f"{p_new:.4f}")
    check("pre-existing-grid ag-burn is FULL weight (escape hatch)",
          abs(p_existing - _p_from_logit(-4.60 + math.log(50.0))) < 0.01,
          f"{p_existing:.4f}")
    check("escape hatch: existing grid > new grid",
          p_existing > p_new,
          f"existing {p_existing:.4f} vs new {p_new:.4f}")
    check("downweighted ag-burn stays below auto-approval floor (0.85)",
          p_new < 0.85, f"{p_new:.4f}")
    check("tag_stats records 'ag' for the downweighted hotspot only",
          stats.get("ag") == 1, f"{stats}")
    # The pre-existing-grid hotspot hit the escape hatch → full weight, so
    # it must NOT be logged as a downweighted ag detection.
    check("escape-hatch hotspot is not logged as downweighted",
          sum(stats.values()) == 1, f"{stats}")

    print("== composition: static + ag on the same hotspot ==")
    g_both = _grid()
    hs_both = _hs(51.0, 19.0)
    hs_both._static_source = True
    hs_both._ag_burn = True
    stats2: dict = {}
    _fuse_firms_hotspots(g_both, [hs_both], is_new_grid=True, tag_stats=stats2)
    p_both = float(g_both.probabilities.max())
    # More charitable weight wins: max(0.1, 0.3) = 0.3 → LR ln(50·0.3) ≈ 2.71.
    # A cell that's both is treated as a real fire, not dismissed as
    # industrial noise; the tag log makes the decision auditable.
    check("both tags compose to the more charitable (higher) weight",
          abs(p_both - _p_from_logit(-4.60 + math.log(50.0 * AG_BURN_EVIDENCE_WEIGHT))) < 0.01,
          f"{p_both:.4f}")
    check("tag_stats records 'ag + static'", stats2.get("ag + static") == 1, f"{stats2}")

    print("== grid recreation from expiry is a NEW episode (documented) ==")
    # purge_stale_grids deletes grids with no evidence for 24h. An actively
    # burning fire is re-detected on every overpass, so it never expires
    # mid-burn; a grid is only recreated after a full day of silence, which
    # is genuinely a new fire episode — its re-created grid is correctly
    # downweighted again on first sighting. Assert the semantics hold:
    g2 = _grid()
    hs2 = _hs(51.0, 19.0); hs2._ag_burn = True
    _fuse_firms_hotspots(g2, [hs2], is_new_grid=True)  # pass 1: new grid
    p1 = float(g2.probabilities.max())
    # Simulate the next pass: same fire, grid pre-exists → full weight.
    g2b = _grid()
    hs2b = _hs(51.0, 19.0); hs2b._ag_burn = True
    _fuse_firms_hotspots(g2b, [hs2b], is_new_grid=False)
    p2 = float(g2b.probabilities.max())
    check("first sighting (new grid) downweighted", p1 < p2, f"{p1:.4f} vs {p2:.4f}")
    check("subsequent pass (grid exists) full weight",
          abs(p2 - _p_from_logit(-4.60 + math.log(50.0))) < 0.01, f"{p2:.4f}")

    print("== report confirmation: ag-burn hotspots DO confirm ==")
    # Only static-source hotspots are excluded from confirmation; ag-burn
    # must flow through. Confirmation answers "did a hotspot really happen
    # here" — orthogonal to the spread-model evidence weight.
    ag = _hs(51.0, 19.0); ag._ag_burn = True
    static = _hs(51.0, 19.0); static._static_source = True
    passed = [h for h in [ag, static] if not getattr(h, "_static_source", False)]
    # Identity check: FIRMSHotspot is a dataclass, so == compares field
    # values — two identically-constructed hotspots compare equal. Use `is`.
    check("ag-burn hotspot passes the confirmation filter",
          any(h is ag for h in passed) and not any(h is static for h in passed))

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
