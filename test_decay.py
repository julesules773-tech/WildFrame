#!/usr/bin/env python3
"""
test_decay.py — Regression test for the evidence-gated decay in predict()
=========================================================================

Locks down the decay formula so it can never silently regress (an earlier
edit squared the base half-life, making decay ~10800x too slow — only an
ad-hoc script caught it).

Run from the project root:  .venv/bin/python test_decay.py
Exit code is non-zero on failure.
"""

from __future__ import annotations

import math
import sys

import numpy as np

from bayesian_filter import (
    DECAY_HALF_LIFE_S,
    DECAY_LAMBDA,
    EVIDENCE_SHELF_LIFE_S,
    BayesianFireGrid,
    Evidence,
)

T0 = 1_700_000_000.0  # arbitrary "now" for the replay


def _grid() -> BayesianFireGrid:
    return BayesianFireGrid(center_lat=51.0, center_lon=19.0, cell_size_m=100, nx=60, ny=60)


def _fade(gap_h: float, decay_scale: float = 1.0, burn_threshold: float = 0.9999) -> float:
    """Advance a single-evidence grid by gap_h hours with spread disabled
    (burn_threshold ~1) — isolates pure decay. Returns the centre cell p."""
    g = _grid()
    g.update(Evidence.satellite_hotspot(51.0, 19.0), at=T0)
    p0 = float(g.probabilities[g.ny // 2, g.nx // 2])
    t, remaining = T0, gap_h * 3600.0
    while remaining > 0:
        step = min(remaining, 600.0)
        t += step
        remaining -= step
        g.predict(
            dt=step, wind_speed=0.0, wind_dir_deg=0.0,
            burn_threshold=burn_threshold, decay_scale=decay_scale, at=t,
        )
    return p0, float(g.probabilities[g.ny // 2, g.nx // 2])


def main() -> int:
    checks = []

    # 1) Fresh evidence within the shelf window: p follows the analytic
    #    integral  p = p0 * exp(-ln2 * T^2 / (2 * BASE * SHELF))  for T < SHELF.
    B, S = DECAY_HALF_LIFE_S, EVIDENCE_SHELF_LIFE_S
    T = 10 * 3600.0
    assert T < S, "test gap must sit strictly inside the shelf window"
    p0, p = _fade(T / 3600.0)
    expect = p0 * math.exp(-math.log(2) * T * T / (2 * B * S))
    checks.append(("fresh-evidence 10h gap matches analytic integral",
                   abs(p - expect) < 0.01, f"p={p:.4f} expect={expect:.4f}"))

    # 2) Stale / never-observed cells decay EXACTLY as the old uniform model
    #    (ramp = 1): p = p0 * exp(-DECAY_LAMBDA * dt / max(1, decay_scale)).
    #    Simulate by aging a cell past the shelf window: stamp evidence at
    #    T0 - SHELF - 1h, then run a 3h gap entirely outside the window.
    g = _grid()
    g.update(Evidence.satellite_hotspot(51.0, 19.0), at=T0 - S - 3600.0)
    p0_stale = float(g.probabilities[g.ny // 2, g.nx // 2])
    t, remaining = T0, 3 * 3600.0
    while remaining > 0:
        step = min(remaining, 600.0)
        t += step
        remaining -= step
        g.predict(dt=step, wind_speed=0.0, wind_dir_deg=0.0,
                  burn_threshold=0.9999, at=t)
    expect_stale = p0_stale * math.exp(-DECAY_LAMBDA * 3 * 3600.0)
    checks.append(("stale evidence reduces to the old uniform decay",
                   abs(g.probabilities[g.ny // 2, g.nx // 2] - expect_stale) < 0.01,
                   f"p={g.probabilities[g.ny // 2, g.nx // 2]:.4f} expect={expect_stale:.4f}"))

    # 3) decay_scale (EFFIS DMC) still lengthens the half-life.
    _, p_ds1 = _fade(24.0, decay_scale=1.0)
    _, p_ds2 = _fade(24.0, decay_scale=2.0)
    checks.append(("decay_scale 2x fades slower than 1x",
                   p_ds2 > p_ds1, f"ds=1 -> {p_ds1:.4f}, ds=2 -> {p_ds2:.4f}"))

    # 4) Never-observed background cells decay back toward their prior.
    g = _grid()
    g.update(Evidence.satellite_hotspot(51.0, 19.0), at=T0)
    t, remaining = T0, 5 * 3600.0
    while remaining > 0:
        step = min(remaining, 600.0)
        t += step
        remaining -= step
        g.predict(dt=step, wind_speed=0.0, wind_dir_deg=0.0, at=t)
    bg = float(g.probabilities[5, 5])
    checks.append(("background cell decays to its prior", bg <= 0.0101,
                   f"p={bg:.6f}"))

    # 5) Evidence-gated decay must not depend on wall clock when `at` is
    #    given (replay determinism): same input -> same output.
    r1 = _fade(6.0)
    r2 = _fade(6.0)
    checks.append(("replay with `at` is deterministic", r1 == r2, f"{r1}"))

    print(f"evidence shelf life = {S / 3600:.1f}h, base half-life = {B / 3600:.1f}h")
    failed = 0
    for name, ok, detail in checks:
        print(f"  {'OK ' if ok else 'FAIL'} {name}  ({detail})")
        failed += 0 if ok else 1
    print(f"\n{'ALL PASS' if failed == 0 else f'{failed} FAILED'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
