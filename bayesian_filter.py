#!/usr/bin/env python3
"""
bayesian_filter.py — Bayesian Wildfire Probability Grid
========================================================

A probabilistic fire-tracking grid that maintains P(cell is burning | all
evidence) for every cell in a local planar grid.  Implements the classic
predict–update cycle of a Bayesian filter (structurally identical to a
Kalman or particle filter), where:

  - **Predict**  advances probability forward using a simplified Rothermel
    spread model (elliptical kernel + exponential decay).
  - **Update**  fuses new evidence via Bayes' rule in log-odds space.

Evidence sources
----------------
  - Satellite hotspot (VIIRS/FIRMS):  high LR (~20:1)
  - Photo showing visible flame:      moderate-high LR (~10:1)
  - Photo showing smoke only:         weak LR (~3:1)

Likelihood ratios are discounted by the spatial uncertainty of the
triangulation ellipse — a tight ellipse concentrates evidence; a wide
ellipse smears it across many cells.

Usage
-----
    from bayesian_filter import BayesianFireGrid, Evidence

    grid = BayesianFireGrid(center_lat=37.727, center_lon=-119.637)
    grid.predict(dt=600.0, wind_speed=15, wind_dir=225)
    grid.update(Evidence.flame(lat=37.727, lon=-119.637, lr=10.0))
    state = grid.export_state()  # { ref, cell_size, nx, ny, probabilities }
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator

import numpy as np


def _convolve_same(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pure-numpy equivalent of scipy.signal.fftconvolve(a, b, mode='same').

    Avoids a scipy dependency (and any scipy install/binary issues) since
    numpy's FFT is all this needs.
    """
    s1 = np.array(a.shape)
    s2 = np.array(b.shape)
    full_shape = s1 + s2 - 1
    fshape = [int(sz) for sz in full_shape]
    fa = np.fft.rfft2(a, fshape)
    fb = np.fft.rfft2(b, fshape)
    full = np.fft.irfft2(fa * fb, fshape)
    # Crop the full convolution down to the 'same' (size of a) output,
    # centered the same way scipy does.
    startind = (full_shape - s1) // 2
    endind = startind + s1
    return full[startind[0]:endind[0], startind[1]:endind[1]]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
R_EARTH = 6_371_000.0  # Earth radius in metres

# Default grid geometry
DEFAULT_CELL_SIZE_M = 100.0   # 100 m — matches the design doc recommendation
DEFAULT_GRID_NX = 120         # 120 cells × 100 m → 12 km coverage
DEFAULT_GRID_NY = 120         # 12 km N–S

# Predict-step defaults
DEFAULT_BURN_THRESHOLD = 0.15   # cells above this probability can spread
DEFAULT_BASE_SPREAD_RATE = 5.0   # m/min — grass fast, forest slower
PROB_MAX = 0.9999                # clamp to avoid log(0)
PROB_MIN = 1e-6                  # clamp to avoid log(0) / div-by-zero

# Decay: halve ≈ every 3 hours without corroboration
DECAY_HALF_LIFE_S = 10800.0  # seconds (3 hours)
DECAY_LAMBDA = math.log(2) / DECAY_HALF_LIFE_S

# Evidence shelf-life: the expected satellite revisit interval. A cell
# whose last evidence is FRESHER than this is treated as an actively
# burning fire — its probability holds, because the absence of a new
# detection is just the sensor cadence, not proof the fire is out. Only
# cells with evidence OLDER than this window fade at the base
# DECAY_HALF_LIFE_S. Default 12 h ≈ the VIIRS revisit cadence that
# produces our FIRMS detections (backtest_grids.py measured that the
# old uniform 3 h half-life erased fires between passes); env-tunable
# via WILDFRAME_EVIDENCE_SHELF_LIFE_S (seconds).
EVIDENCE_SHELF_LIFE_S = float(
    os.environ.get("WILDFRAME_EVIDENCE_SHELF_LIFE_S", str(12 * 3600))
)

# Wind-to-ellipse parameters (simplified Rothermel)
# Head-to-flank ratio at 10 m/s wind
WIND_HEAD_FACTOR = 0.15       # m⁻¹·s — spread multiplier per m/s of wind
WIND_BACK_FACTOR = 0.04       # m⁻¹·s — backing spread multiplier
MAX_ECCENTRICITY = 0.85       # cap ellipse eccentricity
MIN_ECCENTRICITY = 0.10       # near-circular in calm conditions

# Slope factor: spread increase per degree uphill
SLOPE_FACTOR_PER_DEG = 0.04   # ~4 % per degree

# ---------------------------------------------------------------------------
# Geometry helpers (same equirectangular projection as triangulation.py)
# ---------------------------------------------------------------------------

def _deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0


def equirectangular_project(
    lat: float, lon: float, lat0: float, lon0: float
) -> tuple[float, float]:
    """(lat, lon) → (x, y) in metres, local to (lat0, lon0)."""
    rlat = _deg2rad(lat)
    rlon = _deg2rad(lon)
    rlat0 = _deg2rad(lat0)
    rlon0 = _deg2rad(lon0)
    x = (rlon - rlon0) * math.cos(rlat0) * R_EARTH
    y = (rlat - rlat0) * R_EARTH
    return x, y


def equirectangular_unproject(
    x: float, y: float, lat0: float, lon0: float
) -> tuple[float, float]:
    """(x, y) in metres → (lat, lon)."""
    rlat0 = _deg2rad(lat0)
    rlon0 = _deg2rad(lon0)
    lat = math.degrees(y / R_EARTH + rlat0)
    lon = math.degrees(x / (R_EARTH * math.cos(rlat0)) + rlon0)
    return lat, lon


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    """
    A single piece of evidence to fuse into the grid.

    Attributes
    ----------
    lat, lon : float
        Centre of the evidence in decimal degrees.
    log_likelihood_ratio : float
        ln(P(e|burning) / P(e|not burning)). Positive → supports burning.
        Negative → supports not burning (rare but possible).
    spatial_radius_m : float
        Standard deviation of the Gaussian spatial spread. 0 → pin to one cell.
        Set from triangulation semi-major axis: a wide ellipse → large radius.
    source : str
        Human-readable source tag (e.g. "viirs", "photo-flame", "photo-smoke").
    """

    lat: float
    lon: float
    log_likelihood_ratio: float
    spatial_radius_m: float = 0.0
    source: str = "unknown"

    @classmethod
    def satellite_hotspot(cls, lat: float, lon: float) -> "Evidence":
        """VIIRS/FIRMS hotspot — high confidence, known false-positive rate ~2%."""
        # Modern VIIRS thermal anomaly detection has >98% probability of
        # detecting an active fire when a fire is present → LR ≈ 50:1
        # ln(50) ≈ 3.91
        return cls(
            lat=lat, lon=lon,
            log_likelihood_ratio=math.log(50.0),
            spatial_radius_m=DEFAULT_CELL_SIZE_M / 2.0,  # 50m — tight concentration
            source="viirs",
        )

    @classmethod
    def agency_confirm(cls, lat: float, lon: float) -> "Evidence":
        """Government-confirmed incident (CAP alert / agency feed) — the
        strongest evidence WildFrame can receive: a verified ground-truth
        fire, so it outranks even a satellite hotspot. LR ≈ 100:1 → ln ≈ 4.6."""
        return cls(
            lat=lat, lon=lon,
            log_likelihood_ratio=math.log(100.0),
            spatial_radius_m=DEFAULT_CELL_SIZE_M / 2.0,  # tight — confirmed location
            source="agency-confirm",
        )

    @classmethod
    def agency_cancel(cls, lat: float, lon: float) -> "Evidence":
        """Agency 'contained/cancelled' message — the inverse of a confirm.
        Negative LR (ln(1/100) ≈ -4.6) supports *not burning*, pulling the
        grid's probability back down. Because the LR is negative, update()
        does NOT stamp ``last_updated`` — a cancel must never keep a grid
        alive or reset its expiry clock."""
        return cls(
            lat=lat, lon=lon,
            log_likelihood_ratio=math.log(1.0 / 100.0),
            spatial_radius_m=DEFAULT_CELL_SIZE_M / 2.0,
            source="agency-cancel",
        )

    @classmethod
    def visible_flame(
        cls, lat: float, lon: float,
        semi_major: float = 0.0, semi_minor: float = 0.0,
    ) -> "Evidence":
        """
        Photo showing visible flame.  LR discounted by the triangulation
        uncertainty ellipse: wider ellipse → less confident in location →
        lower effective LR and wider spatial smear.

        Base LR = 10:1 → ln ≈ 2.30.  Discounted by ellipse area relative to
        a single cell (~10 000 m²).
        """
        base_lr = 10.0
        # Discount: if ellipse area >> cell area, reduce LR proportionally
        if semi_major > 0 and semi_minor > 0:
            ellipse_area = math.pi * semi_major * semi_minor
            cell_area = DEFAULT_CELL_SIZE_M ** 2
            discount = max(cell_area / (ellipse_area + cell_area), 0.15)
            effective_lr = 1.0 + (base_lr - 1.0) * discount
        else:
            effective_lr = base_lr
            semi_major = semi_minor = DEFAULT_CELL_SIZE_M

        spatial_radius = max(semi_major, semi_minor) * 0.5
        return cls(
            lat=lat, lon=lon,
            log_likelihood_ratio=math.log(effective_lr),
            spatial_radius_m=spatial_radius,
            source="photo-flame",
        )

    @classmethod
    def smoke_only(
        cls, lat: float, lon: float, wind_dir_deg: float | None = None,
    ) -> "Evidence":
        """
        Photo showing smoke only (no visible flame).  Smoke drifts downwind,
        so evidence is spread along the upwind bearing from the smoke.

        Base LR = 3:1 → ln ≈ 1.10.  If wind direction is known, shift the
        evidence centre upwind by a drift offset.
        """
        effective_lat, effective_lon = lat, lon
        if wind_dir_deg is not None:
            # Smoke likely originated upwind; shift ~500 m upwind
            shift_m = 500.0
            θ = _deg2rad(wind_dir_deg)
            dx = shift_m * math.sin(θ)
            dy = shift_m * math.cos(θ)
            # Unproject the shift from local to geographic
            lat_rad = _deg2rad(lat)
            dlat = -dy / R_EARTH
            dlon = -dx / (R_EARTH * math.cos(lat_rad))
            effective_lat += math.degrees(dlat)
            effective_lon += math.degrees(dlon)

        return cls(
            lat=effective_lat, lon=effective_lon,
            log_likelihood_ratio=math.log(3.0),
            spatial_radius_m=500.0,   # smoke drifts — wider uncertainty
            source="photo-smoke",
        )

    @classmethod
    def from_report(
        cls, report: dict, wind_dir_deg: float | None = None,
        ellipse: dict | None = None,
    ) -> "Evidence":
        """
        Build an Evidence object from a report dict.

        Evidence-type resolution (in priority order):
          1. ``smoke_only`` flag explicitly set → use the smoke model
             (which shifts upwind by 500 m when ``wind_dir_deg`` is known).
          2. Uncertainty ellipse from triangulation → use visible-flame model
             with LR discounted by ellipse area.
          3. Report has a ``device_heading`` → likely pointing at flames →
             visible-flame model.
          4. No heading, no ellipse, no ``smoke_only`` flag → likely a smoke
             sighting (conservative), use smoke model with upwind shift
             (Gap 2 fix: this is the path that exercises the upwind-drift
             logic for the majority of citizen reports).

        Parameters
        ----------
        wind_dir_deg : float | None
            Compass wind direction for smoke-drift correction.  Passed through
            from the grid entry's stored wind so that per-grid weather is
            respected (Gap 2 + Gap 3 together).
        ellipse : dict | None
            Triangulation uncertainty ellipse with keys ``semi_major`` and
            ``semi_minor``.
        """
        lat = report["lat"]
        lon = report["lon"]
        source = report.get("source_type", "citizen")

        # Determine evidence type (see docstring for priority)
        if report.get("smoke_only"):
            return cls.smoke_only(lat, lon, wind_dir_deg)

        if ellipse and ellipse.get("semi_major", 0) > 0:
            return cls.visible_flame(
                lat, lon,
                semi_major=ellipse.get("semi_major", 0),
                semi_minor=ellipse.get("semi_minor", 0),
            )

        # Has a heading → likely pointing at visible flames
        if report.get("device_heading") is not None:
            return cls.visible_flame(lat, lon)

        # No heading, no ellipse, no explicit smoke flag → conservative:
        # treat as smoke observation so the upwind-shift logic fires
        # for the majority of citizen reports (Gap 2 actually works now).
        return cls.smoke_only(lat, lon, wind_dir_deg)


# ---------------------------------------------------------------------------
# Elliptical spread kernel
# ---------------------------------------------------------------------------

class SpreadKernel:
    """
    An elliptical 2D Gaussian kernel oriented along the wind direction.

    Parameters
    ----------
    wind_speed : float
        Wind speed in m/s.
    wind_dir_deg : float
        Wind direction in degrees (compass: 0° = north, 90° = east).
    slope_pct : float
        Slope in percent (rise/run × 100).  Positive = uphill.
    slope_aspect_deg : float
        Direction the slope faces (compass degrees).  0 = north-facing.
    base_spread_rate : float
        Base Rothermel spread rate in m/min.
    moisture_factor : float
        Spread-rate multiplier from fuel moisture (EFFIS FFMC). 1.0 = the
        pre-EFFIS wind-only behaviour; <1 damp fuels slow the fire, >1
        bone-dry fuels speed it up. Scales base rate, so head/back/flank
        all expand or contract together.
    """

    def __init__(
        self,
        wind_speed: float = 0.0,
        wind_dir_deg: float = 0.0,
        slope_pct: float = 0.0,
        slope_aspect_deg: float = 0.0,
        base_spread_rate: float = DEFAULT_BASE_SPREAD_RATE,
        moisture_factor: float = 1.0,
    ):
        self.wind_speed = wind_speed
        self.wind_dir = wind_dir_deg
        self.slope_pct = slope_pct
        self.slope_aspect = slope_aspect_deg
        self.base_rate = base_spread_rate * max(0.1, moisture_factor)

        # Compute ellipse parameters
        self._compute_ellipse()

    def _compute_ellipse(self) -> None:
        # Effective wind: vector sum of wind + slope-driven spread
        # Wind-driven head acceleration
        head_factor = 1.0 + self.wind_speed * WIND_HEAD_FACTOR
        back_factor = 1.0 + self.wind_speed * WIND_BACK_FACTOR

        # Slope factor (fire spreads faster uphill)
        slope_deg = math.atan(self.slope_pct / 100.0) * 180.0 / math.pi
        slope_factor = 1.0 + slope_deg * SLOPE_FACTOR_PER_DEG if slope_deg > 0 else 1.0

        # Head: downwind + uphill (if aspect aligns) — but for simplicity
        # we take the dominant direction as wind_dir and scale the ellipse.
        # The spread ellipse: head = fastest, flank = medium, back = slowest
        self.head = self.base_rate * head_factor * slope_factor
        self.back = self.base_rate * back_factor

        # Flank (semi-minor axis) — standard double-ellipse fire model:
        # with a = (head+back)/2 (semi-major) and c = (head-back)/2
        # (focus offset), b = sqrt(a² - c²), which simplifies to the
        # geometric mean of head and back. This is always well-behaved
        # (no clamping needed) and, unlike a plain average, actually
        # narrows the ellipse as head/back diverge.
        self.flank = math.sqrt(self.head * self.back)

        # Eccentricity, computed from the same (a, b) pair as flank so the
        # two stay consistent — e = sqrt(1 - (b/a)²), capped for reporting.
        if self.head > self.back:
            a = (self.head + self.back) / 2.0
            ratio = self.flank / a
            self.eccentricity = min(math.sqrt(max(1 - ratio * ratio, 0.0)), MAX_ECCENTRICITY)
        else:
            self.eccentricity = MIN_ECCENTRICITY

        # Orientation: wind direction (compass → math angle CCW from east)
        theta_math = (90.0 - self.wind_dir) % 360
        self.orientation_deg = theta_math

    def kernel_at(
        self, dx: float, dy: float, dt_minutes: float
    ) -> float:
        """
        Evaluate the elliptical Gaussian kernel at offset (dx, dy) from a
        burning source, for a given time step.

        Returns a weight in [0, 1] indicating what fraction of the source's
        probability mass transfers to this neighbour cell.

        (dx, dy) are in metres in local coordinates with +x = east, +y = north.
        """
        # Rotate offset into the ellipse's aligned frame
        θ = _deg2rad(self.orientation_deg)
        cosθ = math.cos(θ)
        sinθ = math.sin(θ)
        rx = dx * cosθ + dy * sinθ   # aligned-easting
        ry = -dx * sinθ + dy * cosθ  # aligned-northing

        # Spread distance in the head (rx > 0) and flank (ry) directions
        head_dist = self.head * dt_minutes
        flank_dist = self.flank * dt_minutes
        back_dist = self.back * dt_minutes

        # For rx > 0, use head; for rx < 0, use back
        sig_x = head_dist if rx >= 0 else back_dist
        sig_y = flank_dist

        if sig_x <= 0 or sig_y <= 0:
            return 0.0

        # Gaussian weight
        wx = rx / sig_x
        wy = ry / sig_y
        # Use a quadratic decay (simpler than Gaussian, better for CA spread)
        r2 = wx * wx + wy * wy
        if r2 > 4.0:  # > 2 standard deviations → negligible
            return 0.0
        # Smoothly decaying kernel
        return max(0.0, 1.0 - r2 * 0.25)

    def spread_radius_m(self, dt_minutes: float) -> float:
        """Maximum distance the fire can spread in dt_minutes."""
        return max(self.head, self.back, self.flank) * dt_minutes + DEFAULT_CELL_SIZE_M


# ---------------------------------------------------------------------------
# Marching squares contour extraction
# ---------------------------------------------------------------------------

# Cells of gap bridged by the display-level merge (binary closing) in
# export_contour: two fire pockets up to ~2*r cells apart fuse into one
# perimeter (100 m cells -> ~1 km of bridged gap), so fires that are
# clearly one incident render as a single continuous ring instead of many
# small shapes. Display-level only; the stored probabilities and road-risk
# math are untouched. The 0.3 contour of a genuinely fragmented fire still
# splits into several rings once gaps exceed it.
CONTOUR_MERGE_RADIUS_CELLS = 5


def _shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Shift a boolean mask by (dx, dy) cells, zero-filling at the edges."""
    out = np.zeros_like(mask)
    h, w = mask.shape
    if abs(dx) >= h or abs(dy) >= w:
        return out
    if dx >= 0:
        sx0, sx1, tx0, tx1 = 0, h - dx, dx, h
    else:
        sx0, sx1, tx0, tx1 = -dx, h, 0, h + dx
    if dy >= 0:
        sy0, sy1, ty0, ty1 = 0, w - dy, dy, w
    else:
        sy0, sy1, ty0, ty1 = -dy, w, 0, w + dy
    out[tx0:tx1, ty0:ty1] = mask[sx0:sx1, sy0:sy1]
    return out


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """Grow a boolean mask by a Chebyshev (square) disk of radius ``r`` cells."""
    out = mask.copy()
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if dx == 0 and dy == 0:
                continue
            out |= _shift_mask(mask, dx, dy)
    return out


def _binary_closing(mask: np.ndarray, r_dilate: int, r_erode: int | None = None) -> np.ndarray:
    """Morphological closing (dilate then erode) of a boolean mask.

    Bridges gaps of up to ~2*r_dilate cells between separate regions (so a
    cluster of small fire pockets renders as one continuous perimeter
    instead of many near-identical rings) and fills holes smaller than the
    structuring element. ``r_erode`` defaults to ``r_dilate`` (a symmetric
    closing). Peak-preserving: it operates on the mask, not the probability
    values, so established fires are never diluted away.
    """
    if r_erode is None:
        r_erode = r_dilate
    dilated = _dilate(mask, r_dilate)
    return ~_dilate(~dilated, r_erode)


def _point_in_ring(x: float, y: float, ring: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test against a closed ring (first point
    repeated at the end). Points exactly on the boundary count as outside."""
    inside = False
    n = len(ring) - 1
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def _ring_inside(inner: list[tuple[float, float]], outer: list[tuple[float, float]]) -> bool:
    """True when every vertex of ``inner`` lies strictly inside ``outer``."""
    if len(outer) < 3:
        return False
    return all(_point_in_ring(x, y, outer) for x, y in inner)


def _ring_interior_signal(
    ring: list[tuple[float, float]],
    values: np.ndarray,
    level: float,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    cell_m: float,
) -> int:
    """Classify what a closed ring encloses by sampling the field inside it.

    Returns +1 when the enclosed cells are mostly ABOVE the level (a fire
    boundary), -1 when mostly BELOW (a hole boundary — an unburned gap
    inside a larger fire), 0 when indeterminate. Samples every grid cell
    centre inside the ring's bounding box that passes the point-in-ring
    test and takes the majority, so a fire sitting inside a donut's hole is
    correctly read as a fire (its own cells are above level), not as part of
    the hole."""
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    if cell_m <= 0:
        return 0
    x0 = float(grid_x[0, 0])
    y0 = float(grid_y[0, 0])
    i0 = max(0, int(round((min(xs) - x0) / cell_m)))
    i1 = min(values.shape[0] - 1, int(round((max(xs) - x0) / cell_m)))
    j0 = max(0, int(round((min(ys) - y0) / cell_m)))
    j1 = min(values.shape[1] - 1, int(round((max(ys) - y0) / cell_m)))
    above = below = 0
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            if _point_in_ring(float(grid_x[i, j]), float(grid_y[i, j]), ring):
                if values[i, j] >= level:
                    above += 1
                else:
                    below += 1
    if above + below == 0:
        return 0
    return 1 if above >= below else -1


def _drop_nested_rings(
    chains: list[list[tuple[float, float]]],
    values: np.ndarray,
    level: float,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    cell_m: float,
) -> list[list[tuple[float, float]]]:
    """Drop contours inside contours.

    A closed ring fully inside another ring is either a HOLE boundary (its
    enclosed area is mostly below the level — an unburned gap inside the
    fire, which renders as an inner outline that reads as a bug) or a
    separate fire (enclosed area above the level, e.g. a fire sitting in a
    donut's hole). Hole boundaries are dropped; separate fires are kept.
    Open chains are never dropped (they reach the grid edge, so they cannot
    be nested)."""
    closed = [(i, c) for i, c in enumerate(chains) if len(c) > 2 and c[0] == c[-1]]
    if len(closed) < 2:
        return chains
    drop: set[int] = set()
    for i, ci in closed:
        for j, cj in closed:
            if i == j or i in drop:
                continue
            if _ring_inside(ci, cj) and _ring_interior_signal(
                ci, values, level, grid_x, grid_y, cell_m
            ) < 0:
                drop.add(i)
                break
    if not drop:
        return chains
    return [c for k, c in enumerate(chains) if k not in drop]


def marching_squares_contour(
    values: np.ndarray,
    level: float,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
) -> list[list[tuple[float, float]]]:
    """
    Extract a contour line at a given probability level using marching squares.

    Parameters
    ----------
    values : (nx, ny) numpy array of probabilities.
    level : float — threshold value.
    grid_x, grid_y : (nx, ny) arrays of x,y coordinates at each cell centre.

    Returns
    -------
    List of contour segments. Each segment is a list of (x, y) tuples.
    """
    nx, ny = values.shape
    segments: list[list[tuple[float, float]]] = []

    # Helper: linear interpolation of edge intersection
    def _interp(a_val, b_val, a_pos, b_pos):
        if abs(b_val - a_val) < 1e-12:
            return (a_pos + b_pos) / 2.0
        t = (level - a_val) / (b_val - a_val)
        t = max(0.0, min(1.0, t))
        return a_pos + t * (b_pos - a_pos)

    corner_idx = [(0, 0), (1, 0), (1, 1), (0, 1)]  # vertices of each square

    for i in range(nx - 1):
        for j in range(ny - 1):
            # 4 corner values
            c = [values[i + ci, j + cj] for ci, cj in corner_idx]
            # 4 corner positions (x, y)
            px = [grid_x[i + ci, j + cj] for ci, cj in corner_idx]
            py = [grid_y[i + ci, j + cj] for ci, cj in corner_idx]

            # Binary mask: 1 if above level, 0 if below
            code = sum((1 if c[k] >= level else 0) << k for k in range(4))
            if code == 0 or code == 15:
                continue  # all below or all above

            # Edge intersections (12, 23, 30, 01 → edges e0..e3)
            edges: list[tuple[float, float] | None] = [None] * 4

            # Edge 0: vertex 0 → 1 (bottom)
            if (c[0] >= level) != (c[1] >= level):
                xm = _interp(c[0], c[1], px[0], px[1])
                ym = _interp(c[0], c[1], py[0], py[1])
                edges[0] = (xm, ym)

            # Edge 1: vertex 1 → 2 (right)
            if (c[1] >= level) != (c[2] >= level):
                xm = _interp(c[1], c[2], px[1], px[2])
                ym = _interp(c[1], c[2], py[1], py[2])
                edges[1] = (xm, ym)

            # Edge 2: vertex 2 → 3 (top)
            if (c[2] >= level) != (c[3] >= level):
                xm = _interp(c[2], c[3], px[2], px[3])
                ym = _interp(c[2], c[3], py[2], py[3])
                edges[2] = (xm, ym)

            # Edge 3: vertex 3 → 0 (left)
            if (c[3] >= level) != (c[0] >= level):
                xm = _interp(c[3], c[0], px[3], px[0])
                ym = _interp(c[3], c[0], py[3], py[0])
                edges[3] = (xm, ym)

            # Collect non-None edges
            pts = [e for e in edges if e is not None]
            if len(pts) == 2:
                segments.append([pts[0], pts[1]])

    # Chain the per-cell 2-point fragments into continuous polylines / closed
    # rings. Marching squares emits one isolated segment per cell; without
    # chaining the client strokes disconnected fragments and the contour
    # never forms an enclosed shape. Shared edge intersections are
    # interpolated by both adjacent cells on the same edge, so they agree to
    # floating-point noise — quantize to ``_CHAIN_KEY_DIGITS`` decimals so
    # they key together (points on different edges differ by >= half a cell,
    # far above the noise floor, so the key cannot collide).
    if not segments:
        return segments
    key_digits = 6

    # Cell spacing (m) from the coordinate arrays — used by the micro-fragment
    # filter and the stitch step below.
    cell_m = float(grid_x[1, 0] - grid_x[0, 0]) if nx > 1 else 0.0

    def _key(p: tuple[float, float]) -> tuple[float, float]:
        return (round(float(p[0]), key_digits), round(float(p[1]), key_digits))

    adjacency: dict[tuple[float, float], list[tuple[float, float]]] = {}
    for a, b in segments:
        ka, kb = _key(a), _key(b)
        adjacency.setdefault(ka, []).append(kb)
        adjacency.setdefault(kb, []).append(ka)

    chained: list[list[tuple[float, float]]] = []
    used: set[tuple[float, float]] = set()
    # Boundary endpoints (degree 1) exist exactly when a region touches the
    # array edge. Starting the walk from one makes a single forward walk
    # cover the whole open chain; starting from an interior point would
    # split it in two and falsely look like a ring.
    boundary = sorted(p for p, nbrs in adjacency.items() if len(nbrs) == 1)
    candidates = boundary + sorted(p for p in adjacency if len(adjacency[p]) != 1)
    for start in candidates:
        if start in used:
            continue
        chain = [start]
        used.add(start)
        # Single forward walk. On a closed ring this loops all the way back
        # to ``start``'s neighbour; on an open chain it ends on the opposite
        # boundary endpoint.
        cur = start
        while True:
            nxt = None
            for nb in adjacency.get(cur, ()):
                if nb not in used:
                    nxt = nb
                    break
            if nxt is None:
                break
            chain.append(nxt)
            used.add(nxt)
            cur = nxt

        if start in adjacency.get(cur, ()):
            # Ring: the walk looped back to ``start``'s neighbour — append
            # ``start`` to close it. No artificial boundary invented.
            chained.append(chain + [start])
        else:
            # Open chain (region touches the grid edge): left open on
            # purpose — closing it would invent a boundary where the fire
            # actually extends past the grid.
            chained.append(chain)

    # Drop micro contours: the discrete spread kernel leaves isolated
    # single-cell probability pockets (a lone p>0.3 cell among ~10%
    # neighbours) whose contours are tiny rings that render as absurdly
    # small shapes on the map. With the display-level merge bridging ~1 km
    # gaps, any contour below 8 cells of perimeter is a fragment — a
    # pocket the merge should have absorbed or a neck-split artifact — so
    # it is dropped (open or closed). A fire that small isn't meaningful to
    # display and the road-risk / contour math never relied on it.
    if cell_m > 0:
        chained = [
            c for c in chained
            if _chain_perimeter(c) >= 8.0 * cell_m
        ]

    # Stitch open chains whose endpoints are within ~1 cell of each other.
    # These are the same contour point separated by marching-squares
    # interpolation noise (e.g. around a diagonally-touching pocket), which
    # leaves an almost-closed ring with a tiny visible gap. Joining them
    # makes the ring enclosed again. Endpoints on the grid edge are excluded
    # — those are genuine open contours (fire extends past the grid).
    if cell_m > 0:
        stitch_tol = 1.0 * cell_m
        stitched: list[list[tuple[float, float]]] = []
        for c in chained:
            if len(c) < 2 or c[0] == c[-1]:
                stitched.append(c)
                continue
            a, b = c[0], c[-1]
            a_edge = _on_grid_edge(a, grid_x, grid_y)
            b_edge = _on_grid_edge(b, grid_x, grid_y)
            if not a_edge and not b_edge and _dist(a, b) <= stitch_tol:
                stitched.append(c + [a])  # close the ring
            else:
                stitched.append(c)
        chained = stitched

    # Drop contours inside contours: a donut's hole ring (its enclosed area
    # is below the level) renders as an inner outline that reads as a bug.
    # A separate fire inside a hole keeps its ring. Open chains (grid-edge
    # regions) are never nested and stay.
    chained = _drop_nested_rings(chained, values, level, grid_x, grid_y, cell_m)

    return chained


def _chain_perimeter(chain: list[tuple[float, float]]) -> float:
    """Total length of a contour chain (grid units). For a closed ring the
    closing edge (last point back to first) is counted by the loop since the
    chain already ends where it started."""
    if len(chain) < 2:
        return 0.0
    total = 0.0
    last = chain[0]
    for pt in chain[1:]:
        total += math.hypot(pt[0] - last[0], pt[1] - last[1])
        last = pt
    return total


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _on_grid_edge(
    p: tuple[float, float],
    grid_x: np.ndarray,
    grid_y: np.ndarray,
) -> bool:
    """True when ``p`` sits on the outermost grid coordinate line (the fire
    region reaches the array boundary there)."""
    x, y = p
    return (
        abs(x - float(grid_x.min())) < 1e-6
        or abs(x - float(grid_x.max())) < 1e-6
        or abs(y - float(grid_y.min())) < 1e-6
        or abs(y - float(grid_y.max())) < 1e-6
    )


# ---------------------------------------------------------------------------
# The Bayesian Grid
# ---------------------------------------------------------------------------

class BayesianFireGrid:
    """
    A 2D grid of cells, each storing P(cell is burning | all evidence).

    Internally operates in log-odds space for numerical stability.

    Parameters
    ----------
    center_lat, center_lon : float
        Geographic centre of the grid.
    cell_size_m : float
        Size of each cell in metres (square cells).
    nx, ny : int
        Number of cells in the x (E–W) and y (N–S) directions.
    """

    def __init__(
        self,
        center_lat: float,
        center_lon: float,
        cell_size_m: float = DEFAULT_CELL_SIZE_M,
        nx: int = DEFAULT_GRID_NX,
        ny: int = DEFAULT_GRID_NY,
    ):
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.cell_size = cell_size_m
        self.nx = nx
        self.ny = ny

        # Grid extent in local coordinates
        self.half_x = nx * cell_size_m / 2.0
        self.half_y = ny * cell_size_m / 2.0

        # Log-odds grid: ln(p/(1-p)) for each cell
        # Start uniform: p = 0.01 → logit = ln(0.01/0.99) ≈ -4.60
        self.logits: np.ndarray = np.full((nx, ny), math.log(0.01 / 0.99), dtype=np.float64)
        self.probabilities: np.ndarray = np.full((nx, ny), 0.01, dtype=np.float64)

        # Last-updated timestamps (unix epoch seconds)
        self.last_updated: np.ndarray = np.zeros((nx, ny), dtype=np.float64)

        # Precompute grid coordinates (cell centres)
        self._build_coords()

        self.last_predict_time: float = 0.0

    def _build_coords(self) -> None:
        """Precompute the (x, y) and (lat, lon) for each cell centre.

        Uses numpy-vectorized equirectangular unprojection instead of a
        Python double loop — this is ~1000× faster for large grids.
        """
        x0 = -self.half_x + self.cell_size / 2.0
        y0 = -self.half_y + self.cell_size / 2.0

        xs = np.linspace(x0, x0 + (self.nx - 1) * self.cell_size, self.nx)
        ys = np.linspace(y0, y0 + (self.ny - 1) * self.cell_size, self.ny)
        self.grid_x, self.grid_y = np.meshgrid(xs, ys, indexing="ij")

        # Vectorized equirectangular unprojection
        rlat0 = np.radians(self.center_lat)
        rlon0 = np.radians(self.center_lon)
        self.grid_lat = np.degrees(self.grid_y / R_EARTH + rlat0)
        self.grid_lon = np.degrees(self.grid_x / (R_EARTH * np.cos(rlat0)) + rlon0)

    def _compute_probs(self) -> None:
        """Sync the probability array from logits."""
        # Clamp to avoid overflow
        clipped = np.clip(self.logits, -30.0, 30.0)
        self.probabilities = 1.0 / (1.0 + np.exp(-clipped))
        # Ensure bounds
        self.probabilities = np.clip(self.probabilities, PROB_MIN, PROB_MAX)

    def _logit_to_prob(self, logit: float) -> float:
        if logit > 30:
            return PROB_MAX
        if logit < -30:
            return PROB_MIN
        return max(PROB_MIN, min(PROB_MAX, 1.0 / (1.0 + math.exp(-logit))))

    def _prob_to_logit(self, prob: float) -> float:
        p = max(PROB_MIN, min(PROB_MAX, prob))
        return math.log(p / (1.0 - p))

    @staticmethod
    def _prob_to_logit_array(prob: np.ndarray) -> np.ndarray:
        """Vectorized version of _prob_to_logit for numpy arrays."""
        p = np.clip(prob, PROB_MIN, PROB_MAX)
        return np.log(p / (1.0 - p))

    # ------------------------------------------------------------------
    # Coordinate conversions
    # ------------------------------------------------------------------

    def latlon_to_cell(self, lat: float, lon: float) -> tuple[int, int]:
        """Get the grid cell (i, j) containing a geographic point."""
        x, y = equirectangular_project(lat, lon, self.center_lat, self.center_lon)
        i = int((x - (-self.half_x)) / self.cell_size)
        j = int((y - (-self.half_y)) / self.cell_size)
        return max(0, min(self.nx - 1, i)), max(0, min(self.ny - 1, j))

    def cell_to_latlon(self, i: int, j: int) -> tuple[float, float]:
        """Geographic centre of cell (i, j)."""
        return self.grid_lat[i, j], self.grid_lon[i, j]

    # ------------------------------------------------------------------
    # Predict step
    # ------------------------------------------------------------------

    def predict(
        self,
        dt: float,
        wind_speed: float = 0.0,
        wind_dir_deg: float = 0.0,
        slope_pct: float = 0.0,
        slope_aspect_deg: float = 0.0,
        burn_threshold: float = DEFAULT_BURN_THRESHOLD,
        moisture_factor: float = 1.0,
        decay_scale: float = 1.0,
        at: float | None = None,
    ) -> None:
        """
        Advance the probability grid by time dt (seconds).

        1. For each cell with p > burn_threshold, spread probability mass
           to neighbouring cells using an elliptical spread kernel (scaled
           by ``moisture_factor`` from EFFIS FFMC when fuel-moisture data
           is available).
        2. Apply evidence-gated exponential decay: a cell's probability
           only fades once its last evidence has aged past the expected
           satellite revisit window (``EVIDENCE_SHELF_LIFE_S``); while the
           evidence is fresh the effective half-life stretches, so a fire
           detected once per pass stays visible between passes. Stale and
           never-observed cells decay with half-life DECAY_HALF_LIFE_S as
           before. ``decay_scale`` > 1 lengthens the base half-life —
           EFFIS DMC's "deep duff keeps the fire smouldering" effect.

        ``at`` overrides the wall clock (used by the backtest harness to
        replay history faithfully); production always leaves it None.
        """
        now = at if at is not None else datetime.now(timezone.utc).timestamp()
        dt_minutes = dt / 60.0

        # Compute the spread kernel
        kernel = SpreadKernel(
            wind_speed=wind_speed,
            wind_dir_deg=wind_dir_deg,
            slope_pct=slope_pct,
            slope_aspect_deg=slope_aspect_deg,
            moisture_factor=moisture_factor,
        )

        spread_radius = kernel.spread_radius_m(dt_minutes)
        radius_cells = int(math.ceil(spread_radius / self.cell_size))

        # --- Spread probability from burning cells ---
        # We build a temporary array of delta-logits to apply in one pass
        delta_logits = np.zeros_like(self.logits)

        burning_mask = self.probabilities > burn_threshold

        if np.any(burning_mask):
            # The kernel weight between a source cell and a neighbour depends
            # only on their relative offset (dx, dy) — never on which cell is
            # actually burning. So instead of looping over every
            # (burning cell, neighbour) pair in Python (which scales as
            # burning_cells * radius_cells^2, and both of those blow up
            # together as cell_size shrinks), build the weight stencil ONCE
            # and apply it to every burning cell at once via a 2D
            # convolution. This is the same math, just computed in
            # vectorized/compiled form instead of a Python double loop.
            offsets = np.arange(-radius_cells, radius_cells + 1)
            stencil = np.zeros((len(offsets), len(offsets)))
            for oi, di in enumerate(offsets):
                for oj, dj in enumerate(offsets):
                    if di == 0 and dj == 0:
                        continue
                    dx = di * self.cell_size
                    dy = dj * self.cell_size
                    stencil[oi, oj] = kernel.kernel_at(dx, dy, dt_minutes)

            mass_transferred = stencil.sum()

            if mass_transferred > 0:
                # Max 15% of source mass moves per dt, distributed across
                # the stencil in proportion to kernel weight (same as before)
                fraction_stencil = 0.15 * stencil / mass_transferred

                # Probability mass only where actually burning
                source_field = np.where(burning_mask, self.probabilities, 0.0)

                # contribution[n] = sum_b source_field[b] * fraction_stencil[n - b]
                # — exactly what the nested loop computed, as a convolution.
                # NOTE: near the grid edge, 'same' mode zero-pads rather than
                # renormalizing over fewer in-bounds neighbours (unlike the
                # old per-cell renormalization), so boundary-adjacent cells
                # transfer slightly less than 15% of their mass instead of
                # redistributing it inward. In practice this just means fire
                # spreading toward the mapped edge "exits" the grid rather
                # than piling up against it — arguably more physically
                # sensible, and only matters right at the grid boundary.
                contribution = _convolve_same(source_field, fraction_stencil)
                contribution = np.clip(contribution, 0.0, None)

                changed = contribution > 1e-12
                if np.any(changed):
                    current_prob = self.probabilities[changed]
                    new_prob = np.minimum(PROB_MAX, current_prob + contribution[changed])
                    delta_logits[changed] += (
                        self._prob_to_logit_array(new_prob) - self._prob_to_logit_array(current_prob)
                    )

            # The source cells lose a small fraction of their probability
            # (fire physically consuming fuel — happens over hours, not minutes)
            source_decay = 0.02 * (dt_minutes / 60.0)  # ~2% per hour
            source_decay = min(source_decay, 0.5)  # cap at 50% max loss per predict

            current_prob = self.probabilities[burning_mask]
            new_source_prob = np.maximum(
                PROB_MIN, current_prob * (1.0 - source_decay / (1.0 + current_prob))
            )
            delta_logits[burning_mask] += (
                self._prob_to_logit_array(new_source_prob) - self._prob_to_logit_array(current_prob)
            )

        # Apply delta logits
        self.logits += delta_logits

        # --- Apply evidence-gated temporal decay ---
        # Per cell: the effective half-life stretches while the evidence is
        # fresh (age < EVIDENCE_SHELF_LIFE_S) and converges to the base
        # half-life once the evidence ages past the expected revisit
        # window. ramp = shelf_life / age for fresh cells (tau grows as
        # evidence gets fresher), exactly 1 for stale/never-observed
        # cells. age is floored at 1 s so a just-stamped cell has a finite
        # (very long) tau instead of an infinite one.
        self._compute_probs()

        # Effective half-life = DECAY_HALF_LIFE_S * decay_scale * ramp, so
        # the per-step factor is exp(-DECAY_LAMBDA * dt / (decay_scale *
        # ramp)) — with ramp = 1 this reduces EXACTLY to the original
        # uniform decay (DECAY_LAMBDA already embeds the base half-life).
        age = np.maximum(now - self.last_updated, 1.0)  # s since last evidence
        ramp = np.maximum(1.0, EVIDENCE_SHELF_LIFE_S / age)
        denom = max(1.0, decay_scale) * ramp
        self.probabilities *= np.exp(-DECAY_LAMBDA * dt / denom)
        # But keep at least a tiny residual
        self.probabilities = np.clip(self.probabilities, PROB_MIN, PROB_MAX)

        # Re-sync logits
        self.logits = np.log(self.probabilities / (1.0 - self.probabilities))

        # NOTE: last_updated is NOT touched here. It tracks when a cell was
        # last confirmed by real evidence (see update()), not when the model
        # merely predicted forward. Stamping it here would make every cell
        # look "just observed" after every predict step, which defeats its
        # purpose (e.g. rendering confirmed vs. predicted-only cells
        # differently).
        self.last_predict_time = now

    # ------------------------------------------------------------------
    # Update step (Bayes' rule)
    # ------------------------------------------------------------------

    def update(self, evidence: Evidence, at: float | None = None) -> None:
        """
        Fuse a piece of evidence into the grid using Bayes' rule in log-odds.

        For pin-point evidence (spatial_radius_m ≈ 0), the update is applied
        to exactly one cell.  For spatially-uncertain evidence, the log-LR is
        spread as a 2D Gaussian across neighbouring cells, weighted so that
        the total information content equals the original LR.

        ``at`` overrides the wall clock for the per-cell ``last_updated``
        stamp (the backtest harness replays historical detections with
        their real acquisition times; production always leaves it None).
        """
        # Find the central cell
        ci, cj = self.latlon_to_cell(evidence.lat, evidence.lon)

        if evidence.spatial_radius_m <= self.cell_size * 0.5:
            # Pin-point update — all evidence goes to the centre cell
            self.logits[ci, cj] += evidence.log_likelihood_ratio
        else:
            # Spatial spread: 2D Gaussian kernel
            # The centre cell receives the *full* log-LR; neighbouring cells
            # receive a Gaussian-decayed fraction.  This is a soft-evidence
            # model — the total integrated update exceeds the original LR,
            # but the centre cell always gets the full signal regardless of
            # cell size, making the heatmap responsive to sparse evidence.
            sigma = evidence.spatial_radius_m
            radius_cells = int(math.ceil(3.0 * sigma / self.cell_size))

            i_min = max(0, ci - radius_cells)
            i_max = min(self.nx, ci + radius_cells + 1)
            j_min = max(0, cj - radius_cells)
            j_max = min(self.ny, cj + radius_cells + 1)

            for i in range(i_min, i_max):
                for j in range(j_min, j_max):
                    dx = self.grid_x[i, j] - self.grid_x[ci, cj]
                    dy = self.grid_y[i, j] - self.grid_y[ci, cj]
                    r2 = (dx * dx + dy * dy) / (sigma * sigma)
                    if r2 < 9.0:  # within 3 sigma
                        w = math.exp(-0.5 * r2)
                        self.logits[i, j] += evidence.log_likelihood_ratio * w

        # Clamp and re-sync
        np.clip(self.logits, -30.0, 30.0, out=self.logits)
        self._compute_probs()

        # Only positive (supports-burning) evidence stamps a cell as freshly
        # observed. Negative evidence (agency cancels, etc.) must NOT advance
        # last_updated — otherwise a cancel would keep the grid alive in
        # expire_stale and reset last_evidence_at to "confirmed just now"
        # when the opposite is true.
        if evidence.log_likelihood_ratio > 0:
            now = at if at is not None else datetime.now(timezone.utc).timestamp()
            self.last_updated[ci, cj] = now
            if evidence.spatial_radius_m > self.cell_size:
                # Update timestamps for all affected cells
                i_min = max(0, ci - 5)
                i_max = min(self.nx, ci + 6)
                j_min = max(0, cj - 5)
                j_max = min(self.ny, cj + 6)
                self.last_updated[i_min:i_max, j_min:j_max] = now

    # ------------------------------------------------------------------
    # Export / Serialization
    # ------------------------------------------------------------------

    def export_state(self, threshold: float = 0.0) -> dict[str, Any]:
        """
        Export the current grid state for the frontend.

        Only includes cells with probability > threshold (to reduce payload).
        Returns a dict that the frontend can directly render as a heatmap.
        """
        # Ensure probabilities are synced from logits before exporting
        self._compute_probs()

        if threshold > 0:
            mask = self.probabilities > threshold
            indices = np.argwhere(mask)
            cells = []
            for i, j in indices:
                cells.append({
                    "lat": float(self.grid_lat[i, j]),
                    "lon": float(self.grid_lon[i, j]),
                    "p": float(self.probabilities[i, j]),
                })
        else:
            cells = []
            for i in range(self.nx):
                for j in range(self.ny):
                    cells.append({
                        "lat": float(self.grid_lat[i, j]),
                        "lon": float(self.grid_lon[i, j]),
                        "p": float(self.probabilities[i, j]),
                    })

        return {
            "ref_lat": self.center_lat,
            "ref_lon": self.center_lon,
            "cell_size_m": self.cell_size,
            "nx": self.nx,
            "ny": self.ny,
            "cells": cells,
            "count": len(cells),
            "last_predict_time": self.last_predict_time,
        }

    def export_contour(self, level: float = 0.6) -> list[list[list[float]]]:
        """
        Extract a contour (isoline) at the given probability level.

        Returns a list of contour segments, each segment being a list of
        [lat, lon] coordinate pairs.
        """
        # Ensure probabilities are synced from logits before extracting contour
        self._compute_probs()

        # Light 3x3 smoothing (pure numpy, separable 1-2-1) before contouring:
        # the discrete spread kernel leaves isolated single-cell probability
        # pockets (a lone p>0.3 cell among ~10% neighbours) whose contours are
        # tiny disconnected fragments that render as stray, unenclosed line
        # bits on the map. Smoothing merges those pockets into the fire region
        # so the contour can actually enclose. Display-level only — the grid's
        # probabilities are untouched. Full 3x3 Gaussian (weights 1,2,1
        # squared) ≈ sigma 0.7: below one cell, so genuinely separate fires
        # stay separate.
        probs = self.probabilities
        # Smoothing dilutes the peak (a lone 0.33 pocket drops to ~0.09), so
        # only apply it to well-established fires (peak well above the
        # contour level). Borderline fires keep their raw field — chaining +
        # micro-fragment filtering still clean those up.
        established = (
            probs.shape[0] >= 3 and probs.shape[1] >= 3
            and float(np.max(probs)) > 2.0 * level
        )
        if established:
            p = np.asarray(probs, dtype=np.float64)
            p = (p[:-2, :] + 2.0 * p[1:-1, :] + p[2:, :]) / 4.0
            p = (p[:, :-2] + 2.0 * p[:, 1:-1] + p[:, 2:]) / 4.0
            # Re-expand to the full grid with the mean baseline at the rim so
            # grid_x/grid_y stay aligned (the rim is below any contour level
            # anyway).
            padded = np.empty_like(p, shape=probs.shape)
            padded[:] = np.mean(probs)
            padded[1:-1, 1:-1] = p
            probs = padded

        contour_level = level
        if established:
            # Merge nearby shapes into one perimeter: binary closing on the
            # above-level mask bridges gaps of up to ~2*r cells, so a cluster
            # of small fire pockets renders as one continuous ring instead of
            # many near-identical small shapes. Peak-preserving (works on the
            # mask, not the probabilities). Display-level only. The closed
            # mask is re-marched at 0.5 with a light smoothing pass so the
            # outline follows the fused boundary instead of a staircase.
            mask = probs > level
            closed = _binary_closing(mask, CONTOUR_MERGE_RADIUS_CELLS)
            field = closed.astype(np.float64)
            field = (field[:-2, :] + 2.0 * field[1:-1, :] + field[2:, :]) / 4.0
            field = (field[:, :-2] + 2.0 * field[:, 1:-1] + field[:, 2:]) / 4.0
            padded = np.zeros_like(field, shape=probs.shape)
            padded[1:-1, 1:-1] = field
            probs = padded
            contour_level = 0.5

        segments_xy = marching_squares_contour(
            probs, contour_level, self.grid_x, self.grid_y,
        )

        result = []
        for seg in segments_xy:
            geo_seg = []
            for x, y in seg:
                lat, lon = equirectangular_unproject(
                    x, y, self.center_lat, self.center_lon,
                )
                geo_seg.append([lat, lon])
            if len(geo_seg) >= 2:
                result.append(geo_seg)
        return result

    def get_statistics(self) -> dict[str, Any]:
        """Summary statistics of the current grid state."""
        self._compute_probs()
        p = self.probabilities

        # Area estimates
        cell_area_ha = (self.cell_size ** 2) / 10000.0  # m² → hectares

        # Cells above various thresholds
        p05 = np.sum(p > 0.05)
        p10 = np.sum(p > 0.10)
        p25 = np.sum(p > 0.25)
        p50 = np.sum(p > 0.50)

        return {
            "cell_size_m": self.cell_size,
            "grid_cells": self.nx * self.ny,
            "cells_p_above_0_05": int(p05),
            "cells_p_above_0_10": int(p10),
            "cells_p_above_0_25": int(p25),
            "cells_p_above_0_50": int(p50),
            "area_ha_p_above_0_05": round(p05 * cell_area_ha, 1),
            "area_ha_p_above_0_10": round(p10 * cell_area_ha, 1),
            "area_ha_p_above_0_25": round(p25 * cell_area_ha, 1),
            "area_ha_p_above_0_50": round(p50 * cell_area_ha, 1),
            "max_p": float(np.max(p)),
            "mean_p": float(np.mean(p)),
            "last_predict_time": self.last_predict_time,
        }

    def reset(self) -> None:
        """Reset the grid to uniform low probability."""
        self.logits.fill(math.log(0.01 / 0.99))
        self._compute_probs()
        self.last_updated.fill(0.0)
        self.last_predict_time = 0.0

    # ------------------------------------------------------------------
    # Persistence (Postgres JSONB) — serialize the numpy state
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the grid's full state so it can be persisted (Postgres
        JSONB) and reconstructed exactly later, surviving restarts and
        multiple workers.

        The big numpy arrays (logits, last_updated) are stored as base64
        of raw float32 bytes — compact and fast to round-trip. Grid
        coordinates (grid_x/grid_y/grid_lat/grid_lon) are derived purely
        from (center, cell_size, nx, ny), so they are recomputed by
        ``from_dict`` instead of stored.
        """
        import base64

        def _b64(arr: np.ndarray) -> str:
            return base64.b64encode(arr.astype(np.float32).tobytes()).decode("ascii")

        return {
            "format": "b64-f32",
            "center_lat": self.center_lat,
            "center_lon": self.center_lon,
            "cell_size_m": self.cell_size,
            "nx": self.nx,
            "ny": self.ny,
            "logits": _b64(self.logits),
            "last_updated": _b64(self.last_updated),
            "last_predict_time": self.last_predict_time,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BayesianFireGrid":
        """Reconstruct a grid from the dict produced by ``to_dict``."""
        import base64

        def _unb64(b64: str, nx: int, ny: int) -> np.ndarray:
            raw = base64.b64decode(b64)
            return np.frombuffer(raw, dtype=np.float32).astype(np.float64).reshape(nx, ny)

        nx = int(data["nx"])
        ny = int(data["ny"])
        grid = cls(
            center_lat=float(data["center_lat"]),
            center_lon=float(data["center_lon"]),
            cell_size_m=float(data["cell_size_m"]),
            nx=nx,
            ny=ny,
        )
        grid.logits = _unb64(data["logits"], nx, ny)
        grid.last_updated = _unb64(data["last_updated"], nx, ny)
        grid.last_predict_time = float(data.get("last_predict_time", 0.0))
        grid._compute_probs()
        return grid


# ---------------------------------------------------------------------------
# Convenience: build a grid centred on a set of reports
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Grid sizing helper (shared by auto_grid and historical demo)
# ---------------------------------------------------------------------------

# Maximum number of grid cells before coarsening resolution
MAX_GRID_CELLS = 200 * 200  # 40 000 cells max
# Absolute largest cell size (beyond this the heatmap is too blurry)
MAX_CELL_SIZE_M = 5000.0


def auto_grid_size(
    lats: list[float],
    lons: list[float],
    margin_m: float = 5000.0,
) -> dict:
    """
    Compute optimal grid dimensions from a set of latitude/longitude coordinates.

    Returns a dict with keys:
        nx, ny        — number of cells
        cell_size_m   — cell size in metres (may be > 100 if capped)
        center_lat    — centroid latitude
        center_lon    — centroid longitude

    Returns an empty dict if there are fewer than 1 coordinate.
    """
    if len(lats) < 1 or len(lons) < 1:
        return {}

    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)

    # Bounding box in local coords
    xs, ys = [], []
    for lat, lon in zip(lats, lons):
        x, y = equirectangular_project(lat, lon, center_lat, center_lon)
        xs.append(x)
        ys.append(y)

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    span_x = (x_max - x_min) + 2 * margin_m
    span_y = (y_max - y_min) + 2 * margin_m

    cell_size = DEFAULT_CELL_SIZE_M
    nx = max(40, int(math.ceil(span_x / cell_size)))
    ny = max(40, int(math.ceil(span_y / cell_size)))

    if nx * ny > MAX_GRID_CELLS:
        # Coarsen resolution to fit within MAX_GRID_CELLS
        ratio = math.sqrt(nx * ny / MAX_GRID_CELLS)
        cell_size = DEFAULT_CELL_SIZE_M * ratio
        cell_size = min(cell_size, MAX_CELL_SIZE_M)
        nx = max(40, int(math.ceil(span_x / cell_size)))
        ny = max(40, int(math.ceil(span_y / cell_size)))
        # Re-check in case the cap stopped us from reaching the target
        if nx * ny > MAX_GRID_CELLS:
            scale = math.sqrt(nx * ny / MAX_GRID_CELLS)
            nx = max(40, int(nx / scale))
            ny = max(40, int(ny / scale))

    # Round to a reasonable number
    nx = int(math.ceil(nx / 10)) * 10
    ny = int(math.ceil(ny / 10)) * 10

    return {
        "nx": nx,
        "ny": ny,
        "cell_size_m": cell_size,
        "center_lat": center_lat,
        "center_lon": center_lon,
    }


def auto_grid(
    reports: list[dict],
    margin_m: float = 5000.0,
) -> BayesianFireGrid | None:
    """
    Create a BayesianFireGrid auto-sized to cover all confirmed reports.

    Computes the centroid and bounding box of the reports, then creates a
    grid with enough cells to cover the box plus a margin.

    Returns None if there are no confirmed reports.
    """
    confirmed = [r for r in reports if r.get("status") == "confirmed"]
    if not confirmed:
        return None

    lats = [r["lat"] for r in confirmed]
    lons = [r["lon"] for r in confirmed]

    sizing = auto_grid_size(lats, lons, margin_m=margin_m)
    if not sizing:
        return None

    return BayesianFireGrid(
        center_lat=sizing["center_lat"],
        center_lon=sizing["center_lon"],
        cell_size_m=sizing["cell_size_m"],
        nx=sizing["nx"],
        ny=sizing["ny"],
    )


# ---------------------------------------------------------------------------
# Convenience: seed with initial evidence from reports
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Road-risk overlay — continuous spread-rate function
# ---------------------------------------------------------------------------


def effective_spread_rate(
    phi_deg: float,
    head_rate: float,
    back_rate: float,
    flank_rate: float,
) -> float:
    """
    Closed-form effective radial spread rate (m/min) at bearing φ relative to wind.

    φ = 0° means directly downwind (head fire rate).
    φ = 180° means directly upwind (back fire rate).
    φ = ±90° means crosswind (flank fire rate).

    Derivation: the fire ellipse in wind-aligned coords is
      (rx / head)² + (ry / flank)² = 1   for rx ≥ 0 (downwind)
      (rx / back)² + (ry / flank)² = 1   for rx < 0 (upwind)

    In polar form (rx = r·cosφ, ry = r·sinφ), the fire edge at angle φ is at
    distance r = 1 / √( cos²φ/rate² + sin²φ/flank² ), so the effective
    radial spread rate in that direction is r (m/min).
    """
    phi = ((phi_deg + 180) % 360) - 180  # normalise to [-180, 180]
    phi_rad = math.radians(phi)
    cos_phi = math.cos(phi_rad)
    sin_phi = math.sin(phi_rad)

    rate = head_rate if abs(phi) <= 90 else back_rate

    denom = (cos_phi / rate) ** 2 + (sin_phi / flank_rate) ** 2
    if denom < 1e-12:
        return max(head_rate, back_rate, flank_rate)
    return 1.0 / math.sqrt(denom)


ROAD_RISK_TIERS: list[tuple[str, float]] = [
    ("critical", 30.0),    # < 30 minutes
    ("high", 120.0),       # < 2 hours
    ("moderate", 360.0),   # < 6 hours
    ("low", float("inf")),
]


def _risk_haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two lat/lon points."""
    R = 6_371_000
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _initial_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing (compass degrees) from point 1 to point 2."""
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    λ1, λ2 = math.radians(lon1), math.radians(lon2)
    y = math.sin(λ2 - λ1) * math.cos(φ2)
    x = math.cos(φ1) * math.sin(φ2) - math.sin(φ1) * math.cos(φ2) * math.cos(λ2 - λ1)
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360) % 360


# How many wind-at-arrival iterations before we give up on convergence.
MAX_FORECAST_ITERS = 3


def _risk_tier(t_arrival: float) -> str:
    """Bucket an arrival time (minutes) into a risk tier."""
    for tier_name, tier_minutes in ROAD_RISK_TIERS:
        if t_arrival < tier_minutes:
            return tier_name
    return "low"


def _rate_toward(
    segment_bearing: float,
    wind_speed: float,
    wind_dir_deg: float,
    moisture_factor: float,
) -> float:
    """Closed-form effective spread rate (m/min) toward a road bearing under
    a given wind (direction flipped to spread-TOWARD convention)."""
    kernel = SpreadKernel(
        wind_speed=wind_speed,
        wind_dir_deg=wind_dir_deg,
        moisture_factor=moisture_factor,
    )
    phi = (segment_bearing - wind_dir_deg + 360) % 360
    return effective_spread_rate(phi, kernel.head, kernel.back, kernel.flank)


def _wind_at_arrival(
    series: list[dict],
    now_epoch: float,
    t_arrival_min: float,
) -> tuple[float, float]:
    """Nearest-hour forecast wind (speed, dir) to ``now + t_arrival_min``.

    Known v1 limitation (hour-snapping): a road whose arrival time straddles
    an hour boundary gets the adjacent hour's wind, which can flip a tier for
    no physical reason. Linear interpolation on speed is easy; direction
    needs a circular mean — deferred, not silently solved."""
    target = now_epoch + t_arrival_min * 60.0
    best = min(series, key=lambda h: abs(h["ts"] - target))
    return best["speed"], best["dir"]


def _converged_arrival(
    segment_bearing: float,
    best_dist: float,
    wind_speed: float,
    wind_dir_deg: float,
    moisture_factor: float,
    series: list[dict],
    rate_modifier: Any | None,
    now_epoch: float,
) -> tuple[float, float, float, bool]:
    """Wind-at-arrival fixed-point iteration for one road segment.

    Returns ``(t_arrival_min, wind_speed_used, wind_dir_used, converged)``:

    - Starts from the current-wind estimate (the fallback).
    - Re-looks-up the forecast wind at the current arrival time, rebuilds
      the ellipse, and recomputes arrival — so a road reached in 90 min is
      assessed with the wind that will actually be blowing in 90 min.
    - Stops early when the risk TIER (not raw minutes) is stable across two
      iterations — a tier-stable result is good enough and cheap. A tier that
      merely REPEATS earlier in the sequence (e.g. high → moderate → high) is
      a 2-cycle between forecast-hour buckets, not stability — convergence
      requires a tier that holds steady across consecutive iterations.
    - Caps at ``MAX_FORECAST_ITERS``; if it never stabilizes (arrival time
      oscillates between forecast-hour buckets), falls back to the
      current-wind-only estimate rather than returning whichever iteration
      happened to run last.
    - **Critical tier (<30 min) is intentionally unchanged**: arrival that
      fast reads the current/first forecast hour anyway, so iteration buys
      nothing — behavior is byte-identical to the pre-forecast code.

    ``rate_modifier`` is a ``(rate_m_min, t_arrival_min) -> rate_m_min`` hook
    (None = identity). It is re-evaluated INSIDE the loop with the current
    arrival time, so a future precip dampener (which scales rate by the rain
    expected in the arrival window) converges together with the wind instead
    of being bolted on as a post-hoc scalar.
    """
    rate0 = _rate_toward(segment_bearing, wind_speed, wind_dir_deg, moisture_factor)
    t_cur = best_dist / rate0 if rate0 > 0 else float("inf")

    # Critical tier: keep byte-identical behavior (no iteration).
    if t_cur < ROAD_RISK_TIERS[0][1]:
        return t_cur, wind_speed, wind_dir_deg, False

    t_prev, ws, wd = t_cur, wind_speed, wind_dir_deg
    seen_tiers = [_risk_tier(t_cur)]
    for _ in range(MAX_FORECAST_ITERS):
        ws, wd = _wind_at_arrival(series, now_epoch, t_prev)
        rate = _rate_toward(segment_bearing, ws, wd, moisture_factor)
        if rate_modifier is not None:
            rate = rate_modifier(rate, t_prev)
        t_new = best_dist / rate if rate > 0 else float("inf")
        if not math.isfinite(t_new) or abs(t_new - t_prev) < 1e-9:
            return t_new, ws, wd, True
        tier_new = _risk_tier(t_new)
        if tier_new == seen_tiers[-1]:
            return t_new, ws, wd, True  # tier stable across consecutive iterations
        if tier_new in seen_tiers[:-1]:
            # The tier repeated an EARLIER value → a 2+ cycle between
            # forecast-hour buckets, not convergence. Fall back to the
            # current-wind estimate below.
            break
        seen_tiers.append(tier_new)
        t_prev = t_new
    # Never stabilized → current-wind-only fallback.
    return t_cur, wind_speed, wind_dir_deg, False


def compute_road_risk(
    grid: BayesianFireGrid,
    road_segments: list[list[tuple[float, float]]],
    wind_speed: float,
    wind_dir_deg: float,
    contour_level: float = 0.3,
    contour: list[list[list[float]]] | None = None,
    moisture_factor: float = 1.0,
    forecast_series: list[dict] | None = None,
    rate_modifier: Any | None = None,
) -> list[dict]:
    """
    Assess risk to road segments from a spreading fire using the same
    head/back/flank ellipse as a continuous function (Part 2 of the v4
    spread model).

    For each road segment:
      1. Find the nearest point on the fire's contour.
      2. Compute bearing φ from contour point toward the road.
      3. effective_rate(φ) = closed-form radial spread rate.
      4. t_arrival = d / effective_rate(φ) — minutes.
      5. Bucket into risk tiers, weighted by grid probability.

    When ``forecast_series`` (hourly wind/precip from weather.get_forecast_series)
    is provided, step 4 becomes a wind-at-arrival fixed point: the ellipse is
    rebuilt with the forecast wind at the current arrival time and iterated
    until the risk tier is stable (see ``_converged_arrival``). Roads reached
    in <30 min (critical tier) and roads whose iteration oscillates fall back
    to the current-wind estimate unchanged. ``rate_modifier`` is an optional
    ``(rate_m_min, t_arrival_min) -> rate_m_min`` hook evaluated inside the
    loop (None = identity) — designed for a future precipitation dampener so
    wind and precip converge together.

    Parameters
    ----------
    grid : BayesianFireGrid
        The fire grid with current probability state.
    road_segments : list of list of (lat, lon) tuples
        Each road segment is a polyline of (lat, lon) pairs.
    wind_speed : float
        Current wind speed in m/s.
    wind_dir_deg : float
        Current wind direction in compass degrees.
    contour_level : float
        Probability level defining the "established fire edge" (default 0.30).
    contour : list of list of [lat, lon], optional
        Pre-computed contour from grid.export_contour(). Computed if None.
    moisture_factor : float
        Spread-rate multiplier from EFFIS fuel moisture (1.0 = neutral).
    forecast_series : list of dict, optional
        Hourly forecast (ts, speed, dir, precip_mm, ...) for this fire's
        ~55 km cell; enables the wind-at-arrival correction. None = the
        pre-forecast current-wind-only behavior.
    rate_modifier : callable, optional
        ``(rate_m_min, t_arrival_min) -> rate_m_min`` hook for future
        dampening (e.g. precip). None = identity.

    Returns
    -------
    list of dict, each with keys:
        segment, risk_tier, t_arrival_min, nearest_distance_m,
        nearest_contour_point, nearest_road_point, bearing_from_wind_deg,
        effective_spread_rate_m_min, probability_at_contour,
        head_rate_m_min, back_rate_m_min, flank_rate_m_min, wind_source
    """
    if contour is None:
        contour = grid.export_contour(level=contour_level)

    # Flatten contour to a list of (lat, lon) points
    contour_points: list[tuple[float, float]] = []
    for seg in contour:
        for pt in seg:
            contour_points.append((pt[0], pt[1]))

    if not contour_points:
        return [
            {
                "segment": seg,
                "risk_tier": "low",
                "t_arrival_min": None,
                "message": "No fire contour at this level",
            }
            for seg in road_segments
        ]

    results = []
    for segment in road_segments:
        best_dist = float("inf")
        best_contour_pt = None
        nearest_road_pt = None

        # Brute-force nearest point search (n is small — roads near fire)
        for rlat, rlon in segment:
            for clat, clon in contour_points:
                d = _risk_haversine(rlat, rlon, clat, clon)
                if d < best_dist:
                    best_dist = d
                    best_contour_pt = (clat, clon)
                    nearest_road_pt = (rlat, rlon)

        if best_contour_pt is None:
            results.append({
                "segment": segment,
                "risk_tier": "low",
                "t_arrival_min": None,
                "nearest_distance_m": None,
                "message": "No contour point found",
            })
            continue

        # Bearing from contour point toward road point
        bearing = _initial_bearing(
            best_contour_pt[0], best_contour_pt[1],
            nearest_road_pt[0], nearest_road_pt[1],
        )

        # Wind-at-arrival correction (forecast present): rebuild the ellipse
        # with the wind that will actually be blowing when the fire reaches
        # this road. Falls back to current wind on oscillation / no forecast.
        if forecast_series:
            t_arrival, ws_used, wd_used, converged = _converged_arrival(
                bearing, best_dist, wind_speed, wind_dir_deg,
                moisture_factor, forecast_series, rate_modifier, time.time(),
            )
        else:
            ws_used, wd_used, converged = wind_speed, wind_dir_deg, False

        # Ellipse under the WIND ACTUALLY USED (forecast-corrected or current)
        # so the reported rates and t_arrival agree with the same kernel.
        used_kernel = SpreadKernel(
            wind_speed=ws_used,
            wind_dir_deg=wd_used,
            moisture_factor=moisture_factor,
        )
        phi_deg = (bearing - wd_used + 360) % 360
        rate = effective_spread_rate(phi_deg, used_kernel.head, used_kernel.back, used_kernel.flank)
        if not forecast_series:
            # Pre-forecast behavior: current wind, no iteration.
            t_arrival = best_dist / rate if rate > 0 else float("inf")

        # Risk tier
        risk_tier = _risk_tier(t_arrival)

        # Grid probability at nearest contour point
        ci, cj = grid.latlon_to_cell(best_contour_pt[0], best_contour_pt[1])
        if 0 <= ci < grid.nx and 0 <= cj < grid.ny:
            prob = float(grid.probabilities[ci, cj])
        else:
            prob = 0.0

        results.append({
            "segment": segment,
            "risk_tier": risk_tier,
            "t_arrival_min": round(t_arrival, 1) if t_arrival != float("inf") else None,
            "nearest_distance_m": round(best_dist, 1),
            "nearest_contour_point": [round(best_contour_pt[0], 6), round(best_contour_pt[1], 6)],
            "nearest_road_point": [round(nearest_road_pt[0], 6), round(nearest_road_pt[1], 6)],
            "bearing_from_wind_deg": round(phi_deg, 1),
            "effective_spread_rate_m_min": round(rate, 2),
            "head_rate_m_min": round(used_kernel.head, 2),
            "back_rate_m_min": round(used_kernel.back, 2),
            "flank_rate_m_min": round(used_kernel.flank, 2),
            "probability_at_contour": round(prob, 4),
            "wind_source": "forecast" if converged else "current",
        })

    return results


def seed_from_reports(
    grid: BayesianFireGrid,
    reports: list[dict],
    clusters: list[dict],
    wind_dir_deg: float | None = None,
) -> None:
    """
    Feed existing confirmed reports into the grid as initial evidence.
    Also uses triangulation results (if available) to set spatial uncertainty.

    Parameters
    ----------
    wind_dir_deg : float | None
        Wind direction in compass degrees for smoke-drift correction.
        When provided, smoke-only evidence is shifted upwind by 500 m
        (previously this was dead code since nothing passed it).
    """
    cluster_lookup: dict[str, dict] = {}
    for c in clusters:
        for rid in c.get("report_ids", []):
            cluster_lookup[rid] = c

    for r in reports:
        if r.get("status") != "confirmed":
            continue

        c = cluster_lookup.get(r["id"])
        ellipse = None
        if c and c.get("triangulation"):
            t = c["triangulation"]
            if t.get("status") == "ok":
                ellipse = {
                    "semi_major": t.get("ellipse_semi_major", 0),
                    "semi_minor": t.get("ellipse_semi_minor", 0),
                }

        evidence = Evidence.from_report(r, wind_dir_deg=wind_dir_deg, ellipse=ellipse)
        grid.update(evidence)