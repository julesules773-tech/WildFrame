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
    """

    def __init__(
        self,
        wind_speed: float = 0.0,
        wind_dir_deg: float = 0.0,
        slope_pct: float = 0.0,
        slope_aspect_deg: float = 0.0,
        base_spread_rate: float = DEFAULT_BASE_SPREAD_RATE,
    ):
        self.wind_speed = wind_speed
        self.wind_dir = wind_dir_deg
        self.slope_pct = slope_pct
        self.slope_aspect = slope_aspect_deg
        self.base_rate = base_spread_rate

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

    return segments


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
    ) -> None:
        """
        Advance the probability grid by time dt (seconds).

        1. For each cell with p > burn_threshold, spread probability mass
           to neighbouring cells using an elliptical spread kernel.
        2. Apply exponential decay to all cells (uncorroborated probability
           fades with half-life DECAY_HALF_LIFE_S).
        """
        now = datetime.now(timezone.utc).timestamp()
        dt_minutes = dt / 60.0

        # Compute the spread kernel
        kernel = SpreadKernel(
            wind_speed=wind_speed,
            wind_dir_deg=wind_dir_deg,
            slope_pct=slope_pct,
            slope_aspect_deg=slope_aspect_deg,
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

        # --- Apply temporal decay (uncorroborated cells) ---
        # Decay factor per cell: p *= exp(-lambda * dt)
        # In log-odds: this is more complex...  Let's work in probability space.
        self._compute_probs()

        decay_factor = math.exp(-DECAY_LAMBDA * dt)
        # Decay probability toward 0 for all cells
        self.probabilities *= decay_factor
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

    def update(self, evidence: Evidence) -> None:
        """
        Fuse a piece of evidence into the grid using Bayes' rule in log-odds.

        For pin-point evidence (spatial_radius_m ≈ 0), the update is applied
        to exactly one cell.  For spatially-uncertain evidence, the log-LR is
        spread as a 2D Gaussian across neighbouring cells, weighted so that
        the total information content equals the original LR.
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

        now = datetime.now(timezone.utc).timestamp()
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

        segments_xy = marching_squares_contour(
            self.probabilities, level, self.grid_x, self.grid_y,
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


def compute_road_risk(
    grid: BayesianFireGrid,
    road_segments: list[list[tuple[float, float]]],
    wind_speed: float,
    wind_dir_deg: float,
    contour_level: float = 0.3,
    contour: list[list[list[float]]] | None = None,
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

    Returns
    -------
    list of dict, each with keys:
        segment, risk_tier, t_arrival_min, nearest_distance_m,
        nearest_contour_point, nearest_road_point, bearing_from_wind_deg,
        effective_spread_rate_m_min, probability_at_contour,
        head_rate_m_min, back_rate_m_min, flank_rate_m_min
    """
    kernel = SpreadKernel(wind_speed=wind_speed, wind_dir_deg=wind_dir_deg)

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

        # Bearing relative to wind direction
        phi_deg = (bearing - wind_dir_deg + 360) % 360

        # Effective spread rate in this direction (m/min)
        rate = effective_spread_rate(phi_deg, kernel.head, kernel.back, kernel.flank)

        # Time to arrival (minutes)
        t_arrival = best_dist / rate if rate > 0 else float("inf")

        # Risk tier
        risk_tier = "low"
        for tier_name, tier_minutes in ROAD_RISK_TIERS:
            if t_arrival < tier_minutes:
                risk_tier = tier_name
                break

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
            "head_rate_m_min": round(kernel.head, 2),
            "back_rate_m_min": round(kernel.back, 2),
            "flank_rate_m_min": round(kernel.flank, 2),
            "probability_at_contour": round(prob, 4),
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