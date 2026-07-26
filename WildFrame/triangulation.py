#!/usr/bin/env python3
"""
triangulation.py — Bearings-Only Fire Origin Triangulation
===========================================================

Given a set of reports, each with a (lat, lon) location and a compass
bearing (degrees clockwise from north), this module estimates the most
likely fire origin point F = (x_fire, y_fire) in local planar coordinates,
along with a confidence ellipse.

Algorithm
---------
1. Convert lat/lon → local planar coordinates (equirectangular projection).
2. Each report becomes a ray P_i + t·d_i where d_i is the unit direction
   vector derived from the compass bearing.
3. Weighted least-squares intersection: minimise the sum of squared
   perpendicular distances from F to all rays.

    A = Σ w_i · n_i n_i^T      (2×2 matrix)
    b = Σ w_i · (n_i · P_i) n_i

    F = A⁻¹ b                  (solved via Cramer's rule)

4. Uncertainty ellipse from Cov(F) = σ² · A⁻¹.

Usage (as module)
-----------------
    from triangulation import triangulate

    result = triangulate([
        {"lat": 38.172, "lon": 23.717, "device_heading": 45.0},
        {"lat": 38.176, "lon": 23.722, "device_heading": 48.0},
    ])

    if result["status"] == "ok":
        print(f"Fire at {result['fire_lat']}, {result['fire_lon']}")
        print(f"Confidence: {result['confidence']}")
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
R_EARTH = 6_371_000.0  # Earth radius in metres
DEFAULT_BEARING_ERROR_DEG = 15.0  # default σ_θ when not otherwise estimated

# Enhanced weighting constants
DISTANCE_DECAY_SCALE_M = 500.0   # characteristic distance (metres) for Gaussian decay
TIME_DECAY_HALF_LIFE_S = 7200.0  # half-life (seconds) for Lorentzian time decay (~2 hours)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0


def equirectangular_projection(
    lat: float, lon: float, lat0: float, lon0: float
) -> tuple[float, float]:
    """
    Convert (lat, lon) in decimal degrees to local planar (x, y) in metres
    using an equirectangular projection centred on (lat0, lon0).

    x = (lon - lon0) · cos(lat0) · R
    y = (lat - lat0) · R
    """
    rlat = _deg2rad(lat)
    rlon = _deg2rad(lon)
    rlat0 = _deg2rad(lat0)
    rlon0 = _deg2rad(lon0)

    x = (rlon - rlon0) * math.cos(rlat0) * R_EARTH
    y = (rlat - rlat0) * R_EARTH
    return x, y


def inverse_equirectangular(
    x: float, y: float, lat0: float, lon0: float
) -> tuple[float, float]:
    """
    Convert local planar (x, y) in metres back to (lat, lon) in decimal
    degrees, using the same reference point.
    """
    rlat0 = _deg2rad(lat0)
    rlon0 = _deg2rad(lon0)

    lat = math.degrees(y / R_EARTH + rlat0)
    lon = math.degrees(x / (R_EARTH * math.cos(rlat0)) + rlon0)
    return lat, lon


# ---------------------------------------------------------------------------
# Ray and normal computation
# ---------------------------------------------------------------------------

def bearing_to_direction(bearing_deg: float) -> tuple[float, float]:
    """
    Convert compass bearing (degrees clockwise from north) to a unit
    direction vector (dx, dy) in local coordinates.

    Compass convention: 0° = north (+y), 90° = east (+x).
    """
    θ = _deg2rad(bearing_deg)
    return math.sin(θ), math.cos(θ)


def bearing_to_normal(bearing_deg: float) -> tuple[float, float]:
    """
    Unit normal vector to the ray (d_i rotated 90° clockwise).

    n_i = (cos θ, -sin θ)
    """
    θ = _deg2rad(bearing_deg)
    return math.cos(θ), -math.sin(θ)


# ---------------------------------------------------------------------------
# Weight functions
# ---------------------------------------------------------------------------

def default_weighting(reports_planar: list[dict]) -> list[float]:
    """
    Enhanced weighting scheme combining three factors:

    1. **Bearing accuracy** — w_bear = 1 / σ²
       Reports with wider bearing error (σ_θ) get lower weight.

    2. **Distance decay** — w_dist = exp(-(d / D_scale)²)
       Reports far from the cluster centroid contribute less angular
       resolution over the search area.

    3. **Time decay** — w_time = 1 / (1 + (age / T_half)²)
       Older reports (age > T_half) are downweighted with a Lorentzian
       tail, so reports beyond ~2 hours become increasingly irrelevant.

    Final weight: w_i = w_bear * w_dist * w_time

    Planar dicts must contain:
        - "bearing_error_deg"  (float)
        - "dist_from_centroid"  (float, metres)  — 0 if not set
        - "age_s"               (float, seconds)  — 0 if not set
    """
    weights: list[float] = []
    for r in reports_planar:
        # 1) Bearing accuracy
        sigma_deg = r.get("bearing_error_deg", DEFAULT_BEARING_ERROR_DEG)
        w_bear = 1.0 / (sigma_deg * sigma_deg)

        # 2) Distance decay (Gaussian)
        d = r.get("dist_from_centroid", 0.0)
        w_dist = math.exp(-(d / DISTANCE_DECAY_SCALE_M) ** 2)

        # 3) Time decay (Lorentzian — smoother than exponential cutoff)
        age = r.get("age_s", 0.0)
        w_time = 1.0 / (1.0 + (age / TIME_DECAY_HALF_LIFE_S) ** 2) if age > 0 else 1.0

        # Combined weight
        w = w_bear * w_dist * w_time
        weights.append(w)
    return weights


# ---------------------------------------------------------------------------
# Triangulation core
# ---------------------------------------------------------------------------

def _parse_age(captured_at: Any, now: datetime) -> float:
    """
    Compute the age of a report in seconds from its ``captured_at``
    timestamp.  ``captured_at`` can be:
        - An ISO-8601 string (e.g. ``"2026-07-10T17:50:48.524Z"``)
        - A ``datetime`` instance
        - ``None`` or empty string → age = 0 (no downweighting)
    """
    if captured_at is None:
        return 0.0
    if isinstance(captured_at, str):
        captured_at = captured_at.strip()
        if not captured_at:
            return 0.0
        # Normalise Z suffix (Python < 3.11 compat)
        if captured_at.endswith("Z"):
            captured_at = captured_at[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(captured_at)
        except (ValueError, TypeError):
            return 0.0
    elif isinstance(captured_at, datetime):
        dt = captured_at
    else:
        return 0.0

    # Ensure both are timezone-aware for subtraction
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now_utc = now.replace(tzinfo=timezone.utc)
    else:
        now_utc = now

    age = (now_utc - dt).total_seconds()
    return max(age, 0.0)


def triangulate(
    reports: list[dict],
    weight_fn: Callable | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """
    Triangulate the fire origin from one or more bearing reports.

    Parameters
    ----------
    reports : list[dict]
        Each dict must contain:
            - "lat" (float)
            - "lon" (float)
            - "device_heading" (float | None) — compass bearing in degrees,
              clockwise from north. Reports with None are skipped.
        May contain:
            - "bearing_error_deg" (float) — per-report σ_θ estimate.
            - "captured_at" (str | datetime | None) — timestamp for
              time-based downweighting. ISO-8601 string or datetime.
    weight_fn : Callable | None
        Function that takes the list of planar reports and returns a list of
        weights. If None, uses `default_weighting`.
    now : datetime | None
        Reference time for age computation.  Defaults to ``utcnow()``.

    Returns
    -------
    dict with keys:
        "status"          : "ok" | "insufficient_reports" | "nearly_parallel"
        "fire_lat"        : float — estimated fire latitude  (if ok)
        "fire_lon"        : float — estimated fire longitude (if ok)
        "fire_x"          : float — local x (metres)         (if ok)
        "fire_y"          : float — local y (metres)         (if ok)
        "ellipse_semi_major" : float — uncertainty ellipse (if ok)
        "ellipse_semi_minor" : float — uncertainty ellipse (if ok)
        "ellipse_angle_deg"  : float — rotation of major axis (if ok)
        "condition_number"   : float — A matrix condition (if ok)
        "confidence"      : "high" | "medium" | "low"
        "num_reports"     : int — number of reports with bearing
        "det"             : float — determinant of A
        "message"         : str — human-readable status
    """
    # Filter to reports that have a bearing
    bearing_reports = [r for r in reports if r.get("device_heading") is not None]

    if len(bearing_reports) < 2:
        return {
            "status": "insufficient_reports",
            "num_reports": len(bearing_reports),
            "confidence": "low",
            "message": (
                f"Need ≥ 2 reports with bearings to triangulate "
                f"(got {len(bearing_reports)})"
            ),
        }

    # --- Step 1: Compute centroid for local projection reference ---
    lat0 = sum(r["lat"] for r in bearing_reports) / len(bearing_reports)
    lon0 = sum(r["lon"] for r in bearing_reports) / len(bearing_reports)

    # --- Step 2: Project to local planar coordinates ---
    if now is None:
        now = datetime.now(timezone.utc)

    planar: list[dict] = []
    for r in bearing_reports:
        x, y = equirectangular_projection(r["lat"], r["lon"], lat0, lon0)
        dx, dy = bearing_to_direction(r["device_heading"])
        nx, ny = bearing_to_normal(r["device_heading"])

        # Distance from the reporter centroid (projection origin)
        dist_from_centroid = math.hypot(x, y)

        # Age in seconds
        age_s = _parse_age(r.get("captured_at"), now)

        planar.append({
            "x": x,
            "y": y,
            "dx": dx,
            "dy": dy,
            "nx": nx,
            "ny": ny,
            "bearing_error_deg": r.get("bearing_error_deg", DEFAULT_BEARING_ERROR_DEG),
            "dist_from_centroid": dist_from_centroid,
            "age_s": age_s,
        })

    # --- Step 3: Compute weights ---
    if weight_fn is None:
        weights = default_weighting(planar)
    else:
        weights = weight_fn(planar)

    # --- Step 4: Build A matrix and b vector (Cramer's rule form) ---
    Axx = 0.0
    Axy = 0.0
    Ayy = 0.0
    bx = 0.0
    by = 0.0

    for p, w in zip(planar, weights):
        nx, ny = p["nx"], p["ny"]
        # n_i · P_i
        n_dot_P = nx * p["x"] + ny * p["y"]

        # A += w · n n^T
        Axx += w * nx * nx
        Axy += w * nx * ny
        Ayy += w * ny * ny

        # b += w · (n · P) · n
        bx += w * n_dot_P * nx
        by += w * n_dot_P * ny

    # --- Step 5: Solve 2×2 system via Cramer's rule ---
    det = Axx * Ayy - Axy * Axy

    if abs(det) < 1e-12:
        return {
            "status": "nearly_parallel",
            "num_reports": len(bearing_reports),
            "det": det,
            "confidence": "low",
            "message": (
                "A matrix is near-singular (det ≈ 0) — all bearing rays "
                "are nearly parallel. Cannot triangulate confidently."
            ),
        }

    fx = (bx * Ayy - by * Axy) / det
    fy = (Axx * by - Axy * bx) / det

    # --- Step 6: Convert back to lat/lon ---
    fire_lat, fire_lon = inverse_equirectangular(fx, fy, lat0, lon0)

    # --- Step 7: Uncertainty ellipse ---
    # Cov(F) ≈ σ² · A⁻¹
    # Use the RMS residual as σ² estimate, or fall back to default error
    # Inverse of A:
    invAxx = Ayy / det
    invAxy = -Axy / det
    invAyy = Axx / det

    # Compute residuals for σ² estimate
    residuals: list[float] = []
    for p in planar:
        nx, ny = p["nx"], p["ny"]
        dist = abs(nx * (fx - p["x"]) + ny * (fy - p["y"]))
        residuals.append(dist)
    rms_residual = math.sqrt(sum(r * r for r in residuals) / len(residuals)) if residuals else 1.0

    # Scale factor: use RMS residual or a minimum of 1 m
    sigma2 = max(rms_residual, 1.0) ** 2

    cov_xx = sigma2 * invAxx
    cov_xy = sigma2 * invAxy
    cov_yy = sigma2 * invAyy

    # Eigenvalues of the 2×2 covariance matrix
    trace = cov_xx + cov_yy
    disc = math.sqrt((cov_xx - cov_yy) ** 2 + 4 * cov_xy * cov_xy)
    eig1 = (trace + disc) / 2.0  # larger eigenvalue → semi-major axis
    eig2 = (trace - disc) / 2.0  # smaller eigenvalue → semi-minor axis

    semi_major = math.sqrt(eig1) if eig1 > 0 else 0.0
    semi_minor = math.sqrt(eig2) if eig2 > 0 else 0.0

    # Rotation angle of the major axis (in degrees)
    # tan(2θ) = 2·cov_xy / (cov_xx - cov_yy)
    if abs(cov_xx - cov_yy) > 1e-12:
        theta = 0.5 * math.atan2(2 * cov_xy, cov_xx - cov_yy)
    else:
        theta = math.pi / 4.0 if cov_xy > 0 else 0.0
    ellipse_angle_deg = math.degrees(theta)

    # Condition number of A (ratio of largest to smallest eigenvalue of A^-1)
    condition_number = eig1 / eig2 if eig2 > 1e-12 else float("inf")

    # --- Step 8: Confidence rating ---
    if condition_number < 5.0 and semi_major < 500.0:
        confidence = "high"
    elif condition_number < 20.0 and semi_major < 2000.0:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "status": "ok",
        "fire_lat": fire_lat,
        "fire_lon": fire_lon,
        "fire_x": fx,
        "fire_y": fy,
        "ellipse_semi_major": semi_major,
        "ellipse_semi_minor": semi_minor,
        "ellipse_angle_deg": ellipse_angle_deg,
        "condition_number": condition_number,
        "confidence": confidence,
        "num_reports": len(bearing_reports),
        "det": det,
        "message": (
            f"Triangulated from {len(bearing_reports)} bearing reports. "
            f"Confidence: {confidence}. "
            f"Uncertainty: {semi_major:.0f} × {semi_minor:.0f} m"
        ),
    }


# ---------------------------------------------------------------------------
# Convenience: batch triangulate clusters from the server
# ---------------------------------------------------------------------------

def triangulate_cluster(
    reports_in_cluster: list[dict],
    now: datetime | None = None,
) -> dict[str, Any]:
    """
    Wrapper for triangulating a single cluster's reports.

    Forwards the ``captured_at`` timestamp for time-based weighting.

    Returns the same dict as `triangulate()`.
    """
    input_reports = []
    for r in reports_in_cluster:
        input_reports.append({
            "lat": r["lat"],
            "lon": r["lon"],
            "device_heading": r.get("device_heading"),
            "bearing_error_deg": r.get("bearing_error_deg", DEFAULT_BEARING_ERROR_DEG),
            "captured_at": r.get("captured_at"),
        })
    return triangulate(input_reports, now=now)
