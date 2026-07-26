"""
nasa_firms.py — NASA FIRMS Active Fire Data Fetcher
====================================================

Fetches near-real-time satellite fire hotspot data from the NASA FIRMS API
(VIIRS 375m and MODIS 1km) and returns structured records suitable for
injection into the Bayesian fire grid as Evidence.

API docs: https://firms.modaps.eosdis.nasa.gov/api/
"""

from __future__ import annotations

import csv
import io
import logging
import math
import os
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api"

# Valid source identifiers
VIIRS_SNPP_NRT = "VIIRS_SNPP_NRT"  # Suomi-NPP, 375m resolution
VIIRS_NOAA20_NRT = "VIIRS_NOAA20_NRT"  # NOAA-20, 375m
VIIRS_NOAA21_NRT = "VIIRS_NOAA21_NRT"  # NOAA-21, 375m
MODIS_NRT = "MODIS_NRT"  # Aqua/Terra, 1km resolution

# Default source: VIIRS Suomi-NPP (highest resolution, most widely used)
DEFAULT_SOURCE = VIIRS_SNPP_NRT

# Max days the FIRMS API allows per request
MAX_DAY_RANGE = 5

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class FIRMSHotspot:
    """A single active fire / thermal anomaly detection from FIRMS."""

    latitude: float
    longitude: float
    brightness: float  # brightness temperature (K)
    scan: float
    track: float
    acq_date: str  # "YYYY-MM-DD"
    acq_time: str  # "HHMM" UTC
    satellite: str  # e.g. "N" (Suomi-NPP), "20" (NOAA-20)
    instrument: str  # "VIIRS" or "MODIS"
    confidence: str  # "low", "nominal", "high", or percentage string
    version: str
    frp: float  # Fire Radiative Power (MW)
    daynight: str  # "D" or "N"

    @property
    def is_high_confidence(self) -> bool:
        """Return True if this detection is high-confidence."""
        # FIRMS confidence values: "low", "nominal", "high", or "0-100%"
        try:
            pct = int(self.confidence)
            return pct >= 80
        except (ValueError, TypeError):
            return self.confidence.lower() == "high"

    @property
    def is_nominal_or_higher(self) -> bool:
        """Return True if this detection is nominal or high confidence."""
        try:
            pct = int(self.confidence)
            return pct >= 50
        except (ValueError, TypeError):
            return self.confidence.lower() in ("nominal", "high")


# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------


def _get_api_key() -> str | None:
    """Return the NASA FIRMS API key from the environment."""
    return os.environ.get("NASA_FIRMS_API_KEY") or os.environ.get("FIRMS_API_KEY")


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------


def fetch_fire_data(
    api_key: str,
    bbox: tuple[float, float, float, float] | str = "world",
    source: str = DEFAULT_SOURCE,
    day_range: int = 1,
) -> list[FIRMSHotspot]:
    """Fetch active fire data from NASA FIRMS.

    Parameters
    ----------
    api_key : str
        NASA FIRMS API key (free registration at
        https://firms.modaps.eosdis.nasa.gov/api/map_key/).
    bbox : tuple[float, float, float, float] | str
        Bounding box as (west, south, east, north) or the string "world".
    source : str
        Satellite source identifier (e.g. VIIRS_SNPP_NRT, MODIS_NRT).
    day_range : int
        Number of days to look back (1–5).

    Returns
    -------
    list[FIRMSHotspot]
        Parsed hotspot detections.

    Raises
    ------
    ValueError
        If the API key is missing or the response is unparseable.
    ConnectionError
        If the API call fails due to network/HTTP errors.
    """
    if not api_key:
        raise ValueError(
            "NASA FIRMS API key is required. Set NASA_FIRMS_API_KEY env var."
        )

    day_range = max(1, min(day_range, MAX_DAY_RANGE))

    # Build the area coordinates string
    if isinstance(bbox, str) and bbox.lower() == "world":
        coords_str = "world"
    else:
        w, s, e, n = bbox
        coords_str = f"{w},{s},{e},{n}"

    url = f"{FIRMS_BASE_URL}/area/csv/{api_key}/{source}/{coords_str}/{day_range}"

    logger.info("Fetching FIRMS data: %s", url.replace(api_key, "***API_KEY***"))

    try:
        req = Request(url, headers={"User-Agent": "WildFrame/1.0"})
        with urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
    except HTTPError as exc:
        if exc.code == 403:
            raise ConnectionError(
                "FIRMS API returned 403: invalid API key or rate limit exceeded."
            ) from exc
        if exc.code == 404:
            raise ConnectionError(
                "FIRMS API returned 404: invalid endpoint. "
                "Check source and parameters."
            ) from exc
        raise ConnectionError(f"FIRMS API HTTP {exc.code}: {exc.reason}") from exc
    except URLError as exc:
        raise ConnectionError(f"FIRMS API network error: {exc.reason}") from exc

    if not raw.strip():
        return []  # No data returned

    return _parse_firms_csv(raw)


def _parse_firms_csv(csv_text: str) -> list[FIRMSHotspot]:
    """Parse a FIRMS CSV response into a list of FIRMSHotspot records."""
    reader = csv.DictReader(io.StringIO(csv_text))
    if not reader.fieldnames:
        return []

    hotspots: list[FIRMSHotspot] = []
    for row in reader:
        try:
            hotspot = FIRMSHotspot(
                latitude=float(row.get("latitude", 0) or 0),
                longitude=float(row.get("longitude", 0) or 0),
                brightness=float(
                    row.get("bright_ti4") or row.get("bright_t31", 0) or 0
                ),
                scan=float(row.get("scan", 0) or 0),
                track=float(row.get("track", 0) or 0),
                acq_date=str(row.get("acq_date", "")),
                acq_time=str(row.get("acq_time", "")),
                satellite=str(row.get("satellite", "")),
                instrument=str(row.get("instrument", "")),
                confidence=str(row.get("confidence", "low")),
                version=str(row.get("version", "")),
                frp=float(row.get("frp", 0) or 0),
                daynight=str(row.get("daynight", "D")),
            )
            hotspots.append(hotspot)
        except (ValueError, TypeError) as exc:
            logger.warning("Skipping unparseable FIRMS row: %s", exc)
            continue

    return hotspots


# ---------------------------------------------------------------------------
# Convenience: fetch hotspots near a given center point
# ---------------------------------------------------------------------------


def fetch_hotspots_near(
    api_key: str,
    center_lat: float,
    center_lon: float,
    radius_km: float = 25.0,
    source: str = DEFAULT_SOURCE,
    day_range: int = 1,
    min_confidence: str = "nominal",
) -> list[FIRMSHotspot]:
    """Fetch FIRMS hotspots within a radius of a center point.

    Parameters
    ----------
    api_key : str
        NASA FIRMS API key.
    center_lat, center_lon : float
        Center point in decimal degrees.
    radius_km : float
        Search radius in kilometres (default 25 km).
    source : str
        Satellite source.
    day_range : int
        Days to look back (1–5).
    min_confidence : str
        Minimum confidence filter: "low", "nominal", or "high".

    Returns
    -------
    list[FIRMSHotspot]
        Hotspots within the bounding box.
    """
    # Convert radius to degrees (approximate)
    dlat = radius_km / 111.0
    cos_lat = abs(math.cos(math.radians(center_lat))) or 1.0
    dlon = radius_km / (111.0 * cos_lat)

    bbox = (
        center_lon - dlon,  # west
        center_lat - dlat,  # south
        center_lon + dlon,  # east
        center_lat + dlat,  # north
    )

    all_hotspots = fetch_fire_data(api_key, bbox, source, day_range)

    # Filter by confidence
    if min_confidence == "high":
        return [h for h in all_hotspots if h.is_high_confidence]
    elif min_confidence == "nominal":
        return [h for h in all_hotspots if h.is_nominal_or_higher]
    else:
        return all_hotspots


# ---------------------------------------------------------------------------
# Global fetch (all fires worldwide in the last 1-5 days)
# ---------------------------------------------------------------------------


def fetch_global_fires(
    api_key: str,
    source: str = DEFAULT_SOURCE,
    day_range: int = 1,
    min_confidence: str = "nominal",
) -> list[FIRMSHotspot]:
    """Fetch active fire hotspots worldwide.

    Parameters
    ----------
    api_key : str
        NASA FIRMS API key.
    source : str
        Satellite source.
    day_range : int
        Days to look back (1–5).
    min_confidence : str
        Minimum confidence filter.

    Returns
    -------
    list[FIRMSHotspot]
        All global hotspot detections matching the criteria.
    """
    all_hotspots = fetch_fire_data(api_key, "world", source, day_range)

    if min_confidence == "high":
        return [h for h in all_hotspots if h.is_high_confidence]
    elif min_confidence == "nominal":
        return [h for h in all_hotspots if h.is_nominal_or_higher]
    else:
        return all_hotspots
