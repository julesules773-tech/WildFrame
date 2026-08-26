#!/usr/bin/env python3
"""
WildFrame — Wildfire Detection Prototype Backend
=================================================
Flask server that:
  - Accepts photo uploads with GPS coordinates
  - Stores reports as JSON
  - Runs single-pass clustering to group nearby reports into candidate fires
  - Reports clusters via REST API for the Leaflet frontend
"""

import hashlib
import hmac
import json
import logging
import math
import os
import random
import threading
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Load .env (WILDFRAME_DATABASE_URL, NASA_FIRMS_API_KEY, WILDFRAME_ADMIN_SECRET,
# ROBOFLOW_API_KEY) before any project module reads environment variables.
# dotenv refuses to override existing shell vars — including EMPTY ones (a
# leftover `export NASA_FIRMS_API_KEY=` shadows the real key in .env). Fill
# in values for any key that is missing OR empty so .env actually applies.
from dotenv import dotenv_values

_env_path = Path(__file__).parent / ".env"
for _key, _value in dotenv_values(_env_path).items():
    if _value and not os.environ.get(_key):
        os.environ[_key] = _value

from triangulation import triangulate, triangulate_cluster
from bayesian_filter import (
    BayesianFireGrid, Evidence, auto_grid_size, seed_from_reports,
    compute_road_risk,
    DEFAULT_CELL_SIZE_M, CITIZEN_CELL_SIZE_M, DEFAULT_BASE_SPREAD_RATE, WIND_HEAD_FACTOR,
)

from fire_vision import scan_photo
import nasa_firms
import corine
import worldcover
import land_mask
import db
import photo_storage
import weather
import effis_fwi

# The Procrastinate job app (jobs.py) is used to enqueue on-demand jobs
# (e.g. a manual FIRMS fetch) from the web process. The worker opens its
# own copy (async) at startup and lazily imports this module from inside
# jobs — so a module-level sync open() would crash there with
# NotImplementedError (sync open on an already-open async connector). The
# server's copy is therefore opened lazily, exactly once, right before the
# first defer (see _job_defer).
from jobs import app as _job_app

_job_app_lock = threading.Lock()
_job_app_opened = False


def _job_defer(task_name: str, **kwargs):
    """Open the job app exactly once, then defer ``task_name``.

    Web process: opens the sync connector on first use so
    ``configure_task(...).defer(...)`` works. Worker process: never calls
    this (jobs execute the task directly), so importing this module there
    no longer opens the app — fixing the crash where ``import server``
    inside a job hit ``NotImplementedError`` from the sync ``open()`` on
    top of the worker's already-open async app.
    """
    global _job_app_opened
    if not _job_app_opened:
        with _job_app_lock:
            if not _job_app_opened:
                _job_app.open()
                _job_app_opened = True
    return _job_app.configure_task(name=task_name).defer(**kwargs)

from flask import Flask, jsonify, request, send_from_directory, redirect
from werkzeug.utils import secure_filename
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
UPLOAD_DIR = Path("uploads")
DATA_DIR = Path("data")
STATIC_DIR = Path("static")

CLUSTER_RADIUS_M = 500.0          # metres — group reports within this distance
CLUSTER_TIME_WINDOW_MINUTES = 120 # 2 hours
ACTIVE_REPORT_HOURS = 48          # keep reports visible on map for 48h
# Top-N active fires (by peak probability) shown in the admin dashboard's
# live-fire overview — spot-checking data, not an archive.
ADMIN_GRIDS_LIMIT = 500

# Grid matching radius lives in db.py (shared with the persistence layer).
GRID_MATCH_RADIUS_M = db.GRID_MATCH_RADIUS_M
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp", "heic", "heif"}


# ---------------------------------------------------------------------------
# Application-level rate limiter (in-memory, sliding window)
# ---------------------------------------------------------------------------
# Complements nginx rate limiting: catches abuse that slips through
# (e.g. distributed low-rate floods, or direct-origin bypass).
# No external dependencies — uses a dict of (ip → [timestamps]).

class RateLimiter:
    """Per-IP sliding-window rate limiter.

    Usage:
        limiter = RateLimiter()
        if not limiter.allow(request.remote_addr, "upload", max_hits=5, window_s=60):
            return jsonify({"error": "rate limit exceeded"}), 429
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._windows: dict[str, dict[str, list[float]]] = {}  # ip → {key → [ts]}

    def allow(self, ip: str, key: str, max_hits: int, window_s: float) -> bool:
        """Return True if the request is allowed, False if rate-limited."""
        now = time.time()
        cutoff = now - window_s
        bucket_key = f"{ip}:{key}"

        with self._lock:
            buckets = self._windows.setdefault(ip, {})
            timestamps = buckets.get(key, [])

            # Prune old entries outside the window
            timestamps = [t for t in timestamps if t > cutoff]

            if len(timestamps) >= max_hits:
                buckets[key] = timestamps
                return False

            timestamps.append(now)
            buckets[key] = timestamps
            return True

    def _cleanup(self) -> None:
        """Periodic cleanup of stale entries (called lazily)."""
        now = time.time()
        with self._lock:
            stale_ips = []
            for ip, buckets in self._windows.items():
                stale_keys = [k for k, ts in buckets.items()
                              if not ts or ts[-1] < now - 3600]
                for k in stale_keys:
                    del buckets[k]
                if not buckets:
                    stale_ips.append(ip)
            for ip in stale_ips:
                del self._windows[ip]


_limiter = RateLimiter()
# Cleanup every ~500 requests (rough — no need for a timer thread).
# Use itertools.count (atomic increment) instead of bare int += 1 to
# avoid the GIL read-modify-write race under concurrent requests.
import itertools as _itertools
_request_counter = _itertools.count()


def _check_rate_limit(ip: str, key: str, max_hits: int, window_s: float) -> Optional[str]:
    """Check rate limit; return error message or None if allowed.

    Also triggers lazy cleanup every ~500 requests."""
    n = next(_request_counter)
    if n % 500 == 0:
        _limiter._cleanup()
        _abuse_tracker._cleanup()

    if not _limiter.allow(ip, key, max_hits, window_s):
        return f"rate limit: max {max_hits} {key} requests per {window_s:.0f}s"
    return None


# Rate-limit configs: (key, max_hits, window_seconds)
UPLOAD_RATE_LIMIT = ("upload", 5, 60)        # 5 uploads per minute
SAT_CONFIRM_RATE_LIMIT = ("sat_check", 10, 60)  # 10 satellite checks per minute


# ---------------------------------------------------------------------------
# IP abuse tracker — blocks IPs that repeatedly upload non-fire photos
# ---------------------------------------------------------------------------
# After the AI scan, reports with verdict "nothing" (no fire/smoke detected)
# are tracked per IP.  If an IP exceeds ABUSE_NOTHING_THRESHOLD rejections
# within ABUSE_WINDOW_S, the IP is blocked for ABUSE_BLOCK_DURATION_S.
# The block is in-memory and resets on server restart — suitable for
# stopping automated abuse; persistent bans should use nginx or a WAF.
ABUSE_NOTHING_THRESHOLD = 5     # nothing-verdicts before block
ABUSE_WINDOW_S = 3600           # sliding window: 1 hour
ABUSE_BLOCK_DURATION_S = 86400  # block duration: 24 hours


class AbuseTracker:
    """Per-IP tracker for repeated nothing-verdict uploads.

    After each AI scan that returns "nothing", call ``record_nothing(ip)``.
    ``is_blocked(ip)`` returns True once the threshold is exceeded.
    """

    def __init__(self):
        self._lock = threading.Lock()
        # ip → list of timestamps of nothing-verdict uploads
        self._nothing_hits: dict[str, list[float]] = {}
        # ip → expiry timestamp of active block
        self._blocked_until: dict[str, float] = {}

    def is_blocked(self, ip: str) -> Optional[float]:
        """Return block expiry timestamp if IP is blocked, else None."""
        with self._lock:
            until = self._blocked_until.get(ip)
            if until and until > time.time():
                return until
            # Block expired — clear it
            if until:
                del self._blocked_until[ip]
            return None

    def record_nothing(self, ip: str) -> None:
        """Record a nothing-verdict upload and block the IP if over threshold."""
        now = time.time()
        cutoff = now - ABUSE_WINDOW_S

        with self._lock:
            hits = self._nothing_hits.setdefault(ip, [])
            # Prune old entries
            hits[:] = [t for t in hits if t > cutoff]
            hits.append(now)

            if len(hits) >= ABUSE_NOTHING_THRESHOLD:
                self._blocked_until[ip] = now + ABUSE_BLOCK_DURATION_S
                logger.warning(
                    "[abuse] IP %s blocked for %dh — %d nothing-verdicts in %ds",
                    ip, ABUSE_BLOCK_DURATION_S // 3600,
                    len(hits), ABUSE_WINDOW_S,
                )

    def get_blocked_ips(self) -> list[dict]:
        """Return list of currently blocked IPs with expiry info."""
        now = time.time()
        with self._lock:
            result = []
            for ip, until in list(self._blocked_until.items()):
                if until > now:
                    result.append({
                        "ip": ip,
                        "blocked_until": datetime.fromtimestamp(until, tz=timezone.utc).isoformat(),
                        "remaining_s": round(until - now),
                    })
                    # Also include their nothing count
                    hits = self._nothing_hits.get(ip, [])
                    cutoff = now - ABUSE_WINDOW_S
                    recent = [t for t in hits if t > cutoff]
                    result[-1]["recent_nothing_count"] = len(recent)
                else:
                    del self._blocked_until[ip]
            return result

    def unblock(self, ip: str) -> bool:
        """Manually unblock an IP. Returns True if it was blocked."""
        with self._lock:
            if ip in self._blocked_until:
                del self._blocked_until[ip]
                # Also clear the hit history so a fresh start
                self._nothing_hits.pop(ip, None)
                return True
            return False

    def _cleanup(self) -> None:
        """Remove expired blocks and old hit records."""
        now = time.time()
        cutoff = now - ABUSE_WINDOW_S
        with self._lock:
            # Expired blocks
            expired = [ip for ip, until in self._blocked_until.items()
                       if until <= now]
            for ip in expired:
                del self._blocked_until[ip]
            # Old hit records (for unblocked IPs)
            stale = [ip for ip, hits in self._nothing_hits.items()
                     if not hits or all(t <= cutoff for t in hits)]
            for ip in stale:
                del self._nothing_hits[ip]


_abuse_tracker = AbuseTracker()


# Allowed report source types
SOURCE_TYPES = {"citizen", "NASA", "Sentinel", "CCTV", "drone", "IoT", "ranger", "emergency services"}

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB
# Static assets are served with no-cache so deploys (rsync/scp of new
# files) reach visitors on their next load instead of after Flask's default
# 12h SEND_FILE_MAX_AGE window. Small files + conditional requests make
# the revalidation cost negligible; asset URLs in the pages also carry ?v=
# version params as a hard bust for any intermediate cache.
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

# Admin secret for dashboard access.
# Set env var WILDFRAME_ADMIN_SECRET or use the default (change in production!).
ADMIN_SECRET = os.environ.get("WILDFRAME_ADMIN_SECRET", "wildframe-admin")

# --- Corroboration-gated auto-approval -------------------------------
# A report with a positive AI verdict is auto-confirmed ONLY when an
# independent source corroborates it (a confirmed report within 500 m / 2 h,
# or a live FIRMS hotspot within 3 km / 12 h). Photo confidence alone never
# auto-approves. Set WILDFRAME_AUTO_APPROVE=0 to keep every report in the
# human-review queue. (Local-only WILDFRAME_AUTO_APPROVE_RELAXED=1 relaxes
# this: above-floor fires auto-confirm with no corroboration, and nearby
# pending positive-AI reports corroborate like confirmed ones — see the
# config block below.)
AUTO_APPROVE_ENABLED = os.environ.get("WILDFRAME_AUTO_APPROVE", "1") != "0"
# Class-specific confidence floors, tuned via sweep_yolo.py over the
# fire_dataset (755 fire / 244 clean) for the LOCAL ENSEMBLE — YOLOv26
# detector + fine-tuned EfficientNet-B0 classifier (fire_vision.py,
# WILDFRAME_VISION_ENSEMBLE=1). Merging the two by max per-class confidence
# lifts recall from 66% (YOLO alone @ 0.80/0.70) to 100% at (fire >= 0.60,
# smoke >= 0.70) with 5/244 false passes — best F1 99.7%. There is no
# zero-false-pass threshold anymore (the classifier's non_fire.178 scores
# fire=1.0), but the corroboration gate means a false pass alone never
# auto-approves; recall is the direction that matters here.
AUTO_APPROVE_FLAME_MIN_CONF = float(os.environ.get("WILDFRAME_AUTO_APPROVE_FLAME_CONF", "0.60"))
AUTO_APPROVE_SMOKE_MIN_CONF = float(os.environ.get("WILDFRAME_AUTO_APPROVE_SMOKE_CONF", "0.70"))

# Client-side pre-filter gate (MobileNetV3, static/app.js) — a soft veto.
# The gate's "no fire" verdict alone never blocks a report (it's a
# progressive enhancement and can be wrong), but under the relaxed flag it
# RAISES the flame floor so a low-confidence scan can't breeze through on
# the confidence tier alone. GATE_MIN_PROB mirrors the model's min_prob
# (train_gate.py threshold; the client warns the user below it).
GATE_MIN_PROB = float(os.environ.get("WILDFRAME_GATE_MIN_PROB", "0.4"))
GATE_VETO_FLAME_FLOOR = float(os.environ.get("WILDFRAME_GATE_VETO_FLAME_FLOOR", "0.85"))

# --- Local-only auto-approval relaxations (default OFF — production keeps
# corroboration-gated approval exactly as before) ---
# WILDFRAME_AUTO_APPROVE_RELAXED=1 (enabled in local .env, NOT the VM)
# relaxes two things, justified by the ensemble's benchmark (99.7% F1,
# 100% recall at 0.60/0.70 over 755 fire / 244 clean):
#   1. Above-floor fires auto-confirm with NO corroboration — the model
#      tier alone is enough, so high-confidence reports never sit in the
#      human review queue.
#   2. A nearby *pending* report with a positive AI verdict (from a
#      DIFFERENT session id) corroborates like a confirmed report or a
#      FIRMS hotspot — two independent observers are evidence, so reports
#      auto-accept when another fire report is nearby, not just on FIRMS.
#   3. The client-side pre-filter gate (MobileNetV3) gets a soft veto: a
#      "no-fire" verdict raises the flame floor to GATE_VETO_FLAME_FLOOR,
#      so a low-confidence scan can't confirm on the confidence tier alone.
AUTO_APPROVE_RELAXED = os.environ.get("WILDFRAME_AUTO_APPROVE_RELAXED", "0") != "0"


UPLOAD_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Reports and Bayesian grids now persist in PostgreSQL (see db.py) instead
# of data/*.json + in-memory registries, so concurrent requests and
# multiple workers can't lose updates or state on restart.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _allowed_file(name: str) -> bool:
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _load_reports(demo: bool = False) -> list[dict]:
    """Load all reports for a mode from Postgres."""
    return db.list_reports("demo" if demo else "production")


def _exif_gps(path: Path) -> Optional[tuple[float, float]]:
    """Extract (lat, lon) from image EXIF GPS metadata, if available."""
    try:
        img = Image.open(path)
        exif = img.getexif()
        if not exif:
            return None

        gps_info = {}
        for tag_id, value in exif.items():
            tag_name = TAGS.get(tag_id, tag_id)
            if tag_name == "GPSInfo":
                for gps_tag_id in value:
                    gps_tag_name = GPSTAGS.get(gps_tag_id, gps_tag_id)
                    gps_info[gps_tag_name] = value[gps_tag_id]
                break

        if not gps_info:
            return None

        def _to_decimal(values, ref):
            # PIL EXIF GPSLatitude/GPSLongitude is a tuple of (degrees, minutes, seconds)
            # each element is a Rational like (37, 1), unpack to float
            d = float(values[0]) if hasattr(values[0], 'numerator') else float(values[0])
            m = float(values[1]) if hasattr(values[1], 'numerator') else float(values[1])
            s = float(values[2]) if hasattr(values[2], 'numerator') else float(values[2])
            dec = d + m / 60.0 + s / 3600.0
            if ref in ("S", "W"):
                dec = -dec
            return dec

        lat = _to_decimal(gps_info["GPSLatitude"], gps_info["GPSLatitudeRef"]) if "GPSLatitude" in gps_info else None
        lon = _to_decimal(gps_info["GPSLongitude"], gps_info["GPSLongitudeRef"]) if "GPSLongitude" in gps_info else None

        if lat is not None and lon is not None:
            return (lat, lon)
    except Exception:
        pass
    return None


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two lat/lon points."""
    R = 6_371_000  # Earth radius in metres
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Clustering (single-pass, DBSCAN-inspired)
# ---------------------------------------------------------------------------

def _convert_heic_to_jpeg(filepath: Path) -> Path:
    """Convert an iPhone HEIC/HEIF upload to JPEG, preserving EXIF/GPS.

    Returns the path of the new JPEG (same stem, ``.jpg`` suffix) and removes
    the original HEIC staging file. Raises ValueError if conversion fails (e.g.
    ``pillow-heif`` not installed) so the caller can return a clean 400.
    """
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError as exc:
        raise ValueError(
            "HEIC support unavailable — install pillow-heif to accept iPhone photos"
        ) from exc

    try:
        im = Image.open(filepath)
        exif_bytes = im.info.get("exif")
        jpeg_path = filepath.with_suffix(".jpg")
        im.convert("RGB").save(jpeg_path, format="JPEG", quality=92, exif=exif_bytes)
    except Exception as exc:
        raise ValueError(f"Could not convert HEIC image: {exc}") from exc

    filepath.unlink(missing_ok=True)
    return jpeg_path


def _compute_clusters(reports: list[dict]) -> list[dict]:
    """
    Single-pass clustering:
      - Only active reports with status == 'confirmed' are clustered.
      - Pending reports show as individual markers awaiting moderation.
      - For each confirmed report, check distance to all existing cluster
        centroids. If within CLUSTER_RADIUS_M, merge; otherwise start a new
        cluster.
      - For each cluster with 2+ reports that have bearings, run
        triangulation to estimate the fire origin.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=CLUSTER_TIME_WINDOW_MINUTES)

    active = [
        r for r in reports
        if r.get("status", "pending") == "confirmed"
        and _parse_ts(r.get("captured_at", "")) >= cutoff
    ]

    clusters: list[dict] = []
    # Build a lookup from id -> report for fast access during triangulation
    active_by_id = {r["id"]: r for r in active}

    for r in active:
        rlat, rlon = r["lat"], r["lon"]
        best_idx = -1
        best_dist = float("inf")

        for i, c in enumerate(clusters):
            d = _haversine(rlat, rlon, c["centroid_lat"], c["centroid_lon"])
            if d < best_dist:
                best_dist = d
                best_idx = i

        if best_idx >= 0 and best_dist <= CLUSTER_RADIUS_M:
            c = clusters[best_idx]
            c["report_ids"].append(r["id"])
            c["points"].append([rlat, rlon])
            c["count"] = len(c["report_ids"])
            # Running average
            n = c["count"]
            c["centroid_lat"] = (c["centroid_lat"] * (n - 1) + rlat) / n
            c["centroid_lon"] = (c["centroid_lon"] * (n - 1) + rlon) / n
        else:
            clusters.append({
                "centroid_lat": rlat,
                "centroid_lon": rlon,
                "report_ids": [r["id"]],
                "points": [[rlat, rlon]],
                "count": 1,
            })

    # --- Run triangulation on each cluster ---
    for c in clusters:
        cluster_reports = [active_by_id[rid] for rid in c["report_ids"] if rid in active_by_id]
        t_result = triangulate_cluster(cluster_reports)
        if t_result["status"] == "ok":
            c["triangulation"] = t_result
        else:
            c["triangulation"] = None

    return clusters


def _parse_ts(ts: str) -> datetime:
    try:
        # Python < 3.11 does not support the "Z" suffix for UTC.
        # Normalize "Z" to "+00:00" so fromisoformat works everywhere.
        if isinstance(ts, str) and ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return datetime.min.replace(tzinfo=timezone.utc)


def _request_mode() -> str:
    """
    Resolve which data mode an API request is asking for.

    Reads ``?mode=`` from the query string (GET) or the ``"mode"`` JSON
    body field (POST). Anything other than "demo" resolves to
    "production" — the safe default — so demo data can never leak into
    operational endpoints by accident.
    """
    mode = request.args.get("mode")
    if mode is None and request.method == "POST":
        body = request.get_json(silent=True) or {}
        mode = body.get("mode")
    return "demo" if mode == "demo" else "production"


# ---------------------------------------------------------------------------
# Corroboration-gated auto-approval
#
# A positive AI photo verdict is treated as *suggestion*, not truth: the
# model's confidence is miscalibrated, so a report only auto-confirms when an
# INDEPENDENT source corroborates it — a nearby confirmed report, or a live
# FIRMS hotspot. Everything else stays "pending" for human moderation.
# ---------------------------------------------------------------------------

def _cluster_corroborated(report: dict, reports: list[dict]) -> bool:
    """True if an independent report exists within the cluster radius AND the
    2h time window of ``report``.

    Independent means: a *confirmed* report (any source), or — when
    AUTO_APPROVE_RELAXED=1 — a *pending* report whose own AI scan is
    positive (flame/smoke/both). Same-session submissions never corroborate
    (a phone re-submitting the same fire isn't independent evidence), and
    the candidate itself is excluded by id."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(minutes=CLUSTER_TIME_WINDOW_MINUTES)
    for other in reports:
        if other.get("id") == report.get("id"):
            continue
        ts = _parse_ts(other.get("captured_at", ""))
        if ts < cutoff:
            continue
        if (other.get("session_id") and report.get("session_id")
                and other.get("session_id") == report.get("session_id")):
            continue
        status = other.get("status")
        if status == "confirmed":
            pass
        elif AUTO_APPROVE_RELAXED and status == "pending":
            verdict = (other.get("ai_analysis") or {}).get("verdict")
            if verdict not in ("flame", "smoke", "both"):
                continue
        else:
            continue
        d = _haversine(report["lat"], report["lon"], other["lat"], other["lon"])
        if d <= CLUSTER_RADIUS_M:
            return True
    return False


def _satellite_corroboration(report: dict) -> dict:
    """Check live NASA FIRMS hotspots near the report. Fail-closed: a missing
    key or API outage returns confirmed=False with the error surfaced, so a
    scan can never be auto-approved on a broken satellite lookup."""
    api_key = nasa_firms._get_api_key()
    if not api_key:
        return {"confirmed": False, "error": "NASA_FIRMS_API_KEY not set"}
    try:
        hotspots = nasa_firms.fetch_hotspots_near(
            api_key=api_key,
            center_lat=report["lat"],
            center_lon=report["lon"],
            radius_km=_FIRMS_CONFIRM_RADIUS_M / 1000.0,
            day_range=2,
            min_confidence="low",
        )
    except (ConnectionError, ValueError) as exc:
        return {"confirmed": False, "error": str(exc)}
    result = _match_report_to_hotspots(
        report, hotspots,
        radius_m=_FIRMS_CONFIRM_RADIUS_M,
        window_hours=_FIRMS_CONFIRM_WINDOW_HOURS,
    )
    result["source"] = "auto-approval"
    return result


def _auto_approval_decision(
    verdict: Optional[str],
    fire_conf: float,
    smoke_conf: float,
    cluster_ok: bool,
    sat_ok: bool,
    gate_prob: Optional[float] = None,
    relaxed: bool = False,
) -> tuple[bool, Optional[str]]:
    """Decide auto-approval from an AI verdict plus corroboration.

    Rules:
      - verdict must be positive (flame/smoke/both); nothing/error/None never
        auto-approve, even with high confidence.
      - default (AUTO_APPROVE_RELAXED=0): the confidence floor AND at least
        one independent corroboration (cluster or satellite) are both
        required — the classic corroboration-gated rule.
      - relaxed (AUTO_APPROVE_RELAXED=1): EITHER is enough — above-floor
        fires auto-confirm with no corroboration (no human review), and a
        corroboration (nearby report or FIRMS) auto-confirms even a
        positive-but-below-floor scan.
      - client pre-filter gate soft veto (relaxed only): when the client-side
        gate says "no fire" (gate_prob below GATE_MIN_PROB), the flame floor
        RAISES to GATE_VETO_FLAME_FLOOR — a low-confidence scan can't ride
        the confidence tier alone. Corroboration (cluster/satellite) still
        wins. Absent/None gate_prob (gate not loaded client-side) vetoes
        nothing.
    Returns (approved, source) with sources like "confidence", "cluster",
    "satellite", "confidence+cluster", ... or (False, None).
    """
    if verdict not in ("flame", "smoke", "both"):
        return False, None
    flame_floor = AUTO_APPROVE_FLAME_MIN_CONF
    if relaxed and gate_prob is not None and gate_prob < GATE_MIN_PROB:
        # Client pre-filter found no fire — demand much more from the server
        # scan before the confidence tier alone can confirm.
        flame_floor = GATE_VETO_FLAME_FLOOR
    above_floor = fire_conf >= flame_floor or smoke_conf >= AUTO_APPROVE_SMOKE_MIN_CONF
    corr_ok = cluster_ok or sat_ok
    if relaxed:
        approved = above_floor or corr_ok
    else:
        approved = above_floor and corr_ok
    if not approved:
        return False, None
    sources = []
    if above_floor:
        sources.append("confidence")
    if cluster_ok:
        sources.append("cluster")
    if sat_ok:
        sources.append("satellite")
    return True, "+".join(sources)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/api/reports", methods=["GET"])
def list_reports():
    demo = _request_mode() == "demo"
    reports = _load_reports(demo=demo)
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=ACTIVE_REPORT_HOURS)
    active = [r for r in reports if _parse_ts(r.get("captured_at", "")) >= cutoff]
    # Photos are only exposed for APPROVED (confirmed) reports on the public
    # map. Pending/rejected/cancelled reports stay visible as markers for
    # context, but their photos are withheld until a moderator confirms
    # them — the admin endpoints (which must show photos pre-approval) read
    # the database directly and are unaffected.
    for r in active:
        if r.get("status") != "confirmed":
            r.pop("photo_url", None)
    return jsonify({"reports": active, "count": len(active), "mode": "demo" if demo else "production"})


@app.route("/api/reports", methods=["POST"])
def create_report():
    """Accept multipart upload with photo + GPS data."""
    # --- Rate limit (application-level, per IP) ---
    ip = request.headers.get("CF-Connecting-IP") or request.remote_addr or "unknown"
    rl_err = _check_rate_limit(ip, *UPLOAD_RATE_LIMIT)
    if rl_err:
        logger.warning("[rate-limit] upload blocked for %s: %s", ip, rl_err)
        return jsonify({"error": "too many uploads — please wait a minute"}), 429

    # --- IP abuse check: block IPs repeatedly uploading non-fire photos ---
    blocked_until = _abuse_tracker.is_blocked(ip)
    if blocked_until:
        remaining_h = max(1, int((blocked_until - time.time()) / 3600))
        logger.warning("[abuse] upload blocked for %s — blocked for ~%dh", ip, remaining_h)
        return jsonify({
            "error": f"Your IP has been temporarily blocked due to repeated non-fire uploads. Try again in {remaining_h}h.",
            "blocked_until": datetime.fromtimestamp(blocked_until, tz=timezone.utc).isoformat(),
        }), 403

    # --- Parse form fields ---
    lat = request.form.get("lat")
    lon = request.form.get("lon")
    heading = request.form.get("heading")  # nullable
    session_id = request.form.get("session_id", str(uuid.uuid4()))
    captured_at = request.form.get("captured_at", datetime.now(timezone.utc).isoformat())

    # --- GPS validation ---
    if lat and lon:
        try:
            lat = float(lat)
            lon = float(lon)
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid lat/lon values"}), 400
    else:
        # Fall back to device GPS not provided — try EXIF
        lat, lon = None, None

    # --- Photo ---
    file = request.files.get("photo")
    if not file or not file.filename:
        return jsonify({"error": "No photo provided"}), 400

    if not _allowed_file(file.filename):
        return jsonify({"error": f"File type not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    ext = secure_filename(file.filename).rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = UPLOAD_DIR / filename
    file.save(str(filepath))

    # --- iPhone HEIC/HEIF: convert to JPEG before anything reads the file ---
    # EXIF extraction, the AI scan, and photo storage can't handle raw HEIC
    # bytes, so normalize it to JPEG (EXIF/GPS preserved) right after staging.
    if ext in ("heic", "heif"):
        try:
            filepath = _convert_heic_to_jpeg(filepath)
            filename = filepath.name
            print(f"[photo] converted {ext} upload to JPEG ({filename})")
        except ValueError as exc:
            filepath.unlink(missing_ok=True)
            return jsonify({"error": str(exc)}), 400

    # --- Try EXIF GPS fallback ---
    if lat is None or lon is None:
        exif_gps = _exif_gps(filepath)
        if exif_gps:
            lat, lon = exif_gps
            print(f"[EXIF] Extracted GPS: {lat}, {lon}")

    if lat is None or lon is None:
        # Clean up uploaded file
        filepath.unlink(missing_ok=True)
        return jsonify({"error": "No GPS coordinates provided — enable location services or upload a photo with GPS EXIF data"}), 400

    # --- Run AI-powered fire/smoke detection FIRST ---
    # A positive verdict means the report is created for (possible)
    # auto-approval. A "nothing" verdict is NOT grounds to delete the
    # photo: the hosted Roboflow API is nondeterministic across time — the
    # exact same image bytes can score fire=0.62 one day and 0.000 the next
    # (measured with a 9796x3846 panorama) — so a borderline-but-real fire
    # would be silently lost. "Nothing" scans therefore become ordinary
    # PENDING reports for a human moderator to judge, never auto-approved.
    # If the scan errors out (no API key, server down, etc.), we still
    # allow the report through — better a false alarm than a missed fire.
    ai_verdict = None
    try:
        ai_result = scan_photo(filepath)
        ai_verdict = ai_result["verdict"]
        print(f"[AI-vision] verdict={ai_verdict} (confidence={ai_result['confidence']:.2f}, "
              f"fire={ai_result['fire_confidence']:.2f}, smoke={ai_result['smoke_confidence']:.2f}) — {filename}")
    except Exception as exc:
        print(f"[AI-vision] Scan failed: {exc} — proceeding with report anyway")
        ai_result = None

    # --- Track nothing-verdicts for IP abuse detection ---
    # If the AI scan completed and found no fire/smoke, record it.
    # Repeated nothing-verdicts from the same IP indicate abuse.
    if ai_verdict == "nothing":
        _abuse_tracker.record_nothing(ip)

    # --- Persist photo: S3 (if configured) or local disk ---
    # The file was staged on local disk for EXIF + AI scanning; now move
    # the accepted photo to its real home. photo_storage returns the URL
    # to store (S3 URL in S3 mode, /uploads/... otherwise) and removes
    # the staging copy in S3 mode.
    try:
        photo_url = photo_storage.store_photo(filename, filepath)
    except Exception as exc:
        # Upload to S3 failed — don't leave an orphaned file on ephemeral
        # disk; clean up the staging copy and tell the caller.
        filepath.unlink(missing_ok=True)
        print(f"[photo-storage] Failed to store photo {filename}: {exc}")
        return jsonify({"error": "Failed to store photo — check S3 configuration and try again."}), 500

    # --- Build report (reached for positive verdicts, "nothing" scans kept
    # for human review, and scan errors — never for missing GPS) ---
    report = {
        "id": uuid.uuid4().hex,
        "lat": lat,
        "lon": lon,
        "photo_url": photo_url,
        "captured_at": captured_at,
        "device_heading": float(heading) if heading else None,
        "session_id": session_id,
        "status": "pending",
        "source_type": "citizen",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # Attach AI analysis (if available) — store trimmed fields only
    if ai_result:
        report["ai_analysis"] = {
            "verdict": ai_result["verdict"],
            "confidence": ai_result["confidence"],
            "fire_confidence": ai_result["fire_confidence"],
            "smoke_confidence": ai_result["smoke_confidence"],
            "detection_count": ai_result["detection_count"],
            "model": ai_result["model"],
            "error": ai_result["error"],
        }
    else:
        report["ai_analysis"] = {
            "verdict": "error",
            "confidence": 0.0,
            "fire_confidence": 0.0,
            "smoke_confidence": 0.0,
            "detection_count": 0,
            "model": "roboflow-universe-projects/fire-and-smoke-segmentation/11",
            "error": "AI scan failed — report created for manual review",
        }

    # --- Client-side pre-filter gate (soft veto input) ---
    # static/app.js sends the MobileNetV3 fire probability it computed at
    # submit time (0..1, absent if the gate never loaded client-side). It
    # never blocks the upload — only the auto-approval floor reacts to it.
    gate_prob = None
    raw_gate = request.form.get("gate_prob")
    if raw_gate:
        try:
            gate_prob = float(raw_gate)
            gate_prob = max(0.0, min(1.0, gate_prob))
        except (TypeError, ValueError):
            gate_prob = None
    if gate_prob is not None:
        report["client_gate"] = {
            "prob": round(gate_prob, 4),
            "verdict": "fire" if gate_prob >= GATE_MIN_PROB else "no-fire",
        }

    # --- Auto-approval ---
    # Default (production): corroboration-gated — a positive AI verdict
    # auto-confirms only with independent corroboration (a nearby confirmed
    # report, or a live FIRMS hotspot); everything else stays "pending" for
    # human moderation. With WILDFRAME_AUTO_APPROVE_RELAXED=1 (local only):
    # above-floor fires auto-confirm with no corroboration, and nearby
    # pending reports with positive AI verdicts corroborate like confirmed
    # ones. The client gate's "no fire" verdict raises the flame floor
    # (soft veto) under the relaxed flag.
    if AUTO_APPROVE_ENABLED and ai_verdict in ("flame", "smoke", "both"):
        fire_conf = ai_result.get("fire_confidence", 0.0) if ai_result else 0.0
        smoke_conf = ai_result.get("smoke_confidence", 0.0) if ai_result else 0.0
        # Mirror the decision's soft veto so "above_floor" for the FIRMS
        # skip-check and the approval_class label use the same floor.
        flame_floor = AUTO_APPROVE_FLAME_MIN_CONF
        if (AUTO_APPROVE_RELAXED and gate_prob is not None
                and gate_prob < GATE_MIN_PROB):
            flame_floor = GATE_VETO_FLAME_FLOOR
        above_floor = fire_conf >= flame_floor or smoke_conf >= AUTO_APPROVE_SMOKE_MIN_CONF
        cluster_ok = _cluster_corroborated(report, _load_reports())
        sat = None
        # Skip the FIRMS lookup when the decision is already reachable: a
        # relaxed above-floor scan, or an existing cluster corroboration.
        if not cluster_ok and not (AUTO_APPROVE_RELAXED and above_floor):
            sat = _satellite_corroboration(report)
            report["satellite_confirmation"] = sat
        auto_approved, approval_source = _auto_approval_decision(
            ai_verdict, fire_conf, smoke_conf,
            cluster_ok, bool(sat and sat.get("confirmed")),
            gate_prob=gate_prob, relaxed=AUTO_APPROVE_RELAXED,
        )
        if auto_approved:
            report["status"] = "confirmed"
            report["auto_approved"] = True
            report["approval_source"] = approval_source
            report["approval_class"] = (
                "flame" if fire_conf >= flame_floor else "smoke"
            )
            report["approval_confidence"] = round(max(fire_conf, smoke_conf), 4)
            veto_note = (
                " (gate soft-veto: flame floor raised to "
                f"{GATE_VETO_FLAME_FLOOR:.2f})" if flame_floor > AUTO_APPROVE_FLAME_MIN_CONF else ""
            )
            print(f"[auto-approve] {report['id']} auto-confirmed via {approval_source} "
                  f"(class={report['approval_class']} "
                  f"@{report['approval_confidence']:.2f}, "
                  f"fire={fire_conf:.2f} smoke={smoke_conf:.2f}, "
                  f"gate={gate_prob}{veto_note})")
        else:
            report["auto_approved"] = False

    # Single-row INSERT — concurrent uploads can never clobber each other.
    db.insert_reports([report], "production")

    # Auto-approved reports feed the Bayesian grid immediately (same as an
    # admin accept), so the fire starts tracking right away.
    if report.get("status") == "confirmed":
        _feed_reports_into_grid([report])

    # --- Recompute clusters ---
    clusters = _compute_clusters(_load_reports())

    return jsonify({"report": report, "clusters": clusters, "cluster_count": len(clusters)}), 201


@app.route("/api/clusters", methods=["GET"])
def get_clusters():
    demo = _request_mode() == "demo"
    reports = _load_reports(demo=demo)
    clusters = _compute_clusters(reports)
    return jsonify({"clusters": clusters, "count": len(clusters), "mode": "demo" if demo else "production"})


@app.route("/api/reports/<report_id>/status", methods=["PUT"])
def update_status(report_id: str):
    data = request.get_json(silent=True) or {}
    new_status = data.get("status")
    if new_status not in ("pending", "confirmed", "rejected"):
        return jsonify({"error": "Invalid status. Use: pending, confirmed, rejected"}), 400

    # Atomic single-row UPDATE — no read-modify-write, so concurrent
    # moderations can't lose each other's changes.
    r = db.update_report_status(report_id, new_status)
    if r is None:
        return jsonify({"error": "Report not found"}), 404
    clusters = _compute_clusters(_load_reports())
    return jsonify({"report": r, "clusters": clusters, "cluster_count": len(clusters)})


@app.route("/api/reports/<report_id>/check-satellite", methods=["POST"])
def check_report_satellite(report_id: str):
    """
    On-demand: check a single report against live NASA FIRMS hotspot data
    near its location, and persist the result on the report.

    Unlike the background poller (which reuses one global fetch for all
    reports), this makes a single targeted FIRMS request scoped to the
    report's location, so it works even if the poller hasn't run yet.

    JSON body (optional):
      { "radius_km": 3, "window_hours": 12, "day_range": 2 }
    """
    report = db.get_report(report_id)
    if report is None:
        return jsonify({"error": "Report not found"}), 404

    # --- Rate limit (application-level, per IP) ---
    ip = request.headers.get("CF-Connecting-IP") or request.remote_addr or "unknown"
    rl_err = _check_rate_limit(ip, *SAT_CONFIRM_RATE_LIMIT)
    if rl_err:
        logger.warning("[rate-limit] sat-check blocked for %s: %s", ip, rl_err)
        return jsonify({"error": "too many satellite checks — please wait"}), 429

    api_key = nasa_firms._get_api_key()
    if not api_key:
        return jsonify({
            "error": "NASA_FIRMS_API_KEY not set — cannot check satellite data.",
        }), 400

    data = request.get_json(silent=True) or {}
    radius_km = float(data.get("radius_km", _FIRMS_CONFIRM_RADIUS_M / 1000.0))
    window_hours = float(data.get("window_hours", _FIRMS_CONFIRM_WINDOW_HOURS))
    day_range = int(data.get("day_range", 2))

    try:
        hotspots = nasa_firms.fetch_hotspots_near(
            api_key=api_key,
            center_lat=report["lat"],
            center_lon=report["lon"],
            radius_km=radius_km,
            day_range=day_range,
            min_confidence="low",
        )
    except (ConnectionError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 502

    result = _match_report_to_hotspots(
        report, hotspots, radius_m=radius_km * 1000.0, window_hours=window_hours,
    )
    result["source"] = "manual"
    report["satellite_confirmation"] = result
    db.update_report(report, "production")

    return jsonify({"report_id": report_id, "satellite_confirmation": result})


@app.route("/api/triangulate", methods=["POST"])
def api_triangulate():
    """
    Independently triangulate a set of bearing reports.

    Accepts JSON:
        {
            "reports": [
                {"lat": 38.174, "lon": 23.717, "device_heading": 45.0},
                {"lat": 38.176, "lon": 23.722, "device_heading": 48.0},
            ]
        }

    Returns the same dict as triangulate() plus an "input_reports"
    array echoing back what was received.
    """
    data = request.get_json(silent=True)
    if not data or "reports" not in data:
        return jsonify({"error": "Missing 'reports' array in request body"}), 400

    reports_in = data["reports"]
    if not isinstance(reports_in, list) or len(reports_in) == 0:
        return jsonify({"error": "'reports' must be a non-empty array"}), 400

    # Validate each report has the required fields
    for r in reports_in:
        if "lat" not in r or "lon" not in r:
            return jsonify({"error": "Each report requires 'lat' and 'lon'"}), 400
        if "device_heading" not in r:
            return jsonify({"error": "Each report requires 'device_heading' (use null if unknown)"}), 400

    result = triangulate(reports_in)
    result["input_reports"] = reports_in

    status_code = 200 if result["status"] == "ok" else 422
    return jsonify(result), status_code


@app.route("/api/seed", methods=["POST"])
def seed_data():
    """Populate the database with test wildfire reports from Greece."""
    import random as _random

    # Greece wildfire-prone locations, grouped by cluster with a common
    # fire origin per cluster so that headings converge for triangulation.
    # Each entry: (lat, lon, fire_origin_lat, fire_origin_lon)
    # fire_origin is the approximate true fire source for that cluster.
    greece_hotspots = [
        # Cluster 1: Mount Parnitha — fire origin near the mountain
        (38.1723, 23.7171, 38.174, 23.718),
        (38.1760, 23.7220, 38.174, 23.718),
        (38.1690, 23.7100, 38.174, 23.718),
        (38.1790, 23.7250, 38.174, 23.718),
        (38.1740, 23.7150, 38.174, 23.718),
        # Cluster 2: Evia island
        (38.6500, 23.9000, 38.653, 23.902),
        (38.6550, 23.9050, 38.653, 23.902),
        (38.6470, 23.8920, 38.653, 23.902),
        (38.6600, 23.9100, 38.653, 23.902),
        # Cluster 3: Ancient Olympia area
        (37.6500, 21.6250, 37.651, 21.624),
        (37.6540, 21.6300, 37.651, 21.624),
        (37.6470, 21.6180, 37.651, 21.624),
        # Cluster 4: Rhodes
        (36.1500, 28.0000, 36.153, 28.003),
        (36.1550, 28.0050, 36.153, 28.003),
        (36.1470, 27.9950, 36.153, 28.003),
        (36.1600, 28.0100, 36.153, 28.003),
        # Cluster 5: Crete / Chania area
        (35.4500, 23.9000, 35.452, 23.903),
        (35.4550, 23.9080, 35.452, 23.903),
        (35.4450, 23.8950, 35.452, 23.903),
        # Lone scattered reports (no cluster convergence — random headings)
        (38.0000, 22.5000, None, None),
        (39.5000, 22.0000, None, None),
        (40.5000, 23.0000, None, None),
    ]

    now = datetime.now(timezone.utc)
    # Seed data is DEMO data — it must never touch the production store.
    new_reports = []
    new_ids = []

    for i, pt in enumerate(greece_hotspots):
        lat, lon, fire_lat, fire_lon = pt
        # Add tiny jitter so reports aren't perfectly colocated
        lat += _random.uniform(-0.0005, 0.0005)
        lon += _random.uniform(-0.0005, 0.0005)

        # Stagger timestamps over the last 90 minutes
        ts = now - timedelta(minutes=_random.randint(0, 90))
        session = f"seed-gr-{_random.choice(['alpha','beta','gamma','delta','epsilon'])}"

        # Compute heading toward the cluster's fire origin (with noise)
        if fire_lat is not None and fire_lon is not None and _random.random() < 0.85:
            # Bearing from reporter toward fire origin (compass, clockwise from north)
            dlat = fire_lat - lat
            dlon = fire_lon - lon
            # Math angle: atan2(dlat, dlon) gives radians CCW from east
            # Compass bearing: (90 - math_deg + 360) % 360
            math_bearing = math.degrees(math.atan2(dlat, dlon))
            heading = ((90 - math_bearing) + 360) % 360
            # Add noise (±10°) for realism
            heading += _random.uniform(-10, 10)
            heading %= 360
        else:
            heading = None

        report = {
            "id": uuid.uuid4().hex,
            "lat": lat,
            "lon": lon,
            "photo_url": _gen_photo_url(i + 50),  # Offset to avoid overlapping with forest photos
            "captured_at": ts.isoformat(),
            "device_heading": heading,
            "session_id": session,
            "status": "confirmed",
            "source_type": "citizen",
            "created_at": now.isoformat(),
        }
        new_reports.append(report)
        new_ids.append(report["id"])

    db.insert_reports(new_reports, "demo")
    reports = _load_reports(demo=True)
    clusters = _compute_clusters(reports)

    # Build a fresh Bayesian grid sized for Greece only — not merged with
    # any existing reports from other regions.
    _seed_new_grid(new_reports, demo=True)

    return jsonify({
        "message": f"Seeded {len(greece_hotspots)} test reports across Greece",
        "seeded_count": len(greece_hotspots),
        "total_reports": len(reports),
        "clusters": clusters,
        "cluster_count": len(clusters),
        "mode": "demo",
    }), 201


@app.route("/uploads/<filename>")
def uploaded_file(filename: str):
    return send_from_directory(str(UPLOAD_DIR), filename)


@app.route("/healthz")
def healthz():
    """Cheap liveness/readiness probe for the host (PaaS healthchecks)."""
    try:
        db.ping()
        return jsonify({"status": "ok", "db": "ok"}), 200
    except Exception:
        # Never expose exception details (may contain DB credentials)
        return jsonify({"status": "degraded", "db": "error"}), 503


@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "landing.html")


@app.route("/map")
def map_page():
    return send_from_directory(str(STATIC_DIR), "map.html")


@app.route("/contact")
def contact_page():
    """Public contact / partnerships page for investors and agencies."""
    return send_from_directory(str(STATIC_DIR), "contact.html")


@app.route("/privacy")
def privacy_page():
    """Public privacy policy page."""
    return send_from_directory(str(STATIC_DIR), "privacy.html")


@app.route("/about")
def about_page():
    """Public "Who we are" page."""
    return send_from_directory(str(STATIC_DIR), "about.html")


@app.route("/faq")
def faq_page():
    """Public FAQ page with a suggestion / bug-report submission form."""
    return send_from_directory(str(STATIC_DIR), "faq.html")


@app.route("/wildfire-early-detection-for-emergency-services")
def emergency_services_page():
    """Public page targeting emergency services, agencies, utilities and insurers."""
    return send_from_directory(str(STATIC_DIR), "emergency-services.html")


@app.route("/investors")
def investors_page():
    """Public page for investors and potential partners."""
    return send_from_directory(str(STATIC_DIR), "investors.html")


@app.route("/nasa-firms-wildfire-map")
def nasa_firms_map_page():
    """SEO intent page: live NASA FIRMS satellite fire detection map."""
    return send_from_directory(str(STATIC_DIR), "nasa-firms-wildfire-map.html")


@app.route("/how-early-wildfire-detection-works")
def how_detection_works_page():
    """SEO intent page: how early wildfire detection works."""
    return send_from_directory(str(STATIC_DIR), "how-early-wildfire-detection-works.html")


@app.route("/wildfire-spread-risk-map")
def wildfire_spread_risk_page():
    """SEO intent page: wildfire spread risk map and predictive modelling."""
    return send_from_directory(str(STATIC_DIR), "wildfire-spread-risk-map.html")


# Poland bbox for the /map/poland SEO landing page (west,south,east,north).
POLAND_BBOX = (14.07, 49.00, 24.15, 54.84)  # full bbox for SEO page / PostGIS pre-filter

# Simplified Poland boundary polygon (~20 vertices) for accurate
# centroid-in-country filtering.  Replaces the bbox check which
# incorrectly includes CZ/SK/DE/BY/LT/RU territory.
POLAND_POLYGON: list[tuple[float, float]] = [
    (54.84, 18.95),   # north coast (Gdansk area)
    (54.50, 22.80),   # NE — Kaliningrad border
    (54.00, 23.50),   # Sejny
    (53.50, 24.00),   # Grodno border
    (52.10, 24.00),   # Brest border
    (51.25, 24.00),   # Wlodawa
    (50.40, 24.00),   # Lublin SE
    (49.80, 23.50),   # Przemysl
    (49.00, 22.50),   # Bieszczady
    (49.00, 20.00),   # Tatras
    (49.20, 18.50),   # Cieszyn
    (49.50, 17.50),   # Jeseniky border
    (50.10, 17.00),   # Glubczyce
    (50.50, 16.50),   # Kłodzko
    (51.10, 15.50),   # Zgorzelec
    (51.80, 14.70),   # Słubice
    (52.80, 14.30),   # Schwedt border
    (53.80, 14.25),   # Świnoujście
    (54.50, 16.50),   # Darłowo
]


def _point_in_poland(lat: float, lon: float) -> bool:
    """Fast ray-cast point-in-polygon for the simplified Poland boundary."""
    # POLAND_POLYGON is defined as (lat, lon) pairs; _point_in_polygon
    # expects (lon, lat) pairs, so we swap each point.
    pts = [(p[1], p[0]) for p in POLAND_POLYGON]
    return _point_in_polygon(lon, lat, pts)


@app.route("/map/poland")
def map_poland_page():
    """SEO landing page: "Live wildfire map of Poland".

    Serves the static page with two server-side substitutions so the HTML
    that crawlers see contains the live numbers (the page is otherwise
    identical to the on-disk file):

      {{ACTIVE_FIRES}} — number of currently-tracked fires in Poland
                         (grids with max_p >= the map's 0.1 visibility floor)
      {{COVERAGE}}     — coverage label

    Falls back gracefully: if the DB query fails, the placeholders remain
    and the page still renders (the client-side map also re-fetches the
    live count).
    """
    html = (STATIC_DIR / "map_poland.html").read_text(encoding="utf-8")
    try:
        grids = db.list_grid_meta("production", bbox=POLAND_BBOX)
        active = sum(1 for g in grids if (g.get("max_p") or 0) >= 0.1)
    except Exception:  # noqa: BLE001 — landing page must never 500
        active = None
    if active is not None:
        html = html.replace("{{ACTIVE_FIRES}}", str(active))
    html = html.replace("{{COVERAGE}}", "Poland")
    return html, 200, {"Content-Type": "text/html; charset=utf-8"}


# ---------------------------------------------------------------------------
# Public feedback endpoint (FAQ page form)
# ---------------------------------------------------------------------------

FEEDBACK_CATEGORIES = {"suggestion", "bug", "question", "other"}
FEEDBACK_MAX_MESSAGE = 4000
FEEDBACK_THROTTLE_S = 30.0  # per-IP cooldown between submissions


@app.route("/api/feedback", methods=["POST"])
def feedback_submit():
    """Store a suggestion / bug report / question from the FAQ page.

    Light anti-abuse: per-IP cooldown via kv_store (30 s), category
    whitelist, and length caps. No auth — the page is public by design.
    """
    # --- Rate limit (application-level, per IP) ---
    ip = request.headers.get("CF-Connecting-IP") or request.remote_addr or "unknown"
    rl_err = _check_rate_limit(ip, "feedback", 3, 60)  # 3 per minute
    if rl_err:
        return jsonify({"error": "please wait before submitting again"}), 429

    body = request.get_json(silent=True) or {}
    category = str(body.get("category", "")).strip().lower()
    message = str(body.get("message", "")).strip()

    if category not in FEEDBACK_CATEGORIES:
        return jsonify({
            "error": "category must be one of: " + ", ".join(sorted(FEEDBACK_CATEGORIES))
        }), 400
    if not message:
        return jsonify({"error": "message is required"}), 400
    if len(message) > FEEDBACK_MAX_MESSAGE:
        return jsonify({"error": f"message must be under {FEEDBACK_MAX_MESSAGE} characters"}), 400

    ip = request.remote_addr or "unknown"
    throttle_key = f"feedback_throttle:{ip}"
    last = db.kv_get(throttle_key) or {}
    if last.get("at") and time.time() - last.get("at") < FEEDBACK_THROTTLE_S:
        return jsonify({"error": "please wait a moment before submitting again"}), 429
    db.kv_set(throttle_key, {"at": time.time()})

    name = str(body.get("name", "")).strip()[:120] or None
    email = str(body.get("email", "")).strip()[:200] or None
    fid = db.submit_feedback(
        category=category,
        message=message,
        name=name,
        email=email,
        page="faq",
    )
    logger.info("Feedback submitted id=%s category=%s from %s", fid, category, ip)
    return jsonify({"ok": True, "id": fid}), 201


# ---------------------------------------------------------------------------
# Admin Helpers & Routes
# ---------------------------------------------------------------------------

def _require_admin():
    """Check the X-Admin-Secret header against the configured secret.

    Uses hmac.compare_digest for constant-time comparison to prevent
    timing attacks on the admin secret."""
    auth = request.headers.get("X-Admin-Secret", "")
    if not auth or not ADMIN_SECRET:
        return False
    return hmac.compare_digest(auth, ADMIN_SECRET)


# Cookie proving this browser already validated the admin key. HttpOnly so
# page JS can't read it; the X-Admin-Secret header check remains the real
# gate on every admin data endpoint.
ADMIN_COOKIE = "wf_admin_ok"
ADMIN_COOKIE_MAX_AGE = 12 * 3600  # 12 hours


def _admin_cookie_value() -> str:
    """Signed cookie value — an HMAC of the admin secret. A plain static
    value (e.g. "1") would be forgeable: any visitor could set
    ``document.cookie = "wf_admin_ok=1"`` from the same-origin map page and
    the /admin gate would open for them. Signing keeps the gate honest
    without putting the secret itself in the cookie."""
    return hmac.new(ADMIN_SECRET.encode(), b"wf-admin-page", hashlib.sha256).hexdigest()


@app.route("/admin")
def admin_page():
    """Serve the admin dashboard only to browsers that proved the secret.

    The page itself is a login shell — every admin data endpoint still
    requires the X-Admin-Secret header — but gating the page keeps the
    admin surface undiscoverable on a public deployment:

      * /admin            -> 404 unless the signed wf_admin_ok cookie is present
      * /admin?key=<sec>  -> constant-time-compares the key, sets the
                             HttpOnly cookie, redirects to /admin (so the
                             key leaves the address bar)

    The key only ever appears in the operator's own request (it is logged
    once by the server for that single redirect request).
    """
    if request.cookies.get(ADMIN_COOKIE) == _admin_cookie_value():
        return send_from_directory(str(STATIC_DIR), "admin.html")

    key = request.args.get("key", "")
    if key:
        # Rate-limit admin login attempts (brute-force protection)
        ip = request.headers.get("CF-Connecting-IP") or request.remote_addr or "unknown"
        rl_err = _check_rate_limit(ip, "admin_login", 5, 300)  # 5 per 5 min
        if rl_err:
            logger.warning("[rate-limit] admin login blocked for %s", ip)
            return jsonify({"error": "too many login attempts — try again later"}), 429
        if hmac.compare_digest(key, ADMIN_SECRET):
            resp = redirect("/admin", code=302)
            resp.set_cookie(
                ADMIN_COOKIE, _admin_cookie_value(),
                max_age=ADMIN_COOKIE_MAX_AGE, httponly=True, samesite="Lax",
                secure=request.is_secure,
            )
            return resp

    return jsonify({"error": "Not found"}), 404


@app.route("/api/admin/status", methods=["GET"])
def admin_status():
    """Public: does this browser hold the admin cookie? The frontend calls
    this on boot to decide whether to show the Admin button. Deliberately
    not secret-gated — it only reveals whether the visitor already proved
    the key (button stays hidden otherwise, fail-closed)."""
    return jsonify({"authed": request.cookies.get(ADMIN_COOKIE) == _admin_cookie_value()})


@app.route("/api/admin/pending", methods=["GET"])
def admin_list_pending():
    if not _require_admin():
        return jsonify({"error": "Unauthorized"}), 401
    pending = db.list_reports("production", status="pending")
    return jsonify({"reports": pending, "count": len(pending)})


@app.route("/api/admin/feedback", methods=["GET"])
def admin_list_feedback():
    """List user feedback submissions (FAQ page form), newest first."""
    if not _require_admin():
        return jsonify({"error": "Unauthorized"}), 401
    items = db.list_feedback(limit=100)
    return jsonify({"feedback": items, "count": len(items)})


@app.route("/api/admin/feedback/<feedback_id>", methods=["DELETE"])
def admin_delete_feedback(feedback_id: str):
    """Delete one feedback submission."""
    if not _require_admin():
        return jsonify({"error": "Unauthorized"}), 401
    if not db.delete_feedback(feedback_id):
        return jsonify({"error": "not found"}), 404
    return jsonify({"ok": True}), 200


@app.route("/api/admin/auto-approved", methods=["GET"])
def admin_list_auto_approved():
    """Recently auto-approved (confirmed) reports, so a human can spot-check
    them and reject any where the corroboration was wrong. Auto-approval is
    fast-tracked, not irreversible — this list is the backstop."""
    if not _require_admin():
        return jsonify({"error": "Unauthorized"}), 401
    # "Recently auto-approved" — bound to the last 48h so we never scan the
    # full confirmed list (which includes FIRMS-ingested reports) on every
    # 8s poll; the strip is a human backstop, not an archive.
    reports = db.list_reports("production", status="confirmed", since_hours=ACTIVE_REPORT_HOURS)
    auto = [r for r in reports if r.get("auto_approved")][:20]
    return jsonify({"reports": auto, "count": len(auto)})


@app.route("/api/admin/grids", methods=["GET"])
def admin_list_grids():
    """Live-fire overview for the admin dashboard: the active production
    Bayesian grids with their current probability, wind and EFFIS
    fuel-moisture context (FFMC/DMC/ISI + the moisture_factor the model is
    actually applying). Lets a human spot-check the fire-weather data
    feeding the spread model.

    Bounded to the top ``limit`` by peak probability so the 8s admin poll
    never scans thousands of stale FIRMS grids."""
    if not _require_admin():
        return jsonify({"error": "Unauthorized"}), 401
    # Spot-checking order is applied IN SQL (fwi_first) so the LIMIT window
    # can never drop the fires that actually have fuel-moisture data — they
    # rank first regardless of how far down the max_p ordering they sit.
    rows = db.list_grid_meta(
        "production", limit=ADMIN_GRIDS_LIMIT, fwi_first=True,
    )
    now = time.time()
    grids = []
    for r in rows:
        ffmc = float(r["ffmc"] or 0.0)
        moisture_factor = effis_fwi.moisture_factor(ffmc) if ffmc > 0 else 1.0
        evidence_age_h = (
            (now - float(r["last_evidence_at"])) / 3600.0
            if float(r["last_evidence_at"] or 0.0) > 0
            else None
        )
        grids.append({
            "id": r["id"],
            "lat": r["centroid_lat"],
            "lon": r["centroid_lon"],
            "max_p": float(r["max_p"]),
            "wind_speed": float(r["wind_speed"]),
            "wind_dir_deg": float(r["wind_dir_deg"]),
            "ffmc": ffmc,
            "dmc": float(r["dmc"] or 0.0),
            "isi": float(r["isi"] or 0.0),
            "moisture_factor": moisture_factor,
            "evidence_age_h": evidence_age_h,
        })
    return jsonify({"grids": grids, "count": len(grids)})


@app.route("/api/admin/blocked-ips", methods=["GET"])
def admin_list_blocked_ips():
    """List currently blocked IPs and their abuse stats."""
    if not _require_admin():
        return jsonify({"error": "Unauthorized"}), 401
    blocked = _abuse_tracker.get_blocked_ips()
    return jsonify({"blocked_ips": blocked, "count": len(blocked)})


@app.route("/api/admin/unblock-ip/<ip>", methods=["POST"])
def admin_unblock_ip(ip: str):
    """Manually unblock an IP address."""
    if not _require_admin():
        return jsonify({"error": "Unauthorized"}), 401
    was_blocked = _abuse_tracker.unblock(ip)
    return jsonify({"ok": True, "was_blocked": was_blocked})


@app.route("/api/admin/accept/<report_id>", methods=["POST"])
def admin_accept(report_id: str):
    if not _require_admin():
        return jsonify({"error": "Unauthorized"}), 401
    r = db.update_report_status(report_id, "confirmed")
    if r is None:
        return jsonify({"error": "Report not found"}), 404
    clusters = _compute_clusters(_load_reports())
    # Feed into Bayesian grid
    _feed_reports_into_grid([r])
    return jsonify({"report": r, "clusters": clusters,
                    "cluster_count": len(clusters)})


@app.route("/api/admin/accept-all", methods=["POST"])
def admin_accept_all():
    """Accept all pending reports at once."""
    if not _require_admin():
        return jsonify({"error": "Unauthorized"}), 401
    # One atomic UPDATE confirms every pending report at once.
    accepted_reports = db.accept_all_pending("production")
    count = len(accepted_reports)
    clusters = _compute_clusters(_load_reports())
    # Feed into Bayesian grid
    if accepted_reports:
        _feed_reports_into_grid(accepted_reports)
    return jsonify({
        "success": True,
        "accepted_count": count,
        "clusters": clusters,
        "cluster_count": len(clusters),
        "message": f"Accepted {count} pending report(s)",
    })


@app.route("/api/admin/reject/<report_id>", methods=["POST"])
def admin_reject(report_id: str):
    if not _require_admin():
        return jsonify({"error": "Unauthorized"}), 401
    r = db.get_report(report_id)
    if r is None:
        return jsonify({"error": "Report not found"}), 404

    # Delete the uploaded photo (S3 object or local file)
    photo_url = r.get("photo_url", "")
    if photo_url:
        photo_storage.delete_photo(photo_url)

    db.delete_report(report_id)
    clusters = _compute_clusters(_load_reports())
    return jsonify({"success": True,
                    "clusters": clusters,
                    "cluster_count": len(clusters)})


# ---------------------------------------------------------------------------
# Forest / Natural Park Seed Data
# ---------------------------------------------------------------------------

# Realistic citizen phone photo placeholders (nature/scenery from Picsum)
# Each URL uses a seed so the same report always shows the same photo
_PHOTO_SEEDS = ["forest1","forest2","forest3","forest4","forest5","tree1","tree2","tree3",
                "mountain1","mountain2","smoke1","smoke2","fire1","fire2","wildfire1","wildfire2",
                "landscape1","landscape2","landscape3","valley1","valley2","ridge1","ridge2"]


def _gen_photo_url(idx: int) -> str:
    """Generate a consistent photo URL for a seed report."""
    seed = _PHOTO_SEEDS[idx % len(_PHOTO_SEEDS)]
    return f"https://picsum.photos/seed/{seed}/400/300"


def _build_reports_from_hotspots(
    hotspots: list[tuple],
    prefix: str = "seed",
) -> tuple[list[dict], list[str]]:
    """
    Shared helper: given a list of hotspot tuples of the form
        (lat, lon, fire_origin_lat, fire_origin_lon, source_type, session_suffix)
    produce report dicts with computed headings toward the fire origin.

    Returns (list_of_report_dicts, list_of_new_ids).
    """
    import random as _random

    now = datetime.now(timezone.utc)
    reports = []
    new_ids = []

    for i, pt in enumerate(hotspots):
        lat, lon, fire_lat, fire_lon, source_type, session_suffix = pt
        # Add tiny jitter
        lat += _random.uniform(-0.0005, 0.0005)
        lon += _random.uniform(-0.0005, 0.0005)

        ts = now - timedelta(minutes=_random.randint(0, 90))
        session = f"{prefix}-{session_suffix}-{_random.choice(['a','b','c','d','e'])}"

        # Compute heading toward fire origin (with noise)
        if fire_lat is not None and fire_lon is not None and _random.random() < 0.85:
            dlat = fire_lat - lat
            dlon = fire_lon - lon
            math_bearing = math.degrees(math.atan2(dlat, dlon))
            heading = ((90 - math_bearing) + 360) % 360
            heading += _random.uniform(-12, 12)
            heading %= 360
        else:
            heading = None

        report = {
            "id": uuid.uuid4().hex,
            "lat": lat,
            "lon": lon,
            "photo_url": _gen_photo_url(i),
            "captured_at": ts.isoformat(),
            "device_heading": heading,
            "session_id": session,
            "status": "confirmed",
            "source_type": "citizen",  # All seed reports are citizen phone photos
            "created_at": now.isoformat(),
        }
        reports.append(report)
        new_ids.append(report["id"])

    return reports, new_ids


@app.route("/api/seed/forest", methods=["POST"])
def seed_forest():
    """
    Seed realistic wildfire reports around Yosemite National Park and surrounding
    Sierra Nevada forest — all as citizen phone-photo reports.

    Four fire clusters:
      1. Yosemite Valley floor / Half Dome area
      2. Mariposa Grove (giant sequoias)
      3. Tuolumne Meadows (high country)
      4. Hetch Hetchy / Grand Canyon of the Tuolumne
      5. Lone scattered reports in the park
    """

    # Each hotspot: (lat, lon, fire_origin_lat, fire_origin_lon, source_type, session_suffix)
    # source_type is always "citizen" — the helper will set it
    yosemite_hotspots = [
        # ---- Cluster 1: Yosemite Valley / Half Dome ----
        # Fire origin near Half Dome (37.744, -119.533)
        (37.7210, -119.5380, 37.742, -119.533, "citizen", "yose-valley"),
        (37.7270, -119.5450, 37.742, -119.533, "citizen", "yose-valley"),
        (37.7360, -119.5500, 37.742, -119.533, "citizen", "yose-valley"),
        (37.7460, -119.5570, 37.742, -119.533, "citizen", "yose-valley"),
        (37.7410, -119.5220, 37.742, -119.533, "citizen", "yose-valley"),
        (37.7150, -119.5250, 37.742, -119.533, "citizen", "yose-valley"),
        # ---- Cluster 2: Mariposa Grove (south of park) ----
        # Fire origin in the grove 37.514, -119.604
        (37.5050, -119.6000, 37.514, -119.604, "citizen", "mariposa"),
        (37.5180, -119.6100, 37.514, -119.604, "citizen", "mariposa"),
        (37.5100, -119.5950, 37.514, -119.604, "citizen", "mariposa"),
        (37.5220, -119.6080, 37.514, -119.604, "citizen", "mariposa"),
        # ---- Cluster 3: Tuolumne Meadows ----
        # Fire origin near Lembert Dome 37.878, -119.348
        (37.8700, -119.3550, 37.878, -119.348, "citizen", "tuolumne"),
        (37.8750, -119.3400, 37.878, -119.348, "citizen", "tuolumne"),
        (37.8830, -119.3500, 37.878, -119.348, "citizen", "tuolumne"),
        (37.8670, -119.3450, 37.878, -119.348, "citizen", "tuolumne"),
        (37.8850, -119.3600, 37.878, -119.348, "citizen", "tuolumne"),
        # ---- Cluster 4: Hetch Hetchy ----
        # Fire origin 37.947, -119.788
        (37.9400, -119.7820, 37.947, -119.788, "citizen", "hetch"),
        (37.9500, -119.7920, 37.947, -119.788, "citizen", "hetch"),
        (37.9350, -119.7800, 37.947, -119.788, "citizen", "hetch"),
        (37.9530, -119.7950, 37.947, -119.788, "citizen", "hetch"),
        # ---- Lone scattered reports (no cluster convergence) ----
        (37.8000, -119.8000, None, None, "citizen", "scatter"),
        (37.6500, -119.4500, None, None, "citizen", "scatter"),
        (37.9000, -119.6000, None, None, "citizen", "scatter"),
    ]

    reports_list, new_ids = _build_reports_from_hotspots(yosemite_hotspots, prefix="yose")

    # Seed data is DEMO data — it must never touch the production store.
    db.insert_reports(reports_list, "demo")
    existing = _load_reports(demo=True)
    clusters = _compute_clusters(existing)

    # Build a fresh Bayesian grid sized for Yosemite only — not merged with
    # any existing reports from other regions.
    _seed_new_grid(reports_list, demo=True)

    return jsonify({
        "message": f"Seeded {len(yosemite_hotspots)} test reports across Yosemite National Park",
        "seeded_count": len(yosemite_hotspots),
        "total_reports": len(existing),
        "clusters": clusters,
        "cluster_count": len(clusters),
        "centroid_lat": 37.745,
        "centroid_lon": -119.593,
        "mode": "demo",
    }), 201


# ---------------------------------------------------------------------------
# Live Demo — Progressive Triangulation Simulation
# ---------------------------------------------------------------------------

# In-memory state for the live demo
_live_demo: Optional[dict] = None

# Bayesian probability grids now persist in PostgreSQL (see db.py). There
# are TWO independent registries (production vs demo, selected by mode) so
# that demo/simulation grids can never influence operational outputs — the
# same guarantee as before, but durable across restarts and multiple workers.
#
# Every mutation runs as a row-locked transaction (db.mutate_grid) so
# concurrent requests/workers serialize per grid instead of losing each
# other's evidence.


# ---------------------------------------------------------------------------
# Bayesian Grid Helpers
#
# One BayesianFireGrid models the local spread of ONE fire. Since a batch of
# reports can span multiple, geographically unrelated fires (e.g. the Greece
# seed data covers 5 clusters spread across the whole country), we keep a
# separate grid per fire cluster instead of one grid sized to fit everything
# at once — that would force a huge cell size (and misses small fires
# entirely once the span gets large enough that a fire's whole footprint is
# smaller than one cell). Clusters are matched to grids by centroid
# proximity so an evolving fire keeps using the same grid across calls.
#
# Every helper takes ``demo`` so production grids and demo grids are never
# mixed: seed data / live demo / historic replay / simulated satellite all
# use ``demo=True``; real reports and real FIRMS data use ``demo=False``.
# ---------------------------------------------------------------------------

def _find_or_create_grid_for_cluster(cluster: dict, demo: bool = False, cell_size_m: float = DEFAULT_CELL_SIZE_M) -> tuple[str, BayesianFireGrid]:
    """Return (grid_id, grid) tracking this cluster's fire, creating one if
    needed. Creation is serialized in Postgres (advisory lock on a location
    bucket), so concurrent workers can't create duplicate grids.

    New grids (and any grid that never received real weather — e.g. ones
    created before weather.py existed, whose ``wind_updated_at`` is 0) get
    real wind from Open-Meteo at their centroid; matched grids keep their
    stored wind, which the worker's periodic refresh keeps current."""
    mode = "demo" if demo else "production"
    grid_id, entry = db.find_or_create_grid(mode, cluster, cell_size_m=cell_size_m)

    # Only fetch weather for PRODUCTION fires (demo/seed grids keep the
    # deterministic defaults) and only when the grid has no fresh wind yet
    # (brand-new, or pre-weather and never refreshed). fetch succeeded?
    if not demo and (entry.get("wind_updated_at") or 0) <= 0:
        speed, dir_deg, fetched_at = weather.get_wind_full(
            cluster["centroid_lat"], cluster["centroid_lon"],
        )
        if fetched_at > 0 and db.update_grid_wind(mode, grid_id, speed, dir_deg):
            entry["wind_speed"] = speed
            entry["wind_dir_deg"] = dir_deg
            print(f"[weather] {grid_id} wind from Open-Meteo: {speed:.1f} m/s "
                  f"toward {dir_deg:.0f}° ({cluster['centroid_lat']:.3f}, "
                  f"{cluster['centroid_lon']:.3f})")

    return grid_id, entry["grid"]


def _sync_grids_from_clusters(reports: list[dict], clusters: list[dict], demo: bool = False) -> None:
    """Ensure each cluster has a backing grid, seeded with its own reports.

    Each grid is seeded inside a row-locked transaction so concurrent
    workers never clobber each other's evidence."""
    mode = "demo" if demo else "production"
    for cluster in clusters:
        grid_id, _entry = _find_or_create_grid_for_cluster(cluster, demo=demo, cell_size_m=CITIZEN_CELL_SIZE_M)
        cluster_reports = [r for r in reports if r["id"] in cluster["report_ids"]]

        def _seed(grid: BayesianFireGrid, entry: dict, _cr=cluster_reports, _cl=clusters) -> None:
            seed_from_reports(grid, _cr, _cl, wind_dir_deg=entry["wind_dir_deg"])

        db.mutate_grid(mode, grid_id, _seed)


def _init_bayesian_grid(demo: bool = False) -> None:
    """(Re)build all per-cluster grids from scratch from confirmed reports."""
    mode = "demo" if demo else "production"
    db.delete_grids(mode)
    reports = _load_reports(demo=demo)
    clusters = _compute_clusters(reports)
    _sync_grids_from_clusters(reports, clusters, demo=demo)


def _seed_new_grid(seed_reports: list[dict], demo: bool = True) -> None:
    """
    Build fresh grid(s) scoped ONLY to the given batch (one per fire cluster
    within it), replacing whatever grids currently exist in the requested
    mode's registry. Used by seed endpoints so a demo dataset doesn't merge
    with unrelated existing fires — and (since seeds are always demo data)
    never touches the production registry.
    """
    mode = "demo" if demo else "production"
    db.delete_grids(mode)

    all_reports = _load_reports(demo=demo)
    clusters = _compute_clusters(all_reports)

    seed_ids = {r["id"] for r in seed_reports}
    relevant_clusters = [c for c in clusters if set(c["report_ids"]) & seed_ids]

    _sync_grids_from_clusters(seed_reports, relevant_clusters, demo=demo)


def _feed_reports_into_grid(reports: list[dict], demo: bool = False) -> None:
    """Feed newly confirmed reports into their per-cluster grid(s)."""
    mode = "demo" if demo else "production"
    if db.count_grids(mode) == 0:
        _init_bayesian_grid(demo=demo)
        return

    all_reports = _load_reports(demo=demo)
    clusters = _compute_clusters(all_reports)
    new_ids = {r["id"] for r in reports}
    relevant_clusters = [c for c in clusters if set(c["report_ids"]) & new_ids]
    _sync_grids_from_clusters(all_reports, relevant_clusters, demo=demo)


def _find_or_create_grid_for_point(lat: float, lon: float, demo: bool = False) -> tuple[str, BayesianFireGrid]:
    """Like _find_or_create_grid_for_cluster, but for a single lat/lon (used
    by the manual evidence-injection endpoint)."""
    fake_cluster = {
        "centroid_lat": lat, "centroid_lon": lon,
        "points": [[lat, lon]],
    }
    return _find_or_create_grid_for_cluster(fake_cluster, demo=demo, cell_size_m=CITIZEN_CELL_SIZE_M)


# Maximum grids serialized in one /api/bayesian/state response. The map
# only renders what's in the viewport (bbox filter), but a world-wide view
# can still contain hundreds of fires — cap it so a single request never
# serializes ~30s of contours.
MAX_STATE_GRIDS = 120

# Overflow tier: grids ranked beyond MAX_STATE_GRIDS (up to this many
# more) still ship, but COARSE — only cells above COARSE_CELL_THRESHOLD
# and no contour — so a dense viewport renders ~2x the fires in roughly
# the same payload budget. Without it, the 120-grid cap silently dropped
# ~700 of ~800 visible fires in the Africa fire belt (blobs rendered "in
# parts" / cut on zoom). The client culls cells it can't see, so shipping
# a coarse core for the overflow fires is strictly better than dropping
# them.
COARSE_GRID_OVERFLOW = 120
COARSE_CELL_THRESHOLD = 0.10

# Total candidates considered per full-detail request: the top
# MAX_STATE_GRIDS ship full resolution, the next COARSE_GRID_OVERFLOW ship
# coarse, and any beyond that are returned as cheap "overflow" dots (meta
# shape) so a dense viewport never shows holes — every fire is represented
# (blob, coarse blob, or dot) instead of silently vanishing. Capped at the
# same budget as the meta-dot view.
STATE_CANDIDATE_CAP = 600

# detail=meta (low-zoom dot view) only ships id/centroid/max_p/wind — no
# state loading, no contour extraction — so it can afford a larger cap.
META_MAX_GRIDS = 600

# Contour extraction (marching squares) is the single most expensive grid
# operation (~30ms/grid). Keep the response small by flooring the cell
# threshold so we never ship every cell of every grid (at threshold 0.02 a
# typical grid ships 1-10 cells; at 0.001 it ships ~13k).
MIN_STATE_THRESHOLD = 0.005

# In-process cache for serialized grid exports (the read-only path of
# /api/bayesian/state). Keyed by (mode, grid_id, persisted-state version,
# elapsed bucket, threshold, contour). The map's 5s polls only recompute
# a grid's expensive contour when the grid actually changed (worker
# checkpoint, evidence injection) or the elapsed bucket rolls over —
# otherwise they're served from here. Bounded; cleared wholesale on
# overflow (rare, and cheap to rebuild).
_EXPORT_CACHE: dict = {}
_EXPORT_CACHE_LOCK = threading.Lock()
# Trimmed from 4000 for the 1 GB Lightsail VM: each entry holds a full
# serialized export (cells + statistics + contour), so 4000 could eat
# ~100 MB+ of the VM's scarce RAM (it was already swapping). 1200 keeps
# the hot viewport cached while freeing memory for numpy state loads.
_EXPORT_CACHE_MAX = 1200
# Quantize elapsed time into buckets so exports stay stable (and cached)
# between worker checkpoints while the fire still visibly advances.
# 45s (was 15s): the bucket rolls over 3x less often, so the expensive
# full recompute (numpy load + predict + contour) happens 3x less — on
# the 1 GB VM a bucket roll over a busy viewport was a multi-second stall
# on every 15s. Fires still animate between checkpoints, just in 45s
# steps instead of 15s (imperceptible on a heatmap).
_EXPORT_DT_BUCKET_S = 45.0

# Read-path fallback predict gate: if a grid hasn't been checkpointed by
# the worker for this long, the read path extrapolates it in-memory (never
# persisted — grids.advance owns persistence). Raised from 2s so steady
# 5s polls don't churn predict() for every grid on every poll.
_PREDICT_GATE_S = 10.0


def _export_cache_get(key: tuple) -> Optional[dict]:
    with _EXPORT_CACHE_LOCK:
        return _EXPORT_CACHE.get(key)


def _export_cache_set(key: tuple, value: dict) -> None:
    with _EXPORT_CACHE_LOCK:
        if len(_EXPORT_CACHE) >= _EXPORT_CACHE_MAX:
            # Evict the oldest 20 % instead of clearing the entire cache.
            # Full clear causes a thundering herd: every concurrent poller
            # recomputes contours simultaneously, spiking latency on the
            # 1 GB VM.  Dicts preserve insertion order in Python 3.7+, so
            # iterating the first N keys removes the oldest entries.
            evict_count = max(1, _EXPORT_CACHE_MAX // 5)
            for i, old_key in enumerate(_EXPORT_CACHE):
                if i >= evict_count:
                    break
                _EXPORT_CACHE.pop(old_key, None)
        _EXPORT_CACHE[key] = value


def _parse_bbox_param(raw: Optional[str]) -> Optional[tuple[float, float, float, float]]:
    """Parse a 'west,south,east,north' bbox string into a tuple, or None."""
    if not raw:
        return None
    parts = raw.split(",")
    if len(parts) != 4:
        return None
    try:
        w, s, e, n = (float(p) for p in parts)
    except ValueError:
        return None
    if w >= e or s >= n:
        return None
    return (w, s, e, n)


# How many stale-wind grids in the visible viewport one poll may hand to
# the background wind-refresh thread. Covers the full-detail cap (120) and
# most of the meta-dot cap (600); repeat polls cost nothing once the
# shared per-cell weather cache is warm (24h TTL, ~55 km cells — see
# weather.py's sizing note: a finer grid would blow the free-tier budget).
VIEWPORT_WIND_LIMIT = 600

# How many stale grids ONE refresh thread refreshes before it exits. The
# map polls every few seconds, so instead of one thread crawling through
# hundreds of sequential live fetches (tens of minutes for a cold region),
# each poll's thread refreshes a bounded slice — the most probable fires
# first — and the next poll (after the guard clears) continues with the
# rest. Keeps visible progress fast and the budget spend per poll bounded.
VIEWPORT_WIND_SLICE = 100

# Per-process guard: the map polls every few seconds; without this, a user
# staring at a cold region would pile up one refresh thread per poll, each
# re-scanning the same rows. Only one viewport thread runs per web worker
# at a time (different gunicorn workers still parallelize, and the shared
# per-cell cache + daily budget bound total API usage).
_VIEWPORT_WIND_LOCK = threading.Lock()
_VIEWPORT_WIND_ACTIVE = False


def _spawn_viewport_wind_refresh(mode: str, bbox: Optional[tuple[float, float, float, float]]) -> None:
    """Refresh real wind for the most probable fires in ``bbox`` in the
    background, so the visible map shows real wind within a poll cycle.

    The worker's global sweep (grids.advance) walks grids in
    ``wind_updated_at`` order and can take hours to reach a region the
    user just panned to — and it shares the daily weather budget with
    everything else. Instead of waiting on it, every /api/bayesian/state
    poll hands a bounded slice of the stale grids in the CURRENT viewport
    to a short-lived daemon thread that fetches + persists their wind.
    The shared ~28 km cell cache (weather.py) makes repeat polls free, the
    per-poll slice keeps progress visible fast, and the daily budget caps
    total API usage regardless of how many users poll.

    Demo grids keep the deterministic defaults by design, so this only
    runs for production.
    """
    global _VIEWPORT_WIND_ACTIVE
    if mode != "production" or not bbox:
        return
    with _VIEWPORT_WIND_LOCK:
        if _VIEWPORT_WIND_ACTIVE:
            return  # previous poll's thread is still working
        _VIEWPORT_WIND_ACTIVE = True

    def _run() -> None:
        global _VIEWPORT_WIND_ACTIVE
        try:
            rows = db.list_grid_meta(mode, bbox=bbox, limit=VIEWPORT_WIND_LIMIT)
            cutoff = time.time() - weather.CACHE_TTL_S
            stale = [r for r in rows if (r.get("wind_updated_at") or 0) < cutoff]
            for row in stale[:VIEWPORT_WIND_SLICE]:
                speed, dir_, fetched = weather.get_wind_full(
                    row["centroid_lat"], row["centroid_lon"],
                )
                if fetched > 0:
                    db.update_grid_wind(mode, row["id"], speed, dir_)
        except Exception as exc:
            logger.warning("[weather] viewport wind refresh failed: %s", exc)
        finally:
            with _VIEWPORT_WIND_LOCK:
                _VIEWPORT_WIND_ACTIVE = False

    threading.Thread(target=_run, daemon=True, name="viewport-wind").start()


def _round_export_state(state: dict) -> dict:
    """Round cell lat/lon/p to trim the JSON payload (a few decimals are
    invisible at map zoom levels but cut the state blob by ~30%)."""
    cells = state.get("cells")
    if cells:
        state["cells"] = [
            {"lat": round(c["lat"], 5), "lon": round(c["lon"], 5), "p": round(c["p"], 4)}
            for c in cells
        ]
    return state


def _round_contour(contour: list) -> list:
    """Round contour polyline coordinates (lat/lon) to 5 decimals."""
    return [
        [[round(pt[0], 5), round(pt[1], 5)] for pt in segment]
        for segment in contour
    ]


def _grid_to_json(
    threshold: float = 0.02,
    contour_level: float = 0.6,
    demo: bool = False,
    bbox: Optional[tuple[float, float, float, float]] = None,
    max_grids: int = MAX_STATE_GRIDS,
    detail: str = "full",
    include_contour: bool = True,
) -> dict:
    """
    Serialize Bayesian grids in the requested viewport for the frontend.

    Read-only: prediction is checkpointed by the worker (grids.advance);
    this function only extrapolates grids in-memory between checkpoints and
    NEVER writes to the database — the old per-poll predict+persist under a
    row lock is gone (it made every map poll hammer Postgres).

    ``detail="meta"`` (low-zoom dot view) ships only id/centroid/max_p/wind
    — no state loading, no marching squares — so world-scale views stay
    fast even with thousands of fires.

    Serialized exports are cached in-process keyed by the grid's persisted
    state version (``updated_at``) plus a 15s elapsed bucket, so the map's
    5s polls don't recompute contours for every grid on every poll.
    """
    mode = "demo" if demo else "production"
    meta_only = detail == "meta"
    # Full-detail path: MAX_STATE_GRIDS grids at full resolution PLUS
    # COARSE_GRID_OVERFLOW more at coarse resolution (see the constants).
    # The ordering from list_grid_meta is visible-first, so the overflow
    # tier is always filled with on-screen fires before any off-screen
    # margin-only grid is considered.
    effective_max = META_MAX_GRIDS if meta_only else STATE_CANDIDATE_CAP

    # Candidates come straight from Postgres: viewport-filtered via PostGIS
    # and ordered by peak probability (denormalized max_p column), capped so
    # one request never serializes ~30s of contours.
    items = db.list_grid_meta(mode, bbox=bbox, limit=effective_max)

    # Exclude grids whose centroid falls inside a conflict zone.
    # The FIRMS pass already drops new hotspots in these zones, but grids
    # created before the filter was activated remain in the DB.
    if SOURCE_FILTER_CONFLICT in ("drop", "downweight"):
        _czones = db.kv_get("conflict_zones") or []
        if _czones:
            _prepared_cz = _prepare_conflict_zones(_czones)
            if _prepared_cz:
                items = [
                    row for row in items
                    if not _point_in_prepared_zones(
                        row["centroid_lon"], row["centroid_lat"], _prepared_cz
                    )
                ]

    # Optional country filter — only applied when the client sends
    # ``country=pl`` (the /map/poland SEO page).  The main map has
    # no country filter so fires everywhere are visible.
    country = request.args.get("country", "")
    if country == "pl":
        items = [
            row for row in items
            if _point_in_poland(row["centroid_lat"], row["centroid_lon"])
        ]

    if meta_only:
        # Cheap dot view — no state loads, no contour extraction.
        grids_out = [
            {
                "id": row["id"],
                "lat": round(row["centroid_lat"], 5),
                "lon": round(row["centroid_lon"], 5),
                "max_p": round(row["max_p"], 4),
                # No real weather yet (wind_updated_at = 0 means the grid was
                # never refreshed from Open-Meteo) → null, never the fake
                # 3.0 m/s West default. The UI shows "N/A" for null.
                "wind_speed": round(row["wind_speed"], 1) if (row.get("wind_updated_at") or 0) > 0 else None,
                "wind_dir_deg": round(row["wind_dir_deg"], 0) if (row.get("wind_updated_at") or 0) > 0 else None,
            }
            for row in items
        ]
        return {
            "grids": grids_out,
            "returned_grids": len(grids_out),
            "detail": "meta",
        }

    now_ts = time.time()
    grids_out = []

    # Cache-first pass: the key only needs the meta row (updated_at + the
    # last_predict_time we now expose from the state JSONB in SQL), so a
    # repeated poll of the same viewport hits the export cache WITHOUT
    # loading/deserializing any numpy state. Only the grids that missed the
    # cache are batch-loaded below (one query, not one per grid).
    missing = []
    for idx, row in enumerate(items):
        # Rows beyond the full+coarse budget are only used as overflow dots
        # (see below) — never serialized.
        if idx >= max_grids + COARSE_GRID_OVERFLOW:
            continue
        grid_id = row["id"]
        # Ranks beyond MAX_STATE_GRIDS ship coarse (fewer cells, no contour)
        # so the cap never silently drops an on-screen fire in dense areas.
        coarse = idx >= max_grids
        eff_threshold = COARSE_CELL_THRESHOLD if coarse else threshold
        eff_contour = include_contour and not coarse
        # Full microsecond precision (TIMESTAMPTZ) so two writes to one grid
        # within the same second still invalidate the cache.
        updated_epoch = row["updated_at"].timestamp() if row.get("updated_at") else 0.0

        # In-memory extrapolation bucket: the export is keyed by how much
        # wall-clock time has passed since the last persisted predict, so
        # fires keep advancing in ~15s steps between worker checkpoints
        # while the cache absorbs every poll within a bucket.
        lpt = row.get("last_predict_time") or 0.0
        elapsed = max(0.0, now_ts - lpt) if lpt > 0 else 0.0
        bucket = int(elapsed / _EXPORT_DT_BUCKET_S)

        key = (
            mode, grid_id, updated_epoch, bucket,
            round(eff_threshold, 4), round(contour_level, 4), eff_contour,
        )
        cached = _export_cache_get(key)
        if cached is not None:
            grids_out.append(cached)
            continue
        missing.append((row, key, coarse))

    # Batch-load only the grids that missed the cache (one query total).
    if missing:
        entries = db.get_grid_entries_batch(mode, [row["id"] for row, _, _ in missing])
        for row, key, coarse in missing:
            entry = entries.get(row["id"])
            if entry is None:
                continue  # grid deleted between queries
            grid = entry["grid"]
            lpt = grid.last_predict_time
            elapsed = max(0.0, now_ts - lpt) if lpt > 0 else 0.0

            # Expand grid if fire is near the boundary — even when predict
            # is skipped (recent checkpoint).  Without this, the export
            # shows the old DB dimensions and the fire appears clipped at
            # a straight grid edge.
            _radius = max(10, int(math.ceil(
                (DEFAULT_BASE_SPREAD_RATE * (1.0 + 15.0 * WIND_HEAD_FACTOR))
                * (min(elapsed, 600.0) / 60.0) / grid.cell_size
            ))) if lpt > 0 else 10
            _did_expand = grid._expand_for_predict(_radius)

            # Fallback extrapolation (never persisted — the worker owns
            # persistence via grids.advance). Raised gate (~10s) keeps steady
            # state cheap; dt capped at 10 minutes so fires can't run away
            # between checkpoints.
            #
            # Same fuel-moisture factors as the worker's checkpoint
            # (grids.advance → db.advance_grids) so the between-poll
            # animation doesn't silently drift toward wind-only behaviour
            # on a dry-fuel fire (ffmc=0 / outside coverage → neutral 1.0).
            #
            # When the grid was JUST expanded (fire was at boundary), always
            # run a short predict to spread probability into the new cells —
            # without this, the new cells stay at prior (0.01) and the heatmap
            # still looks clipped even though the grid grew.
            _needs_predict = (
                lpt > 0 and (elapsed > _PREDICT_GATE_S or _did_expand)
            )
            if _needs_predict:
                _ffmc = float(entry.get("ffmc") or 0.0)
                _dmc = float(entry.get("dmc") or 0.0)
                grid.predict(
                    dt=min(elapsed, 600.0),
                    wind_speed=entry.get("wind_speed", 3.0),
                    wind_dir_deg=entry.get("wind_dir_deg", 270.0),
                    moisture_factor=effis_fwi.moisture_factor(_ffmc) if _ffmc > 0 else 1.0,
                    decay_scale=effis_fwi.decay_scale(_dmc) if _dmc > 0 else 1.0,
                    slope_lookup=db.dem_terrain_lookup,
                )

            # Null (not a fake 3.0/270) until the grid has real weather.
            has_wind = (entry.get("wind_updated_at") or 0) > 0
            eff_threshold = COARSE_CELL_THRESHOLD if coarse else threshold
            eff_contour = include_contour and not coarse
            out = {
                "id": row["id"],
                "state": _round_export_state(grid.export_state(threshold=eff_threshold)),
                "statistics": grid.get_statistics(),
                "wind_speed": round(entry["wind_speed"], 1) if has_wind else None,
                "wind_dir_deg": round(entry["wind_dir_deg"], 0) if has_wind else None,
            }
            if eff_contour:
                out["contour"] = _round_contour(grid.export_contour(level=contour_level))

            _export_cache_set(key, out)
            grids_out.append(out)

    # Overflow dots: candidates beyond the full+coarse budget ship as cheap
    # meta-shaped dots (id/centroid/max_p — no state loads) so the client
    # can still represent every fire in a dense viewport instead of leaving
    # a hole. The visible-first ordering means these are always the
    # weakest / most marginal fires, never an on-screen one.
    overflow = [
        {
            "id": row["id"],
            "lat": round(row["centroid_lat"], 5),
            "lon": round(row["centroid_lon"], 5),
            "max_p": round(row["max_p"], 4),
        }
        for row in items[max_grids + COARSE_GRID_OVERFLOW:]
    ]

    return {
        "grids": grids_out,
        "overflow": overflow,
        "returned_grids": len(grids_out),
        "capped": bool(overflow),
        "detail": "full",
    }


@app.route("/api/conflict-zones", methods=["GET"])
def get_conflict_zones():
    """Return conflict zone geometries for client-side rendering.

    Public endpoint — no auth needed (the zones are informational).
    The client renders these as hover overlays explaining why satellite
    fires are hidden in conflict regions.
    """
    raw = db.kv_get("conflict_zones") or []
    zones = []
    for z in raw:
        ztype = z.get("type")
        if ztype == "bbox":
            zones.append({
                "type": "bbox",
                "bounds": [z["s"], z["w"], z["n"], z["e"]],  # Leaflet LatLngBounds
                "label": z.get("label", "Conflict zone"),
            })
        elif ztype == "polygon":
            pts = z.get("points", [])
            if pts:
                zones.append({
                    "type": "polygon",
                    "points": [[p[1], p[0]] for p in pts],  # [lat,lon] for Leaflet
                    "label": z.get("label", "Conflict zone"),
                })
    return jsonify({"zones": zones})


@app.route("/api/bayesian/state", methods=["GET"])
def bayesian_get_state():
    """
    Get the current Bayesian probability grid state.

    Query params:
      - threshold: minimum probability to include (default 0.02; floored
        to MIN_STATE_THRESHOLD so we never ship every cell of a grid)
      - contour:   contour level (default 0.6; 0 skips contour extraction)
      - bbox:      "west,south,east,north" viewport — only fires in this
                   area are serialized (keeps global FIRMS data fast)
      - detail:    "full" (default) or "meta" — meta ships cheap
                   id/centroid/max_p dots for low-zoom views

    Read-only: fires are checkpointed by the worker (grids.advance); this
    route extrapolates in-memory between checkpoints and never writes.
    Serialized exports are cached per grid state version.
    """
    threshold = request.args.get("threshold", 0.02, type=float)
    if threshold < MIN_STATE_THRESHOLD:
        threshold = MIN_STATE_THRESHOLD
    contour_level = request.args.get("contour", 0.6, type=float)
    demo = _request_mode() == "demo"
    bbox = _parse_bbox_param(request.args.get("bbox"))

    detail = request.args.get("detail", "full")
    if detail not in ("meta", "full"):
        detail = "full"
    # contour=0 is the frontend's sentinel for "skip marching squares"
    # (the cache key still needs a sane contour value).
    include_contour = contour_level > 0
    if not include_contour:
        contour_level = 0.6

    try:
        data = _grid_to_json(
            threshold=threshold,
            contour_level=contour_level,
            demo=demo,
            bbox=bbox,
            detail=detail,
            include_contour=include_contour,
        )
        # Lazy on-demand wind: refresh the visible viewport in the
        # background so wherever the user looks, fires get real wind on the
        # next poll — regardless of where the worker's global sweep is.
        _spawn_viewport_wind_refresh("demo" if demo else "production", bbox)
        data["mode"] = "demo" if demo else "production"
        return jsonify(data)
    except Exception:
        # Never expose exception details (may contain DB credentials)
        logger.exception("grid data endpoint error")
        return jsonify({"error": "internal error"}), 500



# ---------------------------------------------------------------------------
# Road-risk overlay — continuous ellipse spread rate to roads
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# OSM Road Data Cache (in-memory + disk-persisted)
#
# Roads don't move, so once we've successfully fetched road segments for a
# given location, we NEVER need to fetch them again. The cache is persisted
# to ``data/osm_road_cache.json`` so it survives server restarts.
#
# Multiple Overpass API endpoints are tried in order if one fails, and the
# default search radius is 2 km (not 5) to keep queries light and fast.
# ---------------------------------------------------------------------------

OSM_CACHE_FILE = DATA_DIR / "osm_road_cache.json"

# Multiple Overpass API endpoints to try in order (fallback on failure)
# OSM API endpoints tried in order. Overpass mirrors first, then the main
# OSM API (which returns XML, not JSON, but is generally more reliable).
_OSM_API_ENDPOINTS = [
    {"url": "https://overpass-api.de/api/interpreter", "type": "overpass"},
    {"url": "https://overpass.kumi.systems/api/interpreter", "type": "overpass"},
    {"url": "https://api.openstreetmap.org/api/0.6/map", "type": "osmapi"},
]

# The OSM road cache persists in Postgres (osm_road_cache table, see
# db.py) instead of a JSON file + in-memory dict, so it's shared correctly
# across multiple workers and never written with a racy read-modify-write.
_osm_cache_lock = threading.Lock()

# Track request timestamps per cache_key to enforce our own rate limit
_osm_request_timestamps: dict[str, list[float]] = {}

# Single-flight: per-key locks so N concurrent misses for the same
# location trigger exactly one Overpass HTTP request (the others
# wait on the lock and reuse the cached result).
_osm_inflight_locks: dict[str, threading.Lock] = {}

# Last-failure timestamps per cache_key: if every endpoint just failed
# for a key, short-circuit repeat calls for a cooldown window instead
# of hammering the mirrors again (they were already down or
# rate-limited seconds ago).
_osm_failed_at: dict[str, float] = {}
_OSM_FAIL_COOLDOWN_S = 30.0

# How many Overpass HTTP requests we allow per cache_key per minute before
# falling back to cached/stale data (protects both Overpass and our own
# error handling).
_OSM_REQ_BUDGET_PER_MIN = 20  # accounts for multiple fire clusters × endpoint fallbacks


def _osm_cache_key(lat: float, lon: float, radius_km: float) -> str:
    """Cache key: ~100m spatial precision, exact radius."""
    return f"{round(lat, 3)},{round(lon, 3)},{radius_km}"


def _osm_enforce_client_rate_limit(cache_key: str) -> bool:
    """
    Enforce our own per-key request budget so we don't hammer Overpass.
    Returns True if the request is allowed, False if we should use
    cached/stale data instead.
    """
    now = time.time()
    window_start = now - 60.0
    with _osm_cache_lock:
        timestamps = _osm_request_timestamps.get(cache_key, [])
        timestamps = [t for t in timestamps if t > window_start]
        if len(timestamps) >= _OSM_REQ_BUDGET_PER_MIN:
            print(f"[road-cache] REQUEST BUDGET EXCEEDED for {cache_key} ({len(timestamps)}/{_OSM_REQ_BUDGET_PER_MIN} in last 60s)")
            return False
        timestamps.append(now)
        _osm_request_timestamps[cache_key] = timestamps
        return True


def _parse_osm_xml_roads(raw_xml: bytes) -> list[list[tuple[float, float]]]:
    """
    Parse an OSM XML response (from the main OSM API /api/0.6/map) into
    road segment polylines. Only includes major highways
    (motorway/trunk/primary/secondary/tertiary and their _link variants).

    Uses only the standard library ``xml.etree.ElementTree``.
    """
    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(raw_xml)
    except ET.ParseError as e:
        raise ValueError(f"Invalid OSM XML response: {e}")

    # Collect all nodes: id -> (lat, lon)
    nodes: dict[int, tuple[float, float]] = {}
    for node_el in root.findall("node"):
        nid = int(node_el.get("id", "0"))
        lat = float(node_el.get("lat", "0"))
        lon = float(node_el.get("lon", "0"))
        nodes[nid] = (lat, lon)

    # Regex to match major highway types
    import re
    highway_pattern = re.compile(r"^(motorway|trunk|primary|secondary|tertiary)(_link)?$")

    segments: list[list[tuple[float, float]]] = []
    for way_el in root.findall("way"):
        # Check for highway tag
        tags = {t.get("k"): t.get("v") for t in way_el.findall("tag")}
        highway = tags.get("highway", "")
        if not highway_pattern.match(highway):
            continue

        # Build segment from node references
        seg = []
        for nd_el in way_el.findall("nd"):
            ref = int(nd_el.get("ref", "0"))
            if ref in nodes:
                seg.append(nodes[ref])

        if len(seg) >= 2:
            segments.append(seg)

    return segments


def _fetch_osm_roads(
    center_lat: float,
    center_lon: float,
    radius_km: float = 2.0,
) -> list[list[tuple[float, float]]]:
    """
    Fetch road segments from OpenStreetMap near a given point.

    Tries multiple API endpoints (Overpass mirrors → main OSM API), with
    persistent caching so each location is fetched at most once.

    **Caching**: roads don't move, so once fetched, data is saved to the
    Postgres ``osm_road_cache`` table (see db.py) forever. Uses fuzzy
    matching: if the exact contour centroid isn't cached, any cached
    centroid within 1.5 km with the same radius is reused (the contour
    shifts slightly on every poll as the fire spreads).

    **Endpoint order**: Overpass mirrors first, then the main OSM API
    (returns XML, more reliable). Each endpoint is tried until one
    succeeds or all fail.

    **Single-flight**: N concurrent misses for the same cache key trigger
    exactly one HTTP request; the other callers wait on a per-key lock and
    reuse the result.

    **Scoped fallback**: when only the main OSM API succeeds it covers a
    smaller area (capped at 1.5 km), so its result is cached under a
    scoped 1.5 km key — never under the requested full-radius key (that
    would poison the cache with a partial dataset).

    **Empty results are never cached**: a 0-segment response can mean the
    API degraded, so we return ``[]`` without persisting it.

    **Rate limiting**: max ``_OSM_REQ_BUDGET_PER_MIN`` (20) requests per
    cache key per minute. Fuzzy-matched lookups don't consume budget.

    **Failure short-circuit**: if every endpoint just failed for a key,
    repeat calls within ``_OSM_FAIL_COOLDOWN_S`` skip the network entirely
    and serve stale data (or raise) instead of hammering the mirrors.

    Parameters
    ----------
    radius_km : float
        Search radius in km. Default 2.0 (not 5.0) to reduce query weight.

    Returns a list of road segment polylines, each being a list of
    (lat, lon) coordinate pairs.

    Raises Exception if ALL endpoints fail AND no cached/stale data exists.
    """
    import urllib.request
    import urllib.error
    import urllib.parse

    cache_key = _osm_cache_key(center_lat, center_lon, radius_km)

    def _parse_cache_key(k: str):
        parts = k.split(",")
        if len(parts) == 3:
            try:
                return float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
                pass
        return None

    # ---- Exact cache hit (no lock needed) ----
    entry = db.osm_get(cache_key)
    if entry and entry.get("segments"):
        print(f"[road-cache] HIT for {cache_key} ({len(entry['segments'])} segments)")
        return entry["segments"]

    # ---- Fuzzy cache lookup: same radius, centroid within 1.5 km ----
    # The contour centroid shifts slightly on every poll as the fire
    # spreads, so an exact key match may miss even though we have data
    # for a nearby location. Scan all cache entries with the same radius
    # and find the closest one within FUZZY_MATCH_DISTANCE_M.
    FUZZY_MATCH_DISTANCE_M = 1500.0
    for k, v in db.osm_iter():
        parsed = _parse_cache_key(k)
        if not parsed or not v.get("segments"):
            continue
        ck_lat, ck_lon, ck_radius = parsed
        if abs(ck_radius - radius_km) > 0.01:
            continue  # different radius — skip
        d = _haversine(center_lat, center_lon, ck_lat, ck_lon)
        if d < FUZZY_MATCH_DISTANCE_M:
            print(f"[road-cache] FUZZY HIT for {cache_key} (matched {k}, {d:.0f}m away, {len(v['segments'])} segments)")
            # Migrate to exact key for faster lookup next time
            db.osm_set(cache_key, v["segments"], v.get("stored_at", time.time()))
            return v["segments"]

    # ---- Failure short-circuit: don't re-hammer mirrors we just lost ----
    last_fail = _osm_failed_at.get(cache_key)
    if last_fail is not None and time.time() - last_fail < _OSM_FAIL_COOLDOWN_S:
        cached = db.osm_get(cache_key)
        if cached and cached.get("segments"):
            print(f"[road-cache] SHORT-CIRCUIT (failed {time.time() - last_fail:.0f}s ago) — serving stale for {cache_key}")
            return cached["segments"]
        raise Exception("All OSM API endpoints failed.")

    # ---- Single-flight: one fetch per key, N waiters reuse the result ----
    with _osm_inflight_locks.setdefault(cache_key, threading.Lock()):
        # A waiter may have filled the cache while we waited for the lock.
        entry = db.osm_get(cache_key)
        if entry and entry.get("segments"):
            print(f"[road-cache] HIT for {cache_key} ({len(entry['segments'])} segments)")
            return entry["segments"]

        # ---- Enforce our own client-side rate budget ----
        if not _osm_enforce_client_rate_limit(cache_key):
            cached = db.osm_get(cache_key)
            if cached and cached.get("segments"):
                print(f"[road-cache] BUDGET EXCEEDED — using cached data for {cache_key}")
                return cached["segments"]
            raise Exception(
                f"Request budget exceeded for this area ({_OSM_REQ_BUDGET_PER_MIN}/min). "
                f"Try again in a few minutes."
            )

        # ---- Cache miss — fetch from Overpass, trying endpoints in order ----
        print(f"[road-cache] MISS for {cache_key} — fetching from Overpass...")

        radius_m = int(radius_km * 1000)

        # Optimised query: exact-tag union (indexed lookups, NOT a regex
        # scan) restricted to major road types only, to keep payloads small
        # and avoid Overpass query-timeout rejections.
        query = f"""
        [out:json][timeout:15];
        (
          way["highway"="motorway"](around:{radius_m},{center_lat},{center_lon});
          way["highway"="motorway_link"](around:{radius_m},{center_lat},{center_lon});
          way["highway"="trunk"](around:{radius_m},{center_lat},{center_lon});
          way["highway"="trunk_link"](around:{radius_m},{center_lat},{center_lon});
          way["highway"="primary"](around:{radius_m},{center_lat},{center_lon});
          way["highway"="primary_link"](around:{radius_m},{center_lat},{center_lon});
          way["highway"="secondary"](around:{radius_m},{center_lat},{center_lon});
          way["highway"="secondary_link"](around:{radius_m},{center_lat},{center_lon});
          way["highway"="tertiary"](around:{radius_m},{center_lat},{center_lon});
          way["highway"="tertiary_link"](around:{radius_m},{center_lat},{center_lon});
        );
        out body;
        >;
        out skel qt;
        """.strip()

        user_agent = "WildFrame/1.0 (wildfire-risk-assessment; contact@wildframe.example)"

        contexts_attempted: list = []

        for endpoint in _OSM_API_ENDPOINTS:
            ep_url = endpoint["url"]
            ep_type = endpoint["type"]
            ep_name = ep_url.split("/")[2]

            try:
                post_data: Optional[bytes] = None

                if ep_type == "overpass":
                    # Overpass API — POST the query as form data. Overpass's
                    # own docs recommend POST for anything beyond trivial
                    # queries; putting the query on the URL (GET) risks
                    # length limits and WAF/proxy rejections (we were seeing
                    # 406s from this).
                    post_data = urllib.parse.urlencode({"data": query}).encode("utf-8")
                    url = ep_url
                else:
                    # Main OSM API — use bbox query. This endpoint returns
                    # ALL data in the bbox (buildings, POIs, everything —
                    # not just roads) and hard-caps at 50,000 nodes, so we
                    # deliberately use a smaller radius here than the caller
                    # asked for to avoid tripping that limit (this is a
                    # last-resort fallback, not an equivalent replacement
                    # for Overpass).
                    osmapi_radius_m = min(radius_m, 1500)
                    lat_rad = math.radians(center_lat)
                    dlat = osmapi_radius_m / 6_371_000.0  # radians
                    dlon = osmapi_radius_m / (6_371_000.0 * math.cos(lat_rad))
                    bbox = (
                        center_lon - math.degrees(dlon),
                        center_lat - math.degrees(dlat),
                        center_lon + math.degrees(dlon),
                        center_lat + math.degrees(dlat),
                    )
                    url = f"{ep_url}?bbox={bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"

                headers = {"User-Agent": user_agent}
                if ep_type == "osmapi":
                    headers["Accept"] = "application/xml"
                else:
                    headers["Accept"] = "application/json"
                req = urllib.request.Request(url, data=post_data, headers=headers)

                with urllib.request.urlopen(req, timeout=15) as resp:
                    raw = resp.read()

                if ep_type == "overpass":
                    # Parse JSON response from Overpass
                    data = json.loads(raw.decode("utf-8"))

                    # Overpass reports rate-limiting and query errors IN-BAND
                    # via a "remark" field, still with HTTP 200 — it does not
                    # always use a 429 status. Treat this the same as a hard
                    # failure so we fall through to the next endpoint instead
                    # of caching a false "zero roads here" result forever.
                    remark = data.get("remark")
                    if remark:
                        raise ValueError(f"Overpass remark: {remark}")

                    nodes: dict[int, tuple[float, float]] = {}
                    for element in data.get("elements", []):
                        if element["type"] == "node":
                            nodes[element["id"]] = (element["lat"], element["lon"])

                    result: list[list[tuple[float, float]]] = []
                    for element in data.get("elements", []):
                        if element["type"] == "way" and "nodes" in element:
                            seg = []
                            for node_id in element["nodes"]:
                                if node_id in nodes:
                                    seg.append(nodes[node_id])
                            if len(seg) >= 2:
                                result.append(seg)
                else:
                    # Parse XML response from main OSM API
                    result = _parse_osm_xml_roads(raw)

                # Never cache an empty result: it can mean the API degraded
                # (or the area genuinely has no major roads), and freezing
                # that into the cache would mask future fixes.
                if not result:
                    print(f"[road-cache] EMPTY result from {ep_name} for {cache_key} — not cached")
                    return []

                if ep_type == "osmapi":
                    # Degraded fallback: it covered a smaller area than the
                    # caller asked for, so DON'T cache under the requested
                    # key (that would poison the full-radius cache entry).
                    # Cache under a scoped 1.5 km key instead.
                    scoped_key = f"{round(center_lat, 3)},{round(center_lon, 3)},1.5"
                    db.osm_set(scoped_key, result, time.time())
                    print(f"[road-cache] STORED {scoped_key} ({len(result)} segments, endpoint={ep_name}) — scoped fallback")
                    return result

                # Full-radius success (Overpass mirror) — cache under the
                # requested key.
                db.osm_set(cache_key, result, time.time())
                print(f"[road-cache] STORED {cache_key} ({len(result)} segments, endpoint={ep_name})")
                return result

            except urllib.error.HTTPError as e:
                ctx = f"{ep_name}: HTTP {e.code}"
                contexts_attempted.append(ctx)
                print(f"[road-cache] {ctx} for {cache_key}")
                # A 429 means THIS endpoint is rate-limiting us, not that
                # every mirror is — move on and try the remaining endpoints
                # instead of abandoning the whole chain.
                continue

            except urllib.error.URLError as e:
                ctx = f"{ep_name}: {e.reason}"
                contexts_attempted.append(ctx)
                print(f"[road-cache] {ctx} for {cache_key}")
                continue

            except (OSError, ValueError, json.JSONDecodeError) as e:
                ctx = f"{ep_name}: {e}"
                contexts_attempted.append(ctx)
                print(f"[road-cache] {ctx} for {cache_key}")
                continue

        # ---- All endpoints exhausted ----
        # Remember the failure so repeat calls short-circuit during the
        # cooldown window instead of hammering mirrors that are down.
        _osm_failed_at[cache_key] = time.time()
        ctx = " (" + "; ".join(contexts_attempted) + ")" if contexts_attempted else ""
        entry = db.osm_get(cache_key)
        if entry and entry.get("segments"):
            age_s = time.time() - entry.get("stored_at", 0)
            print(f"[road-cache]{ctx} All endpoints failed — serving stale data "
                  f"({len(entry['segments'])} segments, {age_s:.0f}s old)")
            return entry["segments"]
        raise Exception("All OSM API endpoints failed.")
@app.route("/api/bayesian/road-risk", methods=["POST"])
def bayesian_road_risk():
    """
    Compute fire spread risk to roads near active fire grids.

    Uses the head/back/flank ellipse as a continuous function (the closed-form
    effective spread rate) to estimate time-to-arrival for each road segment.

    JSON body:
      {
        "grid_id": "all" (default) | "grid-1",   // which fire to assess
        "contour_level": 0.3,                      // fire edge probability
        "radius_km": 5.0,                           // search radius for roads
        "road_segments": [[[lat, lon], ...], ...]  // optional: pre-fetched
      }

    If ``road_segments`` is omitted, the endpoint fetches roads from
    OpenStreetMap Overpass API within ``radius_km`` of each grid's
    contour centroid.

    Returns a GeoJSON FeatureCollection where each feature is a road segment
    with risk-tier properties: ``risk_tier`` (critical/high/moderate/low),
    ``t_arrival_min``, ``nearest_distance_m``, and the ellipse rates.
    """
    data = request.get_json(silent=True) or {}
    contour_level = data.get("contour_level", 0.3)
    radius_km = data.get("radius_km", 5.0)
    target_grid = data.get("grid_id", "all")
    pre_fetched = data.get("road_segments")
    demo = data.get("mode") == "demo"
    # Restrict road-risk assessment to fires in the current viewport (same
    # bbox the frontend sends for the Bayesian state) so a global FIRMS
    # dataset doesn't trigger hundreds of OSM fetches per poll.
    bbox = _parse_bbox_param(data.get("bbox"))
    mode = "demo" if demo else "production"

    if db.count_grids(mode) == 0:
        return jsonify({"error": "No active fire grids. Seed some data first or enable the Bayesian overlay."}), 400

    # Restrict to fires in the current viewport at the SQL level (PostGIS),
    # loaded in ONE batch query (the per-grid N+1 used to cost N round trips).
    _meta_rows = db.list_grid_meta(mode, bbox=bbox)
    _viewport_entries = db.get_grid_entries_batch(mode, [r["id"] for r in _meta_rows])
    viewport_items = [
        (r["id"], _viewport_entries[r["id"]])
        for r in _meta_rows
        if r["id"] in _viewport_entries
    ]
    if target_grid != "all":
        viewport_items = [(k, v) for k, v in viewport_items if k == target_grid]
    if not viewport_items:
        return jsonify({
            "type": "FeatureCollection",
            "features": [],
            "metadata": {
                "grids_assessed": 0,
                "contour_level": contour_level,
                "radius_km": radius_km,
                "grids_without_contour": 0,
                "grids_without_roads": 0,
                "empty_reason": "no active fires in the current viewport",
            },
        })

    all_features: list[dict] = []
    grids_without_contour = 0   # fire hasn't crossed contour_level yet
    grids_without_roads = 0     # established fire edge, but 0 roads nearby
    grids_fetch_failed = 0      # grids whose OSM fetch failed (all endpoints down)
    first_fetch_error: Optional[str] = None
    max_peak_probability = 0.0  # highest peak prob among no-contour grids —
                                 # helps distinguish "close" from "nowhere near"

    # Batch-load forecast series for the viewport's unique ~55 km cells in
    # ONE query (same pattern as get_grid_entries_batch — no per-grid N+1).
    # The series is populated as a side-effect of the wind refresh; cells
    # missing it self-heal via weather.get_forecast_series (one fetch each).
    _cell_centroids: dict[str, tuple[float, float]] = {}
    for _r in _meta_rows:
        _cell_centroids.setdefault(
            weather._cell_key(_r["centroid_lat"], _r["centroid_lon"]),
            (_r["centroid_lat"], _r["centroid_lon"]),
        )
    _fc_hits = db.kv_get_many([weather._fc_kv_key(c) for c in _cell_centroids])
    _forecast_by_cell: dict[str, Optional[list]] = {}
    for _cell, (_lat, _lon) in _cell_centroids.items():
        _cached = (_fc_hits.get(weather._fc_kv_key(_cell)) or {}).get("hours")
        _forecast_by_cell[_cell] = _cached or weather.get_forecast_series(_lat, _lon)
    _forecast_used_count = 0

    for key, entry in viewport_items:
        grid = entry["grid"]
        wind_speed = entry.get("wind_speed", 3.0)
        wind_dir = entry.get("wind_dir_deg", 270.0)
        # EFFIS fuel moisture scales the spread ellipse (0 = no data /
        # outside coverage → neutral 1.0).
        ffmc = float(entry.get("ffmc") or 0.0)
        dmc = float(entry.get("dmc") or 0.0)
        moisture_factor = effis_fwi.moisture_factor(ffmc) if ffmc > 0 else 1.0

        # Get contour to find where the fire edge is
        contour = grid.export_contour(level=contour_level)

        # Flatten contour to find centroid for road fetch
        all_cpts: list[tuple[float, float]] = []
        for seg in contour:
            for pt in seg:
                all_cpts.append((pt[0], pt[1]))

        if not all_cpts or not contour:
            # No established fire edge yet — skip this grid. Track the peak
            # probability so callers can tell "just seeded, still building
            # evidence" apart from something actually being wrong.
            grids_without_contour += 1
            peak = float(grid.probabilities.max()) if grid.probabilities.size else 0.0
            max_peak_probability = max(max_peak_probability, peak)
            continue

        # Get road segments
        if pre_fetched:
            segments = pre_fetched
        else:
            # Find contour centroid to fetch roads nearby
            clat = sum(p[0] for p in all_cpts) / len(all_cpts)
            clon = sum(p[1] for p in all_cpts) / len(all_cpts)
            try:
                segments = _fetch_osm_roads(clat, clon, radius_km)
            except Exception as exc:
                # One grid's fetch failing (Overpass mirrors down/rate-limited)
                # shouldn't 502 the whole overlay — log it, skip this grid,
                # assess the rest. Only a total failure (every grid failed)
                # returns an error below.
                grids_fetch_failed += 1
                if first_fetch_error is None:
                    first_fetch_error = str(exc)
                print(f"[road-risk] {key}: OSM fetch failed ({exc}) — skipping grid")
                continue

        # Compute risk for this grid's roads, with the wind-at-arrival
        # forecast correction when the cell has an hourly series (None →
        # current-wind-only, exactly the pre-forecast behavior).
        _cell = weather._cell_key(
            entry.get("centroid_lat") or grid.center_lat,
            entry.get("centroid_lon") or grid.center_lon,
        )
        forecast_series = _forecast_by_cell.get(_cell)
        if forecast_series:
            _forecast_used_count += 1
        risk_results = compute_road_risk(
            grid, segments, wind_speed, wind_dir,
            contour_level=contour_level, contour=contour,
            moisture_factor=moisture_factor,
            forecast_series=forecast_series,
        )

        if not risk_results:
            grids_without_roads += 1

        # Build GeoJSON features
        for result in risk_results:
            seg = result["segment"]
            if not seg:
                continue
            # GeoJSON uses [lon, lat] order
            coords = [[lon, lat] for lat, lon in seg]
            feature: dict = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords,
                },
                "properties": {
                    "risk_tier": result["risk_tier"],
                    "t_arrival_min": result.get("t_arrival_min"),
                    "nearest_distance_m": result.get("nearest_distance_m"),
                    "grid_id": key,
                    # EFFIS fuel-moisture context (0 = unavailable) and the
                    # resulting spread multiplier — so a "critical" road can
                    # be traced back to WHY it's critical.
                    "ffmc": ffmc,
                    "dmc": dmc,
                    "moisture_factor": moisture_factor,
                },
            }
            # Add optional detail fields for tooltip
            if result.get("head_rate_m_min") is not None:
                feature["properties"]["head_rate_m_min"] = result["head_rate_m_min"]
                feature["properties"]["back_rate_m_min"] = result["back_rate_m_min"]
                feature["properties"]["flank_rate_m_min"] = result["flank_rate_m_min"]
                feature["properties"]["effective_spread_rate_m_min"] = result.get("effective_spread_rate_m_min")
                feature["properties"]["bearing_from_wind_deg"] = result.get("bearing_from_wind_deg")
                feature["properties"]["probability_at_contour"] = result.get("probability_at_contour")
                feature["properties"]["nearest_contour_point"] = result.get("nearest_contour_point")
            all_features.append(feature)

    # Build a human-readable reason when we're returning no features, so the
    # frontend doesn't have to guess between "fire not established yet",
    # "no roads nearby", and (implicitly) "fetch succeeded but was empty".
    empty_reason = None
    # 502 ONLY when every grid with an established contour failed its fetch
    # (all mirrors down) — a genuine outage. Grids skipped for having no
    # contour yet, or that succeeded and simply found no roads, are NOT
    # failures, so a partial/legitimately-empty result isn't misreported.
    contoured = len(viewport_items) - grids_without_contour
    if (not all_features and grids_fetch_failed > 0
            and grids_fetch_failed == contoured):
        return jsonify({
            "error": f"Failed to fetch road data from OpenStreetMap: {first_fetch_error}",
        }), 502
    if not all_features:
        if grids_without_contour and not grids_without_roads:
            empty_reason = (
                f"no established fire edge yet at {contour_level} probability "
                f"(peak so far: {max_peak_probability:.2f}) — needs more "
                f"corroborating evidence"
            )
        elif grids_without_roads and not grids_without_contour:
            empty_reason = "fire edge established, but no major roads found within radius"
        elif grids_without_contour and grids_without_roads:
            empty_reason = "some fires not yet established; others have no roads nearby"

    return jsonify({
        "type": "FeatureCollection",
        "features": all_features,
        "metadata": {
            "grids_assessed": len(viewport_items),
            "contour_level": contour_level,
            "radius_km": radius_km,
            "grids_without_contour": grids_without_contour,
            "grids_without_roads": grids_without_roads,
            "grids_fetch_failed": grids_fetch_failed,
            "forecast_wind_grids": _forecast_used_count,
            "empty_reason": empty_reason,
        },
    })


AGENCY_API_KEY_HEADER = "X-Agency-Key"
_warned_missing_agency_key = False


def _agency_api_key() -> str:
    """Shared secret for the agency ingest endpoints.

    Resolution: env ``WILDFRAME_AGENCY_API_KEY`` (from .env) first, then the
    ``kv_store`` key ``agency_api_key``. ENV WINS — if the env var is set,
    the kv_store value is ignored (so a runtime rotation via kv_store only
    takes effect when env is unset; set the env var for a fixed key). Read
    per-request so kv rotation is otherwise instant.
    """
    key = os.environ.get("WILDFRAME_AGENCY_API_KEY") or ""
    if not key:
        stored = db.kv_get("agency_api_key") or {}
        key = stored.get("key") or ""
    return key


def _require_agency_key():
    """Enforce the shared secret on agency ingest endpoints.

    Fail-open when no key is configured (local dev) — but log a one-time
    warning so exposing the server publicly without a key is a visible
    mistake. Once a key is set (env or kv_store), every request must send it
    in the ``X-Agency-Key`` header; comparison is constant-time.
    """
    global _warned_missing_agency_key
    expected = _agency_api_key()
    if not expected:
        if not _warned_missing_agency_key:
            _warned_missing_agency_key = True
            logger.warning(
                "Agency ingest is OPEN (no WILDFRAME_AGENCY_API_KEY set) — "
                "set one in .env or kv_store before exposing the server publicly."
            )
        return None
    provided = request.headers.get(AGENCY_API_KEY_HEADER, "")
    if not provided or not hmac.compare_digest(provided, expected):
        logger.warning(
            "Agency ingest rejected (bad/missing X-Agency-Key) from %s — "
            "possible probe or misconfigured client.",
            request.remote_addr or "unknown",
        )
        return jsonify({"error": "invalid or missing X-Agency-Key"}), 401
    return None


@app.route("/api/agencies/ingest", methods=["POST"])
def agencies_ingest():
    """Ingest one normalized agency incident (CAP / GeoJSON / GeoRSS adapter
    output). This is the front door for the government-feed pipeline: a
    Lambda (push) or a poller job (pull) calls this with a canonical
    incident dict and gets idempotent, staleness-guarded storage plus
    Bayesian grid evidence fusion in one shot.

    Auth: when ``WILDFRAME_AGENCY_API_KEY`` (or kv_store ``agency_api_key``)
    is configured, the request must carry the matching ``X-Agency-Key``
    header; otherwise the endpoint fails open with a warning (local dev).

    Expected body (all adapter outputs must converge on this shape):

    .. code-block:: json

        {
          "agency": "gov-cap:meteoalarm",   // CAP sender, or namespaced feed name
          "incident_id": "cap-2026-0810-004",  // CAP identifier / feed item id
          "action": "create",               // create | update | cancel | delete
          "sent_at": "2026-08-10T09:30:00Z", // staleness clock — only newer wins
          "lat": 51.106,
          "lon": 18.941,
          "status": "confirmed",            // optional; create defaults confirmed
          "source_type": "government",
          "severity": "Extreme",            // optional, stored in data blob
          "data": {}                         // optional extra metadata
        }

    Returns the authoritative stored report (newest version wins) plus
    ``created``/``stale`` flags and the grid mutation outcome.

    Grid dispatch:
      - create/update (confirmed) → find-or-create grid at the point and fuse
        strong positive evidence (Evidence.agency_confirm).
      - cancel/delete → mark the report ``cancelled``, find the NEAREST
        EXISTING grid (never create one) and fuse negative evidence
        (Evidence.agency_cancel), decaying the fire rather than deleting it.
    """
    auth_error = _require_agency_key()
    if auth_error is not None:
        return auth_error

    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "production")
    if mode not in ("production", "demo"):
        return jsonify({"error": "invalid mode"}), 400

    agency = data.get("agency")
    incident_id = data.get("incident_id")
    action = data.get("action", "create")
    if not agency or not incident_id:
        return jsonify({"error": "agency and incident_id are required"}), 400
    if action not in ("create", "update", "cancel", "delete"):
        return jsonify({"error": f"invalid action: {action}"}), 400
    if data.get("lat") is None or data.get("lon") is None:
        return jsonify({"error": "lat and lon are required"}), 400

    # Normalize into the canonical report shape the rest of the system uses.
    incident = dict(data)
    incident.setdefault("id", uuid.uuid4().hex)
    incident.setdefault("source_type", "government")
    incident.setdefault("sent_at", datetime.now(timezone.utc).isoformat())
    incident.setdefault("captured_at", incident["sent_at"])
    if action == "cancel" or action == "delete":
        incident["status"] = "cancelled"
    else:
        # An explicit status (e.g. "pending" from an adapter) is respected;
        # anything falsy defaults to the high-trust confirmed lane.
        incident["status"] = incident.get("status") or "confirmed"
    # Keep the full normalized envelope (including CAP fields like severity)
    # inside the round-trip data blob so the UI/admin can render it.
    incident["data"] = incident.get("data") or {}
    incident["data"]["action"] = action
    incident["data"]["agency"] = agency
    incident["data"]["incident_id"] = incident_id

    stored, created, applied = db.upsert_agency_incident(incident, mode)

    # NOTE: ``sent_at`` must live inside the data blob (upsert_agency_incident
    # stores the full incident dict), because _row_to_report rebuilds the
    # report from that blob — that's what makes this comparison meaningful.
    stale = (not created) and stored.get("sent_at") != incident.get("sent_at")
    duplicate = (not created) and (not applied) and (not stale)

    grid_result = {"fused": False, "grid_id": None}
    demo = mode == "demo"
    # Only mutate grid evidence when this message actually changed the row
    # (applied). A stale (guard-rejected) or duplicate (same sent_at retry)
    # message must NOT touch the grid: the row is unchanged, so fusing would
    # double-count evidence (e.g. a redelivered create) or contradict the
    # stored state (e.g. a stale cancel degrading a confirmed fire).
    took_effect = applied
    if took_effect and stored.get("status") == "cancelled":
        # Never create a grid for a cancel — only fuse into an existing one.
        grid_id = db.find_grid_near(mode, float(data["lat"]), float(data["lon"]))
        if grid_id:
            db.mutate_grid(mode, grid_id, lambda g, _e: g.update(
                Evidence.agency_cancel(float(data["lat"]), float(data["lon"]))
            ))
            grid_result = {"fused": True, "grid_id": grid_id}
    elif took_effect and stored.get("status") == "confirmed":
        grid_id, _grid = _find_or_create_grid_for_point(
            float(data["lat"]), float(data["lon"]), demo=demo,
        )
        db.mutate_grid(mode, grid_id, lambda g, _e: g.update(
            Evidence.agency_confirm(float(data["lat"]), float(data["lon"]))
        ))
        grid_result = {"fused": True, "grid_id": grid_id}

    return jsonify({
        "report": stored,
        "action": action,
        "created": created,
        "stale": stale,
        "duplicate": duplicate,
        "grid": grid_result,
    })


# ---------------------------------------------------------------------------
# Fire Spread Simulation
# ---------------------------------------------------------------------------
# On-demand forecast: user clicks "Play" → compute N future timesteps
# (default 6 × 15 min = 90 min) and return heatmap, road risk, and
# containment probability at each frame.  The client animates through
# the frames.  No storage — computed fresh per request.

def _containment_estimate(
    grid: "BayesianFireGrid",
    wind_speed: float,
    wind_dir_deg: float,
    moisture_factor: float,
    steps_done: int,
    total_steps: int,
) -> float:
    """Heuristic containment probability (0–1) for a fire.

    Starts at 0.1 (fire just detected) and increases as the simulation
    progresses — a fire that hasn't spread much after many timesteps is
    more likely to be containable.  Wind and moisture modulate the rate:
    high wind makes containment harder (slower climb); low moisture
    makes it easier (faster climb).  This is a *rough* heuristic for
    the public-facing simulation — not a physics-based containment model.
    """
    # Base: linear climb from 0.1 to 0.8 over the full simulation
    base = 0.1 + 0.7 * (steps_done / max(total_steps, 1))
    # Wind penalty: >10 m/s pushes containment down
    wind_penalty = max(0.0, (wind_speed - 10.0) * 0.015)
    # Moisture bonus: drier fuel (mf < 1) = harder to contain
    moisture_bonus = max(0.0, (1.0 - moisture_factor) * 0.1)
    return max(0.05, min(0.95, base - wind_penalty + moisture_bonus))


@app.route("/api/simulate/<grid_id>", methods=["POST"])
def simulate_fire(grid_id: str):
    """
    Compute a multi-step fire spread forecast for one grid.

    The grid is cloned (not mutated) and ``predict()`` is called N times
    with 15-min timesteps.  Each frame returns:
      - sparse heatmap cells (probability > 0.03)
      - contour polygons at 0.3 and 0.6 levels
      - road risk assessment (critical/high/moderate/low)
      - containment probability estimate

    JSON body (all optional):
      {
        "steps": 24,            // number of 15-min timesteps (1-24 = 6 h max)
        "contour_level": 0.3,   // primary contour level
        "radius_km": 5.0        // road search radius
      }

    Returns:
      {
        "grid_id": "grid-123",
        "frames": [...],
        "metadata": {wind, cell_size, center, ...}
      }
    """
    from bayesian_filter import BayesianFireGrid
    import copy

    data = request.get_json(silent=True) or {}
    steps = max(1, min(int(data.get("steps", 24)), 24))  # 6 h max (24 × 15 min)
    contour_level = float(data.get("contour_level", 0.3))
    radius_km = float(data.get("radius_km", 5.0))
    demo = data.get("mode") == "demo"
    mode = "demo" if demo else "production"

    # Load the grid entry (read-only — we clone before mutating)
    entry = db.get_grid_entry(mode, grid_id)
    if entry is None:
        return jsonify({"error": f"Grid '{grid_id}' not found"}), 404

    grid: BayesianFireGrid = entry["grid"]
    wind_speed = float(entry.get("wind_speed", 3.0))
    wind_dir_deg = float(entry.get("wind_dir_deg", 270.0))
    ffmc = float(entry.get("ffmc") or 0.0)
    dmc = float(entry.get("dmc") or 0.0)
    mf = effis_fwi.moisture_factor(ffmc) if ffmc > 0 else 1.0

    # Road segments: cache-only lookup (no live Overpass fetch).
    # The road-risk endpoint populates the cache; if the user hasn't
    # viewed road risk for this fire yet, roads are simply empty —
    # the heatmap + containment still render correctly.
    contour = grid.export_contour(level=contour_level)
    all_cpts = [pt for seg in contour for pt in seg]
    road_segments = []
    if all_cpts:
        clat = sum(p[0] for p in all_cpts) / len(all_cpts)
        clon = sum(p[1] for p in all_cpts) / len(all_cpts)
        cache_key = _osm_cache_key(clat, clon, radius_km)
        cached = db.osm_get(cache_key)
        if cached and cached.get("segments"):
            road_segments = cached["segments"]
        else:
            # Fuzzy lookup: same radius, centroid within 1.5 km
            for k, v in db.osm_iter():
                parsed_k = k.split(",")
                if len(parsed_k) == 3:
                    try:
                        ck_lat, ck_lon, ck_rad = float(parsed_k[0]), float(parsed_k[1]), float(parsed_k[2])
                    except ValueError:
                        continue
                    if abs(ck_rad - radius_km) > 0.01:
                        continue
                    d = _haversine(clat, clon, ck_lat, ck_lon)
                    if d < 1500.0 and v.get("segments"):
                        road_segments = v["segments"]
                        break
        if road_segments:
            print(f"[simulate] road cache hit for {grid_id}: {len(road_segments)} segments")

    # Clone the grid so we never mutate the live state
    grid_copy = BayesianFireGrid.from_dict(grid.to_dict())
    dt_s = 900.0  # 15 minutes per step

    frames = []
    for step in range(1, steps + 1):
        # Advance one timestep
        grid_copy.predict(
            dt=dt_s,
            wind_speed=wind_speed,
            wind_dir_deg=wind_dir_deg,
            moisture_factor=mf,
            slope_lookup=db.dem_terrain_lookup,
        )
        grid_copy._compute_probs()

        # Sparse heatmap (threshold = 0.03 to keep payload small)
        state = grid_copy.export_state(threshold=0.03)

        # Perimeter contour: use a very low level (0.05) so the line
        # wraps the OUTER boundary of the predicted spread area — not
        # deep inside the fire body.  The 0.3 contour would sit inside
        # because most cells are >0.9 probability.
        c_perimeter = grid_copy.export_contour(level=0.05)
        c_hot = grid_copy.export_contour(level=0.6)

        # Road risk at this timestep
        road_risk = []
        if road_segments:
            try:
                risk = compute_road_risk(
                    grid_copy, road_segments, wind_speed, wind_dir_deg,
                    contour_level=0.05, contour=c_perimeter,
                    moisture_factor=mf,
                )
                # Only return non-low roads to keep payload small
                road_risk = [r for r in risk if r.get("risk_tier") != "low"]
            except Exception:
                pass

        # Containment estimate
        containment = _containment_estimate(
            grid_copy, wind_speed, wind_dir_deg, mf, step, steps,
        )

        t_min = step * 15
        frames.append({
            "step": step,
            "t_label": f"+{t_min} min",
            "t_min": t_min,
            "cells": state["cells"],
            "cell_count": state["count"],
            "contour_perimeter": c_perimeter,
            "contour_hot": c_hot,
            "road_risk": road_risk,
            "containment": round(containment, 3),
            "max_p": float(grid_copy.probabilities.max()),
        })

    return jsonify({
        "grid_id": grid_id,
        "frames": frames,
        "metadata": {
            "wind_speed": wind_speed,
            "wind_dir_deg": wind_dir_deg,
            "moisture_factor": round(mf, 3),
            "cell_size_m": grid.cell_size,
            "center_lat": grid.center_lat,
            "center_lon": grid.center_lon,
            "steps": steps,
            "step_duration_min": 15,
            "total_duration_min": steps * 15,
            "road_segments_count": len(road_segments),
        },
    })


@app.route("/api/bayesian/predict", methods=["POST"])
def bayesian_predict():
    """
    Manually trigger a predict step with custom weather parameters.

    JSON body:
      {
        "dt": 600,           // time step in seconds
        "wind_speed": 5.0,   // m/s
        "wind_dir_deg": 270, // compass degrees
        "slope_pct": 0.0,    // percent
        "grid_id": "grid-1"  // optional: target a specific grid; omit for all
      }

    When a single grid is targeted, its per-grid wind parameters are also
    updated so subsequent auto-predict polls use the new wind (Gap 3 fix).
    When omitted, all grids are advanced with the given wind but per-grid
    stored wind is NOT overwritten (so auto-predict keeps using its own).
    """
    data = request.get_json(silent=True) or {}
    dt = data.get("dt", 600.0)
    wind_speed = data.get("wind_speed", 5.0)
    wind_dir = data.get("wind_dir_deg", 270.0)
    slope_pct = data.get("slope_pct", 0.0)
    grid_id = data.get("grid_id")
    demo = data.get("mode") == "demo"

    mode = "demo" if demo else "production"

    if grid_id:
        if db.get_grid_entry(mode, grid_id) is None:
            return jsonify({"error": f"Grid '{grid_id}' not found"}), 404

        def _predict_single(grid: BayesianFireGrid, entry: dict) -> None:
            grid.predict(
                dt=dt, wind_speed=wind_speed, wind_dir_deg=wind_dir, slope_pct=slope_pct,
                slope_lookup=db.dem_terrain_lookup,
            )
            # Persist the new wind for this grid (so auto-predict inherits it)
            entry["wind_speed"] = wind_speed
            entry["wind_dir_deg"] = wind_dir

        db.mutate_grid(mode, grid_id, _predict_single)
        message = f"Predicted {dt:.0f}s ahead for grid '{grid_id}' (wind now {wind_speed} m/s from {wind_dir}°)."
    else:
        ids = [row["id"] for row in db.list_grid_meta(mode)]
        for gid in ids:
            db.mutate_grid(
                mode, gid,
                lambda g, e: g.predict(
                    dt=dt, wind_speed=wind_speed, wind_dir_deg=wind_dir, slope_pct=slope_pct,
                    slope_lookup=db.dem_terrain_lookup,
                ),
            )
        message = f"Predicted {dt:.0f}s ahead for all {len(ids)} grid(s). Per-grid wind NOT overwritten."

    return jsonify({"status": "ok", "message": message})


# ---------------------------------------------------------------------------
# Geocoding — search box (city / address lookup)
# ---------------------------------------------------------------------------

_GEOCODE_LOCK = threading.Lock()
_GEOCODE_LAST_TS = 0.0
_GEOCODE_MIN_INTERVAL_S = 1.2  # Nominatim usage policy: max 1 request/second
_GEOCODE_UA = "WildFrame-Pyrae/1.0 (https://pyrae.co)"


@app.route("/api/geocode", methods=["GET"])
def api_geocode():
    """Search for a place (city, address, …) via OpenStreetMap Nominatim.

    Proxied server-side so we can (a) send a proper identifying User-Agent
    as Nominatim's usage policy requires, (b) throttle to <=1 req/s, and
    (c) cache normalized queries in Postgres for 30 days, so the vast
    majority of searches never touch Nominatim at all.

    Query params: ``q`` (>= 2 chars), ``limit`` (default 5, max 8).

    Returns ``{"results": [...]}`` where each result has ``lat``, ``lon``,
    ``label``, ``name``, ``type``, ``class`` and ``importance``. If the
    upstream geocoder is unreachable we return ``{"results": [],
    "degraded": true}`` so the UI can show a quiet "unavailable" row
    instead of an error page.
    """
    q = (request.args.get("q") or "").strip()
    if len(q) < 2:
        return jsonify({"results": [], "error": "query_too_short"}), 400
    try:
        limit = max(1, min(int(request.args.get("limit", 5)), 8))
    except (TypeError, ValueError):
        limit = 5

    key = " ".join(q.lower().split())

    # Cache is best-effort: if Postgres is down we still want to serve
    # results straight from Nominatim rather than 500.
    try:
        cached = db.geocode_get(key)
    except Exception:
        cached = None
    if cached is not None:
        return jsonify({"results": cached})

    import urllib.request
    import urllib.error
    import urllib.parse

    global _GEOCODE_LAST_TS
    with _GEOCODE_LOCK:
        wait = _GEOCODE_MIN_INTERVAL_S - (time.time() - _GEOCODE_LAST_TS)
        if wait > 0:
            time.sleep(wait)
        try:
            url = (
                "https://nominatim.openstreetmap.org/search?"
                + urllib.parse.urlencode(
                    {
                        "q": q,
                        "format": "jsonv2",
                        "limit": limit,
                        "accept-language": "en",
                    }
                )
            )
            req = urllib.request.Request(url, headers={"User-Agent": _GEOCODE_UA})
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
            # Geocoder unavailable — degrade gracefully.
            return jsonify({"results": [], "degraded": True})
        finally:
            _GEOCODE_LAST_TS = time.time()

    results = []
    for r in raw:
        try:
            lat = float(r["lat"])
            lon = float(r["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        results.append(
            {
                "lat": lat,
                "lon": lon,
                "label": r.get("display_name", ""),
                "name": r.get("name", ""),
                "type": r.get("addresstype") or r.get("type") or "place",
                "class": r.get("class") or r.get("addresstype") or "place",
                "importance": r.get("importance") or 0.0,
            }
        )

    try:
        db.geocode_set(key, results)
    except Exception:
        pass  # caching is best-effort
    return jsonify({"results": results})


@app.route("/api/bayesian/update", methods=["POST"])
def bayesian_update():
    """
    Inject an evidence observation into the grid manually.

    JSON body:
      {
        "lat": 37.727,
        "lon": -119.637,
        "log_lr": 2.3,        // ln(likelihood ratio)
        "spatial_radius_m": 0,
        "source": "photo-flame"
      }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Missing request body"}), 400

    lat = data.get("lat")
    lon = data.get("lon")
    log_lr = data.get("log_lr", math.log(10.0))

    if lat is None or lon is None:
        return jsonify({"error": "Missing 'lat' and 'lon'"}), 400

    demo = data.get("mode") == "demo"

    evidence = Evidence(
        lat=lat,
        lon=lon,
        log_likelihood_ratio=log_lr,
        spatial_radius_m=data.get("spatial_radius_m", 0.0),
        source=data.get("source", "api"),
    )

    grid_id, _grid = _find_or_create_grid_for_point(lat, lon, demo=demo)
    mode = "demo" if demo else "production"
    db.mutate_grid(mode, grid_id, lambda g, e: g.update(evidence))

    return jsonify({"status": "ok", "evidence": data, "message": "Evidence fused."})


@app.route("/api/bayesian/reset", methods=["POST"])
def bayesian_reset():
    """Reset Bayesian grids (clears the whole registry for the requested
    mode; grids are rebuilt on demand from confirmed reports)."""
    demo = _request_mode() == "demo"
    mode = "demo" if demo else "production"
    n = db.delete_grids(mode)
    return jsonify({"status": "ok", "message": f"Grid(s) reset ({n} removed).", "mode": mode})


# ---------------------------------------------------------------------------
# Satellite Hotspot Simulation — bridges Gap 1 (satellite path for live grids)
# ---------------------------------------------------------------------------

# Poller state (active flags + parameters) now lives in the kv_store table
# (db.py) and the periodic work runs in the Procrastinate job queue
# (jobs.py) — no process-local threads, so it survives restarts and works
# correctly with multiple web workers.


def _simulate_satellite_pass(
    probability: float = 0.6,
    min_hotspots: int = 1,
    max_hotspots: int = 4,
    jitter_km: float = 0.5,
    guarantee_hit: bool = False,
) -> dict:
    """
    Simulate a satellite (VIIRS/FIRMS) pass over all active Bayesian grids.

    For each grid, with *probability*, drop 1–4 satellite hotspot evidence
    observations near the grid's tracked centroid.  This gives the live
    citizen-report pipeline the satellite-corroboration story that previously
    only existed in the isolated Creek Fire demo.

    Parameters
    ----------
    guarantee_hit : bool
        If True and at least one grid exists, force at least one grid to
        receive hotspots even if every per-grid coin flip missed. Intended
        for a manually-triggered pass (a deliberate button click shouldn't
        silently do nothing just because of an unlucky roll — with only 1-2
        active fires and probability=0.6, an all-miss result happens ~16-40%
        of the time). The background poller should leave this False so its
        behavior stays organically probabilistic over many passes.

    Returns a summary dict with counts of grids hit and evidence injected.
    """
    # SIMULATED passes only ever touch the DEMO registry — they must never
    # inject fake evidence into production grids that drive real outputs.
    rows = db.list_grid_meta("demo")
    if not rows:
        return {"injected": 0, "grids_hit": 0, "grids_considered": 0}

    total_injected = 0
    grids_hit = 0
    grid_ids_hit = []

    def _pass(grid: BayesianFireGrid, entry: dict, chance: float) -> int:
        """Inject hotspots with probability `chance`; returns count injected."""
        if random.random() >= chance:
            return 0
        clat, clon = entry["centroid_lat"], entry["centroid_lon"]
        num = random.randint(min_hotspots, max_hotspots)
        for _ in range(num):
            jlat = clat + random.uniform(-jitter_km / 111.0, jitter_km / 111.0)
            jlon = clon + random.uniform(
                -jitter_km / (111.0 * math.cos(math.radians(clat))),
                jitter_km / (111.0 * math.cos(math.radians(clat))),
            )
            grid.update(Evidence.satellite_hotspot(lat=jlat, lon=jlon))
        return num

    all_ids = [row["id"] for row in rows]
    for grid_id in all_ids:
        # Row-locked mutation: concurrent passes (multi-worker) serialize
        # per grid instead of double-injecting or losing evidence.
        n = db.mutate_grid(
            "demo", grid_id,
            lambda g, e, _ch=probability: _pass(g, e, _ch),
        )
        n = n or 0
        if n > 0:
            total_injected += n
            grids_hit += 1
            grid_ids_hit.append(grid_id)

    if guarantee_hit and grids_hit == 0 and all_ids:
        # Every grid missed its coin flip — force one, chosen at random,
        # so a deliberate manual trigger always shows something happened.
        grid_id = random.choice(all_ids)
        n = db.mutate_grid(
            "demo", grid_id,
            lambda g, e, _ch=1.0: _pass(g, e, _ch),
        )
        n = n or 0
        total_injected += n
        grids_hit += 1
        grid_ids_hit.append(grid_id)

    return {
        "injected": total_injected,
        "grids_hit": grids_hit,
        "grids_considered": len(rows),
        "grid_ids_hit": grid_ids_hit,
    }


@app.route("/api/satellite/simulate-pass", methods=["POST"])
def satellite_simulate_pass():
    """
    Manually trigger a simulated satellite pass over all active fire grids.

    JSON body (all optional):
      {
        "probability": 0.6,      // probability any given grid gets hotspots
        "min_hotspots": 1,       // min hotspots per-hit grid
        "max_hotspots": 4,       // max hotspots per-hit grid
        "jitter_km": 0.5,        // random offset radius from centroid (km)
        "guarantee_hit": true    // force >=1 hit if any grid exists, so a
                                  // deliberate click isn't lost to an
                                  // all-miss coin flip (default true)
      }

    Returns a summary of how many grids were hit and how many evidence
    observations were injected into the Bayesian filter.
    """
    data = request.get_json(silent=True) or {}
    result = _simulate_satellite_pass(
        probability=data.get("probability", 0.6),
        min_hotspots=data.get("min_hotspots", 1),
        max_hotspots=data.get("max_hotspots", 4),
        jitter_km=data.get("jitter_km", 0.5),
        guarantee_hit=data.get("guarantee_hit", True),
    )

    if result["grids_considered"] == 0:
        message = "No active fires are being tracked yet — nothing to scan."
    else:
        message = (
            f"Satellite pass complete: {result['injected']} hotspot(s) "
            f"injected across {result['grids_hit']}/{result['grids_considered']} grid(s)."
        )

    return jsonify({
        "status": "ok",
        "message": message,
        **result,
    })


@app.route("/api/satellite/poller/start", methods=["POST"])
def satellite_poller_start():
    """
    Enable the periodic simulated-satellite job in the queue. The work runs
    in a Procrastinate worker (see jobs.py), scheduled every ~20 s; this
    endpoint just flips the flag the job checks — so it survives restarts
    and behaves correctly with multiple web workers.

    JSON body (all optional):
      {
        "interval_s": 20,      // seconds between simulated passes (hint)
        "probability": 0.6,    // pass probability per grid
        "min_hotspots": 1,
        "max_hotspots": 3
      }
    """
    data = request.get_json(silent=True) or {}
    interval = data.get("interval_s", 20.0)
    probability = data.get("probability", 0.6)
    min_hotspots = data.get("min_hotspots", 1)
    max_hotspots = data.get("max_hotspots", 3)

    db.set_poller_active("satellite", True, {
        "interval_s": interval,
        "probability": probability,
        "min_hotspots": min_hotspots,
        "max_hotspots": max_hotspots,
    })

    # The job runs on a minute-cron and self-throttles to interval_s, so
    # sub-minute intervals are effectively 60 s (see jobs.py).
    effective = max(60.0, float(interval))
    return jsonify({
        "status": "started",
        "message": f"Satellite poller scheduled (interval={effective:.0f}s, p={probability}).",
        "interval_s": interval,
        "effective_interval_s": effective,
        "probability": probability,
    })


@app.route("/api/satellite/poller/stop", methods=["POST"])
def satellite_poller_stop():
    """Disable the periodic simulated-satellite job."""
    was_active = db.get_poller_config("satellite", {})["active"]
    db.set_poller_active("satellite", False)

    return jsonify({
        "status": "stopped" if was_active else "idle",
        "message": "Satellite poller stopped." if was_active else "Poller was not running.",
    })


# ===================================================================
# NASA FIRMS Real Satellite Data Integration
# ===================================================================
# Fetches actual VIIRS / MODIS hotspot data from the NASA FIRMS API and
# feeds it into the Bayesian fire grids as real satellite evidence.

# FIRMS poller state lives in the kv_store table; the periodic work runs
# in the Procrastinate job queue (jobs.py).


# Cluster radius for grouping nearby FIRMS hotspot points into a single fire.
# VIIRS has 375m resolution — hotspots within 1km are almost certainly the
# same fire, but an active fire front spans far more than one hotspot: a
# single wildfire routinely produces dozens of detections spread over
# 1–10+ km (each overpass, each VIIRS/MODIS pixel, each scan line). The
# old 1 km radius fragmented one fire into many candidate clusters (and
# therefore many grids — the production grid count grew past 15k, each
# with its own full state blob, which is what makes /api/bayesian/state
# heavy: 100k+ cells serialized per full-detail poll). 3 km merges the
# pixels of one fire front into one grid while still separating genuinely
# distinct fires, and matches the existing satellite-confirmation radius
# (_FIRMS_CONFIRM_RADIUS_M = 3000.0).
_FIRMS_CLUSTER_RADIUS_M = float(os.environ.get("WILDFRAME_FIRMS_CLUSTER_RADIUS_M", "3000.0"))

# Hard wall-clock timeout for a FIRMS pass (seconds).  If the pass exceeds
# this, we abort and let the next periodic run retry.  Prevents a stuck
# pass from starving the worker queue indefinitely.
_PASS_TIMEOUT_S = float(os.environ.get("WILDFRAME_PASS_TIMEOUT_S", "600"))  # 10 min

# ---------------------------------------------------------------------------
# Static-source downweight (Step 2 of the filtering plan)
# ---------------------------------------------------------------------------
# A FIRMS hotspot on a persistent industrial cell (refinery flare, steel
# mill — the `static_sources` mask built by build_static_mask.py) is real
# heat but almost certainly not a wildfire. Instead of dropping it (the
# mask is derived from detection persistence, not an authoritative
# inventory, so it can misfire), inject it at reduced evidence weight:
# weight 0.1 scales the satellite LR from ln(50) ≈ 3.91 down to ln(5) ≈
# 1.61, so a static source can never push a grid toward the auto-approval
# floor on its own — it renders as a low-confidence dot, not a crimson
# blob. The same reduced-weight path is what the agricultural-burning
# downweight (Step 3) will reuse.
STATIC_SOURCE_EVIDENCE_WEIGHT = float(
    os.environ.get("WILDFRAME_STATIC_SOURCE_WEIGHT", "0.1")
)

# ---------------------------------------------------------------------------
# Agricultural-burning downweight (Step 3 of the filtering plan)
# ---------------------------------------------------------------------------
# A FIRMS detection on CORINE cropland (arable fields, orchards, vineyards)
# is real fire but frequently agricultural burning — stubble/slash burns
# that are short-lived and not the spread-relevant wildfire the Bayesian
# grid models. Instead of dropping it, inject at reduced evidence weight
# (0.3 → LR ln(50·0.3) ≈ 2.71 vs the full ln(50) ≈ 3.91): still real
# positive evidence that "something is burning here", just less than a
# full spread-relevant detection. The escape hatch promotes it back to
# full weight: a fire that survives to a later FIRMS pass (its grid
# pre-exists) is a persisting fire, not a one-off burn, so subsequent
# detections fuse at full strength.
#
# FRP cap is OFF by default: the FIRMS-archive calibration (240 daily
# files, 2024-01 → 2026-04, joined to CORINE classes) found NO bimodal
# FRP split — cropland p50 = 2.8 MW vs forest-shrub p50 = 3.2 MW, with
# ~92% of both classes below 10 MW, and day/night shows no separation
# either (41.7% vs 42.4% daytime). A hard FRP threshold would wrongly
# downweight ~92% of real forest fires. Land class is the signal; FRP is
# an optional extra cap if a deployment wants it (0/absent disables).
AG_BURN_EVIDENCE_WEIGHT = float(
    os.environ.get("WILDFRAME_AG_BURN_WEIGHT", "0.3")
)
AG_BURN_FRP_MAX_MW = float(
    os.environ.get("WILDFRAME_AG_BURN_FRP_MAX_MW", "0")  # 0 = disabled
)

# ---------------------------------------------------------------------------
# Source filter (Step 4 of the filtering plan): volcanic / conflict
# ---------------------------------------------------------------------------
# FIRMS reports *thermal anomalies*, not just wildfires. Two remaining
# non-wildfire source classes can masquerade as spread-relevant fire:
# volcanic/geothermal heat and conflict fires (settlements, strikes,
# explosions in war zones). Neither is relevant for the Poland beta, so
# this is a configurable per-source policy knob, off by default:
#
#   WILDFRAME_SOURCE_FILTER_VOLCANIC  = full | downweight | drop | passthrough
#   WILDFRAME_SOURCE_FILTER_CONFLICT  = full | downweight | drop | passthrough
#
#   full         current behavior: no tag, full evidence weight (default)
#   downweight   inject at reduced weight (volcanic: 0.1 like static —
#                persistent heat, not spread)
#   drop         exclude from injection entirely
#   passthrough  skip the source tag AND skip the land-cover gate for
#                these hotspots — the operator explicitly wants them shown
#
# Volcanic detections are cross-referenced against a volcano location table
# (Smithsonian GVP Holocene database). The import script is DEFERRED until
# geographic expansion — db.volcano_hits_batch is a fail-open stub that
# returns no hits, so the wiring is live but inert today. Conflict zones
# are operator-supplied geometry in kv_store (``conflict_zones``): a JSON
# list of polygons/bboxes, editable without redeploy.
#
# Volcanic weight matches STATIC_SOURCE_EVIDENCE_WEIGHT (0.1): "known
# persistent heat source, don't let it dominate". No escape hatch for
# volcanic — unlike ag-burn there's no bimodality; a fumarole doesn't
# "escape" into a wildfire, and if it ignites surrounding vegetation that
# new hotspot falls outside the volcano radius and gets full weight on its
# own.
SOURCE_FILTER_VOLCANIC = os.environ.get(
    "WILDFRAME_SOURCE_FILTER_VOLCANIC", "downweight"
)
SOURCE_FILTER_CONFLICT = os.environ.get(
    "WILDFRAME_SOURCE_FILTER_CONFLICT", "full"
)
VOLCANIC_EVIDENCE_WEIGHT = float(
    os.environ.get("WILDFRAME_VOLCANIC_WEIGHT", "0.1")
)
CONFLICT_EVIDENCE_WEIGHT = float(
    os.environ.get("WILDFRAME_CONFLICT_WEIGHT", "0.1")
)

# ---------------------------------------------------------------------------
# Satellite confirmation of crowdsourced reports
# ---------------------------------------------------------------------------
# A crowdsourced report is considered "satellite-confirmed" if a FIRMS
# hotspot was detected within this radius and within this time window of
# the report's captured_at timestamp. The radius is generous relative to
# VIIRS's 375m pixel size because citizen GPS (phone GPS / EXIF) is much
# noisier than satellite geolocation. The time window reflects that VIIRS/
# MODIS overpasses are not continuous — a satellite may not have passed
# over the exact moment of the report, but a hotspot within ~12h either
# side of the report is strong corroborating evidence of an active fire.
_FIRMS_CONFIRM_RADIUS_M = 3000.0
_FIRMS_CONFIRM_WINDOW_HOURS = 12.0


def _match_report_to_hotspots(
    report: dict,
    hotspots: list["nasa_firms.FIRMSHotspot"],
    radius_m: float = _FIRMS_CONFIRM_RADIUS_M,
    window_hours: float = _FIRMS_CONFIRM_WINDOW_HOURS,
) -> dict:
    """
    Check a single report against a list of FIRMS hotspots and return a
    satellite-confirmation summary.

    A hotspot "matches" if it is within `radius_m` of the report AND its
    acquisition time is within `window_hours` of the report's captured_at
    (hotspots with unparseable timestamps are matched on distance only).

    Returns
    -------
    dict with keys:
        "confirmed"    : bool
        "hotspot_count": int   — number of matching hotspots
        "nearest_km"   : float | None
        "nearest_frp"  : float | None — Fire Radiative Power (MW) of nearest match
        "checked_at"   : str (ISO timestamp)
    """
    report_time = _parse_ts(report.get("captured_at", ""))
    have_report_time = report_time > datetime.min.replace(tzinfo=timezone.utc)

    matches = []
    for hs in hotspots:
        d_m = _haversine(report["lat"], report["lon"], hs.latitude, hs.longitude)
        if d_m > radius_m:
            continue
        if have_report_time:
            hs_time = hs.acquired_at
            if hs_time is not None:
                delta_h = abs((hs_time - report_time).total_seconds()) / 3600.0
                if delta_h > window_hours:
                    continue
        matches.append((d_m, hs))

    matches.sort(key=lambda pair: pair[0])

    result = {
        "confirmed": len(matches) > 0,
        "hotspot_count": len(matches),
        "nearest_km": round(matches[0][0] / 1000.0, 3) if matches else None,
        "nearest_frp": matches[0][1].frp if matches else None,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    return result


# ---------------------------------------------------------------------------
# FIRMS hotspot clustering — O(n) spatial hash
#
# The original loop compared every hotspot against every existing cluster
# (O(n²), ~30+ minutes for the ~100k hotspots a global day-range fetch
# returns). Hotspots are now bucketed into cells sized to the merge radius
# (~2.2 × _FIRMS_CLUSTER_RADIUS_M, see _FIRMS_CELL_DEG) and each hotspot
# only compares against clusters in its own + the 8 neighbouring cells —
# the same single-pass semantics (merge into the nearest cluster whose
# centroid is within _FIRMS_CLUSTER_RADIUS_M), but O(n) overall.
# ---------------------------------------------------------------------------

_FIRMS_CELL_DEG = max(0.06, 2.2 * _FIRMS_CLUSTER_RADIUS_M / 111_320.0)


def _firms_cell(lat: float, lon: float, cell_deg: float = _FIRMS_CELL_DEG) -> tuple[int, int]:
    """Spatial-hash cell for a lat/lon.

    Cell size is derived from the merge radius (≈2.2 × radius, floored at
    0.06° ≈ 6.7 km) so the 3×3-neighbourhood lookup always covers twice
    the merge radius — the invariant the O(n) clustering relies on. The
    longitude cell is widened by 1/cos(lat) so every cell is square (not
    just at the equator), which keeps the 3×3-neighbourhood lookup correct
    at higher latitudes. The 0.3 clamp keeps cells ≥ 1 km wide in
    longitude up to ~78° latitude — beyond that (polar regions, where
    FIRMS active fires don't occur) the 3×3 lookup can in theory miss a
    mergeable cluster, which only means one extra cluster there.
    """
    c = max(math.cos(math.radians(lat)), 0.3)
    return (math.floor(lon / (cell_deg / c)), math.floor(lat / cell_deg))


def _cluster_firms_hotspots(hotspots: list) -> list[dict]:
    """Group hotspots into candidate fires with single-pass clustering.

    Identical semantics to the original O(n²) loop — merge each hotspot
    into the nearest cluster whose centroid is within
    ``_FIRMS_CLUSTER_RADIUS_M``, otherwise start a new cluster — but each
    hotspot only compares against clusters in its own + neighbouring
    spatial cells, so the whole pass is O(n) instead of O(n²).

    Returns a list of {"centroid_lat", "centroid_lon", "points", "hotspots"}.
    """
    clusters: list[dict] = []
    # cell key -> indices of clusters whose current centroid is in that cell
    by_cell: dict[tuple[int, int], list[int]] = {}

    for hs in hotspots:
        cx, cy = _firms_cell(hs.latitude, hs.longitude)
        best_idx = -1
        best_dist = float("inf")

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for ci in by_cell.get((cx + dx, cy + dy), ()):
                    c = clusters[ci]
                    d = _haversine(hs.latitude, hs.longitude, c["centroid_lat"], c["centroid_lon"])
                    if d < best_dist:
                        best_dist = d
                        best_idx = ci

        if best_idx >= 0 and best_dist <= _FIRMS_CLUSTER_RADIUS_M:
            c = clusters[best_idx]
            c["points"].append([hs.latitude, hs.longitude])
            c["hotspots"].append(hs)
            # Running centroid average
            n = len(c["points"])
            c["centroid_lat"] = (c["centroid_lat"] * (n - 1) + hs.latitude) / n
            c["centroid_lon"] = (c["centroid_lon"] * (n - 1) + hs.longitude) / n
            # Keep the bucket index in sync with the (moving) centroid.
            new_cell = _firms_cell(c["centroid_lat"], c["centroid_lon"])
            if new_cell != c["_cell"]:
                bucket = by_cell.get(c["_cell"])
                if bucket is not None and best_idx in bucket:
                    bucket.remove(best_idx)
                    if not bucket:
                        del by_cell[c["_cell"]]
                by_cell.setdefault(new_cell, []).append(best_idx)
                c["_cell"] = new_cell
        else:
            ci = len(clusters)
            cell = _firms_cell(hs.latitude, hs.longitude)
            clusters.append({
                "centroid_lat": hs.latitude,
                "centroid_lon": hs.longitude,
                "points": [[hs.latitude, hs.longitude]],
                "hotspots": [hs],
                "_cell": cell,
            })
            by_cell.setdefault(cell, []).append(ci)

    for c in clusters:
        c.pop("_cell", None)
    return clusters


def _fuse_firms_hotspots(
    grid: "BayesianFireGrid", hotspots: list, is_new_grid: bool = False,
    tag_stats: Optional[dict] = None,
) -> int:
    """Fuse new FIRMS detections into a grid, skipping already-fused ones.

    FIRMS re-returns the FULL past-24h window on every pass, so without
    dedup every poll would (a) re-add +log(50) per hotspot — ratcheting
    probabilities up to the 0.9999 clamp — and (b) re-stamp
    last_updated = now, so evidence never aged past the 12h shelf life
    and evidence-gated decay never engaged. That pinned ~99% of
    production grids at max_p >= 0.85 (the all-crimson wall).

    Dedup is per cell: a hotspot is only fused if that cell has not yet
    seen a detection at least as recent (by acquisition time). Each fused
    detection stamps ``last_updated`` with its acquisition time (not fetch
    time), so the cutoff is "newest detection this cell has seen" and
    decay/purge run off real detection recency.

    Returns the number of hotspots actually injected.
    """
    # Sort oldest-first so per-cell last_updated stamps advance
    # monotonically — the dedup is then deterministic regardless of the
    # order FIRMS returns detections.
    ordered = sorted(
        hotspots,
        key=lambda h: h.acquired_at.timestamp() if h.acquired_at is not None else 0.0,
    )
    injected = 0
    for hs in ordered:
        at = hs.acquired_at
        ts = at.timestamp() if at is not None else None
        if ts is not None:
            ci, cj = grid.latlon_to_cell(hs.latitude, hs.longitude)
            if ts <= grid.last_updated[ci, cj]:
                continue  # already fused this detection
        # Downweight composition. Two independent tags can land on one
        # hotspot (e.g. an industrial flare on a cropland cell is both
        # static-source AND ag-burn): take the most charitable weight — the
        # MAX — and log which tag(s) fired so the composed weight is
        # explainable; if the both-tags case turns out common, it's visible
        # rather than inferable from the number alone.
        #
        #   static-source (Step 2): real heat, probably not fire at all → 0.1
        #   ag-burn (Step 3): real fire, probably not *our* fire → 0.3,
        #       and only when the grid was created this pass — a pre-existing
        #       grid means the fire persisted, so it fuses at full weight
        #       (the escape hatch).
        #
        # max() is the deliberate choice: when a cell is both, err toward
        # treating it as a real fire (0.3) rather than dismissing it as
        # industrial noise (0.1). Either reading is defensible; the tag log
        # makes the decision auditable. Collect the applicable weights and
        # take the max OF THEM (never vs the 1.0 baseline — that would
        # nullify a lone ag-burn downweight).
        applicable: list[float] = []
        tags: list[str] = []
        if getattr(hs, "_static_source", False):
            applicable.append(STATIC_SOURCE_EVIDENCE_WEIGHT)
            tags.append("static")
        if getattr(hs, "_ag_burn", False) and is_new_grid:
            applicable.append(AG_BURN_EVIDENCE_WEIGHT)
            tags.append("ag")
        # Step 4 source filter: volcanic and conflict detections, when the
        # policy is ``downweight``, inject at reduced weight. Unlike ag-burn
        # there is NO escape hatch: a volcano is always persistent, so the
        # hatch would defeat the downweight entirely (a fumarole doesn't
        # "escape" into a wildfire — if it ignites surrounding vegetation,
        # that new hotspot lands outside the volcano radius and gets full
        # weight on its own). ``passthrough``/``full`` never reach here with
        # a tag set.
        if getattr(hs, "_volcanic", False) and SOURCE_FILTER_VOLCANIC == "downweight":
            applicable.append(VOLCANIC_EVIDENCE_WEIGHT)
            tags.append("volcanic")
        if getattr(hs, "_conflict", False) and SOURCE_FILTER_CONFLICT == "downweight":
            applicable.append(CONFLICT_EVIDENCE_WEIGHT)
            tags.append("conflict")
        weight = max(applicable) if applicable else 1.0
        if tags and tag_stats is not None:
            key = " + ".join(sorted(tags))
            tag_stats[key] = tag_stats.get(key, 0) + 1
        grid.update(
            Evidence.satellite_hotspot(lat=hs.latitude, lon=hs.longitude, weight=weight),
            at=ts,
        )
        injected += 1
    return injected


def _gate_firms_by_land_cover(hotspots: list) -> list:
    """Drop FIRMS detections that fall on water or non-burnable land.

    Three-layer lookup (applied in order):
      0. Land mask (Natural Earth 110m) — drop water detections globally
         (ocean, sea, large lakes).  Runs first because it's the cheapest
         and catches the bulk of false positives.
      1. CORINE CLC2018 (100 m vector, Europe only) — authoritative for
         the Poland/EU beta footprint.
      2. ESA WorldCover 2021 (10 m raster, global) — fallback for points
         outside CORINE coverage (the FIRMS API returns worldwide data).

    Points outside ALL layers pass through permissively — the gate can
    never silently delete fires in uncovered regions.

    Side effect: each kept hotspot gets ``h._clc_code`` set to its
    land-cover class code (CORINE string when in Europe, WorldCover int
    when in the global fallback, None when outside both).  The Step 3
    ag-burn tagging reuses this one lookup instead of hitting PostGIS
    again.
    """
    if not hotspots:
        return hotspots

    # Layer 0: Land mask (Natural Earth 110m) — drop water globally
    land_hits = db.land_mask_batch(
        [(h.latitude, h.longitude) for h in hotspots]
    )
    water_dropped = 0
    hotspots_after_land = []
    for i, h in enumerate(hotspots):
        if i in land_hits:
            hotspots_after_land.append(h)
        else:
            water_dropped += 1
    if water_dropped:
        logger.info(
            "[firms] Land mask dropped %d hotspot(s) on water (ocean/sea/lake)",
            water_dropped,
        )
    hotspots = hotspots_after_land

    if not hotspots:
        return hotspots

    # Layer 1: CORINE (Europe)
    corine_codes = db.land_cover_codes_batch(
        [(h.latitude, h.longitude) for h in hotspots]
    )

    # Layer 2: WorldCover (global) — only for points NOT in CORINE
    uncov_indices = [i for i in range(len(hotspots)) if i not in corine_codes]
    wc_codes: dict[int, Optional[int]] = {}
    if uncov_indices:
        wc_points = [(hotspots[i].latitude, hotspots[i].longitude)
                     for i in uncov_indices]
        wc_raw = db.worldcover_code_batch(wc_points)
        # Remap indices back to the original hotspot list positions
        for j, orig_i in enumerate(uncov_indices):
            if j in wc_raw:
                wc_codes[orig_i] = wc_raw[j]

    kept: list = []
    for i, h in enumerate(hotspots):
        # Step 4: passthrough-classified sources (volcanic/conflict) are
        # exempt from the land-cover gate — the operator explicitly wants
        # them shown regardless of the underlying land class.
        if _is_source_passthrough(h):
            h._clc_code = None
            kept.append(h)
            continue

        # Try CORINE first (Europe), then WorldCover (global fallback)
        corine_code = corine_codes.get(i)
        if corine_code is not None:
            if not corine.is_burnable(corine_code):
                continue
            h._clc_code = corine_code
            kept.append(h)
            continue

        # Outside CORINE — try WorldCover
        wc_code = wc_codes.get(i)
        if wc_code is not None:
            if not worldcover.is_burnable(wc_code):
                continue
            h._clc_code = wc_code
            kept.append(h)
            continue

        # Outside both layers — pass through permissively
        h._clc_code = None
        kept.append(h)

    return kept


def _tag_ag_burn(hotspots: list) -> int:
    """Tag FIRMS detections on cropland as agricultural burning (Step 3).

    Checks both CORINE (Europe) and WorldCover (global) cropland classes.
    The signal is the land-cover class alone: the archive calibration found
    no FRP or day/night separation between cropland and forest detections,
    so the land class is the discriminator.  An optional FRP cap (0 = off)
    still exists for deployments that want it.

    Returns the number of hotspots tagged.  Fail-open: hotspots without a
    code (outside both CORINE and WorldCover) are never tagged.
    """
    tagged = 0
    for h in hotspots:
        code = getattr(h, "_clc_code", None)
        if not corine.is_cropland(code) and not worldcover.is_cropland(code):
            continue
        if AG_BURN_FRP_MAX_MW > 0 and h.frp >= AG_BURN_FRP_MAX_MW:
            continue
        h._ag_burn = True
        tagged += 1
    return tagged


def _is_source_passthrough(hotspot) -> bool:
    """True if a tagged hotspot's policy is ``passthrough`` (exempt from
    the land-cover gate — the operator explicitly wants it shown)."""
    if getattr(hotspot, "_volcanic", False) and SOURCE_FILTER_VOLCANIC == "passthrough":
        return True
    if getattr(hotspot, "_conflict", False) and SOURCE_FILTER_CONFLICT == "passthrough":
        return True
    return False


def _tag_volcanic(hotspots: list) -> int:
    """Tag FIRMS detections near a volcano (Step 4 source filter).

    Cross-references hotspots against the ``volcanoes`` table via
    ``db.volcano_hits_batch`` — currently a fail-open stub (the GVP import
    is deferred until geographic expansion), so nothing is tagged on
    current deployments.

    Runs under every non-``full`` policy: ``downweight`` and
    ``passthrough`` need the tag set so the fuse/gate can act on it, and
    ``drop`` needs it so the pass can exclude the hotspot.

    Returns the number of hotspots tagged.
    """
    if SOURCE_FILTER_VOLCANIC == "full":
        return 0
    hits = db.volcano_hits_batch(
        [(h.latitude, h.longitude) for h in hotspots]
    )
    tagged = 0
    for i, h in enumerate(hotspots):
        if hits.get(i):
            h._volcanic = True
            tagged += 1
    return tagged


def _tag_conflict(hotspots: list) -> int:
    """Tag FIRMS detections inside operator-supplied conflict zones.

    Reads ``conflict_zones`` from kv_store (editable without redeploy, same
    pattern as the ``firms_poller`` config): a JSON list where each entry
    is either ``{"type": "bbox", "w": .., "s": .., "e": .., "n": ..}`` or
    ``{"type": "polygon", "points": [[lon,lat], ...]}``. Fail-open: no
    zones stored (or malformed) → nothing tagged.

    Returns the number of hotspots tagged.
    """
    if SOURCE_FILTER_CONFLICT == "full":
        return 0
    zones = db.kv_get("conflict_zones") or []
    if not zones:
        return 0
    prepared = _prepare_conflict_zones(zones)
    if not prepared:
        return 0
    tagged = 0
    for h in hotspots:
        if _point_in_prepared_zones(h.longitude, h.latitude, prepared):
            h._conflict = True
            tagged += 1
    return tagged


def _prepare_conflict_zones(zones: list) -> list:
    """Precompute each zone's bounding box once so the per-hotspot hot loop
    can reject with a 4-comparison bbox test instead of ray-casting every
    polygon (a 651-point Ukraine ring × 400k+ hotspots was adding minutes
    to each FIRMS pass). Returns ``[(ztype, (w, s, e, n), payload), ...]``
    where ``payload`` is the point list for polygons (None for bboxes).
    Malformed zones are skipped (same fail-open semantics as before).
    """
    prepared: list[tuple] = []
    for zone in zones:
        if not isinstance(zone, dict):
            continue
        ztype = zone.get("type")
        if ztype == "bbox":
            try:
                bounds = (zone["w"], zone["s"], zone["e"], zone["n"])
            except KeyError:
                continue
            prepared.append(("bbox", bounds, None))
        elif ztype == "polygon":
            pts = zone.get("points")
            if not pts or len(pts) < 3:
                continue
            lons = [p[0] for p in pts]
            lats = [p[1] for p in pts]
            bounds = (min(lons), min(lats), max(lons), max(lats))
            prepared.append(("polygon", bounds, pts))
    return prepared


def _point_in_prepared_zones(lon: float, lat: float, prepared: list) -> bool:
    """Fast point-in-zone test against precomputed zones: bbox-reject first,
    ray-cast a polygon only when the point is inside its bounds."""
    for ztype, (w, s, e, n), payload in prepared:
        if not (w <= lon <= e and s <= lat <= n):
            continue
        if ztype == "bbox" or _point_in_polygon(lon, lat, payload):
            return True
    return False


def _point_in_conflict_zones(lon: float, lat: float, zones: list) -> bool:
    """Point-in-zone test for raw kv_store ``conflict_zones`` geometry.

    Simple ad-hoc helper (used by validation/debug scripts); the FIRMS pass
    itself uses the precomputed fast path (``_prepare_conflict_zones`` +
    ``_point_in_prepared_zones``) to avoid per-hotspot bounds recomputation.
    """
    for zone in zones:
        if not isinstance(zone, dict):
            continue
        ztype = zone.get("type")
        if ztype == "bbox":
            try:
                if (zone["w"] <= lon <= zone["e"]
                        and zone["s"] <= lat <= zone["n"]):
                    return True
            except KeyError:
                continue
        elif ztype == "polygon":
            pts = zone.get("points")
            if not pts or len(pts) < 3:
                continue
            if _point_in_polygon(lon, lat, pts):
                return True
    return False


def _point_in_polygon(lon: float, lat: float, pts: list) -> bool:
    """Ray-casting point-in-polygon for a closed [[lon,lat], ...] ring."""
    inside = False
    n = len(pts)
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if ((yi > lat) != (yj > lat)) and (
            lon < (xj - xi) * (lat - yi) / (yj - yi) + xi
        ):
            inside = not inside
        j = i
    return inside


def _fetch_nasa_firms_pass(
    min_confidence: str = "nominal",
    day_range: int = 2,
) -> dict:
    """
    Fetch global NASA FIRMS hotspot data and inject into Bayesian grids.

    Unlike the simulation pass (which only works with existing grids), this
    function fetches ALL active fires worldwide from NASA FIRMS, clusters
    nearby hotspot detections into candidate fires, and for each candidate:

      - If a Bayesian grid already exists nearby (within GRID_MATCH_RADIUS_M),
        injects the hotspots as evidence into that grid.
      - If no grid exists, automatically creates a new Bayesian grid for this
        fire, then injects the evidence.

    This means the FIRMS integration is self-driving: it discovers new fires
    from satellite data without requiring any user reports or seed data.  The
    newly created grids appear in /api/bayesian/state on the next poll.

    The query covers the past ``day_range`` days (default 2 / 48h, matching
    NASA's map view) and merges every VIIRS 375m instrument
    (``nasa_firms.DEFAULT_SOURCES``: Suomi-NPP + NOAA-20 + NOAA-21) so
    fires any single satellite missed are still caught — the combined view
    NASA's FIRMS map shows. Each satellite overpass is an independent
    detection; the per-cell acquisition-time dedup in _fuse_firms_hotspots
    prevents re-injection within a source across passes.

    Parameters
    ----------
    min_confidence : str
        Minimum FIRMS confidence: "low", "nominal", or "high".
    day_range : int
        Look-back window in days (1–5; the FIRMS API minimum is 1).

    Returns
    -------
    dict with keys:
        "injected"         : int — total evidence items injected across all grids
        "grids_hit"        : int — grids that received at least one hotspot
        "grids_considered" : int — total grids (existing + newly created)
        "firms_hotspots"   : int — total hotspots returned by FIRMS
        "new_grids"        : int — grids auto-created for previously-unknown fires
        "stale_grids_purged": int — production grids expired (>24h no evidence)
        "api_error"        : str | None — error message if the API call failed
    """
    pass_start = time.time()
    pass_deadline = pass_start + _PASS_TIMEOUT_S

    def _check_timeout(label: str = "") -> None:
        """Raise if the pass has exceeded its wall-clock budget."""
        if time.time() > pass_deadline:
            elapsed = time.time() - pass_start
            raise TimeoutError(
                f"FIRMS pass exceeded {_PASS_TIMEOUT_S:.0f}s limit "
                f"({elapsed:.0f}s elapsed at {label})"
            )

    api_key = nasa_firms._get_api_key()
    if not api_key:
        return {
            "injected": 0, "grids_hit": 0, "grids_considered": 0,
            "firms_hotspots": 0, "new_grids": 0,
            "api_error": "NASA_FIRMS_API_KEY not set",
        }

    # Fetch ALL global hotspots (not just near existing grids)
    print(f"[firms] API key found (len={len(api_key)}), fetching global data "
          f"(day_range={day_range}, min_confidence={min_confidence})...")

    try:
        all_hotspots = nasa_firms.fetch_global_fires(
            api_key=api_key,
            day_range=day_range,
            min_confidence=min_confidence,
            sources=nasa_firms.DEFAULT_SOURCES,
        )
        print(f"[firms] Fetched {len(all_hotspots)} hotspots from FIRMS API")
    except (ConnectionError, ValueError) as exc:
        print(f"[firms] FIRMS fetch FAILED: {exc}")
        return {
            "injected": 0, "grids_hit": 0, "grids_considered": 0,
            "firms_hotspots": 0, "new_grids": 0,
            "api_error": str(exc),
        }

    if not all_hotspots:
        # Still run the expiry sweep — an empty fetch (e.g. no active fires
        # in the last 24h) is exactly when stale grids should be cleaned up.
        print("[firms] FIRMS returned 0 hotspots — CSV was empty or all rows filtered out")
        stale_purged = db.purge_stale_grids("production", max_age_hours=24.0)
        if stale_purged:
            logger.info("[firms] Expired %d stale production grid(s) (no evidence in 24h)", stale_purged)
        return {
            "injected": 0, "grids_hit": 0, "grids_considered": 0,
            "firms_hotspots": 0, "new_grids": 0,
            "stale_grids_purged": stale_purged,
        }

    # --- Source filter (Step 4 of the filtering plan): volcanic/conflict ---
    # Tag non-wildfire source classes BEFORE the land-cover gate. Order
    # matters: a volcano's slopes are often bare rock (CORINE 332 —
    # non-burnable), so the gate would silently drop Etna before the
    # volcanic policy could ever act. Tag first, then apply the policy:
    #   drop        → exclude the tagged hotspots outright
    #   passthrough → the gate below skips them (operator wants them shown)
    #   downweight  → the tag flows to _fuse_firms_hotspots (reduced weight)
    #   full        → no tag, current behavior
    # Both are fail-open: volcano_hits_batch is a stub (GVP import deferred)
    # and an absent/empty conflict_zones list tags nothing.
    volcanic_tagged = _tag_volcanic(all_hotspots)
    conflict_tagged = _tag_conflict(all_hotspots)
    source_dropped = 0
    if SOURCE_FILTER_VOLCANIC == "drop":
        before = len(all_hotspots)
        all_hotspots = [h for h in all_hotspots if not getattr(h, "_volcanic", False)]
        source_dropped += before - len(all_hotspots)
    if SOURCE_FILTER_CONFLICT == "drop":
        before = len(all_hotspots)
        all_hotspots = [h for h in all_hotspots if not getattr(h, "_conflict", False)]
        source_dropped += before - len(all_hotspots)
    if source_dropped:
        logger.info(
            "[firms] Source filter dropped %d hotspot(s) (volcanic/conflict 'drop' policy)",
            source_dropped,
        )

    # --- CORINE land-cover gate (Step 1 of the filtering plan) ---
    # Drop detections on non-burnable land: industrial sites, urban
    # reflections, water glare and bare ground are not wildfires. One
    # batched PostGIS lookup against the loaded CLC2018 layer; hotspots
    # outside CORINE coverage (non-European fires) pass through untouched.
    # ``passthrough``-classified sources are exempt (kept unconditionally).
    raw_hotspots = len(all_hotspots)
    all_hotspots = _gate_firms_by_land_cover(all_hotspots)
    land_cover_dropped = raw_hotspots - len(all_hotspots)
    if land_cover_dropped:
        logger.info(
            "[firms] Land-cover gate dropped %d hotspot(s) on non-burnable CORINE classes (%d kept)",
            land_cover_dropped, len(all_hotspots),
        )

    if not all_hotspots:
        # Everything was industrial/urban noise — still run the expiry sweep
        # so old fires fade, mirroring the empty-fetch path above.
        print("[firms] All hotspots filtered out by the land-cover gate")
        stale_purged = db.purge_stale_grids("production", max_age_hours=24.0)
        if stale_purged:
            logger.info("[firms] Expired %d stale production grid(s) (no evidence in 24h)", stale_purged)
        return {
            "injected": 0, "grids_hit": 0, "grids_considered": 0,
            "firms_hotspots": 0, "new_grids": 0,
            "land_cover_dropped": land_cover_dropped,
            "stale_grids_purged": stale_purged,
        }

    # --- Static-source mask (Step 2 of the filtering plan) ---
    # Tag hotspots that land on a persistent industrial cell (refinery
    # flare, steel mill) so the fusion step below injects them at reduced
    # evidence weight instead of full ln(50). Fail-open: if the mask table
    # isn't built yet, nothing is tagged and everything flows at full
    # weight — the gate can never accidentally starve real fires.
    static_hits = db.static_source_hits_batch(
        [(h.latitude, h.longitude) for h in all_hotspots]
    )
    static_downweighted = 0
    for i, h in enumerate(all_hotspots):
        if static_hits.get(i):
            h._static_source = True
            static_downweighted += 1
    if static_downweighted:
        logger.info(
            "[firms] Static-source mask downweighted %d hotspot(s) on persistent industrial cells (%d at full weight)",
            static_downweighted, len(all_hotspots) - static_downweighted,
        )

    # --- Agricultural-burning tag (Step 3 of the filtering plan) ---
    # Hotspots on CORINE cropland (arable, orchards, vineyards) are real
    # fire but frequently short-lived ag-burning, not spread-relevant
    # wildfire. Tag them so fusion injects at reduced weight (0.3) — unless
    # the fire persists to a later pass (escape hatch in _fuse_firms_hotspots).
    # Reuses the CLC codes captured by _gate_firms_by_land_cover, so no
    # extra DB roundtrips. Fail-open: outside CORINE coverage nothing is
    # tagged.
    ag_downweighted = _tag_ag_burn(all_hotspots)
    if ag_downweighted:
        logger.info(
            "[firms] Agricultural-burn tag marked %d hotspot(s) on cropland (%d at full weight)",
            ag_downweighted, len(all_hotspots) - ag_downweighted,
        )

    # --- Cluster hotspots into candidate fires ---
    # O(n) spatial-hash clustering (see _cluster_firms_hotspots): hotspots
    # within _FIRMS_CLUSTER_RADIUS_M of each other are grouped as one fire.
    clusters = _cluster_firms_hotspots(all_hotspots)
    _check_timeout("clustering")

    logger.info(
        "[firms] Clustered %d hotspots into %d candidate fires",
        len(all_hotspots), len(clusters),
    )

    # Real FIRMS ingestion is PRODUCTION-only: it reads from (and injects
    # into) the production registry so real satellite data never mixes with
    # demo/simulated evidence. Grids persist in Postgres and each fire's
    # evidence injection is a row-locked transaction, so concurrent
    # fetches (multi-worker) can't lose each other's hotspots.
    mode = "production"
    pre_count = db.count_grids(mode)

    total_injected = 0
    grids_hit = 0
    new_grids = 0
    total_hotspots = len(all_hotspots)

    # --- Assign every cluster to a grid and inject evidence, in bulk ---
    # The sequential path (find_or_create_grid + mutate_grid per cluster/
    # grid) costs one DB transaction + advisory lock + PostGIS nearest query
    # per cluster (~25 ms each → 10–40 minutes on a global fetch of
    # ~10k-100k clusters). The bulk path below does the whole pass in a
    # handful of transactions via in-memory spatial matching:
    #   db.bulk_find_or_create_grids — one centroid query + one bulk INSERT
    #   db.bulk_mutate_grids        — chunked id=ANY() loads + one UPDATE batch
    # New grids start with the default wind and get real Open-Meteo values
    # on the next grids.advance wind sweep (wind_updated_at = 0 sorts first).
    grid_ids, new_grid_ids = db.bulk_find_or_create_grids(mode, clusters)
    _check_timeout("grid-matching")

    # Merge clusters that map to the same grid (two FIRMS clusters can sit
    # within GRID_MATCH_RADIUS_M of the same existing grid) so each grid
    # receives ALL its hotspots in ONE mutation — identical to the old
    # sequential merge behaviour, and avoids bulk_mutate_grids overwriting
    # a grid's evidence when the same id appears in two jobs.
    hotspots_by_grid: dict[str, list] = {}
    for grid_id, c in zip(grid_ids, clusters):
        hotspots_by_grid.setdefault(grid_id, []).extend(c["hotspots"])

    jobs = []
    # Composition stats: which tag(s) fired into the composed weight (e.g.
    # "static", "ag", "static + ag"). Logged once after the pass so the
    # both-tags case is visible without per-hotspot log spam. Shared by all
    # job closures — bulk_mutate_grids runs them sequentially in one process.
    tag_stats: dict[str, int] = {}
    for grid_id, hotspots in hotspots_by_grid.items():
        # Escape hatch: only ag-burn detections injected into a grid created
        # THIS pass are downweighted. A grid that already existed means the
        # fire survived a previous pass — it's persisting, not a one-off
        # burn, so its ag-burn detections fuse at full weight. (A grid can't
        # expire mid-burn: purge_stale_grids only deletes grids with no
        # evidence for 24h, and an active fire is re-detected on every
        # overpass, so recreation from expiry is a genuinely new episode.)
        is_new = grid_id in new_grid_ids

        def _inject(grid: BayesianFireGrid, entry: dict, _hs=hotspots,
                    _new=is_new, _stats=tag_stats) -> int:
            return _fuse_firms_hotspots(grid, _hs, is_new_grid=_new,
                                        tag_stats=_stats)

        jobs.append((grid_id, _inject))

    _check_timeout("before-fusion")
    results = db.bulk_mutate_grids(mode, jobs)
    _check_timeout("fusion-complete")
    total_injected = sum(results.values())
    grids_hit = sum(1 for v in results.values() if v)

    post_count = db.count_grids(mode)
    new_grids = max(0, post_count - pre_count)

    logger.info(
        "[firms] Injected %d hotspots across %d grids (%d new)",
        total_injected, grids_hit, new_grids,
    )

    # Composition breakdown: which downweight tag(s) fired ("static",
    # "ag", "static + ag"). Logged once per pass so the both-tags case is
    # visible — if it turns out common, it's explainable rather than
    # inferred from the weights alone.
    if tag_stats:
        summary = ", ".join(f"{k}={v}" for k, v in sorted(tag_stats.items()))
        logger.info("[firms] Downweight composition: %s", summary)

    # --- Cross-check crowdsourced reports against this pass's hotspots ---
    # Reuses the hotspot list already fetched above — no extra API call.
    # Static-source detections (refinery flares) must not confirm citizen
    # reports: the heat is real, but it is not a wildfire, so it should not
    # lend satellite corroboration to a crowdsourced report. Ag-burn
    # detections DO confirm: a citizen photographing a field fire is a true
    # positive sighting, regardless of the reduced grid weight — report
    # confirmation answers "did a hotspot really happen here", which is
    # orthogonal to the spread-model evidence weight.
    # Volcanic detections (a photo of lava is not a wildfire sighting) are
    # excluded from confirmation, same as static sources; conflict fires DO
    # confirm (fire is fire). passthrough volcanic flows through.
    reports_confirmed = _confirm_reports_against_hotspots(
        [h for h in all_hotspots
         if not getattr(h, "_static_source", False)
         and not (getattr(h, "_volcanic", False)
                  and SOURCE_FILTER_VOLCANIC == "downweight")]
    )

    # Expire fires that have stopped being detected: any production grid
    # whose newest evidence is older than 24h is deleted so old fires
    # disappear from the map (and the DB). Runs after every fetch so the
    # map stays current even if the periodic purge job is idle.
    stale_purged = db.purge_stale_grids("production", max_age_hours=24.0)
    if stale_purged:
        logger.info("[firms] Expired %d stale production grid(s) (no evidence in 24h)", stale_purged)

    return {
        "injected": total_injected,
        "grids_hit": grids_hit,
        "grids_considered": post_count,
        "firms_hotspots": total_hotspots,
        "land_cover_dropped": land_cover_dropped,
        "static_downweighted": static_downweighted,
        "ag_downweighted": ag_downweighted,
        "volcanic_tagged": volcanic_tagged,
        "conflict_tagged": conflict_tagged,
        "source_dropped": source_dropped,
        "new_grids": new_grids,
        "stale_grids_purged": stale_purged,
        "reports_confirmed": reports_confirmed,
        "timed_out": False,
    }


def _confirm_reports_against_hotspots(
    hotspots: list["nasa_firms.FIRMSHotspot"],
) -> int:
    """
    Cross-check all 'confirmed' (accepted) crowdsourced reports against a
    list of FIRMS hotspots, updating each report's `satellite_confirmation`
    field in place. Persists changes to disk.

    Only accepted reports are checked — pending/rejected reports haven't
    been vetted by a moderator yet, so satellite corroboration for them
    isn't meaningful in the same way.

    Returns the number of reports newly or still marked as confirmed.
    """
    # Only PRODUCTION reports are cross-checked — demo/seed reports are
    # never "satellite-confirmed" against real FIRMS data. Each report is
    # updated with a single-row UPDATE (no read-modify-write of the store).
    #
    # Hotspots are bucketed into ~4.4 km cells ONCE, then each report only
    # compares against the handful of hotspots in its own + neighbouring
    # cells. (The original code scanned every hotspot for every report —
    # O(reports × ~100k) haversines per pass.)
    reports = db.list_reports("production", status="confirmed")
    cell_deg = 0.04  # ≈4.4 km ≥ the 3 km confirmation radius
    buckets: dict[tuple[int, int], list] = {}
    for hs in hotspots:
        buckets.setdefault(_firms_cell(hs.latitude, hs.longitude, cell_deg), []).append(hs)

    changed = 0
    newly_confirmed = 0

    for r in reports:
        cx, cy = _firms_cell(r["lat"], r["lon"], cell_deg)
        nearby: list = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nearby.extend(buckets.get((cx + dx, cy + dy), ()))

        result = _match_report_to_hotspots(r, nearby)
        result["source"] = "firms-poll"
        prev = r.get("satellite_confirmation") or {}
        meaningfully_changed = (
            prev.get("confirmed") != result["confirmed"]
            or prev.get("hotspot_count") != result["hotspot_count"]
        )
        if meaningfully_changed or "satellite_confirmation" not in r:
            r["satellite_confirmation"] = result
            if db.update_report(r, "production"):
                changed += 1
        if result["confirmed"]:
            newly_confirmed += 1

    return newly_confirmed


def _firms_fetch_message(result: dict) -> str:
    """Human-readable summary of a FIRMS pass result dict."""
    parts = []
    if result.get("new_grids"):
        parts.append(f"Auto-created {result['new_grids']} new grid(s) for previously unknown fires")
    if result.get("injected"):
        parts.append(f"Injected {result['injected']} evidence items across {result['grids_hit']} grid(s)")
    if result.get("firms_hotspots"):
        parts.append(f"from {result['firms_hotspots']} FIRMS hotspot(s)")
    if result.get("static_downweighted"):
        parts.append(
            f"downweighted {result['static_downweighted']} static-source hotspot(s)"
        )
    if result.get("ag_downweighted"):
        parts.append(
            f"downweighted {result['ag_downweighted']} ag-burn hotspot(s) on cropland"
        )
    if result.get("stale_grids_purged"):
        parts.append(f"Expired {result['stale_grids_purged']} stale fire(s) (>24h no evidence)")
    if not parts:
        parts.append("No FIRMS hotspots found worldwide in the last 24 hours")
    return ", ".join(parts)


@app.route("/api/satellite/firms-fetch", methods=["POST"])
def satellite_firms_fetch():
    """
    Manually trigger a global NASA FIRMS data fetch.

    The pass itself (global FIRMS API call, clustering ~100k hotspots,
    grid injection) runs as a background job in the Procrastinate worker:
    it can take a minute or two, and running it inline blocked the request
    thread (the button spun for minutes with no feedback).

    This endpoint checks the API key, queues the job (bypassing the
    poller's interval throttle via ``force=True``) and returns immediately
    with ``{"accepted": true}``. Poll ``/api/satellite/poller/status`` for
    ``firms_fetch_in_progress`` / ``firms_fetch_last_result``.

    JSON body (all optional):
      {
        "day_range": 2,             // look-back days (1-5); default 2 = 48h
        "min_confidence": "nominal"  // minimum confidence: low/nominal/high
      }
    """
    data = request.get_json(silent=True) or {}

    if not nasa_firms._get_api_key():
        return jsonify({
            "error": "NASA_FIRMS_API_KEY not set — cannot fetch FIRMS data.",
        }), 400

    # Look-back window, same as the poller: default to the configured
    # day_range (currently 1 = 24h; historically 2 = 48h, matching NASA's
    # map view), clamped to the FIRMS API's 1-5 day range. Every caller
    # funnels through _fetch_nasa_firms_pass, which merges all VIIRS
    # satellites, so this is the single knob for how far back we look.
    day_range = int(data.get(
        "day_range",
        db.get_poller_config("firms", {}).get("day_range", 2),
    ))
    day_range = max(1, min(day_range, 5))
    min_confidence = data.get("min_confidence", "nominal")

    # One manual fetch at a time: reject a second click while a fetch is
    # already queued/running (stale flags older than 15 min are ignored —
    # e.g. the worker was restarted mid-fetch).
    in_prog = db.kv_get("firms_fetch_in_progress") or {}
    if in_prog.get("at") and time.time() - in_prog.get("at") < 900:
        return jsonify({
            "error": "A FIRMS fetch is already running in the background.",
        }), 409

    # Mark a manual fetch as in progress and queue the job. The worker
    # clears the flag + stores the result summary in kv_store when done.
    db.kv_set("firms_fetch_in_progress", {"at": time.time()})
    db.kv_set("firms_fetch_last_result", None)
    try:
        _job_defer(
            "firms.fetch",
            force=True,
            day_range=int(day_range),
            min_confidence=str(min_confidence),
        )
    except Exception:
        # Queue unavailable — undo the claim so the next click isn't
        # blocked with a stale 409 for the next 15 minutes.
        db.kv_set("firms_fetch_in_progress", False)
        db.kv_set("firms_fetch_last_result", None)
        logger.exception("Failed to defer firms.fetch job")
        return jsonify({
            "error": "Could not queue the FIRMS fetch — is the worker database reachable?",
        }), 500

    return jsonify({
        "accepted": True,
        "message": "FIRMS fetch started — processing in the background. "
                   "The map will refresh automatically when it finishes.",
        "in_progress": True,
    })


@app.route("/api/satellite/firms-poller/start", methods=["POST"])
def satellite_firms_poller_start():
    """Enable the periodic real-NASA-FIRMS job in the queue (global,
    self-driving). Runs in the Procrastinate worker every ~10 min; this
    endpoint flips the flag the job checks."""
    data = request.get_json(silent=True) or {}
    interval = data.get("interval_s", 600.0)  # default: 10 min
    # Look-back window, same as the manual fetch: default 2 days (48h),
    # clamped to the FIRMS API's 1-5 day range.
    day_range = int(data.get("day_range", 2))
    day_range = max(1, min(day_range, 5))
    min_confidence = data.get("min_confidence", "nominal")

    # Validate the API key early
    if not nasa_firms._get_api_key():
        return jsonify({
            "status": "error",
            "error": "NASA_FIRMS_API_KEY environment variable is not set. "
                     "Set it to your free NASA FIRMS API key and restart.",
        }), 400

    db.set_poller_active("firms", True, {
        "interval_s": interval,
        "day_range": day_range,
        "min_confidence": min_confidence,
    })

    return jsonify({
        "status": "started",
        "message": f"FIRMS poller scheduled (interval={interval}s, confidence={min_confidence}).",
        "interval_s": interval,
    })


@app.route("/api/satellite/firms-poller/stop", methods=["POST"])
def satellite_firms_poller_stop():
    """Disable the periodic real-NASA-FIRMS job."""
    was_active = db.get_poller_config("firms", {})["active"]
    db.set_poller_active("firms", False)

    return jsonify({
        "status": "stopped" if was_active else "idle",
        "message": "FIRMS poller stopped." if was_active else "FIRMS poller was not running.",
    })


@app.route("/api/satellite/poller/status", methods=["GET"])
def satellite_poller_status():
    """Check poller flags and worker heartbeats (durable, cross-process)."""
    now = time.time()
    sat = db.get_poller_config("satellite", {})
    firms = db.get_poller_config("firms", {})
    sat_hb = (db.kv_get("satellite_poller_heartbeat") or {}).get("at", 0)
    firms_hb = (db.kv_get("firms_poller_heartbeat") or {}).get("at", 0)
    sat_alive = sat["active"] and (now - sat_hb) < max(60.0, float(sat.get("interval_s", 20)) * 3)
    firms_alive = firms["active"] and (now - firms_hb) < max(300.0, float(firms.get("interval_s", 600)) * 3)
    return jsonify({
        "active": sat["active"] or firms["active"],
        "alive": sat_alive or firms_alive,
        "satellite_poller_active": sat["active"],
        "satellite_alive": sat_alive,
        "firms_poller_active": firms["active"],
        "firms_alive": firms_alive,
        "firms_fetch_in_progress": bool(db.kv_get("firms_fetch_in_progress")),
        "firms_fetch_last_result": db.kv_get("firms_fetch_last_result"),
        "grids_tracked": db.count_grids("production") + db.count_grids("demo"),
        "production_grids": db.count_grids("production"),
        "demo_grids": db.count_grids("demo"),
    })



@app.route("/api/live-demo/start", methods=["POST"])
def live_demo_start():
    """
    Start a live simulation of a wildfire unfolding.

    Creates a sequence of 12 citizen reports over 6 steps.
    The frontend polls /api/live-demo/step to reveal reports progressively,
    showing how triangulation converges as more bearing reports come in.

    All observers are within the 500m cluster radius so they always form
    a single cluster. Early reports are close together (nearly parallel
    rays → low confidence), later reports spread wider (better baseline →
    high confidence).

    Scenario: a fire starts near El Capitan Meadow in Yosemite Valley.
    """
    global _live_demo

    import random as _random

    # Fire origin: El Capitan Meadow area, Yosemite Valley
    fire_lat = 37.7270
    fire_lon = -119.6370

    # Observers around the fire origin, in concentric rings of increasing radius.
    # Each pair is roughly symmetric so the cluster centroid stays near the fire.
    # All distances are well within the 500m CLUSTER_RADIUS_M.
    # (lat, lon, _unused_source, offset_seconds, bearing_error_deg)
    observers = [
        # Step 0: ~60m from fire (very close → poor baseline, high bearing error)
        (37.72755, -119.63645, "hiker", 0, 14.0),
        (37.72645, -119.63755, "hiker", 3, 14.0),
        # Step 1: ~150m ring
        (37.72835, -119.63700, "citizen", 7, 11.0),
        (37.72565, -119.63700, "citizen", 10, 11.0),
        # Step 2: ~230m ring
        (37.72900, -119.63855, "ranger", 14, 9.0),
        (37.72500, -119.63545, "citizen", 17, 9.0),
        # Step 3: ~310m ring
        (37.72980, -119.63520, "ranger", 21, 7.0),
        (37.72420, -119.63880, "citizen", 24, 7.0),
        # Step 4: ~390m ring
        (37.73050, -119.63700, "emergency services", 28, 5.0),
        (37.72350, -119.63700, "citizen", 31, 5.0),
        # Step 5: ~450m ring (at cluster edge)
        (37.73100, -119.63955, "citizen", 35, 4.0),
        (37.72300, -119.63445, "ranger", 38, 4.0),
    ]

    now = datetime.now(timezone.utc)

    demo_reports = []
    for i, obs in enumerate(observers):
        lat, lon, _source, offset_sec, bearing_err = obs
        captured_at = now - timedelta(minutes=45) + timedelta(seconds=offset_sec)

        # Compute heading toward fire origin with gaussian noise proportional to error
        dlat = fire_lat - lat
        dlon = fire_lon - lon
        math_bearing = math.degrees(math.atan2(dlat, dlon))
        heading = ((90 - math_bearing) + 360) % 360
        heading += _random.gauss(0, bearing_err)
        heading %= 360

        report = {
            "id": uuid.uuid4().hex,
            "lat": lat,
            "lon": lon,
            "photo_url": _gen_photo_url(i + 100),
            "captured_at": captured_at.isoformat(),
            "device_heading": heading,
            "bearing_error_deg": bearing_err,
            "session_id": "live-demo-session",
            "status": "confirmed",
            "source_type": "citizen",
            "created_at": now.isoformat(),
        }
        demo_reports.append(report)

    # 6 steps, 2 reports each
    steps = [
        [0, 1],       # step 0 (t=0s): ~60m ring — barely any baseline
        [2, 3],       # step 1 (t=7s): ~150m ring — better baseline
        [4, 5],       # step 2 (t=14s): ~230m ring
        [6, 7],       # step 3 (t=21s): ~310m ring
        [8, 9],       # step 4 (t=28s): ~390m ring
        [10, 11],     # step 5 (t=35s): ~450m ring — widest baseline
    ]

    _live_demo = {
        "fire_lat": fire_lat,
        "fire_lon": fire_lon,
        "reports": demo_reports,
        "steps": steps,
        "current_step": -1,  # no steps revealed yet
        "started_at": now.isoformat(),
        "active": True,
    }

    return jsonify({
        "status": "started",
        "total_reports": len(demo_reports),
        "total_steps": len(steps),
        "fire_lat": fire_lat,
        "fire_lon": fire_lon,
        "message": "Live demo started. Poll /api/live-demo/step to reveal reports progressively.",
    }), 200


@app.route("/api/live-demo/step", methods=["POST"])
def live_demo_step():
    """
    Advance the live demo by one step, seeding the next batch of reports
    into the persistent store. Returns the newly seeded reports and updated
    clusters/triangulation.

    The frontend calls this every ~6-8 seconds to simulate real-time
    report inflow.
    """
    global _live_demo

    if not _live_demo or not _live_demo.get("active"):
        return jsonify({"error": "No active live demo. Call /api/live-demo/start first."}), 400

    step = _live_demo["current_step"] + 1
    steps = _live_demo["steps"]

    if step >= len(steps):
        return jsonify({
            "status": "complete",
            "message": "All reports have been revealed. Demo complete.",
            "step": step,
            "total_steps": len(steps),
            "reports_seeded": [],
            "total_seeded": len(_live_demo["reports"]),
        }), 200

    # Get the report indices for this step
    indices = steps[step]
    reports_to_add = [_live_demo["reports"][i] for i in indices]

    # Save to the DEMO store — live-demo reports are simulated and must
    # never enter the production reports store. Pure INSERT, so concurrent
    # steps can't lose reports.
    db.insert_reports(reports_to_add, "demo")
    existing = _load_reports(demo=True)

    # Update step counter
    _live_demo["current_step"] = step

    # Compute clusters and triangulation
    clusters = _compute_clusters(existing)

    # Feed new reports into the DEMO Bayesian grid registry
    _feed_reports_into_grid(reports_to_add, demo=True)

    # Check if we've revealed all
    all_revealed = (step + 1) >= len(steps)
    if all_revealed:
        _live_demo["active"] = False

    # Find triangulation for the fire cluster (the one closest to our fire origin)
    fire_cluster = None
    for c in clusters:
        if c.get("triangulation"):
            dist = _haversine(
                c["centroid_lat"], c["centroid_lon"],
                _live_demo["fire_lat"], _live_demo["fire_lon"]
            )
            if dist < 2000:  # within 2 km of the demo fire origin
                fire_cluster = c
                break

    return jsonify({
        "status": "step",
        "step": step,
        "total_steps": len(steps),
        "reports_seeded": reports_to_add,
        "total_seeded": sum(len(s) for s in steps[:step + 1]),
        "total_reports": len(_live_demo["reports"]),
        "all_revealed": all_revealed,
        "clusters": clusters,
        "cluster_count": len(clusters),
        "fire_cluster": fire_cluster,
        "demo_fire_lat": _live_demo["fire_lat"],
        "demo_fire_lon": _live_demo["fire_lon"],
    }), 200


@app.route("/api/live-demo/reset", methods=["POST"])
def live_demo_reset():
    """
    Clear all reports created by the live demo.
    """
    global _live_demo

    if _live_demo:
        for rid in {r["id"] for r in _live_demo["reports"]}:
            db.delete_report(rid, "demo")

    _live_demo = None
    # Also reset the DEMO Bayesian grid registry so no stale evidence persists
    db.delete_grids("demo")

    return jsonify({"status": "reset", "message": "Live demo reset. Demo reports removed."}), 200


# ---------------------------------------------------------------------------
# Bayesian Historic Demo — 2020 Creek Fire (Sierra National Forest, CA)
# ---------------------------------------------------------------------------

# In-memory state for the historic Bayesian demo
_bayesian_demo: Optional[dict] = None


# Creek Fire 2020 scenario: satellite hotspot sequence with evolving weather.
# Each step provides new VIIRS detections, changing wind, and an approximate
# true fire perimeter at that point in the fire's progression.
#
# The Creek Fire started Sept 4, 2020 near Shaver Lake, CA (~37.135, -119.283)
# and grew to 380,000 acres driven by strong easterly (downslope) winds.
# This demo compresses ~4 days of growth into 9 steps for an ~45-second demo.

CREEK_FIRE_SCENARIO = {
    "name": "2020 Creek Fire",
    "location": "Sierra National Forest, CA",
    "fire_lat": 37.135,
    "fire_lon": -119.283,
    "start_date": "2020-09-04",
    "total_acres": 379895,
    "steps": [
        {  # Step 0: Early afternoon — first VIIRS detection
            "label": "Day 1 — Initial Detection",
            "description": "VIIRS satellite detects first hotspot near Shaver Lake. Light easterly winds.",
            "wind_speed": 2.0,
            "wind_dir_deg": 270,  # from east, blowing west
            "hotspots": [
                [37.1348, -119.2825],
                [37.1352, -119.2838],
            ],
            # ~50 acre initial perimeter (tiny ellipse)
            "perimeter": [
                [37.1345, -119.2820], [37.1355, -119.2820],
                [37.1358, -119.2835], [37.1355, -119.2845],
                [37.1345, -119.2845], [37.1340, -119.2835],
            ],
        },
        {  # Step 1: Late afternoon — fire picking up
            "label": "Day 1 — Afternoon Growth",
            "description": "Multiple satellite hotspots confirm active fire. Easterly winds strengthening.",
            "wind_speed": 5.0,
            "wind_dir_deg": 270,
            "hotspots": [
                [37.1340, -119.2830],
                [37.1355, -119.2815],
                [37.1360, -119.2845],
                [37.1340, -119.2850],
            ],
            # ~200 acre perimeter
            "perimeter": [
                [37.1338, -119.2815], [37.1362, -119.2815],
                [37.1368, -119.2835], [37.1362, -119.2855],
                [37.1340, -119.2858], [37.1332, -119.2840],
            ],
        },
        {  # Step 2: Evening — fire growing
            "label": "Day 1 — Evening Escalation",
            "description": "Winds picking up dramatically. Fire beginning to spread west toward Huntington Lake.",
            "wind_speed": 10.0,
            "wind_dir_deg": 275,
            "hotspots": [
                [37.1335, -119.2860],
                [37.1348, -119.2805],
                [37.1365, -119.2810],
                [37.1358, -119.2870],
                [37.1330, -119.2840],
                [37.1360, -119.2885],
            ],
            # ~1,000 acre perimeter — fire starting to elongate westward
            "perimeter": [
                [37.1330, -119.2805], [37.1370, -119.2805],
                [37.1380, -119.2835], [37.1375, -119.2890],
                [37.1355, -119.2900], [37.1330, -119.2890],
                [37.1325, -119.2860],
            ],
        },
        {  # Step 3: Day 2 morning — rapid growth begins
            "label": "Day 2 — Rapid Growth Phase",
            "description": "Strong downslope easterly winds (35 mph gusts) drive explosive westward spread. Fire jumping across canyons.",
            "wind_speed": 16.0,
            "wind_dir_deg": 275,
            "hotspots": [
                [37.1320, -119.2900],
                [37.1345, -119.2920],
                [37.1365, -119.2895],
                [37.1375, -119.2850],
                [37.1335, -119.2790],
                [37.1350, -119.2780],
                [37.1370, -119.2910],
                [37.1315, -119.2880],
                [37.1340, -119.2950],
                [37.1360, -119.2940],
            ],
            # ~5,000 acres — fire expanding rapidly westward
            "perimeter": [
                [37.1315, -119.2790], [37.1375, -119.2785],
                [37.1390, -119.2840], [37.1385, -119.2920],
                [37.1360, -119.2955], [37.1325, -119.2940],
                [37.1305, -119.2900], [37.1300, -119.2850],
            ],
        },
        {  # Step 4: Day 2 evening — peak winds, maximum spread
            "label": "Day 2 — Peak Wind Event",
            "description": "Peak easterly winds (45 mph). Fire makes its biggest run, advancing miles westward. Crowning in timber.",
            "wind_speed": 20.0,
            "wind_dir_deg": 280,
            "hotspots": [
                [37.1300, -119.2980],
                [37.1325, -119.3000],
                [37.1350, -119.2970],
                [37.1370, -119.2960],
                [37.1380, -119.2930],
                [37.1330, -119.2850],
                [37.1310, -119.2820],
                [37.1360, -119.2825],
                [37.1340, -119.3020],
                [37.1360, -119.3005],
                [37.1315, -119.3040],
                [37.1335, -119.3060],
                [37.1300, -119.2920],
                [37.1375, -119.2985],
                [37.1350, -119.3080],
            ],
            # ~25,000 acres — fire extends far west
            "perimeter": [
                [37.1290, -119.2800], [37.1390, -119.2800],
                [37.1405, -119.2860], [37.1400, -119.2950],
                [37.1380, -119.3060], [37.1350, -119.3110],
                [37.1315, -119.3100], [37.1290, -119.3040],
                [37.1275, -119.2960], [37.1270, -119.2880],
            ],
        },
        {  # Step 5: Day 3 — fire continues spreading
            "label": "Day 3 — Sustained Growth",
            "description": "Winds remain strong. Fire continues westward push through the San Joaquin River canyon. Mandatory evacuations in effect.",
            "wind_speed": 14.0,
            "wind_dir_deg": 270,
            "hotspots": [
                [37.1280, -119.3120],
                [37.1310, -119.3140],
                [37.1340, -119.3110],
                [37.1365, -119.3070],
                [37.1375, -119.3020],
                [37.1330, -119.3080],
                [37.1295, -119.3160],
                [37.1320, -119.3180],
                [37.1355, -119.3140],
                [37.1275, -119.3060],
                [37.1300, -119.2850],
                [37.1380, -119.2950],
            ],
            # ~80,000 acres — fire growing both west and north
            "perimeter": [
                [37.1260, -119.2780], [37.1400, -119.2780],
                [37.1420, -119.2860], [37.1415, -119.2980],
                [37.1390, -119.3120], [37.1355, -119.3200],
                [37.1310, -119.3195], [37.1275, -119.3140],
                [37.1255, -119.3040], [37.1245, -119.2920],
            ],
        },
        {  # Step 6: Day 4 — fire approaching full extent
            "label": "Day 4 — Approaching Peak",
            "description": "Fire has grown to over 150,000 acres. Winds finally easing. Firefighters focusing on structure defense.",
            "wind_speed": 8.0,
            "wind_dir_deg": 260,
            "hotspots": [
                [37.1260, -119.3200],
                [37.1290, -119.3220],
                [37.1325, -119.3205],
                [37.1350, -119.3170],
                [37.1370, -119.3100],
                [37.1305, -119.3240],
                [37.1270, -119.3100],
                [37.1330, -119.2800],
            ],
            # ~200,000 acres — near full extent
            "perimeter": [
                [37.1230, -119.2760], [37.1410, -119.2760],
                [37.1435, -119.2860], [37.1430, -119.3000],
                [37.1405, -119.3160], [37.1370, -119.3250],
                [37.1315, -119.3260], [37.1265, -119.3220],
                [37.1235, -119.3120], [37.1215, -119.2980],
                [37.1210, -119.2860],
            ],
        },
        {  # Step 7: Day 5 — slowing down
            "label": "Day 5 — Slowing",
            "description": "Winds subsiding. Fire growth slowing. Over 250,000 acres burned. Crews making progress on containment.",
            "wind_speed": 5.0,
            "wind_dir_deg": 250,
            "hotspots": [
                [37.1250, -119.3180],
                [37.1280, -119.3260],
                [37.1315, -119.3245],
                [37.1340, -119.3150],
                [37.1285, -119.3280],
                [37.1260, -119.3120],
            ],
            # ~300,000 acres
            "perimeter": [
                [37.1210, -119.2740], [37.1420, -119.2740],
                [37.1445, -119.2860], [37.1440, -119.3020],
                [37.1415, -119.3180], [37.1380, -119.3280],
                [37.1320, -119.3295], [37.1260, -119.3260],
                [37.1225, -119.3160], [37.1200, -119.3020],
                [37.1195, -119.2880],
            ],
        },
        {  # Step 8: Final — full extent
            "label": "Final Extent",
            "description": "380,000 acres burned. Full containment achieved after several weeks. One of California's largest wildfires.",
            "wind_speed": 3.0,
            "wind_dir_deg": 270,
            "hotspots": [
                [37.1260, -119.3240],
                [37.1300, -119.3300],
                [37.1325, -119.3260],
                [37.1280, -119.3200],
                [37.1310, -119.3220],
                [37.1270, -119.3160],
            ],
            # ~380,000 acres — final perimeter
            "perimeter": [
                [37.1190, -119.2720], [37.1430, -119.2720],
                [37.1455, -119.2860], [37.1450, -119.3040],
                [37.1425, -119.3200], [37.1390, -119.3310],
                [37.1325, -119.3325], [37.1255, -119.3290],
                [37.1215, -119.3180], [37.1185, -119.3060],
                [37.1175, -119.2900],
            ],
        },
    ],
}


@app.route("/api/bayesian-demo/start", methods=["POST"])
def bayesian_demo_start():
    """
    Start the Bayesian historic fire simulation (2020 Creek Fire replay).

    Initializes a Bayesian grid centered on the fire origin and seeds
    it with the first step's satellite hotspot evidence.  Frontend calls
    this once, then polls /api/bayesian-demo/step to advance.
    """
    global _bayesian_demo

    scenario = CREEK_FIRE_SCENARIO
    start_step = scenario["steps"][0]

    # Compute the Creek Fire's full extent from all steps (hotspots + perimeters)
    # so the grid is auto-sized and capped the same way as seed data grids.
    all_lats, all_lons = [], []
    for step in scenario["steps"]:
        for hs in step.get("hotspots", []):
            all_lats.append(hs[0])
            all_lons.append(hs[1])
        for pt in step.get("perimeter", []):
            all_lats.append(pt[0])
            all_lons.append(pt[1])

    sizing = auto_grid_size(all_lats, all_lons, margin_m=5000.0)
    if sizing:
        demo_grid = BayesianFireGrid(
            center_lat=sizing["center_lat"],
            center_lon=sizing["center_lon"],
            cell_size_m=sizing["cell_size_m"],
            nx=sizing["nx"],
            ny=sizing["ny"],
        )
    else:
        # Fallback (shouldn't happen, but just in case)
        demo_grid = BayesianFireGrid(
            center_lat=scenario["fire_lat"],
            center_lon=scenario["fire_lon"],
            nx=140,
            ny=100,
        )

    # Register under a fixed key in the DEMO grid registry (rather than a
    # separate standalone variable) so it's picked up by _grid_to_json the
    # same way any other cluster's grid is. It is deliberately registered in
    # the demo registry — a historic replay must never pollute the
    # production grids that drive real operational outputs. The grid row
    # persists in Postgres, so it survives restarts.
    db.upsert_grid(
        "demo", "creek-fire-demo", demo_grid,
        scenario["fire_lat"], scenario["fire_lon"],
        wind_speed=start_step["wind_speed"],
        wind_dir_deg=start_step["wind_dir_deg"],
    )

    # Seed with first step's satellite hotspots + initial predict, atomically.
    def _seed_start(grid: BayesianFireGrid, entry: dict, _hs=start_step["hotspots"], _ws=start_step["wind_speed"], _wd=start_step["wind_dir_deg"]) -> None:
        for hs in _hs:
            grid.update(Evidence.satellite_hotspot(lat=hs[0], lon=hs[1]))
        grid.predict(dt=300.0, wind_speed=_ws, wind_dir_deg=_wd, slope_lookup=db.dem_terrain_lookup)

    db.mutate_grid("demo", "creek-fire-demo", _seed_start)

    # Store demo state
    _bayesian_demo = {
        "scenario": scenario,
        "current_step": 0,
        "total_steps": len(scenario["steps"]),
        "active": True,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    return jsonify({
        "status": "started",
        "name": scenario["name"],
        "location": scenario["location"],
        "fire_lat": scenario["fire_lat"],
        "fire_lon": scenario["fire_lon"],
        "total_steps": len(scenario["steps"]),
        "step": 0,
        "step_label": start_step["label"],
        "step_description": start_step["description"],
        "wind_speed": start_step["wind_speed"],
        "wind_dir_deg": start_step["wind_dir_deg"],
        "hotspots": start_step["hotspots"],
        "perimeter": start_step["perimeter"],
    }), 200


@app.route("/api/bayesian-demo/step", methods=["POST"])
def bayesian_demo_step():
    """
    Advance the Bayesian historic demo by one step.

    Each step:
      1. Injects new satellite hotspot evidence into the Bayesian grid
      2. Updates wind conditions
      3. Runs a predict step to simulate fire spread under those winds

    Returns the updated satellite hotspots, fire perimeter, wind, grid stats,
    and the full Bayesian grid state (cells + contour) for frontend rendering.
    """
    global _bayesian_demo

    if not _bayesian_demo or not _bayesian_demo.get("active"):
        return jsonify({"error": "No active Bayesian demo. Call /api/bayesian-demo/start first."}), 400

    if db.get_grid_entry("demo", "creek-fire-demo") is None:
        return jsonify({"error": "Demo grid missing. Call /api/bayesian-demo/start first."}), 400

    step = _bayesian_demo["current_step"] + 1
    steps = _bayesian_demo["scenario"]["steps"]

    if step >= len(steps):
        # Demo complete — return final state one last time (demo registry)
        grid_data = _grid_to_json(threshold=0.02, contour_level=0.6, demo=True)
        return jsonify({
            "status": "complete",
            "message": "Historical replay complete.",
            "step": step,
            "total_steps": len(steps),
            **grid_data,
        }), 200

    step_data = steps[step]

    # 1+2. Inject new satellite hotspot evidence and run the predict step
    # (30 minutes of spread at this step's wind), atomically.
    def _advance(grid: BayesianFireGrid, entry: dict, _sd=step_data) -> None:
        for hs in _sd["hotspots"]:
            grid.update(Evidence.satellite_hotspot(lat=hs[0], lon=hs[1]))
        grid.predict(
            dt=1800.0,
            wind_speed=_sd["wind_speed"],
            wind_dir_deg=_sd["wind_dir_deg"],
            slope_lookup=db.dem_terrain_lookup,
        )

    db.mutate_grid("demo", "creek-fire-demo", _advance)

    # 3. Update demo state
    _bayesian_demo["current_step"] = step
    all_complete = (step + 1) >= len(steps)
    if all_complete:
        _bayesian_demo["active"] = False

    # 4. Export the Bayesian grid state for the frontend
    # Use a dynamic contour level: 40% of max probability, min 0.3 so
    # the contour is visible even when the fire hasn't peaked yet
    demo_entry = db.get_grid_entry("demo", "creek-fire-demo")
    demo_grid = demo_entry["grid"] if demo_entry else None
    max_p = demo_grid.get_statistics()["max_p"] if demo_grid else 0.0
    dynamic_contour = max(0.3, min(0.7, max_p * 0.65))
    # The read path is read-only now; the explicit predict above is what
    # advances the grid, and the in-memory extrapolation gate (~10s) won't
    # double-predict right after it.
    grid_data = _grid_to_json(threshold=0.02, contour_level=dynamic_contour, demo=True)

    return jsonify({
        "status": "step",
        "step": step,
        "total_steps": len(steps),
        "step_label": step_data["label"],
        "step_description": step_data["description"],
        "wind_speed": step_data["wind_speed"],
        "wind_dir_deg": step_data["wind_dir_deg"],
        "hotspots": step_data["hotspots"],
        "perimeter": step_data["perimeter"],
        "all_complete": all_complete,
        **grid_data,
    }), 200


@app.route("/api/bayesian-demo/reset", methods=["POST"])
def bayesian_demo_reset():
    """Reset the Bayesian historic demo and clear the grid."""
    global _bayesian_demo
    _bayesian_demo = None
    db.delete_grid("demo", "creek-fire-demo")
    return jsonify({"status": "reset", "message": "Bayesian demo reset."}), 200


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("🔥 WildFrame — Wildfire Detection Prototype")
    print(f"   Listening on http://localhost:4141")
    print(f"   Cluster radius: {CLUSTER_RADIUS_M}m | Time window: {CLUSTER_TIME_WINDOW_MINUTES}min")
    # debug=True keeps the useful traceback pages for development, but it
    # also enables the Werkzeug debugger console — an RCE risk if the port
    # is ever exposed. Production runs gunicorn; this dev entry point stays
    # debug-off unless WILDFRAME_DEBUG=1 is explicitly set.
    debug = os.environ.get("WILDFRAME_DEBUG", "0") == "1"
    # use_reloader=False prevents the reloader from crashing on file edits.
    app.run(host="0.0.0.0", port=4141, debug=debug, use_reloader=False)