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

import json
import math
import os
import random
import threading
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from triangulation import triangulate, triangulate_cluster
from bayesian_filter import (
    BayesianFireGrid, Evidence, auto_grid, auto_grid_size, seed_from_reports,
    compute_road_risk,
    DEFAULT_CELL_SIZE_M,
)

from fire_vision import scan_photo
import nasa_firms

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
UPLOAD_DIR = Path("uploads")
DATA_DIR = Path("data")
REPORTS_FILE = DATA_DIR / "reports.json"
STATIC_DIR = Path("static")

CLUSTER_RADIUS_M = 500.0          # metres — group reports within this distance
CLUSTER_TIME_WINDOW_MINUTES = 120 # 2 hours
ACTIVE_REPORT_HOURS = 48          # keep reports visible on map for 48h

# A Bayesian grid models ONE fire's local spread. GRID_MATCH_RADIUS_M is how
# close a cluster's centroid must be to an existing grid's tracked centroid
# to be considered "the same fire" (reuse that grid) rather than a distinct
# fire (spin up a new, independently-sized grid). Comfortably larger than
# CLUSTER_RADIUS_M + the grid's own margin, comfortably smaller than the gap
# between genuinely separate fires.
GRID_MATCH_RADIUS_M = 10000.0

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}

# Allowed report source types
SOURCE_TYPES = {"citizen", "NASA", "Sentinel", "CCTV", "drone", "IoT", "ranger", "emergency services"}

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

# Admin secret for dashboard access.
# Set env var WILDFRAME_ADMIN_SECRET or use the default (change in production!).
ADMIN_SECRET = os.environ.get("WILDFRAME_ADMIN_SECRET", "wildframe-admin")

UPLOAD_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
if not REPORTS_FILE.exists():
    REPORTS_FILE.write_text("[]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _allowed_file(name: str) -> bool:
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _load_reports() -> list[dict]:
    try:
        return json.loads(REPORTS_FILE.read_text())
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def _save_reports(reports: list[dict]) -> None:
    REPORTS_FILE.write_text(json.dumps(reports, indent=2, default=str))


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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/api/reports", methods=["GET"])
def list_reports():
    reports = _load_reports()
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=ACTIVE_REPORT_HOURS)
    active = [r for r in reports if _parse_ts(r.get("captured_at", "")) >= cutoff]
    return jsonify({"reports": active, "count": len(active)})


@app.route("/api/reports", methods=["POST"])
def create_report():
    """Accept multipart upload with photo + GPS data."""
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
    # If the AI is confident there's no fire, delete the photo and reject
    # the upload. If the scan errors out (no API key, server down, etc.),
    # we still allow the report through — better a false alarm than a
    # missed fire.
    ai_verdict = None
    try:
        ai_result = scan_photo(filepath)
        ai_verdict = ai_result["verdict"]

        if ai_verdict == "nothing":
            # AI says no fire — delete photo and return early
            filepath.unlink(missing_ok=True)
            print(f"[AI-vision] NOTHING detected in {filename} — photo discarded")
            return jsonify({
                "accepted": False,
                "reason": "AI analysis detected no fire or smoke in the photo.",
                "ai_analysis": {
                    "verdict": "nothing",
                    "confidence": ai_result["confidence"],
                    "fire_confidence": ai_result["fire_confidence"],
                    "smoke_confidence": ai_result["smoke_confidence"],
                    "detection_count": ai_result["detection_count"],
                    "model": ai_result["model"],
                    "error": ai_result["error"],
                },
            }), 200

        if ai_verdict != "error":
            print(f"[AI-vision] {ai_verdict.upper()} detected "
                  f"(confidence={ai_result['confidence']:.2f}, "
                  f"fire={ai_result['fire_confidence']:.2f}, "
                  f"smoke={ai_result['smoke_confidence']:.2f}) — creating report")

    except Exception as exc:
        print(f"[AI-vision] Scan failed: {exc} — proceeding with report anyway")
        ai_result = None

    # --- Build report (only reached if fire/smoke detected or scan errored) ---
    report = {
        "id": uuid.uuid4().hex,
        "lat": lat,
        "lon": lon,
        "photo_url": f"/uploads/{filename}",
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

    reports = _load_reports()
    reports.append(report)
    _save_reports(reports)

    # --- Recompute clusters ---
    clusters = _compute_clusters(reports)

    return jsonify({"report": report, "clusters": clusters, "cluster_count": len(clusters)}), 201


@app.route("/api/clusters", methods=["GET"])
def get_clusters():
    reports = _load_reports()
    clusters = _compute_clusters(reports)
    return jsonify({"clusters": clusters, "count": len(clusters)})


@app.route("/api/reports/<report_id>/status", methods=["PUT"])
def update_status(report_id: str):
    data = request.get_json(silent=True) or {}
    new_status = data.get("status")
    if new_status not in ("pending", "confirmed", "rejected"):
        return jsonify({"error": "Invalid status. Use: pending, confirmed, rejected"}), 400

    reports = _load_reports()
    for r in reports:
        if r["id"] == report_id:
            r["status"] = new_status
            _save_reports(reports)
            clusters = _compute_clusters(reports)
            return jsonify({"report": r, "clusters": clusters, "cluster_count": len(clusters)})
    return jsonify({"error": "Report not found"}), 404


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
    reports = _load_reports()
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
        reports.append(report)
        new_ids.append(report["id"])

    _save_reports(reports)
    clusters = _compute_clusters(reports)

    # Build a fresh Bayesian grid sized for Greece only — not merged with
    # any existing reports from other regions.
    _seed_new_grid(reports[-len(greece_hotspots):])

    return jsonify({
        "message": f"Seeded {len(greece_hotspots)} test reports across Greece",
        "seeded_count": len(greece_hotspots),
        "total_reports": len(reports),
        "clusters": clusters,
        "cluster_count": len(clusters),
    }), 201


@app.route("/uploads/<filename>")
def uploaded_file(filename: str):
    return send_from_directory(str(UPLOAD_DIR), filename)


@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


# ---------------------------------------------------------------------------
# Admin Helpers & Routes
# ---------------------------------------------------------------------------

def _require_admin():
    """Check the X-Admin-Secret header against the configured secret."""
    auth = request.headers.get("X-Admin-Secret", "")
    if auth != ADMIN_SECRET:
        return False
    return True


@app.route("/admin")
def admin_page():
    return send_from_directory(str(STATIC_DIR), "admin.html")


@app.route("/api/admin/pending", methods=["GET"])
def admin_list_pending():
    if not _require_admin():
        return jsonify({"error": "Unauthorized"}), 401
    reports = _load_reports()
    pending = [r for r in reports if r.get("status") == "pending"]
    return jsonify({"reports": pending, "count": len(pending)})


@app.route("/api/admin/accept/<report_id>", methods=["POST"])
def admin_accept(report_id: str):
    if not _require_admin():
        return jsonify({"error": "Unauthorized"}), 401
    reports = _load_reports()
    for r in reports:
        if r["id"] == report_id:
            r["status"] = "confirmed"
            _save_reports(reports)
            clusters = _compute_clusters(reports)
            # Feed into Bayesian grid
            _feed_reports_into_grid([r])
            return jsonify({"report": r, "clusters": clusters,
                            "cluster_count": len(clusters)})
    return jsonify({"error": "Report not found"}), 404


@app.route("/api/admin/accept-all", methods=["POST"])
def admin_accept_all():
    """Accept all pending reports at once."""
    if not _require_admin():
        return jsonify({"error": "Unauthorized"}), 401
    reports = _load_reports()
    count = 0
    accepted_reports = []
    for r in reports:
        if r.get("status") == "pending":
            r["status"] = "confirmed"
            accepted_reports.append(r)
            count += 1
    _save_reports(reports)
    clusters = _compute_clusters(reports)
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
    reports = _load_reports()
    for i, r in enumerate(reports):
        if r["id"] == report_id:
            # Delete the uploaded photo from disk
            photo_url = r.get("photo_url", "")
            if photo_url:
                # photo_url looks like "/uploads/filename"
                p = Path(photo_url.lstrip("/"))
                try:
                    if p.exists():
                        p.unlink()
                        print(f"[admin] Deleted photo: {p}")
                except Exception as exc:
                    print(f"[admin] Failed to delete photo {p}: {exc}")

            # Remove the report from the list
            reports.pop(i)
            _save_reports(reports)
            clusters = _compute_clusters(reports)
            return jsonify({"success": True,
                            "clusters": clusters,
                            "cluster_count": len(clusters)})
    return jsonify({"error": "Report not found"}), 404


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

    existing = _load_reports()
    existing.extend(reports_list)
    _save_reports(existing)
    clusters = _compute_clusters(existing)

    # Build a fresh Bayesian grid sized for Yosemite only — not merged with
    # any existing reports from other regions.
    _seed_new_grid(reports_list)

    return jsonify({
        "message": f"Seeded {len(yosemite_hotspots)} test reports across Yosemite National Park",
        "seeded_count": len(yosemite_hotspots),
        "total_reports": len(existing),
        "clusters": clusters,
        "cluster_count": len(clusters),
        "centroid_lat": 37.745,
        "centroid_lon": -119.593,
    }), 201


# ---------------------------------------------------------------------------
# Live Demo — Progressive Triangulation Simulation
# ---------------------------------------------------------------------------

# In-memory state for the live demo
_live_demo: Optional[dict] = None

# In-memory Bayesian probability grid
_bayesian_grids: dict[str, dict] = {}   # key -> {"grid": BayesianFireGrid, "centroid_lat": float, "centroid_lon": float}
_next_grid_id = 1


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
# ---------------------------------------------------------------------------

def _find_or_create_grid_for_cluster(cluster: dict) -> BayesianFireGrid:
    """Return the grid tracking this cluster's fire, creating one if needed."""
    global _next_grid_id

    clat, clon = cluster["centroid_lat"], cluster["centroid_lon"]

    best_key, best_dist = None, float("inf")
    for key, entry in _bayesian_grids.items():
        d = _haversine(clat, clon, entry["centroid_lat"], entry["centroid_lon"])
        if d < best_dist:
            best_dist, best_key = d, key

    if best_key is not None and best_dist <= GRID_MATCH_RADIUS_M:
        entry = _bayesian_grids[best_key]
        # Keep the tracked centroid current as the cluster grows/shifts
        entry["centroid_lat"], entry["centroid_lon"] = clat, clon
        return entry["grid"]

    # No existing grid tracks a fire near here — spin up a fresh one sized
    # to just this cluster (so it gets a fine cell size, not a continent-
    # spanning one)
    lats = [p[0] for p in cluster["points"]]
    lons = [p[1] for p in cluster["points"]]
    sizing = auto_grid_size(lats, lons) or {
        "center_lat": clat, "center_lon": clon,
        "cell_size_m": DEFAULT_CELL_SIZE_M, "nx": 40, "ny": 40,
    }
    grid = BayesianFireGrid(
        center_lat=sizing["center_lat"], center_lon=sizing["center_lon"],
        cell_size_m=sizing["cell_size_m"], nx=sizing["nx"], ny=sizing["ny"],
    )

    key = f"grid-{_next_grid_id}"
    _next_grid_id += 1
    _bayesian_grids[key] = {
        "grid": grid, "centroid_lat": clat, "centroid_lon": clon,
        "wind_speed": 3.0, "wind_dir_deg": 270.0,
    }
    return grid


def _get_grid_entry(grid: BayesianFireGrid) -> Optional[dict]:
    """Return the registry entry dict for a given grid object."""
    for entry in _bayesian_grids.values():
        if entry["grid"] is grid:
            return entry
    return None


def _sync_grids_from_clusters(reports: list[dict], clusters: list[dict]) -> None:
    """Ensure each cluster has a backing grid, seeded with its own reports."""
    for cluster in clusters:
        grid = _find_or_create_grid_for_cluster(cluster)
        entry = _get_grid_entry(grid)
        wind_dir = entry["wind_dir_deg"] if entry else None
        cluster_reports = [r for r in reports if r["id"] in cluster["report_ids"]]
        seed_from_reports(grid, cluster_reports, clusters, wind_dir_deg=wind_dir)


def _init_bayesian_grid() -> None:
    """(Re)build all per-cluster grids from scratch from confirmed reports."""
    global _bayesian_grids
    _bayesian_grids = {}
    reports = _load_reports()
    clusters = _compute_clusters(reports)
    _sync_grids_from_clusters(reports, clusters)


def _ensure_bayesian_grids() -> dict[str, dict]:
    """Return the grid registry, building it if it doesn't exist yet."""
    if not _bayesian_grids:
        _init_bayesian_grid()
    return _bayesian_grids


def _seed_new_grid(seed_reports: list[dict]) -> None:
    """
    Build fresh grid(s) scoped ONLY to the given batch (one per fire cluster
    within it), replacing whatever grids currently exist. Used by seed
    endpoints so a demo dataset doesn't merge with unrelated existing fires.
    """
    global _bayesian_grids
    _bayesian_grids = {}

    all_reports = _load_reports()
    clusters = _compute_clusters(all_reports)

    seed_ids = {r["id"] for r in seed_reports}
    relevant_clusters = [c for c in clusters if set(c["report_ids"]) & seed_ids]

    _sync_grids_from_clusters(seed_reports, relevant_clusters)


def _feed_reports_into_grid(reports: list[dict]) -> None:
    """Feed newly confirmed reports into their per-cluster grid(s)."""
    if not _bayesian_grids:
        _init_bayesian_grid()
        return

    all_reports = _load_reports()
    clusters = _compute_clusters(all_reports)
    new_ids = {r["id"] for r in reports}
    relevant_clusters = [c for c in clusters if set(c["report_ids"]) & new_ids]
    _sync_grids_from_clusters(all_reports, relevant_clusters)


def _find_or_create_grid_for_point(lat: float, lon: float) -> BayesianFireGrid:
    """Like _find_or_create_grid_for_cluster, but for a single lat/lon (used
    by the manual evidence-injection endpoint)."""
    fake_cluster = {
        "centroid_lat": lat, "centroid_lon": lon,
        "points": [[lat, lon]],
    }
    return _find_or_create_grid_for_cluster(fake_cluster)


def _grid_to_json(
    threshold: float = 0.02,
    contour_level: float = 0.6,
    auto_predict: bool = True,
) -> dict:
    """
    Get the current state of ALL Bayesian grids (one per fire cluster),
    optionally running an automatic predict step on each for any elapsed
    time since its last prediction.

    Parameters
    ----------
    auto_predict : bool
        If True (default), advance each grid by elapsed time.  Set to False
        when the caller has already run an explicit predict (e.g., during
        Bayesian demo steps, to avoid double-predicting).
    """
    registry = _ensure_bayesian_grids()

    grids_out = []
    for key, entry in registry.items():
        grid = entry["grid"]

        if auto_predict:
            wind_speed = entry.get("wind_speed", 3.0)
            wind_dir_deg = entry.get("wind_dir_deg", 270.0)
            now = datetime.now(timezone.utc)
            if grid.last_predict_time > 0:
                elapsed = (now.timestamp() - grid.last_predict_time)
                # Gate is intentionally smaller than the frontend's 5s poll
                # interval, so nearly every poll advances the grid a little
                # instead of sitting frozen for 30s and then jumping.
                if elapsed > 2:
                    dt = min(elapsed, 600.0)
                    grid.predict(
                        dt=dt,
                        wind_speed=wind_speed,
                        wind_dir_deg=wind_dir_deg,
                    )
            else:
                grid.predict(dt=60.0, wind_speed=wind_speed, wind_dir_deg=wind_dir_deg)

        grids_out.append({
            "id": key,
            "state": grid.export_state(threshold=threshold),
            "contour": grid.export_contour(level=contour_level),
            "statistics": grid.get_statistics(),
            "wind_speed": entry.get("wind_speed", 3.0),
            "wind_dir_deg": entry.get("wind_dir_deg", 270.0),
        })

    return {"grids": grids_out}


@app.route("/api/bayesian/state", methods=["GET"])
def bayesian_get_state():
    """
    Get the current Bayesian probability grid state.

    Query params:
      - threshold: minimum probability to include (default 0.02)
      - contour:   contour level (default 0.6)

    Automatically runs a predict step proportional to the time elapsed
    since the last prediction, so the fire spread animates in real time.
    """
    threshold = request.args.get("threshold", 0.02, type=float)
    contour_level = request.args.get("contour", 0.6, type=float)

    try:
        data = _grid_to_json(threshold=threshold, contour_level=contour_level)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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

# In-memory cache (mirrors the on-disk file). Keyed by cache_key.
_osm_road_cache: dict[str, dict] = {}
_osm_cache_lock = threading.Lock()

# Track request timestamps per cache_key to enforce our own rate limit
_osm_request_timestamps: dict[str, list[float]] = {}

# How many Overpass HTTP requests we allow per cache_key per minute before
# falling back to cached/stale data (protects both Overpass and our own
# error handling).
_OSM_REQ_BUDGET_PER_MIN = 20  # accounts for multiple fire clusters × endpoint fallbacks


def _load_osm_cache() -> None:
    """Load OSM road cache from disk on startup."""
    global _osm_road_cache
    if OSM_CACHE_FILE.exists():
        try:
            data = json.loads(OSM_CACHE_FILE.read_text())
            if isinstance(data, dict):
                # Convert string keys to proper types if needed
                _osm_road_cache = data
                count = sum(1 for v in data.values() if v.get("segments"))
                print(f"[road-cache] Loaded {count} cached locations from disk")
        except (json.JSONDecodeError, OSError) as e:
            print(f"[road-cache] Failed to load disk cache: {e}")


def _save_osm_cache() -> None:
    """Persist OSM road cache to disk. Called after each successful fetch."""
    try:
        OSM_CACHE_FILE.write_text(json.dumps(_osm_road_cache, indent=2, default=str))
    except OSError as e:
        print(f"[road-cache] Failed to save disk cache: {e}")


# Load persisted cache at module import time
_load_osm_cache()


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
    disk-persistent caching so each location is fetched at most once.

    **Caching**: roads don't move, so once fetched, data is saved to
    ``data/osm_road_cache.json`` forever. Uses fuzzy matching: if the
    exact contour centroid isn't cached, any cached centroid within
    1.5 km with the same radius is reused (the contour shifts slightly
    on every poll as the fire spreads).

    **Endpoint order**: Overpass mirrors first, then the main OSM API
    (returns XML, more reliable). Each endpoint is tried until one
    succeeds or all fail.

    **Rate limiting**: max ``_OSM_REQ_BUDGET_PER_MIN`` (20) requests per
    cache key per minute. Fuzzy-matched lookups don't consume budget.

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

    # ---- Helper: read cache entry under lock ----
    def _get_cache():
        with _osm_cache_lock:
            return _osm_road_cache.get(cache_key)

    # ---- Helper: return stale data with a log message, or raise ----
    def _serve_stale_or_raise(
        error_msg: str,
        contexts: Optional[list[str]] = None,
    ) -> list[list[tuple[float, float]]]:
        entry = _get_cache()
        if entry and entry.get("segments"):
            age_s = time.time() - entry.get("stored_at", 0)
            ctx = f" ({'; '.join(contexts)}) " if contexts else " "
            print(f"[road-cache]{ctx}{error_msg} — serving stale data "
                  f"({len(entry['segments'])} segments, {age_s:.0f}s old)")
            return entry["segments"]
        raise Exception(error_msg)

    # ---- Helper: parse a cache key into (lat, lon, radius_km) ----
    def _parse_cache_key(k: str) -> Optional[tuple[float, float, float]]:
        parts = k.split(",")
        if len(parts) == 3:
            try:
                return float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
                pass
        return None

    # ---- Check in-memory / disk cache (exact match) ----
    # Since roads don't move, if we have segments for this key they're
    # always valid regardless of age.
    with _osm_cache_lock:
        entry = _osm_road_cache.get(cache_key)
        if entry and entry.get("segments"):
            print(f"[road-cache] HIT for {cache_key} ({len(entry['segments'])} segments)")
            return entry["segments"]

    # ---- Fuzzy cache lookup: same radius, centroid within 1km ----
    # The contour centroid shifts slightly on every poll as the fire
    # spreads, so exact key match may miss even though we have data
    # for a nearby location. Scan all cache entries with the same
    # radius and find the closest one within FUZZY_MATCH_DISTANCE_M.
    FUZZY_MATCH_DISTANCE_M = 1500.0
    with _osm_cache_lock:
        for k, v in _osm_road_cache.items():
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
                _osm_road_cache[cache_key] = v
                return v["segments"]

    # ---- Enforce our own client-side rate budget ----
    if not _osm_enforce_client_rate_limit(cache_key):
        cached = _get_cache()
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

    # Optimised query: only major road types (not footpaths, service roads,
    # tracks, etc.) to minimise payload and timeout risk.
    query = f"""
    [out:json][timeout:8];
    (
      way["highway"~"^(motorway|trunk|primary|secondary|tertiary)(_link)?$"](around:{radius_m},{center_lat},{center_lon});
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
                # Overpass API — POST the query as form data. Overpass's own
                # docs recommend POST for anything beyond trivial queries;
                # putting the query on the URL (GET) risks length limits and
                # WAF/proxy rejections (we were seeing 406s from this).
                post_data = urllib.parse.urlencode({"data": query}).encode("utf-8")
                url = ep_url
            else:
                # Main OSM API — use bbox query. This endpoint returns ALL
                # data in the bbox (buildings, POIs, everything — not just
                # roads) and hard-caps at 50,000 nodes, so we deliberately
                # use a smaller radius here than the caller asked for to
                # avoid tripping that limit (this is a last-resort fallback,
                # not an equivalent replacement for Overpass).
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

                segment_dicts: list[dict] = []
                for element in data.get("elements", []):
                    if element["type"] == "way" and "nodes" in element:
                        seg = []
                        for node_id in element["nodes"]:
                            if node_id in nodes:
                                seg.append(nodes[node_id])
                        if len(seg) >= 2:
                            highway_type = element.get("tags", {}).get("highway", "unknown")
                            name = element.get("tags", {}).get("name", "")
                            segment_dicts.append({"coords": seg, "highway": highway_type, "name": name})

                result = [s["coords"] for s in segment_dicts]
            else:
                # Parse XML response from main OSM API
                result = _parse_osm_xml_roads(raw)

            # Store in cache (in-memory + disk)
            with _osm_cache_lock:
                _osm_road_cache[cache_key] = {
                    "segments": result,
                    "stored_at": time.time(),
                }
            _save_osm_cache()

            print(f"[road-cache] STORED {cache_key} ({len(result)} segments, endpoint={ep_name})")
            return result

        except urllib.error.HTTPError as e:
            ctx = f"{ep_name}: HTTP {e.code}"
            contexts_attempted.append(ctx)
            print(f"[road-cache] {ctx} for {cache_key}")

            # A 429 means THIS endpoint is rate-limiting us, not that every
            # mirror is — move on and try the remaining endpoints instead
            # of abandoning the whole chain.
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

    # ---- All endpoints exhausted — serve stale or raise ----
    return _serve_stale_or_raise(
        "All OSM API endpoints failed.",
        contexts=contexts_attempted,
    )


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

    registry = _ensure_bayesian_grids()
    if not registry:
        return jsonify({"error": "No active fire grids. Seed some data first or enable the Bayesian overlay."}), 400

    all_features: list[dict] = []
    grids_without_contour = 0   # fire hasn't crossed contour_level yet
    grids_without_roads = 0     # established fire edge, but 0 roads nearby
    max_peak_probability = 0.0  # highest peak prob among no-contour grids —
                                 # helps distinguish "close" from "nowhere near"

    for key, entry in registry.items():
        if target_grid != "all" and key != target_grid:
            continue

        grid = entry["grid"]
        wind_speed = entry.get("wind_speed", 3.0)
        wind_dir = entry.get("wind_dir_deg", 270.0)

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
                return jsonify({
                    "error": f"Failed to fetch road data from OpenStreetMap: {exc}",
                }), 502

        # Compute risk for this grid's roads
        risk_results = compute_road_risk(
            grid, segments, wind_speed, wind_dir,
            contour_level=contour_level, contour=contour,
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
            "grids_assessed": len(registry),
            "contour_level": contour_level,
            "radius_km": radius_km,
            "grids_without_contour": grids_without_contour,
            "grids_without_roads": grids_without_roads,
            "empty_reason": empty_reason,
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

    registry = _ensure_bayesian_grids()

    if grid_id:
        if grid_id not in registry:
            return jsonify({"error": f"Grid '{grid_id}' not found"}), 404
        entry = registry[grid_id]
        entry["grid"].predict(
            dt=dt, wind_speed=wind_speed, wind_dir_deg=wind_dir, slope_pct=slope_pct,
        )
        # Persist the new wind for this grid (so auto-predict inherits it)
        entry["wind_speed"] = wind_speed
        entry["wind_dir_deg"] = wind_dir
        message = f"Predicted {dt:.0f}s ahead for grid '{grid_id}' (wind now {wind_speed} m/s from {wind_dir}°)."
    else:
        for entry in registry.values():
            entry["grid"].predict(
                dt=dt, wind_speed=wind_speed, wind_dir_deg=wind_dir, slope_pct=slope_pct,
            )
        message = f"Predicted {dt:.0f}s ahead for all {len(registry)} grid(s). Per-grid wind NOT overwritten."

    return jsonify({"status": "ok", "message": message})


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

    evidence = Evidence(
        lat=lat,
        lon=lon,
        log_likelihood_ratio=log_lr,
        spatial_radius_m=data.get("spatial_radius_m", 0.0),
        source=data.get("source", "api"),
    )

    grid = _find_or_create_grid_for_point(lat, lon)
    grid.update(evidence)

    return jsonify({"status": "ok", "evidence": data, "message": "Evidence fused."})


@app.route("/api/bayesian/reset", methods=["POST"])
def bayesian_reset():
    """Reset all Bayesian grids (clears the whole registry; grids are
    rebuilt on demand from confirmed reports)."""
    global _bayesian_grids
    _bayesian_grids = {}
    return jsonify({"status": "ok", "message": "Grid(s) reset."})


# ---------------------------------------------------------------------------
# Satellite Hotspot Simulation — bridges Gap 1 (satellite path for live grids)
# ---------------------------------------------------------------------------

_satellite_poller_active = False
_satellite_poller_thread: Optional[threading.Thread] = None


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
    # Lazily rebuild the registry from persisted confirmed reports if this
    # is a fresh process (e.g. right after the debug reloader restarted)
    # and nobody's hit /api/bayesian/state yet to build it. Without this,
    # a satellite pass right after a restart always reports zero grids,
    # even though confirmed reports — and their fires — are sitting right
    # there on disk.
    registry = _ensure_bayesian_grids()
    if not registry:
        return {"injected": 0, "grids_hit": 0, "grids_considered": 0}

    total_injected = 0
    grids_hit = 0
    grid_ids_hit = []

    all_keys = list(registry.items())
    for key, entry in all_keys:
        grid = entry["grid"]
        clat, clon = entry["centroid_lat"], entry["centroid_lon"]

        if random.random() >= probability:
            continue

        num = random.randint(min_hotspots, max_hotspots)
        for _ in range(num):
            # Jitter within ~jitter_km of the centroid
            jlat = clat + random.uniform(
                -jitter_km / 111.0,
                jitter_km / 111.0,
            )
            jlon = clon + random.uniform(
                -jitter_km / (111.0 * math.cos(math.radians(clat))),
                jitter_km / (111.0 * math.cos(math.radians(clat))),
            )
            evidence = Evidence.satellite_hotspot(lat=jlat, lon=jlon)
            grid.update(evidence)

        total_injected += num
        grids_hit += 1
        grid_ids_hit.append(key)

    if guarantee_hit and grids_hit == 0 and all_keys:
        # Every grid missed its coin flip — force one, chosen at random,
        # so a deliberate manual trigger always shows something happened.
        key, entry = random.choice(all_keys)
        grid = entry["grid"]
        clat, clon = entry["centroid_lat"], entry["centroid_lon"]
        num = random.randint(min_hotspots, max_hotspots)
        for _ in range(num):
            jlat = clat + random.uniform(-jitter_km / 111.0, jitter_km / 111.0)
            jlon = clon + random.uniform(
                -jitter_km / (111.0 * math.cos(math.radians(clat))),
                jitter_km / (111.0 * math.cos(math.radians(clat))),
            )
            grid.update(Evidence.satellite_hotspot(lat=jlat, lon=jlon))
        total_injected += num
        grids_hit += 1
        grid_ids_hit.append(key)

    return {
        "injected": total_injected,
        "grids_hit": grids_hit,
        "grids_considered": len(registry),
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
    Start a lightweight background thread that periodically simulates
    satellite passes, giving the live pipeline a continuous stream of
    satellite corroboration evidence.

    JSON body (all optional):
      {
        "interval_s": 30,      // seconds between simulated passes
        "probability": 0.6,    // pass probability per grid
        "min_hotspots": 1,
        "max_hotspots": 3
      }
    """
    global _satellite_poller_active, _satellite_poller_thread

    if _satellite_poller_active:
        return jsonify({"status": "ok", "message": "Poller already running."}), 200

    data = request.get_json(silent=True) or {}
    interval = data.get("interval_s", 30.0)
    probability = data.get("probability", 0.6)
    min_hotspots = data.get("min_hotspots", 1)
    max_hotspots = data.get("max_hotspots", 3)

    _satellite_poller_active = True

    def _poller_loop():
        while _satellite_poller_active:
            time.sleep(interval)
            if not _satellite_poller_active:
                break
            try:
                result = _simulate_satellite_pass(
                    probability=probability,
                    min_hotspots=min_hotspots,
                    max_hotspots=max_hotspots,
                )
                if result["injected"] > 0:
                    print(
                        f"[satellite-poller] Injected {result['injected']} hotspot(s) "
                        f"across {result['grids_hit']} grid(s)."
                    )
            except Exception as exc:
                print(f"[satellite-poller] Error: {exc}")

    _satellite_poller_thread = threading.Thread(target=_poller_loop, daemon=True)
    _satellite_poller_thread.start()

    return jsonify({
        "status": "started",
        "message": f"Satellite poller started (interval={interval}s, p={probability}).",
        "interval_s": interval,
        "probability": probability,
    })


@app.route("/api/satellite/poller/stop", methods=["POST"])
def satellite_poller_stop():
    """Stop the background satellite simulation poller."""
    global _satellite_poller_active, _satellite_poller_thread

    was_active = _satellite_poller_active
    _satellite_poller_active = False

    if _satellite_poller_thread and _satellite_poller_thread.is_alive():
        _satellite_poller_thread.join(timeout=5.0)

    _satellite_poller_thread = None

    return jsonify({
        "status": "stopped" if was_active else "idle",
        "message": "Satellite poller stopped." if was_active else "Poller was not running.",
    })


# ===================================================================
# NASA FIRMS Real Satellite Data Integration
# ===================================================================
# Fetches actual VIIRS / MODIS hotspot data from the NASA FIRMS API and
# feeds it into the Bayesian fire grids as real satellite evidence.

_firms_poller_active = False
_firms_poller_thread: Optional[threading.Thread] = None


# Cluster radius for grouping nearby FIRMS hotspot points into a single fire.
# VIIRS has 375m resolution — hotspots within 1km are almost certainly the
# same fire.
_FIRMS_CLUSTER_RADIUS_M = 1000.0


def _fetch_nasa_firms_pass(
    day_range: int = 1,
    min_confidence: str = "nominal",
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

    Parameters
    ----------
    day_range : int
        Days of FIRMS data to query (1–5).
    min_confidence : str
        Minimum FIRMS confidence: "low", "nominal", or "high".

    Returns
    -------
    dict with keys:
        "injected"         : int — total evidence items injected across all grids
        "grids_hit"        : int — grids that received at least one hotspot
        "grids_considered" : int — total grids (existing + newly created)
        "firms_hotspots"   : int — total hotspots returned by FIRMS
        "new_grids"        : int — grids auto-created for previously-unknown fires
        "api_error"        : str | None — error message if the API call failed
    """
    api_key = nasa_firms._get_api_key()
    if not api_key:
        return {
            "injected": 0, "grids_hit": 0, "grids_considered": 0,
            "firms_hotspots": 0, "new_grids": 0,
            "api_error": "NASA_FIRMS_API_KEY not set",
        }

    # Fetch ALL global hotspots (not just near existing grids)
    try:
        all_hotspots = nasa_firms.fetch_global_fires(
            api_key=api_key,
            day_range=day_range,
            min_confidence=min_confidence,
        )
    except (ConnectionError, ValueError) as exc:
        logger.warning("[firms] Global FIRMS fetch failed: %s", exc)
        return {
            "injected": 0, "grids_hit": 0, "grids_considered": 0,
            "firms_hotspots": 0, "new_grids": 0,
            "api_error": str(exc),
        }

    if not all_hotspots:
        return {
            "injected": 0, "grids_hit": 0, "grids_considered": 0,
            "firms_hotspots": 0, "new_grids": 0,
        }

    # --- Cluster hotspots into candidate fires ---
    # Simple single-pass clustering: hotspots within _FIRMS_CLUSTER_RADIUS_M
    # of each other are grouped as one fire.
    clusters: list[dict] = []  # each: {"centroid_lat": ..., "centroid_lon": ..., "points": [[lat,lon],...], "hotspots": [...]}
    for hs in all_hotspots:
        hlats = [hs.latitude, hs.longitude]
        best_idx = -1
        best_dist = float("inf")

        for i, c in enumerate(clusters):
            d = _haversine(hs.latitude, hs.longitude, c["centroid_lat"], c["centroid_lon"])
            if d < best_dist:
                best_dist = d
                best_idx = i

        if best_idx >= 0 and best_dist <= _FIRMS_CLUSTER_RADIUS_M:
            c = clusters[best_idx]
            c["points"].append([hs.latitude, hs.longitude])
            c["hotspots"].append(hs)
            # Running centroid average
            n = len(c["points"])
            c["centroid_lat"] = (c["centroid_lat"] * (n - 1) + hs.latitude) / n
            c["centroid_lon"] = (c["centroid_lon"] * (n - 1) + hs.longitude) / n
        else:
            clusters.append({
                "centroid_lat": hs.latitude,
                "centroid_lon": hs.longitude,
                "points": [[hs.latitude, hs.longitude]],
                "hotspots": [hs],
            })

    logger.info(
        "[firms] Clustered %d hotspots into %d candidate fires",
        len(all_hotspots), len(clusters),
    )

    # --- Ensure grids exist, inject evidence ---
    # Lazily initialise the registry so _find_or_create_grid_for_cluster works
    registry = _ensure_bayesian_grids()

    total_injected = 0
    grids_hit = 0
    new_grids = 0
    total_hotspots = len(all_hotspots)
    pre_count = len(registry)

    for c in clusters:
        clat, clon = c["centroid_lat"], c["centroid_lon"]

        # Find or create a Bayesian grid for this fire cluster
        grid = _find_or_create_grid_for_cluster(c)

        # Inject each hotspot as satellite evidence
        num_injected = 0
        for hs in c["hotspots"]:
            evidence = Evidence.satellite_hotspot(lat=hs.latitude, lon=hs.longitude)
            grid.update(evidence)
            num_injected += 1

        total_injected += num_injected
        grids_hit += 1

    post_count = len(_ensure_bayesian_grids())
    new_grids = max(0, post_count - pre_count)

    logger.info(
        "[firms] Injected %d hotspots across %d grids (%d new)",
        total_injected, grids_hit, new_grids,
    )

    return {
        "injected": total_injected,
        "grids_hit": grids_hit,
        "grids_considered": post_count,
        "firms_hotspots": total_hotspots,
        "new_grids": new_grids,
    }


@app.route("/api/satellite/firms-fetch", methods=["POST"])
def satellite_firms_fetch():
    """
    Manually trigger a global NASA FIRMS data fetch.

    Fetches all active fires worldwide from NASA FIRMS, auto-creates
    Bayesian grids for any newly-discovered fires, and injects hotspot
    evidence into existing grids.

    JSON body (all optional):
      {
        "day_range": 1,             // days of FIRMS data to query (1-5)
        "min_confidence": "nominal"  // minimum confidence: low/nominal/high
      }

    Returns a summary of how many grids were hit, evidence injected,
    and new grids auto-created.
    """
    data = request.get_json(silent=True) or {}
    result = _fetch_nasa_firms_pass(
        day_range=data.get("day_range", 1),
        min_confidence=data.get("min_confidence", "nominal"),
    )

    if result.get("api_error"):
        return jsonify({
            "error": result["api_error"],
            "injected": 0,
            "grids_hit": 0,
            "grids_considered": result["grids_considered"],
            "new_grids": 0,
        }), 400

    msg_parts = []
    if result["new_grids"] > 0:
        msg_parts.append(f"Auto-created {result['new_grids']} new grid(s) for previously unknown fires")
    if result["injected"] > 0:
        msg_parts.append(f"Injected {result['injected']} evidence items across {result['grids_hit']} grid(s)")
    if result["firms_hotspots"] > 0:
        msg_parts.append(f"from {result['firms_hotspots']} FIRMS hotspot(s)")
    if not msg_parts:
        msg_parts.append(f"No FIRMS hotspots found worldwide in the last {data.get('day_range', 1)} day(s)")

    return jsonify({
        **result,
        "message": ", ".join(msg_parts),
    })


@app.route("/api/satellite/firms-poller/start", methods=["POST"])
def satellite_firms_poller_start():
    """Start the real NASA FIRMS background poller (global, self-driving)."""
    global _firms_poller_active, _firms_poller_thread

    data = request.get_json(silent=True) or {}
    interval = data.get("interval_s", 600.0)  # default: 10 min
    day_range = data.get("day_range", 1)
    min_confidence = data.get("min_confidence", "nominal")

    # Validate the API key early
    if not nasa_firms._get_api_key():
        return jsonify({
            "status": "error",
            "error": "NASA_FIRMS_API_KEY environment variable is not set. "
                     "Set it to your free NASA FIRMS API key and restart.",
        }), 400

    _firms_poller_active = True

    def _firms_poller_loop():
        while _firms_poller_active:
            time.sleep(interval)
            if not _firms_poller_active:
                break
            try:
                result = _fetch_nasa_firms_pass(
                    day_range=day_range,
                    min_confidence=min_confidence,
                )
                if result.get("api_error"):
                    logger.warning(
                        "[firms-poller] API error: %s", result["api_error"]
                    )
                elif result["injected"] > 0:
                    logger.info(
                        "[firms-poller] Injected %d hotspot(s) across %d grid(s) "
                        "(%d new grids created).",
                        result["injected"], result["grids_hit"],
                        result.get("new_grids", 0),
                    )
                else:
                    logger.info(
                        "[firms-poller] No FIRMS hotspots found (worldwide, "
                        "last %d day(s), confidence=%s).",
                        day_range, min_confidence,
                    )
            except Exception as exc:
                logger.error("[firms-poller] Error: %s", exc)

    _firms_poller_thread = threading.Thread(target=_firms_poller_loop, daemon=True)
    _firms_poller_thread.start()

    return jsonify({
        "status": "started",
        "message": f"FIRMS poller started (interval={interval}s, confidence={min_confidence}).",
        "interval_s": interval,
    })


@app.route("/api/satellite/firms-poller/stop", methods=["POST"])
def satellite_firms_poller_stop():
    """Stop the real NASA FIRMS background poller."""
    global _firms_poller_active, _firms_poller_thread

    was_active = _firms_poller_active
    _firms_poller_active = False

    if _firms_poller_thread and _firms_poller_thread.is_alive():
        _firms_poller_thread.join(timeout=5.0)

    _firms_poller_thread = None

    return jsonify({
        "status": "stopped" if was_active else "idle",
        "message": "FIRMS poller stopped." if was_active else "FIRMS poller was not running.",
    })


@app.route("/api/satellite/poller/status", methods=["GET"])
def satellite_poller_status():
    """Check whether the satellite poller is currently running."""
    return jsonify({
        "active": _satellite_poller_active,
        "alive": bool(_satellite_poller_thread and _satellite_poller_thread.is_alive()),
        "grids_tracked": len(_ensure_bayesian_grids()),
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

    # Save to persistent store
    existing = _load_reports()
    existing.extend(reports_to_add)
    _save_reports(existing)

    # Update step counter
    _live_demo["current_step"] = step

    # Compute clusters and triangulation
    clusters = _compute_clusters(existing)

    # Feed new reports into Bayesian grid
    _feed_reports_into_grid(reports_to_add)

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
    global _live_demo, _bayesian_grids

    if _live_demo:
        demo_ids = set(r["id"] for r in _live_demo["reports"])
        existing = _load_reports()
        existing = [r for r in existing if r["id"] not in demo_ids]
        _save_reports(existing)

    _live_demo = None
    # Also reset the Bayesian grid registry so no stale evidence persists
    _bayesian_grids = {}

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
    global _bayesian_demo, _bayesian_grids

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

    # Register under a fixed key in the shared grid registry (rather than a
    # separate standalone variable) so it's picked up by _grid_to_json the
    # same way any other cluster's grid is.
    _bayesian_grids["creek-fire-demo"] = {
        "grid": demo_grid,
        "centroid_lat": scenario["fire_lat"],
        "centroid_lon": scenario["fire_lon"],
    }

    # Seed with first step's satellite hotspots
    for hs in start_step["hotspots"]:
        evidence = Evidence.satellite_hotspot(lat=hs[0], lon=hs[1])
        demo_grid.update(evidence)

    # Run initial small predict to spread from hotspots
    demo_grid.predict(
        dt=300.0,  # 5 minutes
        wind_speed=start_step["wind_speed"],
        wind_dir_deg=start_step["wind_dir_deg"],
    )

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
    global _bayesian_demo, _bayesian_grids

    if not _bayesian_demo or not _bayesian_demo.get("active"):
        return jsonify({"error": "No active Bayesian demo. Call /api/bayesian-demo/start first."}), 400

    demo_entry = _bayesian_grids.get("creek-fire-demo")
    if demo_entry is None:
        return jsonify({"error": "Demo grid missing. Call /api/bayesian-demo/start first."}), 400
    demo_grid = demo_entry["grid"]

    step = _bayesian_demo["current_step"] + 1
    steps = _bayesian_demo["scenario"]["steps"]

    if step >= len(steps):
        # Demo complete — return final state one last time
        grid_data = _grid_to_json(threshold=0.02, contour_level=0.6)
        return jsonify({
            "status": "complete",
            "message": "Historical replay complete.",
            "step": step,
            "total_steps": len(steps),
            **grid_data,
        }), 200

    step_data = steps[step]

    # 1. Inject new satellite hotspot evidence
    for hs in step_data["hotspots"]:
        evidence = Evidence.satellite_hotspot(lat=hs[0], lon=hs[1])
        demo_grid.update(evidence)

    # 2. Run predict step with current wind conditions
    # Simulate 30 minutes of spread at the given wind speed
    demo_grid.predict(
        dt=1800.0,  # 30 minutes
        wind_speed=step_data["wind_speed"],
        wind_dir_deg=step_data["wind_dir_deg"],
    )

    # 3. Update demo state
    _bayesian_demo["current_step"] = step
    all_complete = (step + 1) >= len(steps)
    if all_complete:
        _bayesian_demo["active"] = False

    # 4. Export the Bayesian grid state for the frontend
    # Use a dynamic contour level: 40% of max probability, min 0.3 so
    # the contour is visible even when the fire hasn't peaked yet
    max_p = demo_grid.get_statistics()["max_p"]
    dynamic_contour = max(0.3, min(0.7, max_p * 0.65))
    # Skip auto-predict since we just ran the explicit predict above
    grid_data = _grid_to_json(threshold=0.02, contour_level=dynamic_contour, auto_predict=False)

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
    global _bayesian_demo, _bayesian_grids
    _bayesian_demo = None
    _bayesian_grids.pop("creek-fire-demo", None)
    return jsonify({"status": "reset", "message": "Bayesian demo reset."}), 200


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("🔥 WildFrame — Wildfire Detection Prototype")
    print(f"   Listening on http://localhost:4141")
    print(f"   Cluster radius: {CLUSTER_RADIUS_M}m | Time window: {CLUSTER_TIME_WINDOW_MINUTES}min")
    app.run(host="0.0.0.0", port=4141, debug=True)