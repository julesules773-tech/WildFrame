"""
cap_adapter.py — Common Alerting Protocol (CAP) → WildFrame incident adapter
=============================================================================

Parses CAP 1.0 / 1.1 / 1.2 XML alerts into the canonical incident dict that
``POST /api/agencies/ingest`` expects, and pushes them through that endpoint
— the same front door a Lambda (push) or a poller job (pull) would use.
The ingest endpoint then does idempotent, staleness-guarded storage plus
Bayesian grid evidence fusion in one shot.

Pipeline::

    CAP XML (feed URL or webhook body)
        → parse_cap()               namespace-agnostic, msgType-aware
        → canonical incident dict   (one per alert, keyed by sender+identifier)
        → post_incident()           POST /api/agencies/ingest
        → reports table + grid evidence (existing machinery, unchanged)

Design decisions
----------------
* **One incident per CAP alert.** CAP's native dedup pair is
  ``(sender, identifier)``; we use it verbatim as ``(agency, incident_id)``,
  so retries, Updates and Cancels collapse onto the same row for free.
  Cancel/Delete messages even carry ``references`` back to the original
  alert, which we preserve for traceability.

* **msgType → action:** Alert→create, Update→update, Cancel→cancel,
  Delete→delete. Ack and Error are acknowledgements, not incidents, and are
  skipped. Alerts with ``status`` Exercise/Test/Draft are skipped unless
  ``include_test=True`` — you don't want drill alerts on the live map.

* **Namespace-agnostic parsing.** CAP versions differ only in their XML
  namespace URI; all element lookups match on the local tag name, so
  1.0/1.1/1.2 parse identically.

* **Location from the first placed area.** The first ``<info>/<area>`` that
  has a polygon or circle supplies the point: polygons get an
  area-weighted (shoelace) centroid, circles their centre. The full area
  set is preserved in ``data.cap.areas`` so nothing is lost. Alerts with no
  geometry anywhere are skipped (they can't be placed on the map).

* **Trust fields.** CAP's ``severity`` / ``certainty`` / ``urgency`` are
  preserved verbatim, plus a derived ``confidence`` (0..1) in ``data`` —
  a placeholder for the source-tier trust model, not consumed yet.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)


class CAPAdapterError(Exception):
    """Raised on malformed CAP XML, feed fetch failures, or ingest failures."""


# ---------------------------------------------------------------------------
# msgType / status handling
# ---------------------------------------------------------------------------

ACTION_BY_MSGTYPE = {
    "Alert": "create",
    "Update": "update",
    "Cancel": "cancel",
    "Delete": "delete",
}

#: Ack/Error messages acknowledge receipt; they are not incidents.
SKIPPED_MSGTYPES = {"Ack", "Error"}

#: Drill / test alerts must never reach the live map unless explicitly asked.
SKIPPED_STATUSES = {"Exercise", "Test", "Draft"}

#: Heuristic certainty → confidence contribution (placeholder trust tier).
_CERTAINTY_CONF = {
    "Observed": 1.0,
    "Likely": 0.8,
    "Possible": 0.6,
    "Unlikely": 0.3,
    "Unknown": 0.4,
}

#: Heuristic severity → confidence contribution (placeholder trust tier).
_SEVERITY_CONF = {
    "Extreme": 1.0,
    "Severe": 0.85,
    "Moderate": 0.7,
    "Minor": 0.5,
    "Unknown": 0.5,
}


# ---------------------------------------------------------------------------
# Namespace-agnostic XML helpers
# ---------------------------------------------------------------------------


def _local(tag: str) -> str:
    """Strip any ``{namespace}`` prefix from an ElementTree tag."""
    return tag.rsplit("}", 1)[-1]


def _find(el: ET.Element, name: str) -> Optional[ET.Element]:
    """First direct child whose LOCAL tag name is ``name`` (any CAP ns)."""
    for child in el:
        if _local(child.tag) == name:
            return child
    return None


def _find_all(el: ET.Element, name: str) -> List[ET.Element]:
    return [child for child in el if _local(child.tag) == name]


def _text(el: ET.Element, name: str, default: Optional[str] = None) -> Optional[str]:
    child = _find(el, name)
    if child is not None and child.text:
        return child.text.strip()
    return default


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _parse_polygon(text: str) -> List[Tuple[float, float]]:
    """``"lat,lon lat,lon …"`` (CAP polygon, vertices clockwise or not)."""
    points = []
    for pair in text.split():
        try:
            lat_s, lon_s = pair.split(",")
            points.append((float(lat_s.strip()), float(lon_s.strip())))
        except (ValueError, TypeError):
            continue
    return points


def _parse_circle(text: str) -> Optional[Tuple[float, float]]:
    """``"lat,lon radius"`` → the circle centre (radius discarded)."""
    parts = text.split()
    if not parts:
        return None
    try:
        lat_s, lon_s = parts[0].split(",")
        return (float(lat_s.strip()), float(lon_s.strip()))
    except (ValueError, IndexError, TypeError):
        return None


def _centroid(points: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """Area-weighted centroid (shoelace) of a polygon, in degrees.

    Degenerate inputs (1–2 points, or zero area) fall back to the vertex
    mean so we always return *a* sensible point rather than erroring.
    """
    if not points:
        return None
    if len(points) == 1:
        return points[0]
    if len(points) == 2:
        return ((points[0][0] + points[1][0]) / 2.0,
                (points[0][1] + points[1][1]) / 2.0)
    ring = points if points[0] == points[-1] else points + [points[0]]
    area = 0.0
    cx = 0.0
    cy = 0.0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:]):
        cross = x0 * y1 - x1 * y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    area *= 0.5
    if abs(area) < 1e-12:  # degenerate — mean of vertices
        return (sum(p[0] for p in points) / len(points),
                sum(p[1] for p in points) / len(points))
    return (cx / (6.0 * area), cy / (6.0 * area))


# ---------------------------------------------------------------------------
# Alert → canonical incident mapping
# ---------------------------------------------------------------------------


def _normalize_ts(value: Optional[str]) -> Optional[str]:
    """Canonicalize a CAP timestamp to a UTC ISO-8601 string.

    CAP senders emit ``sent`` in many spellings (``...Z``, ``...+00:00``,
    ``...-05:00``). The ingest route compares ``sent_at`` as raw strings to
    compute its ``stale``/``duplicate`` flags, so equal instants in
    different formats would mislabel a retry. We normalize everything to
    one canonical UTC form; unparseable values pass through verbatim.
    """
    if not value:
        return value
    try:
        v = value.strip()
        if v.endswith("Z"):
            v = v[:-1] + "+00:00"
        dt = datetime.fromisoformat(v)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except ValueError:
        return value


def _derive_confidence(certainty: Optional[str], severity: Optional[str]) -> float:
    """0..1 heuristic from CAP certainty + severity (see module docstring)."""
    c = _CERTAINTY_CONF.get(certainty or "", 0.4)
    s = _SEVERITY_CONF.get(severity or "", 0.5)
    return round(0.6 * c + 0.4 * s, 3)


def _alert_to_incident(root: ET.Element, include_test: bool = False) -> Optional[dict]:
    """Map one CAP ``<alert>`` to the canonical incident dict, or None."""
    status = (_text(root, "status") or "Actual").strip()
    if status in SKIPPED_STATUSES and not include_test:
        logger.info("Skipping CAP alert: drill status %r", status)
        return None

    msg_type = (_text(root, "msgType") or "Alert").strip()
    if msg_type in SKIPPED_MSGTYPES:
        logger.info("Skipping CAP alert: %s messages are not incidents", msg_type)
        return None

    sender = _text(root, "sender")
    identifier = _text(root, "identifier")
    if not sender or not identifier:
        logger.warning("Skipping CAP alert without sender/identifier")
        return None

    action = ACTION_BY_MSGTYPE.get(msg_type, "create")

    # --- locate the first placed area, preserving every area for the blob ---
    lat = lon = None
    areas = []
    for info in _find_all(root, "info"):
        for area in _find_all(info, "area"):
            area_desc = _text(area, "areaDesc") or ""
            polygon_text = _text(area, "polygon")
            circle_text = _text(area, "circle")
            entry = {"areaDesc": area_desc}
            if polygon_text:
                pts = _parse_polygon(polygon_text)
                entry["polygon"] = pts
                if lat is None and pts:
                    c = _centroid(pts)
                    if c:
                        lat, lon = c
            elif circle_text:
                entry["circle"] = circle_text
                if lat is None:
                    c = _parse_circle(circle_text)
                    if c:
                        lat, lon = c
            areas.append(entry)
        if lat is not None:
            break  # only need the first info block with a placed area

    if lat is None or lon is None:
        logger.warning(
            "Skipping CAP alert %s: no polygon/circle geometry to place it",
            identifier,
        )
        return None
    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        logger.warning("Skipping CAP alert %s: out-of-range geometry", identifier)
        return None

    severity = _text(info, "severity") if info is not None else None
    certainty = _text(info, "certainty") if info is not None else None
    urgency = _text(info, "urgency") if info is not None else None

    # One clean value for the top-level fields; the full envelope rides along
    # in data.cap so the UI/admin can render it later.
    sent_at = _normalize_ts(_text(root, "sent"))
    incident = {
        "agency": sender,
        "incident_id": identifier,
        "action": action,
        # Only set sent_at when CAP actually provides it: an empty string
        # would reach the upsert as NULL, and the guard's (both-NULL)
        # clause would let retries re-apply — the double-fusion bug. When
        # absent, the ingest route defaults it to now.
        **({"sent_at": sent_at} if sent_at else {}),
        "lat": lat,
        "lon": lon,
        "source_type": "government",
        # cancel/delete → the ingest route flips status to "cancelled";
        # anything else rides the high-trust confirmed lane by default.
        "status": None,
        "severity": severity,
        "data": {
            "confidence": _derive_confidence(certainty, severity),
            "cap": {
                "msg_type": msg_type,
                "status": status,
                "event": _text(info, "event") if info is not None else None,
                "headline": _text(info, "headline") if info is not None else None,
                "description": _text(info, "description") if info is not None else None,
                "urgency": urgency,
                "severity": severity,
                "certainty": certainty,
                "effective": _text(info, "effective") if info is not None else None,
                "onset": _text(info, "onset") if info is not None else None,
                "expires": _text(info, "expires") if info is not None else None,
                "sender_name": _text(info, "senderName") if info is not None else None,
                "references": _text(root, "references"),
                "areas": areas,
            },
        },
    }
    return incident


def parse_cap(xml_text: str, include_test: bool = False) -> List[dict]:
    """Parse CAP XML into a list of canonical incident dicts.

    One dict per ``<alert>`` that is placeable (has geometry) and not an
    Ack/Error/drill message. Never raises on a bad *feed* — malformed XML
    raises :class:`CAPAdapterError`.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise CAPAdapterError(f"malformed CAP XML: {exc}") from exc

    incidents = []
    for alert in _find_all(root, "alert") if _local(root.tag) != "alert" else [root]:
        incident = _alert_to_incident(alert, include_test=include_test)
        if incident is not None:
            incidents.append(incident)
    return incidents


# ---------------------------------------------------------------------------
# Push through POST /api/agencies/ingest
# ---------------------------------------------------------------------------


def post_incident(base_url: str, incident: dict, timeout: int = 15,
                  api_key: Optional[str] = None) -> Tuple[int, dict]:
    """POST one canonical incident to ``POST /api/agencies/ingest``.

    ``api_key``, when given, is sent as the ``X-Agency-Key`` header — the
    shared secret the endpoint enforces once configured.

    Returns ``(http_status, response_json)``. Raises :class:`CAPAdapterError`
    on transport errors or non-2xx responses (the ingest endpoint itself
    returns 400 for validation failures — surface its ``error`` message).
    """
    url = base_url.rstrip("/") + "/api/agencies/ingest"
    body = json.dumps(incident).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-Agency-Key"] = api_key
    req = Request(url, data=body, method="POST", headers=headers)
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8")[:300]
        except Exception:
            pass
        raise CAPAdapterError(f"ingest HTTP {exc.code}: {detail}") from exc
    except (URLError, OSError) as exc:
        raise CAPAdapterError(f"ingest endpoint unreachable: {exc}") from exc


def fetch_feed(url: str, timeout: int = 30) -> str:
    """Download a CAP feed's raw XML (the pull path — poller job friendly)."""
    req = Request(url, headers={"User-Agent": "WildFrame-cap-adapter/1.0"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (HTTPError, URLError, OSError) as exc:
        raise CAPAdapterError(f"feed fetch failed: {exc}") from exc


def consume_cap_feed(feed_url: str, base_url: str, mode: str = "demo",
                     include_test: bool = False,
                     api_key: Optional[str] = None) -> dict:
    """Fetch a CAP feed and push every placeable alert through the ingest
    endpoint. Returns a per-alert summary — the shape a poller job would
    report and store in ``kv_store``. ``api_key`` is forwarded to
    :func:`post_incident` as the ``X-Agency-Key`` header.
    """
    xml_text = fetch_feed(feed_url)
    incidents = parse_cap(xml_text, include_test=include_test)
    results = []
    for incident in incidents:
        incident.setdefault("mode", mode)
        try:
            status, body = post_incident(base_url, incident, api_key=api_key)
            results.append({
                "agency": incident["agency"],
                "incident_id": incident["incident_id"],
                "action": incident["action"],
                "http": status,
                "created": body.get("created"),
                "stale": body.get("stale"),
                "duplicate": body.get("duplicate"),
                "grid_fused": (body.get("grid") or {}).get("fused"),
            })
        except CAPAdapterError as exc:
            logger.error("[cap] ingest failed for %s: %s", incident["incident_id"], exc)
            results.append({
                "agency": incident["agency"],
                "incident_id": incident["incident_id"],
                "action": incident["action"],
                "error": str(exc),
            })
    return {"fetched": len(incidents), "results": results}


# ---------------------------------------------------------------------------
# Sample CAP 1.2 alert + demo
# ---------------------------------------------------------------------------

SAMPLE_CAP_XML = """<?xml version="1.0" encoding="UTF-8"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>PL-2026-08-10-00123</identifier>
  <sender>polfire@kgpsp.gov.pl</sender>
  <sent>2026-08-10T11:00:00Z</sent>
  <status>Actual</status>
  <msgType>Alert</msgType>
  <scope>Public</scope>
  <code>wildframe-demo</code>
  <info>
    <language>pl-PL</language>
    <category>Fire</category>
    <event>Wildfire</event>
    <urgency>Immediate</urgency>
    <severity>Severe</severity>
    <certainty>Observed</certainty>
    <effective>2026-08-10T11:00:00Z</effective>
    <onset>2026-08-10T10:45:00Z</onset>
    <expires>2026-08-11T11:00:00Z</expires>
    <senderName>Komenda Główna Państwowej Straży Pożarnej</senderName>
    <headline>Pożar lasu w okolicy miejscowości WildFrame</headline>
    <description>Unconfirmed wildfire detected in forest area; fire brigade dispatched.</description>
    <instruction>Stay clear of the area. Follow instructions from local authorities.</instruction>
    <area>
      <areaDesc>Forest parcel near 51.905N 19.900E</areaDesc>
      <polygon>51.9000,19.9000 51.9050,19.9100 51.9100,19.9000 51.9050,19.8900 51.9000,19.9000</polygon>
    </area>
  </info>
</alert>
"""


_MSGTYPE_BY_ACTION = {"create": "Alert", "update": "Update", "cancel": "Cancel", "delete": "Delete"}


def _demo_alert_copy(source: dict, action: str, sent_at: str, **overrides) -> dict:
    inc = json.loads(json.dumps(source))
    inc["action"] = action
    inc["sent_at"] = _normalize_ts(sent_at)
    inc["status"] = None
    # Keep the embedded CAP envelope consistent with the action being sent.
    inc.setdefault("data", {}).setdefault("cap", {})["msg_type"] = _MSGTYPE_BY_ACTION.get(action, "Alert")
    inc.update(overrides)
    return inc


def demo(base_url: str = "http://localhost:4141", mode: str = "demo") -> None:
    """Parse the sample alert and push the full lifecycle through the real
    ingest endpoint: Alert → Update → Cancel.

    Sends the ``X-Agency-Key`` header when ``WILDFRAME_AGENCY_API_KEY`` is
    set (matching the endpoint's shared-secret gate)."""
    api_key = os.environ.get("WILDFRAME_AGENCY_API_KEY") or None
    (incident,) = parse_cap(SAMPLE_CAP_XML)
    incident["mode"] = mode

    print("=== CAP adapter demo —", base_url, "mode:", mode, "===\n")
    for label, inc in [
        ("ALERT  (create)", incident),
        ("UPDATE (severity bump)",
         _demo_alert_copy(incident, "update", "2026-08-10T12:00:00Z",
                          severity="Extreme")),
        ("CANCEL (fire contained)",
         _demo_alert_copy(incident, "cancel", "2026-08-10T13:00:00Z")),
    ]:
        print(f"── {label} ──")
        status, body = post_incident(base_url, inc, api_key=api_key)
        print(f"  HTTP {status}  created={body.get('created')}  "
              f"stale={body.get('stale')}  duplicate={body.get('duplicate')}  "
              f"grid={body.get('grid')}")
        print(f"  status={body.get('report', {}).get('status')}  "
              f"sent_at={body.get('report', {}).get('sent_at')}")
    print("\nDemo rows live in mode=%s — clean up with:" % mode)
    print("  psql -d wildframe -c \"DELETE FROM reports "
          "WHERE agency='polfire@kgpsp.gov.pl' AND mode='%s';\"" % mode)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:4141"
    mode = sys.argv[2] if len(sys.argv) > 2 else "demo"
    demo(base, mode)
