#!/usr/bin/env python3
"""Tests for the government agency incident pipeline.

Covers:
  * cap_adapter.parse_cap — CAP XML → canonical incident dicts
  * cap_adapter._centroid — shoelace polygon centroid
  * cap_adapter._derive_confidence — certainty/severity → 0..1
  * POST /api/agencies/ingest — create, update, cancel, idempotency, staleness
  * db.upsert_agency_incident — dedup key, staleness guard
  * Evidence.agency_confirm / agency_cancel — Bayesian grid fusion

Run from the project root:  .venv/bin/python test_agency_pipeline.py
"""

import json
import math
import os
import sys
import uuid
from datetime import datetime, timezone

import cap_adapter
import db

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


# ---------------------------------------------------------------------------
# 1. CAP XML parsing
# ---------------------------------------------------------------------------

SAMPLE_CAP = """\
<?xml version="1.0" encoding="UTF-8"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>PL-TEST-001</identifier>
  <sender>polfire@kgpsp.gov.pl</sender>
  <sent>2026-08-10T11:00:00Z</sent>
  <status>Actual</status>
  <msgType>Alert</msgType>
  <scope>Public</scope>
  <info>
    <language>pl-PL</language>
    <category>Fire</category>
    <event>Wildfire</event>
    <urgency>Immediate</urgency>
    <severity>Severe</severity>
    <certainty>Observed</certainty>
    <effective>2026-08-10T11:00:00Z</effective>
    <senderName>KG PSP</senderName>
    <headline>Forest fire near WildFrame</headline>
    <description>Test wildfire alert.</description>
    <area>
      <areaDesc>Forest near 51.9N 19.9E</areaDesc>
      <polygon>51.90,19.90 51.91,19.91 51.91,19.89 51.90,19.90</polygon>
    </area>
  </info>
</alert>
"""


def test_parse_cap_basic():
    print("[parse_cap: basic]")
    incidents = cap_adapter.parse_cap(SAMPLE_CAP)
    check("parses one alert", len(incidents) == 1)
    inc = incidents[0]
    check("agency from sender", inc["agency"] == "polfire@kgpsp.gov.pl")
    check("incident_id from identifier", inc["incident_id"] == "PL-TEST-001")
    check("action is create for Alert", inc["action"] == "create")
    check("source_type is government", inc["source_type"] == "government")
    check("has lat", inc.get("lat") is not None)
    check("has lon", inc.get("lon") is not None)
    check("severity preserved", inc.get("severity") == "Severe")
    check("has cap envelope in data", "cap" in inc.get("data", {}))
    cap_data = inc["data"]["cap"]
    check("cap msg_type is Alert", cap_data.get("msg_type") == "Alert")
    check("cap event is Wildfire", cap_data.get("event") == "Wildfire")
    check("cap headline preserved", cap_data.get("headline") == "Forest fire near WildFrame")


def test_parse_cap_circle():
    print("[parse_cap: circle geometry]")
    xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>CIRCLE-001</identifier>
  <sender>test@example.com</sender>
  <sent>2026-08-10T12:00:00Z</sent>
  <status>Actual</status>
  <msgType>Alert</msgType>
  <scope>Public</scope>
  <info>
    <event>Fire</event>
    <severity>Moderate</severity>
    <certainty>Likely</certainty>
    <area>
      <areaDesc>Circle area</areaDesc>
      <circle>48.85,2.35 5000</circle>
    </area>
  </info>
</alert>
"""
    incidents = cap_adapter.parse_cap(xml)
    check("parses circle alert", len(incidents) == 1)
    inc = incidents[0]
    check("lat ~ 48.85", abs(inc["lat"] - 48.85) < 0.01)
    check("lon ~ 2.35", abs(inc["lon"] - 2.35) < 0.01)


def test_parse_cap_update():
    print("[parse_cap: Update msgType]")
    xml = SAMPLE_CAP.replace("<msgType>Alert</msgType>", "<msgType>Update</msgType>")
    xml = xml.replace("<identifier>PL-TEST-001</identifier>",
                       "<identifier>PL-TEST-001</identifier>\n  <references>PL-TEST-001 polfire@kgpsp.gov.pl 2026-08-10T11:00:00Z</references>")
    incidents = cap_adapter.parse_cap(xml)
    check("parses update", len(incidents) == 1)
    check("action is update", incidents[0]["action"] == "update")


def test_parse_cap_cancel():
    print("[parse_cap: Cancel msgType]")
    xml = SAMPLE_CAP.replace("<msgType>Alert</msgType>", "<msgType>Cancel</msgType>")
    incidents = cap_adapter.parse_cap(xml)
    check("parses cancel", len(incidents) == 1)
    check("action is cancel", incidents[0]["action"] == "cancel")


def test_parse_cap_skip_ack():
    print("[parse_cap: skips Ack]")
    xml = SAMPLE_CAP.replace("<msgType>Alert</msgType>", "<msgType>Ack</msgType>")
    incidents = cap_adapter.parse_cap(xml)
    check("ack produces no incidents", len(incidents) == 0)


def test_parse_cap_skip_error():
    print("[parse_cap: skips Error]")
    xml = SAMPLE_CAP.replace("<msgType>Alert</msgType>", "<msgType>Error</msgType>")
    incidents = cap_adapter.parse_cap(xml)
    check("error produces no incidents", len(incidents) == 0)


def test_parse_cap_skip_test():
    print("[parse_cap: skips Test status]")
    xml = SAMPLE_CAP.replace("<status>Actual</status>", "<status>Test</status>")
    incidents = cap_adapter.parse_cap(xml)
    check("test alert skipped by default", len(incidents) == 0)
    incidents_with_test = cap_adapter.parse_cap(xml, include_test=True)
    check("test alert included with include_test=True", len(incidents_with_test) == 1)


def test_parse_cap_skip_exercise():
    print("[parse_cap: skips Exercise status]")
    xml = SAMPLE_CAP.replace("<status>Actual</status>", "<status>Exercise</status>")
    incidents = cap_adapter.parse_cap(xml)
    check("exercise alert skipped", len(incidents) == 0)


def test_parse_cap_no_geometry():
    print("[parse_cap: no geometry → skipped]")
    xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>NOGEO-001</identifier>
  <sender>test@example.com</sender>
  <sent>2026-08-10T12:00:00Z</sent>
  <status>Actual</status>
  <msgType>Alert</msgType>
  <scope>Public</scope>
  <info>
    <event>Fire</event>
    <severity>Minor</severity>
    <area>
      <areaDesc>No geometry here</areaDesc>
    </area>
  </info>
</alert>
"""
    incidents = cap_adapter.parse_cap(xml)
    check("no-geometry alert skipped", len(incidents) == 0)


def test_parse_cap_malformed_xml():
    print("[parse_cap: malformed XML]")
    try:
        cap_adapter.parse_cap("<not valid xml><<<")
        check("malformed XML raises CAPAdapterError", False)
    except cap_adapter.CAPAdapterError:
        check("malformed XML raises CAPAdapterError", True)


# ---------------------------------------------------------------------------
# 2. Geometry helpers
# ---------------------------------------------------------------------------

def test_centroid():
    print("[centroid]")
    # Simple square
    pts = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
    c = cap_adapter._centroid(pts)
    check("square centroid ~ (0.5, 0.5)", c is not None and abs(c[0] - 0.5) < 0.01 and abs(c[1] - 0.5) < 0.01)
    # Single point
    c1 = cap_adapter._centroid([(3.0, 4.0)])
    check("single point returns itself", c1 == (3.0, 4.0))
    # Empty
    c0 = cap_adapter._centroid([])
    check("empty returns None", c0 is None)
    # Two points → midpoint
    c2 = cap_adapter._centroid([(0.0, 0.0), (2.0, 4.0)])
    check("two points → midpoint", c2 is not None and abs(c2[0] - 1.0) < 0.01 and abs(c2[1] - 2.0) < 0.01)


# ---------------------------------------------------------------------------
# 3. Confidence derivation
# ---------------------------------------------------------------------------

def test_derive_confidence():
    print("[derive_confidence]")
    # Observed + Extreme = highest
    h = cap_adapter._derive_confidence("Observed", "Extreme")
    check("Observed+Extreme ≈ 1.0", abs(h - 1.0) < 0.01, f"got {h}")
    # Unknown + Unknown = low
    lo = cap_adapter._derive_confidence("Unknown", "Unknown")
    check("Unknown+Unknown ≈ 0.44", abs(lo - 0.44) < 0.02, f"got {lo}")
    # Likely + Severe
    m = cap_adapter._derive_confidence("Likely", "Severe")
    check("Likely+Severe ≈ 0.83", abs(m - 0.83) < 0.02, f"got {m}")
    # None inputs
    n = cap_adapter._derive_confidence(None, None)
    check("None+None returns float", isinstance(n, float))


# ---------------------------------------------------------------------------
# 4. Timestamp normalization
# ---------------------------------------------------------------------------

def test_normalize_ts():
    print("[normalize_ts]")
    z = cap_adapter._normalize_ts("2026-08-10T11:00:00Z")
    check("Z suffix → +00:00", z is not None and "+00:00" in z)
    offset = cap_adapter._normalize_ts("2026-08-10T13:00:00+02:00")
    check("offset converted to UTC", offset is not None and "+00:00" in offset)
    # Both should represent the same instant
    check("Z and +02:00 normalize to same UTC hour",
          z is not None and offset is not None and z[:13] == offset[:13],
          f"z={z}, offset={offset}")
    none_ts = cap_adapter._normalize_ts(None)
    check("None returns None", none_ts is None)
    empty_ts = cap_adapter._normalize_ts("")
    check("empty string returns empty", empty_ts == "")


# ---------------------------------------------------------------------------
# 5. Ingest endpoint (requires running DB + server context)
# ---------------------------------------------------------------------------

def _make_incident(**overrides) -> dict:
    """Build a minimal valid agency incident dict."""
    base = {
        "id": uuid.uuid4().hex,
        "agency": "test-agency",
        "incident_id": f"test-{uuid.uuid4().hex[:8]}",
        "action": "create",
        "sent_at": datetime.now(timezone.utc).isoformat(),
        "lat": 51.9,
        "lon": 19.9,
        "source_type": "government",
        "status": "confirmed",
        "data": {},
    }
    base.update(overrides)
    return base


def test_upsert_create():
    print("[upsert: create]")
    inc = _make_incident()
    report, created, applied = db.upsert_agency_incident(inc, "demo")
    check("first insert creates", created is True)
    check("applied is True", applied is True)
    check("report has id", report.get("id") is not None)
    check("report status is confirmed", report.get("status") == "confirmed")
    # Cleanup
    db.delete_report(report["id"], "demo")


def test_upsert_idempotent():
    print("[upsert: idempotent retry]")
    inc = _make_incident()
    r1, created1, applied1 = db.upsert_agency_incident(inc, "demo")
    check("first creates", created1 is True)
    # Same sent_at → duplicate
    r2, created2, applied2 = db.upsert_agency_incident(inc, "demo")
    check("retry with same sent_at is not created", created2 is False)
    check("retry with same sent_at is not applied", applied2 is False)
    check("returned report matches original", r2.get("id") == r1.get("id"))
    # Cleanup
    db.delete_report(r1["id"], "demo")


def test_upsert_staleness_guard():
    print("[upsert: staleness guard]")
    inc = _make_incident(sent_at="2026-08-10T10:00:00+00:00")
    r1, _, _ = db.upsert_agency_incident(inc, "demo")
    # Older sent_at should be rejected
    old = _make_incident(incident_id=inc["incident_id"],
                         agency=inc["agency"],
                         sent_at="2026-08-10T09:00:00+00:00")
    r2, created2, applied2 = db.upsert_agency_incident(old, "demo")
    check("older sent_at is not created", created2 is False)
    check("older sent_at is not applied", applied2 is False)
    check("returned report is the newer one", r2.get("sent_at") == "2026-08-10T10:00:00+00:00")
    # Newer sent_at should win
    new = _make_incident(incident_id=inc["incident_id"],
                         agency=inc["agency"],
                         sent_at="2026-08-10T11:00:00+00:00")
    r3, created3, applied3 = db.upsert_agency_incident(new, "demo")
    check("newer sent_at is applied", applied3 is True)
    check("status updated to confirmed", r3.get("status") == "confirmed")
    # Cleanup
    db.delete_report(r1["id"], "demo")


def test_upsert_cancel():
    print("[upsert: cancel]")
    inc = _make_incident(sent_at="2026-08-10T10:00:00+00:00")
    r1, _, _ = db.upsert_agency_incident(inc, "demo")
    # Cancel with newer sent_at
    cancel = _make_incident(incident_id=inc["incident_id"],
                            agency=inc["agency"],
                            action="cancel",
                            status="cancelled",
                            sent_at="2026-08-10T12:00:00+00:00")
    r2, _, applied = db.upsert_agency_incident(cancel, "demo")
    check("cancel is applied", applied is True)
    check("status is cancelled", r2.get("status") == "cancelled")
    # Cleanup
    db.delete_report(r1["id"], "demo")


# ---------------------------------------------------------------------------
# 6. Evidence objects
# ---------------------------------------------------------------------------

def test_evidence_agency_confirm():
    print("[Evidence.agency_confirm]")
    from bayesian_filter import Evidence
    e = Evidence.agency_confirm(51.9, 19.9)
    check("source is agency-confirm", e.source == "agency-confirm")
    check("log LR > 0 (positive evidence)", e.log_likelihood_ratio > 0)
    check("log LR ≈ ln(100)", abs(e.log_likelihood_ratio - math.log(100.0)) < 0.01)
    check("has lat/lon", e.lat == 51.9 and e.lon == 19.9)


def test_evidence_agency_cancel():
    print("[Evidence.agency_cancel]")
    from bayesian_filter import Evidence
    e = Evidence.agency_cancel(51.9, 19.9)
    check("source is agency-cancel", e.source == "agency-cancel")
    check("log LR < 0 (negative evidence)", e.log_likelihood_ratio < 0)
    check("log LR ≈ ln(1/100)", abs(e.log_likelihood_ratio - math.log(1.0 / 100.0)) < 0.01)


# ---------------------------------------------------------------------------
# 7. CAP adapter demo dry-run (parse only, no HTTP)
# ---------------------------------------------------------------------------

def test_cap_adapter_parse_and_fields():
    print("[cap_adapter: full field extraction]")
    incidents = cap_adapter.parse_cap(SAMPLE_CAP)
    inc = incidents[0]
    cap = inc["data"]["cap"]
    check("certainty is Observed", cap.get("certainty") == "Observed")
    check("urgency is Immediate", cap.get("urgency") == "Immediate")
    check("sender_name preserved", cap.get("sender_name") == "KG PSP")
    check("description preserved", cap.get("description") == "Test wildfire alert.")
    check("confidence > 0.8", inc["data"].get("confidence", 0) > 0.8,
          f"confidence={inc['data'].get('confidence')}")
    check("areas list has one entry", len(cap.get("areas", [])) == 1)
    check("area has polygon", "polygon" in cap["areas"][0])


# ---------------------------------------------------------------------------
# 8. Integration: full CAP feed → HTTP ingest → DB → grid
# ---------------------------------------------------------------------------

def _get_test_client():
    """Lazy-import Flask test client (requires server.py + DB)."""
    from server import app
    app.config["TESTING"] = True
    return app.test_client()


def _post_ingest(client, incident, api_key=None):
    """POST to /api/agencies/ingest and return (status_code, json)."""
    headers = {"Content-Type": "application/json"}
    key = api_key or os.environ.get("WILDFRAME_AGENCY_API_KEY")
    if key:
        headers["X-Agency-Key"] = key
    resp = client.post("/api/agencies/ingest",
                       data=json.dumps(incident),
                       headers=headers)
    return resp.status_code, resp.get_json()


def test_integration_create_via_http():
    print("[integration: create via HTTP]")
    client = _get_test_client()
    inc = _make_incident(mode="demo")
    status, body = _post_ingest(client, inc)
    check("HTTP 200", status == 200, f"got {status}")
    check("created=True", body.get("created") is True)
    check("applied (grid fused)", body.get("grid", {}).get("fused") is True)
    check("grid_id returned", body.get("grid", {}).get("grid_id") is not None)
    check("report status confirmed", body.get("report", {}).get("status") == "confirmed")
    # Cleanup
    db.delete_report(body["report"]["id"], "demo")


def test_integration_idempotent_via_http():
    print("[integration: idempotent retry via HTTP]")
    client = _get_test_client()
    inc = _make_incident(mode="demo")
    # First POST
    s1, b1 = _post_ingest(client, inc)
    check("first: HTTP 200", s1 == 200)
    check("first: created", b1.get("created") is True)
    # Second POST (same body → duplicate)
    s2, b2 = _post_ingest(client, inc)
    check("retry: HTTP 200", s2 == 200)
    check("retry: not created", b2.get("created") is False)
    check("retry: not duplicate (same sent_at)", b2.get("duplicate") is True)
    check("retry: grid NOT re-fused", b2.get("grid", {}).get("fused") is False)
    # Cleanup
    db.delete_report(b1["report"]["id"], "demo")


def test_integration_staleness_via_http():
    print("[integration: staleness guard via HTTP]")
    client = _get_test_client()
    inc = _make_incident(mode="demo", sent_at="2026-08-10T10:00:00+00:00")
    s1, b1 = _post_ingest(client, inc)
    check("create: HTTP 200", s1 == 200)
    # Older message
    old = _make_incident(incident_id=inc["incident_id"], agency=inc["agency"],
                         mode="demo", sent_at="2026-08-10T09:00:00+00:00")
    s2, b2 = _post_ingest(client, old)
    check("older: HTTP 200", s2 == 200)
    check("older: stale=True", b2.get("stale") is True)
    check("older: grid NOT fused", b2.get("grid", {}).get("fused") is False)
    # Newer message
    new = _make_incident(incident_id=inc["incident_id"], agency=inc["agency"],
                         mode="demo", sent_at="2026-08-10T11:00:00+00:00")
    s3, b3 = _post_ingest(client, new)
    check("newer: HTTP 200", s3 == 200)
    check("newer: created=False (update)", b3.get("created") is False)
    check("newer: stale=False", b3.get("stale") is False)
    check("newer: grid fused", b3.get("grid", {}).get("fused") is True)
    # Cleanup
    db.delete_report(b1["report"]["id"], "demo")


def test_integration_cancel_via_http():
    print("[integration: cancel via HTTP]")
    client = _get_test_client()
    # Create
    inc = _make_incident(mode="demo", sent_at="2026-08-10T10:00:00+00:00")
    s1, b1 = _post_ingest(client, inc)
    check("create: HTTP 200", s1 == 200)
    grid_id = b1.get("grid", {}).get("grid_id")
    check("grid created", grid_id is not None)
    # Cancel
    cancel = _make_incident(incident_id=inc["incident_id"], agency=inc["agency"],
                            mode="demo", action="cancel", status="cancelled",
                            sent_at="2026-08-10T12:00:00+00:00")
    s2, b2 = _post_ingest(client, cancel)
    check("cancel: HTTP 200", s2 == 200)
    check("cancel: status=cancelled", b2.get("report", {}).get("status") == "cancelled")
    check("cancel: grid fused (negative evidence)", b2.get("grid", {}).get("fused") is True)
    check("cancel: same grid_id", b2.get("grid", {}).get("grid_id") == grid_id)
    # Cleanup
    db.delete_report(b1["report"]["id"], "demo")


def test_integration_validation():
    print("[integration: validation errors via HTTP]")
    client = _get_test_client()
    # Missing agency
    s, b = _post_ingest(client, {"incident_id": "x", "lat": 1, "lon": 1})
    check("missing agency → 400", s == 400)
    check("error message", "agency" in b.get("error", ""))
    # Missing lat/lon
    s, b = _post_ingest(client, {"agency": "test", "incident_id": "x"})
    check("missing lat/lon → 400", s == 400)
    # Invalid action
    s, b = _post_ingest(client, {"agency": "test", "incident_id": "x",
                                 "lat": 1, "lon": 1, "action": "bogus"})
    check("invalid action → 400", s == 400)


def test_integration_cap_feed_full_lifecycle():
    print("[integration: CAP feed full lifecycle]")
    client = _get_test_client()
    # Parse a sample CAP alert
    incidents = cap_adapter.parse_cap(SAMPLE_CAP)
    check("sample parses to 1 incident", len(incidents) == 1)
    inc = incidents[0]
    inc["mode"] = "demo"

    # 1. CREATE
    s1, b1 = _post_ingest(client, inc)
    check("Alert: HTTP 200", s1 == 200)
    check("Alert: created", b1.get("created") is True)
    check("Alert: grid fused", b1.get("grid", {}).get("fused") is True)
    report_id = b1["report"]["id"]
    grid_id = b1["grid"]["grid_id"]

    # 2. UPDATE (severity bump)
    update_inc = json.loads(json.dumps(inc))
    update_inc["action"] = "update"
    update_inc["id"] = report_id  # keep consistent id in data blob
    update_inc["sent_at"] = cap_adapter._normalize_ts("2026-08-10T12:00:00Z")
    update_inc["severity"] = "Extreme"
    update_inc["data"]["cap"]["msg_type"] = "Update"
    s2, b2 = _post_ingest(client, update_inc)
    check("Update: HTTP 200", s2 == 200)
    check("Update: not created (update)", b2.get("created") is False)
    check("Update: stale=False", b2.get("stale") is False)
    check("Update: grid fused", b2.get("grid", {}).get("fused") is True)
    check("Update: same report id", b2["report"]["id"] == report_id)

    # 3. CANCEL
    cancel_inc = json.loads(json.dumps(inc))
    cancel_inc["action"] = "cancel"
    cancel_inc["status"] = "cancelled"
    cancel_inc["sent_at"] = cap_adapter._normalize_ts("2026-08-10T13:00:00Z")
    cancel_inc["data"]["cap"]["msg_type"] = "Cancel"
    s3, b3 = _post_ingest(client, cancel_inc)
    check("Cancel: HTTP 200", s3 == 200)
    check("Cancel: status=cancelled", b3["report"]["status"] == "cancelled")
    check("Cancel: grid fused (negative evidence)", b3["grid"]["fused"] is True)
    check("Cancel: same grid_id", b3["grid"]["grid_id"] == grid_id)

    # Cleanup
    db.delete_report(report_id, "demo")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_parse_cap_basic()
    test_parse_cap_circle()
    test_parse_cap_update()
    test_parse_cap_cancel()
    test_parse_cap_skip_ack()
    test_parse_cap_skip_error()
    test_parse_cap_skip_test()
    test_parse_cap_skip_exercise()
    test_parse_cap_no_geometry()
    test_parse_cap_malformed_xml()
    test_centroid()
    test_derive_confidence()
    test_normalize_ts()
    test_evidence_agency_confirm()
    test_evidence_agency_cancel()
    test_cap_adapter_parse_and_fields()

    # DB tests — only run if DB is available
    try:
        import psycopg
        with psycopg.connect(db.DATABASE_URL) as conn:
            pass  # DB reachable
        test_upsert_create()
        test_upsert_idempotent()
        test_upsert_staleness_guard()
        test_upsert_cancel()

        # Integration tests (need Flask + DB)
        test_integration_create_via_http()
        test_integration_idempotent_via_http()
        test_integration_staleness_via_http()
        test_integration_cancel_via_http()
        test_integration_validation()
        test_integration_cap_feed_full_lifecycle()
    except Exception as exc:
        print(f"\n  SKIP  DB/integration tests (cannot connect: {exc})")

    print(f"\n{'ALL PASS' if FAIL == 0 else 'SOME FAILED'} ({PASS} pass, {FAIL} fail)")
    sys.exit(1 if FAIL else 0)
