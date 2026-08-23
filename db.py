#!/usr/bin/env python3
"""
db.py — PostgreSQL/PostGIS persistence layer for WildFrame
===========================================================

Replaces the JSON-file + in-memory stores that previously lost updates
under concurrent requests and vanished across restarts / multiple
workers. Every read-modify-write is now a single-row SQL statement or a
row-locked transaction, so concurrent web workers and the job-queue
worker can never clobber each other's writes.

Stores
------
- ``reports``           — citizen reports + seed/demo reports (mode column
                          keeps production and demo data fully isolated)
- ``bayesian_grids``    — one row per fire; the numpy grid state lives in
                          a JSONB blob (see BayesianFireGrid.to_dict) and
                          per-grid mutations run under ``SELECT ... FOR
                          UPDATE`` row locks
- ``osm_road_cache``    — OSM road segments keyed by rounded lat/lon/radius
- ``kv_store``          — tiny config/state store (poller flags, grid id
                          counters, heartbeats)

Connection
----------
A thread-safe psycopg connection pool. Configure with the
``WILDFRAME_DATABASE_URL`` env var; defaults to ``postgresql:///wildframe``
(Homebrew Postgres on macOS, current-user auth, unix socket).
"""

import json
import math
import os
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Iterator, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from bayesian_filter import BayesianFireGrid, auto_grid_size, DEFAULT_CELL_SIZE_M, CITIZEN_CELL_SIZE_M

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

DATABASE_URL = os.environ.get(
    "WILDFRAME_DATABASE_URL",
    "postgresql:///wildframe",
)

# How close a cluster's centroid must be to an existing grid's tracked
# centroid to reuse that grid rather than create a new fire. Shared with
# server.py (which imported the same constant).
GRID_MATCH_RADIUS_M = 10000.0

# Rows per chunk for the batched point-in-polygon lookups
# (land_cover_codes_batch / static_source_hits_batch). Each row is 3 bind
# params (idx, lon, lat); 10k rows = 30k params, safely under Postgres's
# 65,535-param statement limit so a full multi-satellite 48h pass (~250k
# hotspots) never trips "number of parameters must be between 0 and 65535"
# — which the server raises before it would ever raise UndefinedTable, so
# the fail-open path would otherwise never fire.
_BATCH_LOOKUP_CHUNK = 10_000

# Cached coverage bbox (minlon, minlat, maxlon, maxlat) per spatial lookup
# table. The CORINE + static-source layers only cover the Poland beta
# footprint, but the FIRMS pass now feeds them ~255k global hotspots —
# gating every one of them is pure waste (and, with the batched VALUES
# joins, enough transient memory to OOM the 1 GB production VM). Points
# outside the layer's own extent fail open anyway (no polygon → burnable /
# not static), so pre-filtering to the extent is behaviour-identical but
# drops the query work by ~99%.
_BATCH_EXTENT_CACHE: dict[str, Optional[tuple[float, float, float, float]]] = {}


def _table_bbox(table: str) -> Optional[tuple[float, float, float, float]]:
    """Coverage bbox of a spatial table, or None if missing/empty (fail-open)."""
    if table in _BATCH_EXTENT_CACHE:
        return _BATCH_EXTENT_CACHE[table]
    try:
        with _conn() as conn:
            row = conn.execute(
                f"SELECT ST_XMin(e) AS minlon, ST_YMin(e) AS minlat, "
                f"ST_XMax(e) AS maxlon, ST_YMax(e) AS maxlat "
                f"FROM (SELECT ST_Extent(geom) AS e FROM {table}) s"
            ).fetchone()
    except psycopg.errors.UndefinedTable:
        _BATCH_EXTENT_CACHE[table] = None
        return None
    if not row or row["minlon"] is None:
        _BATCH_EXTENT_CACHE[table] = None
        return None
    ext = (float(row["minlon"]), float(row["minlat"]),
           float(row["maxlon"]), float(row["maxlat"]))
    _BATCH_EXTENT_CACHE[table] = ext
    return ext


def _in_bbox(lat: float, lon: float, bbox: tuple[float, float, float, float]) -> bool:
    minlon, minlat, maxlon, maxlat = bbox
    return minlon <= lon <= maxlon and minlat <= lat <= maxlat

_pool: Optional[ConnectionPool] = None


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            DATABASE_URL,
            min_size=1,
            max_size=10,
            open=False,
            # Close idle connections after 5 minutes so stale/server-killed
            # connections don't cause sporadic OperationalError on the first
            # request after a quiet period.
            max_idle=300,
            kwargs={
                "row_factory": dict_row,
                # Fail fast on dead DB so PaaS health checks (e.g. Fly's
                # 3 s probe) get a 503 instead of a ~10 s hang.
                "connect_timeout": 5,
            },
        )
        _pool.open()
        _pool.wait(timeout=10.0)
    return _pool


def _conn():
    """Context manager yielding a pooled connection (with commit semantics
    controlled by the caller via transactions)."""
    return _get_pool().connection()


def check_connection() -> bool:
    """Return True if the database is reachable."""
    try:
        with _conn() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS reports (
    id                 TEXT PRIMARY KEY,
    mode               TEXT NOT NULL DEFAULT 'production',
    lat                DOUBLE PRECISION NOT NULL,
    lon                DOUBLE PRECISION NOT NULL,
    geom               geometry(Point, 4326),
    photo_url          TEXT,
    captured_at        TIMESTAMPTZ,
    device_heading     DOUBLE PRECISION,
    session_id         TEXT,
    status             TEXT NOT NULL DEFAULT 'pending',
    source_type        TEXT,
    created_at         TIMESTAMPTZ,
    data               JSONB NOT NULL          -- full report dict (round-trip exact)
);
CREATE INDEX IF NOT EXISTS idx_reports_mode_status ON reports (mode, status);
CREATE INDEX IF NOT EXISTS idx_reports_mode_captured ON reports (mode, captured_at);
CREATE INDEX IF NOT EXISTS idx_reports_geom ON reports USING GIST (geom);
-- External agency identity: (agency, incident_id) is the dedup key for
-- government feeds (CAP sender+identifier, or a namespaced feed item id).
-- Citizen/photo reports keep both NULL; a plain unique index ignores NULLs
-- in Postgres, so existing rows never collide. Idempotent migration for
-- databases created before these columns existed.
ALTER TABLE reports ADD COLUMN IF NOT EXISTS agency TEXT;
ALTER TABLE reports ADD COLUMN IF NOT EXISTS incident_id TEXT;
ALTER TABLE reports ADD COLUMN IF NOT EXISTS sent_at TIMESTAMPTZ;
CREATE UNIQUE INDEX IF NOT EXISTS uq_reports_agency_incident
    ON reports (mode, agency, incident_id);

CREATE TABLE IF NOT EXISTS bayesian_grids (
    id               TEXT PRIMARY KEY,
    mode             TEXT NOT NULL DEFAULT 'production',
    centroid_lat     DOUBLE PRECISION NOT NULL,
    centroid_lon     DOUBLE PRECISION NOT NULL,
    geom             geometry(Point, 4326),
    wind_speed       DOUBLE PRECISION NOT NULL DEFAULT 3.0,
    wind_dir_deg     DOUBLE PRECISION NOT NULL DEFAULT 270.0,
    state            JSONB NOT NULL,
    max_p            DOUBLE PRECISION NOT NULL DEFAULT 0.01,
    -- Epoch (unix) time of the NEWEST evidence fused into this grid
    -- (max of the grid's per-cell last_updated array). Only evidence
    -- injection updates it — predict()/map polling never does — so it
    -- is a true "is this fire still being detected?" signal. Grids whose
    -- last evidence is older than the expiry window are purged.
    last_evidence_at DOUBLE PRECISION NOT NULL DEFAULT 0,
    -- Epoch (unix) time when this grid's wind was last refreshed from the
    -- weather API. 0 = never (created before weather existed, or the fetch
    -- fell back to defaults) — the periodic wind-refresh job picks those up.
    wind_updated_at   DOUBLE PRECISION NOT NULL DEFAULT 0,
    -- Fuel-moisture / fire-weather indices from EFFIS (daily; 0 = not yet
    -- fetched, fetch fell back, or the fire is outside EFFIS coverage).
    -- ffmc/dmc/isi are raw FWI-system index values; fwi_updated_at is the
    -- epoch of the last successful fetch (0 = never — the periodic
    -- refresh picks those up, and the advance job scales spread by them).
    ffmc             DOUBLE PRECISION NOT NULL DEFAULT 0,
    dmc              DOUBLE PRECISION NOT NULL DEFAULT 0,
    isi              DOUBLE PRECISION NOT NULL DEFAULT 0,
    fwi_updated_at   DOUBLE PRECISION NOT NULL DEFAULT 0,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- Existing databases: add the column idempotently (CREATE IF NOT EXISTS
-- above won't touch an existing table).
ALTER TABLE bayesian_grids ADD COLUMN IF NOT EXISTS last_evidence_at DOUBLE PRECISION NOT NULL DEFAULT 0;
ALTER TABLE bayesian_grids ADD COLUMN IF NOT EXISTS wind_updated_at DOUBLE PRECISION NOT NULL DEFAULT 0;
ALTER TABLE bayesian_grids ADD COLUMN IF NOT EXISTS ffmc DOUBLE PRECISION NOT NULL DEFAULT 0;
ALTER TABLE bayesian_grids ADD COLUMN IF NOT EXISTS dmc DOUBLE PRECISION NOT NULL DEFAULT 0;
ALTER TABLE bayesian_grids ADD COLUMN IF NOT EXISTS isi DOUBLE PRECISION NOT NULL DEFAULT 0;
ALTER TABLE bayesian_grids ADD COLUMN IF NOT EXISTS fwi_updated_at DOUBLE PRECISION NOT NULL DEFAULT 0;
CREATE INDEX IF NOT EXISTS idx_grids_mode_geom ON bayesian_grids USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_grids_mode_maxp ON bayesian_grids (mode, max_p DESC);
CREATE INDEX IF NOT EXISTS idx_grids_mode_evidence ON bayesian_grids (mode, last_evidence_at);

CREATE TABLE IF NOT EXISTS osm_road_cache (
    cache_key  TEXT PRIMARY KEY,
    segments   JSONB,
    stored_at  DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS geocode_cache (
    query_key  TEXT PRIMARY KEY,
    results    JSONB NOT NULL,
    stored_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS kv_store (
    key    TEXT PRIMARY KEY,
    value  JSONB
);
-- Add created_at for TTL-based cleanup (idempotent — safe to re-run)
ALTER TABLE kv_store ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT now();
UPDATE kv_store SET created_at = now() WHERE created_at IS NULL;

-- Public feedback / suggestions / bug reports submitted from the FAQ page.
-- Unauthenticated by design (anyone can submit); rate-limited per IP in
-- the endpoint. Messages are reviewed by the operator in the admin flow.
CREATE TABLE IF NOT EXISTS feedback (
    id          TEXT PRIMARY KEY,
    category    TEXT NOT NULL,          -- suggestion | bug | question | other
    name        TEXT,
    email       TEXT,
    message     TEXT NOT NULL,
    page        TEXT,                   -- which page the form was on ('faq')
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_feedback_created ON feedback (created_at DESC);
"""


def init_schema(install_procrastinate: bool = True) -> None:
    """Create all WildFrame tables plus indexes. Idempotent.

    ``install_procrastinate`` also installs the Procrastinate job-queue
    schema (needed before any worker starts)."""
    with _conn() as conn:
        with conn.transaction():
            conn.execute(SCHEMA_SQL)
    if install_procrastinate:
        # Same path the `procrastinate schema --apply` CLI uses: open the
        # app (initializes the connector's pool) then apply the schema.
        # The schema SQL contains PL/pgSQL bodies with `%` format specifiers,
        # so it must go through Procrastinate's own apply_schema() (which
        # escapes/unescapes them correctly) rather than a raw execute.
        # NOTE: the schema SQL is NOT idempotent, so only apply it when the
        # queue tables don't exist yet.
        from procrastinate import App, PsycopgConnector

        with _conn() as conn:
            exists = conn.execute(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_name = 'procrastinate_jobs'"
            ).fetchone()
        if not exists:
            app = App(connector=PsycopgConnector(conninfo=DATABASE_URL))
            with app.open():
                app.schema_manager.apply_schema()


# ---------------------------------------------------------------------------
# Row ↔ dict mapping
# ---------------------------------------------------------------------------

def _row_to_report(row: dict) -> dict:
    """Reconstruct the full report dict from a row. ``data`` is the exact
    original dict; ``status`` is authoritative (kept in sync on writes)."""
    report = dict(row.get("data") or {})
    report["status"] = row["status"]
    return report


def _parse_ts(ts: Any) -> Optional[datetime]:
    """Best-effort parse of an ISO timestamp string into an aware datetime."""
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    if not isinstance(ts, str) or not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def insert_reports(reports: list[dict], mode: str) -> int:
    """Insert report rows. Pure INSERT — no read-modify-write, so
    concurrent inserts can never lose updates."""
    if not reports:
        return 0
    with _conn() as conn:
        with conn.transaction():
            for r in reports:
                captured = _parse_ts(r.get("captured_at"))
                created = _parse_ts(r.get("created_at"))
                conn.execute(
                    """
                    INSERT INTO reports
                        (id, mode, lat, lon, geom, photo_url, captured_at,
                         device_heading, session_id, status, source_type,
                         created_at, data)
                    VALUES (%s, %s, %s, %s,
                            ST_SetSRID(ST_MakePoint(%s, %s), 4326),
                            %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        r["id"], mode, r["lat"], r["lon"],
                        r["lon"], r["lat"],
                        r.get("photo_url"), captured,
                        r.get("device_heading"), r.get("session_id"),
                        r.get("status", "pending"), r.get("source_type"),
                        created, Jsonb(r),
                    ),
                )
    return len(reports)


def upsert_agency_incident(incident: dict, mode: str) -> dict:
    """Insert or update one agency incident, keyed by ``(mode, agency, incident_id)``.

    Idempotent by construction: a retry of the same message (or an older
    one) is a no-op thanks to the unique index + staleness guard, so a
    redelivered CAP alert or a Lambda retry can never create a duplicate
    fire or resurrect a cancelled one.

    ``incident`` is a full report dict plus at least:
      - ``agency``       : source (CAP sender, or namespaced feed name)
      - ``incident_id``  : CAP identifier, or the feed item id
      - ``sent_at``      : message time (ISO) — the staleness clock. Only a
                           NEWER sent_at can overwrite an existing row, so
                           out-of-order delivery can't regress a report.

    ``mode`` must be "production" or "demo". Returns ``(report, created, applied)``
    where ``report`` is the authoritative dict as stored (newest version wins),
    ``created`` is True only when this call actually inserted a fresh row, and
    ``applied`` is True only when the row was actually inserted or updated
    (False for a stale rejection or a pure duplicate retry — use it to gate
    downstream effects like grid evidence fusion).
    """
    assert incident.get("agency") and incident.get("incident_id")
    captured = _parse_ts(incident.get("captured_at"))
    created_ts = _parse_ts(incident.get("created_at"))
    sent_at = _parse_ts(incident.get("sent_at"))
    with _conn() as conn:
        with conn.transaction():
            # RETURNING (xmax = 0) is Postgres' classic "was this a fresh
            # insert?" probe: a brand-new tuple has xmax = 0, an updated one
            # (DO UPDATE fired) does not. Note: when the staleness WHERE
            # clause rejects the update, the statement returns NO row at all
            # (neither inserted nor updated), so we fall back to a re-select
            # below.
            result = conn.execute(
                """
                INSERT INTO reports
                    (id, mode, lat, lon, geom, photo_url, captured_at,
                     device_heading, session_id, status, source_type,
                     created_at, data, agency, incident_id, sent_at)
                VALUES (%s, %s, %s, %s,
                        ST_SetSRID(ST_MakePoint(%s, %s), 4326),
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (mode, agency, incident_id) DO UPDATE SET
                    lat = EXCLUDED.lat,
                    lon = EXCLUDED.lon,
                    geom = EXCLUDED.geom,
                    photo_url = EXCLUDED.photo_url,
                    captured_at = EXCLUDED.captured_at,
                    device_heading = EXCLUDED.device_heading,
                    session_id = EXCLUDED.session_id,
                    status = EXCLUDED.status,
                    source_type = EXCLUDED.source_type,
                    created_at = EXCLUDED.created_at,
                    data = EXCLUDED.data,
                    sent_at = EXCLUDED.sent_at
                WHERE EXCLUDED.sent_at > reports.sent_at
                   OR (EXCLUDED.sent_at IS NULL AND reports.sent_at IS NULL)
                RETURNING id, status, data, (xmax = 0) AS inserted
                """,
                (
                    incident["id"], mode, incident["lat"], incident["lon"],
                    incident["lon"], incident["lat"],
                    incident.get("photo_url"), captured,
                    incident.get("device_heading"), incident.get("session_id"),
                    (incident.get("status") or "pending"), incident.get("source_type"),
                    created_ts, Jsonb(incident),
                    incident["agency"], incident["incident_id"], sent_at,
                ),
            ).fetchone()
            if result is None:
                # The staleness guard rejected the update — an existing row
                # is newer, so nothing changed. Return the authoritative row.
                row = conn.execute(
                    "SELECT id, status, data FROM reports "
                    "WHERE mode = %s AND agency = %s AND incident_id = %s",
                    (mode, incident["agency"], incident["incident_id"]),
                ).fetchone()
                return _row_to_report(row), False, False
    return _row_to_report(result), bool(result["inserted"]), True


def find_grid_near(
    mode: str,
    lat: float,
    lon: float,
    max_dist_m: float = GRID_MATCH_RADIUS_M,
) -> Optional[str]:
    """Return the id of the nearest grid within ``max_dist_m`` of a point,
    or None. Used to target an existing grid with evidence (e.g. an agency
    cancel) WITHOUT creating a new grid — cancels must not spin up fires."""
    with _conn() as conn:
        nearest = _nearest_grid(conn, mode, lat, lon)
        if nearest is not None and nearest["dist_m"] <= max_dist_m:
            return nearest["id"]
    return None


def list_reports(
    mode: str,
    since_hours: Optional[float] = None,
    status: Optional[str] = None,
) -> list[dict]:
    """Fetch reports for a mode, optionally filtered by recency/status."""
    clauses = ["mode = %s"]
    params: list[Any] = [mode]
    if since_hours is not None:
        clauses.append("captured_at >= %s")
        params.append(datetime.now(timezone.utc) - timedelta(hours=since_hours))
    if status is not None:
        clauses.append("status = %s")
        params.append(status)
    sql = (
        "SELECT id, status, data FROM reports WHERE "
        + " AND ".join(clauses)
        + " ORDER BY captured_at DESC NULLS LAST"
    )
    with _conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_report(r) for r in rows]


def get_report(report_id: str, mode: str = "production") -> Optional[dict]:
    """Fetch one report. Mode-scoped so a demo id can never resolve a
    production report (or vice versa) — the DB layer enforces isolation
    even if a caller forgets."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT id, status, data FROM reports WHERE id = %s AND mode = %s",
            (report_id, mode),
        ).fetchone()
    return _row_to_report(row) if row else None


def update_report_status(
    report_id: str, new_status: str, mode: str = "production"
) -> Optional[dict]:
    """Atomic single-row status update (mode-scoped). Returns the updated
    report or None."""
    with _conn() as conn:
        with conn.transaction():
            row = conn.execute(
                """
                UPDATE reports
                   SET status = %s,
                       data = jsonb_set(data, '{status}', to_jsonb(%s::text))
                 WHERE id = %s AND mode = %s
                 RETURNING id, status, data
                """,
                (new_status, new_status, report_id, mode),
            ).fetchone()
    return _row_to_report(row) if row else None


def update_report(report: dict, mode: str) -> bool:
    """Wholesale update of one report's stored dict (used when nested
    fields like ``satellite_confirmation`` change). Mode-scoped; returns
    True if found and updated."""
    captured = _parse_ts(report.get("captured_at"))
    created = _parse_ts(report.get("created_at"))
    with _conn() as conn:
        with conn.transaction():
            cur = conn.execute(
                """
                UPDATE reports
                   SET data = %s,
                       lat = %s, lon = %s,
                       geom = ST_SetSRID(ST_MakePoint(%s, %s), 4326),
                       captured_at = %s, created_at = %s,
                       status = %s, source_type = %s
                 WHERE id = %s AND mode = %s
                """,
                (
                    Jsonb(report), report["lat"], report["lon"],
                    report["lon"], report["lat"],
                    captured, created,
                    report.get("status", "pending"),
                    report.get("source_type"),
                    report["id"], mode,
                ),
            )
            return cur.rowcount > 0


def accept_all_pending(mode: str) -> list[dict]:
    """Confirm every pending report in one atomic UPDATE. Returns the
    updated reports (for Bayesian grid feeding)."""
    with _conn() as conn:
        with conn.transaction():
            rows = conn.execute(
                """
                UPDATE reports
                   SET status = 'confirmed',
                       data = jsonb_set(data, '{status}', '"confirmed"')
                 WHERE mode = %s AND status = 'pending'
                 RETURNING id, status, data
                """,
                (mode,),
            ).fetchall()
    return [_row_to_report(r) for r in rows]


def delete_report(report_id: str, mode: str = "production") -> Optional[dict]:
    """Delete one report (mode-scoped). Returns the deleted report or None.

    Also reverses the report's grid evidence: a deleted report must not
    keep a fire alive on the map (the grid it seeded would otherwise keep
    its fused probability — and its wind badge — until the 24h stale
    purge). Negative evidence never stamps ``last_updated``, so a delete
    can't keep the grid alive either.
    """
    with _conn() as conn:
        with conn.transaction():
            row = conn.execute(
                "DELETE FROM reports WHERE id = %s AND mode = %s "
                "RETURNING id, status, data",
                (report_id, mode),
            ).fetchone()
    deleted = _row_to_report(row) if row else None
    if deleted is not None:
        _reverse_report_evidence(deleted, mode)
    return deleted


def _reverse_report_evidence(report: dict, mode: str) -> None:
    """Reverse a deleted confirmed report's contribution to its grid.

    The grid near the report's location gets the same evidence the report
    originally fused, but with a NEGATED log-likelihood ratio (the
    ``agency_cancel`` pattern). Because ``update()`` only stamps
    ``last_updated`` for positive evidence, a delete never resets the
    grid's expiry clock.
    """
    from bayesian_filter import Evidence  # lazy: db.py already imports the grid

    if report.get("status") != "confirmed":
        return  # only confirmed reports ever fed the grid
    lat, lon = report.get("lat"), report.get("lon")
    if lat is None or lon is None:
        return
    grid_id = find_grid_near(mode, float(lat), float(lon))
    if grid_id is None:
        return

    def _reverse(grid: "BayesianFireGrid", entry: dict) -> None:
        ev = Evidence.from_report(report, wind_dir_deg=entry.get("wind_dir_deg"))
        ev.log_likelihood_ratio = -ev.log_likelihood_ratio
        ev.source = f"report-removed:{report.get('id')}"
        grid.update(ev)

    mutate_grid(mode, grid_id, _reverse)


def count_reports(mode: str) -> int:
    with _conn() as conn:
        return conn.execute(
            "SELECT count(*) AS n FROM reports WHERE mode = %s", (mode,)
        ).fetchone()["n"]


# ---------------------------------------------------------------------------
# Bayesian grids
# ---------------------------------------------------------------------------

def _grid_from_row(row: dict) -> BayesianFireGrid:
    return BayesianFireGrid.from_dict(row["state"])


def _entry_from_row(row: dict) -> dict:
    """Registry-entry-shaped dict (matches server.py's old entry shape)."""
    return {
        "grid": _grid_from_row(row),
        "centroid_lat": row["centroid_lat"],
        "centroid_lon": row["centroid_lon"],
        "wind_speed": row["wind_speed"],
        "wind_dir_deg": row["wind_dir_deg"],
        "wind_updated_at": row["wind_updated_at"],
        "ffmc": row["ffmc"],
        "dmc": row["dmc"],
        "isi": row["isi"],
        "fwi_updated_at": row["fwi_updated_at"],
        "max_p": row["max_p"],
    }


def count_grids(mode: str) -> int:
    with _conn() as conn:
        return conn.execute(
            "SELECT count(*) AS n FROM bayesian_grids WHERE mode = %s",
            (mode,),
        ).fetchone()["n"]


def list_grid_meta(
    mode: str,
    bbox: Optional[tuple[float, float, float, float]] = None,
    limit: Optional[int] = None,
    fwi_first: bool = False,
) -> list[dict]:
    """List grid metadata (id, centroid, wind, max_p) for a mode, filtered
    to a viewport bbox (west,south,east,north) via PostGIS and ordered by
    peak probability. Does NOT load the heavy numpy state.

    ``fwi_first=True`` orders grids WITH EFFIS fuel-moisture data
    (ffmc > 0) before the rest — done IN SQL, before any LIMIT, so a
    bounded window can never exclude the data-bearing fires the admin
    dashboard exists to spot-check."""
    clauses = ["mode = %s"]
    params: list[Any] = [mode]
    if bbox is not None:
        w, s, e, n = bbox
        # Include any grid whose CELL EXTENT could reach into the viewport,
        # not just grids whose centroid is inside it. Without this, zooming
        # into the edge of a big fire (or between two fires) drops the whole
        # grid — the fire's centroid is outside the viewport bbox — so the
        # visible part of the fire silently vanishes from the heatmap.
        #
        # IMPORTANT: this must stay a CONSTANT expansion, not a per-row
        # expression. A per-row ST_Expand(GREATEST(COALESCE(state->>...)...))
        # reads the state JSONB for every row in the table, which defeats
        # the GiST index on geom and falls back to a full Parallel Seq Scan
        # (~940ms on the production fleet vs ~15ms indexed). Grids are
        # bounded (nx*cell/2 half-extent, metres->degrees via the E-W
        # conversion ÷cos(lat)) — the largest half-extent on the production
        # fleet is ~0.23°, so a constant 0.3° margin covers every grid's
        # cells (with headroom for future larger grids) while keeping the
        # index usable. The client culls off-screen cells anyway, so a hair
        # of over-fetch near the viewport edge is harmless.
        clauses.append(
            "geom && ST_Expand(ST_MakeEnvelope(%s, %s, %s, %s, 4326), 0.3)"
        )
        params.extend([w, s, e, n])
        if not fwi_first:
            # Visible-first ordering: grids whose CELL EXTENT reaches the
            # actual viewport always rank above margin-only grids (those
            # whose centroid is within the 0.3° margin but whose cells are
            # off-screen). The caller applies a LIMIT cap to bound the
            # payload — with a plain "max_p DESC" order, the cap could
            # drop a fire whose cells are ON SCREEN in favour of an
            # off-screen fire (in dense regions ~700 of ~800 visible fires
            # were silently dropped, rendering blobs "cut" / in parts).
            # The half-extent is read from each candidate row's state JSONB
            # AFTER the indexed geom filter — the same pattern as the
            # last_predict_time extraction below — so only the ~hundreds of
            # filtered rows pay the JSONB read, never a full-table scan.
            half_m = (
                "((state->>'nx')::float "
                "* COALESCE(NULLIF(state->>'cell_size_m','')::float, 100)) / 2.0"
            )
            half_lon = f"({half_m}) / (111320.0 * cos(radians(centroid_lat)))"
            half_lat = f"({half_m}) / 111320.0"
            visible = (
                f"(centroid_lon + ({half_lon}) >= {w!r} "
                f"AND centroid_lon - ({half_lon}) <= {e!r} "
                f"AND centroid_lat + ({half_lat}) >= {s!r} "
                f"AND centroid_lat - ({half_lat}) <= {n!r})"
            )
            order = f"CASE WHEN {visible} THEN 0 ELSE 1 END, max_p DESC"
        else:
            # fwi_first with a viewport: keep the admin's fuel-moisture
            # ordering (no current caller combines the two).
            order = "(ffmc > 0) DESC, max_p DESC"
    elif fwi_first:
        # Admin dashboard path: fuel-moisture-bearing grids first, always.
        order = "(ffmc > 0) DESC, max_p DESC"
    else:
        order = "max_p DESC"
    sql = (
        "SELECT id, centroid_lat, centroid_lon, wind_speed, wind_dir_deg, "
        "max_p, updated_at, wind_updated_at, last_evidence_at, "
        "ffmc, dmc, isi, fwi_updated_at, "
        # last_predict_time lives inside the state JSONB; expose it here so
        # the export-cache key can be computed WITHOUT loading/deserializing
        # the full numpy state (the state endpoint checks its cache first).
        # Mirror BayesianFireGrid.from_dict's fallback (0.0) so the
        # cache-first key always matches the miss-path key for legacy grids
        # whose state predates the last_predict_time key.
        "COALESCE((state->>'last_predict_time')::float, 0.0) AS last_predict_time "
        "FROM bayesian_grids WHERE " + " AND ".join(clauses) +
        " ORDER BY " + order
    )
    if limit:
        sql += " LIMIT %s"
        params.append(limit)
    with _conn() as conn:
        return conn.execute(sql, params).fetchall()


def get_grid_entry(mode: str, grid_id: str) -> Optional[dict]:
    """Load one grid (no lock) as an entry dict, or None if missing."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM bayesian_grids WHERE id = %s AND mode = %s",
            (grid_id, mode),
        ).fetchone()
    return _entry_from_row(row) if row else None


def get_grid_entries_batch(mode: str, grid_ids: list[str]) -> dict[str, dict]:
    """Load many grids in ONE query, returned as entry dicts keyed by id.

    Replaces the per-grid ``get_grid_entry`` N+1 pattern in the state
    export / road-risk paths (121 queries for 120 grids → 1). Returns an
    empty dict for an empty input; missing ids are simply absent."""
    if not grid_ids:
        return {}
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM bayesian_grids WHERE mode = %s AND id = ANY(%s)",
            (mode, grid_ids),
        ).fetchall()
    return {r["id"]: _entry_from_row(r) for r in rows}


def mutate_grid(
    mode: str,
    grid_id: str,
    fn: Callable[[BayesianFireGrid, dict], Any],
) -> Optional[Any]:
    """
    Atomically load → mutate → save one grid.

    The row is locked with ``SELECT ... FOR UPDATE`` inside a transaction,
    ``fn(grid, entry)`` is called, and any in-place changes to ``grid``
    (and to the entry's wind/centroid fields) are persisted before the
    lock releases. Concurrent workers therefore serialize per grid instead
    of clobbering each other's evidence/predict steps.

    Returns fn's return value, or None if the grid doesn't exist.
    """
    with _conn() as conn:
        with conn.transaction():
            row = conn.execute(
                "SELECT * FROM bayesian_grids WHERE id = %s AND mode = %s FOR UPDATE",
                (grid_id, mode),
            ).fetchone()
            if row is None:
                return None
            grid = _grid_from_row(row)
            entry = {
                "centroid_lat": row["centroid_lat"],
                "centroid_lon": row["centroid_lon"],
                "wind_speed": row["wind_speed"],
                "wind_dir_deg": row["wind_dir_deg"],
                "ffmc": row["ffmc"],
                "dmc": row["dmc"],
                "isi": row["isi"],
            }
            result = fn(grid, entry)
            max_p = float(grid.get_statistics()["max_p"])
            # Newest evidence timestamp in the grid (0 if never updated).
            last_evidence_at = float(grid.last_updated.max())
            conn.execute(
                """
                UPDATE bayesian_grids
                   SET state = %s,
                       centroid_lat = %s, centroid_lon = %s,
                       geom = ST_SetSRID(ST_MakePoint(%s, %s), 4326),
                       wind_speed = %s, wind_dir_deg = %s,
                       ffmc = %s, dmc = %s, isi = %s,
                       max_p = %s,
                       last_evidence_at = %s,
                       updated_at = now()
                 WHERE id = %s AND mode = %s
                """,
                (
                    Jsonb(grid.to_dict()),
                    entry["centroid_lat"], entry["centroid_lon"],
                    entry["centroid_lon"], entry["centroid_lat"],
                    entry["wind_speed"], entry["wind_dir_deg"],
                    entry["ffmc"], entry["dmc"], entry["isi"],
                    max_p,
                    last_evidence_at,
                    grid_id, mode,
                ),
            )
            return result


def _next_grid_id(conn, mode: str) -> int:
    """Atomically bump the GLOBAL grid id counter.

    ``id`` is the primary key of ``bayesian_grids``, so ids must be unique
    across ALL modes — a per-mode counter could emit ``grid-1`` for two
    different modes and collide. Demo grids get the ``demo-`` prefix, which
    already keeps them distinct from production ids."""
    key = "grid_counter:global"
    row = conn.execute(
        """
        INSERT INTO kv_store (key, value) VALUES (%s, '{"n": 1}')
        ON CONFLICT (key) DO UPDATE
            SET value = jsonb_build_object('n', (kv_store.value->>'n')::int + 1)
        RETURNING value->>'n' AS n
        """,
        (key,),
    ).fetchone()
    return int(row["n"])


def _nearest_grid(
    conn,
    mode: str,
    clat: float,
    clon: float,
) -> Optional[dict]:
    """Nearest grid row for a point (via PostGIS <-> ordering)."""
    return conn.execute(
        """
        SELECT id, centroid_lat, centroid_lon, wind_speed, wind_dir_deg, max_p,
               wind_updated_at, ffmc, dmc, isi, fwi_updated_at,
               ST_Distance(
                   geom::geography,
                   ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography
               ) AS dist_m
        FROM bayesian_grids
        WHERE mode = %s
        ORDER BY geom <-> ST_SetSRID(ST_MakePoint(%s, %s), 4326)
        LIMIT 1
        """,
        (clon, clat, mode, clon, clat),
    ).fetchone()


def find_or_create_grid(
    mode: str,
    cluster: dict,
    wind_speed: float = 3.0,
    wind_dir_deg: float = 270.0,
    wind_updated_at: float = 0.0,
    cell_size_m: float = DEFAULT_CELL_SIZE_M,
) -> tuple[str, dict]:
    """
    Return the grid tracking this cluster's fire, creating one if needed.

    Creation is serialized across workers with a Postgres advisory lock on
    a coarse location bucket, and the nearest-grid check is repeated inside
    the lock, so two concurrent requests can't spin up duplicate grids for
    the same new fire.

    Returns (grid_id, entry) where entry has a freshly-loaded grid.
    """
    clat, clon = cluster["centroid_lat"], cluster["centroid_lon"]
    # Coarse (~0.1° ≈ 11 km) bucket: coarser than GRID_MATCH_RADIUS_M so
    # clusters of the same new fire (which are found by a 10 km nearest
    # search) contend on the same lock. A finer bucket could let two
    # concurrent requests for one fire grab different locks and both
    # create a duplicate grid.
    lock_key = f"gridloc:{mode}:{clat:.1f}:{clon:.1f}"

    with _conn() as conn:
        with conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (lock_key,))

            nearest = _nearest_grid(conn, mode, clat, clon)
            if nearest is not None and nearest["dist_m"] <= GRID_MATCH_RADIUS_M:
                # Same fire — keep the tracked centroid current.
                conn.execute(
                    """
                    UPDATE bayesian_grids
                       SET centroid_lat = %s, centroid_lon = %s,
                           geom = ST_SetSRID(ST_MakePoint(%s, %s), 4326)
                     WHERE id = %s
                    """,
                    (clat, clon, clon, clat, nearest["id"]),
                )
                return nearest["id"], {
                    "grid": _grid_from_row(
                        conn.execute(
                            "SELECT * FROM bayesian_grids WHERE id = %s",
                            (nearest["id"],),
                        ).fetchone()
                    ),
                    "centroid_lat": clat,
                    "centroid_lon": clon,
                    "wind_speed": nearest["wind_speed"],
                    "wind_dir_deg": nearest["wind_dir_deg"],
                    "wind_updated_at": nearest["wind_updated_at"],
                    "ffmc": nearest["ffmc"],
                    "dmc": nearest["dmc"],
                    "isi": nearest["isi"],
                    "fwi_updated_at": nearest["fwi_updated_at"],
                    "max_p": nearest["max_p"],
                }

            # --- No nearby grid — create a fresh one sized to this cluster ---
            lats = [p[0] for p in cluster.get("points", [[clat, clon]])]
            lons = [p[1] for p in cluster.get("points", [[clat, clon]])]
            sizing = auto_grid_size(lats, lons, cell_size_m=cell_size_m) or {
                "center_lat": clat, "center_lon": clon,
                "cell_size_m": cell_size_m, "nx": 40, "ny": 40,
            }
            grid = BayesianFireGrid(
                center_lat=sizing["center_lat"],
                center_lon=sizing["center_lon"],
                cell_size_m=sizing["cell_size_m"],
                nx=sizing["nx"],
                ny=sizing["ny"],
            )
            prefix = "demo-" if mode == "demo" else ""
            grid_id = f"{prefix}grid-{_next_grid_id(conn, mode)}"

            conn.execute(
                """
                INSERT INTO bayesian_grids
                    (id, mode, centroid_lat, centroid_lon, geom,
                     wind_speed, wind_dir_deg, state, max_p,
                     last_evidence_at, wind_updated_at)
                VALUES (%s, %s, %s, %s,
                        ST_SetSRID(ST_MakePoint(%s, %s), 4326),
                        %s, %s, %s, %s, %s, %s)
                """,
                (
                    grid_id, mode, clat, clon, clon, clat,
                    wind_speed, wind_dir_deg,
                    Jsonb(grid.to_dict()),
                    float(grid.get_statistics()["max_p"]),
                    float(grid.last_updated.max()),  # 0 — fresh grid, no evidence yet
                    wind_updated_at,
                ),
            )
            return grid_id, {
                "grid": grid,
                "centroid_lat": clat,
                "centroid_lon": clon,
                "wind_speed": wind_speed,
                "wind_dir_deg": wind_dir_deg,
                "wind_updated_at": wind_updated_at,
                "ffmc": 0.0,
                "dmc": 0.0,
                "isi": 0.0,
                "fwi_updated_at": 0.0,
                "max_p": float(grid.get_statistics()["max_p"]),
            }


def list_grids_needing_wind(
    mode: str,
    limit: int = 200,
    max_age_s: float = 24 * 60 * 60,
) -> list[dict]:
    """Return the oldest-refreshed grids whose wind is stale (or was never
    set — ``wind_updated_at = 0`` covers grids created before weather
    existed). The periodic wind-refresh job feeds these centroids to the
    weather API and calls :func:`update_grid_wind`."""
    cutoff = time.time() - max_age_s
    with _conn() as conn:
        return conn.execute(
            "SELECT id, centroid_lat, centroid_lon FROM bayesian_grids "
            "WHERE mode = %s AND wind_updated_at < %s "
            "ORDER BY wind_updated_at ASC LIMIT %s",
            (mode, cutoff, limit),
        ).fetchall()


def update_grid_wind(
    mode: str,
    grid_id: str,
    wind_speed: float,
    wind_dir_deg: float,
) -> bool:
    """Persist a freshly-fetched wind value on a grid and stamp
    ``wind_updated_at`` so the refresh job doesn't immediately re-fetch.
    Returns True if the grid was updated."""
    with _conn() as conn:
        with conn.transaction():
            cur = conn.execute(
                "UPDATE bayesian_grids "
                "SET wind_speed = %s, wind_dir_deg = %s, wind_updated_at = %s, "
                "    updated_at = now() "
                "WHERE id = %s AND mode = %s",
                (wind_speed, wind_dir_deg, time.time(), grid_id, mode),
            )
            return cur.rowcount > 0


def list_grids_needing_fwi(
    mode: str,
    limit: int = 200,
    max_age_s: float = 12 * 3600,
) -> list[dict]:
    """Return the oldest-refreshed grids whose fuel-moisture values are
    stale (or never set — ``fwi_updated_at = 0`` covers grids created
    before EFFIS integration). The periodic refresh job feeds these
    centroids to EFFIS and calls :func:`update_grid_fwi`."""
    cutoff = time.time() - max_age_s
    with _conn() as conn:
        return conn.execute(
            "SELECT id, centroid_lat, centroid_lon FROM bayesian_grids "
            "WHERE mode = %s AND fwi_updated_at < %s "
            "ORDER BY fwi_updated_at ASC LIMIT %s",
            (mode, cutoff, limit),
        ).fetchall()


def update_grid_fwi(
    mode: str,
    grid_id: str,
    ffmc: float,
    dmc: float,
    isi: float,
) -> bool:
    """Persist freshly-fetched EFFIS values on a grid and stamp
    ``fwi_updated_at`` so the refresh job doesn't immediately re-fetch.
    Returns True if the grid was updated."""
    with _conn() as conn:
        with conn.transaction():
            cur = conn.execute(
                "UPDATE bayesian_grids "
                "SET ffmc = %s, dmc = %s, isi = %s, fwi_updated_at = %s, "
                "    updated_at = now() "
                "WHERE id = %s AND mode = %s",
                (ffmc, dmc, isi, time.time(), grid_id, mode),
            )
            return cur.rowcount > 0


def touch_grid_fwi(mode: str, grid_id: str) -> bool:
    """Stamp ``fwi_updated_at`` WITHOUT values — used for grids that will
    never have EFFIS data (outside EMNA coverage) so the daily sweep stops
    rescanning them. Returns True if the grid was updated."""
    with _conn() as conn:
        with conn.transaction():
            cur = conn.execute(
                "UPDATE bayesian_grids "
                "SET fwi_updated_at = %s, updated_at = now() "
                "WHERE id = %s AND mode = %s",
                (time.time(), grid_id, mode),
            )
            return cur.rowcount > 0


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres (duplicated from server.py to avoid
    an import cycle — db.py is imported BY server.py)."""
    R = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _loc_cell(lat: float, lon: float, cell_deg: float) -> tuple[int, int]:
    """Spatial-hash cell for a lat/lon (longitude widened by 1/cos(lat))."""
    c = max(math.cos(math.radians(lat)), 0.3)
    return (math.floor(lon / (cell_deg / c)), math.floor(lat / cell_deg))


def _next_grid_ids(conn, mode: str, n: int) -> list[int]:
    """Atomically bump the GLOBAL grid id counter by n; return the n new ids.

    Same counter/prefix rules as _next_grid_id, but for bulk creation: one
    upsert instead of one per new grid."""
    if n <= 0:
        return []
    key = "grid_counter:global"
    row = conn.execute(
        """
        INSERT INTO kv_store (key, value) VALUES (%s, jsonb_build_object('n', %s))
        ON CONFLICT (key) DO UPDATE
            SET value = jsonb_build_object('n', (kv_store.value->>'n')::int + %s)
        RETURNING value->>'n' AS n
        """,
        (key, n, n),
    ).fetchone()
    end = int(row["n"])
    return list(range(end - n + 1, end + 1))


_BULK_GRID_INSERT_SQL = """
INSERT INTO bayesian_grids
    (id, mode, centroid_lat, centroid_lon, geom,
     wind_speed, wind_dir_deg, state, max_p,
     last_evidence_at, wind_updated_at)
VALUES (%s, %s, %s, %s,
        ST_SetSRID(ST_MakePoint(%s, %s), 4326),
        %s, %s, %s, %s, %s, %s)
"""

_BULK_GRID_CENTROID_SQL = """
UPDATE bayesian_grids
   SET centroid_lat = %s, centroid_lon = %s,
       geom = ST_SetSRID(ST_MakePoint(%s, %s), 4326)
 WHERE id = %s
"""


def bulk_find_or_create_grids(
    mode: str,
    clusters: list[dict],
    wind_speed: float = 3.0,
    wind_dir_deg: float = 270.0,
    wind_updated_at: float = 0.0,
    cell_size_m: float = DEFAULT_CELL_SIZE_M,
) -> list[str]:
    """Return one grid_id per cluster (in order), creating grids in bulk.

    Batch equivalent of find_or_create_grid() for the global FIRMS pass,
    which feeds ~10k-100k clusters per fetch. The per-cluster version costs
    one transaction + advisory lock + PostGIS nearest query PER CLUSTER
    (~25 ms each -> ~10-40 minutes). This version:

      1. Loads every existing grid's centroid ONCE (single query).
      2. Matches each cluster to the nearest grid within GRID_MATCH_RADIUS_M
         in memory via a spatial hash (same semantics as the DB nearest
         query, O(n) instead of O(n) round trips). A grid created earlier
         in THIS pass is added to the hash, so a later cluster can reuse
         it too — identical to the sequential find_or_create_grid
         behaviour.
      3. Bulk-inserts only the genuinely-new grids in ONE transaction
         (executemany + a single counter bump).

    Returns ``(grid_ids, new_ids)``: grid ids aligned with ``clusters``
    (same length, same order; matched grids keep their existing id, new
    grids get a fresh one) and the set of ids that were created by THIS
    call. The new-ids set is what lets the FIRMS pass key the Step 3
    escape hatch on "grid created this pass" (a fire whose grid pre-exists
    has persisted → full weight).

    Unlike find_or_create_grid(), this does NOT take the per-bucket
    Postgres advisory lock. Duplicate creation is prevented instead by the
    pass-level ``firms_fetch_in_progress`` flag (one fetch at a time); the
    advisory lock is dropped deliberately because it was the ~25ms/cluster
    round trip that made the sequential path too slow.
    """
    if not clusters:
        return []

    cell_deg = 0.15  # ~16 km — > 2x the 10 km match radius
    existing = list_grid_meta(mode)  # one query for all centroids

    # grid_id -> (lat, lon) tracked centroid; a brand-new grid gets a
    # placeholder key ("__new_0") so later clusters can match it in-memory
    # before real ids are allocated.
    centroids: dict[str, tuple[float, float]] = {
        row["id"]: (row["centroid_lat"], row["centroid_lon"]) for row in existing
    }
    by_cell: dict[tuple[int, int], list[str]] = {}
    for gid, (glat, glon) in centroids.items():
        by_cell.setdefault(_loc_cell(glat, glon, cell_deg), []).append(gid)

    def _recenter(gid: str, clat: float, clon: float) -> None:
        """Move a grid's tracked centroid (and its hash bucket) to clat/clon."""
        old_lat, old_lon = centroids[gid]
        if old_lat == clat and old_lon == clon:
            return
        centroids[gid] = (clat, clon)
        old_key = _loc_cell(old_lat, old_lon, cell_deg)
        bucket = by_cell.get(old_key)
        if bucket is not None:
            try:
                bucket.remove(gid)
            except ValueError:
                pass
            if not bucket:
                del by_cell[old_key]
        by_cell.setdefault(_loc_cell(clat, clon, cell_deg), []).append(gid)

    def _nearest(clat: float, clon: float):
        """(grid_id, dist_m) of the nearest known grid centroid."""
        best_id, best_d = None, float("inf")
        cx, cy = _loc_cell(clat, clon, cell_deg)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for gid in by_cell.get((cx + dx, cy + dy), ()):
                    glat, glon = centroids[gid]
                    d = _haversine_m(clat, clon, glat, glon)
                    if d < best_d:
                        best_d, best_id = d, gid
        return best_id, best_d

    # --- Single pass: assign every cluster to a grid (match or create) ---
    prefix = "demo-" if mode == "demo" else ""
    result: list[str] = []
    new_clusters: list[dict] = []        # clusters needing a brand-new grid
    new_centroids: list[tuple[float, float]] = []  # matching (clat, clon)
    centroid_moves: dict[str, tuple[float, float]] = {}  # existing grid -> new centroid
    placeholder_n = 0

    for c in clusters:
        clat, clon = c["centroid_lat"], c["centroid_lon"]
        gid, dist = _nearest(clat, clon)
        if gid is not None and dist <= GRID_MATCH_RADIUS_M:
            result.append(gid)
            _recenter(gid, clat, clon)
            if not gid.startswith("__new_"):
                centroid_moves[gid] = (clat, clon)
        else:
            # Reserve a fresh grid in-memory so later clusters can reuse it.
            placeholder = f"__new_{placeholder_n}"
            placeholder_n += 1
            result.append(placeholder)
            centroids[placeholder] = (clat, clon)
            by_cell.setdefault(_loc_cell(clat, clon, cell_deg), []).append(placeholder)
            new_clusters.append(c)
            new_centroids.append((clat, clon))

    # --- Bulk-create the new grids, one transaction ---
    new_ids: list[str] = []
    if new_clusters:
        with _conn() as conn:
            with conn.transaction():
                ids = [
                    f"{prefix}grid-{n}"
                    for n in _next_grid_ids(conn, mode, len(new_clusters))
                ]
                rows = []
                for grid_id, c, (clat, clon) in zip(ids, new_clusters, new_centroids):
                    lats = [p[0] for p in c.get("points", [[clat, clon]])]
                    lons = [p[1] for p in c.get("points", [[clat, clon]])]
                    sizing = auto_grid_size(lats, lons, cell_size_m=cell_size_m) or {
                        "center_lat": clat, "center_lon": clon,
                        "cell_size_m": cell_size_m, "nx": 40, "ny": 40,
                    }
                    grid = BayesianFireGrid(
                        center_lat=sizing["center_lat"],
                        center_lon=sizing["center_lon"],
                        cell_size_m=sizing["cell_size_m"],
                        nx=sizing["nx"], ny=sizing["ny"],
                    )
                    rows.append((
                        grid_id, mode, clat, clon, clon, clat,
                        wind_speed, wind_dir_deg,
                        Jsonb(grid.to_dict()),
                        float(grid.get_statistics()["max_p"]),
                        float(grid.last_updated.max()),  # 0 — fresh grid
                        wind_updated_at,
                    ))
                with conn.cursor() as cur:
                    cur.executemany(_BULK_GRID_INSERT_SQL, rows)
                new_ids = ids

    # --- Persist centroid moves for matched EXISTING grids (one UPDATE batch) ---
    if centroid_moves:
        moves = [
            (clat, clon, clon, clat, gid)
            for gid, (clat, clon) in centroid_moves.items()
        ]
        with _conn() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.executemany(_BULK_GRID_CENTROID_SQL, moves)

    # --- Swap placeholders for real ids ---
    id_by_placeholder = {
        f"__new_{i}": real for i, real in enumerate(new_ids)
    }
    resolved = [id_by_placeholder.get(gid, gid) for gid in result]
    return resolved, set(new_ids)


def bulk_mutate_grids(
    mode: str,
    jobs: list[tuple[str, Callable[[BayesianFireGrid, dict], Any]]],
    chunk: int = 250,
) -> dict[str, Any]:
    """Run ``fn(grid, entry)`` for many grids, persisting all in bulk.

    Batch equivalent of mutate_grid() for the global FIRMS pass (which
    feeds one grid per cluster — ~10k-100k grids). The per-grid version
    opens a transaction + FOR UPDATE lock + full reload + UPDATE for EACH
    grid; this version loads up to ``chunk`` grids with one ``id = ANY()``
    query, runs the callable on each in memory, and writes them back with
    one executemany — a handful of transactions instead of one per grid.

    jobs: list of (grid_id, fn) where fn(grid, entry) mutates the grid in
    place and returns a result value (stored under its grid_id).

    Returns {grid_id: fn_result} for grids that were loaded successfully.
    """
    if not jobs:
        return {}

    results: dict[str, Any] = {}
    for start in range(0, len(jobs), chunk):
        batch = jobs[start:start + chunk]
        ids = [gid for gid, _ in batch]
        with _conn() as conn:
            with conn.transaction():
                rows = conn.execute(
                    "SELECT * FROM bayesian_grids WHERE id = ANY(%s) AND mode = %s FOR UPDATE",
                    (ids, mode),
                ).fetchall()
                row_by_id = {r["id"]: r for r in rows}
                updates = []
                for gid, fn in batch:
                    row = row_by_id.get(gid)
                    if row is None:
                        continue
                    grid = _grid_from_row(row)
                    entry = {
                        "centroid_lat": row["centroid_lat"],
                        "centroid_lon": row["centroid_lon"],
                        "wind_speed": row["wind_speed"],
                        "wind_dir_deg": row["wind_dir_deg"],
                        "ffmc": row["ffmc"],
                        "dmc": row["dmc"],
                        "isi": row["isi"],
                    }
                    results[gid] = fn(grid, entry)
                    if not results[gid]:
                        # No-op mutation (e.g. the FIRMS dedup skipped every
                        # hotspot for this grid) — don't write it back. A
                        # write would bump updated_at for zero state change
                        # and invalidate the export cache keyed on it,
                        # forcing re-serialization on the next map poll.
                        continue
                    max_p = float(grid.get_statistics()["max_p"])
                    last_evidence_at = float(grid.last_updated.max())
                    updates.append((
                        Jsonb(grid.to_dict()),
                        entry["centroid_lat"], entry["centroid_lon"],
                        entry["centroid_lon"], entry["centroid_lat"],
                        entry["wind_speed"], entry["wind_dir_deg"],
                        entry["ffmc"], entry["dmc"], entry["isi"],
                        max_p, last_evidence_at, gid, mode,
                    ))
                with conn.cursor() as cur:
                    cur.executemany(
                        """
                        UPDATE bayesian_grids
                           SET state = %s,
                               centroid_lat = %s, centroid_lon = %s,
                               geom = ST_SetSRID(ST_MakePoint(%s, %s), 4326),
                               wind_speed = %s, wind_dir_deg = %s,
                               ffmc = %s, dmc = %s, isi = %s,
                               max_p = %s,
                               last_evidence_at = %s,
                               updated_at = now()
                         WHERE id = %s AND mode = %s
                        """,
                        updates,
                    )
    return results


def upsert_grid(
    mode: str,
    grid_id: str,
    grid: BayesianFireGrid,
    centroid_lat: float,
    centroid_lon: float,
    wind_speed: float = 3.0,
    wind_dir_deg: float = 270.0,
) -> None:
    """Insert or replace a grid under a fixed id (used by the historic
    Creek Fire replay which uses the stable key 'creek-fire-demo')."""
    with _conn() as conn:
        with conn.transaction():
            conn.execute(
                """
                INSERT INTO bayesian_grids
                    (id, mode, centroid_lat, centroid_lon, geom,
                     wind_speed, wind_dir_deg, state, max_p,
                     last_evidence_at)
                VALUES (%s, %s, %s, %s,
                        ST_SetSRID(ST_MakePoint(%s, %s), 4326),
                        %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    mode = EXCLUDED.mode,
                    centroid_lat = EXCLUDED.centroid_lat,
                    centroid_lon = EXCLUDED.centroid_lon,
                    geom = EXCLUDED.geom,
                    wind_speed = EXCLUDED.wind_speed,
                    wind_dir_deg = EXCLUDED.wind_dir_deg,
                    state = EXCLUDED.state,
                    max_p = EXCLUDED.max_p,
                    last_evidence_at = EXCLUDED.last_evidence_at,
                    updated_at = now()
                """,
                (
                    grid_id, mode, centroid_lat, centroid_lon,
                    centroid_lon, centroid_lat,
                    wind_speed, wind_dir_deg,
                    Jsonb(grid.to_dict()),
                    float(grid.get_statistics()["max_p"]),
                    float(grid.last_updated.max()),
                ),
            )


def seed_grid_counter_from_existing() -> Optional[int]:
    """Set the global grid-id counter above the highest numeric id already
    stored, so new grids never collide with existing ones (e.g. legacy
    grids imported before the counter existed). Returns the seeded value
    or None if there are no grids yet."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT max((regexp_replace(id, '^.*-', ''))::int) AS m "
            "FROM bayesian_grids WHERE id ~ '-[0-9]+$'"
        ).fetchone()
        if not row or row["m"] is None:
            return None
        top = int(row["m"])
        with conn.transaction():
            conn.execute(
                """
                INSERT INTO kv_store (key, value) VALUES ('grid_counter:global', %s)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                """,
                (Jsonb({"n": top}),),
            )
    return top


def delete_grids(mode: str) -> int:
    with _conn() as conn:
        with conn.transaction():
            cur = conn.execute(
                "DELETE FROM bayesian_grids WHERE mode = %s", (mode,)
            )
            return cur.rowcount


def delete_grid(mode: str, grid_id: str) -> bool:
    with _conn() as conn:
        with conn.transaction():
            cur = conn.execute(
                "DELETE FROM bayesian_grids WHERE id = %s AND mode = %s",
                (grid_id, mode),
            )
            return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Evidence-age expiry
#
# A grid's ``last_evidence_at`` column (epoch of its newest fused evidence)
# lets us expire fires that have stopped being detected: real FIRMS fires
# whose newest hotspot is older than the window are deleted so old fires
# don't linger on the map (or in the DB) indefinitely.
# ---------------------------------------------------------------------------


def advance_grids(
    mode: str,
    limit: int = 500,
    min_age_s: float = 60.0,
) -> int:
    """Advance a bounded, rotating slice of grids by elapsed time and
    persist the result (called by the worker's periodic 'grids.advance'
    job).

    Why rotation: with a global FIRMS dataset (thousands of grids) the
    worker cannot advance everything every minute — each advance rewrites
    the grid's JSONB state under a row lock. So each run advances the next
    ``limit`` grids by id (cursor kept in kv_store); every grid gets
    checkpointed on a rolling basis while no single run is expensive. The
    map's read path independently extrapolates fires in-memory between
    checkpoints, so animation stays smooth regardless.

    Each grid is advanced under its own row lock (mutate_grid), so
    concurrent evidence injection is never clobbered.

    Returns the number of grids actually advanced.
    """
    # EFFIS fuel-moisture conversions for the advance step. Lazy import at
    # function level: effis_fwi imports db, so a module-level import here
    # would be an import cycle (imported once per job run, not per grid).
    from effis_fwi import moisture_factor, decay_scale

    cursor_key = f"grids_advance_cursor:{mode}"
    cursor = kv_get(cursor_key) or ""

    with _conn() as conn:
        if cursor:
            rows = conn.execute(
                "SELECT id FROM bayesian_grids WHERE mode = %s AND id > %s "
                "ORDER BY id LIMIT %s",
                (mode, cursor, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id FROM bayesian_grids WHERE mode = %s "
                "ORDER BY id LIMIT %s",
                (mode, limit),
            ).fetchall()

    if not rows:
        kv_set(cursor_key, "")  # wrapped around — restart the sweep
        return 0

    now = time.time()
    advanced = 0
    for row in rows:
        grid_id = row["id"]

        def _advance(grid: BayesianFireGrid, entry: dict) -> bool:
            wind_speed = entry.get("wind_speed", 3.0)
            wind_dir_deg = entry.get("wind_dir_deg", 270.0)
            # EFFIS fuel moisture (0 = no data / outside coverage → neutral).
            ffmc = float(entry.get("ffmc") or 0.0)
            dmc = float(entry.get("dmc") or 0.0)
            mf = moisture_factor(ffmc) if ffmc > 0 else 1.0
            ds = decay_scale(dmc) if dmc > 0 else 1.0
            if grid.last_predict_time > 0:
                elapsed = now - grid.last_predict_time
                if elapsed < min_age_s:
                    return False  # checkpointed recently — skip
                grid.predict(
                    dt=min(elapsed, 600.0),
                    wind_speed=wind_speed,
                    wind_dir_deg=wind_dir_deg,
                    moisture_factor=mf,
                    decay_scale=ds,
                )
            else:
                grid.predict(
                    dt=60.0, wind_speed=wind_speed, wind_dir_deg=wind_dir_deg,
                    moisture_factor=mf, decay_scale=ds,
                )
            return True

        if mutate_grid(mode, grid_id, _advance):
            advanced += 1

    kv_set(cursor_key, rows[-1]["id"])
    return advanced


def purge_stale_grids(mode: str, max_age_hours: float = 24.0) -> int:
    """Delete grids whose newest evidence is older than ``max_age_hours``.

    Only grids that HAVE received evidence (``last_evidence_at > 0``) are
    candidates — a grid created but never fed (e.g. an admin-accepted
    report waiting for satellite confirmation) is not "stale" in this
    sense and is left alone.

    Returns the number of grids deleted.
    """
    cutoff = time.time() - max_age_hours * 3600.0
    with _conn() as conn:
        with conn.transaction():
            cur = conn.execute(
                "DELETE FROM bayesian_grids "
                "WHERE mode = %s AND last_evidence_at > 0 AND last_evidence_at < %s",
                (mode, cutoff),
            )
            return cur.rowcount


def backfill_last_evidence_at() -> int:
    """One-time migration: backfill ``last_evidence_at`` from each grid's
    serialized state (the per-cell ``last_updated`` array inside the JSONB
    state blob). Needed so an expiry sweep doesn't immediately delete
    grids that were created before the column existed (they'd all read 0).

    Returns the number of grids backfilled.
    """
    import base64
    import numpy as np

    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, state FROM bayesian_grids WHERE last_evidence_at = 0"
        ).fetchall()

    n = 0
    for row in rows:
        state = row["state"] or {}
        b64 = state.get("last_updated")
        ts = 0.0
        if isinstance(b64, str):
            try:
                raw = base64.b64decode(b64)
                arr = np.frombuffer(raw, dtype=np.float32)
                if arr.size:
                    ts = float(arr.max())
            except Exception:
                ts = 0.0
        if ts <= 0:
            continue
        with _conn() as conn:
            with conn.transaction():
                conn.execute(
                    "UPDATE bayesian_grids SET last_evidence_at = %s WHERE id = %s",
                    (ts, row["id"]),
                )
        n += 1
    return n


# ---------------------------------------------------------------------------
# kv_store — poller flags, counters, heartbeats
# ---------------------------------------------------------------------------

def kv_get(key: str, default: Any = None) -> Any:
    with _conn() as conn:
        row = conn.execute(
            "SELECT value FROM kv_store WHERE key = %s", (key,)
        ).fetchone()
    return row["value"] if row else default


def kv_get_many(keys: list[str]) -> dict[str, Any]:
    """Batch kv_store read in ONE query; returns ``{key: value}`` for the keys
    that exist (missing keys are simply absent). Empty input → {}."""
    if not keys:
        return {}
    with _conn() as conn:
        rows = conn.execute(
            "SELECT key, value FROM kv_store WHERE key = ANY(%s)", (keys,)
        ).fetchall()
    return {row["key"]: row["value"] for row in rows}


def submit_feedback(
    category: str,
    message: str,
    name: Optional[str] = None,
    email: Optional[str] = None,
    page: Optional[str] = None,
) -> str:
    """Persist a user-submitted suggestion / bug report / question.

    Returns the new feedback row id.
    """
    fid = uuid.uuid4().hex
    with _conn() as conn:
        with conn.transaction():
            conn.execute(
                """
                INSERT INTO feedback (id, category, name, email, message, page)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (fid, category, name, email, message, page),
            )
    return fid


def list_feedback(limit: int = 100) -> list[dict]:
    """List recent feedback submissions, newest first (admin review)."""
    with _conn() as conn:
        rows = conn.execute(
            """
            SELECT id, category, name, email, message, page, created_at
            FROM feedback
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (limit,),
        ).fetchall()
    out: list[dict] = []
    for row in rows:
        out.append({
            "id": row["id"],
            "category": row["category"],
            "name": row["name"],
            "email": row["email"],
            "message": row["message"],
            "page": row["page"],
            "created_at": (
                row["created_at"].isoformat() if row["created_at"] else None
            ),
        })
    return out


def delete_feedback(feedback_id: str) -> bool:
    """Delete one feedback submission. Returns True if a row was removed."""
    with _conn() as conn:
        with conn.transaction():
            cur = conn.execute(
                "DELETE FROM feedback WHERE id = %s", (feedback_id,)
            )
    return cur.rowcount > 0


def land_cover_codes_batch(points: list[tuple[float, float]]) -> dict[int, Optional[str]]:
    """Look up the CORINE land-cover class (``Code_18``) at each point.

    ``points`` is a list of ``(lat, lon)``. Returns ``{index: code_18}``
    for points that fall on a loaded ``land_cover`` polygon; points outside
    CORINE coverage (or on the sea) are simply absent from the result.

    Uses a temp table with a GiST index for the spatial join — much faster
    than the previous VALUES approach for large batches (>5k points), because
    Postgres can optimize the join plan and avoid re-parsing a massive SQL
    string with 30k+ bind parameters.
    """
    if not points:
        return {}
    bbox = _table_bbox("land_cover")
    if bbox is None:
        # Fail-open: land_cover not loaded on this environment (the CORINE
        # import hasn't run yet) — every point is treated as burnable rather
        # than crashing the FIRMS pass.
        return {}
    # Only points inside the layer's own coverage can hit a polygon — skip
    # the rest (they fail open as burnable anyway).
    keep = [(i, (lat, lon)) for i, (lat, lon) in enumerate(points)
            if _in_bbox(lat, lon, bbox)]
    if not keep:
        return {}
    out: dict[int, Optional[str]] = {}
    with _conn() as conn:
        # Temp table with a GiST index: Postgres can optimize the join plan
        # and avoid re-parsing a massive VALUES SQL string.
        conn.execute("CREATE TEMP TABLE _lc_pts (idx int, geom geometry) ON COMMIT DROP")
        conn.execute("CREATE INDEX ON _lc_pts USING gist (geom)")
        # Batch inserts (1k rows each) to stay under memory limits
        for start in range(0, len(keep), 1000):
            batch = keep[start:start + 1000]
            conn.execute(
                "INSERT INTO _lc_pts (idx, geom) VALUES "
                + ", ".join(
                    "(%s, ST_SetSRID(ST_MakePoint(%s, %s), 4326))"
                    for _ in batch
                ),
                [p for idx, (lat, lon) in batch for p in (idx, lon, lat)],
            )
        rows = conn.execute(
            """
            SELECT DISTINCT ON (p.idx) p.idx, lc.code_18
            FROM _lc_pts p
            JOIN land_cover lc ON ST_Intersects(lc.geom, p.geom)
            ORDER BY p.idx, lc.code_18
            """
        ).fetchall()
    for r in rows:
        out[int(r["idx"])] = str(r["code_18"])
    return out


def static_source_hits_batch(points: list[tuple[float, float]]) -> dict[int, bool]:
    """Return ``{index: True}`` for points landing on a flagged static-source
    cell (the ``static_sources`` mask built by ``build_static_mask.py``).

    Same batched GiST join shape as ``land_cover_codes_batch``. Fail-open:
    if the mask table is missing (not built yet on this environment), every
    point is absent from the result, so callers treat all hotspots as
    normal wildfire candidates.
    """
    if not points:
        return {}
    bbox = _table_bbox("static_sources")
    if bbox is None:
        # Fail-open: static_sources not built yet — no downweighting.
        return {}
    # Same coverage pre-filter + chunking rationale as land_cover_codes_batch
    # (see _table_bbox): the mask only covers the Poland footprint, so only
    # points inside it can ever be flagged static — trimming the global
    # ~250k-point pass to a handful of rows and keeping peak memory off the
    # 1 GB production VM. 10k rows/chunk stays under the 65,535-param limit.
    keep = [(i, (lat, lon)) for i, (lat, lon) in enumerate(points)
            if _in_bbox(lat, lon, bbox)]
    out: dict[int, bool] = {}
    for start in range(0, len(keep), _BATCH_LOOKUP_CHUNK):
        chunk = keep[start:start + _BATCH_LOOKUP_CHUNK]
        values_sql = ", ".join(
            "(%s, ST_SetSRID(ST_MakePoint(%s, %s), 4326))" for _ in chunk
        )
        params: list[Any] = []
        for idx, (lat, lon) in chunk:
            params.extend([idx, lon, lat])
        with _conn() as conn:
            rows = conn.execute(
                f"""
                SELECT DISTINCT ON (hs.idx) hs.idx
                FROM (VALUES {values_sql}) AS hs(idx, geom)
                JOIN static_sources ss ON ss.is_static
                    AND ST_Intersects(ss.geom, hs.geom)
                ORDER BY hs.idx
                """,
                params,
            ).fetchall()
        for r in rows:
            out[int(r["idx"])] = True
    return out


def worldcover_code_batch(points: list[tuple[float, float]]) -> dict[int, Optional[int]]:
    """Look up the ESA WorldCover class code at each point (global fallback).

    ``points`` is a list of ``(lat, lon)``.  Returns ``{index: code}`` for
    points that fall inside a loaded ``worldcover_polygons`` geometry row;
    points outside WorldCover coverage are absent from the result.

    This is the **global fallback** for the land-cover gate: CORINE covers
    Europe at 100 m; WorldCover covers the globe at 10 m.  The caller
    (``_gate_firms_by_land_cover``) tries CORINE first and falls back to
    this function when CORINE has no coverage.

    Same batched-GiST pattern as ``land_cover_codes_batch``: temp table,
    spatial pre-filter, chunked inserts, ST_Contains join.  Fail-open:
    if ``worldcover_polygons`` doesn't exist yet, returns empty (every
    point treated as burnable).
    """
    if not points:
        return {}
    # Check if table exists (fail-open)
    with _conn() as conn:
        row = conn.execute(
            "SELECT EXISTS ("
            "  SELECT 1 FROM information_schema.tables"
            "  WHERE table_name = 'worldcover_polygons'"
            ")"
        ).fetchone()
        exists = row["exists"]
    if not exists:
        return {}
    bbox = _table_bbox("worldcover_polygons")
    if bbox is None:
        return {}
    keep = [(i, (lat, lon)) for i, (lat, lon) in enumerate(points)
            if _in_bbox(lat, lon, bbox)]
    if not keep:
        return {}
    out: dict[int, Optional[int]] = {}
    with _conn() as conn:
        conn.execute(
            "CREATE TEMP TABLE _wc_pts (idx int, geom geometry) ON COMMIT DROP"
        )
        conn.execute("CREATE INDEX ON _wc_pts USING gist (geom)")
        for start in range(0, len(keep), 1000):
            batch = keep[start:start + 1000]
            conn.execute(
                "INSERT INTO _wc_pts (idx, geom) VALUES "
                + ", ".join(
                    "(%s, ST_SetSRID(ST_MakePoint(%s, %s), 4326))"
                    for _ in batch
                ),
                [p for idx, (lat, lon) in batch for p in (idx, lon, lat)],
            )
        rows = conn.execute(
            """
            SELECT DISTINCT ON (p.idx) p.idx, wp.class_code
            FROM _wc_pts p
            JOIN worldcover_polygons wp ON ST_Intersects(wp.geom, p.geom)
            ORDER BY p.idx, wp.class_code
            """
        ).fetchall()
    for r in rows:
        val = r["class_code"]
        if val is not None:
            out[int(r["idx"])] = int(val)
    return out


def land_mask_batch(points: list[tuple[float, float]]) -> dict[int, bool]:
    """Return ``{index: True}`` for points that fall ON LAND.

    Uses the ``land_mask`` table (Natural Earth 110m land polygon) to
    determine if each point is on land or water.  Points outside the
    land-mask coverage (e.g. far ocean) are treated as water (fail-closed
    for the water filter — we drop them to avoid false positives).

    Returns ``{index: True}`` for land points; water points are absent.
    The caller drops water points by checking membership.

    Fail-open if ``land_mask`` table doesn't exist: returns all points
    as land (no filtering), so the system degrades gracefully.
    """
    if not points:
        return {}
    # Check if table exists (fail-open)
    with _conn() as conn:
        row = conn.execute(
            "SELECT EXISTS ("
            "  SELECT 1 FROM information_schema.tables"
            "  WHERE table_name = 'land_mask'"
            ")"
        ).fetchone()
        exists = row["exists"]
    if not exists:
        # Fail-open: no land mask loaded — treat all points as land
        return {i: True for i in range(len(points))}
    bbox = _table_bbox("land_mask")
    if bbox is None:
        return {i: True for i in range(len(points))}
    # Points outside the bbox are water (fail-closed for water filter)
    keep = [(i, (lat, lon)) for i, (lat, lon) in enumerate(points)
            if _in_bbox(lat, lon, bbox)]
    if not keep:
        # All points outside land mask bbox — treat as water
        return {}
    out: dict[int, bool] = {}
    with _conn() as conn:
        conn.execute(
            "CREATE TEMP TABLE _lm_pts (idx int, geom geometry) ON COMMIT DROP"
        )
        conn.execute("CREATE INDEX ON _lm_pts USING gist (geom)")
        for start in range(0, len(keep), 1000):
            batch = keep[start:start + 1000]
            conn.execute(
                "INSERT INTO _lm_pts (idx, geom) VALUES "
                + ", ".join(
                    "(%s, ST_SetSRID(ST_MakePoint(%s, %s), 4326))"
                    for _ in batch
                ),
                [p for idx, (lat, lon) in batch for p in (idx, lon, lat)],
            )
        rows = conn.execute(
            """
            SELECT DISTINCT p.idx
            FROM _lm_pts p
            JOIN land_mask lm ON ST_Contains(lm.geom, p.geom)
            """
        ).fetchall()
    for r in rows:
        out[int(r["idx"])] = True
    return out


def volcano_hits_batch(points: list[tuple[float, float]]) -> dict[int, bool]:
    """Return ``{index: True}`` for points near a volcano (Step 4 stub).

    The real implementation joins points against a ``volcanoes`` table
    built from the Smithsonian GVP Holocene volcano database (~1,400
    entries) via a ``corine_import.py``-shaped import script, using the
    same batched GiST join shape as ``static_source_hits_batch`` (with the
    same 10k-row chunking and coverage-bbox pre-filter).

    The GVP import is DEFERRED until geographic expansion — volcanic heat
    is irrelevant for the Poland beta — so this stub always returns no
    hits. Fail-open by construction: callers treat every point as
    non-volcanic, exactly as they would before the table existed.
    """
    return {}


def kv_set(key: str, value: Any) -> None:
    with _conn() as conn:
        with conn.transaction():
            conn.execute(
                """
                INSERT INTO kv_store (key, value, created_at)
                VALUES (%s, %s, now())
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                """,
                (key, Jsonb(value)),
            )


def kv_cleanup(prefixes: list[str], max_age_hours: float = 48.0) -> int:
    """Delete kv_store entries matching prefixes older than max_age_hours.

    Returns the number of deleted rows.  Call periodically from the worker
    to prevent unbounded growth of weather/EFFIS/feedback cache rows.
    """
    if not prefixes:
        return 0
    with _conn() as conn:
        # Build prefix match: key LIKE 'weather_%' OR key LIKE 'effis_%'
        clauses = " OR ".join("key LIKE %s" for _ in prefixes)
        params = [f"{p}%" for p in prefixes] + [int(max_age_hours * 3600)]
        result = conn.execute(
            f"""
            DELETE FROM kv_store
            WHERE ({clauses})
              AND created_at < now() - make_interval(secs := %s)
            """,
            params,
        )
        return result.rowcount or 0


# ---------------------------------------------------------------------------
# OSM road cache
# ---------------------------------------------------------------------------

def osm_get(cache_key: str) -> Optional[dict]:
    with _conn() as conn:
        row = conn.execute(
            "SELECT segments, stored_at FROM osm_road_cache WHERE cache_key = %s",
            (cache_key,),
        ).fetchone()
    if not row:
        return None
    return {"segments": row["segments"], "stored_at": row["stored_at"]}


def osm_set(cache_key: str, segments: list, stored_at: float) -> None:
    with _conn() as conn:
        with conn.transaction():
            conn.execute(
                """
                INSERT INTO osm_road_cache (cache_key, segments, stored_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (cache_key) DO UPDATE SET
                    segments = EXCLUDED.segments,
                    stored_at = EXCLUDED.stored_at
                """,
                (cache_key, Jsonb(segments), stored_at),
            )


def osm_iter() -> Iterator[tuple[str, dict]]:
    """Yield (cache_key, entry) for every cached location (fuzzy matching
    scans these instead of a process-local dict)."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT cache_key, segments, stored_at FROM osm_road_cache"
        ).fetchall()
    for r in rows:
        yield r["cache_key"], {"segments": r["segments"], "stored_at": r["stored_at"]}


def osm_count() -> int:
    with _conn() as conn:
        return conn.execute(
            "SELECT count(*) AS n FROM osm_road_cache"
        ).fetchone()["n"]


# ---------------------------------------------------------------------------
# Geocode cache — search-box results keyed by normalized query
# ---------------------------------------------------------------------------

GEOCODE_TTL_S = 30 * 24 * 3600  # 30 days


def geocode_get(query_key: str, ttl_s: float = GEOCODE_TTL_S) -> Optional[list]:
    """Return cached geocode results for a normalized query key, or None if
    missing or expired (expired rows are treated as a miss and overwritten
    on the next successful fetch)."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT results, stored_at FROM geocode_cache WHERE query_key = %s",
            (query_key,),
        ).fetchone()
    if not row:
        return None
    age = (datetime.now(timezone.utc) - row["stored_at"]).total_seconds()
    if age > ttl_s:
        return None
    return row["results"]


def geocode_set(query_key: str, results: list) -> None:
    """Cache geocode results under a normalized query key (upsert)."""
    with _conn() as conn:
        with conn.transaction():
            conn.execute(
                """
                INSERT INTO geocode_cache (query_key, results, stored_at)
                VALUES (%s, %s, now())
                ON CONFLICT (query_key) DO UPDATE SET
                    results = EXCLUDED.results,
                    stored_at = now()
                """,
                (query_key, Jsonb(results)),
            )


# ---------------------------------------------------------------------------
# One-time import of the legacy JSON files
# ---------------------------------------------------------------------------

def import_json_data(
    reports_file: str = "data/reports.json",
    demo_reports_file: str = "data/demo_reports.json",
    osm_cache_file: str = "data/osm_road_cache.json",
) -> dict:
    """
    Import existing JSON stores into Postgres. Skips any mode that already
    has rows so re-running is safe. Returns a summary dict.
    """
    from pathlib import Path

    summary = {"reports": 0, "demo_reports": 0, "osm_cache": 0, "skipped": []}

    def _load(path: str):
        p = Path(path)
        if not p.exists():
            return []
        try:
            data = json.loads(p.read_text())
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    if count_reports("production") == 0:
        reports = _load(reports_file)
        if reports:
            insert_reports(reports, "production")
            summary["reports"] = len(reports)
    else:
        summary["skipped"].append("reports (production store not empty)")

    if count_reports("demo") == 0:
        demo = _load(demo_reports_file)
        if demo:
            insert_reports(demo, "demo")
            summary["demo_reports"] = len(demo)
    else:
        summary["skipped"].append("demo_reports (demo store not empty)")

    if osm_count() == 0:
        p = Path(osm_cache_file)
        if p.exists():
            try:
                data = json.loads(p.read_text())
                if isinstance(data, dict):
                    n = 0
                    for key, entry in data.items():
                        if isinstance(entry, dict) and "segments" in entry:
                            osm_set(key, entry["segments"], entry.get("stored_at", time.time()))
                            n += 1
                    summary["osm_cache"] = n
            except (json.JSONDecodeError, OSError):
                pass
    else:
        summary["skipped"].append("osm_cache (cache not empty)")

    return summary


# ---------------------------------------------------------------------------
# Poller flag helpers (used by server routes and jobs)
# ---------------------------------------------------------------------------

def get_poller_config(kind: str, defaults: dict) -> dict:
    """kind: 'satellite' | 'firms' | 'cap'. Returns merged {active, params...}."""
    stored = kv_get(f"{kind}_poller", {}) or {}
    merged = dict(defaults)
    merged.update({k: v for k, v in stored.items() if k != "active"})
    merged["active"] = bool(stored.get("active", False))
    return merged


def set_poller_active(kind: str, active: bool, params: Optional[dict] = None) -> None:
    stored = kv_get(f"{kind}_poller", {}) or {}
    if params:
        stored.update(params)
    stored["active"] = active
    kv_set(f"{kind}_poller", stored)


def ping() -> None:
    """Cheap liveness probe for health checks: round-trips one query through
    the connection pool (raises on DB outage)."""
    with _conn() as conn:
        conn.execute("SELECT 1")
