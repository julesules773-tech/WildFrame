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
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Iterator, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from bayesian_filter import BayesianFireGrid, auto_grid_size, DEFAULT_CELL_SIZE_M

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

_pool: Optional[ConnectionPool] = None


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            DATABASE_URL,
            min_size=1,
            max_size=10,
            open=False,
            kwargs={"row_factory": dict_row},
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
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- Existing databases: add the column idempotently (CREATE IF NOT EXISTS
-- above won't touch an existing table).
ALTER TABLE bayesian_grids ADD COLUMN IF NOT EXISTS last_evidence_at DOUBLE PRECISION NOT NULL DEFAULT 0;
CREATE INDEX IF NOT EXISTS idx_grids_mode_geom ON bayesian_grids USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_grids_mode_maxp ON bayesian_grids (mode, max_p DESC);
CREATE INDEX IF NOT EXISTS idx_grids_mode_evidence ON bayesian_grids (mode, last_evidence_at);

CREATE TABLE IF NOT EXISTS osm_road_cache (
    cache_key  TEXT PRIMARY KEY,
    segments   JSONB,
    stored_at  DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS kv_store (
    key    TEXT PRIMARY KEY,
    value  JSONB
);
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
    """Delete one report (mode-scoped). Returns the deleted report or None."""
    with _conn() as conn:
        with conn.transaction():
            row = conn.execute(
                "DELETE FROM reports WHERE id = %s AND mode = %s "
                "RETURNING id, status, data",
                (report_id, mode),
            ).fetchone()
    return _row_to_report(row) if row else None


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
) -> list[dict]:
    """List grid metadata (id, centroid, wind, max_p) for a mode, filtered
    to a viewport bbox (west,south,east,north) via PostGIS and ordered by
    peak probability. Does NOT load the heavy numpy state."""
    clauses = ["mode = %s"]
    params: list[Any] = [mode]
    if bbox is not None:
        w, s, e, n = bbox
        clauses.append("geom && ST_MakeEnvelope(%s, %s, %s, %s, 4326)")
        params.extend([w, s, e, n])
    sql = (
        "SELECT id, centroid_lat, centroid_lon, wind_speed, wind_dir_deg, "
        "max_p, updated_at "
        "FROM bayesian_grids WHERE " + " AND ".join(clauses) +
        " ORDER BY max_p DESC"
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
                    "max_p": nearest["max_p"],
                }

            # --- No nearby grid — create a fresh one sized to this cluster ---
            lats = [p[0] for p in cluster.get("points", [[clat, clon]])]
            lons = [p[1] for p in cluster.get("points", [[clat, clon]])]
            sizing = auto_grid_size(lats, lons) or {
                "center_lat": clat, "center_lon": clon,
                "cell_size_m": DEFAULT_CELL_SIZE_M, "nx": 40, "ny": 40,
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
                     last_evidence_at)
                VALUES (%s, %s, %s, %s,
                        ST_SetSRID(ST_MakePoint(%s, %s), 4326),
                        %s, %s, %s, %s, %s)
                """,
                (
                    grid_id, mode, clat, clon, clon, clat,
                    wind_speed, wind_dir_deg,
                    Jsonb(grid.to_dict()),
                    float(grid.get_statistics()["max_p"]),
                    float(grid.last_updated.max()),  # 0 — fresh grid, no evidence yet
                ),
            )
            return grid_id, {
                "grid": grid,
                "centroid_lat": clat,
                "centroid_lon": clon,
                "wind_speed": wind_speed,
                "wind_dir_deg": wind_dir_deg,
                "max_p": float(grid.get_statistics()["max_p"]),
            }


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
            if grid.last_predict_time > 0:
                elapsed = now - grid.last_predict_time
                if elapsed < min_age_s:
                    return False  # checkpointed recently — skip
                grid.predict(
                    dt=min(elapsed, 600.0),
                    wind_speed=wind_speed,
                    wind_dir_deg=wind_dir_deg,
                )
            else:
                grid.predict(dt=60.0, wind_speed=wind_speed, wind_dir_deg=wind_dir_deg)
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


def kv_set(key: str, value: Any) -> None:
    with _conn() as conn:
        with conn.transaction():
            conn.execute(
                """
                INSERT INTO kv_store (key, value) VALUES (%s, %s)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                """,
                (key, Jsonb(value)),
            )


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
    """kind: 'satellite' | 'firms'. Returns merged {active, params...}."""
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
