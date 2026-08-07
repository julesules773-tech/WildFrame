# WildFrame

Wildfire detection prototype: citizen photo reports + satellite/FIRMS hotspot
ingestion fused with a Bayesian fire-propagation model, visualised on an
interactive map.

**Highlights**

- Citizen reports, NASA FIRMS hotspots, and optional AI photo scanning
  (`fire_vision.py`) fused into one probabilistic fire model
  (`bayesian_filter.py`)
- Demo / seed / simulated data strictly isolated from operational outputs
  (see *Demo vs production isolation*)
- All state in PostgreSQL + PostGIS with a Procrastinate job queue — no JSON
  files, no in-memory registries, no lost updates under concurrency, and
  state survives restarts
- FIRMS ingestion locked to the past 24 hours; fires older than that expire
  automatically

## Architecture

```
static/   (browser UI: index.html, admin.html, app.js)
   │
server.py (Flask API — uploads, reports, map queries, admin)
   │
worker.py (Procrastinate job queue — periodic jobs + job execution)
   │
jobs.py ───┐
db.py  ────┼──► PostgreSQL 18 + PostGIS   (reports, grids, caches, queue)
           │
migrate.py (one-time schema creation + legacy JSON import)
```

**PostgreSQL/PostGIS replaces the old JSON files + in-memory state.** Every
write is now an atomic SQL statement or a row-locked transaction, so
concurrent requests and multiple workers can no longer lose updates, and
all state survives restarts.

| Store          | Table            | Notes                                             |
| -------------- | ---------------- | ------------------------------------------------- |
| Reports        | `reports`        | full dict in `data` JSONB; `mode` column isolates demo/production |
| Bayesian grids | `bayesian_grids` | numpy grid state in `state` JSONB; `FOR UPDATE` row locks serialize per-grid mutations |
| OSM road cache | `osm_road_cache` | road segments keyed by rounded lat/lon/radius     |
| Flags/counters | `kv_store`       | poller flags, heartbeats, grid-id counter         |
| Job queue      | `procrastinate_*`| installed by `migrate.py`                          |

## Project layout

| File                 | Role |
| -------------------- | ---- |
| `server.py`          | Flask API: uploads, reports, clusters, Bayesian state, poller & admin endpoints |
| `worker.py`          | job-queue worker: runs periodic satellite/FIRMS jobs + `grids.advance` / `grids.expire_stale` |
| `jobs.py`            | Procrastinate task definitions + periodic schedules |
| `db.py`              | all SQL/PostGIS access (reports, grids, kv_store, schema helpers) |
| `migrate.py`         | one-time setup: create schema, import legacy JSON |
| `nasa_firms.py`      | NASA FIRMS API client (24h hotspot fetches) |
| `fire_vision.py`     | optional AI fire/smoke detection on uploaded photos (Roboflow) |
| `bayesian_filter.py` | Bayesian fire-propagation grid model |
| `triangulation.py`   | multi-report triangulation helpers |
| `static/`            | frontend: `index.html` (map), `admin.html` (dashboard), `app.js`, styles |

## Configuration

Copy the template and fill in what you have:

```bash
cp .env.example .env
```

The entry-point scripts (`server.py`, `worker.py`, `migrate.py`) load `.env`
automatically via `python-dotenv`. **Values already exported in your shell
take precedence** — `load_dotenv()` never overrides an existing env var.

| Variable | Default | Purpose |
| --- | --- | --- |
| `WILDFRAME_DATABASE_URL` | `postgresql:///wildframe` | Postgres/PostGIS connection string |
| `NASA_FIRMS_API_KEY` | — | NASA FIRMS hotspot ingestion — free key at <https://firms.modaps.eosdis.nasa.gov/api/map_key/>. `FIRMS_API_KEY` is accepted as an alias. |
| `ROBOFLOW_API_KEY` | — | optional AI fire/smoke photo scanning — free account at <https://app.roboflow.com/> |
| `WILDFRAME_ADMIN_SECRET` | `wildframe-admin` | shared secret for the admin dashboard (`/admin.html`, sent as the `X-Admin-Secret` header). **Change it in production!** |

`.env` is gitignored; only `.env.example` is committed.

## Demo vs production isolation

Demo/seed/simulation/historic-replay data never touches operational
outputs: every report, grid, and evidence item carries a `mode`
(`production` | `demo`) and is read/written through mode-filtered SQL.
Demo grids are also id-prefixed (`demo-grid-N`) so ids can never collide
with production grids. Poller flags and heartbeats live in `kv_store`,
shared but namespaced per poller.

## Periodic jobs

The simulated satellite pass and real FIRMS fetch are Procrastinate
periodic jobs running in `worker.py`, toggled by the API (start/stop just
flips a flag in `kv_store`). One worker process handles both execution and
periodic deferral. Cadence is `interval_s` (defaults: satellite 20 s,
FIRMS 600 s) enforced by minute-cron + self-throttling (see `jobs.py`).

Additional periodic jobs:

- `grids.advance` — advances and persists Bayesian grid state on a schedule,
  keeping the map's read path read-only
- `grids.expire_stale` — hourly purge of fires whose newest evidence is
  older than 24 h

## Setup (macOS / Homebrew)

```bash
brew install postgresql postgis
brew services start postgresql
createdb wildframe
psql -d wildframe -c 'CREATE EXTENSION IF NOT EXISTS postgis;'

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

cp .env.example .env      # then fill in NASA_FIRMS_API_KEY etc.
```

Connection string: `WILDFRAME_DATABASE_URL` env var (default
`postgresql:///wildframe` — Homebrew current-user socket auth).

## First run / migration

```bash
.venv/bin/python migrate.py        # creates tables + imports legacy JSON data
```

Idempotent: re-running skips any store that already has rows.

## Run

```bash
.venv/bin/python server.py         # web app on http://localhost:4141
.venv/bin/python worker.py         # job-queue worker (periodic satellite/FIRMS polls)
```

## Poller API

| Endpoint | Effect |
| --- | --- |
| `POST /api/satellite/poller/start` | enable simulated satellite pass (`interval_s`, `probability`, `min_hotspots`, `max_hotspots`) |
| `POST /api/satellite/poller/stop` | disable it |
| `POST /api/satellite/firms-poller/start` | enable real FIRMS fetch (`interval_s`, `day_range`, `min_confidence`) |
| `POST /api/satellite/firms-poller/stop` | disable it |
| `POST /api/satellite/firms-fetch` | manual FIRMS fetch — **asynchronous**: returns `{accepted: true}` immediately and runs the global pass (fetch + clustering + grid injection) as a job in the queue worker. Poll `GET /api/satellite/poller/status` for `firms_fetch_in_progress` / `firms_fetch_last_result` (409 if one is already running). **Always queries the past 24 hours** — `day_range` is locked to 1 server-side (button, poller, and job all clamp to it) |
| `GET /api/satellite/poller/status` | flags + worker heartbeats + grid counts + manual-fetch progress |

The FIRMS pass itself is fast even on the ~100k hotspots a global day-range
fetch returns: clustering is O(n) spatial-hash (not O(n²)), each fire's
evidence is injected in one row-locked transaction, and the whole thing
runs in the worker so the UI never blocks.

### 24-hour window + fire expiry

Every FIRMS query (manual button, FIRMS Live poller, queued job) is locked
to the past 24 hours: `day_range` is hard-capped to 1 in the routes, the
job, and inside `_fetch_nasa_firms_pass` itself, so no caller can widen it.

Fires also age out: each grid tracks the timestamp of its *newest* evidence
(`last_evidence_at`, fed only by evidence injection — map polling never
touches it). Grids whose newest evidence is older than 24 hours are
deleted — at the end of every FIRMS fetch and by an hourly periodic job
(`grids.expire_stale`) — so old fires disappear from both the map and the
DB. `migrate.py` backfills `last_evidence_at` for pre-existing grids.
