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
| `bayesian_filter.py` | Bayesian fire-propagation grid model + road-risk assessment |
| `weather.py`         | wind + hourly forecast for grids (WeatherAPI.com primary, Open-Meteo fallback) |
| `effis_fwi.py`       | EFFIS fuel-moisture indices (FFMC/DMC/ISI) scaling the spread model |
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
| `WILDFRAME_EVIDENCE_SHELF_LIFE_S` | `43200` (12 h) | evidence-gated decay: seconds a freshly-detected fire's probability holds before fading (matches the VIIRS revisit cadence). Only cells whose evidence is OLDER than this fade at the 3 h base half-life — verified by `backtest_grids.py`. |
| `ROBOFLOW_API_KEY` | — | optional AI fire/smoke photo scanning — free account at <https://app.roboflow.com/> |
| `WILDFRAME_ENV` | `development` | Set to `production` or `prod` to enable production-only safety checks. In production, the server **refuses to start** with the default admin secret. Accepts `FLASK_ENV` as a fallback. |
| `WILDFRAME_ADMIN_SECRET` | `wildframe-admin` | shared secret for the admin dashboard. `/admin` returns 404 publicly; visit `/admin?key=<secret>` once to set the login cookie (HttpOnly, 12h), then every admin API still requires the `X-Admin-Secret` header. **The server refuses to start if this is still the default in production** (`WILDFRAME_ENV=production`). |
| `WILDFRAME_S3_BUCKET` | — | S3 bucket for uploaded photos. **When set, accepted photos are stored in S3** (required on PaaS with ephemeral disk); when unset, photos stay on local disk (`uploads/`). |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | — | AWS credentials for S3 (boto3 default credential chain — an IAM role also works) |
| `AWS_REGION` | `us-east-1` | Region of the S3 bucket |
| `WILDFRAME_S3_PUBLIC_URL` | — | optional public base URL for photos (e.g. a CloudFront domain) — defaults to the bucket's S3 URL |
| `WILDFRAME_S3_PREFIX` | `photos` | object-key prefix inside the bucket |
| `WILDFRAME_AUTO_APPROVE` | `1` | corroboration-gated auto-approval (see below). Set to `0` to keep every report in the human-review queue |
| `WILDFRAME_AUTO_APPROVE_FLAME_CONF` | `0.80` | minimum **fire-class** confidence for auto-approval. Flame is the noisy class (false positives up to 0.94 on clean images), so its bar is high |
| `WILDFRAME_AUTO_APPROVE_SMOKE_CONF` | `0.40` | minimum **smoke-class** confidence for auto-approval. Smoke detections stay precise at low confidence (91.8% precision at 0.40), so its bar is low. Both floors calibrated with `threshold_sweep.py` against the fire dataset |
| `WILDFRAME_WEATHERAPI_KEY` | — | **primary wind provider** — WeatherAPI.com key (<https://www.weatherapi.com/>). `WEATHERAPI_API_KEY` is accepted as an alias. When set, wind + the hourly forecast series come from the paid, commercial-licensed API; when unset, Open-Meteo's free tier is used directly |
| `WILDFRAME_WEATHER` | `1` | set to `0` to disable live weather entirely (grids then use the neutral defaults: 3.0 m/s @ 270°) |
| `WILDFRAME_WEATHER_DAILY_BUDGET` | `9500` | per-day request cap for the **Open-Meteo fallback** (under its 10k/day free tier; the counter lives in `kv_store`) |
| `WILDFRAME_WEATHERAPI_DAILY_BUDGET` | `60000` | per-day request cap for **WeatherAPI.com** (~66.7k/day on a 2M/mo plan — cap is a runaway-loop guard, not a plan limit) |

## Corroboration-gated auto-approval

A positive AI photo verdict is treated as a *suggestion*, not truth: the
model's confidence is miscalibrated (see `backtest_vision.py` — false
positives were measured at up to 0.93, higher than the 0.84 scored by a
real fire). So a report is **never auto-confirmed on photo confidence
alone**. When `WILDFRAME_AUTO_APPROVE` is enabled (default), a report with
a positive verdict (`flame` / `smoke` / `both`) is auto-confirmed only
when **both** of these hold:

- **Its class clears the confidence floor** — `fire_confidence >= 0.80`
  **or** `smoke_confidence >= 0.40` (flame is the noisy class, smoke is
  precise at low confidence — calibrated with `threshold_sweep.py`, see
  the config table for overrides); **and**
- **An independent source corroborates it:**
  - **Cluster corroboration** — an already-confirmed report exists within
    `CLUSTER_RADIUS_M` (500 m) and the cluster time window (2 h). This
    path accepts a fire when nearby reports already agree even if FIRMS
    hasn't had a satellite pass yet; or
  - **Satellite corroboration** — a live FIRMS hotspot is found within
    3 km / 12 h of the report (the same matcher as the admin dashboard's
    manual check; fail-closed on API errors).

Auto-approved reports are flagged (`auto_approved`, `approval_source`,
`approval_class`, `approval_confidence`) and feed the Bayesian grid
immediately, but stay visible in the **"Recently auto-approved" strip on
the admin dashboard** (showing the class + confidence that cleared the
floor), so a human can still reject anywhere the corroboration was wrong.

`.env` is gitignored; only `.env.example` is committed.

## Wind, weather budgets, and forecast-driven road risk

Every Bayesian grid carries real 10 m wind (`wind_speed`, `wind_dir_deg`)
that shapes the spread ellipse, the smoke-drift upwind shift, and the
road-risk model — see `weather.py`.

**Providers (primary → fallback → neutral):**

1. **WeatherAPI.com** — used whenever `WILDFRAME_WEATHERAPI_KEY` is set.
   Paid plans carry a commercial license and a large per-day quota. One
   `forecast.json` call per cell returns *both* the current wind and the
   hourly forecast series (3 days: wind, precip, humidity, temp), so the
   forecast costs no extra requests.
2. **Open-Meteo** (free, budget-gated) — the default when no key is set,
   and the automatic fallback if WeatherAPI errors or its budget is out.
   Free tier is non-commercial only, so a paid plan is the compliant
   choice for commercial deployments.
3. **Neutral defaults** (3.0 m/s @ 270°) — only if both providers fail or
   are budget-exhausted. Weather never blocks report ingestion or grid
   creation.

**Direction convention:** both providers report the direction the wind
comes FROM; WildFrame stores the direction the fire spreads TOWARD (the
head of the spread ellipse, 0° = north). `weather.py` flips by 180°.

**Caching & budgets:** wind and the forecast series are cached per ~55 km
cell (0.5°) with a 24 h TTL in `kv_store` (`weather:{cell}`,
`weather_fc:{cell}`) plus an in-process cache. Two independent per-day
budget counters guard the providers (`weather_budget`, `weatherapi_budget`
in `kv_store`) so neither the free tier nor the paid quota can be burned
by a runaway loop; exhausted budgets fall through to the other provider.
Failed fetches refund their budget slot.

### Road risk (`POST /api/bayesian/road-risk`)

For each fire, road segments near the contour (Overpass/OSM, cached in
`osm_road_cache`) are assessed against the head/back/flank spread ellipse:
`t_arrival = distance / effective_rate(bearing)` bucketed into risk tiers
(**critical** < 30 min, **high** < 2 h, **moderate** < 6 h, else **low**),
weighted by the grid probability at the contour point.

When the cell has an hourly forecast series, arrival becomes a
**wind-at-arrival fixed point** (`_converged_arrival` in
`bayesian_filter.py`): a road reached in 90 min is assessed with the wind
that will actually be blowing in 90 min, iterating until the risk **tier**
holds steady (max 3 iterations). Two guardrails, both reviewed in:

- **Critical tier (< 30 min) is intentionally unchanged** — arrival that
  fast reads the current/first forecast hour anyway, so iteration is
  skipped and behavior is byte-identical to the pre-forecast code.
- **Oscillation fallback** — if the tier cycles between forecast-hour
  buckets (e.g. high → moderate → high) instead of stabilizing, the road
  falls back to the current-wind estimate rather than returning a
  mid-cycle wind.

A future precipitation dampener slots in via the `rate_modifier` hook
(`(rate, t_arrival) -> rate`, evaluated inside the convergence loop) —
wind and precip would then converge together instead of precip being a
post-hoc scalar. The road-risk response includes a `forecast_wind_grids`
metadata counter so forecast coverage is observable.

## Photo storage (S3)

Uploaded photos are the one piece of state that doesn't live in Postgres.
They're staged on local disk first (EXIF GPS + AI scan both read from
disk), then stored in S3 when `WILDFRAME_S3_BUCKET` is set — otherwise
they stay in local `uploads/` (local dev). `photo_url` on a report is just
a URL, so the frontend and DB schema need no changes.

**One-time bucket setup** (AWS console / CLI):

```bash
aws s3 mb s3://wildframe-photos --region us-east-1
```

Photos are uploaded public-read. If your bucket uses *Bucket owner
enforced* object ownership (the modern default), object ACLs are rejected
and you must grant public read via a bucket policy instead:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": "*",
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::wildframe-photos/photos/*"
  }]
}
```

For production, prefer a private bucket + CloudFront (set
`WILDFRAME_S3_PUBLIC_URL` to the distribution domain) over public reads.
Rejected reports and admin-rejected photos are deleted from S3
(`photo_storage.delete_photo`).

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
