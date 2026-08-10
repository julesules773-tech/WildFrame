# Deployment — Render (web + worker) + Neon (PostGIS database)

WildFrame is an **always-on Flask + Postgres/PostGIS + background-worker** app,
so it cannot run on serverless hosts (Vercel, etc.). This guide deploys it on
**Render** (two persistent services: `web` + `worker`) with a **Neon** Postgres
database that has the **PostGIS** extension (Render's own managed Postgres does
*not* support PostGIS, which is why we use Neon).

Estimated time: ~25 minutes. Cost: ~$14–26/mo (two Render starter instances +
Neon free tier).

---

## 0. Prerequisites

- This repo pushed to GitHub (Render deploys from git).
- Your existing `.env` values handy (S3 keys, NASA FIRMS key, Roboflow key,
  admin secret, agency key) — you'll paste them into Render once.

## 1. Create the database on Neon

1. Sign up at <https://neon.tech> → **Create project** (any region; `wildframe` DB).
2. Open the **SQL Editor** and run:
   ```sql
   CREATE EXTENSION IF NOT EXISTS postgis;
   ```
3. Copy the **connection string** from *Dashboard → Connection Details* (use
   the `psql`/driver string, **not** the pooled one) and append
   `?sslmode=require`:
   ```
   postgresql://USER:PASSWORD@ep-XXXX.us-east-1.aws.neon.tech/wildframe?sslmode=require
   ```
4. **Important:** open *Project settings → Compute* and set **Auto-suspend to 0
   (never)**. Otherwise Neon scales the DB to zero after 5 min idle and the
   worker's per-minute jobs will hit cold starts.

## 2. Create the schema (run once, from your machine)

```bash
cd /Users/julianuhres/WildFrame
WILDFRAME_DATABASE_URL='postgresql://USER:PASSWORD@ep-XXXX.us-east-1.aws.neon.tech/wildframe?sslmode=require' \
  .venv/bin/python migrate.py
```

This creates the WildFrame tables (reports, clusters, bayesian_grids,
osm_road_cache, …) **and** the Procrastinate job-queue tables. It's idempotent —
safe to re-run.

> ℹ️ `migrate.py` prints the PostGIS hint — you already ran
> `CREATE EXTENSION postgis` in step 1, so ignore it.

## 3. Deploy to Render

1. Sign up at <https://render.com> → **New → Blueprint** → connect the GitHub repo.
2. Render detects `render.yaml` and creates two services:
   - **wildframe** (web) — `gunicorn server:app`
   - **wildframe-worker** (worker) — `python worker.py`
3. Render will prompt for the `sync: false` env vars. Paste the **same values in
   both services** (Render's "Environment Groups" are the easiest way):

   | Key | Value |
   |---|---|
   | `WILDFRAME_DATABASE_URL` | Neon string from step 1 |
   | `NASA_FIRMS_API_KEY` | from your `.env` |
   | `ROBOFLOW_API_KEY` | from your `.env` |
   | `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | from your `.env` |
   | `WILDFRAME_S3_BUCKET` | `photo-hold-542489917871-us-east-1-an` |
   | `WILDFRAME_ADMIN_SECRET` | from your `.env` |
   | `WILDFRAME_AGENCY_API_KEY` | from your `.env` |

4. Hit **Apply**. Both services build and start.

## 4. Verify

- Open the web service URL → `GET /healthz` should return `200`.
- The map should load, FIRMS fires should appear, and within a minute the worker
  logs should show `firms.fetch` / `grids.advance` / weather sweep activity.
- Upload a test photo → it should be scanned (Roboflow) and stored in S3
  (uploads go to S3 automatically because `WILDFRAME_S3_BUCKET` is set — no
  persistent disk needed on Render).
- Test the agency ingest endpoint with your key:
  ```bash
  curl -s -X POST https://<your-app>.onrender.com/api/agencies/ingest \
    -H "Content-Type: application/json" -H "X-Agency-Key: <your key>" \
    -d '{"agency":"gov-test","incident_id":"deploy-check","action":"create","lat":51.1,"lon":18.9,"sent_at":"2026-08-10T12:00:00Z"}'
  ```

## 5. Operational notes

- **Both services must stay up.** The worker is what fetches FIRMS hotspots,
  advances the fire grid, and refreshes wind every minute. Render's *free*
  instances sleep after inactivity — that's why this blueprint uses `starter`.
- **Secrets:** `.env` is gitignored and never deployed; all secrets live in
  Render's dashboard.
- **Photos:** stored in S3 via boto3 (lazy-imported — no heavy deps).
- **Open-Meteo:** the free tier is non-commercial; for a public beta, switch to
  the paid Standard plan ($29/mo) — it also lifts the ~9,500 calls/day cap.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Web starts, then 500s on map data | DB not migrated — re-run step 2, or the Neon `sslmode=require` is missing from the URL |
| No fires appearing | Worker not running or `NASA_FIRMS_API_KEY` empty — check worker logs |
| Grid shows default wind 3.0/270° | Open-Meteo budget exhausted or DB auto-suspend — check `[weather]` log lines |
| `psycopg` SSL error | Ensure the connection string ends in `?sslmode=require` |

## Switching hosts (alternatives)

- **Fly.io** — same Neon DB; write a `Dockerfile` + `fly.toml` with two
  `processes:` (web: `gunicorn server:app`; worker: `python worker.py`).
- **Hetzner VPS** — install Postgres + `postgis` via apt, systemd units for
  gunicorn + worker, nginx + TLS. Cheapest at scale, most manual.
