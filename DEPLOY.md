# Deployment — Fly.io (web + worker) + Neon (PostGIS database)

WildFrame is an **always-on Flask + Postgres/PostGIS + background-worker** app,
so it cannot run on serverless hosts (Vercel, etc.). This guide deploys it on
**Fly.io** as two process groups in one Docker image:
- **web** — gunicorn (`server:app`), public HTTP
- **worker** — Procrastinate job worker (`python worker.py`), always-on

The database is **Neon** (managed Postgres with the **PostGIS** extension).

Estimated time: ~30 minutes. Cost: Fly free allowance then ~$3–6/mo per
machine (web + worker), plus Neon's free tier.

---

## 0. Prerequisites

- Docker installed (Fly builds your image locally with `fly deploy`).
- `flyctl` installed: `brew install flyctl`, then `fly auth login`.
- Your existing `.env` values handy (S3 keys, NASA FIRMS, Roboflow, admin
  secret, agency key) — you'll set them as Fly secrets.

## 1. Create the database on Neon

1. Sign up at <https://neon.tech> → **Create project** (any region; `wildframe` DB).
2. Open the **SQL Editor** and run:
   ```sql
   CREATE EXTENSION IF NOT EXISTS postgis;
   ```
3. Copy the **connection string** (*Dashboard → Connection Details*, the
   non-pooled driver string) and append `?sslmode=require`:
   ```
   postgresql://USER:PASSWORD@ep-XXXX.us-east-1.aws.neon.tech/wildframe?sslmode=require
   ```
4. **Important:** *Project settings → Compute* → set **Auto-suspend to 0
   (never)**. Otherwise Neon scales to zero after 5 min idle and the worker's
   per-minute jobs hit cold starts.

## 2. Create the schema (run once, from your machine)

```bash
cd /Users/julianuhres/WildFrame
WILDFRAME_DATABASE_URL='postgresql://USER:PASSWORD@ep-XXXX.us-east-1.aws.neon.tech/wildframe?sslmode=require' \
  .venv/bin/python migrate.py
```

This creates the WildFrame tables (reports, clusters, bayesian_grids,
osm_road_cache, …) **and** the Procrastinate job-queue tables. Idempotent —
safe to re-run.

## 3. Create the Fly app

```bash
cd /Users/julianuhres/WildFrame
fly apps create wildframe     # pick a unique name and update `app` in fly.toml if taken
```

## 4. Set secrets

```bash
fly secrets set \
  WILDFRAME_DATABASE_URL='postgresql://USER:PASSWORD@ep-XXXX.us-east-1.aws.neon.tech/wildframe?sslmode=require' \
  NASA_FIRMS_API_KEY='your-key' \
  ROBOFLOW_API_KEY='your-key' \
  AWS_ACCESS_KEY_ID='your-key' \
  AWS_SECRET_ACCESS_KEY='your-secret' \
  WILDFRAME_ADMIN_SECRET='your-secret' \
  WILDFRAME_AGENCY_API_KEY='your-secret'
```

(`WILDFRAME_S3_BUCKET` is already set in `fly.toml` — it's not a credential.)

## 5. Deploy

```bash
fly deploy
```

Fly builds the image (a few minutes the first time), then starts **two
machines** — one `web`, one `worker`. `fly deploy` again after every push.

## 6. Verify

- `fly open` → the map should load; `https://<app>.fly.dev/healthz` → `200`.
- `fly logs` → the **worker** log should show `firms.fetch`, `grids.advance`,
  and weather sweep activity within a minute (that's the proof the worker is
  alive and the DB is reachable).
- Upload a test photo → it should be scanned (Roboflow) and stored in S3
  (no persistent disk needed — `WILDFRAME_S3_BUCKET` routes uploads to S3).
- Test the agency ingest endpoint:
  ```bash
  curl -s -X POST https://<app>.fly.dev/api/agencies/ingest \
    -H "Content-Type: application/json" -H "X-Agency-Key: <your key>" \
    -d '{"agency":"gov-test","incident_id":"deploy-check","action":"create","lat":51.1,"lon":18.9,"sent_at":"2026-08-10T12:00:00Z"}'
  ```

## 7. Operational notes

- **Both processes must run.** The worker drives FIRMS fetches, grid advance,
  and wind refresh every minute. Fly machines don't sleep, so a starter
  `shared-cpu-1x` machine per process is enough.
- **The worker is the CPU-heavy one** (grid advance over ~780 grids). If it
  gets slow: `fly scale memory 1024` (or split sizing per process later).
- **Scaling:** `fly scale count 2 -p web` to add web machines; `fly scale show`
  to inspect.
- **Secrets:** never in the repo — `fly secrets set` encrypts them per-app.
- **Photos:** S3 via boto3 (lazy-imported — no heavy deps in the image).
- **Open-Meteo:** the free tier is non-commercial; for a public beta switch to
  the paid Standard plan ($29/mo) — it also lifts the ~9,500 calls/day cap.
- **Logs:** `fly logs` (worker + web); the gunicorn access log goes to stdout
  (`--access-logfile -`).

## Troubleshooting

| Symptom | Fix |
|---|---|
| Web starts, 500s on map data | DB not migrated — re-run step 2, or the Neon `sslmode=require` is missing from the secret |
| No fires / no grid advance | Worker not running — `fly logs`; check `NASA_FIRMS_API_KEY` secret |
| Wind stuck at 3.0/270° | Open-Meteo budget exhausted or Neon auto-suspend — check `[weather]` log lines |
| `psycopg` SSL error | Connection string must end in `?sslmode=require` |
| Port clash / no HTTP | `internal_port` in fly.toml must match gunicorn's `--bind` port (8080) |

## Alternatives

- **Render** — a complete blueprint (`render.yaml`, web + worker, health
  check) is preserved in git history: `git show 92894b5:render.yaml > render.yaml`.
  Same Neon DB; two always-on `starter` instances (~$14–26/mo).
- **Hetzner VPS** — cheapest at scale (~€4–8/mo for everything), most manual:
  apt Postgres + `postgis`, systemd units for gunicorn + worker, nginx + TLS.
