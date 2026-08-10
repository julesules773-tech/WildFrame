# Deployment — AWS Lightsail (web + worker) + Neon (PostGIS database)

WildFrame is an **always-on Flask + Postgres/PostGIS + background-worker** app,
so it cannot run on serverless hosts (Vercel, etc.). This guide deploys it on
**AWS Lightsail** as a single 1 GB Ubuntu VM running two systemd services:

- **wildframe-web** — gunicorn (`server:app`) on 127.0.0.1:8000, fronted by
  nginx (TLS)
- **wildframe-worker** — Procrastinate job worker (`python worker.py`) with
  `Restart=always` (the Fly outage root cause was a worker that stopped and
  never came back — systemd makes that impossible)

Everything else is external: **Neon** (managed Postgres + PostGIS),
**AWS S3** (photo storage), **Cloudflare** (DNS + edge TLS, Full strict).

Estimated time: ~45–60 min first time. Cost: **$7/mo** Lightsail 1 GB
(`micro_3_0`) + Neon free tier + S3 (pennies) + Cloudflare Free.

> Historical note: Fly.io (`fly.toml`, `Dockerfile`) and Render (`render.yaml`
> in git history) configs are preserved but no longer used. The current
> deployment is the Lightsail VM described here.

---

## 0. Prerequisites

- AWS account; an IAM user with Lightsail permissions
  (`AmazonLightsailFullAccess` or `AdministratorAccess`), creds in `.env`
- boto3 (already in `requirements.txt`)
- Neon project with PostGIS (see §1)
- Cloudflare zone for `pyrae.co` on Cloudflare nameservers
  (see `CUSTOM_DOMAIN.md`)

## 1. Create the database on Neon

1. Sign up at <https://neon.tech> → **Create project** (any region).
2. In the SQL Editor: `CREATE EXTENSION IF NOT EXISTS postgis;`
3. Copy the connection string and append `?sslmode=require`.
4. *Project settings → Compute* → set **Auto-suspend to 0 (never)** — the
   worker's per-minute jobs must not hit cold starts.

## 2. Create the schema (once, from your machine)

```bash
WILDFRAME_DATABASE_URL='postgresql://USER:PASSWORD@ep-XXXX.us-east-1.aws.neon.tech/wildframe?sslmode=require' \
  .venv/bin/python migrate.py
```
Idempotent — safe to re-run.

## 3. Create the Lightsail instance

```bash
.venv/bin/python deploy/aws/create_instance.py
```
This creates the 1 GB Ubuntu 24.04 instance, an SSH key
(`~/.ssh/wildframe-prod-key.pem`), a static IP, and opens ports 22/80/443.
Print the resulting static IP (e.g. `3.222.128.25`).

> ⚠️ Use the `micro_3_0` bundle — the cheaper `micro_ipv6_3_0` **cannot
> attach a static IP** (the IP must be stable for DNS).

## 4. Push code + secrets to the VM

```bash
IP=<static-ip>; KEY=~/.ssh/wildframe-prod-key.pem
rsync -az -e "ssh -i $KEY" \
  --exclude .venv --exclude node_modules --exclude .git \
  --exclude __pycache__ --exclude '*.pyc' --exclude uploads \
  --exclude sample_test_images --exclude .aws \
  ./ ubuntu@$IP:/home/ubuntu/wildframe/
```

**`.env` handling (critical):** copy `.env` to the VM, then make sure
`WILDFRAME_DATABASE_URL` is the **Neon** URL — the local dev `.env` points at
a local Postgres, and bootstrapping with it silently breaks the app
(worker `PoolTimeout`, `/healthz` 500). `bootstrap.sh` now refuses to run if
the URL targets localhost. Keep `.env` out of git; it lives on the VM.

## 5. Bootstrap the VM

```bash
ssh -i $KEY ubuntu@$IP 'bash /home/ubuntu/wildframe/deploy/aws/bootstrap.sh'
```
Installs nginx/certbot, adds a 2 GB swap, builds the venv + requirements,
starts **wildframe-web** and **wildframe-worker** (systemd, enabled on boot),
and puts nginx in stage-1 (port 80 only, ACME challenge ready).

## 6. Point Cloudflare at the VM

In **Cloudflare → pyrae.co → DNS → Records**, for BOTH `@` and `www`:
delete the old record, add **A → <static-ip>** with **Proxied (orange)**.
Keep SSL/TLS mode **Full (strict)**. The site 525s until step 7 issues the
origin cert — that's expected.

## 7. TLS (Let's Encrypt, through Cloudflare)

```bash
ssh -i $KEY ubuntu@$IP 'bash /home/ubuntu/wildframe/deploy/aws/tls.sh'
```
Runs certbot (HTTP-01 via the ACME webroot, one cert for `pyrae.co` +
`www.pyrae.co`), installs the full nginx config (80→443 redirect + 443 proxy
to gunicorn, 20 MB upload cap), and enables certbot's auto-renewal timer.

## 8. Verify

```bash
curl -s -o /dev/null -w '%{http_code}\n' https://pyrae.co/          # 200
curl -s -o /dev/null -w '%{http_code}\n' https://www.pyrae.co/      # 200
curl -s https://pyrae.co/healthz          # {"db":"ok","status":"ok"}
```
- Upload a photo → scanned (Roboflow) + stored in **S3**
  (`WILDFRAME_S3_BUCKET` is set in the VM `.env`).
- Within a minute, `journalctl -u wildframe-worker` should show
  `firms.fetch`, `grids.advance`, and weather jobs.

## Operational notes

- **Both services must run.** `systemctl status wildframe-web
  wildframe-worker`; both are `Restart=always` and enabled on boot.
- **Memory:** 1 GB fits gunicorn (2 workers) + the worker; a 2 GB swap file
  is the safety net. `free -m` to watch.
- **Cert renewal:** certbot's systemd timer renews automatically
  (`systemctl list-timers certbot.timer`).
- **Deploys:** rsync the repo again (step 4), then
  `ssh ... 'sudo systemctl restart wildframe-web wildframe-worker'`.
- **Photos:** S3 via boto3 — no persistent disk needed.
- **Open-Meteo:** free tier is non-commercial; for a public beta consider the
  paid Standard plan (~$29/mo) if the ~9,500 calls/day cap is hit.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `/healthz` 500 / worker `PoolTimeout` | VM `.env` has a local DB URL — set the Neon URL (bootstrap refuses to proceed) |
| 525 after DNS flip | Origin cert not issued yet — run `tls.sh` once DNS propagates |
| certbot: `NXDOMAIN looking up A for www.pyrae.co` | The `www` A record is missing/not propagated — add it, query `dig @1.1.1.1`, rerun `tls.sh` |
| nginx: `unknown directive "http2"` | Ubuntu 24.04 ships nginx 1.24 — `tls.sh` already uses `listen 443 ssl http2;` |
| Site 525 later, out of nowhere | Origin down or cert expired — `ssh` in, `systemctl status`, `certbot certificates` |
| Wind stuck at 3.0/270° | Open-Meteo budget exhausted — check `[weather]` lines in the worker journal |
