# Custom Domain — pyrae.co (AWS Lightsail)

Documentation of how `pyrae.co` and `www.pyrae.co` are wired to the WildFrame
app on AWS Lightsail. Updated 2026-08-10 (migrated from Fly.io after the Fly
trial ended and shut the app down).

## Current architecture

```
Browser ─► https://pyrae.co ────► Cloudflare edge (proxied) ──► Lightsail VM
Browser ─► https://www.pyrae.co ─► Cloudflare edge (proxied) ──► 3.222.128.25
                                                                      │
                                                         nginx :443 (TLS)
                                                                      │
                                                     gunicorn :8000 (web)
                                                     worker.py  (jobs)
```

- **Registrar:** GoDaddy (owns the `.co` registration)
- **DNS:** Cloudflare (nameservers `jean.ns.cloudflare.com` + `ignat.ns.cloudflare.com`)
- **Hosting:** AWS Lightsail instance `wildframe-prod`
  (`micro_3_0`, 1 GB / 2 vCPU / 40 GB, Ubuntu 24.04, us-east-1a)
- **Static IP:** `3.222.128.25` (attached to the instance — free while
  attached; **never release it** without attaching a replacement first)
- **Origin:** nginx on the VM terminates TLS; gunicorn on 127.0.0.1:8000
- **Database:** Neon (PostGIS) — `WILDFRAME_DATABASE_URL` in the VM `.env`
- **Photos:** AWS S3 (`photo-hold-542489917871-us-east-1-an`)

## DNS records (Cloudflare)

| Type | Name | Target | Proxy status |
|---|---|---|---|
| A | `@` | `3.222.128.25` | Proxied (orange) |
| A | `www` | `3.222.128.25` | Proxied (orange) |

Both are proxied so the origin IP stays hidden behind Cloudflare's edge, and
Cloudflare's Universal SSL covers both hostnames at the edge.

## Certificates

- **Origin (VM):** Let's Encrypt cert for `pyrae.co` + `www.pyrae.co`,
  issued by `deploy/aws/tls.sh` (certbot, webroot). Renewal is automatic via
  certbot's systemd timer. Certs live in `/etc/letsencrypt/live/pyrae.co/`.
- **Cloudflare (edge):** Universal SSL.
- **SSL/TLS mode:** **Full (strict)** — Cloudflare requires a valid origin
  cert; an invalid/missing one shows **525**.

## How to rebuild (if it ever needs redoing)

1. **GoDaddy** → Domain → Nameservers → Cloudflare's two nameservers.
2. **Cloudflare** → Add site `pyrae.co` → add the two A records above
   (proxied).
3. **Lightsail** → create instance + static IP via
   `deploy/aws/create_instance.py` (see `DEPLOY.md` §3).
4. **VM** → rsync code + `.env` (with the **Neon** DB URL), run
   `deploy/aws/bootstrap.sh` (services + stage-1 nginx).
5. **TLS** → once DNS points at the VM, run `deploy/aws/tls.sh`
   (certbot + full nginx config).
6. **Cloudflare** → SSL/TLS → **Full (strict)**.

## Verification commands

```bash
dig +short NS pyrae.co                     # expect *.ns.cloudflare.com
dig @1.1.1.1 +short pyrae.co A             # Cloudflare edge IPs (proxied)
ssh -i ~/.ssh/wildframe-prod-key.pem ubuntu@3.222.128.25 \
  'sudo certbot certificates'              # origin cert status
curl -s -o /dev/null -w '%{http_code}\n' https://pyrae.co/        # 200
curl -s -o /dev/null -w '%{http_code}\n' https://www.pyrae.co/    # 200
curl -s https://pyrae.co/healthz           # {"db":"ok","status":"ok"}
```

## Pitfalls learned (important!)

1. **A missing `www` record fails the whole cert.** Certbot requests one cert
   for both hostnames — if `www` has no DNS record it fails with
   `NXDOMAIN looking up A for www.pyrae.co` and **no cert is issued at all**.
   Add the record, wait for propagation, rerun `tls.sh`.
2. **DNS cache skew.** The local resolver can serve a stale answer for a
   deleted record (looks fine, isn't). Always check with
   `dig @1.1.1.1` / `@8.8.8.8`.
3. **525 in Full (strict) = origin cert problem.** Cloudflare only serves the
   site once the origin presents a valid cert for the hostname. During the
   gap between DNS flip and `tls.sh` the site 525s — that's expected.
4. **nginx `http2` directive.** Ubuntu 24.04 ships nginx 1.24, which rejects
   `http2 on;` — use `listen 443 ssl http2;` (already correct in `tls.sh`).
5. **The `micro_ipv6_3_0` Lightsail bundle can't attach a static IP.** Use
   `micro_3_0` ($7/mo) for a stable DNS target.
6. **VM `.env` must point at Neon, not localhost.** The local dev `.env` has a
   local DB URL; copying it to the VM breaks the app (worker `PoolTimeout`,
   `/healthz` 500). `bootstrap.sh` refuses to run on a localhost URL.

## Related files

- `deploy/aws/` — `create_instance.py`, `bootstrap.sh`, `tls.sh`, systemd units
- `DEPLOY.md` — full Lightsail + Neon deployment walkthrough
- `fly.toml` / `Dockerfile` — preserved from the previous Fly.io era (unused)
