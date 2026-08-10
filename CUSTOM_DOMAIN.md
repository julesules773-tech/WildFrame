# Custom Domain — pyrae.co

Documentation of how `pyrae.co` and `www.pyrae.co` are wired to the WildFrame
app on Fly.io. Written 2026-08-10, after the initial setup.

## Current architecture

```
Browser ──► https://pyrae.co ────────► Fly.io edge (wildframe.fly.dev)
             (DNS-only, goes direct)      │
                                         ▼
                                     web machines
                                        (gunicorn)

Browser ──► https://www.pyrae.co ──► Cloudflare edge (proxied) ──► Fly.io origin
             (orange cloud)                │
                                           ▼
                                       wildframe.fly.dev
```

- **Registrar:** GoDaddy (owns the `.co` registration)
- **DNS:** Cloudflare (nameservers `jean.ns.cloudflare.com` + `ignat.ns.cloudflare.com`)
- **Hosting:** Fly.io app `wildframe` (2 web machines + 1 worker + 1 standby),
  primary region `iad`
- **Origin:** `wildframe.fly.dev` (the app's fly.dev URL, always works)
- **Database:** Neon (PostGIS) — reachable from the app via `WILDFRAME_DATABASE_URL`

## DNS records (Cloudflare)

| Type | Name | Target | Proxy status |
|---|---|---|---|
| CNAME | `@` | `wildframe.fly.dev` | DNS only (grey) |
| CNAME | `www` | `wildframe.fly.dev` | Proxied (orange) |

Notes:
- The apex record is DNS-only because it works fine direct-to-Fly (Fly serves
  its own Let's Encrypt cert for `pyrae.co`). It can be set to **proxied
  (orange)** for consistency — the origin handshake still works because
  Cloudflare uses the CNAME target's SNI.
- Do **not** use plain `A` records pointing at Fly's shared IPv4
  (`66.241.125.28`) with proxying enabled — Cloudflare then connects to the
  origin with SNI `pyrae.co`, which Fly's edge won't answer (see pitfalls).

## Certificates

- **Fly (origin):** Let's Encrypt certs for `pyrae.co` + `www.pyrae.co`,
  managed automatically (`fly certs add`). Renewal is automatic.
- **Cloudflare (edge):** Universal SSL covers `pyrae.co` + `www.pyrae.co`
  automatically.
- **Cloudflare SSL/TLS mode:** **Full (strict)** — Cloudflare validates the
  origin cert for `wildframe.fly.dev`.

## How it was set up (if it ever needs rebuilding)

1. **GoDaddy** → Domain → Nameservers → set Cloudflare's two nameservers.
2. **Cloudflare** → Add site `pyrae.co` (Free plan) → delete imported
   `A`/`AAAA` records for `@` and `www` → add the two CNAME rows above.
3. **Fly** → `fly certs add pyrae.co` and `fly certs add www.pyrae.co`
   (from the repo dir; runs against the linked app).
4. **Cloudflare** → SSL/TLS → **Full (strict)**.

## Verification commands

```bash
# DNS points at Cloudflare nameservers?
dig +short NS pyrae.co                       # expect *.ns.cloudflare.com

# Records served (use a clean resolver to dodge cache):
dig @8.8.8.8 +short pyrae.co A
dig @8.8.8.8 +short www.pyrae.co A

# Certificates on Fly:
fly certs list                                # expect Issued
fly certs check pyrae.co                      # live validation status

# End-to-end HTTPS:
curl -s -o /dev/null -w '%{http_code}\n' https://pyrae.co/        # 200
curl -s -o /dev/null -w '%{http_code}\n' https://www.pyrae.co/    # 200
curl -s https://pyrae.co/healthz              # {"db":"ok","status":"ok"}
```

## Pitfalls learned (important!)

1. **Fly edge propagation can take ~2 hours.** After `fly certs add`, `fly
   certs list` may show `Issued` while `fly certs check` says `Not verified`
   and HTTPS fails with a TLS reset (`SSL_ERROR_SYSCALL` / `EOF in violation
   of protocol`). This is normal — the cert reaches the edge nodes eventually.
   Re-adding certs or redeploying does **not** speed it up.
2. **`fly certs list` vs `fly certs check` disagree** right after issuance —
   `list` flips to `Issued` first; trust the HTTPS test, not the status text.
3. **Don't proxy `A` records to the shared IPv4.** Cloudflare proxied
   `A 66.241.125.28` → origin TLS with SNI `pyrae.co` → **HTTP 525**. Use
   `CNAME → wildframe.fly.dev` so the origin handshake uses SNI
   `wildframe.fly.dev` (which Fly always serves).
4. **Trial org limits:** Fly trial orgs (no credit card) can't allocate a
   dedicated IPv4 (`fly ips allocate-v4` → *"disabled for trial
   organizations"*). Custom-domain certs DO work on trial orgs — it just
   looks broken until edge propagation finishes.
5. **`force_https = true` does not block Let's Encrypt** — Fly's edge answers
   ACME challenges before applying the redirect.
6. **DNS caching** can make it look like nothing changed: query
   `dig @8.8.8.8` / `@1.1.1.1` directly to bypass a stale local resolver.

## Related files

- `fly.toml` — Fly app config (processes, port 8080, health check, `iad`)
- `Dockerfile` / `.dockerignore` — container build
- `DEPLOY.md` — full Fly + Neon deployment walkthrough
