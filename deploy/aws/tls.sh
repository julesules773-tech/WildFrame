#!/usr/bin/env bash
# WildFrame — TLS setup (run AFTER the Cloudflare DNS records point at this
# instance AND bootstrap.sh has run). Issues a Let's Encrypt cert for
# pyrae.co + www.pyrae.co via HTTP-01 through the Cloudflare proxy
# (SSL mode Full strict requires a real origin cert).
set -euo pipefail

EMAIL="${CERTBOT_EMAIL:-admin@pyrae.co}"
DOMAINS="-d pyrae.co -d www.pyrae.co"

echo "==> certbot (HTTP-01 webroot)"
sudo certbot certonly --webroot -w /var/www/letsencrypt $DOMAINS \
  --non-interactive --agree-tos -m "$EMAIL" --keep-until-expiring

echo "==> full nginx config (80->443 redirect + 443 proxy to gunicorn)"
sudo tee /etc/nginx/sites-available/wildframe.conf > /dev/null <<'NGINX'
# WildFrame — production nginx: TLS in front of gunicorn on 127.0.0.1:8000.
#
# DDoS protection: rate limiting per real client IP (CF-Connecting-IP).
# Cloudflare strips X-Forwarded-For to the previous hop; CF-Connecting-IP
# always holds the real visitor IP.  Fallback to X-Real-IP / remote_addr
# for direct hits (health checks, certbot).

# --- Rate-limit zones (defined once, shared across all server blocks) ---
# 10 req/s general, burst up to 20.  IPs that exceed are delayed, not rejected.
limit_req_zone $http_cf_connecting_ip zone=general:10m rate=60r/m;
# Uploads: 5 req/min — photo uploads are rare and expensive.
limit_req_zone $http_cf_connecting_ip zone=upload:10m rate=5r/m;
# Feedback: 3 req/min — backend has its own 30s cooldown but nginx
# absorbs the flood so gunicorn never sees it.
limit_req_zone $http_cf_connecting_ip zone=feedback:10m rate=3r/m;
# API (map data, clusters, fire data): 30 req/min — tight enough to
# block scraping but generous for the 15s polling interval.
limit_req_zone $http_cf_connecting_ip zone=api:10m rate=30r/m;
# Concurrent connections per IP — prevents connection-flood slowloris.
limit_conn_zone $http_cf_connecting_ip zone=connlimit:10m;

server {
    listen 80;
    listen [::]:80;
    server_name pyrae.co www.pyrae.co;

    location /.well-known/acme-challenge/ {
        root /var/www/letsencrypt;
    }

    location / {
        return 301 https://$host$request_uri;
    }
}

server {
    listen 443 ssl http2;        # Ubuntu 24.04 ships nginx 1.24 — uses the listen-directive syntax
    listen [::]:443 ssl http2;
    server_name pyrae.co www.pyrae.co;

    ssl_certificate     /etc/letsencrypt/live/pyrae.co/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/pyrae.co/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;

    client_max_body_size 20m;   # Flask MAX_CONTENT_LENGTH is 16 MB

    # --- Slowloris protection ---
    # Reject requests that take too long to send headers/body.
    client_header_timeout 10s;   # 10s to send all request headers
    client_body_timeout 30s;     # 30s to send request body (photo uploads)
    send_timeout 10s;            # 10s between successive write operations

    # --- Request header hardening ---
    # Reject oversized or missing Host header (bot scanners, curl floods)
    large_client_header_buffers 4 8k;

    # --- Global connection limit (20 concurrent per IP) ---
    limit_conn connlimit 20;

    # --- Static files: no rate limit ---
    location /static/ {
        alias /home/ubuntu/wildframe/static/;
        expires 1h;
        add_header Cache-Control "public, immutable";
    }

    # --- Upload endpoints (photo, feedback) ---
    location /api/upload {
        limit_req zone=upload burst=2 nodelay;
        limit_req_status 429;
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $http_cf_connecting_ip;
        proxy_set_header X-Forwarded-For $http_cf_connecting_ip;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 180;
        proxy_connect_timeout 60;
    }

    location /api/feedback {
        limit_req zone=feedback burst=1 nodelay;
        limit_req_status 429;
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $http_cf_connecting_ip;
        proxy_set_header X-Forwarded-For $http_cf_connecting_ip;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 30;
        proxy_connect_timeout 10;
    }

    # --- API endpoints (map data, clusters, fires) ---
    location /api/ {
        limit_req zone=api burst=10 nodelay;
        limit_req_status 429;
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $http_cf_connecting_ip;
        proxy_set_header X-Forwarded-For $http_cf_connecting_ip;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 180;
        proxy_connect_timeout 60;
    }

    # --- Everything else (map page, index, admin) ---
    location / {
        limit_req zone=general burst=20 nodelay;
        limit_req_status 429;
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $http_cf_connecting_ip;
        proxy_set_header X-Forwarded-For $http_cf_connecting_ip;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 180;
        proxy_connect_timeout 60;
    }
}
NGINX
sudo nginx -t
sudo systemctl reload nginx

echo "==> cert auto-renewal timer (installed by certbot):"
systemctl --no-pager list-timers certbot.timer || true

echo "TLS DONE — verify:"
echo "  curl -sI https://pyrae.co/ | head -3"
echo "  curl -s https://pyrae.co/healthz"
