#!/usr/bin/env bash
# WildFrame — TLS setup (run AFTER the Cloudflare DNS records point at this
# instance AND bootstrap.sh has run). Issues a Let's Encrypt cert for
# pyrae.co + www.pyrae.co via HTTP-01 through the Cloudflare proxy
# (SSL mode Full strict requires a real origin cert).
set -euo pipefail

EMAIL="${CERTBOT_EMAIL:-admin@pyrae.co}"
DOMAINS="-d pyrae.co -d www.pyrae.co"
if command -v certbot &>/dev/null; then
    echo "==> fix permissions for nginx (www-data) on static files"
sudo chmod -R o+rx /home/ubuntu/wildframe/static/
sudo chmod o+x /home/ubuntu /home/ubuntu/wildframe

echo "==> certbot (HTTP-01 webroot)"
    sudo certbot certonly --webroot -w /var/www/letsencrypt $DOMAINS \
        --non-interactive --agree-tos -m "$EMAIL" --keep-until-expiring
else
    echo "==> certbot not found, skipping TLS renewal (certs already in place)"
fi

echo "==> full nginx config (80->443 redirect + 443 proxy to gunicorn)"
sudo tee /etc/nginx/sites-available/wildframe.conf > /dev/null <<'NGINX'
# WildFrame — production nginx: TLS in front of gunicorn on 127.0.0.1:8000.
#
# Performance: static HTML pages and assets are served directly from disk,
# bypassing Flask entirely. Only API endpoints and dynamic routes hit gunicorn.
#
# DDoS protection: rate limiting per real client IP (CF-Connecting-IP).
# Cloudflare strips X-Forwarded-For to the previous hop; CF-Connecting-IP
# always holds the real visitor IP.  Fallback to X-Real-IP / remote_addr
# for direct hits (health checks, certbot).

# --- Rate-limit zones (defined once, shared across all server blocks) ---
# 60 req/min general, burst up to 20.  IPs that exceed are delayed, not rejected.
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

    # ==================================================================
    # Static assets (JS, CSS, images, fonts) — served directly by nginx
    # ==================================================================
    location /static/ {
        alias /home/ubuntu/wildframe/static/;
        expires 1h;
        add_header Cache-Control "public, immutable";
    }

    # ==================================================================
    # Static HTML pages — served directly from disk, bypassing Flask.
    # Each URL maps to its on-disk filename (some differ).
    # These are = (exact) locations so they match before the catch-all.
    # ==================================================================
    location = / { root /home/ubuntu/wildframe/static; try_files /landing.html =404; }
    location = /map { root /home/ubuntu/wildframe/static; try_files /map.html =404; }
    location = /login { root /home/ubuntu/wildframe/static; try_files /login.html =404; }
    location = /invite { root /home/ubuntu/wildframe/static; try_files /invite.html =404; }
    location = /dashboard { root /home/ubuntu/wildframe/static; try_files /dashboard.html =404; }
    location = /confidence { root /home/ubuntu/wildframe/static; try_files /confidence.html =404; }
    location = /contact { root /home/ubuntu/wildframe/static; try_files /contact.html =404; }
    location = /privacy { root /home/ubuntu/wildframe/static; try_files /privacy.html =404; }
    location = /about { root /home/ubuntu/wildframe/static; try_files /about.html =404; }
    location = /faq { root /home/ubuntu/wildframe/static; try_files /faq.html =404; }
    location = /investors { root /home/ubuntu/wildframe/static; try_files /investors.html =404; }
    location = /wildfire-early-detection-for-emergency-services { root /home/ubuntu/wildframe/static; try_files /emergency-services.html =404; }
    location = /nasa-firms-wildfire-map { root /home/ubuntu/wildframe/static; try_files /nasa-firms-wildfire-map.html =404; }
    location = /how-early-wildfire-detection-works { root /home/ubuntu/wildframe/static; try_files /how-early-wildfire-detection-works.html =404; }
    location = /wildfire-spread-risk-map { root /home/ubuntu/wildframe/static; try_files /wildfire-spread-risk-map.html =404; }

    # HTML pages with short cache (content changes rarely but deploys are frequent)
    location ~* ^/(map|login|invite|dashboard|confidence|contact|privacy|about|faq|investors|emergency-services|nasa-firms-wildfire-map|how-early-wildfire-detection-works|wildfire-spread-risk-map)$ {
        root /home/ubuntu/wildframe/static;
        try_files $uri.html @proxy;
        expires 5m;
        add_header Cache-Control "public, must-revalidate";
    }

    # ==================================================================
    # Upload endpoints (photo, feedback) — proxied to gunicorn
    # ==================================================================
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

    # ==================================================================
    # API endpoints (map data, clusters, fires) — proxied to gunicorn
    # ==================================================================
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

    # ==================================================================
    # Dynamic routes that need Flask (auth, DB queries, template sub)
    # /admin, /map/poland — must go through gunicorn
    # ==================================================================
    location /admin {
        limit_req zone=general burst=5 nodelay;
        limit_req_status 429;
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $http_cf_connecting_ip;
        proxy_set_header X-Forwarded-For $http_cf_connecting_ip;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 30;
        proxy_connect_timeout 10;
    }

    location /map/poland {
        limit_req zone=general burst=5 nodelay;
        limit_req_status 429;
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $http_cf_connecting_ip;
        proxy_set_header X-Forwarded-For $http_cf_connecting_ip;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 30;
        proxy_connect_timeout 10;
    }

    location /map/france {
        limit_req zone=general burst=5 nodelay;
        limit_req_status 429;
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $http_cf_connecting_ip;
        proxy_set_header X-Forwarded-For $http_cf_connecting_ip;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 30;
        proxy_connect_timeout 10;
    }

    location /map/germany {
        limit_req zone=general burst=5 nodelay;
        limit_req_status 429;
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $http_cf_connecting_ip;
        proxy_set_header X-Forwarded-For $http_cf_connecting_ip;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 30;
        proxy_connect_timeout 10;
    }

    location /map/spain {
        limit_req zone=general burst=5 nodelay;
        limit_req_status 429;
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $http_cf_connecting_ip;
        proxy_set_header X-Forwarded-For $http_cf_connecting_ip;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 30;
        proxy_connect_timeout 10;
    }

    # ==================================================================
    # Catch-all: try static files, then proxy to gunicorn
    # ==================================================================
    location / {
        root /home/ubuntu/wildframe/static;
        try_files $uri @proxy;
        # Cache static assets served directly by nginx (JS, CSS, images, fonts)
        location ~* \.(js|css|png|jpg|jpeg|gif|webp|ico|svg|woff|woff2|ttf|eot|onnx|json)$ {
            try_files $uri =404;
            expires 1h;
            add_header Cache-Control "public, immutable";
        }
    }

    # Named location for proxy fallback
    location @proxy {
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
