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

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
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
