#!/usr/bin/env bash
# WildFrame — Lightsail VM bootstrap (run as the ubuntu user, once).
# Prereq: the project is rsynced to /home/ubuntu/wildframe (including .env
# and deploy/aws/). Run:  bash /home/ubuntu/wildframe/deploy/aws/bootstrap.sh
set -euo pipefail

echo "==> sanity: production DB URL present on the VM"
if grep -qE 'localhost|127\\.0\\.0\\.1' .env; then
  echo "ERROR: .env WILDFRAME_DATABASE_URL points at a local database." >&2
  echo "       Set the Neon production URL (sslmode=require) before bootstrapping." >&2
  exit 1
fi

echo "==> apt packages"
sudo apt-get update -y
sudo apt-get install -y nginx certbot python3-certbot-nginx python3-venv python3-pip

echo "==> 2G swap (insurance on the 1 GB instance)"
if [ ! -f /swapfile ]; then
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

echo "==> python venv + requirements"
cd /home/ubuntu/wildframe
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip wheel
./.venv/bin/pip install -r requirements.txt

echo "==> systemd services (web + worker)"
sudo cp deploy/aws/wildframe-web.service deploy/aws/wildframe-worker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now wildframe-web wildframe-worker

echo "==> nginx (port 80 only, ACME challenge; full TLS conf comes from tls.sh)"
sudo mkdir -p /var/www/letsencrypt
sudo tee /etc/nginx/sites-available/wildframe.conf > /dev/null <<'NGINX'
# WildFrame — stage 1: HTTP only, serve ACME challenges.
# tls.sh replaces this with the full 80->443 redirect + 443 proxy config.
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name _;

    location /.well-known/acme-challenge/ {
        root /var/www/letsencrypt;
    }

    location / {
        return 503 "WildFrame origin — waiting for TLS setup\n";
    }
}
NGINX
sudo ln -sf /etc/nginx/sites-available/wildframe.conf /etc/nginx/sites-enabled/wildframe.conf
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx

echo "==> status"
systemctl --no-pager status wildframe-web --lines=5 || true
systemctl --no-pager status wildframe-worker --lines=5 || true
echo "BOOTSTRAP DONE — next: flip DNS, then run deploy/aws/tls.sh"
