#!/usr/bin/env bash
# WildFrame — provision a local PostgreSQL + PostGIS on this host and point
# .env at it (WILDFRAME_DATABASE_URL → 127.0.0.1). Idempotent: safe to
# re-run; each run rotates the local DB password.
#
# Rationale: a self-hosted DB on the same VM has ZERO metered egress, so the
# Neon-style "data transfer quota exceeded" outage cannot happen again.
#
# Usage:  bash /home/ubuntu/wildframe/deploy/aws/setup_local_db.sh
set -euo pipefail

cd /home/ubuntu/wildframe

echo "==> installing postgresql + postgis (if missing)"
if ! command -v pg_isready >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y postgresql postgis
fi

echo "==> starting postgres"
sudo systemctl enable --now postgresql
pg_isready -h 127.0.0.1 -p 5432

echo "==> creating wildframe role + database + postgis extension"
PW=$(openssl rand -hex 16)
if ! sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='wildframe'" | grep -q 1; then
  sudo -u postgres psql -c "CREATE ROLE wildframe LOGIN PASSWORD '$PW'"
  echo "   role created"
else
  sudo -u postgres psql -c "ALTER ROLE wildframe WITH LOGIN PASSWORD '$PW'"
  echo "   role exists, password rotated"
fi
if ! sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='wildframe'" | grep -q 1; then
  sudo -u postgres createdb -O wildframe wildframe
  echo "   database created"
fi
sudo -u postgres psql -d wildframe -c "CREATE EXTENSION IF NOT EXISTS postgis"

echo "==> pointing .env at the local database"
if grep -q '^WILDFRAME_DATABASE_URL=' .env; then
  sed -i "s|^WILDFRAME_DATABASE_URL=.*|WILDFRAME_DATABASE_URL=postgresql://wildframe:${PW}@127.0.0.1:5432/wildframe|" .env
else
  echo "WILDFRAME_DATABASE_URL=postgresql://wildframe:${PW}@127.0.0.1:5432/wildframe" >> .env
fi
echo "   WILDFRAME_DATABASE_URL=$(grep '^WILDFRAME_DATABASE_URL=' .env | sed -E 's#(://[^:]+:)[^@]+@#\1***@#')"

echo "==> schema + procrastinate queue tables"
./.venv/bin/python migrate.py 2>&1 | tail -6

echo "LOCAL DB READY"
