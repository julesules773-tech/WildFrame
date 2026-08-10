# WildFrame — production image
# One image, two processes (see fly.toml [processes]):
#   web    = gunicorn serving the Flask API + static UI
#   worker = Procrastinate job worker (FIRMS, grid advance, weather)
#
# The Dockerfile CMD is only a fallback — Fly overrides it with the
# [processes] start commands.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first (layer is cached until requirements.txt changes).
# requirements.txt is deliberately lean: no inference-sdk / scipy / torch —
# the production scan path uses the hosted Roboflow API via stdlib urllib.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Application code (secrets are NOT baked in — .env is dockerignored,
# everything comes from `fly secrets set`).
COPY . .

EXPOSE 8080

CMD ["gunicorn", "server:app", "--workers", "2", "--bind", "0.0.0.0:8080", "--timeout", "120", "--access-logfile", "-"]
