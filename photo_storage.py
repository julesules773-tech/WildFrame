#!/usr/bin/env python3
"""
photo_storage.py — Uploaded-photo storage (local disk or S3)
============================================================

Where accepted report photos live:

* **S3 mode** — if ``WILDFRAME_S3_BUCKET`` is set, accepted photos are
  uploaded to S3 (publicly readable) and the report's ``photo_url`` is the
  full public S3 URL. This is what you want when running on a PaaS with an
  ephemeral filesystem (Render/Railway/Fly), or with more than one server.
* **Local mode** — if no bucket is configured (local dev, no AWS account),
  photos are saved under ``uploads/`` and served by Flask's
  ``/uploads/<filename>`` route, exactly as before.

Photos are always staged on local disk first, because EXIF GPS extraction
and the Roboflow AI scan both read from disk. Once a report is accepted,
the staging file is pushed to S3 (S3 mode) or kept in place (local mode).

Configuration (env vars)
------------------------
``WILDFRAME_S3_BUCKET``           bucket name; its presence enables S3 mode
``AWS_ACCESS_KEY_ID``             standard boto3 credential
``AWS_SECRET_ACCESS_KEY``         standard boto3 credential
``AWS_REGION`` / ``AWS_DEFAULT_REGION``  bucket region (default ``us-east-1``)
``WILDFRAME_S3_PUBLIC_URL``       optional public base URL (e.g. a CloudFront
                                  domain); defaults to the bucket's S3 URL
``WILDFRAME_S3_PREFIX``           object-key prefix (default ``photos``)

Usage
-----
    import photo_storage

    url = photo_storage.store_photo("abc123.jpg", Path("uploads/abc123.jpg"))
    # → "https://bucket.s3.us-east-1.amazonaws.com/photos/abc123.jpg"

    photo_storage.delete_photo(url)   # handles S3 URLs and /uploads/ paths
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from urllib.parse import urlparse

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

S3_BUCKET = (os.environ.get("WILDFRAME_S3_BUCKET") or "").strip()
S3_REGION = (
    os.environ.get("AWS_REGION")
    or os.environ.get("AWS_DEFAULT_REGION")
    or "us-east-1"
)
S3_PUBLIC_URL = (os.environ.get("WILDFRAME_S3_PUBLIC_URL") or "").rstrip("/")
S3_PREFIX = (os.environ.get("WILDFRAME_S3_PREFIX") or "photos").strip("/")

UPLOAD_DIR = Path("uploads")

_CONTENT_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def s3_enabled() -> bool:
    """True when S3 storage is configured and enabled."""
    return bool(S3_BUCKET)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_s3_client_cache = None
_s3_client_lock = threading.Lock()


def _s3_client():
    """Lazily-created boto3 client — only built when S3 mode is used."""
    global _s3_client_cache
    if _s3_client_cache is None:
        with _s3_client_lock:
            if _s3_client_cache is None:  # double-check after acquiring lock
                import boto3  # heavy import kept out of the local-dev path
                _s3_client_cache = boto3.client("s3", region_name=S3_REGION)
    return _s3_client_cache


def _key_for(filename: str) -> str:
    """Object key for a stored photo filename."""
    return f"{S3_PREFIX}/{filename}"


def _is_s3_host(host: str) -> bool:
    """True if *host* is a host we manage (our bucket or public URL)."""
    if host == f"{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com":
        return True
    if host == f"{S3_BUCKET}.s3.amazonaws.com":
        return True
    if S3_PUBLIC_URL:
        if host == urlparse(S3_PUBLIC_URL).netloc:
            return True
    return False


def _key_from_url(url: str) -> str | None:
    """
    Extract the S3 object key from a stored ``photo_url``, or None if the
    URL isn't one of ours (e.g. a picsum.photos seed URL or a local path).
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return None
    if not _is_s3_host(parsed.netloc):
        return None
    path = parsed.path.lstrip("/")
    if path.startswith(f"{S3_PREFIX}/") and path[len(S3_PREFIX) + 1:]:
        return path
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def store_photo(filename: str, staging_path: Path) -> str:
    """
    Persist an accepted photo and return the URL to store in ``photo_url``.

    S3 mode: uploads the staging file to S3 (publicly readable), removes
    the staging copy, and returns the public URL. Local mode: returns
    ``/uploads/<filename>`` and leaves the file in place.

    Raises on S3 failure, leaving the staging file untouched so the caller
    can decide how to respond.
    """
    if not s3_enabled():
        return f"/uploads/{filename}"

    key = _key_for(filename)
    content_type = _CONTENT_TYPES.get(Path(filename).suffix.lower(), "application/octet-stream")

    client = _s3_client()
    # Prefer ACL-style public read; modern buckets with "Bucket owner
    # enforced" object ownership reject ACLs and need a bucket policy
    # instead — retry without the ACL in that case.
    #
    # NOTE: upload_file() raises boto3's S3UploadFailedError (not a raw
    # ClientError) when PutObject fails, so match on the message text.
    try:
        client.upload_file(
            str(staging_path), S3_BUCKET, key,
            ExtraArgs={"ACL": "public-read", "ContentType": content_type},
        )
    except Exception as exc:
        if "AccessControlListNotSupported" not in str(exc):
            raise
        log.warning(
            "Bucket %s rejects object ACLs — uploaded without public-read ACL. "
            "Add a public-read bucket policy (see README) or photos will 403.",
            S3_BUCKET,
        )
        client.upload_file(
            str(staging_path), S3_BUCKET, key,
            ExtraArgs={"ContentType": content_type},
        )

    # Staging copy is no longer needed once safely in S3.
    staging_path.unlink(missing_ok=True)

    if S3_PUBLIC_URL:
        return f"{S3_PUBLIC_URL}/{key}"
    return f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{key}"


def delete_photo(photo_url: str) -> None:
    """
    Delete a stored photo by its ``photo_url``. Handles both S3 URLs and
    local ``/uploads/`` paths; foreign URLs (seed placeholders) are no-ops.

    Best-effort: failures are logged, never raised — the DB row is the
    source of truth and deletion is cleanup.
    """
    if not photo_url:
        return

    key = _key_from_url(photo_url)
    if key:
        try:
            _s3_client().delete_object(Bucket=S3_BUCKET, Key=key)
            log.info("Deleted S3 object %s/%s", S3_BUCKET, key)
        except Exception as exc:
            log.warning("Failed to delete S3 object %s/%s: %s", S3_BUCKET, key, exc)
        return

    if photo_url.startswith("/uploads/"):
        p = UPLOAD_DIR / Path(photo_url).name
        try:
            if p.exists():
                p.unlink()
                log.info("Deleted local photo %s", p)
        except OSError as exc:
            log.warning("Failed to delete local photo %s: %s", p, exc)
