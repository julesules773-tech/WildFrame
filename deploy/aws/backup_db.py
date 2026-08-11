#!/usr/bin/env python3
"""Nightly pg_dump of the WildFrame database, uploaded to S3.

Keeps the last 14 dumps on S3 and the last 7 locally. Credentials come from
the same .env the app uses (dotenv), so no extra setup is needed.

Cron (root's crontab or the ubuntu user via crontab -e):
    15 3 * * * cd /home/ubuntu/wildframe && ./.venv/bin/python deploy/aws/backup_db.py >> /home/ubuntu/backups/backup.log 2>&1
"""
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

BACKUP_DIR = Path("/home/ubuntu/backups")
BACKUP_DIR.mkdir(parents=True, exist_ok=True)
S3_BUCKET = os.environ.get("WILDFRAME_S3_BUCKET", "")
S3_PREFIX = "db-backups"
KEEP_S3 = 14
KEEP_LOCAL = 7


def main() -> int:
    url = os.environ.get("WILDFRAME_DATABASE_URL")
    if not url or not S3_BUCKET:
        print("missing WILDFRAME_DATABASE_URL or WILDFRAME_S3_BUCKET", file=sys.stderr)
        return 1

    ts = time.strftime("%Y%m%d-%H%M%S")
    local = BACKUP_DIR / f"wildframe-{ts}.dump"

    print(f"[backup] dumping to {local} ...")
    subprocess.run(
        ["pg_dump", url, "-Fc", "-f", str(local)],
        check=True,
        capture_output=True,
    )
    print(f"[backup] dump OK ({local.stat().st_size / 1024 / 1024:.1f} MB)")

    import boto3

    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION") or "us-east-1")
    key = f"{S3_PREFIX}/{local.name}"
    s3.upload_file(str(local), S3_BUCKET, key)
    print(f"[backup] uploaded s3://{S3_BUCKET}/{key}")

    # Prune old S3 backups (keep the KEEP_S3 most recent).
    resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PREFIX + "/")
    objs = sorted(resp.get("Contents") or [], key=lambda o: o["LastModified"])
    for o in objs[:-KEEP_S3]:
        s3.delete_object(Bucket=S3_BUCKET, Key=o["Key"])
        print(f"[backup] pruned s3: {o['Key']}")

    # Prune old local backups.
    local_dumps = sorted(BACKUP_DIR.glob("wildframe-*.dump"))
    for f in local_dumps[:-KEEP_LOCAL]:
        f.unlink(missing_ok=True)
        print(f"[backup] pruned local: {f.name}")

    print("[backup] complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
