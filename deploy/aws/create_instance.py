"""Create the WildFrame Lightsail production instance via boto3.

Creates (idempotently enough for a clean rerun):
  - a 1 GB Ubuntu 24.04 instance (`micro_3_0`, $7/mo) — the static-IP-capable
    bundle; the cheaper `micro_ipv6_3_0` cannot attach a static IP
  - an SSH key pair (saved to ~/.ssh/wildframe-prod-key.pem)
  - a static IP (attached — free while attached to a running instance)
  - open firewall ports 22 / 80 / 443

Requires AWS creds in .env (AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY) with
Lightsail permissions. Run from the repo root:
    .venv/bin/python deploy/aws/create_instance.py
"""
import os
import re
import sys
import time

import boto3

REGION = "us-east-1"
INSTANCE_NAME = "wildframe-prod"
KEY_NAME = "wildframe-prod-key"
IP_NAME = "wildframe-prod-ip"
BUNDLE = "micro_3_0"  # 1 GB RAM, 2 vCPU, 40 GB SSD — $7/mo (static-IP capable)
AZ = "us-east-1a"


def load_env_file(path):
    env = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                m = re.match(r"(?:export\s+)?([A-Za-z0-9_]+)\s*=\s*(.*)$", line)
                if m:
                    env[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return env


env = load_env_file(".env")
session = boto3.Session(
    aws_access_key_id=env["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=env["AWS_SECRET_ACCESS_KEY"],
    region_name=REGION,
)
ls = session.client("lightsail", region_name=REGION)

# 0. Permission sanity (raises if the IAM user lacks Lightsail access)
ls.get_regions()

# 1. Pick an Ubuntu blueprint (prefer 24.04 for python3.12)
blueprints = ls.get_blueprints(includeInactive=False)["blueprints"]
ubuntu = [b for b in blueprints if b.get("platform") == "LINUX_UNIX" and "ubuntu" in b["blueprintId"]]
preferred = [b for b in ubuntu if "24_04" in b["blueprintId"] or "24.04" in b.get("description", "")]
bp = (preferred or ubuntu)[0]
print("blueprint:", bp["blueprintId"], "-", bp.get("description"))
bundle = next((b for b in ls.get_bundles()["bundles"] if b["bundleId"] == BUNDLE), None)
if not bundle:
    sys.exit(f"bundle {BUNDLE} not found — run get_bundles() to list current IDs")
print("bundle:", BUNDLE, "-", bundle["ramSizeInGb"], "GB,", bundle["diskSizeInGb"], "GB SSD")

# 2. Key pair (private key is only returned ONCE — keep the local file safe)
os.makedirs(os.path.expanduser("~/.ssh"), exist_ok=True)
key_path = os.path.expanduser("~/.ssh/" + KEY_NAME + ".pem")
if os.path.exists(key_path):
    print("key already exists locally:", key_path)
else:
    try:
        priv = ls.create_key_pair(keyPairName=KEY_NAME).get("privateKeyBase64")
    except Exception as e:
        print("key pair exists but no local file — deleting and recreating:", str(e)[:120])
        ls.delete_key_pair(keyPairName=KEY_NAME)
        time.sleep(5)
        priv = ls.create_key_pair(keyPairName=KEY_NAME).get("privateKeyBase64")
    if not priv:
        sys.exit("could not obtain private key material")
    with open(key_path, "w") as f:
        f.write(priv)
    os.chmod(key_path, 0o600)
    print("key saved:", key_path)

# 3. Create instance
print("creating instance (takes ~1-2 min)...")
try:
    ls.create_instances(
        instanceNames=[INSTANCE_NAME],
        availabilityZone=AZ,
        blueprintId=bp["blueprintId"],
        bundleId=BUNDLE,
        keyPairName=KEY_NAME,
        tags=[{"key": "app", "value": "wildframe"}],
    )
except Exception as e:
    if "already in use" in str(e):
        print("instance already exists — continuing")
    else:
        raise
for _ in range(60):
    st = ls.get_instance(instanceName=INSTANCE_NAME)["instance"]["state"]["name"]
    if st == "running":
        break
    time.sleep(5)
print("instance state:", st)
if st != "running":
    sys.exit("instance not running — check the Lightsail console")

# 4. Static IP + attach (free while attached to a running instance)
try:
    ls.allocate_static_ip(staticIpName=IP_NAME)
except Exception as e:
    print("allocate note:", str(e)[:120])
ls.attach_static_ip(staticIpName=IP_NAME, instanceName=INSTANCE_NAME)
ip = ls.get_static_ip(staticIpName=IP_NAME)["staticIp"]["ipAddress"]
print("static IP:", ip)

# 5. Open ports (SSH is default; ensure 80 + 443)
for port in (22, 80, 443):
    ls.open_instance_public_ports(
        instanceName=INSTANCE_NAME,
        portInfo={"protocol": "tcp", "fromPort": port, "toPort": port},
    )
print("ports open: 22, 80, 443")

print("\nDONE. SSH command:")
print(f"  ssh -i {key_path} ubuntu@{ip}")
print("Then rsync the project (see DEPLOY.md) and run deploy/aws/bootstrap.sh")
