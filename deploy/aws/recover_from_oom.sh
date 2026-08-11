#!/usr/bin/env bash
# recover_from_oom.sh — stabilize the 1 GB Lightsail VM after a
# memory-exhaustion / swap-thrash freeze (pyrae.co unreachable, SSH
# banner timeouts). Safe to re-run: it only inspects, restarts
# WildFrame's own services, and creates a swapfile if one is missing.
#
# Usage:  sudo bash deploy/aws/recover_from_oom.sh
#         (or run as the ubuntu user where sudo is available)
set -u

echo "==> [1/6] Load & memory"
uptime
free -m

echo
echo "==> [2/6] Swap present? (bootstrap.sh provisions /swapfile)"
if [ -f /swapfile ] && swapon --show | grep -q /swapfile; then
  echo "  /swapfile active: $(swapon --show | tail -1)"
else
  echo "  missing/inactive — creating 2G swapfile..."
  if [ ! -f /swapfile ]; then
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile >/dev/null
  fi
  sudo swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
  echo "  swap now: $(swapon --show | tail -1)"
fi

echo
echo "==> [3/6] Was anything OOM-killed?"
sudo dmesg -T 2>/dev/null | grep -iE 'killed process|out of memory|oom' | tail -6 || echo "  (no OOM lines in dmesg — likely swap thrash, not OOM kill)"
echo "  meminfo CommitLimit/Committed: $(grep -E 'CommitLimit|Committed_AS' /proc/meminfo | tr '\n' ' ')"

echo
echo "==> [4/6] Service status (before)"
systemctl --no-pager status wildframe-web wildframe-worker postgresql --lines=0 2>&1 | grep -E '●|Active:' | head -6

echo
echo "==> [5/6] Restart WildFrame services in order"
sudo systemctl restart wildframe-web
sleep 2
sudo systemctl restart wildframe-worker
sleep 2
echo "  postgres: $(systemctl is-active postgresql) | web: $(systemctl is-active wildframe-web) | worker: $(systemctl is-active wildframe-worker)"

echo
echo "==> [6/6] Post-restart sanity"
sleep 5
curl -s -m 15 http://localhost:8000/healthz && echo
echo "  load: $(cut -d' ' -f1-3 /proc/loadavg) | free: $(free -m | awk '/Mem:/{print $3\"MB used / \"$2\"MB\"}')"

echo
echo "DONE. If memory stays tight under the worker's minute-cron advance job,"
echo "consider: (a) resize the instance to 2 GB in Lightsail, and/or"
echo "(b) sudo systemctl edit wildframe-web to set --workers 1 (halves web RAM)."
