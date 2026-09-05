#!/usr/bin/env python3
"""
test_blocked_ips.py — Verify the persistent IP blocklist (blocked_ips table)
survives simulated server restarts.

Because the AbuseTracker now delegates entirely to Postgres, "restarting"
the server is equivalent to calling the db functions from a fresh process.
There is no in-memory state to lose — that IS the thing under test.

Requires a running Postgres with the WildFrame schema (run server.py once
to create tables).  Uses a dedicated test prefix to avoid colliding with
real blocked IPs.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import db

# ---------------------------------------------------------------------------
# Config — all test IPs use a prefix so we never touch real data
# ---------------------------------------------------------------------------
PREFIX = "__test_blocklist__"
TEST_IP = f"{PREFIX}_10.0.0.1"
TEST_IP_2 = f"{PREFIX}_10.0.0.2"
# Aggressive thresholds so the test runs fast
THRESHOLD = 3        # 3 nothing-verdicts to trigger a block
WINDOW_S = 3600      # 1 hour sliding window
BLOCK_S = 120        # 2-minute block (short for testing)

PASS = 0
FAIL = 0
SKIP = 0


def report(status: str, label: str, detail: str = ""):
    global PASS, FAIL, SKIP
    icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⏭️" if status == "SKIP" else "⚠️"
    if status == "PASS":
        PASS += 1
    elif status == "FAIL":
        FAIL += 1
    elif status == "SKIP":
        SKIP += 1
    print(f"  {icon} {label}" + (f"  ({detail})" if detail else ""))


def cleanup():
    """Remove all test rows so we start clean."""
    try:
        with db._conn() as conn:
            conn.execute(
                "DELETE FROM blocked_ips WHERE ip LIKE %s",
                (f"{PREFIX}%",),
            )
    except Exception:
        pass


def assert_is_blocked(ip: str, expect_blocked: bool, label: str):
    """Check whether an IP is blocked and report."""
    until = db.is_ip_blocked(ip)
    blocked = until is not None
    if blocked == expect_blocked:
        report("PASS", label)
    else:
        detail = f"expected {'blocked' if expect_blocked else 'not blocked'}, got "
        detail += f"blocked (until {until})" if blocked else "not blocked"
        report("FAIL", label, detail)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_basic_block():
    """Record enough nothing-verdicts to cross the threshold → IP blocked."""
    print("\n▸ test_basic_block")
    cleanup()

    for i in range(THRESHOLD - 1):
        result = db.record_nothing_hit(TEST_IP, THRESHOLD, WINDOW_S, BLOCK_S)
        if result is not None:
            report("FAIL", f"hit {i+1} should not block yet", f"returned {result}")
            return
    report("PASS", f"first {THRESHOLD-1} hits did not block")

    # This hit should cross the threshold
    result = db.record_nothing_hit(TEST_IP, THRESHOLD, WINDOW_S, BLOCK_S)
    if result is not None and result > time.time():
        report("PASS", f"hit #{THRESHOLD} triggered block", f"expires in {result - time.time():.0f}s")
    else:
        report("FAIL", f"hit #{THRESHOLD} should have blocked", f"got {result}")

    assert_is_blocked(TEST_IP, True, "IP is now blocked")


def test_persistence_across_calls():
    """Simulate a 'restart' by calling is_blocked from a fresh call chain.

    With the old in-memory tracker, the state would be lost here.
    With Postgres, the block must still be active.
    """
    print("\n▸ test_persistence_across_calls")

    # 'Restart' simulation: no object state carries over — every call
    # goes through db.py → Postgres.
    assert_is_blocked(TEST_IP, True, "block persists after simulated restart (1)")
    assert_is_blocked(TEST_IP, True, "block persists after simulated restart (2)")

    # The remaining time should be <= BLOCK_S
    until = db.is_ip_blocked(TEST_IP)
    remaining = until - time.time()
    if 0 < remaining <= BLOCK_S:
        report("PASS", f"remaining time {remaining:.0f}s is within block duration", f"block_s={BLOCK_S}")
    else:
        report("FAIL", f"remaining time {remaining:.0f}s out of range", f"block_s={BLOCK_S}")


def test_already_blocked_does_not_reset():
    """Additional hits while blocked should NOT reset the block timer."""
    print("\n▸ test_already_blocked_does_not_reset")

    until_before = db.is_ip_blocked(TEST_IP)
    if until_before is None:
        report("SKIP", "IP not blocked — run test_basic_block first")
        return

    # Record another hit
    db.record_nothing_hit(TEST_IP, THRESHOLD, WINDOW_S, BLOCK_S)
    until_after = db.is_ip_blocked(TEST_IP)

    if until_after is not None and abs(until_after - until_before) < 2:
        report("PASS", "block timer not reset by hit during active block")
    else:
        report("FAIL", f"block timer changed unexpectedly",
               f"before={until_before}, after={until_after}")


def test_unblock():
    """Manually unblock an IP → is_blocked returns None."""
    print("\n▸ test_unblock")

    was_blocked = db.unblock_ip(TEST_IP)
    if was_blocked:
        report("PASS", "unblock_ip returned True")
    else:
        report("FAIL", "unblock_ip returned False (IP should have been blocked)")

    assert_is_blocked(TEST_IP, False, "IP is no longer blocked after unblock")

    # Verify the row is fully gone (not just expired)
    try:
        with db._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM blocked_ips WHERE ip = %s", (TEST_IP,)
            ).fetchone()
        if row is None:
            report("PASS", "row fully deleted from blocked_ips")
        else:
            report("FAIL", "row still exists after unblock")
    except Exception as exc:
        report("FAIL", f"could not check row: {exc}")


def test_unblock_nonexistent():
    """Unblocking an IP that was never blocked returns False."""
    print("\n▸ test_unblock_nonexistent")
    cleanup()

    was_blocked = db.unblock_ip(f"{PREFIX}_never_blocked")
    if not was_blocked:
        report("PASS", "unblock_ip returned False for unknown IP")
    else:
        report("FAIL", "unblock_ip returned True for unknown IP")


def test_multiple_ips():
    """Block two IPs independently; unblock one; the other stays blocked."""
    print("\n▸ test_multiple_ips")
    cleanup()

    # Block both
    for _ in range(THRESHOLD):
        db.record_nothing_hit(TEST_IP, THRESHOLD, WINDOW_S, BLOCK_S)
        db.record_nothing_hit(TEST_IP_2, THRESHOLD, WINDOW_S, BLOCK_S)

    assert_is_blocked(TEST_IP, True, "IP 1 blocked")
    assert_is_blocked(TEST_IP_2, True, "IP 2 blocked")

    # Unblock IP 1
    db.unblock_ip(TEST_IP)
    assert_is_blocked(TEST_IP, False, "IP 1 unblocked")
    assert_is_blocked(TEST_IP_2, True, "IP 2 still blocked")


def test_cleanup_expired():
    """Expired blocks are removed by cleanup_expired_blocked_ips."""
    print("\n▸ test_cleanup_expired")
    cleanup()

    # Insert a row with blocked_until in the past
    try:
        with db._conn() as conn:
            conn.execute(
                "INSERT INTO blocked_ips (ip, blocked_until, nothing_hits) "
                "VALUES (%s, now() - interval '1 hour', '[]'::jsonb)",
                (f"{PREFIX}_expired",),
            )
    except Exception as exc:
        report("SKIP", f"could not insert expired row: {exc}")
        return

    deleted = db.cleanup_expired_blocked_ips()
    if deleted >= 1:
        report("PASS", f"cleanup deleted {deleted} expired row(s)")
    else:
        report("FAIL", f"cleanup deleted {deleted} rows (expected >= 1)")

    assert_is_blocked(f"{PREFIX}_expired", False, "expired IP not blocked after cleanup")


def test_reblock_after_expiry():
    """After a block expires, the IP can be re-blocked from scratch."""
    print("\n▸ test_reblock_after_expiry")
    cleanup()

    # Insert an expired block
    with db._conn() as conn:
        conn.execute(
            "INSERT INTO blocked_ips (ip, blocked_until, nothing_hits) "
            "VALUES (%s, now() - interval '1 second', '[1.0]'::jsonb)",
            (TEST_IP,),
        )

    assert_is_blocked(TEST_IP, False, "expired block is not active")

    # Now record hits — should start fresh
    for i in range(THRESHOLD - 1):
        result = db.record_nothing_hit(TEST_IP, THRESHOLD, WINDOW_S, BLOCK_S)
        if result is not None:
            report("FAIL", f"hit {i+1} after expiry should not block yet")
            return

    result = db.record_nothing_hit(TEST_IP, THRESHOLD, WINDOW_S, BLOCK_S)
    if result is not None and result > time.time():
        report("PASS", "IP re-blocked after previous block expired")
    else:
        report("FAIL", "IP should be re-blocked after expiry")


def test_get_blocked_ips():
    """get_blocked_ips returns metadata for all active blocks."""
    print("\n▸ test_get_blocked_ips")
    cleanup()

    # Block TEST_IP
    for _ in range(THRESHOLD):
        db.record_nothing_hit(TEST_IP, THRESHOLD, WINDOW_S, BLOCK_S)

    blocked = db.get_blocked_ips()
    test_entry = [b for b in blocked if b["ip"] == TEST_IP]

    if len(test_entry) == 1:
        report("PASS", "get_blocked_ips includes our test IP")
        entry = test_entry[0]
        required_keys = {"ip", "blocked_until", "remaining_s", "recent_nothing_count"}
        missing = required_keys - set(entry.keys())
        if not missing:
            report("PASS", "entry has all required keys")
        else:
            report("FAIL", f"missing keys: {missing}")
    else:
        report("FAIL", f"expected 1 entry for {TEST_IP}, got {len(test_entry)}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("test_blocked_ips.py — IP blocklist persistence tests")
    print("=" * 60)

    try:
        with db._conn() as conn:
            conn.execute("SELECT 1")
    except Exception as exc:
        print(f"\n⏭️  Skipping all tests — cannot connect to Postgres: {exc}")
        sys.exit(0)

    test_basic_block()
    test_persistence_across_calls()
    test_already_blocked_does_not_reset()
    test_unblock()
    test_unblock_nonexistent()
    test_multiple_ips()
    test_cleanup_expired()
    test_reblock_after_expiry()
    test_get_blocked_ips()

    # Always clean up
    cleanup()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed, {SKIP} skipped")
    print("=" * 60)

    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
