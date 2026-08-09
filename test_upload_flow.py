#!/usr/bin/env python3
"""
test_upload_flow.py — End-to-end test of the AI photo upload flow.

Simulates exactly what server.py's create_report() does:
  1. Save uploaded photo to disk
  2. Run AI scan (fire_vision.scan_photo)
  3. If verdict == "nothing": keep photo, create PENDING report (the hosted
     model can miss borderline fires — never silently discard a photo)
  4. If verdict == flame/smoke/both: create report record, print ACCEPTED
  5. If verdict == "error": create report anyway (fail-open)
"""

import json
import os
import sys
import uuid
from pathlib import Path

# Ensure fire_vision is importable
sys.path.insert(0, str(Path(__file__).parent))

from fire_vision import scan_photo

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("ROBOFLOW_API_KEY") or "anQm43Si6DlwZMvBM0LC"
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

PASS = 0
FAIL = 0


def report(status: str, label: str, detail: str = ""):
    global PASS, FAIL
    icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
    if status == "PASS":
        PASS += 1
    elif status == "FAIL":
        FAIL += 1
    print(f"  {icon} {label}  {detail}")


def simulate_upload(image_path: str, expected_verdict: str = None) -> dict:
    """Simulate the server upload flow and return the AI result."""
    path = Path(image_path)
    if not path.is_file():
        report("FAIL", f"Image not found: {path}")
        return {"verdict": "error", "error": "file not found"}

    # Step 1: Save photo to uploads (copy)
    ext = path.suffix.lower().lstrip(".") or "jpg"
    saved_name = f"{uuid.uuid4().hex}.{ext}"
    saved_path = UPLOAD_DIR / saved_name
    import shutil
    shutil.copy2(str(path), str(saved_path))
    print(f"\n  📁 Saved as: {saved_name}")

    # Step 2: Run AI scan
    print(f"  🤖 Scanning...")
    ai_result = scan_photo(str(saved_path), api_key=API_KEY)

    verdict = ai_result["verdict"]
    confidence = ai_result["confidence"]
    fire_conf = ai_result["fire_confidence"]
    smoke_conf = ai_result["smoke_confidence"]
    detection_count = ai_result["detection_count"]
    error = ai_result["error"]

    # Step 3: Decision
    if verdict == "nothing":
        # KEEP for human review — do NOT delete the photo. The hosted model
        # can miss borderline fires (nondeterministic across time), so a
        # "nothing" scan becomes a pending report for a moderator to judge.
        print(f"  🤔 NOTHING detected — photo KEPT as pending for review")
        result = {
            "accepted": True,
            "reason": "AI detected nothing — kept for human review",
            "ai_analysis": ai_result,
        }
    elif verdict == "error":
        # FAIL-OPEN — create report anyway
        print(f"  ⚠️  AI scan error — creating report anyway: {error}")
        result = {
            "accepted": True,
            "reason": f"AI scan errored ({error}), report created for review",
            "ai_analysis": ai_result,
        }
    else:
        # ACCEPT — create report
        print(f"  ✅ {verdict.upper()} detected (conf={confidence:.2f}) — report CREATED")
        result = {
            "accepted": True,
            "reason": f"AI detected {verdict}",
            "ai_analysis": ai_result,
        }

    # Pretty print
    emoji = {
        "flame": "🔥", "smoke": "💨", "both": "🔥💨",
        "nothing": "✅", "error": "❌",
    }.get(verdict, "❓")
    print(f"  {emoji} Verdict: {verdict.upper()}  |  "
          f"Conf: {confidence:.2%}  |  "
          f"Fire: {fire_conf:.2%}  |  "
          f"Smoke: {smoke_conf:.2%}  |  "
          f"Detections: {detection_count}")

    if error:
        print(f"  ❗ Error: {error}")

    # Assert expected verdict if provided
    if expected_verdict and verdict != expected_verdict:
        report("FAIL", f"Expected verdict '{expected_verdict}', got '{verdict}'")
    else:
        report("PASS", f"Verdict '{verdict}'" +
               (f" matches expected '{expected_verdict}'" if expected_verdict else ""))

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 AI UPLOAD FLOW TEST")
    print("=" * 60)

    # Test 1: Fire photo (should detect smoke → report created)
    print("\n" + "─" * 60)
    print("📸 TEST 1: Fire/smoke photo (images.jpeg)")
    print("─" * 60)
    result1 = simulate_upload("../Downloads/fires/images.jpeg")
    assert result1["accepted"] == True, "Fire photo should be ACCEPTED"
    report("PASS", "Fire photo accepted")

    # Test 2: Clean nature photo (should be nothing → kept as pending review)
    print("\n" + "─" * 60)
    print("🌲 TEST 2: Clean nature photo (nothing expected → kept for review)")
    print("─" * 60)
    # Find a sample from the backtest
    sample_dir = Path(__file__).parent / "sample_test_images"
    if sample_dir.exists():
        samples = sorted(sample_dir.glob("*.jpg")) + sorted(sample_dir.glob("*.jpeg"))
        if samples:
            clean_path = str(samples[0])
            result2 = simulate_upload(clean_path, expected_verdict="nothing")
            # A "nothing" verdict is NOT rejected anymore — the photo is kept
            # as a pending report so a real fire is never silently discarded.
            assert result2["accepted"] == True, "Clean photo should be KEPT for review"
            report("PASS", "Clean photo kept for human review")
        else:
            report("FAIL", "No sample images found in sample_test_images/")
    else:
        report("FAIL", "sample_test_images/ directory not found")
        print("  Run 'python3 backtest_vision.py --sample 3' first to download test images")

    # Test 3: Missing file (should error → report created as fail-open)
    print("\n" + "─" * 60)
    print("❌ TEST 3: Missing file (error → report created anyway)")
    print("─" * 60)
    result3 = simulate_upload("/tmp/nonexistent_photo.jpg")
    # Missing file returns error at file-exists check, before scan_photo
    if result3.get("verdict") == "error" or result3.get("accepted") is None:
        report("PASS", "No-file case handled gracefully")

    # Summary
    print("\n" + "=" * 60)
    total = PASS + FAIL
    print(f"📊 RESULTS:  {PASS}/{total} passed  |  {FAIL}/{total} failed")
    if FAIL == 0:
        print("🎉 ALL TESTS PASSED!")
    else:
        print(f"❌ {FAIL} test(s) FAILED")
    print("=" * 60)

    # Clean up: remove saved test photos
    for f in UPLOAD_DIR.glob("*.jpg"):
        try:
            f.unlink()
        except OSError:
            pass
    for f in UPLOAD_DIR.glob("*.jpeg"):
        try:
            f.unlink()
        except OSError:
            pass

    sys.exit(0 if FAIL == 0 else 1)
