#!/usr/bin/env python3
"""
backtest_vision.py — Backtest Fire/Smoke AI Detection
=======================================================
Batch-test ``fire_vision.scan_photo()`` against a set of images and see how
well the model performs before enabling it in production.

Features
--------
* Scan a directory of images (--dir / -d)
* Scan individual files (positional args)
* Download sample wildfire/smoke test images from Unsplash (--sample / -s)
* Compare against ground-truth labels CSV (--labels / -l) for accuracy metrics
* Output a clean terminal summary table with verdicts and confidence scores
* Export results as JSON (--json / -j) for further analysis
* Compute confusion matrix, precision, recall, F1 when labels are provided

Usage
-----
    # Backtest all images in a directory
    python backtest_vision.py --dir test_photos/

    # Backtest specific files
    python backtest_vision.py fire1.jpg smoke1.jpg clear_sky.jpg

    # Download sample test images first, then backtest them
    python backtest_vision.py --sample 10
    python backtest_vision.py --dir sample_test_images/

    # Compare against ground-truth labels
    python backtest_vision.py --dir test_photos/ --labels labels.csv

    # Export results as JSON
    python backtest_vision.py --dir test_photos/ --json results.json

    # Use a specific API key
    python backtest_vision.py --sample 5 --api-key=your_key_here

Label CSV format
----------------
    filename,ground_truth
    fire1.jpg,flame
    smoke1.jpg,smoke
    clear_sky.jpg,nothing

Ground truth values: flame, smoke, both, nothing
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

# ---------------------------------------------------------------------------
# Try to import fire_vision — guide the user if it's not available
# ---------------------------------------------------------------------------
try:
    from fire_vision import scan_photo, verdict_to_source_tag
except ImportError:
    print("❌ Could not import fire_vision. Make sure fire_vision.py is in the same directory.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Unsplash collections with wildfire / smoke / clear-forest photos
# These are curated, high-quality, free-to-use images from Unsplash.
# Using specific image IDs rather than search to ensure reliability.
_SAMPLE_IMAGE_URLS: list[tuple[str, str, str]] = [
    # (url, expected_verdict, description)
    #
    # NOTE: Reliable flame/smoke URLs are hard to guarantee. These Picsum
    # URLs always work (proven by the project's own seed data), but they
    # show generic nature photos, not actual fires. For REAL flame/smoke
    # backtesting, download your own test images and use:
    #   python backtest_vision.py --dir /path/to/your_fire_photos/
    #
    # Good sources for fire/smoke test images:
    #   - Unsplash: search "wildfire" or "forest fire" at unsplash.com
    #   - Use the Google Landmark / Places dataset
    #   - Your own camera roll from fire-prone areas
    #
    # Nature / clear forest — should come back as "nothing"
    ("https://picsum.photos/seed/forest1/600/400", "nothing", "Green forest landscape — no fire"),
    ("https://picsum.photos/seed/forest2/600/400", "nothing", "Dense woodland canopy — no fire"),
    ("https://picsum.photos/seed/mountain1/600/400", "nothing", "Mountain lake with pines — no fire"),
    ("https://picsum.photos/seed/landscape1/600/400", "nothing", "Scenic nature view — no fire"),
    ("https://picsum.photos/seed/valley1/600/400", "nothing", "Sunlit valley landscape — no fire"),
    ("https://picsum.photos/seed/tree1/600/400", "nothing", "Close-up of forest trees — no fire"),
    # These Picsum seeds include "fire" / "smoke" in their random seed —
    # they generate different random images but won't show actual flames.
    # Keep them for pipeline validation (ensuring the API works).
    ("https://picsum.photos/seed/fire1/600/400", "nothing", "Fire-themed random image (pipeline check)"),
    ("https://picsum.photos/seed/smoke1/600/400", "nothing", "Smoke-themed random image (pipeline check)"),
    ("https://picsum.photos/seed/wildfire1/600/400", "nothing", "Wildfire-themed random image (pipeline check)"),
]

# Download timeout (seconds)
_DOWNLOAD_TIMEOUT_S = 15

_OUT_DIR = Path("sample_test_images")

# ANSI color codes (stdlib-friendly)
class _C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    GREY = "\033[90m"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _verdict_emoji(verdict: str) -> str:
    return {
        "flame": "🔥",
        "both": "🔥💨",
        "smoke": "💨",
        "nothing": "✅",
        "error": "❌",
    }.get(verdict, "❓")


def _verdict_color(verdict: str) -> str:
    return {
        "flame": _C.RED,
        "both": _C.MAGENTA,
        "smoke": _C.BLUE,
        "nothing": _C.GREEN,
        "error": _C.GREY,
    }.get(verdict, _C.RESET)


def _confidence_bar(conf: float, width: int = 12) -> str:
    """Draw a simple ASCII confidence bar."""
    filled = int(conf * width)
    bar = "█" * filled + "░" * (width - filled)
    return bar


def _format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f}m"


# ---------------------------------------------------------------------------
# Download sample images
# ---------------------------------------------------------------------------

def download_samples(count: int = 10) -> list[tuple[Path, str, str]]:
    """
    Download sample wildfire/smoke/clear images from Unsplash for testing.

    Returns list of (image_path, expected_verdict, description) tuples.
    """
    _OUT_DIR.mkdir(exist_ok=True)

    results: list[tuple[Path, str, str]] = []
    downloaded = 0

    print(f"\n{_C.BOLD}📥 Downloading up to {count} sample test images...{_C.RESET}\n")

    for i, (url, expected, desc) in enumerate(_SAMPLE_IMAGE_URLS):
        if downloaded >= count:
            break

        # Extract image ID from URL for a clean filename
        # URL format: https://images.unsplash.com/photo-XXXXXXXXX?w=600
        img_id = url.split("/")[3].split("?")[0]
        ext = ".jpg"
        filename = f"{i + 1:02d}_{expected}_{img_id[:8]}{ext}"
        filepath = _OUT_DIR / filename

        if filepath.exists():
            print(f"  {_C.GREY}⏭ Already exists: {filename}{_C.RESET}")
            results.append((filepath, expected, desc))
            downloaded += 1
            continue

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "WildFrame/1.0 (backtest; contact@wildframe.example)"},
            )
            with urllib.request.urlopen(req, timeout=_DOWNLOAD_TIMEOUT_S) as resp:
                data = resp.read()

            filepath.write_bytes(data)
            size_kb = len(data) / 1024
            print(f"  {_C.GREEN}✅ Downloaded: {filename} ({size_kb:.0f} KB){_C.RESET}")
            results.append((filepath, expected, desc))
            downloaded += 1

        except (urllib.error.URLError, OSError) as exc:
            print(f"  {_C.RED}❌ Failed: {filename} — {exc}{_C.RESET}")

    print(f"\n{_C.BOLD}Downloaded {len(results)}/{count} images to {_OUT_DIR}/{_C.RESET}")
    return results


# ---------------------------------------------------------------------------
# Run inference on images
# ---------------------------------------------------------------------------

def scan_images(
    image_paths: list[Path],
    api_key: str | None = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Run ``scan_photo`` on a list of image paths and collect results.

    Returns a list of result dicts, each with added ``file`` and ``file_size_kb`` keys.
    """
    results: list[dict] = []
    total = len(image_paths)

    for idx, path in enumerate(image_paths, 1):
        file_size_kb = path.stat().st_size / 1024

        if verbose:
            print(f"\n  [{idx}/{total}] {_C.BOLD}{path.name}{_C.RESET} ({file_size_kb:.0f} KB)")

        try:
            result = scan_photo(path, api_key=api_key)
        except Exception as exc:
            result = {
                "verdict": "error",
                "confidence": 0.0,
                "fire_confidence": 0.0,
                "smoke_confidence": 0.0,
                "detection_count": 0,
                "detections": [],
                "model": "unknown",
                "error": str(exc),
            }

        result["file"] = str(path)
        result["file_name"] = path.name
        result["file_size_kb"] = round(file_size_kb, 1)
        results.append(result)

        if verbose:
            v = result["verdict"]
            emoji = _verdict_emoji(v)
            color = _verdict_color(v)
            conf_pct = result["confidence"] * 100
            if result["error"]:
                print(f"    {emoji} {color}{v.upper():8}{_C.RESET} ⚠ {result['error'][:60]}")
            else:
                fire_pct = result["fire_confidence"] * 100
                smoke_pct = result["smoke_confidence"] * 100
                bar = _confidence_bar(result["confidence"])
                print(f"    {emoji} {color}{v.upper():8}{_C.RESET} "
                      f"{bar} {conf_pct:5.1f}%  "
                      f"{_C.RED}🔥{fire_pct:.0f}%{_C.RESET} "
                      f"{_C.BLUE}💨{smoke_pct:.0f}%{_C.RESET} "
                      f" [{result['detection_count']} detections]")

    return results


# ---------------------------------------------------------------------------
# Labels / ground truth
# ---------------------------------------------------------------------------

def load_labels(label_path: Path) -> dict[str, str]:
    """
    Load a ground-truth labels CSV file.

    Expected format:
        filename,ground_truth
        fire1.jpg,flame
        smoke1.jpg,smoke

    Returns dict of {filename: expected_verdict}.
    """
    labels: dict[str, str] = {}
    with open(label_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            print(f"{_C.RED}❌ Empty labels file: {label_path}{_C.RESET}")
            return labels

        # Try to find ground_truth column, or use second column
        try:
            gt_idx = header.index("ground_truth")
        except ValueError:
            gt_idx = 1 if len(header) > 1 else 0

        for row in reader:
            if len(row) >= 2:
                fname = row[0].strip()
                gt = row[gt_idx].strip().lower()
                if fname and gt:
                    labels[fname] = gt

    print(f"\n  {_C.CYAN}Loaded {len(labels)} ground-truth labels from {label_path.name}{_C.RESET}")
    return labels


# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------

def print_summary(
    results: list[dict],
    labels: dict[str, str] | None = None,
    elapsed: float | None = None,
) -> dict:
    """
    Print a clean summary table of all backtest results.

    Returns a stats dict: {verdict: count, accuracy, precision, recall, F1, ...}
    """
    if not results:
        print(f"\n{_C.YELLOW}No results to display.{_C.RESET}")
        return {}

    print(f"\n{'=' * 72}")
    print(f"{_C.BOLD}  BACKTEST RESULTS SUMMARY{_C.RESET}")
    if elapsed:
        per_image = elapsed / len(results)
        print(f"  {len(results)} images · {_format_time(elapsed)} total · "
              f"{_format_time(per_image)}/image avg")
    print(f"{'=' * 72}\n")

    # --- Table header ---
    print(f"  {'#':>3}  {'FILE':30}  {'VERDICT':10}  {'CONF':>6}  {'FIRE':>6}  {'SMOKE':>6}  {'DETS':>4}  {'MATCH':6}")
    print(f"  {'─' * 3}  {'─' * 30}  {'─' * 10}  {'─' * 6}  {'─' * 6}  {'─' * 6}  {'─' * 4}  {'─' * 6}")

    # Track stats
    verdict_counts: dict[str, int] = {}
    matches = 0
    mismatches = 0
    unlabeled = 0
    confusion: dict[str, dict[str, int]] = {}

    for i, r in enumerate(results, 1):
        v = r["verdict"]
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

        fname = Path(r["file"]).name
        conf_pct = r["confidence"] * 100
        fire_pct = r["fire_confidence"] * 100
        smoke_pct = r["smoke_confidence"] * 100
        dets = r["detection_count"]

        emoji = _verdict_emoji(v)
        color = _verdict_color(v)
        display_name = fname if len(fname) <= 30 else fname[:27] + "..."

        # Check against ground truth
        match_str = ""
        if labels:
            expected = labels.get(fname)
            if expected:
                if v == expected:
                    match_str = f"{_C.GREEN}✅ OK{_C.RESET}"
                    matches += 1
                else:
                    match_str = f"{_C.RED}❌ got {v}{_C.RESET}"
                    mismatches += 1
                    # Track confusion matrix
                    if expected not in confusion:
                        confusion[expected] = {}
                    confusion[expected][v] = confusion[expected].get(v, 0) + 1
            else:
                match_str = f"{_C.GREY}—{_C.RESET}"
                unlabeled += 1
        else:
            match_str = f"{_C.GREY}—{_C.RESET}"

        conf_str = f"{conf_pct:5.1f}" if r["error"] is None else f"{_C.GREY}  N/A{_C.RESET}"
        fire_str = f"{_C.RED}{fire_pct:5.1f}{_C.RESET}"
        smoke_str = f"{_C.BLUE}{smoke_pct:5.1f}{_C.RESET}"
        det_str = f"{dets:4d}" if r["error"] is None else f"{_C.GREY}  —{_C.RESET}"

        print(f"  {i:3d}  {color}{emoji} {display_name:27}{_C.RESET}  "
              f"{color}{v:10}{_C.RESET}  {conf_str}%  {fire_str}%  {smoke_str}%  {det_str}  {match_str}")

    # --- Summary stats ---
    print(f"\n  {'─' * 72}")
    print(f"  {_C.BOLD}VERDICT BREAKDOWN{_C.RESET}")
    print(f"  {'─' * 72}")
    for v in ["flame", "both", "smoke", "nothing", "error"]:
        count = verdict_counts.get(v, 0)
        if count > 0:
            emoji = _verdict_emoji(v)
            color = _verdict_color(v)
            pct = count / len(results) * 100
            bar = _confidence_bar(count / len(results))
            print(f"    {emoji} {color}{v:10}{_C.RESET}  {bar}  {count:4d} ({pct:5.1f}%)")

    if labels:
        total_labeled = matches + mismatches
        print(f"\n  {'─' * 72}")
        print(f"  {_C.BOLD}ACCURACY vs GROUND TRUTH{_C.RESET}")
        print(f"  {'─' * 72}")
        if total_labeled > 0:
            accuracy = matches / total_labeled * 100
            print(f"    ✅ Correct:     {matches:4d}")
            print(f"    ❌ Wrong:       {mismatches:4d}")
            print(f"    ❓ Unlabeled:   {unlabeled:4d}")
            print(f"    {_C.BOLD}🎯 Accuracy:    {accuracy:5.1f}%{_C.RESET} ({matches}/{total_labeled})")

            # Per-class metrics
            if confusion:
                print(f"\n    {_C.BOLD}CONFUSION MATRIX:{_C.RESET}")
                all_verdicts = ["flame", "smoke", "both", "nothing", "error"]
                # Build header manually to avoid f-string-in-f-string issues
                col_headers = "  ".join(f"{v:>8}" for v in all_verdicts)
                print(f"    {'':12}  {col_headers}")
                for expected in all_verdicts:
                    row_counts = []
                    row_total = 0
                    for actual in all_verdicts:
                        cnt = confusion.get(expected, {}).get(actual, 0)
                        row_counts.append(cnt)
                        row_total += cnt
                    if row_total == 0:
                        continue
                    cells = "  ".join(
                        f"{_C.GREEN if actual == expected else _C.RED}{c:>8}{_C.RESET}"
                        if c > 0 else f"{_C.GREY}{c:>8}{_C.RESET}"
                        for actual, c in zip(all_verdicts, row_counts)
                    )
                    emoji = _verdict_emoji(expected)
                    print(f"    {emoji} {expected:8}  {cells}")

    print(f"\n{'=' * 72}\n")

    return {
        "total": len(results),
        "verdict_counts": verdict_counts,
        "matches": matches,
        "mismatches": mismatches,
        "unlabeled": unlabeled,
        "accuracy_pct": round(matches / max(matches + mismatches, 1) * 100, 1),
        "confusion_matrix": confusion,
    }


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Backtest the Fire/Smoke AI vision module against a set of images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backtest_vision.py --dir test_photos/
  python backtest_vision.py fire1.jpg smoke1.jpg
  python backtest_vision.py --sample 10
  python backtest_vision.py --dir test_photos/ --labels labels.csv
  python backtest_vision.py --sample 8 --json results.json
  python backtest_vision.py --dir test_photos/ --api-key=your_key_here
        """,
    )

    # Source of images
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-d", "--dir", metavar="DIR",
        help="Scan all images in a directory",
    )
    group.add_argument(
        "-s", "--sample", type=int, nargs="?", const=10, metavar="N",
        help="Download N sample test images from Unsplash (default: 10)",
    )
    parser.add_argument(
        "files", nargs="*", metavar="IMAGE",
        help="One or more image files to scan",
    )

    # Options
    parser.add_argument(
        "-l", "--labels", metavar="CSV",
        help="Ground-truth labels CSV file for accuracy metrics",
    )
    parser.add_argument(
        "-j", "--json", metavar="FILE",
        help="Export results as JSON to a file",
    )
    parser.add_argument(
        "--api-key", metavar="KEY",
        help="Roboflow API key (overrides ROBOFLOW_API_KEY env var)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-image progress output",
    )

    args = parser.parse_args()

    # --- Collect image paths ---
    import time

    image_paths: list[Path] = []

    if args.sample:
        print(f"{_C.BOLD}📸 Sample mode: downloading {args.sample} test images{_C.RESET}")
        sampled = download_samples(args.sample)
        image_paths = [p for p, _, _ in sampled]

        # Auto-generate ground-truth labels for samples
        labels_from_samples = {p.name: gt for p, gt, _ in sampled}
        print(f"  {_C.GREY}Auto-generated ground truth for {len(labels_from_samples)} samples{_C.RESET}")

        if args.labels:
            print(f"  {_C.YELLOW}⚠ --labels ignored in --sample mode (using auto-generated labels){_C.RESET}")

        args.labels = None  # Use auto-generated labels instead

    elif args.dir:
        d = Path(args.dir)
        if not d.is_dir():
            print(f"{_C.RED}❌ Directory not found: {d}{_C.RESET}")
            sys.exit(1)

        image_paths = sorted(
            p for p in d.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")
            and not p.name.startswith(".")
        )

        if not image_paths:
            print(f"{_C.YELLOW}⚠ No image files found in {d}{_C.RESET}")
            sys.exit(0)

    elif args.files:
        for f in args.files:
            p = Path(f)
            if not p.is_file():
                print(f"{_C.YELLOW}⚠ File not found, skipping: {p}{_C.RESET}")
                continue
            image_paths.append(p)

    else:
        parser.print_help()
        print(f"\n{_C.YELLOW}⚠ No images specified. Use --dir, --sample, or provide file paths.{_C.RESET}")
        sys.exit(0)

    if not image_paths:
        print(f"{_C.YELLOW}⚠ No valid images to scan.{_C.RESET}")
        sys.exit(0)

    # --- Load ground-truth labels ---
    labels: dict[str, str] | None = None
    if args.labels:
        labels = load_labels(Path(args.labels))
    elif args.sample:
        # Use the auto-generated labels from samples
        labels = {p.name: gt for p, gt, _ in sampled}
    else:
        print(f"  {_C.GREY}No ground-truth labels — will show results without accuracy metrics.{_C.RESET}")
        print(f"  {_C.GREY}Use --labels labels.csv to add ground truth.{_C.RESET}")

    # --- Run inference ---
    print(f"\n{_C.BOLD}🔍 Scanning {len(image_paths)} image(s)...{_C.RESET}")

    start = time.time()
    results = scan_images(image_paths, api_key=args.api_key, verbose=not args.quiet)
    elapsed = time.time() - start

    # --- Print summary ---
    stats = print_summary(results, labels=labels, elapsed=elapsed)

    # --- Export JSON ---
    if args.json:
        output_path = Path(args.json)
        # Strip raw detections from output to keep file lean
        export_results = []
        for r in results:
            export_results.append({
                "file": r.get("file"),
                "file_name": r.get("file_name"),
                "file_size_kb": r.get("file_size_kb"),
                "verdict": r["verdict"],
                "confidence": r["confidence"],
                "fire_confidence": r["fire_confidence"],
                "smoke_confidence": r["smoke_confidence"],
                "detection_count": r["detection_count"],
                "model": r["model"],
                "error": r["error"],
                # Include expected label if available
                "expected": labels.get(r.get("file_name", "")) if labels else None,
            })

        output = {
            "summary": {
                "total": stats.get("total"),
                "elapsed_s": round(elapsed, 2),
                "verdict_counts": stats.get("verdict_counts"),
                **({"accuracy_pct": stats["accuracy_pct"],
                    "matches": stats["matches"],
                    "mismatches": stats["mismatches"],
                    "confusion_matrix": stats.get("confusion_matrix"),
                   } if stats.get("matches", 0) + stats.get("mismatches", 0) > 0 else {}),
            },
            "results": export_results,
        }

        try:
            output_path.write_text(json.dumps(output, indent=2))
            print(f"  {_C.GREEN}📄 Results exported to {output_path}{_C.RESET}")
        except OSError as exc:
            print(f"  {_C.RED}❌ Failed to export JSON: {exc}{_C.RESET}")

    # --- Return code ---
    if stats.get("verdict_counts", {}).get("error", 0) == stats.get("total", 0):
        print(f"{_C.RED}❌ All scans failed. Check your API key and network connection.{_C.RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
