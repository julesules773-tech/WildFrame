#!/usr/bin/env python3
"""
fire_vision.py — AI-Powered Fire & Smoke Detection
===================================================
Detects fire and/or smoke in uploaded photographs using a Roboflow workflow.

**Primary path**: Runs the ``fire-and-smoke-segmentation-alerts`` workflow
via a local Roboflow inference server (http://127.0.0.1:9001) using the
``inference_sdk`` package.

**Fallback path**: If the local server is unreachable, runs the same workflow
via the Roboflow hosted inference API at ``serverless.roboflow.com`` using
stdlib ``urllib`` (JSON body with a base64 image).

Usage
-----
    from fire_vision import scan_photo

    result = scan_photo("uploads/abc123.jpg")
    # → {"verdict": "flame", "confidence": 0.87, ...}

Prerequisites
-------------
1. Install:  pip install inference-sdk
2. Set env var:  export ROBOFLOW_API_KEY=your_key_here
3. (Optional) Run local inference server for faster scans:
       docker run --rm -p 9001:9001 roboflow/roboflow-inference-server-cpu
       # or GPU variant: roboflow/roboflow-inference-server-gpu
4. Optional: Pillow (pillow). Only used to downscale oversized photos
   before sending them to the hosted API, so they don't exceed Roboflow's
   request-body limit (HTTP 413). Without it, large images are sent as-is
   and may be rejected by the hosted API.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

log = logging.getLogger("fire-vision")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Roboflow workspace & workflow (from testpy.pyi reference)
_WORKSPACE_NAME = "juless-workspace-zidwe"
_WORKFLOW_ID = "fire-and-smoke-segmentation-alerts-1784557902550"

# Local inference server (primary — fast, no rate limits)
_LOCAL_INFERENCE_URL = "http://127.0.0.1:9001"

# Hosted fallback — the unified hosted inference API (outline.roboflow.com
# and the old detect/classify/segment hosts are deprecated). Hosted
# workflows run at /infer/workflows/<workspace>/<workflow_id>.
_API_HOST = "https://serverless.roboflow.com"
_USER_AGENT = "WildFrame/1.0 (fire-vision; contact@wildframe.example)"
# Hosted inference can be slow on large photos (we measured read/write
# timeouts right at the 30s mark), so allow up to 60s before giving up.
_TIMEOUT_S = 60

# Cap the long edge of photos sent to the hosted API. The request body is
# base64-encoded (~33% inflation) and Roboflow's gateway rejects oversized
# bodies with HTTP 413 (measured: a 13 MB PNG -> ~17 MB base64 -> 413).
#
# 2560px @ q92 is a hard-won balance, measured with a real 9796x3846
# panorama that contains a small, distant fire:
#   - 1280px @ q88: fire never detected (0.0) — the tiny fire is crushed
#   - 2048px @ q88: cliff edge — detected in one run, missed in another
#   - 2048px @ q92 and up: fire ~0.62-0.65 in the sessions where the API
#     was in a detecting state (see nondeterminism note below)
#   - worst real images at 2560px @ q92: <= ~4 MB base64 (39% of the limit)
#
# IMPORTANT — the hosted API is NONDETERMINISTIC across time: the exact
# same byte-identical JPEG scored fire=0.618/0.652 in one session and 0.000
# eight scans in a row in a later session. Within a session it is
# deterministic (same bytes, same result), so the variance appears to come
# from Roboflow routing to different backends/versions over time, not from
# request noise. Downscaling is therefore necessary-but-not-sufficient: a
# borderline image (small fire in a big frame) may scan as "nothing" on one
# day and "flame" the next, and a re-scan can return a different verdict.
#
# A small fire in a big frame needs enough pixels to survive both our
# downscale and the model's own resize — so we keep the cap generous and
# the JPEG quality high rather than chasing the smallest payload.
_MAX_HOSTED_EDGE_PX = 2560
_JPEG_QUALITY = 92

# Minimum confidence to accept a prediction as valid
_CONFIDENCE_THRESHOLD = 0.10

# Class names for fire vs. smoke
_FIRE_CLASSES = {"fire", "flame", "fire-flame"}
_SMOKE_CLASSES = {"smoke"}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def scan_photo(
    image_path: str | Path,
    api_key: str | None = None,
    confidence_threshold: float = _CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """
    Run fire/smoke detection on a single photograph.

    Tries the local Roboflow inference server first; falls back to the
    hosted REST API if the local server isn't running.

    Parameters
    ----------
    image_path : str | Path
        Path to the image file on disk.
    api_key : str, optional
        Roboflow API key. Falls back to ``ROBOFLOW_API_KEY`` env var.
    confidence_threshold : float, optional
        Minimum prediction confidence (0–1). Default 0.10.

    Returns
    -------
    dict with keys:
        verdict : str
            One of ``"flame"``, ``"smoke"``, ``"both"``, ``"nothing"``,
            or ``"error"``.
        confidence : float
            Highest single-detection confidence across all detected objects.
        fire_confidence : float
            Highest confidence among fire-class detections.
        smoke_confidence : float
            Highest confidence among smoke-class detections.
        detection_count : int
            Number of predictions passing the confidence threshold.
        detections : list[dict]
            Raw prediction dicts.
        model : str
            Model / workflow that was queried.
        error : str | None
            Human-readable error if the scan failed.
    """
    # --- Resolve API key ---
    resolved_key = api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not resolved_key:
        return _error_dict("ROBOFLOW_API_KEY not set. Set this env var or pass api_key=.")

    # --- Validate image file ---
    path = Path(image_path)
    if not path.is_file():
        return _error_dict(f"Image not found: {path}")

    # --- Try primary path: local workflow ---
    try:
        return _scan_via_local_workflow(path, resolved_key, confidence_threshold)
    except Exception as exc:
        log.info("Local workflow failed, falling back to hosted API: %s", exc)

    # --- Fallback: hosted REST API ---
    try:
        return _scan_via_hosted_api(path, resolved_key, confidence_threshold)
    except Exception as exc:
        log.warning("Hosted API also failed: %s", exc)
        return _error_dict(f"All inference paths failed: {exc}")


def verdict_to_source_tag(verdict: str) -> str:
    """Map a verdict to a Bayesian-evidence source tag."""
    return {
        "flame": "photo-flame",
        "both": "photo-flame",
        "smoke": "photo-smoke",
        "nothing": "photo-clear",
        "error": "photo-error",
    }.get(verdict, "photo-unknown")


def verdict_to_likelihood_ratio(verdict: str, confidence: float = 1.0) -> float | None:
    """Map a verdict to a Bayes likelihood ratio, scaled by confidence."""
    base_lr = {
        "flame": 10.0,
        "both": 10.0,
        "smoke": 3.0,
        "nothing": 0.5,
    }.get(verdict)
    if base_lr is None:
        return None
    return 1.0 + (base_lr - 1.0) * max(confidence, 0.0)


# ---------------------------------------------------------------------------
# Primary path — local Roboflow inference server (workflow)
# ---------------------------------------------------------------------------


def _scan_via_local_workflow(
    path: Path,
    api_key: str,
    confidence_threshold: float,
) -> dict[str, Any]:
    """Run the fire/smoke segmentation workflow via the local inference server."""
    try:
        from inference_sdk import InferenceHTTPClient
    except ImportError:
        raise RuntimeError(
            "inference_sdk not installed. Run: pip install inference-sdk"
        )

    client = InferenceHTTPClient(
        api_url=_LOCAL_INFERENCE_URL,
        api_key=api_key,
    )

    t0 = time.time()
    result = client.run_workflow(
        workspace_name=_WORKSPACE_NAME,
        workflow_id=_WORKFLOW_ID,
        images={
            "image": str(path),
        },
    )
    elapsed = time.time() - t0
    log.info("Local workflow returned in %.1fs", elapsed)

    predictions = _extract_predictions_from_workflow(result)
    return _build_result(predictions, confidence_threshold, "workflow:" + _WORKFLOW_ID)


def _extract_predictions_from_workflow(
    workflow_result: list[dict] | dict,
) -> list[dict]:
    """
    Extract prediction dicts from a Roboflow workflow result.

    Workflows return a list of output dicts (one per input image) wrapped
    in various shapes depending on where they run (local server vs hosted
    API). We recursively scan the structure for any list of prediction
    dicts (dicts with ``class`` and ``confidence`` keys), which is the
    shape every workflow output ultimately takes. Handles:

      1. ``[{"predictions": [...]}]`` — bare per-image list
      2. ``{"predictions": {"predictions": [...], "width": ..., "height": ...}}``
         (dict wrapper — Roboflow inference API format)
      3. ``{"outputs": [{"predictions": [...]}]}`` — hosted API envelope
    """
    # ----- Helper: check if a value is a prediction-like list -----
    def _looks_like_predictions(val: object) -> list[dict] | None:
        """If *val* is a non-empty list of prediction dicts, return it; else None."""
        if not isinstance(val, list) or len(val) == 0:
            return None
        first = val[0]
        if isinstance(first, dict) and ("class" in first or "class_name" in first) and "confidence" in first:
            return val
        return None

    def _scan_dict(d: dict) -> list[dict] | None:
        """Depth-first search for a prediction-like list anywhere in *d*."""
        for value in d.values():
            found = _looks_like_predictions(value)
            if found is not None:
                return found
            if isinstance(value, dict):
                found = _scan_dict(value)
                if found is not None:
                    return found
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        found = _scan_dict(item)
                        if found is not None:
                            return found
        return None

    if isinstance(workflow_result, list):
        for item in workflow_result:
            if isinstance(item, dict):
                found = _scan_dict(item)
                if found is not None:
                    return found
        # Bare nested list: only trust it if it is actually prediction-like.
        if workflow_result and isinstance(workflow_result[0], list):
            found = _looks_like_predictions(workflow_result[0])
            if found is not None:
                return found
        return []

    if isinstance(workflow_result, dict):
        found = _scan_dict(workflow_result)
        if found is not None:
            return found

    return []


# ---------------------------------------------------------------------------
# Fallback path — hosted REST API (urllib)
# ---------------------------------------------------------------------------


def _prepare_hosted_image_bytes(path: Path) -> bytes:
    """Return the image bytes to send to the hosted API.

    Small images (long edge <= ``_MAX_HOSTED_EDGE_PX``) are sent as-is with
    no re-encode, preserving their original format and quality. Larger ones
    are downscaled to fit the cap and re-encoded as JPEG (EXIF orientation
    applied first) so the base64 request body stays well under Roboflow's
    gateway limit — the model resizes to its native input size anyway, so
    this costs no detection accuracy.

    Falls back to the raw file bytes if PIL is unavailable (keeps the module
    importable without Pillow, matching the lazy-import pattern of the local
    workflow path) or can't decode the image (e.g. a corrupt file), letting
    the API's own error handling report it.
    """
    try:
        # Lazy import: Pillow is only needed for downscaling, and this module
        # otherwise stays stdlib-only so minimal environments (e.g. a sweep
        # in a bare venv) can still import it without crashing.
        from PIL import Image, ImageOps
    except ImportError:
        return path.read_bytes()
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            if max(img.size) <= _MAX_HOSTED_EDGE_PX:
                return path.read_bytes()
            scale = _MAX_HOSTED_EDGE_PX / max(img.size)
            new_size = (max(1, round(img.width * scale)),
                        max(1, round(img.height * scale)))
            # LANCZOS keeps edges sharp; resize after exif_transpose so the
            # capped dimension is the *oriented* long edge.
            img = img.resize(new_size, Image.LANCZOS)
            if img.mode != "RGB":
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
            return buf.getvalue()
    except Exception:
        # Unreadable/corrupt image — send raw bytes and let the API report it.
        return path.read_bytes()


def _scan_via_hosted_api(
    path: Path,
    api_key: str,
    confidence_threshold: float,
) -> dict[str, Any]:
    """
    Run the fire/smoke workflow via the Roboflow hosted inference API.

    The old per-task hosts (outline/detect/classify/segment.roboflow.com)
    are deprecated; the unified hosted API lives at serverless.roboflow.com
    and takes a JSON body with the API key and a base64 image input. Images
    are downscaled to ``_MAX_HOSTED_EDGE_PX`` first so oversized photos can't
    trigger HTTP 413 on the base64 request body.
    """
    image_bytes = _prepare_hosted_image_bytes(path)
    b64_data = base64.b64encode(image_bytes).decode("ascii")

    url = f"{_API_HOST}/infer/workflows/{_WORKSPACE_NAME}/{_WORKFLOW_ID}"
    payload = json.dumps(
        {
            "api_key": api_key,
            "inputs": {"image": {"type": "base64", "value": b64_data}},
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": _USER_AGENT,
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        # Never let the API key end up in logs or the stored error message.
        body = body.replace(api_key, "[REDACTED]")
        log.warning("Hosted API HTTP %d: %s", exc.code, body[:300])
        raise RuntimeError(f"Hosted API HTTP {exc.code}: {body[:200]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Hosted API connection error: {exc.reason}") from exc

    data = json.loads(raw)
    predictions = _extract_predictions_from_workflow(data)
    return _build_result(predictions, confidence_threshold, f"workflow:{_WORKFLOW_ID}@hosted")


# ---------------------------------------------------------------------------
# Result builder — shared between local & hosted paths
# ---------------------------------------------------------------------------


def _build_result(
    predictions: list[dict],
    confidence_threshold: float,
    model_label: str,
) -> dict[str, Any]:
    """
    Convert a raw prediction list into the uniform result dict.
    """
    detections = [
        p for p in predictions
        if p.get("confidence", 0) >= confidence_threshold
    ]

    fire_conf = 0.0
    smoke_conf = 0.0
    for d in detections:
        cls_name = (d.get("class") or "").lower().strip()
        conf = d.get("confidence", 0)
        if cls_name in _FIRE_CLASSES:
            fire_conf = max(fire_conf, conf)
        elif cls_name in _SMOKE_CLASSES:
            smoke_conf = max(smoke_conf, conf)

    if fire_conf >= confidence_threshold and smoke_conf >= confidence_threshold:
        verdict = "both"
    elif fire_conf >= confidence_threshold:
        verdict = "flame"
    elif smoke_conf >= confidence_threshold:
        verdict = "smoke"
    else:
        verdict = "nothing"

    overall_conf = max(fire_conf, smoke_conf)

    return {
        "verdict": verdict,
        "confidence": round(overall_conf, 4),
        "fire_confidence": round(fire_conf, 4),
        "smoke_confidence": round(smoke_conf, 4),
        "detection_count": len(detections),
        "detections": detections,
        "model": model_label,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _error_dict(msg: str) -> dict[str, Any]:
    """Return a uniform error result dict."""
    return {
        "verdict": "error",
        "confidence": 0.0,
        "fire_confidence": 0.0,
        "smoke_confidence": 0.0,
        "detection_count": 0,
        "detections": [],
        "model": f"workflow:{_WORKFLOW_ID}",
        "error": msg,
    }