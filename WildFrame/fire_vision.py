#!/usr/bin/env python3
"""
fire_vision.py — AI-Powered Fire & Smoke Detection
===================================================
Detects fire and/or smoke in uploaded photographs using a Roboflow workflow.

**Primary path**: Runs the ``fire-and-smoke-segmentation-alerts`` workflow
via a local Roboflow inference server (http://127.0.0.1:9001) using the
``inference_sdk`` package.

**Fallback path**: If the local server is unreachable, calls the Roboflow
hosted inference API at ``outline.roboflow.com`` using stdlib ``urllib``.

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
"""

from __future__ import annotations

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

# Hosted fallback — direct REST API on outline.roboflow.com
_FULL_MODEL_PATH = "roboflow-universe-projects/fire-and-smoke-segmentation/11"
_API_HOST = "https://outline.roboflow.com"
_USER_AGENT = "WildFrame/1.0 (fire-vision; contact@wildframe.example)"
_TIMEOUT_S = 30

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

    Workflows return a list of output dicts (one per input image). Each
    dict can have various field names depending on how the workflow is
    configured. We scan for any field that looks like a list of predictions
    (dicts with ``class`` and ``confidence`` keys).

    Handles two common formats:
      1. ``{"predictions": [{"class": "smoke", ...}, ...]}`` (list)
      2. ``{"predictions": {"predictions": [...], "width": 678, "height": 452}}``
         (dict wrapper — Roboflow inference API format)
    """
    if isinstance(workflow_result, list):
        if len(workflow_result) == 0:
            return []
        candidates = workflow_result[0]
    else:
        candidates = workflow_result

    if not isinstance(candidates, dict):
        if isinstance(candidates, list):
            return candidates
        return []

    # ----- Helper: check if a value is a prediction-like list -----
    def _looks_like_predictions(val: object) -> list[dict] | None:
        """If *val* is a non-empty list of prediction dicts, return it; else None."""
        if not isinstance(val, list) or len(val) == 0:
            return None
        first = val[0]
        if isinstance(first, dict) and ("class" in first or "class_name" in first) and "confidence" in first:
            return val
        return None

    # ----- Scan top-level fields -----
    for value in candidates.values():
        # Case 1: direct list of predictions
        found = _looks_like_predictions(value)
        if found is not None:
            return found

        # Case 2: dict wrapper (e.g. {"predictions": [...], "width": ..., "height": ...})
        if isinstance(value, dict):
            # Check every sub-field for a prediction-like list
            for sub_val in value.values():
                found = _looks_like_predictions(sub_val)
                if found is not None:
                    return found

    # ----- Fallback: if no structured predictions found, return empty -----
    return []


# ---------------------------------------------------------------------------
# Fallback path — hosted REST API (urllib)
# ---------------------------------------------------------------------------


def _scan_via_hosted_api(
    path: Path,
    api_key: str,
    confidence_threshold: float,
) -> dict[str, Any]:
    """Call the Roboflow hosted inference API directly (same as old implementation)."""
    raw_bytes = path.read_bytes()
    import base64
    b64_data = base64.b64encode(raw_bytes).decode("ascii")

    url = f"{_API_HOST}/{_FULL_MODEL_PATH}?api_key={api_key}"
    payload_bytes = b64_data.encode("ascii")

    req = urllib.request.Request(
        url,
        data=payload_bytes,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": _USER_AGENT,
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        log.warning("Hosted API HTTP %d: %s", exc.code, body)
        raise RuntimeError(f"Hosted API HTTP {exc.code}: {body[:200]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Hosted API connection error: {exc.reason}") from exc

    data = json.loads(raw)
    predictions: list[dict] = data.get("predictions", [])
    return _build_result(predictions, confidence_threshold, _FULL_MODEL_PATH)


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