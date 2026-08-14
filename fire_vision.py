#!/usr/bin/env python3
"""
fire_vision.py — AI-Powered Fire & Smoke Detection
===================================================
Detects fire and/or smoke in uploaded photographs.

**Primary path (default)**: a local YOLOv26 fire/smoke detector — the ONNX
export (``models/best.onnx``) served via onnxruntime in production, or the
``.pt`` checkpoint via ultralytics/torch in dev. The engine is selected by
``WILDFRAME_VISION_ENGINE`` (``auto`` | ``yolo`` | ``roboflow``).

**Roboflow fallback**: when no local model is present (or the engine is
forced to ``roboflow``), runs the ``fire-and-smoke-segmentation-alerts``
workflow via a local Roboflow inference server (http://127.0.0.1:9001)
with the hosted API at ``serverless.roboflow.com`` as its own fallback
(needs ROBOFLOW_API_KEY).

**Ensemble mode (optional)**: with ``WILDFRAME_VISION_ENSEMBLE=1`` and a
Roboflow key present, every scan runs the local YOLO model and the Roboflow
workflow IN PARALLEL and merges by max per-class confidence, so either
engine detecting fire/smoke raises that class's confidence — the two models
cover each other's blind spots. Off by default (YOLO-only); the Roboflow leg
uses ``WILDFRAME_ROBOFLOW_WORKFLOW`` when set (point it at a trained model
once one exists). If one engine fails, the other's verdict stands.

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
import threading
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
# Local YOLO engine configuration
# ---------------------------------------------------------------------------
# Engine selection (WILDFRAME_VISION_ENGINE):
#   auto      (default) — local YOLO when models/best.onnx or best.pt exists,
#                          otherwise the Roboflow workflow (needs an API key).
#   yolo      — force the local YOLO model; Roboflow stays as fallback.
#   roboflow  — force the original Roboflow workflow path.
#
# The YOLO model is YOLOv26-S fire/smoke detection, MIT-licensed, from
# https://huggingface.co/SalahALHaismawi/yolov26-fire-detection
# (classes: fire / smoke / other), fine-tuned locally on the Roboflow
# fire/smoke dataset via train_local.py (2 classes: fire/smoke). Locally it
# runs via ultralytics (torch); in production we serve the ONNX export
# through onnxruntime — a ~50 MB pip package that fits the 1 GB Lightsail
# VM, where torch would OOM.
_YOLO_MODEL_LABEL = "yolov26:fire-detection:retrained"
_YOLO_IMG_SIZE = 640
_YOLO_PAD_COLOR = (114, 114, 114)
# Model class ids -> our vocabulary. The retrained model (train_local.py,
# fine-tuned on the Roboflow fire/smoke dataset) has exactly two classes:
# 0=fire, 1=smoke. (The original HF checkpoint had a third catch-all class
# "other" at id 1, deliberately unmapped; the retrain drops it.)
_YOLO_CLASS_NAMES = {0: "fire", 1: "smoke"}

# --- Ensemble mode (parallel YOLO + Roboflow) ---
# When WILDFRAME_VISION_ENSEMBLE=1, every scan runs the local YOLO model AND
# the Roboflow workflow in parallel and merges by MAX per-class confidence,
# so either model detecting fire/smoke raises that class's confidence — the
# two engines cover each other's blind spots. Default OFF (YOLO-only): the
# Roboflow leg needs a trained cloud model (point WILDFRAME_ROBOFLOW_WORKFLOW
# at it), and merged confidences sit above YOLO-alone values, so the
# auto-approval floors (server.py) must be re-tuned via sweep_yolo before
# enabling.
_ENSEMBLE_ENABLED = os.environ.get("WILDFRAME_VISION_ENSEMBLE", "0") == "1"
_ENSEMBLE_WORKFLOW = os.environ.get("WILDFRAME_ROBOFLOW_WORKFLOW") or _WORKFLOW_ID

if _ENSEMBLE_ENABLED and os.environ.get("WILDFRAME_ROBOFLOW_WORKFLOW") is None:
    # The default workflow references the old (now-empty) model — flipping
    # the ensemble on without pointing it at a trained workflow buys zero
    # coverage while taxing every upload with the hosted call's latency.
    log.warning(
        "Ensemble enabled but WILDFRAME_ROBOFLOW_WORKFLOW is not set — the "
        "Roboflow leg will use the default workflow (%s) which currently "
        "returns no detections. Set WILDFRAME_ROBOFLOW_WORKFLOW to a trained "
        "model's workflow before relying on the ensemble.",
        _WORKFLOW_ID,
    )

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

    Engine is chosen by ``WILDFRAME_VISION_ENGINE`` (auto|yolo|roboflow;
    default auto — local YOLO when models/best.onnx or best.pt exists,
    otherwise the Roboflow workflow). YOLO failures fall back to Roboflow
    when an API key is available.

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
    # --- Validate image file ---
    path = Path(image_path)
    if not path.is_file():
        return _error_dict(f"Image not found: {path}")

    # --- Resolve the Roboflow key (needed for the Roboflow path and the
    # ensemble's Roboflow leg) ---
    resolved_key = api_key or os.environ.get("ROBOFLOW_API_KEY")

    # --- Local YOLO engine (primary when configured / model present) ---
    if _resolve_engine() == "yolo":
        # Ensemble mode: YOLO + Roboflow in parallel. Needs a key for the
        # Roboflow leg; without one, fall through to YOLO-only.
        if _ENSEMBLE_ENABLED and resolved_key:
            try:
                return _scan_via_ensemble(path, confidence_threshold, resolved_key)
            except Exception as exc:
                log.warning("Ensemble scan failed: %s — falling back to YOLO-only", exc)
        try:
            return _scan_via_yolo(path, confidence_threshold)
        except Exception as exc:
            log.warning("Local YOLO scan failed: %s", exc)
            # Fall through to Roboflow when a key is available.
            if not resolved_key:
                return _error_dict(
                    f"Local YOLO scan failed: {exc} (and ROBOFLOW_API_KEY not set "
                    "for the Roboflow fallback)."
                )
    else:
        if not resolved_key:
            return _error_dict("ROBOFLOW_API_KEY not set. Set this env var or pass api_key=.")

    # --- Roboflow path: local workflow, then hosted API fallback ---
    return _scan_via_roboflow(path, resolved_key, confidence_threshold)


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
    workflow_id: str | None = None,
) -> dict[str, Any]:
    """Run the fire/smoke segmentation workflow via the local inference server."""
    try:
        from inference_sdk import InferenceHTTPClient
    except ImportError:
        raise RuntimeError(
            "inference_sdk not installed. Run: pip install inference-sdk"
        )

    workflow_id = workflow_id or _WORKFLOW_ID
    client = InferenceHTTPClient(
        api_url=_LOCAL_INFERENCE_URL,
        api_key=api_key,
    )

    t0 = time.time()
    result = client.run_workflow(
        workspace_name=_WORKSPACE_NAME,
        workflow_id=workflow_id,
        images={
            "image": str(path),
        },
    )
    elapsed = time.time() - t0
    log.info("Local workflow returned in %.1fs", elapsed)

    predictions = _extract_predictions_from_workflow(result)
    return _build_result(predictions, confidence_threshold, "workflow:" + workflow_id)


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
    workflow_id: str | None = None,
) -> dict[str, Any]:
    """
    Run the fire/smoke workflow via the Roboflow hosted inference API.

    The old per-task hosts (outline/detect/classify/segment.roboflow.com)
    are deprecated; the unified hosted API lives at serverless.roboflow.com
    and takes a JSON body with the API key and a base64 image input. Images
    are downscaled to ``_MAX_HOSTED_EDGE_PX`` first so oversized photos can't
    trigger HTTP 413 on the base64 request body.
    """
    workflow_id = workflow_id or _WORKFLOW_ID
    image_bytes = _prepare_hosted_image_bytes(path)
    b64_data = base64.b64encode(image_bytes).decode("ascii")

    url = f"{_API_HOST}/infer/workflows/{_WORKSPACE_NAME}/{workflow_id}"
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
    return _build_result(predictions, confidence_threshold, f"workflow:{workflow_id}@hosted")


def _scan_via_roboflow(
    path: Path,
    api_key: str,
    confidence_threshold: float,
    workflow_id: str | None = None,
) -> dict[str, Any]:
    """Run the Roboflow workflow: local inference server first, hosted API as
    fallback. Shared by the single-engine path and the ensemble's Roboflow leg.
    """
    try:
        return _scan_via_local_workflow(path, api_key, confidence_threshold, workflow_id=workflow_id)
    except Exception as exc:
        log.info("Local workflow failed, falling back to hosted API: %s", exc)
    return _scan_via_hosted_api(path, api_key, confidence_threshold, workflow_id=workflow_id)


# ---------------------------------------------------------------------------
# Local YOLO engine — pretrained YOLOv26 fire/smoke detection
# ---------------------------------------------------------------------------


def _yolo_model_path() -> Path | None:
    """Path to the local YOLO weights.

    Honors ``WILDFRAME_YOLO_MODEL`` when set; otherwise looks for
    ``models/best.onnx`` (prod — onnxruntime) then ``models/best.pt``
    (dev — ultralytics) relative to this file.
    """
    override = os.environ.get("WILDFRAME_YOLO_MODEL")
    if override:
        p = Path(override)
        return p if p.is_file() else None
    models_dir = Path(__file__).resolve().parent / "models"
    for candidate in ("best.onnx", "best.pt"):
        p = models_dir / candidate
        if p.is_file():
            return p
    return None


def _resolve_engine() -> str:
    """Return the active vision engine: ``"yolo"`` or ``"roboflow"``.

    Controlled by ``WILDFRAME_VISION_ENGINE`` (``auto`` | ``yolo`` |
    ``roboflow``; default ``auto``) — ``auto`` picks local YOLO whenever the
    weights are on disk, otherwise the Roboflow workflow.
    """
    engine = os.environ.get("WILDFRAME_VISION_ENGINE", "auto").strip().lower()
    if engine == "yolo":
        return "yolo"
    if engine == "roboflow":
        return "roboflow"
    return "yolo" if _yolo_model_path() is not None else "roboflow"


def _scan_via_yolo(path: Path, confidence_threshold: float) -> dict[str, Any]:
    """Run the local YOLOv26 model and normalize to the uniform result dict.

    Prefers the ONNX Runtime backend (prod — tiny footprint, no torch) and
    falls back to ultralytics/torch (dev) when only the ``.pt`` file exists.
    """
    model_path = _yolo_model_path()
    if model_path is None:
        raise RuntimeError(
            "No local YOLO model found (looked for models/best.onnx / best.pt). "
            "Download it from https://huggingface.co/SalahALHaismawi/yolov26-fire-detection"
        )
    if model_path.suffix == ".onnx":
        detections = _run_yolo_onnx(model_path, path, confidence_threshold)
    else:
        detections = _run_yolo_torch(model_path, path, confidence_threshold)
    return _build_result(detections, confidence_threshold, _YOLO_MODEL_LABEL)


def _scan_via_ensemble(
    path: Path,
    confidence_threshold: float,
    api_key: str,
) -> dict[str, Any]:
    """Run the local YOLO model and the Roboflow workflow IN PARALLEL and
    merge by max per-class confidence — either engine detecting fire/smoke
    raises that class's confidence, so the two models cover each other's
    blind spots.

    Both engines run in worker threads and are joined before merging. If one
    engine fails (Roboflow outage, missing key, etc.), the other's verdict
    stands and the failure is surfaced on the result's ``error`` field — a
    degraded-but-useful scan, never a hard failure. If both fail, the result
    is an error dict.

    The merge reuses ``_build_result``: each leg's per-class confidence is
    the max over its own detections, so concatenating both legs' detections
    and deriving the verdict through the shared builder yields identical
    numbers while keeping the verdict rules in ONE place.

    Returns the uniform result dict (same shape as scan_photo) with the
    merged confidence and a ``model`` label like
    ``ensemble:yolov26:fire-detection:retrained+workflow:<id>``.
    """
    results: dict[str, dict[str, Any]] = {}

    def _run_yolo() -> None:
        try:
            results["yolo"] = _scan_via_yolo(path, confidence_threshold)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("Ensemble YOLO leg failed: %s", exc)
            results["yolo"] = _error_dict(f"YOLO leg failed: {exc}")

    def _run_roboflow() -> None:
        try:
            results["roboflow"] = _scan_via_roboflow(
                path, api_key, confidence_threshold, workflow_id=_ENSEMBLE_WORKFLOW,
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("Ensemble Roboflow leg failed: %s", exc)
            results["roboflow"] = _error_dict(f"Roboflow leg failed: {exc}")

    threads = [threading.Thread(target=_run_yolo), threading.Thread(target=_run_roboflow)]
    for t in threads:
        t.start()
    for t in threads:
        # Internal HTTP timeouts bound each leg (hosted API _TIMEOUT_S=60s),
        # so an unbounded join is effectively bounded — but a cap here keeps
        # the ensemble robust to any future leg without its own timeout.
        t.join(timeout=90.0)

    yolo_r = results.get("yolo") or _error_dict("YOLO leg did not return")
    rf_r = results.get("roboflow") or _error_dict("Roboflow leg did not return")

    # Merge through the shared builder: concatenated detections -> identical
    # per-class max confidence + verdict as the single-engine path computes.
    detections = list(yolo_r.get("detections") or []) + list(rf_r.get("detections") or [])
    model_label = "ensemble:" + "+".join(
        m for m in (yolo_r.get("model"), rf_r.get("model")) if m
    )
    result = _build_result(detections, confidence_threshold, model_label)

    # Surface one-leg failures (degraded-but-useful) and force an error
    # verdict only when BOTH legs failed.
    errors = [e for e in (yolo_r.get("error"), rf_r.get("error")) if e]
    if errors:
        result["error"] = " | ".join(errors)
    if yolo_r.get("verdict") == "error" and rf_r.get("verdict") == "error":
        result["verdict"] = "error"
    return result


def _run_yolo_torch(model_path: Path, path: Path, confidence_threshold: float) -> list[dict]:
    """YOLO inference via ultralytics/torch (dev machine)."""
    from ultralytics import YOLO

    model = _yolo_torch_model(model_path)
    results = model.predict(
        str(path),
        conf=confidence_threshold,
        imgsz=_YOLO_IMG_SIZE,
        verbose=False,
        device="cpu",
    )
    detections: list[dict[str, Any]] = []
    boxes = results[0].boxes
    if boxes is None:
        return detections
    for box in boxes:
        cls = int(box.cls[0])
        label = _YOLO_CLASS_NAMES.get(cls)
        if label is None:
            continue  # "other" — never flips a verdict (matches the ONNX path)
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        detections.append({"class": label, "confidence": conf, "bbox": [x1, y1, x2, y2]})
    return detections


def _run_yolo_onnx(model_path: Path, path: Path, confidence_threshold: float) -> list[dict]:
    """YOLO inference via onnxruntime (production).

    The exported graph is end-to-end: a single ``[1, 300, 6]`` output of
    ``[x1, y1, x2, y2, conf, cls]`` with NMS baked in, so the only work left
    is letterboxing the image, un-mapping the boxes, and thresholding.
    """
    import onnxruntime as ort
    from PIL import Image, ImageOps

    session = _yolo_onnx_session(model_path)
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
    blob, ratio, pad_l, pad_t = _yolo_letterbox(img, _YOLO_IMG_SIZE)
    outputs = session.run(None, {session.get_inputs()[0].name: blob})
    return _decode_yolo_end2end(outputs, ratio, pad_l, pad_t, confidence_threshold)


def _yolo_letterbox(img: Any, size: int) -> tuple[Any, float, float, float]:
    """Resize + gray-pad *img* to ``size`` x ``size``.

    Returns ``(NCHW float32 blob normalized to 0-1, scale ratio, pad_left,
    pad_top)`` so detection boxes can be mapped back to the original image.
    """
    import numpy as np
    from PIL import Image

    w, h = img.size
    ratio = min(size / w, size / h)
    nw, nh = max(1, round(w * ratio)), max(1, round(h * ratio))
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), _YOLO_PAD_COLOR)
    pad_l = (size - nw) // 2
    pad_t = (size - nh) // 2
    canvas.paste(resized, (pad_l, pad_t))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    blob = np.ascontiguousarray(arr[None].transpose(0, 3, 1, 2))
    return blob, ratio, float(pad_l), float(pad_t)


def _decode_yolo_end2end(
    outputs: list[Any],
    ratio: float,
    pad_l: float,
    pad_t: float,
    confidence_threshold: float,
) -> list[dict]:
    """Decode the end-to-end ONNX output ``[1, N, 6]`` -> detection dicts.

    Unmaps the NMS-ed boxes back to original pixel coordinates and skips
    detections below the threshold or from unmapped classes ("other").

    NOTE: the exported graph bakes in its own NMS/conf filter (~0.25), so
    ultra-low-confidence detections in the 0.10-0.25 band appear via the
    torch backend but not here — verdicts and max confidences are unaffected
    and both sit far below the auto-approval floors (0.80/0.40).
    """
    detections: list[dict[str, Any]] = []
    if not outputs:
        return detections
    out = outputs[0]
    if out.ndim != 3 or out.shape[-1] != 6:
        log.warning("Unexpected YOLO ONNX output shape %s — treating as no detections", out.shape)
        return detections
    for row in out[0]:
        x1, y1, x2, y2, conf, cls = (float(v) for v in row[:6])
        if conf < confidence_threshold:
            continue
        label = _YOLO_CLASS_NAMES.get(int(cls))
        if label is None:
            continue  # "other" — never flips a verdict
        detections.append(
            {
                "class": label,
                "confidence": conf,
                "bbox": [
                    (x1 - pad_l) / ratio,
                    (y1 - pad_t) / ratio,
                    (x2 - pad_l) / ratio,
                    (y2 - pad_t) / ratio,
                ],
            }
        )
    return detections


_yolo_cache_lock = threading.Lock()
_yolo_torch_cache: tuple[str, Any] | None = None


def _yolo_torch_model(model_path: Path):
    """Module-level cached ultralytics model (loading takes ~1-2s)."""
    global _yolo_torch_cache
    if _yolo_torch_cache is None or _yolo_torch_cache[0] != str(model_path):
        with _yolo_cache_lock:
            if _yolo_torch_cache is None or _yolo_torch_cache[0] != str(model_path):
                from ultralytics import YOLO

                _yolo_torch_cache = (str(model_path), YOLO(model_path))
    return _yolo_torch_cache[1]


_yolo_onnx_cache: tuple[str, Any] | None = None


def _yolo_onnx_session(model_path: Path):
    """Module-level cached ONNX Runtime session (first scan pays load cost)."""
    global _yolo_onnx_cache
    if _yolo_onnx_cache is None or _yolo_onnx_cache[0] != str(model_path):
        with _yolo_cache_lock:
            if _yolo_onnx_cache is None or _yolo_onnx_cache[0] != str(model_path):
                import onnxruntime as ort

                _yolo_onnx_cache = (
                    str(model_path),
                    ort.InferenceSession(
                        str(model_path), providers=["CPUExecutionProvider"]
                    ),
                )
    return _yolo_onnx_cache[1]


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