#!/usr/bin/env python3
"""Train the client-side fire/smoke GATE model.

Purpose
-------
A fast binary pre-flight classifier (fire/smoke vs. not) that runs in the
browser via TF.js, purely to give instant UX feedback before a user submits
a photo. It NEVER decides anything authoritative — the server-side Roboflow
scan and auto-approval logic are untouched. A low fire-probability photo is
soft-rejected (notice + submit-anyway), never blocked or discarded.

Training set
------------
- positives : ~/Downloads/fire_dataset/fire_images/      (real fires)
- negatives : ~/Downloads/fire_dataset/non_fire_images/  (clean scenes;
              ideally incl. sunsets/red-sky hard negatives)

Eval sets
---------
- 15% stratified hold-out of the training set
- sweep.json (200 labeled rows; overlaps the training folders, so its
  accuracy is optimistic — the CANARY pass-rate is the metric that matters)
- canaries = rows with positive=True and fire_conf == 0.0 (the 31 real
  fires the hosted Roboflow model completely missed). The gate must keep
  ~100% of these above the threshold: those are exactly the photos a naive
  filter would wrongly reject.

Output
------
- static/model/gate/model.json + weight shards (TF.js format, served to
  the browser)
- static/model/gate/config.json  {min_prob, metrics, ...} — the threshold
  the frontend uses. Pick the largest threshold that still keeps every
  canary above it (with margin), so we never lose real fires.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

FIRE_DIR = os.path.expanduser("~/Downloads/fire_dataset/fire_images")
NONFIRE_DIR = os.path.expanduser("~/Downloads/fire_dataset/non_fire_images")
SWEEP = os.path.join(os.path.dirname(__file__), "..", "sweep.json")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "static", "model", "gate")
IMG_SIZE = 224
SEED = 7


def _image_paths(directory):
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif")
    return [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if f.lower().endswith(exts)
    ]


def _load_images(paths, labels, size=IMG_SIZE):
    """Load a list of image paths into a float32 [0, 255] tensor stack.

    NOTE: MobileNetV3 in tf.keras embeds its own Rescaling(1/127.5, -1)
    as the first layer (include_preprocessing is NOT a settable kwarg in
    TF 2.16), so we must feed RAW [0, 255] pixels — normalizing here too
    double-scales the input, crushes activations, and squeezes every
    sigmoid output into ~[0.3, 0.6]. The browser gate does the same.
    """
    xs, ys = [], []
    for p, y in zip(paths, labels):
        try:
            raw = tf.io.read_file(p)
            img = tf.image.decode_image(raw, channels=3, expand_animations=False)
            h = tf.shape(img)[0]
            w = tf.shape(img)[1]
            # cover-crop to square — MUST mirror the browser gate exactly
            # (canvas cover-crop in _gatePhoto); stretching instead drifts
            # non-square phone photos out of the training distribution.
            scale = tf.maximum(size / tf.cast(w, tf.float32), size / tf.cast(h, tf.float32))
            cw = tf.cast(tf.round(size / scale), tf.int32)
            ch = tf.cast(tf.round(size / scale), tf.int32)
            ox = (w - cw) // 2
            oy = (h - ch) // 2
            img = img[oy:oy + ch, ox:ox + cw]
            img = tf.image.resize(img, [size, size])
            img = tf.cast(img, tf.float32)  # raw [0, 255] — model preprocesses
            xs.append(img)
            ys.append(y)
        except Exception as exc:  # noqa: BLE001 — skip corrupt/undecodable files
            print(f"  ! skip {os.path.basename(p)}: {exc}")
    return tf.stack(xs), tf.stack(ys)


def _build_model():
    """Build a FUNCTIONAL model — the TF.js converter emits an explicit
    input shape for Functional models, but drops the InputLayer shape for
    Sequential-wrapping-a-Functional-base (the browser then fails with
    'An InputLayer should be passed either a batchInputShape or an
    inputShape'). Functional keeps the exported model loadable."""
    base = MobileNetV3Small(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        minimalistic=False,
        # NOTE: no include_preprocessing arg — MobileNetV3 (unlike NASNet/
        # EfficientNet) always embeds its Rescaling first layer; feed [0,255].
    )
    base.trainable = False
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="gate_input")
    x = base(inp)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(1, activation="sigmoid", name="gate_output")(x)
    model = tf.keras.Model(inp, out, name="fire_gate")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model, base


def _augment(x, y):
    # NOTE: tf.image.random_rotation was removed in TF 2.16 — keep flips +
    # mild contrast only (strong brightness jitter can wash out fire colors).
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_contrast(x, 0.9, 1.1)
    return x, y


def _threshold_report(model, name, xs, ys, canaries=None):
    """Print precision/recall per threshold + canary pass rate."""
    probs = model.predict(xs, verbose=0)[:, 0]
    y = np.asarray(ys)
    print(f"\n  — {name}: {len(y)} images ({int(y.sum())} fire / {int((1 - y).sum())} non-fire) —")
    for label, mask in (("fire", y == 1), ("non-fire", y == 0)):
        p = probs[mask]
        q = np.percentile(p, [5, 25, 50, 75, 95]).round(3)
        print(f"    {label:9s} prob p5/p25/p50/p75/p95: {q}  (min {p.min():.3f} / max {p.max():.3f})")
    # Precompute canary probs ONCE (predicting per-threshold is wasteful).
    cpreds = model.predict(canaries, verbose=0)[:, 0] if canaries is not None else None
    print(f"  {'T':>5} {'PREC':>7} {'RECALL':>7} {'F1':>6}  canary-pass")
    best = (0, None)
    for t in np.arange(0.10, 0.96, 0.05):
        pred = (probs >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        canary = ""
        if cpreds is not None:
            keep = int((cpreds >= t).sum()) / len(cpreds) * 100
            canary = f"{keep:5.0f}%"
            if rec > 0:
                score = f1 + 10.0 * keep / 100.0
                if score > best[0]:
                    best = (score, t)
        print(f"  {t:5.2f} {prec:7.1%} {rec:7.1%} {f1:6.2f}  {canary}")
    if cpreds is not None:
        print(f"  → suggested threshold: {best[1]:.2f} (best F1 + canary-keep)")
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-head", type=int, default=8)
    ap.add_argument("--epochs-finetune", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--skip-train", action="store_true", help="only export/eval existing model")
    args = ap.parse_args()

    random.seed(SEED)
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    fire_paths = _image_paths(FIRE_DIR)
    nonfire_paths = _image_paths(NONFIRE_DIR)
    if not fire_paths or not nonfire_paths:
        sys.exit(f"Dataset missing: {len(fire_paths)} fire / {len(nonfire_paths)} non-fire")
    print(f"dataset: {len(fire_paths)} fire / {len(nonfire_paths)} non-fire")

    all_paths = fire_paths + nonfire_paths
    all_labels = [1] * len(fire_paths) + [0] * len(nonfire_paths)
    # Stratified split preserving fire:non-fire ratio in both halves.
    pairs = list(zip(all_paths, all_labels))
    random.shuffle(pairs)
    n_pos = sum(1 for _, y in pairs if y == 1)
    n_neg = len(pairs) - n_pos
    val_pos = int(n_pos * args.val_split)
    val_neg = int(n_neg * args.val_split)
    train_pairs, val_pairs = [], []
    for p, y in pairs:
        (val_pairs if (y == 1 and len([q for q, yq in val_pairs if yq == 1]) < val_pos
                       or y == 0 and len([q for q, yq in val_pairs if yq == 0]) < val_neg)
         else train_pairs).append((p, y))

    t0 = time.time()
    print("loading train images…")
    xs_tr, ys_tr = _load_images([p for p, _ in train_pairs], [y for _, y in train_pairs])
    print("loading val images…")
    xs_va, ys_va = _load_images([p for p, _ in val_pairs], [y for _, y in val_pairs])
    print(f"train {len(ys_tr)} / val {len(ys_va)} — loaded in {time.time() - t0:.0f}s")

    # Canaries + full sweep (for eval only)
    canary_paths = []
    sweep_items = []
    if os.path.exists(SWEEP):
        with open(SWEEP) as f:
            for r in json.load(f)["rows"]:
                if r.get("error") or not os.path.exists(r["file"]):
                    continue
                sweep_items.append((r["file"], bool(r["positive"])))
                if r["positive"] and r.get("fire_conf") == 0.0:
                    canary_paths.append(r["file"])
    xs_can, _ = _load_images(canary_paths, [1] * len(canary_paths)) if canary_paths else (None, None)
    print(f"eval sets: {len(sweep_items)} sweep rows, {len(canary_paths)} canaries")

    model, base = _build_model()
    if not args.skip_train:
        # Class weights (fire is ~3x the negatives; don't let it dominate).
        total = len(ys_tr)
        w_pos = total / (2 * ys_tr.numpy().sum())
        w_neg = total / (2 * (1 - ys_tr.numpy()).sum())
        class_weight = {0: w_neg, 1: w_pos}

        train_ds = tf.data.Dataset.from_tensor_slices((xs_tr, ys_tr))
        train_ds = train_ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE).batch(args.batch)
        val_ds = tf.data.Dataset.from_tensor_slices((xs_va, ys_va)).batch(args.batch)

        print("\nphase A — training head (base frozen)")
        model.fit(
            train_ds, validation_data=val_ds, epochs=args.epochs_head,
            class_weight=class_weight,
            callbacks=[
                EarlyStopping(monitor="val_auc", patience=2, mode="max", restore_best_weights=True),
            ],
            verbose=2,
        )
        print("\nphase B — fine-tuning last 40 base layers @ lr 1e-5")
        base.trainable = True
        for layer in base.layers[:-40]:
            layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()],
        )
        model.fit(
            train_ds, validation_data=val_ds, epochs=args.epochs_finetune,
            class_weight=class_weight,
            callbacks=[
                EarlyStopping(monitor="val_auc", patience=2, mode="max", restore_best_weights=True),
            ],
            verbose=2,
        )

    print("\n=== hold-out evaluation ===")
    _threshold_report(model, "hold-out", xs_va, ys_va, canaries=xs_can)

    print("\n=== sweep evaluation (optimistic — overlaps training) ===")
    if sweep_items:
        xs_sw, ys_sw = _load_images([p for p, _ in sweep_items], [y for _, y in sweep_items])
        _threshold_report(model, "sweep", xs_sw, ys_sw)

    # Pick the threshold the frontend uses: the largest t that still keeps
    # 100% of canaries above it (with 0.05 margin), so real fires are never
    # soft-rejected — else fall back to 0.40.
    if canary_paths:
        cprobs = model.predict(xs_can, verbose=0)[:, 0]
        min_canary = float(cprobs.min())
        min_prob = round(min(min_canary - 0.05, 0.5), 2)
        print(f"\ncanary min prob: {min_canary:.3f} → gate threshold {min_prob}")
    else:
        min_prob = 0.40
    if min_prob < 0.10:
        print("⚠ warning: threshold very low — gate will rarely soft-reject")
        min_prob = 0.10

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\nsaving keras checkpoint + exporting TF.js graph model → {OUT_DIR}")
    model.save(os.path.join(OUT_DIR, "gate.keras"))
    # tfjs.converters.save_keras_model (layers format) FAILS on this
    # architecture: MobileNetV3 contains TFOpLambda ops (hard-swish etc.)
    # that tfjs-layers cannot deserialize ('Unknown layer: TFOpLambda').
    # The graph-model format converts the real TF ops and loads via
    # tf.loadGraphModel in the browser. Must match static/app.js.
    saved_dir = os.path.join(OUT_DIR, "saved_model")
    try:
        model.export(saved_dir)  # TF 2.15+: exports serving_default signature
    except AttributeError:
        tf.saved_model.save(model, saved_dir)
    # tfjs 4.22: no signature_name kwarg — it defaults to serving_default.
    tfjs.converters.convert_tf_saved_model(saved_dir, OUT_DIR)
    import shutil
    shutil.rmtree(saved_dir, ignore_errors=True)
    config = {
        "model": "mobilenetv3-small-firetuned",
        "min_prob": min_prob,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {"fire": len(fire_paths), "non_fire": len(nonfire_paths)},
        "train": {"fire": int(ys_tr.numpy().sum()), "non_fire": int((1 - ys_tr.numpy()).sum())},
        "val": {"fire": int(ys_va.numpy().sum()), "non_fire": int((1 - ys_va.numpy()).sum())},
        "canaries": len(canary_paths),
    }
    with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"config.json: {config}")


if __name__ == "__main__":
    main()
