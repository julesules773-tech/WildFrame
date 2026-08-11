# Fire gate model — retraining

A small MobileNetV3-Small binary classifier that runs **client-side** (TF.js,
graph model) to give instant UX feedback before a user submits a photo. It
only soft-rejects (notice + submit-anyway); the server-side Roboflow scan and
auto-approval logic are untouched and authoritative.

## Why graph-model format (not layers)

MobileNetV3 contains TFOpLambda ops (hard-swish etc.) that the tfjs **layers**
format cannot deserialize ("Unknown layer: TFOpLambda"), and TF 2.16's Keras 3
also drops InputLayer shapes in the layers export. The **graph-model** path
(keras → SavedModel → `convert_tf_saved_model`) handles both correctly and is
loaded in the browser with `tf.loadGraphModel` — do not switch back to
`save_keras_model`.

## Prereqs (this venv only — prod is untouched)

```bash
python3 -m venv .venv
.venv/bin/pip install 'tensorflow==2.15.0' 'tensorflowjs==4.22.0'
```

Pin those exact versions — TF 2.15 ships Keras 2; newer TF ships Keras 3 and
the converter emits unloadable topologies.

## Retrain

```bash
.venv/bin/python train_gate.py --epochs-head 12 --epochs-finetune 0
```

- Positives: `~/Downloads/fire_dataset/fire_images/` (real fires)
- Negatives: `~/Downloads/fire_dataset/non_fire_images/` (clean scenes —
  **add sunset/red-sky photos as hard negatives**; warm scenes are the classic
  false positive)
- Eval: 15% stratified hold-out + `sweep.json` (200 labeled rows) + the 31
  "canary" fires Roboflow missed at 0.0 confidence — the gate must keep every
  canary above the threshold.

Outputs (committed to `static/model/gate/`):
- `model.json` + `group1-shard1of1.bin` — TF.js **graph** model
- `config.json` — `min_prob` threshold the frontend reads
- `gate.keras` — keras checkpoint for diagnostics / export-only reruns

## Sanity-check the export (browser fidelity)

```bash
node --check ../static/app.js
```

The browser path feeds raw [0,255] float32 224×224 pixels (the model embeds
its own Rescaling first layer) and reads `predict(...).data()[0]` as the fire
probability — mirror that exactly; double-normalizing crushes outputs into
~[0.3, 0.6].
