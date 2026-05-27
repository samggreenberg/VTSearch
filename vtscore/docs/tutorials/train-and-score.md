# Tutorial: Train and Score a Detector End-to-End

This tutorial builds a working detector from scratch. By the end you
will have:

1. Loaded a folder of audio files into `vtscore`.
2. Inspected the resulting media items and their embeddings.
3. Cast labels on a handful of items.
4. Trained an MLP detector with a calibrated threshold.
5. Saved the detector to disk as a `LabelSet` JSON.
6. Reloaded it in a fresh Python process and re-derived its weights.
7. Scored a held-out folder against the detector.
8. Evaluated detector quality using `vtscore.eval`.

The scenario: you have a corpus of mixed audio - barks, music, speech,
environmental sound - and you want a binary detector for "is this a
dog bark?". Six labels is enough to get the MLP off the ground; a few
dozen will make it good.

You'll need `vtscore` installed (`bash scripts/install-cpu.sh` from the
repo root) and a CPU-only Python environment with PyTorch. GPU is fine
too but not required.

## Step 0: Set up the environment

```python
from pathlib import Path
import numpy as np
import torch

# vtscore needs to know where to put its data dir. For this tutorial
# we'll use a scratch directory.
WORKDIR = Path("/tmp/vtscore-tutorial")
WORKDIR.mkdir(exist_ok=True)

from vtscore.config import CoreConfig, register_core_config_builder

def _build_config() -> CoreConfig:
    return CoreConfig(
        data_dir=WORKDIR,
        saved_datasets_dir=WORKDIR / "datasets",
        detectors_dir=WORKDIR / "detectors",
        inclusion=0,
        safe_thresholds=False,
        calibrate_count=2,
        calibration_fraction=0.5,
        enrich_descriptions=False,
        autopilot_goal_diversity=0.5,
        max_concurrent_dataset_downloads=1,
        max_concurrent_dataset_embeddings=1,
    )

register_core_config_builder(_build_config)
print(f"Library data dir: {CoreConfig.from_settings().data_dir}")
```

That's the only setup `vtscore` requires. The library refuses to guess
where to put data - by setting up the config builder, you've answered.

## Step 1: Load a folder

We'll use the ESC-50 demo dataset, which ships with vtscore and contains
2,000 5-second environmental audio clips across 50 categories (one of
which is "dog"). If you don't want to download ESC-50, swap in your
own folder of `.wav` files.

```python
from vtscore.media import audio  # noqa: F401 - register MediaType + embedders
from vtscore.datasets.loader import load_dataset_from_folder

# Replace with your folder of audio files.
AUDIO_FOLDER = Path("/data/esc50/audio")  # or wherever your audio lives

medias: dict[int, dict] = {}
load_dataset_from_folder(
    folder_path=AUDIO_FOLDER,
    media_type="audio",
    medias=medias,
    recursive=True,
)
print(f"Loaded {len(medias)} audio items.")
```

What just happened:

1. `vtscore.media.audio` import registered the audio `MediaType`, the
   LAION-CLAP `MediaEmbedder`, and the default `CLIPPERS`.
2. `load_dataset_from_folder` walked `AUDIO_FOLDER`, ran every `.wav`
   through CLAP, and populated `medias` with the embedding + origin
   metadata for each file.
3. CLAP downloaded its weights on first use to
   `WORKDIR/models/` (about 600 MB; subsequent runs are instant).

The first call takes ~30s for model download and ~10s per 100 files
for embedding. Once cached, the same call runs in seconds.

Peek at the shape of a media item:

```python
sample = next(iter(medias.values()))
print({k: type(v).__name__ for k, v in sample.items()})
# {'id': 'int', 'type': 'str', 'embedder': 'str', 'file_size': 'int',
#  'md5': 'str', 'embedding': 'ndarray', 'filename': 'str', ...}

print(f"Embedding shape: {sample['embedding'].shape}")  # (512,)
print(f"Origin: {sample['origin']}")
# {'importer': 'server_folder', 'params': {'path': '/data/esc50/audio', 'media_type': 'audio'}}
```

The embedding is a 512-D float32 vector; LAION-CLAP's output
dimensionality. Every audio item now has one.

## Step 2: Label a few items

In a real workflow you'd use the VTSearch UI for this. For the tutorial,
hard-code labels by filename:

```python
# Pick six examples - three barks, three not-barks.
labels = {
    "1-100032-A-0.wav": "good",      # dog
    "1-110389-A-0.wav": "good",      # dog
    "2-114587-A-0.wav": "good",      # dog
    "1-30226-A-7.wav": "bad",        # rooster
    "1-103298-A-9.wav": "bad",       # crow
    "1-105019-A-3.wav": "bad",       # cow
}

# Map filename → media item.
by_filename = {m["filename"]: m for m in medias.values()}
labelled = [(m, labels[m["filename"]]) for fn, m in by_filename.items()
            if fn in labels]
print(f"Labelled {len(labelled)} items.")
```

ESC-50 uses `<fold>-<clip_id>-A-<category_id>.wav`; category 0 is "dog",
category 7 is "rooster", etc. Replace these filenames with whatever you
actually have. Six labels is enough - the MLP is small.

## Step 3: Train the MLP

```python
from vtscore.training import train_model, calculate_cross_calibration_threshold

# Build (X, y) tensors.
X_list = [m["embedding"] for m, _ in labelled]
y_list = [1.0 if label == "good" else 0.0 for _, label in labelled]

X = torch.from_numpy(np.stack(X_list).astype(np.float32))
y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)

# Train.
model = train_model(X, y, input_dim=X.shape[1])

# Calibrate the decision threshold via 2-fold cross-validation.
threshold = calculate_cross_calibration_threshold(
    X_list, y_list, input_dim=X.shape[1], inclusion_value=0,
)

print(f"Trained MLP; threshold = {threshold:.3f}")
```

The MLP architecture is `Linear(512, H) → ReLU → Dropout → Linear(H, 1)`
where `H` is auto-sized from the training-set count (`_auto_hidden_dim`
in `vtscore/training/mlp.py`). With 6 examples, `H ≈ 16`. The default
seed makes training deterministic.

## Step 4: Score all loaded items

```python
# Stack every loaded embedding.
ids = list(medias.keys())
embeddings = np.stack([medias[i]["embedding"] for i in ids]).astype(np.float32)

# Forward pass.
with torch.no_grad():
    logits = model(torch.from_numpy(embeddings)).squeeze(1).numpy()
scores = 1.0 / (1.0 + np.exp(-logits))   # sigmoid

# Rank and split at the threshold.
ranked = sorted(zip(ids, scores), key=lambda t: t[1], reverse=True)
above = [(cid, s) for cid, s in ranked if s >= threshold]
print(f"{len(above)} of {len(ranked)} items scored ≥ threshold.")

# Top 10.
for cid, score in ranked[:10]:
    fn = medias[cid]["filename"]
    print(f"  {score:0.3f}  {fn}")
```

You should see other dog clips bubble to the top - even ones you didn't
label, because CLAP's embedding space is good at clustering similar
sounds.

## Step 5: Save the detector

Detectors persist as JSON labelsets - never as model weights. On
reload, the library re-derives weights from the labelset's origins.

```python
from vtscore.datasets.labelset import LabeledElement, LabelSet
from vtscore.detectors.store import save_detector

elements = []
for m, label in labelled:
    elements.append(LabeledElement(
        md5=m["md5"],
        label=label,
        origin_name=m["origin_name"],
        origin=m["origin"],
    ))

labelset = LabelSet(
    labels=elements,
    metadata={
        "detector_name": "dog-barks",
        "media_type": "audio",
        "embedder": "audio_clap",
    },
)

save_detector("dog-barks", labelset)
print(f"Saved detector to {WORKDIR / 'detectors' / 'dog-barks.json'}")
```

Look at the JSON file:

```bash
$ cat /tmp/vtscore-tutorial/detectors/dog-barks.json
```

```json
{
  "detector_name": "dog-barks",
  "media_type": "audio",
  "embedder": "audio_clap",
  "labelset": {
    "labels": [
      {"md5": "5d41…", "label": "good",
       "origin_name": "1-100032-A-0.wav",
       "origin": {"importer": "server_folder",
                  "params": {"path": "/data/esc50/audio", "media_type": "audio"}}},
      …
    ]
  }
}
```

No embeddings. No weights. Six origins. That's the whole detector.

## Step 6: Reload in a fresh process

Open a new Python session - or just re-run the rest of the script after
setting `model = None; threshold = None` to prove the point.

```python
from vtscore.config import CoreConfig, register_core_config_builder
from vtscore.media import audio  # noqa: F401
from vtscore.state import DetectorContext, register_detector_context
from vtscore.detectors.training import train_detector_from_origins

register_core_config_builder(_build_config)   # same builder as before

ctx = DetectorContext(
    detector_id="dog-barks",
    name="dog-barks",
    media_type="audio",
    embedder="audio_clap",
)
register_detector_context(ctx)

# Build (good_origins, bad_origins) from the saved JSON's labelset, then
# resolve every origin to a file, embed each file *with the same embedder
# the detector was trained with* (here: "audio_clap"), and train. Passing
# ctx.embedder is critical - using "" would silently fall back to whatever
# the media type's default embedder is, mixing model outputs.
good_origins = [
    {"origin": e.origin, "origin_name": e.origin_name,
     "filename": e.filename, "md5": e.md5}
    for e in saved_labelset.labels if e.label == "good"
]
bad_origins = [
    {"origin": e.origin, "origin_name": e.origin_name,
     "filename": e.filename, "md5": e.md5}
    for e in saved_labelset.labels if e.label == "bad"
]
weights, threshold = train_detector_from_origins(
    good_origins,
    bad_origins,
    inclusion=0,
    media_type="audio",
    embedder_name=ctx.embedder,
)
ctx.threshold = threshold
# weights is a state_dict-shaped dict; load it onto a model and attach.

print(f"Restored detector: threshold={ctx.threshold:.3f}")
```

The library walked the origins, called the `server_folder`
`MediaSource` to resolve each one to a `Path`, ran CLAP on each file,
rebuilt `(X, y)`, and retrained. If you move the audio folder somewhere
else, you'd update the origin's `path` param (or implement a custom
`MediaSource`).

## Step 7: Score a held-out folder

Apply the reloaded detector to a different folder of audio:

```python
import torch
import numpy as np

held_out: dict[int, dict] = {}
load_dataset_from_folder(
    folder_path=Path("/data/new-audio"),
    media_type="audio",
    medias=held_out,
)

ids = list(held_out.keys())
embeddings = np.stack([held_out[i]["embedding"] for i in ids]).astype(np.float32)
with torch.no_grad():
    logits = ctx.model(torch.from_numpy(embeddings)).squeeze(1).numpy()
scores = 1.0 / (1.0 + np.exp(-logits))

hits = [(held_out[cid]["media_path"], float(s))
        for cid, s in zip(ids, scores) if s >= ctx.threshold]
hits.sort(key=lambda h: h[1], reverse=True)
print(f"{len(hits)} bark candidates in {len(ids)} held-out items.")
for path, score in hits[:10]:
    print(f"  {score:0.3f}  {path}")
```

This is what an autodetect job does - the CLI wraps the same calls in
[`vtscore.cli.autodetect_main`](../packages/cli.md).

## Step 8: Evaluate the detector

If you have ground-truth labels for the held-out set, run the eval
runner to get precision / recall / AP:

```python
from vtscore.eval import eval_learned_sort

# Build a ground-truth label dict for held_out: cid → "good" | "bad"
ground_truth = {cid: ("good" if "dog" in m["filename"] else "bad")
                for cid, m in held_out.items()}

# Run learned-sort evaluation: trains on a slice, scores the rest, repeats.
metrics_list = eval_learned_sort(held_out, ground_truth, n_folds=5)
for i, metrics in enumerate(metrics_list):
    print(f"Fold {i}: AP={metrics.average_precision:.3f}, "
          f"F1={metrics.f1:.3f}, P@10={metrics.precision_at_10:.3f}")
```

`eval_learned_sort` is in `vtscore/eval/runner.py`; it splits the data
into `n_folds` folds, trains a fresh MLP per fold, and computes
classification + ranking metrics. See
[packages/eval.md](../packages/eval.md) for the full evaluation API.

## What you built

By the end of the tutorial:

- **`/tmp/vtscore-tutorial/detectors/dog-barks.json`** - your detector,
  six origins, no weights. ~1 KB.
- **`/tmp/vtscore-tutorial/models/`** - cached LAION-CLAP weights. Reused
  by every future detector with `embedder="audio_clap"`. ~600 MB.
- **`ctx.model`** + **`ctx.threshold`** - in-memory trained MLP, ready
  to score anything.

The same shape generalises to:

- **Images** (`from vtscore.media import image`, SigLIP, 768-D).
- **Text** (`from vtscore.media import text`, E5, 768-D).
- **Video** (`from vtscore.media import video`, X-CLIP, 768-D).
- **Documents** (`from vtscore.media import document` plus a converter).

Mix media types by combining datasets via the
[`combine_datasets` importer](../packages/datasets.md#importers).

## Where to next

- [packages/detectors.md](../packages/detectors.md) - deeper on the
  detector lifecycle (cross-dataset labels, labelset sync, region boxes).
- [packages/eval.md](../packages/eval.md) - the full evaluation framework
  (voting iterations, label curves, plotting).
- [extending/](../extending/) - write your own media type, embedder,
  importer, or exporter.
- [integration.md](../integration.md) - embed `vtscore` in your own
  application.
