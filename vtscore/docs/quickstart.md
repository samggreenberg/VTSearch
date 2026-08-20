# Quickstart

This guide takes about 15 minutes. You'll load a folder of media, train a
small detector from a handful of labels, and score a fresh folder against
it - using only `vtscore` (no Flask, no Angular).

Five short examples follow, in order of increasing depth:

1. [Set up `CoreConfig`](#1-set-up-coreconfig)
2. [Load a folder](#2-load-a-folder-of-audio-files)
3. [Train a detector from labels](#3-train-a-detector-from-labels)
4. [Score a fresh folder](#4-score-a-fresh-folder)
5. [Persist and reload a detector](#5-persist-and-reload-a-detector)
6. [Run a text query](#6-run-a-text-query-no-training-needed)

If you haven't read [concepts.md](concepts.md), do that first - the
vocabulary (`Media`, `Origin`, `LabelSet`, `Embedding`) is assumed below.

> Every snippet on this page is executed by
> `tests_lib/integration/test_docs_quickstart.py` against a synthetic audio
> dataset, so a change that breaks one of them breaks the test suite too. If
> you edit this page, edit that test in the same commit.

## 0. Install

```bash
git clone https://github.com/samggreenberg/vtsearch
cd vtsearch
bash scripts/install.sh   # auto-detects CPU vs GPU
```

The first time you import an embedder, it downloads its model weights
into `data/models/` (~500 MB per backbone, cached for next time). Set
`$VTSEARCH_DATA_DIR` to redirect that cache somewhere else.

## 1. Set up `CoreConfig`

Library-only consumers construct `CoreConfig` directly. The app installs a
builder that reads from `vtsearch.settings`; you don't need one.

```python
from pathlib import Path
from vtscore.config import CoreConfig, register_core_config_builder

DATA = Path("/tmp/vtscore-quickstart")
DATA.mkdir(exist_ok=True)

def _build(settings_path=None) -> CoreConfig:
    return CoreConfig(
        data_dir=DATA,
        saved_datasets_dir=DATA / "datasets",
        detectors_dir=DATA / "detectors",
        max_concurrent_dataset_downloads=1,
        max_concurrent_dataset_embeddings=1,
        autofind_detectors=(),
        dataset_max_age_days=None,
        calibrate_count=2,
        calibration_fraction=0.5,
        enrich_descriptions=False,
        autopilot_goal_diversity=8,
        inclusion=0,
    )

register_core_config_builder(_build)
```

Now `CoreConfig.from_settings()` works anywhere in the library. Two details
that bite:

- **The builder takes an argument.** `CoreConfig.from_settings(settings_path)`
  passes it straight through, so a zero-argument builder raises `TypeError`
  the first time a caller supplies a path. Accept it and ignore it if you
  have nowhere to put it.
- **Every field listed above is required.** `CoreConfig` is a frozen
  dataclass with defaults only for the four newest optional fields
  (`autofind_exporter`, `autofind_exporter_field_values`,
  `signpost_captioner`, `signpost_vocab`) - omitting any of the others is a
  `TypeError`. See [packages/config.md](packages/config.md) for what each one
  means.

## 2. Load a folder of audio files

Loading is **two stages**, and the split matters: `load_dataset_from_folder`
populates the `medias` dict you pass in but deliberately does *not* call the
embedder, and `embed_missing` fills in the vectors afterwards. An importer
that emitted its own pre-computed vectors would skip the second call.

```python
from vtscore.datasets.loader import load_dataset_from_folder
from vtscore.datasets.origin import Origin
from vtscore.datasets.stages.embedding import embed_missing
from vtscore.embedding.media_vectors import media_embedding
from vtscore.media import audio  # noqa: F401 - see the note on auto-discovery below

folder = Path("/data/audio-corpus")

# The origin is how a label finds its file again later; pass one at load
# time or the media dicts come out with `origin=None` and a saved detector
# has nothing to re-resolve.
origin = Origin(
    importer="server_folder",
    params={"path": str(folder), "media_type": "audio"},
).to_dict()

medias: dict[int, dict] = {}
load_dataset_from_folder(
    folder_path=folder,
    media_type="audio",
    medias=medias,
    recursive=True,
    origin=origin,
)
embed_missing(medias)

first = next(iter(medias.values()))
print(f"Loaded {len(medias)} audio items.")
print(f"Embedder: {first['embedder']}")
print(f"First embedding shape: {media_embedding(first).shape}")
# Loaded 532 audio items.
# Embedder: clap_general
# First embedding shape: (512,)
```

Every built-in media type, embedder, and clipper is **auto-discovered when
`vtscore.media` is first imported** - the package scans its own
sub-packages for `MEDIA_TYPE` / `EMBEDDER` / `CLIPPERS` sentinels. So the
`from vtscore.media import audio` line above is documentation of intent, not
a requirement: `vtscore.datasets.loader` pulls the registry in transitively.
An unrecognised name is a `ValueError: Invalid media type: <name>`. The six
shipped types are `audio`, `image`, `text`, `video`, `document`, and `face`.

What's now in each `medias[cid]`:

```python
medias[1]
# {
#   "id": 1,
#   "media_type": "audio",
#   "embedder": "clap_general",
#   "md5": "5d41402abc4b2a76b9719d911017c592",
#   "embeddings": {"clap_general": np.ndarray(shape=(512,), dtype=float32)},
#   "filename": "bark/poodle.wav",
#   "origin": {"importer": "server_folder",
#              "params": {"path": "/data/audio-corpus", "media_type": "audio"}},
#   "origin_name": "bark/poodle.wav",
#   "media_path": "/data/audio-corpus/bark/poodle.wav",
#   "duration": 2.41,
#   …
# }
```

Note `embeddings` (plural): vectors live in a dict keyed by embedder name so
several bound embedders can coexist on one media. There is no
`media["embedding"]`. Always read through
`media_embedding(media)` - it resolves the media's primary embedder - or
`media_embedding(media, "clap_general")` for a specific one.

## 3. Train a detector from labels

Suppose you have a CSV that says which of your loaded items are
"barks" and which aren't. Build `(X, y)`, then train.

```python
import numpy as np
import torch

from vtscore.training import train_model, calculate_cross_calibration_threshold
from vtscore.training.mlp import LINEAR_SVM_HEAD

# Map filenames to labels.
labels = {
    "bark/poodle.wav": "good",
    "bark/labrador.wav": "good",
    "bark/beagle.wav": "good",
    "music/track-01.mp3": "bad",
    "speech/news-clip.wav": "bad",
    "speech/interview.wav": "bad",
}

X_list: list[np.ndarray] = []
y_list: list[float] = []
for m in medias.values():
    label = labels.get(m["filename"])
    if label is None:
        continue
    X_list.append(media_embedding(m))
    y_list.append(1.0 if label == "good" else 0.0)

X = torch.from_numpy(np.stack(X_list).astype(np.float32))
y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)

# LINEAR_SVM_HEAD is the production head: a single Linear(D, 1) fitted to the
# maximum-margin boundary. Omitting hidden_dim auto-sizes an MLP - see docs/ML.md.
model = train_model(X, y, input_dim=X.shape[1], hidden_dim=LINEAR_SVM_HEAD)
threshold = calculate_cross_calibration_threshold(
    X_list, y_list, input_dim=X.shape[1], inclusion_value=0,
    hidden_dim=LINEAR_SVM_HEAD,
)
print(f"Trained detector; calibrated threshold = {threshold:.3f}")
# Trained detector; calibrated threshold = 0.412
```

`train_model` is deterministic given the default seed (42). Six labels is
enough for the linear head to fit - it has `D + 1` parameters and no hidden
layer, which is exactly why a linear head is the production head when positives
are sparse.

For a more idiomatic detector lifecycle (with persistence), see
[example 5](#5-persist-and-reload-a-detector) below.

## 4. Score a fresh folder

Load a different folder, stack its embeddings, run the model, threshold,
and rank.

```python
target_folder = Path("/data/new-audio")
target_medias: dict[int, dict] = {}
load_dataset_from_folder(
    folder_path=target_folder,
    media_type="audio",
    medias=target_medias,
    origin=Origin(
        importer="server_folder",
        params={"path": str(target_folder), "media_type": "audio"},
    ).to_dict(),
)
embed_missing(target_medias)

# Stack embeddings into one matrix.
ids = list(target_medias.keys())
embeddings = np.stack([media_embedding(target_medias[i]) for i in ids]).astype(np.float32)

# Forward pass.
with torch.no_grad():
    logits = model(torch.from_numpy(embeddings)).squeeze(1).numpy()
scores = 1.0 / (1.0 + np.exp(-logits))   # sigmoid

# Rank and filter.
hits = sorted(
    [(target_medias[i]["media_path"], float(s)) for i, s in zip(ids, scores)],
    key=lambda h: h[1],
    reverse=True,
)
above_threshold = [(p, s) for p, s in hits if s >= threshold]
print(f"{len(above_threshold)} of {len(hits)} items scored above threshold.")
for path, score in hits[:10]:
    print(f"  {score:0.3f}  {path}")
```

For larger datasets, build the embedding matrix once on the
`DatasetContext` cache instead of stacking every time - see
[packages/embedding.md](packages/embedding.md#cached-embedding-matrix).

## 5. Persist and reload a detector

Detectors persist as JSON labelsets - never as model weights. On reload,
the library re-derives embeddings from the labelset's origins and
retrains the head.

```python
from vtscore.datasets.labelset import LabeledElement, LabelSet
from vtscore.detectors.store import load_detector, save_detector
from vtscore.detectors.training import train_detector_from_origins
from vtscore.state import DetectorContext, register_detector_context

# Build a LabelSet from the labelled medias.  LabelSet's first positional
# argument is the element list; there is no `labels=` keyword, and the
# attribute holding them is `.elements`.
elements = []
for m in medias.values():
    label = labels.get(m["filename"])
    if label is None:
        continue
    elements.append(LabeledElement(
        md5=m["md5"],
        label=label,
        origin_name=m["origin_name"],
        origin=m["origin"],
        filename=m["filename"],
    ))

labelset = LabelSet(elements)

# Save it. The on-disk path is CoreConfig.detectors_dir / "<slug>.json".
save_detector("barks", labelset, media_type="audio")

# Later, in a different process - reload and train.
ctx = DetectorContext(detector_id="barks", name="barks",
                      media_type="audio", embedder="clap_general")
register_detector_context(ctx)

# Build (good_origins, bad_origins) from the saved labelset's elements,
# then re-derive embeddings + retrain. Pass the same embedder the detector
# was trained with so the re-embedded vectors line up with the saved ones -
# otherwise the head trains on a mix of embedder outputs.
saved = LabelSet.from_dict(load_detector("barks")["labelset"])
good_origins = [
    {"origin": e.origin, "origin_name": e.origin_name,
     "filename": e.filename, "md5": e.md5}
    for e in saved.elements if e.label == "good"
]
bad_origins = [
    {"origin": e.origin, "origin_name": e.origin_name,
     "filename": e.filename, "md5": e.md5}
    for e in saved.elements if e.label == "bad"
]
weights, threshold = train_detector_from_origins(
    good_origins,
    bad_origins,
    inclusion=0,
    media_type="audio",
    embedder_name=ctx.embedder,
)
ctx.threshold = threshold
print(f"Restored detector with threshold {threshold:.3f}")
```

`weights` comes back as a state-dict-shaped dict of plain lists (or `None`
when too few origins re-resolved). Turn it into a live head with
`vtscore.training.mlp.build_model_from_weights(weights)` - it is a transport
format for one process's in-memory head, not a persistence format: nothing
in the detector lifecycle writes it to disk.

What's on disk after `save_detector`:

```json
{
  "name": "barks",
  "media_type": "audio",
  "labelset": {
    "labels": [
      {"md5": "…", "label": "good",
       "origin_name": "bark/poodle.wav",
       "filename": "bark/poodle.wav",
       "origin": {"importer": "server_folder", "params": {"path": "/data/audio-corpus", "media_type": "audio"}}},
      …
    ]
  }
}
```

No embeddings, no model weights - just origins. The library re-derives
everything else on load. This is the invariant described in
[architecture.md](architecture.md#the-no-persisted-vectors-rule).

## 6. Run a text query (no training needed)

For a quick search without training a detector, embed a text query into
the same vector space as your media and rank by cosine similarity.

```python
from vtscore.embedding.helpers import embed_text_query

query_emb = embed_text_query("dog barking", "audio")
# shape (512,), float32 - or None if this media type's embedder has no
# text tower (VideoMAE, DINOv2, AST, …).

# Build a cosine ranking against the live medias.
embeddings = np.stack([media_embedding(m) for m in medias.values()]).astype(np.float32)
cosines = (embeddings @ query_emb) / (
    np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-9
)
ranked = sorted(
    zip(medias.values(), cosines),
    key=lambda t: t[1],
    reverse=True,
)
for media, score in ranked[:10]:
    print(f"  {score:0.3f}  {media['filename']}")
```

`embed_text_query(text, media_type, enrich=False, embedder_name="")` is
LRU-cached per `(text, media_type, enrich, embedder_name)` - calling it twice
with the same arguments is free. Clear the cache with
`vtscore.embedding.clear_text_query_cache()` if you swap embedder
backbones mid-process.

## Where to go next

- **Adding new media types or embedders:**
  [extending/media-types.md](extending/media-types.md),
  [extending/embedders.md](extending/embedders.md).
- **Plugging into your own pipeline:** read
  [architecture.md](architecture.md) for the resolution chain and the
  three hooks the app installs (`register_core_config_builder`,
  `register_*_context_resolver`, `register_plugin_family`).
- **Bulk scoring without writing this script:** use the CLI -
  [packages/cli.md](packages/cli.md) shows the `autodetect` entry points.
- **Evaluating your detector:** see [packages/eval.md](packages/eval.md)
  for the offline evaluation runner (precision/recall, voting iterations).
