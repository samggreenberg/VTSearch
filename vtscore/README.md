# vtscore

The Flask-free, app-free core of [VTSearch](https://github.com/samggreenberg/vtsearch).
A reusable Python library for trainable media search: dataset origins,
MediaSources, clippers, embedders, MLP/detector training and scoring, and
evaluation. The [`vtsearch`](https://pypi.org/project/vtsearch/) Flask + Angular
application wraps this library with the HTTP / SPA / settings layer; everything
described here works without it.

## Status

`vtscore` ships from the same repository as `vtsearch`. The library was carved
out of the monolith in phases (see [`docs/plans/extract-library.md`](../docs/plans/extract-library.md));
its public surface is documented in [`docs/vtscore-api.md`](../docs/vtscore-api.md).

The package version starts at **0.1.0** and is tracked manually in
`vtscore/__init__.py`. There is no auto-bump on commit — the version moves only
when a release is cut. See [CHANGELOG.md](CHANGELOG.md) for what's in each
release.

## Install

`vtscore` is shipped today from the `vtsearch` distribution (one repo, both
packages). Install in editable mode for development:

```bash
git clone https://github.com/samggreenberg/vtsearch
cd vtsearch
bash scripts/install-cpu.sh   # or scripts/install-gpu.sh for CUDA
```

A standalone `vtscore` PyPI distribution is on the roadmap (see Phase 9 of the
extract-library plan); for now, install the repo and import `vtscore` directly.

## Quickstart

The library exports three core flows: **load a dataset**, **train a detector
from votes**, **score media**. The example below stitches them together end to
end.

### 1. Load a dataset from a folder

```python
from pathlib import Path

from vtscore.media import audio  # noqa: F401 — registers the audio MediaType
from vtscore.datasets.loader import load_dataset_from_folder

medias: dict[int, dict] = {}
load_dataset_from_folder(
    folder_path=Path("/path/to/audio/folder"),
    media_type="audio",
    medias=medias,
    recursive=True,
)
print(f"Loaded {len(medias)} media items")
```

`load_dataset_from_folder` walks the folder, runs the default embedder for the
media type (LAION-CLAP for audio, SigLIP for images, X-CLIP for video, E5 for
text), and populates `medias` in place. Each entry carries an `origin`, a
content hash, and an `embedding` ready for training or sorting.

### 2. Train a detector from labels

If you already have label assignments (good / bad) for some media — for
example, from a previous interactive session — train an MLP detector directly:

```python
import numpy as np
import torch

from vtscore.training import train_model, calculate_cross_calibration_threshold

# Collect (embedding, label) pairs from your labelled medias.
X_list = [m["embedding"] for m in labelled_medias]      # list of np.ndarray
y_list = [1.0 if m["label"] == "good" else 0.0 for m in labelled_medias]

X = torch.from_numpy(np.stack(X_list).astype(np.float32))
y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)

model = train_model(X, y, input_dim=X.shape[1])
threshold = calculate_cross_calibration_threshold(
    X_list, y_list, input_dim=X.shape[1], inclusion_value=0,
)
```

The model is a small `torch.nn.Sequential` with one hidden layer; the hidden
width is auto-sized from the training-set size. `train_model` is deterministic
given a fixed `seed` (default 42) and is safe to call from multiple threads.

### 3. Score a folder against the trained detector

```python
import numpy as np
import torch

# Load a fresh folder you want to score.
target_medias: dict[int, dict] = {}
load_dataset_from_folder(
    folder_path=Path("/path/to/score"),
    media_type="audio",
    medias=target_medias,
)

# Stack embeddings, run the model, threshold.
embeddings = np.stack([m["embedding"] for m in target_medias.values()]).astype(np.float32)
with torch.no_grad():
    logits = model(torch.from_numpy(embeddings)).squeeze(1).numpy()
scores = 1.0 / (1.0 + np.exp(-logits))   # sigmoid

hits = [
    (m["media_path"], float(s))
    for m, s in zip(target_medias.values(), scores)
    if s >= threshold
]
hits.sort(key=lambda h: h[1], reverse=True)
for path, score in hits[:10]:
    print(f"{score:0.3f}  {path}")
```

That's the full library flow. Everything else in `vtscore` — diversity-aware
sampling, vote-aware online training, multi-fold evaluation, region-similarity
scoring — builds on these primitives.

## Public API

The full inventory of names `vtscore` exports lives at
[`docs/vtscore-api.md`](../docs/vtscore-api.md). The top-level packages are:

| Package | Purpose |
|---------|---------|
| `vtscore.datasets` | Dataset loaders, importers, origins, labelsets, media sources |
| `vtscore.media` | Per-format `MediaType` plugins (audio, image, text, video, document) plus embedders and clippers |
| `vtscore.embedding` | Embedder façade, torch runtime helpers, cached `(N, D)` matrix |
| `vtscore.training` | Generic MLP / threshold / SVM / region-similarity primitives |
| `vtscore.detectors` | Detector lifecycle, training pipeline, origin resolution, labelset sync |
| `vtscore.eval` | Offline evaluation: text-sort, learned-sort, voting iterations |
| `vtscore.converters` | Cross-format converters (audio→spectrogram, video→keyframes, ASR, OCR) |
| `vtscore.exporters` | Results exporters (JSON, CSV, webhook, email) |
| `vtscore.labels` | Label importers and labelset sync sources |
| `vtscore.plugins` | `PluginRegistry`, `PluginBase`, sentinel-based discovery |
| `vtscore.concurrency` | Async job manager, memory budget, progress tracking |
| `vtscore.state` | `DatasetContext`, `DetectorContext` (no Flask) |
| `vtscore.sync` | `SyncSource[L,S]` ABC |
| `vtscore.security` | Path / URL validation, safe pickle loader |
| `vtscore.config` | `CoreConfig` dataclass + environment-driven constants |

## Plugin discovery

`vtscore` plugins (importers, exporters, label importers, labelset sources,
media types, embedders, clippers, converters) are auto-discovered at import
time via module sentinels. Third-party packages can register plugins without
monkey-patching by declaring entry points under the `vtscore.<family>` groups:

```toml
[project.entry-points."vtscore.importers"]
my_importer = "my_package.importer:MyImporter"
```

See [`docs/EXTENDING.md`](../docs/EXTENDING.md) and the per-family extension
guides for the full plugin-authoring story.

## License

Same as the parent `vtsearch` project — see the repository's `LICENSE` file.
