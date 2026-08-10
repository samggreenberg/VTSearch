# vtscore

The Flask-free, app-free core of [VTSearch](https://github.com/samggreenberg/vtsearch).
A reusable Python library for trainable media search: dataset origins,
MediaSources, clippers, embedders, MLP / detector training and scoring,
and evaluation. The companion `vtsearch` Flask + Angular application wraps
this library with the HTTP / SPA / settings layer; everything described
here works without it.

## Documentation

Comprehensive developer documentation lives under [`docs/`](docs/):

- **[Quickstart](docs/quickstart.md)** - load a folder, train a detector, score new media. Start here.
- **[Architecture](docs/architecture.md)** - system overview, the seven seams between vtscore and vtsearch, the resolution chain for "active context".
- **[Concepts](docs/concepts.md)** - `Media`, `Origin`, `LabelSet`, `Embedding`, `Context`, the MLP detector. The vocabulary every other doc assumes.
- **[Package reference](docs/README.md#package-reference)** - one deep-dive guide per subpackage.
- **[Extending vtscore](docs/extending/README.md)** - eleven plugin families with authoring guides for each.

For the canonical inventory of every name vtscore exports, see
[`docs/vtscore-api.md`](../docs/vtscore-api.md) at the repo root (a
docstring-only API contract sketch).

## Install

`vtscore` ships from the same repository as `vtsearch`. Install in
editable mode for development:

```bash
git clone https://github.com/samggreenberg/vtsearch
cd vtsearch
bash scripts/install.sh   # auto-detects CPU vs GPU
```

A standalone `vtscore` PyPI distribution is deferred until a real
external consumer asks for it. For now, install the repo and import
`vtscore` directly.

## Quickstart (90 seconds)

The library exports three core flows: **load a dataset**, **train a
detector from labels**, **score new media**. Here's the shortest possible
end-to-end script - see [docs/quickstart.md](docs/quickstart.md) for the
walkthrough with all of the details.

```python
from pathlib import Path
import numpy as np, torch

from vtscore.media import audio  # noqa: F401 - registers the audio MediaType
from vtscore.datasets.loader import load_dataset_from_folder
from vtscore.training import train_model, calculate_cross_calibration_threshold

# 1. Load a folder.
medias: dict[int, dict] = {}
load_dataset_from_folder(Path("/path/to/audio"), media_type="audio",
                        medias=medias, recursive=True)

# 2. Train a detector.
X_list = [m["embedding"] for m in labelled_medias]
y_list = [1.0 if m["label"] == "good" else 0.0 for m in labelled_medias]
X = torch.from_numpy(np.stack(X_list).astype(np.float32))
y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
model = train_model(X, y, input_dim=X.shape[1])
threshold = calculate_cross_calibration_threshold(
    X_list, y_list, input_dim=X.shape[1], inclusion_value=0,
)

# 3. Score new media (load another folder, embed, forward-pass, rank).
```

## Public surface

The 17 subpackages and what they do:

| Package | Purpose |
|---------|---------|
| `vtscore.config` | `CoreConfig` dataclass + environment-driven constants |
| `vtscore.media` | `MediaType` plugins (audio, image, text, video, document) + embedders + clippers |
| `vtscore.embedding` | Embedder façade, torch runtime, cached `(N, D)` matrix |
| `vtscore.datasets` | Origins, labelsets, loaders, importers, media sources |
| `vtscore.training` | MLP / threshold / SVM / region-similarity primitives |
| `vtscore.detectors` | Detector lifecycle: train, store, score, labelset sync |
| `vtscore.eval` | Offline evaluation (text-sort, learned-sort, voting iterations) |
| `vtscore.converters` | Cross-format converters (audio→spectrogram, ASR, OCR, …) |
| `vtscore.exporters` | Results exporters (JSON, CSV, webhook, email); JSON/CSV/gui support CLI streaming for sources larger than RAM |
| `vtscore.labels` | Label importers + labelset sync sources |
| `vtscore.plugins` | `PluginRegistry`, sentinel discovery, `importlib.metadata` hooks |
| `vtscore.concurrency` | Async jobs, memory budget, long-running progress trackers |
| `vtscore.state` | `DatasetContext`, `DetectorContext` (no Flask) |
| `vtscore.sync` | `SyncSource[L,S]` ABC |
| `vtscore.security` | Path / URL validation, safe pickle loader |
| `vtscore.utils` | `build_media_hit`, synthetic-media generators |
| `vtscore.cli` | Flask-free CLI entry points (autodetect, pipeline, progress) |

## Plugin discovery

`vtscore` plugins (importers, exporters, label importers, labelset
sources, media types, embedders, clippers, converters, media sources)
are auto-discovered at import time via module sentinels. Third-party
packages register plugins without monkey-patching by declaring entry
points under the `vtscore.<family>` groups:

```toml
[project.entry-points."vtscore.importers"]
my_importer = "my_package.importer:MyImporter"
```

See [docs/extending/](docs/extending/) for per-family authoring guides.

## Conventions

- **No persisted vectors or MLP weights.** Embeddings and trained models
  live in-memory only. Origins are the canonical persisted form; the
  library re-derives `origin → file → embedding → MLP` on demand. The
  single exception is dataset pickle files, which are by design a
  snapshot of media + their embeddings.
- **No hardcoded `data/` paths.** Every reference routes through
  `vtscore.config.DATA_DIR` (honouring `$VTSEARCH_DATA_DIR`), which is
  snapshotted into `CoreConfig.data_dir`.
- **No Flask, no `vtsearch.settings` imports** anywhere in vtscore.
  Verified by `./run-tests.sh vtscore-clean`.

See [docs/architecture.md](docs/architecture.md) for the full set of
architectural invariants.

## Versioning

`vtscore.__version__` is independent semver, bumped manually in
`vtscore/__init__.py` on each release. The companion `vtsearch` package
uses a git-derived timestamp instead. See [`CHANGELOG.md`](CHANGELOG.md)
for per-release notes.

## License

Apache License 2.0, same as the parent `vtsearch` project - see the
repository's [`LICENSE`](../LICENSE) file.

Two carve-outs matter if you are embedding `vtscore` in your own product:

- **Model weights carry their own terms.** `vtscore` downloads embedding
  models at runtime rather than vendoring them, and each publisher licenses
  its own weights. Some are noncommercial (EUPE, under the FAIR
  Noncommercial Research License) or gated (DINOv3). Embedders with a
  restriction expose it through their descriptor's `license_notice` field.
- **Two dependencies are AGPL-3.0, and are skippable**: `ultralytics` (image
  extractor and clipper) and `PyMuPDF` (PDF importer and document
  converters). A default install includes both. The Apache-2.0 grant on
  `vtscore` itself is unaffected, but AGPL terms may attach to a combined
  work you distribute or operate as a service — so if you are shipping a
  closed product on top of `vtscore`, install without them:
  `VTSEARCH_NO_AGPL=1 bash scripts/install.sh`, or
  `pip install -r requirements/base-no-agpl.txt`. Both packages live in the
  `agpl` extra, which every default install path requests; dropping it
  leaves a permissively licensed install in which those four features raise
  an actionable error instead of running (see
  `vtscore/utils/optional_deps.py`).

See [`NOTICE`](../NOTICE) for the full statement.
