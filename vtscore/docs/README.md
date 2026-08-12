# `vtscore` Documentation

The `vtscore` library is the Flask-free, reusable core of
[VTSearch](https://github.com/samggreenberg/vtsearch). It provides everything
needed to load a dataset, embed media, train a detector from labels,
score new media, and evaluate results - without any web framework, settings
system, or UI dependency. The companion `vtsearch` Flask + Angular app wraps
this library with the HTTP / SPA / settings layer.

This documentation is **for developers** - people writing scripts on top of
`vtscore`, embedding it inside other applications, or shipping plugins for it.
For end users of the VTSearch web app, see the
[user guide](../../docs/user/USER_GUIDE.md) instead.

## Start here

- [Quickstart](quickstart.md) - load a folder, train a detector, score new media. ~15 minute read.
- [Architecture](architecture.md) - system overview, the seven seams between vtscore and vtsearch, the resolution chain for "active context".
- [Concepts](concepts.md) - `Media`, `Origin`, `LabelSet`, `Embedding`, `Context`, the linear-head detector. The vocabulary every other doc assumes.

## Package reference

One guide per public subpackage. Each covers the package's purpose, its public
surface, worked examples, and gotchas.

| Package | Purpose | Doc |
|---------|---------|-----|
| `vtscore.config` | `CoreConfig` dataclass + environment-driven constants | [packages/config.md](packages/config.md) |
| `vtscore.media` | `MediaType`, embedder + clipper ABCs, processor ABCs | [packages/media.md](packages/media.md) |
| `vtscore.embedding` | Embedder loaders, torch runtime, cached `(N, D)` matrix | [packages/embedding.md](packages/embedding.md) |
| `vtscore.datasets` | Origins, labelsets, loaders, importers, media sources | [packages/datasets.md](packages/datasets.md) |
| `vtscore.datasource_importers` | Single-item importers for exemplar media | [packages/datasource-importers.md](packages/datasource-importers.md) |
| `vtscore.training` | Head (linear / MLP) / threshold / SVM / region-similarity primitives | [packages/training.md](packages/training.md) |
| `vtscore.detectors` | Detector lifecycle: train, store, score, labelset sync | [packages/detectors.md](packages/detectors.md) |
| `vtscore.eval` | Offline evaluation: text-sort, learned-sort, voting iterations | [packages/eval.md](packages/eval.md) |
| `vtscore.converters` | Cross-format converters (spectrogram, OCR, ASR, keyframes) | [packages/converters.md](packages/converters.md) |
| `vtscore.exporters` | Results exporters (JSON, CSV, webhook, email) | [packages/exporters.md](packages/exporters.md) |
| `vtscore.labels` | Label importers + bidirectional labelset sources | [packages/labels.md](packages/labels.md) |
| `vtscore.plugins` | `PluginRegistry`, sentinel-based discovery, entry-points | [packages/plugins.md](packages/plugins.md) |
| `vtscore.state` | `DatasetContext`, `DetectorContext`, vote / click ops | [packages/state.md](packages/state.md) |
| `vtscore.sync` | `SyncSource[L,S]` generic ABC | [packages/sync.md](packages/sync.md) |
| `vtscore.concurrency` | Async jobs, memory budget, long-running progress | [packages/concurrency.md](packages/concurrency.md) |
| `vtscore.security` | Path / URL validation, allowlist pickle loader | [packages/security.md](packages/security.md) |
| `vtscore.projection` | VTSBrowse: UMAP layout, hex-tile pyramid, region signposts | [packages/projection.md](packages/projection.md) |
| `vtscore.timing` | Measured per-step cost model behind every progress bar | [packages/timing.md](packages/timing.md) |
| `vtscore.utils` | Hit dicts, content hashes, score sanitisation, synthetic media | [packages/utils.md](packages/utils.md) |
| `vtscore.cli` | Flask-free CLI entry points (autodetect, pipeline, progress) | [packages/cli.md](packages/cli.md) |
| `vtscore.io`, `.gpu_backends`, `.single_instance` | Top-level runtime modules: file I/O, cuML routing, the port lock | [packages/runtime-modules.md](packages/runtime-modules.md) |

## Extending vtscore

The plugin families (see the generated inventory in
[concepts.md § Plugin](concepts.md#8-plugin) for the authoritative list)
share a common registry-based architecture. Subclass
the relevant base class, expose a sentinel attribute, drop the module in the
right directory (or register an `importlib.metadata` entry point), and the
library discovers it automatically. See:

- [Plugin authoring overview](extending/README.md) - registries, sentinels, entry-point groups.
- [Dataset importers](extending/dataset-importers.md)
- [Media types](extending/media-types.md)
- [Embedders](extending/embedders.md)
- [Clippers](extending/clippers.md)
- [Media converters](extending/converters.md)
- [Results exporters](extending/results-exporters.md)
- [Label importers](extending/label-importers.md)
- [Labelset sources](extending/labelset-sources.md)

## Conventions

These rules hold across every package - internalise them once and the rest of
the docs make sense:

- **No persisted vectors or model weights.** Embeddings and trained models are
  in-memory artefacts only. The on-disk persistence is `Origin` records
  inside `LabeledElement`s inside a detector's `LabelSet` JSON. On every load,
  the library re-derives `origin → file → embedding → head` from those
  origins. The single sanctioned exception is dataset pickle files, which
  are by design a snapshot of media plus their embeddings.
- **No hardcoded `data/` paths.** Every filesystem reference goes through
  `vtscore.config.DATA_DIR` (which honours `$VTSEARCH_DATA_DIR`) or is
  snapshotted into `CoreConfig.data_dir`. Library consumers can point the
  whole library at any directory by setting the env var or constructing a
  `CoreConfig` with a custom `data_dir`.
- **No Flask, no `vtsearch.settings` imports.** Verified by
  `./run-tests.sh vtscore-clean`, which installs a meta-path import hook that
  refuses `flask` / `werkzeug` / `flask_smorest` before pytest collects
  `tests_lib/`. Library code reads configuration from `CoreConfig` or
  context fields; it never reaches across the boundary.
- **Votes are `dict[int, None]`, not sets.** `votes[cid] = None` adds a
  vote; `del votes[cid]` removes one. The dict shape preserves insertion
  order and serialises consistently.
- **Library is multi-context.** There is no global "active" dataset or
  detector. Every operation resolves a context via the chain: explicit
  `override_*_context()` → installed resolver hook → thread-local → `None`.
  See [Architecture](architecture.md) for the full resolution rules.

## API contract reference

The [package reference](#package-reference) above is the authoritative
inventory of the public surface: one guide per subpackage, each listing the
symbols that package exports and what they guarantee. The guides group
symbols by intent rather than by import path, so pair them with
[Architecture § Import paths](architecture.md#import-paths-read-before-copy-pasting),
which records the real dotted path for every package - including the ones
whose `__init__.py` re-exports nothing.

## Versioning

`vtscore.__version__` is independent semver, bumped manually in
`vtscore/__init__.py` on each release. The companion `vtsearch` package
uses a git-derived timestamp instead, since every commit on `dev` is
effectively a new app release. See the top-level
[`CHANGELOG.md`](../CHANGELOG.md) for per-release notes.

## Doc conventions

Two rules keep this doc set from drifting away from the code, both
enforced by `scripts/check-vtscore-docs.py` (a `./run-tests.sh` gate):

- **Every top-level module and sub-package of `vtscore/` has a
  `packages/` doc, and that doc's Contents table names every module in
  it.** Add a module, and the gate fails until the table lists it. The
  tables are the inventory; write them from the tree, not from memory.
- **Never cite a line number.** A `path.py` plus a colon plus a number
  is wrong by the next edit - in practice every such anchor in this doc
  set had rotted, most by hundreds of lines (which is why the gate
  rejects the pattern outright, including in this sentence). Reference
  the module and the symbol instead
  (``` `pool_box_from_media` in `vtscore/embedding/matrix.py` ```): it
  is stable under any edit that doesn't move the symbol, and it is
  greppable, which a line number never was.

The same reasoning rules out `(~745 lines)`-style size annotations and
absolute machine paths in link text. If a fact about the code will be
wrong after somebody else's unrelated commit, don't write it down.

## Contributing

The library lives in the same git repository as the app
(`samggreenberg/vtsearch`). Run `./run-tests.sh` from the repo root for the
full test suite, or `./run-tests.sh vtscore-clean` to run only the
library-tier tests (`tests_lib/`) under a Flask-blocking import hook -
useful when verifying that a change keeps the library import-clean. New
library code goes under `vtscore/`; new library tests go under
`tests_lib/`.
