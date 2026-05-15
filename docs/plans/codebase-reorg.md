# Codebase Reorganization Plan

**Status:** In Progress (mid-sized refactors landed; harder reshapes
evaluated below)

The first round — `tests/fixtures/medias.py` move, test-suite bucketing,
shared embedder stubs, routes-by-domain, `utils/` split into focused
packages, `docker/`/`requirements/`/`scripts/` moved out of the repo
root, `PatchEmbedOutput` moved into `media/`, plans-doc cleanup — has
all landed on `dev`.

This doc now tracks the **harder, opinionated reshapes** that were
explicitly excluded from the first round. Each item has a verdict
(**Do**, **Maybe**, **Skip**) so the next session can pick up an item,
ship it, and tick it off. The verdicts are recommendations — push back
on any of them if the rationale doesn't hold up.

---

## 1. Split `vtsearch/models/` — **Do (in two steps)**

### Today

`vtsearch/models/` is 19 files, ~4,900 LOC, and conflates four concerns:

| Cluster | Files | LOC |
|---|---|---|
| Detector lifecycle (registry / store / training-glue / dataset-sync / resolver / label-restoration / label-sync / labelset-elements / labelset-training / media-seeding) | 10 | ~3,300 |
| Neural-net training (training, training_workflow, svm_training, region_similarity) | 4 | ~1,300 |
| Embedding helpers (embeddings, embedding_matrix, loader) | 3 | ~330 |
| Sampling (diversity_tree) | 1 | 311 |
| Labeling-session analyzer (labeling_progress) | 1 | 745 |

`detector_*`, `labelset_*`, `label_*`, `resolver.py`, and
`media_seeding.py` are tightly coupled with each other (they share
`DetectorContext` and the resolve→embed→train pipeline) and very loosely
coupled with `training.py` / `diversity_tree.py`. The directory name
also collides with the user-facing concept of "model" (detector), which
is a separate thing from the embedding/torch models that live in
`vtsearch/media/`.

### Why split

- The detector cluster is large enough (3.3k LOC, 10 files) to deserve
  its own package, parallel to `vtsearch/datasets/`. Most external
  imports already say `from vtsearch.models.detector_*`, so the move is
  basically a rename.
- `loader.py` is a thin façade over `vtsearch.media` embedder lookups —
  it doesn't belong next to neural-net training.
- The `models/` name confuses new contributors who expect torch
  `nn.Module`s; the actual torch models live under `vtsearch/media/`.

### Why not

- 40+ external import sites need updating (cheap in aggregate, but a
  big diff).
- `training.py` (882 LOC) mixes "train the detector MLP" with
  "calibrate thresholds", so a clean split needs a small internal cut
  before the package move.

### Proposed shape

```
vtsearch/
  detectors/                ← new package (replaces most of models/)
    registry.py             ← was models/detector_registry.py
    store.py                ← was models/detector_store.py
    training.py             ← was models/detector_training.py + the
                              detector-specific parts of models/training.py
    dataset_sync.py         ← was models/detector_dataset_sync.py
    resolver.py             ← was models/resolver.py
    label_restoration.py
    label_sync.py
    labelset_elements.py
    labelset_training.py
    media_seeding.py
  training/                 ← MLP/SVM training only
    mlp.py                  ← the non-detector half of models/training.py
    svm.py                  ← was models/svm_training.py
    workflow.py             ← was models/training_workflow.py
    region_similarity.py
    thresholds.py           ← calculate_safe_threshold / gmm / cross_calibration
  embedding/                ← embedder façades, not the embedders themselves
    helpers.py              ← was models/embeddings.py
    matrix.py               ← was models/embedding_matrix.py
    loader.py               ← was models/loader.py
```

`diversity_tree.py` is one file — move it into `vtsearch/state/` (it's
already a per-`DatasetContext` field) or leave it under
`vtsearch/sampling/` only if more sampling code arrives. **Do not**
create a one-file `sampling/` package today.

`labeling_progress.py` (745 LOC, formerly `progress.py`) was renamed
out of the way ahead of this split — the original task-#6 "merge with
`vtsearch/concurrency/progress.py`" turned out to be a misdiagnosis
(the two files share nothing but the historical name; one is
long-running-op infrastructure, the other is a per-step model cache
and stopping-condition analyzer). The renamed file lives next to the
detector cluster and should move into `vtsearch/detectors/` in step 1.

### Plan

1. **Step 1 — carve out `vtsearch/detectors/`.** Move the 10 detector-
   cluster files. Update `models/__init__.py` to re-export the moved
   names for one commit, run tests, then update all import sites and
   delete the re-exports. Single PR.
2. **Step 2 — split the remainder.** Cut `training.py` into
   `detectors/training.py` + `training/mlp.py` + `training/thresholds.py`,
   move `embeddings.py` / `embedding_matrix.py` / `loader.py` under
   `embedding/`, move `diversity_tree.py`, delete `vtsearch/models/`.

Run `./run-tests.sh` between steps. No behaviour change — pure moves.

---

## 2. Split the remaining 1000-LOC route files — **Maybe (one of three)**

### Today

After the routes-by-domain reorg, three files are still ≥800 LOC:

| File | LOC | Routes | Comments |
|---|---|---|---|
| `routes/datasets/crud.py` | 970 | 26 | Mixes media-type listings, dataset status/progress, importer staging, import/load, combine |
| `routes/detectors/store.py` | 900 | 13 | Detector CRUD + label CRUD + thumbnail/preview/vote |
| `routes/sorting.py` | 876 | 16 | Sort + learned-sort + votes + textsort + inclusion + safe-thresholds |

### Verdict per file

- **`datasets/crud.py` — split.** It already has four crisp seams:
  - `listings.py` — `/api/media-types`, `/api/embedders`, `/api/clippers`, `/api/converters`, `/api/dataset/importers`, `/api/dataset/all-importers` (small, read-only).
  - `status.py` — `/api/dataset/status`, `/api/dataset/progress`, `/api/dataset/loading-tasks`, `/api/dataset/cancel*`.
  - `staging.py` — `/api/dataset/stage-*`, `/api/dataset/staging`, `/api/dataset/available-files`, `/api/dataset/import/<importer>/options`, `/api/dataset/import/<importer>`, `/api/dataset/combine`.
  - `load.py` — `/api/dataset/load-*`, `/api/dataset/import-local-folder`.
  Each ends up ~250 LOC. The shared helpers (`_extract_clipper_params`,
  `_extract_importer_fields`, `_safe_relative_upload_path`) move into
  `routes/datasets/_helpers.py` next to `_shared.py`.
- **`detectors/store.py` — split.** ✅ Done — split into:
  - `crud.py` — detector CRUD (list/create/get/delete/rename/examples/combine).
  - `labels.py` — label CRUD + import + labels-detail + preview/thumbnail/vote.
  Each module owns its own Blueprint (`detectors_crud_bp` /
  `detectors_labels_bp`), registered directly on the app.
- **`sorting.py` — leave it.** 876 LOC across 16 closely-related routes
  (sort, learned-sort, votes, textsort, inclusion, safe-thresholds) that
  share `_cosine_sort`, `_get_embedder_for_loaded_data`,
  `_load_embedder_with_progress`. Splitting would scatter the helpers
  and produce four ~200-LOC files held together by imports. Not worth
  the churn.

### Risks

- Blueprint registration. Each new module needs to register routes on
  the existing `datasets_bp` / `detectors_bp` — fine, blueprints already
  support being assembled from multiple modules (the existing
  sub-packages do this). No URL changes.
- One PR per file (datasets first, detectors second), not a single
  monster diff.

---

## 3. Split `vtsearch/media/image/media_type.py` (1641 LOC) — **Done**

Audit produced clear seams: **84% of the file was demo-dataset code**,
not core `MediaType` logic. Breakdown of the original 1641 LOC:

| Slice | Lines | What |
|---|---|---|
| Core `MediaType` | ~120 | Identity properties, `display_metadata`, `loops`, `load_media_data`, `media_response` |
| Demo category constants | ~870 | `_PLACES365_*` raw text + parser, six `_DEMO_CATEGORIES_*` lists (Caltech-101/256, Oxford Flowers, Food-101, EuroSAT, Stanford Dogs, UCSF Documents) — pure data |
| `demo_datasets` property | ~303 | Builds 23 `DemoDataset` entries — no instance state used beyond the category constants |
| `load_demo_source` method | ~379 | Per-source download + embed dispatcher — only used `self.type_id` (literal `"image"`) and the category constants |

The demo code is split out into two private helper modules:

- `vtsearch/media/image/_demo_categories.py` (814 LOC) — Places365 raw
  text + `_parse_places365_categories` + the seven category lists,
  exposed as module-level `PLACES365_CATEGORIES`,
  `DEMO_CATEGORIES_CALTECH101`, etc.
- `vtsearch/media/image/_demo_sources.py` (716 LOC) —
  `build_demo_datasets()` returning the catalog and
  `load_demo_source()` as a module-level function (the latter takes
  `clips` and the embedder as arguments instead of using `self`).

`media_type.py` is now 173 LOC — `ImageMediaType` delegates
`demo_datasets` and `load_demo_source` to the helper module. No
behaviour change; the only external touch-ups were swapping
`scripts/run_hac_tree_sweep.py`'s `_PLACES365_CATEGORIES_LIST` import
and one test's `ImageMediaType._PLACES365_CATEGORIES` class-attribute
reference to read the new module-level constant.

`video/media_type.py` (851 LOC) and `audio/media_type.py` (696 LOC) are
smaller and were not audited as part of this step. Apply the same
audit pattern if either grows further.

---

## 4. Unify the plugin-discovery patterns — **Skip (for now)**

There are two real patterns, not three:

1. **`PluginRegistry` + sentinel scan** — used by every plugin family
   under `vtsearch/` (datasets/importers, datasets/sources, exporters,
   labels/importers, labels/sources, settings_io/{importers,exporters,
   sources}, converters). Each family declares a sentinel name
   (`IMPORTER`, `EXPORTER`, `LABEL_IMPORTER`, etc.), and
   `vtsearch.plugins` walks the package looking for modules that expose
   that sentinel.
2. **Media-type / embedder / clipper sentinel scan** in
   `vtsearch/media/__init__.py` — uses `MEDIA_TYPE`, `EMBEDDER`,
   `CLIPPERS` sentinels but is hand-written rather than going through
   `PluginRegistry`.

The "in-memory JSON catalogs" referenced in the old plan was a
mischaracterisation — that's the demo-dataset list, not a plugin
mechanism.

### Verdict

Folding `vtsearch/media/__init__.py`'s scan into `PluginRegistry` is
mechanical but disruptive — every media type, embedder, and clipper
would need an `__init_subclass__`-or-equivalent change, and the scan
logic in `media/__init__.py` includes media-specific quirks (clippers
are a list, embedders are per-media-type, requirements-*.txt files
co-locate with the package) that don't apply to other plugin families.

Cost: high (touches every embedder + clipper). Benefit: a few hundred
lines of dedup. Skip until a third plugin-discovery use case shows up
that would also benefit from unification.

---

## 5. Frontend reorganization — **Do (the dashboard split only)**

### Today

- `frontend/src/app/components/` — 17 flat folders, no `shared/`.
- `frontend/src/app/services/` — 29 flat services.
- 59 `.spec.ts` files, none of them run (no Karma; CLAUDE.md notes they
  must still typecheck).
- Outliers by size:
  - `dashboard.component.ts` — **1,464 LOC**
  - `label-view.component.ts` — 913 LOC
  - `find-view.component.ts` — 385 LOC

### Verdict per slice

- **`dashboard.component.ts` — split.** 1,464 LOC in a single component
  is the worst offender in the frontend. The dashboard has natural tab
  seams (datasets / detectors / processors / settings) that each map to
  a sub-component. Target: ≤400 LOC for the host component, the rest
  pushed into `dashboard/tabs/<tab>/`. Highest value of any frontend
  change.
- **`label-view.component.ts` — audit, then maybe split.** 913 LOC but
  it's one coherent screen with tightly coupled keyboard/mouse/vote
  state. May not split cleanly. Defer until after dashboard.
- **Global "by feature" reorg of `components/` and `services/` —
  skip.** Angular convention is by-type; the codebase is internally
  consistent; the cost (every import path changes) is high; the benefit
  (faster grep?) is marginal. Don't do this unless the team is
  expanding and onboarding pain is real.
- **Dead `.spec.ts` files — leave them.** They typecheck (per
  CLAUDE.md), they document intended behaviour, and deleting them
  forfeits future test-runner coverage at zero current benefit.

---

## 6. CLI package split (`cli.py` → `cli/`) — **Skip**

`vtsearch/cli.py` is 455 LOC and 18 functions — well under the
threshold where splitting helps. Revisit if it crosses 800 LOC or grows
a second `argparse` subcommand cluster. For now, leave it.

---

## 7. Collapsing `vtsearch/auth/` — **Skip (explicit)**

Restated from the previous version of this plan: auth is intentionally
a package because more login providers (Google, SAML, etc.) are
expected to land there following the same pattern used by
`labels/sources/` and `settings_io/sources/`. Do not collapse.

---

## Suggested order of operations

1. ~~Resolve the deferred `vtsearch/models/progress.py` ↔
   `vtsearch/concurrency/progress.py` merge from task #6.~~ ✅ Done
   in PR #1334 — turned out to be a misdiagnosis; resolved as a rename
   of `models/progress.py` to `models/labeling_progress.py`.
2. ~~Carve out `vtsearch/detectors/` (#1 step 1).~~ ✅ Done — the 10
   detector-cluster files plus `labeling_progress.py` now live under
   `vtsearch/detectors/`. `vtsearch/models/` keeps embedders, the torch
   model loader, neural-net training, and the diversity tree until #1
   step 2.
3. ~~Split `routes/datasets/crud.py` (#2).~~ ✅ Done — split into
   `listings.py`, `status.py`, `staging.py`, `load.py`, and a shared
   `_helpers.py`. Each new module owns its own Blueprint (registered
   directly on the app, matching `routes/detectors/`); the old
   `datasets_bp` re-export shim is gone.
4. ~~Finish the `vtsearch/models/` split (#1 step 2).~~ ✅ Done —
   `vtsearch/models/` is gone. Its contents now live under
   `vtsearch/training/` (MLP/SVM/thresholds/region-similarity),
   `vtsearch/embedding/` (helpers, matrix, loader),
   `vtsearch/detectors/` (the vote-aware `train_and_score`,
   `train_detector_from_origins`, `collect_media_origins`, plus the
   apply-and-retrain workflow), and `vtsearch/state/diversity_tree.py`.
5. ~~Split `routes/detectors/store.py` (#2).~~ ✅ Done — split into
   `crud.py` (detector CRUD: list/create/get/delete/rename/examples/combine)
   and `labels.py` (labelset save/import/labels-detail/preview/thumbnail/vote).
   Each new module owns its own Blueprint (`detectors_crud_bp` /
   `detectors_labels_bp`), registered directly on the app; the old
   `detectors_bp` name is gone.
6. ~~Audit `image/media_type.py` (#3).~~ ✅ Done — the audit showed
   demo-dataset code was 84% of the file, so it was split out in the
   same PR. `media_type.py` is now 173 LOC; the demo category
   constants live in `_demo_categories.py` and the
   `build_demo_datasets()` + `load_demo_source()` helpers live in
   `_demo_sources.py`.
7. Split `dashboard.component.ts` (#5).

Each step is its own PR. Don't batch — every move is a large mechanical
diff that's easier to review in isolation.
