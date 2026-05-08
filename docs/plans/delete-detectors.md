# Plan: Delete Detectors, Keep Only Trainable Models

> **Status: Proposed.** Companion to PR #1224 (right pane labelset-driven; CLI
> autorun trainable models). Captures the cleanup we agreed to do *after* that
> PR lands. Origins-as-source-of-truth, MLPs-in-RAM-only.

## Problem

VTSearch carries two parallel concepts for "a thing that classifies media":

1. **Detector** — `data/detectors/<name>.json`, in-memory `autorun_detectors[name]`. Carries (origins | weights | both). Read-only artifact format.
2. **Trainable model** — `data/trainable_models/<name>.json`, in-memory `DetectorContext`. Carries an editable origin-keyed `LabelSet`.

The detector concept exists because, historically, it was the share/export format. After PR #1224 the trainable model is itself origin-keyed and re-importable, so a *separate* detector concept no longer pulls its weight. Worse, several detector code paths violate the new
[no-persisted-vectors-or-MLPs principle](../../CLAUDE.md#no-persisted-vectors-or-mlps-critical):

- `GET /api/autorun-detectors/<name>/export` returns `{weights, threshold}` for the user to download — a serialized MLP.
- `weights_compat.normalize_detector_weights()` has a `fallback_weights` branch that reads weights out of a JSON file when origins fail to resolve.
- The model registry branches on `trainable: bool` and carries `detector_name` / `trainable_model_name` for the same kind of object.

## Goal

Single concept on disk: **trainable model**. Single in-memory cache: **`DetectorContext`**. Single resolve→embed→train pipeline: **`train_from_labelset`** (added in PR #1224).

After this work:

- `data/detectors/` is gone. Existing detector files are no longer read.
- `vtsearch/processors/importers/server_detector_file/` is gone.
- `weights_compat.py` is gone.
- The "weights" field disappears from every JSON-on-disk format.
- The model registry has no `trainable` flag — every entry is a trainable model.
- `autorun_detectors` (the in-memory dict) is gone; `/api/auto-detect` reads `det_ctx.model` directly.
- `autorun_processors` is gone; replaced by `autorun_trainable_models` (already added in PR #1224).

## Non-Goals

- **Backwards compatibility.** Per CLAUDE.md, breaking changes are fine. We do not provide a migration tool that rewrites old detector files into trainable models. Users with `data/detectors/*.json` need to recreate them through the labeling UI.
- **Preserving the browser-download share artifact.** The replacement is "download the trainable-model JSON" (origin-keyed, re-importable). No frozen-MLP file format survives.

## Current State

### On-disk artifacts

| Path | Contents | Written by | Read by |
|---|---|---|---|
| `data/detectors/<n>.json` | `{good_origins, bad_origins, inclusion, media_type, name}` | `POST /api/detector/export-server` (unused by frontend; tests only) | `server_detector_file` processor importer at startup; `import-pkl` route |
| `data/trainable_models/<n>.json` | `{name, media_type, examples, labelset}` | Trainable-model CRUD + sync on vote | Right pane, CLI, Find UI |

### In-memory state

- `autorun_detectors: dict[str, dict]` (`state_core.py:685`) — populated at startup from `autorun_processors` recipes via `ensure_autorun_processors_imported()`. Holds tiny MLP weights for `/api/auto-detect` to use without re-resolving.
- `DetectorContext.model` / `.threshold` / `.label_embeddings` — populated at trainable-model load via `train_from_labelset`. Used by `/api/find-label`, learned-sort, and CLI scoring.

Both are process-scoped, neither persists. Functionally they are the same cache, twice.

### Routes

`vtsearch/routes/detectors_*.py` exposes:
- `/api/autorun-detectors[*]` (CRUD over the in-memory dict)
- `/api/detector/export-server` (write origin-only file to disk; **no frontend caller**)
- `/api/detector/server-files[*]` (list + read disk detector files)
- `/api/auto-detect` (score active dataset against autorun detectors)
- `/api/extract`, `/api/auto-extract`, `/api/localize` (extractors / localizers — unrelated to detector-vs-trainable; out of scope for this cleanup)

`vtsearch/routes/trainable_models.py` and `vtsearch/routes/models_registry.py` are the trainable-model surface.

## Target State

### On-disk

Only `data/trainable_models/<n>.json` exists. `data/detectors/` is removed by the migration step (or just stops being read; users can delete the folder themselves).

### In-memory

Only `DetectorContext`. The `autorun_detectors` dict is deleted. `/api/auto-detect`'s loop iterates the model registry's "autorun-flagged" trainable models and uses each one's `det_ctx.model` directly.

### Routes

Detector routes fold into trainable-model / model-registry routes:

| Old route | Replacement |
|---|---|
| `GET /api/autorun-detectors` | `GET /api/models/registry` (filter `autorun=true`) |
| `POST /api/autorun-detectors` | `POST /api/trainable-models` + `PUT /api/models/registry/<id>/autorun` |
| `DELETE /api/autorun-detectors/<n>` | `DELETE /api/trainable-models/<n>` |
| `PUT /api/autorun-detectors/<n>/rename` | `PUT /api/trainable-models/<n>/rename` |
| `PUT /api/autorun-detectors/<n>/autodetect` | `PUT /api/models/registry/<id>/autorun` |
| `GET /api/autorun-detectors/<n>/export` | **Deleted.** Replacement: `GET /api/trainable-models/<n>` (origin-keyed JSON). |
| `GET /api/autorun-detectors/<n>/examples` | `GET /api/trainable-models/<n>` (already includes `examples`) |
| `PUT /api/autorun-detectors/<n>/examples` | `PUT /api/trainable-models/<n>/examples` |
| `POST /api/autorun-detectors/import-pkl` | **Deleted.** Replacement: `POST /api/trainable-models` with origin-keyed JSON body, or use a label importer. |
| `POST /api/autorun-detectors/import-labels` | `POST /api/trainable-models/<n>/import-labels/<importer>` (already exists) |
| `POST /api/autorun-detectors/from-label-import/<importer>` | New `POST /api/trainable-models/from-label-import/<importer>` (creates a trainable model from import in one shot) |
| `POST /api/detector/export-server` | **Deleted.** Replacement: `GET /api/trainable-models/<n>` returns the JSON; client writes to disk through the file-browser UI. |
| `GET /api/detector/server-files` | **Deleted.** Replacement: file-browser pointed at `data/trainable_models/`. |
| `GET /api/detector/server-files/<n>` | **Deleted.** |
| `POST /api/auto-detect` | Stays. Iterates trainable models flagged for autorun (single registry concept). |

### Settings

- `autorun_processors` (recipe list of detector imports) → **deleted**.
- `autorun_detector_names` (in-memory registry filter) → **deleted**.
- `autorun_trainable_models` (added in PR #1224) → kept; it's the only autorun list now. Each entry is a trainable-model name.

### Model registry

- Drop the `trainable: bool` flag and `detector_name` field. Every entry is a trainable model; `trainable_model_name` is just `name`.
- Add an `autorun: bool` flag (replaces `autodetect` semantics from autorun_detectors).

## Implementation Steps

Each step is independently committable; each leaves the suite green.

### Step 1 — Stop writing weights anywhere

- Remove the `weights` field from `add_autorun_detector()` (`state_processors.py`). The dict still exists in RAM, but the field becomes lazy-derived from `det_ctx.model` only.
- Delete the `fallback_weights` branch in `weights_compat.normalize_detector_weights()`. If origins can't resolve, raise — that's correct semantics under the new principle.
- Delete `GET /api/autorun-detectors/<name>/export`. Update `load-sort-modal.component.ts:121` to fetch `/api/trainable-models/<name>` instead.
- Tests: update `test_detector_export.py`, `test_detectors.py`, `test_multi_detector.py` to assert origins-only.

### Step 2 — Collapse autorun_detectors into DetectorContext

- Delete `state_processors.py:add_autorun_detector / remove_autorun_detector / rename_autorun_detector / get_autorun_detectors`.
- Delete the `autorun_detectors` global at `state_core.py:685`.
- Rewrite `/api/auto-detect` (`detectors_scoring.py:476`) to iterate `model_registry.list_autorun_models()` (new helper) and use each entry's `DetectorContext.model`.
- Trigger `train_from_labelset` for any registry-listed autorun model that doesn't yet have a `DetectorContext` loaded.
- Tests: rewrite `test_detectors.py::TestAutoDetect` to seed via the trainable-model registry instead of `autorun_detectors`.

### Step 3 — Delete detector-on-disk format

- Delete `data/detectors/` references from settings, file-browser, etc.
- Delete `vtsearch/processors/importers/server_detector_file/`.
- Delete `vtsearch/models/weights_compat.py`.
- Delete `/api/detector/export-server`, `/api/detector/server-files`, `/api/detector/server-files/<n>`.
- Delete `POST /api/autorun-detectors/import-pkl`.
- Delete `vtsearch/settings.py:detectors_dir` setting.
- Tests: delete `test_detector_export.py`; trim `test_processor_importers.py`.

### Step 4 — Collapse autorun_processors into autorun_trainable_models

- Delete `autorun_processors`, `autorun_detector_names`, related getters/setters/UI bindings.
- The CLI's `_import_autorun_processors` becomes a no-op; remove it.
- Move any autorun-discovery logic to use `autorun_trainable_models` exclusively.
- Tests: update `test_settings.py`, `test_cli_autodetect.py`.

### Step 5 — Folding routes into trainable-models surface

- Move the still-useful endpoints from `detectors_*.py` (e.g. examples, autorun-flag toggle) onto the trainable-model / model-registry blueprints.
- Delete `detectors_crud.py`, `detectors_training.py` (extractors/localizers stay, in their own files now).
- Update frontend `DetectorsApiService` → renamed `ModelsApiService` (or merged into existing `TrainableModelsApiService`).
- Tests: rename / migrate `test_detectors.py` content.

### Step 6 — Drop trainable/detector_name distinction in registry

- `model_registry.py`: every entry is now a trainable model. Drop `trainable: bool` and `detector_name`. Replace with `autorun: bool`.
- Update `/api/find-label`'s resolution order: cached `det_ctx.model` → train_from_labelset on demand. Drop the `tm_data["weights"]` and `det_name` branches entirely (`detectors_scoring.py:158-191`).
- Tests: simplify multi-detector tests.

### Step 7 — Documentation pass

- Update `docs/ML.md`, `docs/CLI.md`, `docs/EXTENDING.md`, `docs/api/*.md` to describe a single "trainable model" concept.
- Update CLAUDE.md's "Architecture" section: remove detector references; add a one-line summary of the model registry.
- Delete `docs/plans/structural-detectors.md` if obsolete (likely written assuming the two-concept world).

## Test Plan

- After each step, `./run-tests.sh` is green.
- Frontend `npm run build:prod` is clean after every step (the type system catches missing API methods).
- New regression test: load a trainable model in CLI mode against a synthetic dataset whose origins are resolvable; assert MLP scores produce a non-trivial ranking. (Already covered in `test_cli_trainable_models.py`.)
- Before/after benchmark of `/api/auto-detect` cold-start (process restart → first scoring request) — should be within noise of the current implementation, since the work is identical (resolve+embed+train), just owned by `train_from_labelset` instead of `train_detector_from_origins`.

## Open Questions

1. **Import-PKL flow.** Today users can drag a detector JSON file into the dashboard via `/api/autorun-detectors/import-pkl`. The replacement is "drag a trainable-model JSON file" — same origin-keyed shape, but no `weights`. Do we want a one-click "import detector → trainable model" migration in the UI, or just delete the entrypoint? Recommendation: delete; users with old detector files can convert them with a tiny script if needed.
2. **`autodetect` flag semantics.** Currently lives on detector entries; controls whether the detector is included in `/api/auto-detect` runs. After cleanup it becomes `autorun` on the model registry. Do we want it to be the *same* setting as `autorun_trainable_models` (settings list), or separate (registry flag)? Recommendation: settings list is the source of truth, registry flag is a derived view.
3. **Extractors & localizers.** The detector routes file (`detectors_crud.py`) also serves `/api/autorun-extractors[*]` and `/api/autorun-localizers[*]`. These are out of scope for this cleanup but suggest a future "autorun model" concept that subsumes detector / extractor / localizer. Park for a later pass.

## Risk

Low. The cleanup is mechanical: each step deletes code that is either already unused (e.g. `/api/detector/export-server`) or trivially redirected to an existing endpoint. The new trainable-model pipeline (PR #1224) covers every functional case. The biggest risk is hand-rolled scripts in user environments that POST to the old endpoints; per CLAUDE.md's backwards-compat policy, that's acceptable breakage with a clear changelog note.
