# VTSearch Plans

Design docs for features that are proposed, in progress, or recently
landed. New plans live in this folder; once a plan ships and its design
notes are absorbed into [EXTENDING.md](../EXTENDING.md) / its siblings
or [ARCHITECTURE.md](../ARCHITECTURE.md), the plan file is deleted.

## Open plans

| Plan | Status | Summary |
|------|--------|---------|
| [multi-media-import.md](multi-media-import.md) | **In progress** | Importers can mix multiple source media types via `effective_source_specs()`. `server_folder`, `server_files`, `local_folder`, `local_files` migrated; `pickle`, `combine_datasets`, `synthetic`, `http_archive`, `recaller`, `demo` remain on the legacy shim. |
| [patch-embedder.md](patch-embedder.md) | **V1 + V2 shipped; V3 design only** | Six image embedders (DINOv2 / DINOv3 / EUPE × single+patch) are live; region voting via Shift-drag is live. V3 ("one text embedder + one patch embedder per dataset") is designed but not implemented — work plan still a sketch. |
| [clipper-chain.md](clipper-chain.md) | **Phase 1 in flight** | Dataset-load pipeline accepts an ordered list of converter/clipper steps via a new `clipper_chain` field. Phase 2 (frontend chooser), Phase 3 (sidecar/registry schema), Phase 4 (`detector_meta` chain + `input_spec` migration) all deferred — see Open follow-ups. |
| [RCDatasetImporter.md](RCDatasetImporter.md) | **Scaffolds in place; awaiting client code** | ReCaller / DataWrest / PullWrest / Holder plugin scaffolds exist (`hidden_from_picker = True`); the API client stubs (`_rc_fetch_results`, `_dw_get_embedding`, `_pw_fetch_media`, `_holder_*`) still need real implementations. |
| [openapi-schema.md](openapi-schema.md) | **Migration complete; one cosmetic follow-up** | flask-smorest plumbing + Swagger UI + per-plugin runtime validation live across every blueprint. Remaining: per-plugin OpenAPI **spec** types for the six plugin-field route bodies (deferred — runtime validation already captures the field types). |
| [brainstorm.md](brainstorm.md) | **Backlog** | Combined feature + UX backlog (formerly `feature-brainstorm.md` + `ux-brainstorm.md`). Wide-ranging idea backlog — new media types, converters, clippers, demo datasets, UX friction, architecture, experiments. Items graduate into their own plan doc as they mature. |
| [logical-bug-audit.md](logical-bug-audit.md) | **Discovery only** | Multi-agent audit of the codebase for logical bugs (race conditions, context-propagation gaps, silent miscompute, partial-state failures, zip-slip variants). ~95 findings grouped Critical / High / Medium / Low + nine recurring root-cause patterns. No fixes landed. |

## Recently completed (removed)

- **extract-library.md** — Split VTSearch into a reusable `vtscore`
  Python library plus the Flask/Angular `vtsearch` app. All nine phases
  shipped (Phase 6 audited as a no-op, Phase 9 deferred PyPI publication
  until a real external consumer asks). The library now lives at
  `vtscore/` with its own [README](../../vtscore/README.md),
  [CHANGELOG](../../vtscore/CHANGELOG.md), and full developer
  documentation under [`vtscore/docs/`](../../vtscore/docs/). The seven
  seams that were cut between library and app are documented in
  [`vtscore/docs/architecture.md`](../../vtscore/docs/architecture.md).
- **c901-refactor-triage.md** and **c901-refactor-triage-2.md** —
  Per-function refactor/skip decisions for the 2026-05 sweep of
  functions with cyclomatic complexity ≥20. Two waves shipped, twelve
  refactors total (eight in wave 1, four in wave 2); the five "Skip"
  rows carried across both waves stay skipped because the complexity
  is honest dispatch. Running history rolled into
  [brainstorm.md §20.7.1](brainstorm.md).
- **smart-clipper-defaults.md** — "Auto (recommended)" clipper entry for
  audio and video — resolves per-media through pass-through or tiling
  based on each item's duration, with the clipper picker itself hidden
  behind an "Advanced ▾" toggle in the importer modal. All three phases
  shipped: Phase 1 (per-dataset routing), Phase 2 (per-media routing),
  Phase 3 (Advanced toggle in all four importer-modal contexts). The
  only deferred item — per-item routing for image and text — was
  explicitly out of scope.
- **python-quality-tools.md** — pre-commit (ruff + safety hooks),
  deptry, codespell, ruff `S` ruleset, opt-in coverage, vulture audit,
  and the McCabe C901 gate are all in place. GitHub Actions workflows
  were retired; `./run-tests.sh` now runs ruff + codespell + deptry +
  pip-audit + pyright + OpenAPI snapshot drift + the frontend build +
  pytest in one go. Opportunistic maintenance items (C901 noqa
  burn-down, quarterly `pre-commit autoupdate`, possible future
  coverage-delta gate) moved to [brainstorm.md §20.7.x](brainstorm.md).
- **feature-brainstorm.md** and **ux-brainstorm.md** — merged into
  [brainstorm.md](brainstorm.md), with all completed/shipped items
  deleted. Open follow-ups attached to formerly-shipped items kept as
  standalone entries (e.g. faster-whisper backend swap, GPU bulk
  override for audio CLAP / video X-CLIP, folder-browser Phase 2,
  video/document crop overlays).
- **delete-detectors.md** — Collapsed the two-concept "detector vs.
  trainable model" world into a single concept. The old read-only
  detector artifact (with serialized MLP weights), the `autorun_detectors`
  in-memory dict, `weights_compat.py`, `autorun_processors`, the
  registry's `trainable: bool` flag, and the on-disk export routes are
  all gone. What was formerly "trainable model" was then renamed to
  "detector" — so the surviving `detectors_dir` setting and
  `data/detectors/` storage now belong to the new (origin-keyed,
  re-importable) detector concept.
- **pyright-type-checking.md** — Pyright (basic mode) is a hard CI gate
  over the whole `vtsearch/` and `tests/` scope. All seven stages
  shipped: Stage 1 (foundation packages), Stage 2 (`settings`, `state`,
  `security`), Stage 3 (`datasets`, `detectors`, `eval`, `embedding`,
  `training`), Stage 4 (`routes`, `converters`), Stage 5 (`media/`),
  Stage 6 (collapse `include` to `["vtsearch"]`; delete advisory job),
  Stage 7 (`tests/`, including a per-line ignore for a pandas 2.3.3
  Python-3.10 stub gap on `pd.DataFrame(columns=...)`). Reproduce
  CI locally with `python3.10 -m venv /tmp/py310 && source
  /tmp/py310/bin/activate && bash scripts/install-cpu.sh && pyright`.

- **codebase-reorg.md** — Multi-round refactor: tests bucketed into
  group folders, shared embedder stubs in conftest, routes split by
  domain (`datasets/`, `detectors/`, `processors/`, `media/`,
  `settings/`, `labels/`), `vtsearch/models/` package split into
  `detectors/` + `training/` + `embedding/` + `state/diversity_tree.py`,
  `routes/datasets/crud.py` split into `listings.py` / `status.py` /
  `staging.py` / `load.py`, `routes/detectors/store.py` split into
  `crud.py` / `labels.py`, `image/media_type.py` split into
  `_demo_categories.py` + `_demo_sources.py` (1641 → 173 LOC),
  `docker/` / `requirements/` / `scripts/` moved out of repo root,
  `utils/` split into focused packages. Remaining items in the plan
  (`dashboard.component.ts` split, plugin-discovery unification,
  `cli.py` split, `auth/` collapse) were explicit **Skip** decisions
  with documented rationale.
- **gpu-batched-embedding.md** — Phase A (image + text bulk overrides),
  Phase B (bulk `patch_forward`), and Phase C (clip re-embed via
  `embed_media_bulk` with no tempfile) all shipped. Remaining deferred
  follow-ups (audio CLAP / video X-CLIP bulk overrides, fusing DINO
  single-vector + patch forward) live under [brainstorm.md §20.2](brainstorm.md).
- **combine-models-ui.md** — UI for combining two or more trainable
  models into a new one. Backend (`LabelSet.merge` + `POST
  /api/detectors/combine`) shipped earlier; the frontend
  `combine-detectors-modal` component, wired into the dashboard via
  `openCombineDetectorsModal()`, finished the work.
- **sync-sources.md** — Bidirectional sync for settings and detector
  labelsets. Implementation shipped; design notes were folded into the
  "Adding a Settings Source" and "Adding a Labelset Source" sections of
  [EXTENDING-plugins.md](../EXTENDING-plugins.md).
- **active-context-switcher.md** — Top-bar dataset/detector read-only
  fields became click-to-switch pulldowns with compatibility dimming,
  "+ Add New" footers that open the importer / new-detector modals
  in-place, and an explainer overlay for incompatible pairs. Phase 2
  made the URL the source of truth (`/label/:datasetId/:detectorId` and
  `/find/:datasetId/:detectorId` gated by `activeContextGuard`). Phase 3
  added a per-pair spinner glyph backed by `GET /api/jobs/active` plus
  learned-sort rehydration via the `JobManager` signature cache. The
  cross-embedder follow-up closed a latent correctness gap
  (`populate_label_embeddings` now drops `det_ctx.label_embeddings` and
  re-stamps `det_ctx.embedder` when the active dataset's embedder
  changes) and added the "Re-resolving labels for X's embedder…"
  re-embed task path on `POST /api/detectors/registry/load` when the
  detector is already loaded but the embedders disagree.
