# VTSearch Plans

Design docs for features that are proposed, in progress, or recently
landed. New plans live in this folder; once a plan ships and its design
notes are absorbed into [EXTENDING.md](../EXTENDING.md) / its siblings
or [ARCHITECTURE.md](../ARCHITECTURE.md), the plan file is deleted.

## Open plans

| Plan | Status | Summary |
|------|--------|---------|
| [multi-media-import.md](multi-media-import.md) | **In progress** | Importers can mix multiple source media types via `effective_source_specs()`. `server_folder`, `server_files`, `local_folder`, `local_files` migrated; `pickle`, `combine_datasets`, `synthetic`, `http_archive`, `recaller`, `demo` remain on the legacy shim. |
| [delete-detectors.md](delete-detectors.md) | **Mostly shipped** | Steps 1–2 and most of step 3 landed: `vtsearch/models/` is gone, `weights_compat.py` is gone, `/api/autorun-detectors/*` and detector-on-disk routes are gone, `autorun_processors` is gone, the `trainable` flag is gone. Remaining: delete the `detectors_dir` setting (step 3 tail) and the docs pass (step 7). |
| [patch-embedder.md](patch-embedder.md) | **V1 + V2 shipped; V3 design only** | Six image embedders (DINOv2 / DINOv3 / EUPE × single+patch) are live; region voting via Shift-drag is live. V3 ("one text embedder + one patch embedder per dataset") is designed but not implemented — work plan still a sketch. |
| [RCDatasetImporter.md](RCDatasetImporter.md) | **Scaffolds in place; awaiting client code** | ReCaller / DataWrest / PullWrest / Holder plugin scaffolds exist (`hidden_from_picker = True`); the API client stubs (`_rc_fetch_results`, `_dw_get_embedding`, `_pw_fetch_media`, `_holder_*`) still need real implementations. |
| [extract-library.md](extract-library.md) | **Proposed** | Split VTSearch into a `vtscore` Python library plus the Flask/Angular app, gated on a CI job that runs the test suite without Flask installed. Not started. |
| [openapi-schema.md](openapi-schema.md) | **Pilot shipped; rollout in progress** | flask-smorest plumbing + Swagger UI live; `settings/api.py` migrated. Remaining: frontend `SettingsApiService` rewired to the generated client, then the other blueprints (auth, achievements, main, labels, detectors, processors, media, datasets, sorting, eval, file_browser) — and finally delete the legacy permissive `/openapi.json`. |
| [pyright-type-checking.md](pyright-type-checking.md) | **Stage 1 shipped; stages 2–6 open** | `pyrightconfig.json` gates `utils/`, `auth/`, `plugins/`, `sync/`, `concurrency/`, `exporters/`, `labels/`, `settings_io/`, `cli.py`, `config.py`. Remaining stages: 2 (`settings*`, `state/`, `security/`), 3 (`datasets/`, `detectors/`, `eval/`, `models/`), 4 (`routes/`, `converters/`), 5 (`media/`), 6 (whole `vtsearch/` — advisory job removed). |
| [feature-brainstorm.md](feature-brainstorm.md) | **Backlog** | Wide-ranging idea backlog — new media types, converters, clippers, demo datasets, experiments. Items graduate into their own plan doc as they mature. |
| [ux-brainstorm.md](ux-brainstorm.md) | **Backlog** | Audit of friction across importers, labeling, sorting, settings, and progress UX. ~75 ideas across auto-fill, hints, speed-ups, clarity, streamlining, and consistency. Items graduate into their own plan doc as they mature. |

## Recently completed (removed)

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
  single-vector + patch forward) live under feature-brainstorm §12.2.
- **combine-models-ui.md** — UI for combining two or more trainable
  models into a new one. Backend (`LabelSet.merge` + `POST
  /api/detectors/combine`) shipped earlier; the frontend
  `combine-detectors-modal` component, wired into the dashboard via
  `openCombineDetectorsModal()`, finished the work.
- **sync-sources.md** — Bidirectional sync for settings and detector
  labelsets. Implementation shipped; design notes were folded into the
  "Adding a Settings Source" and "Adding a Labelset Source" sections of
  [EXTENDING-plugins.md](../EXTENDING-plugins.md).
