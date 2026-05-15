# VTSearch Plans

Design docs for features that are proposed, in progress, or recently
landed. New plans live in this folder; once a plan ships and its design
notes are absorbed into [EXTENDING.md](../EXTENDING.md) / its siblings
or [ARCHITECTURE.md](../ARCHITECTURE.md), the plan file is deleted.

## Open plans

| Plan | Status | Summary |
|------|--------|---------|
| [codebase-reorg.md](codebase-reorg.md) | **Complete** | Mid-sized refactors: route bucketing (#5), `utils/` split (#6), `docker/`/`requirements/` out of root (#7), docs cleanup (#8), `media → models` import flip (#9). All landed. |
| [combine-models-ui.md](combine-models-ui.md) | Backend shipped; **frontend in progress** | `LabelSet.merge` + `POST /api/detectors/combine` are live; the UI for picking source models and conflict policy is still to build. |
| [multi-media-import.md](multi-media-import.md) | **In progress** | Importers can mix multiple source media types via `effective_source_specs()`. Several importers migrated; `pickle`, `combine_datasets`, `synthetic`, `http_archive`, `recaller`, `demo` remain on the legacy shim. |
| [delete-detectors.md](delete-detectors.md) | **Proposed** | Cleanup follow-up: collapse legacy "detector" concept into trainable models only, origins-as-source-of-truth, MLPs RAM-only. |
| [patch-embedder.md](patch-embedder.md) | **In progress** | Six image embedders (DINOv2 / DINOv3 / EUPE × single+patch). Single-vector variants shipped via PR #1250; patch-region variants and hierarchical region search are the open work. |
| [RCDatasetImporter.md](RCDatasetImporter.md) | **Scaffolds in place; awaiting client code** | ReCaller / DataWrest / PullWrest / Holder plugin scaffolds exist (`hidden_from_picker = True`); the API client stubs need real implementations. |
| [extract-library.md](extract-library.md) | **Proposed** | Split VTSearch into a `vtscore` Python library plus the Flask/Angular app, gated on a CI job that runs the test suite without Flask installed. |
| [gpu-batched-embedding.md](gpu-batched-embedding.md) | **In progress** | Override `_embed_media_bulk_impl` on image + text embedders to batch the GPU forward pass; add `patch_forward_bulk` for the DINO/EUPE patch variants. Targets feature-brainstorm §12.2. |
| [openapi-schema.md](openapi-schema.md) | **In progress** | Replace hand-maintained frontend DTOs with an OpenAPI schema generated from Flask routes via flask-smorest. Settings blueprint migrated as the pilot; remaining blueprints follow. |
| [pyright-type-checking.md](pyright-type-checking.md) | **Stage 1 shipped; in progress** | Adopt pyright basic mode as a hard CI gate, rolled out one package at a time. Targets feature-brainstorm §12.13. Stage 1 (foundation: utils, auth, plugins, sync, concurrency, exporters, labels, settings_io, cli, config) shipped in PR #1349; Stage 2 (settings.py, settings_factory.py, state, security) is next. |
| [feature-brainstorm.md](feature-brainstorm.md) | **Backlog** | Wide-ranging idea backlog — new media types, converters, clippers, demo datasets, experiments. Items graduate into their own plan doc as they mature. |

## Recently completed (removed)

- **sync-sources.md** — Bidirectional sync for settings and detector
  labelsets. Implementation shipped; design notes were folded into the
  "Adding a Settings Source" and "Adding a Labelset Source" sections of
  [EXTENDING-plugins.md](../EXTENDING-plugins.md).
