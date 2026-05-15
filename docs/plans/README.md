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
| [feature-brainstorm.md](feature-brainstorm.md) | **Backlog** | Wide-ranging idea backlog — new media types, converters, clippers, demo datasets, experiments. Items graduate into their own plan doc as they mature. |

## Recently completed (removed)

- **sync-sources.md** — Bidirectional sync for settings and detector
  labelsets. Implementation shipped; design notes were folded into the
  "Adding a Settings Source" and "Adding a Labelset Source" sections of
  [EXTENDING-plugins.md](../EXTENDING-plugins.md).
