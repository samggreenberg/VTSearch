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
| [logical-bug-audit.md](logical-bug-audit.md) | **Discovery only** | Multi-agent audit of the codebase for logical bugs (race conditions, context-propagation gaps, silent miscompute, partial-state failures, zip-slip variants). ~95 findings grouped Critical / High / Medium / Low + nine recurring root-cause patterns. No fixes landed. |
| [frontend-bundle-organization.md](frontend-bundle-organization.md) | **#1 in progress** | Repeated initial-bundle budget bumps (500 → 525 → 540 kB) reflect structural smells. Six enumerated items: shared media-source picker/config widgets (collapses the 1814-line `dataset-importer-modal` and a dupe in `new-detector-modal`), `dashboard.component.ts` split, `label-view.component.ts` split, lazy icon SVGs, API-service audit vs generated client, compressed budget metric. |

