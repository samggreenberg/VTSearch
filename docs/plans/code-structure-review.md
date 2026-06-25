# Code-structure review

**Status:** A systematic, repo-wide structural review. Already shipped:
**Theme A** in full (the `vtscore` ↔ `vtsearch` boundary — fat-controller
extractions, `workflow.py` de-coupling, training-pipeline merge,
inverse-leak sweep), **Theme C quick wins** (the `_ClapBase` mixin and the
settings dispatch-table generation + drift guard), **Theme E quick
wins** (the "shim" rename and the `labelset_ops` facade), and several
**Theme B** mega-file splits (the `settings.py` engine extraction into
`UserSettingsStore`; the `load_pipeline.py` stage extraction; the `app.py`
CLI + port-preflight extraction; and the `importers/base.py` split into a
`base/` package with a thin `ImporterBase` and the rich `DatasetImporter` —
see the Theme B section), and the **Theme F `PluginField` alias collapse**
(the seven per-family `*Field` aliases removed; see Theme F). For the
details of what landed, see the git history / merged PRs on `dev`.
**Everything below is the remaining planned work.**

This review asks: where have design decisions that were right at small scale
been outgrown, and what is worth streamlining, abstracting, or reorganizing?
The codebase is healthy and unusually well-documented; the findings below are
**accretion** problems (modules that started focused and absorbed adjacent
responsibilities), not rot. Nothing here is urgent. Themes are ordered by
leverage-to-effort.

---

## Theme B — Mega-files mixing unrelated concerns

The common shape: a module that started focused and absorbed adjacent
responsibilities.

- **`app.py`** — **DONE (CLI + port-preflight extraction).** The `__main__`
  argparse block + autodetect dispatch moved to `vtsearch/cli_main.py` (its
  `main(app, initialize_server)` is called from a three-line `__main__` in
  `app.py`; `app` and `initialize_server` are passed in rather than imported
  to avoid re-executing `app.py` under a second module name). The Linux
  `/proc` port-preflight helpers + single-instance lock moved to
  `vtsearch/port_preflight.py` (named for precision rather than the plan's
  earlier `vtsearch/app/server.py`, which would have collided with the
  top-level `app` module). `cli_main.main` is itself decomposed into
  `_build_parser` + per-concern helpers (`_maybe_list_plugins`,
  `_maybe_run_pipeline`, `_resolve_plugins`, the `_apply_*` override
  appliers, `_run_autodetect` → `_authenticate_cli_user` /
  `_maybe_import_labels` / `_dispatch_autodetect`, and `_run_server`) so each
  stays under the McCabe gate. `app.py` (down from 1462 → ~630 lines) keeps
  the WSGI `app` object, the request lifecycle hooks, the JSON error
  handlers, blueprint registration, and `initialize_server`.
  **Open follow-up:** the request-lifecycle hooks (`before_request` /
  `after_request` / `teardown_request`) and the global JSON error handlers
  are still inline in `app.py`. They *are* part of the `app` object's
  lifecycle (unlike the CLI/preflight code, which is unrelated to it), so
  the leverage of extracting them to `hooks.py` + `errors.py` is lower and
  the risk (decorator-registration ordering) is higher; left for a scoped
  follow-up.
- **`vtsearch/settings.py`** — **DONE (engine extraction).** The
  lock-ordering-sensitive engine (cross-process file locking, the two-tier
  server/per-user caches, the one-shot legacy migration, and the
  bidirectional sync state-machine) now lives in
  `vtsearch/settings_store.py` as `UserSettingsStore`; `settings.py` (down
  from 1935 → ~1410 lines) keeps the schema/policy layer (Pydantic-driven
  accessor generation, tier routing, CLI fallbacks, effective-value
  resolvers) and delegates engine work to a module-level store instance.
  The shared mutable containers (`_settings_lock`, `_user_caches`,
  `_sync_state`, `_syncing`) stay as module globals and are passed *by
  reference* into the store so external importers (`vtsearch.achievements`,
  the sync-source tests) and the store mutate one set of objects.
  **Open follow-up:** the legacy-migration path is still lazy (runs from
  `UserSettingsStore.ensure_server_loaded` on first server load) rather than
  a one-shot admin script — left as-is deliberately because the lazy
  trigger is what the default-user read-through and the CLI `--settings`
  flat-file flow rely on; moving it to a script is a behavior change worth
  its own scoped task, not a free win.
- **`vtscore/datasets/load_pipeline.py`** — **DONE (stage extraction).**
  `ConcurrencyGate` moved to `vtscore/concurrency/gate.py` (generic
  dynamic-limit semaphore primitive). The six post-import concerns were
  split into a `vtscore/datasets/stages/` package: `_common.py` (shared
  `_TOTAL_LOAD_STEPS`/`_STATUS_TO_STEP` + `_origin_to_str`), `clipper.py`
  (clipper/converter chain + per-clip MD5/embedding/thumbnail fixup),
  `embedding.py` (embed-missing + patch regions), `finalize.py` (drop-none /
  collapse-duplicates / diversity-tree), `projection.py` (opt-in 2-D UMAP
  build + persist), and `registry.py` (auto-register + context-id
  migration). `load_pipeline.py` (down from 1592 → ~640 lines) keeps the
  background-thread orchestration (gate handoff via `_LoadGateController`,
  importer invocation, stage sequencing, failure handling, staging flow)
  plus the request-field parsing helpers (`_parse_bool` /
  `_parse_chain_field` / `_normalize_media_type` / `auto_chunk_size`). The
  dependency DAG is one-way (orchestrator → stages → `stages/_common`); no
  stage imports `load_pipeline`, so there is no import cycle.
- **`vtscore/datasets/importers/base.py`** — **DONE (thin/rich class
  split).** The single 1203-line `base.py` module became a `base/` package:
  `core.py` (`ImporterBase`, the thin base every importer shares — metadata,
  dataset-name resolution, origin building, CLI wrapping, chunked-loading
  scaffolding, the precomputed embedding/MD5/metadata dicts, and the origin
  reload/display/resolve surface, plus an abstract `run()` that raises);
  `dataset_importer.py` (`DatasetImporter(ImporterBase)`, which layers the
  source-spec → converter → ingestion pipeline and the `list_records` /
  `fetch_record` per-record hooks on top); `specs.py` (`SourceSpec` + the
  spec-parsing / converter-ingestion helpers); and `origin.py`
  (origin-serialisation policy helpers + the synthetic dataset-name field).
  Rather than the plan's literal `MultiMediaImporter` rename, `DatasetImporter`
  was **kept as the rich/public base** (identical name, module path, and full
  method set) so the 3 spec-aware folder importers, `recaller`, and all ~25
  test subclasses are unchanged, and **out-of-repo extension importers keep
  working with zero rewrites** (the chosen constraint — see the option
  analysis in the session that shipped this). The 6 truly-thin importers
  (`synthetic`, `pickle`, `local_files`, `combine_datasets`, `demo`,
  `local_folder`) moved onto `ImporterBase`, so they no longer inherit the
  machinery they never used. Discovery is unaffected (the registry keys off
  the `IMPORTER` sentinel + `.name`, never `isinstance`).
- **`vtscore/state/core.py` — `DetectorContext` (31 fields).** A god-object
  spanning vote state, training artifacts, cached embeddings, and
  Find-session state. Splitting into `VoteState` / `TrainingState` /
  `FindSessionState` is the right shape but touches ~346 call sites —
  expensive; defer until already in that code.
- **Frontend:** `dataset-importer-modal.component.ts` (2162; 17 state slices
  + HTTP + 5 picker views) and `browse-canvas.component.ts` (1881; ~200 lines
  of pure pan/zoom/clamp/rubber-band geometry that should be a framework-free
  `ViewTransformService`).

---

## Theme C — Repetitive boilerplate a declarative/table-driven approach collapses

Recurring smell: the **same fact declared in N parallel places**, kept in
sync by hand. (The CLAP-embedder and settings-dispatch instances shipped; the
remaining instances below are still open.)

- **State proxies.** 7 copy-paste `_ProxyDict` / `_ProxyList` declarations in
  `vtsearch/state_proxies.py` could be built from a registry table.
- **Image embedder bases.** Image embedders have single/patch pairs that could
  share a `_SinglePatchBase`; SigLIP/SigLIP2/CLIP have no shared base despite
  identical cross-modal load/warm-up.
- **Downloaders.** `downloader/{audio,image,video,text,docs}.py` each
  re-implement check → download → extract → post-process (~500 lines of
  duplicated zip/tar iteration); a `_DatasetDownloader` base with
  `post_process()` hooks consolidates it.

---

## Theme D — Schema/type drift across boundaries

The same data shapes are independently redeclared in up to four type
systems with nothing enforcing agreement:

- Backend: **Pydantic** (`settings_models.py`) + **Marshmallow**
  (`vtsearch/schemas/`, 3983 lines) for the *same* settings.
- Frontend: **both** a generated OpenAPI client (`ng-openapi-gen` →
  `src/app/generated/api-client`, imported by 53 files) **and** a
  hand-written `api.models.ts` (27 interfaces, imported by 69 files). The
  hand-written types shadow concepts the generated client could cover, and
  several use `[key: string]: unknown` escape hatches that silence drift.

**Fix:** migrate hand-written frontend types onto generated ones where the
endpoint is covered; consider Pydantic→Marshmallow generation server-side.
*Decline* adding `zod`/`io-ts` runtime validation — heavier than the drift
risk warrants.

---

## Theme E — Naming / conceptual overlaps that mislead readers

(The "shim" rename and the `labelset_ops` facade shipped; the remaining bullet
below is still open.)

- **Three overlapping ingestion concepts:** `MediaSource`,
  `DatasetImporter`, and the bare loader functions all mean "get media in,"
  but importers sometimes use a `MediaSource` and sometimes call loaders
  directly, and `FetchedItem.embedding` is bypassed by loader override
  dicts. Draw the boundaries explicitly (or fold `MediaSource` into
  importers). Larger architectural item, not a quick win.

---

## Theme F — Abstractions that aren't paying for themselves

- **`SyncSource[LoadT, SaveT]`** (`vtscore/sync/`) has two subclasses, each
  with one concrete implementation. Don't add more indirection here until a
  third consumer appears.
- ~~**`PluginField` aliases.** Six no-op aliases (`ImporterField = PluginField`
  …) imply field types differ per family. Collapse to `PluginField`.~~
  **Shipped.** Removed all seven per-family aliases (`ImporterField`,
  `ExporterField`, `LabelImporterField`, `LabelsetSourceField`,
  `SettingsSourceField`, `SettingsImporterField`, `SettingsExporterField`);
  every plugin/test now uses `PluginField` directly, re-exported from each
  family's base module. Docs updated to match.
- **`loader.py` façade** re-exports its three sibling loaders without adding
  abstraction.

---

## Evaluated and deliberately **not** touching

The plugin auto-discovery core; the dual `flask.g` / thread-local resolution
mechanism (necessary for the library split); the single `_state_lock` (no
contention symptoms); the frozen request-missing sentinels; the
`concurrency` / `eval` / `security` / `utils` packages (well-sized); and the
route `_shared.py` helpers are all sound. Also declining the proposal to
rename every plugin family's `run()`/`export()`/`load()` to a uniform
`execute()` — domain-meaningful names, churn dwarfs benefit.

---

## Prioritized backlog (remaining)

1. **Theme B** — `importers/base.py` thin/rich split **shipped** (see the
   Theme B bullet above). `settings.py` engine extraction, `load_pipeline.py`
   stage extraction, and the `app.py` CLI + port-preflight extraction also
   shipped. `app.py`'s hooks/errors split remains an open follow-up there.
   Remaining Theme B mega-files: `DetectorContext` sub-context split (item 6
   below) and the two frontend components.
2. **Theme D** — converge frontend types onto the generated client.
3. **Theme E** — `MediaSource` / `DatasetImporter` ingestion-concept overlap.
4. **Theme F** — ~~collapse `PluginField` aliases~~ (shipped); revisit
   `SyncSource` if a third consumer appears.
5. **Theme C remaining** — state-proxy registry table; image single/patch
   base; downloaders base.
6. **Theme B (deferred)** — `DetectorContext` sub-context split (~346 call
   sites; do opportunistically).
7. **Theme A (optional)** — convert the remaining lazy `vtsearch.auth` /
   `vtsearch.achievements` / `vtsearch.logging_config` /
   `vtsearch.routes._shared` reaches in `vtscore/` to injected registration
   hooks (the pattern already used for `register_setting_persister` /
   `register_core_config_builder`), making `vtscore/` import-clean of
   `vtsearch` entirely. They are correct as lazy imports today; do this
   if/when the library tier is actually extracted.
