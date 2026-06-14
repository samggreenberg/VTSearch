# Code-structure review

**Status:** A systematic, repo-wide structural review. Already shipped:
**Theme A** in full (the `vtscore` ↔ `vtsearch` boundary — fat-controller
extractions, `workflow.py` de-coupling, training-pipeline merge,
inverse-leak sweep), **Theme C quick wins** (the `_ClapBase` mixin and the
settings dispatch-table generation + drift guard), and **Theme E quick
wins** (the "shim" rename and the `labelset_ops` facade). For the details of
what landed, see the git history / merged PRs on `dev`. **Everything below
is the remaining planned work.**

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

- **`app.py` (1408 lines).** ~625 lines (783–1408) are the `__main__`
  argparse block + autodetect dispatch; ~180 lines (596–776) are Linux
  `/proc` port-preflight helpers. Neither relates to the WSGI `app` object
  gunicorn imports. Extract `vtsearch/cli_main.py` (argparse/dispatch) and
  `vtsearch/app/server.py` (port preflight); optionally `hooks.py` +
  `errors.py` for the request lifecycle and error handlers.
- **`vtsearch/settings.py` (1887 lines).** Conflates cache state,
  cross-process file locking, two-tier routing, a one-shot legacy migration,
  bidirectional sync state-machines, and dynamic accessor generation. The
  lock-ordering discipline (`file_lock → settings_lock`, re-entrance guards)
  would be safer encapsulated in a `UserSettingsStore`; the legacy-migration
  path could move to a one-shot script.
- **`vtscore/datasets/load_pipeline.py` (1588 lines).** Six concerns:
  `ConcurrencyGate`, progress/step mapping, clipper-chain fixup, embedding,
  dedup/diversity, registry/migration, background-task orchestration. Move
  `ConcurrencyGate` to `vtscore/concurrency/`; split post-import stages into
  a `stages/` package.
- **`vtscore/datasets/importers/base.py` (1202 lines, 35 methods).** A
  trivial single-API-call importer inherits all the multi-media-spec /
  converter / ingestion machinery. Split a thin `ImporterBase` from a
  `MultiMediaImporter` mixin.
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
- **`PluginField` aliases.** Six no-op aliases (`ImporterField = PluginField`
  …) imply field types differ per family. Collapse to `PluginField`.
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

1. **Theme B** — split `app.py`, `load_pipeline.py`, `settings.py`,
   `importers/base.py`.
2. **Theme D** — converge frontend types onto the generated client.
3. **Theme E** — `MediaSource` / `DatasetImporter` ingestion-concept overlap.
4. **Theme F** — collapse `PluginField` aliases; revisit `SyncSource` if a
   third consumer appears.
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
