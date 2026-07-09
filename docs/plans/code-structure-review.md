# Code-structure review

**Status:** The remaining structural work spans Themes B/C/D/E/F and the prioritized backlog below.

This review asks: where have design decisions that were right at small scale been outgrown, and what is worth streamlining, abstracting, or reorganizing? The codebase is healthy and unusually well-documented; the findings are **accretion** problems (modules that started focused and absorbed adjacent responsibilities), not rot. Nothing here is urgent. Themes are ordered by leverage-to-effort.

## Prioritized backlog (remaining)

1. **Theme B remaining** — `DetectorContext` sub-context split (item under Theme B; ~346 call sites, do opportunistically) and the two frontend components (`dataset-importer-modal`, `browse-canvas`). Plus the `app.py` hooks/errors split and the `settings.py` lazy-migration-to-script, both left as scoped open follow-ups (see Theme B).
2. **Theme D** — converge frontend types onto the generated client.
3. **Theme E** — `MediaSource` / `DatasetImporter` ingestion-concept overlap.
4. **Theme F** — revisit `SyncSource` if a third consumer appears; the `loader.py` façade.
5. **Theme C remaining** — state-proxy registry table; image single/patch base; downloaders base.
6. **Theme A (optional)** — convert the remaining lazy `vtsearch.auth` / `vtsearch.achievements` / `vtsearch.logging_config` / `vtsearch.routes._shared` reaches in `vtscore/` to injected registration hooks (the pattern already used for `register_setting_persister` / `register_core_config_builder`), making `vtscore/` import-clean of `vtsearch` entirely. Correct as lazy imports today; do this if/when the library tier is actually extracted.

---

## Theme B — Mega-files mixing unrelated concerns (remaining)

Still open:

- **`app.py` hooks/errors (open follow-up on the shipped CLI split).** The request-lifecycle hooks (`before_request` / `after_request` / `teardown_request`) and the global JSON error handlers are still inline in `app.py`. They *are* part of the `app` object's lifecycle (unlike the CLI/preflight code that was extracted), so the leverage of moving them to `hooks.py` + `errors.py` is lower and the risk (decorator-registration ordering) higher; left for a scoped follow-up.
- **`settings.py` lazy migration (open follow-up on the shipped engine split).** The legacy-migration path is still lazy (runs from `UserSettingsStore.ensure_server_loaded` on first server load) rather than a one-shot admin script — left as-is deliberately because the lazy trigger is what the default-user read-through and the CLI `--settings` flat-file flow rely on; moving it to a script is a behavior change worth its own scoped task.
- **`vtscore/state/core.py` — `DetectorContext` (31 fields).** A god-object spanning vote state, training artifacts, cached embeddings, and Find-session state. Splitting into `VoteState` / `TrainingState` / `FindSessionState` is the right shape but touches ~346 call sites — expensive; defer until already in that code.
- **Frontend:** `dataset-importer-modal.component.ts` (2162; 17 state slices + HTTP + 5 picker views) and `browse-canvas.component.ts` (1881; ~200 lines of pure pan/zoom/clamp/rubber-band geometry that should be a framework-free `ViewTransformService`).

---

## Theme C — Repetitive boilerplate a declarative/table-driven approach collapses (remaining)

Recurring smell: the **same fact declared in N parallel places**, kept in sync by hand. (The CLAP-embedder `_ClapBase` mixin and the settings dispatch-table generation shipped; the rest are open.)

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

## Theme D — Schema/type drift across boundaries (open)

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

## Theme E — Naming / conceptual overlaps that mislead readers (remaining)

(The "shim" rename and the `labelset_ops` facade shipped; this bullet is open.)

- **Three overlapping ingestion concepts:** `MediaSource`,
  `DatasetImporter`, and the bare loader functions all mean "get media in,"
  but importers sometimes use a `MediaSource` and sometimes call loaders
  directly, and `FetchedItem.embedding` is bypassed by loader override
  dicts. Draw the boundaries explicitly (or fold `MediaSource` into
  importers). Larger architectural item, not a quick win.

---

## Theme F — Abstractions that aren't paying for themselves (remaining)

(The `PluginField` alias collapse shipped; these two are open.)

- **`SyncSource[LoadT, SaveT]`** (`vtscore/sync/`) has two subclasses, each
  with one concrete implementation. Don't add more indirection here until a
  third consumer appears.
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
