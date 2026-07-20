# Code-structure review

**Status:** The concrete, shippable findings have been promoted to GitHub issues (see pointers below). What remains in this file is the design narrative that doesn't fit an issue: the gated Theme A note, the items considered and declined, and the "deliberately not touching" list.

This review asked: where have design decisions that were right at small scale been outgrown, and what is worth streamlining, abstracting, or reorganizing? The codebase is healthy and unusually well-documented; the findings are **accretion** problems (modules that started focused and absorbed adjacent responsibilities), not rot. Nothing here is urgent.

## Promoted to issues

Concrete, independently-shippable findings now live as issues (bodies there, not here):

- [ ] #2650 — Collapse the 8 copy-paste state-proxy declarations in `state_proxies.py` to a registry table (Theme C; Sonnet 5)
- [ ] #2651 — Extract a shared cross-modal base for the SigLIP / SigLIP2 / CLIP image embedders (Theme C; Opus 4.8)
- [ ] #2652 — Absorb the residual per-format zip/tar filtering loops in downloaders into `_extract_archive` (Theme C; Haiku 4.5)
- [ ] #2653 — Extract `app.py` request-lifecycle hooks and error handlers into `hooks.py` / `errors.py` (Theme B; Sonnet 5)
- [ ] #2654 — Split the `DetectorContext` god-object into `VoteState` / `TrainingState` / `FindSessionState`, opportunistically (Theme B; Opus 4.8)
- [ ] #2655 — Converge hand-written frontend `api.models.ts` onto generated OpenAPI types, backend-schema-first (Theme D; Sonnet 5)

---

## Open design narrative (not issue-shaped)

<!-- item-sep -->

- **Theme A — make `vtscore/` import-clean of `vtsearch` (gated).** `vtscore/` still reaches into `vtsearch.auth` / `vtsearch.achievements` / `vtsearch.logging_config` / `vtsearch.routes._shared` via lazy in-function imports (~18 sites across `cli.py`, `labels/sync.py`, `embedding/loader.py`, `datasets/load_pipeline.py`, `concurrency/async_jobs.py`, `state/votes.py`, `plugins/normalize.py`, `security/path_validation.py`, `exporters/_template.py`). Converting them to injected registration hooks (the pattern already used for `register_setting_persister` / `register_core_config_builder`) would make the library tier fully import-clean of the app. These are **correct as lazy imports today**; this is only worth doing **if/when the library tier is actually extracted** as a separate package. Left as a gated note, not an issue, because there's no independently-shippable value until that extraction happens.

<!-- item-sep -->

- **Declined — `settings.py` lazy migration → one-shot admin script.** The legacy-migration path (`UserSettingsStore.ensure_server_loaded` → `_maybe_migrate_legacy_settings`, `vtsearch/settings_store.py`) is still lazy (runs on first server-tier load, guarded by `_legacy_migrated`). Moving it to a standalone script was considered and **declined**: the lazy trigger is exactly what the default-user read-through and the CLI `--settings` flat-file flow rely on, and the migration does destructive atomic file rewrites under a cross-process lock, so relocating it is a behavior/data-migration change with upgrade-compatibility risk and little upside. Leave as-is.

<!-- item-sep -->

- **Declined (soft) — backend Pydantic → Marshmallow settings generation.** The same settings fields are hand-declared in both Pydantic (`vtsearch/settings_models.py`, ~533 lines) and Marshmallow (`vtsearch/schemas/settings.py` + `settings_io.py`, ~550 lines). The drift surface is smaller than it looks: the Marshmallow layer already *imports* the enum constants and coercers from `settings_models.py`, so validators are shared — only the flat per-field re-declaration is duplicated. Codegen from Pydantic to Marshmallow would remove that, but it's heavier machinery than a ~550-line, enum-already-shared duplication justifies. Not filed; revisit only if this list of fields starts drifting in practice.

<!-- item-sep -->

## Evaluated and deliberately **not** touching

The plugin auto-discovery core; the dual `flask.g` / thread-local resolution
mechanism (necessary for the library split); the single `_state_lock` (no
contention symptoms); the frozen request-missing sentinels; the
`concurrency` / `eval` / `security` / `utils` packages (well-sized); and the
route `_shared.py` helpers are all sound. Also declining the proposal to
rename every plugin family's `run()`/`export()`/`load()` to a uniform
`execute()` — domain-meaningful names, churn dwarfs benefit.

Two Theme C items shipped earlier (the CLAP-embedder `_ClapBase` mixin and
the settings dispatch-table generation) and two more were found to be
**already substantially done** during this refresh: the downloader
check→download→extract pipeline is consolidated in
`vtscore/datasets/downloader/core.py` (`_download_and_extract` /
`_extract_archive`), leaving only the residual per-format filtering loops in
#2652; and the image single/patch split is already de-duplicated per
backbone (`_Dinov2Base` / `_Dinov3Base` / `_EupeBase`), leaving only the
cross-backbone patch-boilerplate noted as a follow-up inside #2651.
