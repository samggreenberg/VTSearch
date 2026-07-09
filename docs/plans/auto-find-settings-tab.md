# Auto-Find settings tab + results exporter wiring

**Status: Shipped**, including the Phase 2 AutoFind rename + set-and-forget hardening pass. A dedicated "Auto-Find" settings tab houses an editable, per-user detector list and a user-chosen results exporter that an Auto-Find run hands results to automatically; detectors gained the same per-user access-list model as datasets. Open follow-ups below.

## Open follow-ups

- The `/api/auto-detect` route currently has no first-party UI caller (the
  feature is CLI-driven); the server-side auto-export is wired and tested so it
  works the moment a UI flow calls the route.
- Streaming CLI exports (`--stream-results`) still require an explicit
  `--exporter`; the settings fallback only covers the buffered path. Wiring the
  fallback into the streaming pipeline is deferred.
- Scheduling stays external (cron / systemd timers around
  `python app.py --autodetect --user <name> --api-key <key>`); no built-in
  scheduler.
- The email exporter has no template variables in its fields (subject is
  generated); date-stamped subjects could reuse the same template machinery
  if wanted.

### Known limitation: cross-user stale references
Deleting a detector scrubs only the *deleting* user's Auto-Find list; there is no way to enumerate other users' settings files (user data dirs are login-provider-defined), so another user's list may keep a stale name. That staleness is reported, not hidden: their next run lists it under `missing_detectors`.

## What shipped

Icons (frontend):
- `Settings → Browser` tab uses the **eye** glyph (matching Browse buttons); `Settings → Data Imports` uses a new **import** glyph (mirror of the export icon); new **auto-find** glyph (magnifying glass in a ring of three chasing arrows).

Auto-Find settings tab (`Settings → Auto-Find`):
- **Auto-Find detectors** — editable checklist of the user's visible detectors, reusing `PUT /api/detectors/registry/<id>/autorun` (persists to per-user `autofind_detectors`). Moved here from the read-only Server tab, renamed from "Auto-run detectors".
- **Results Exporter** — dropdown (None + each pickable exporter) with a sub-tab per exporter type (Server JSON/CSV File, Send by Email, Webhook); values persist per-exporter.

Settings model (per-user): `autofind_detectors` moved `ServerSettings` → `UserSettings`, joined by `autofind_exporter: str` and `autofind_exporter_field_values: dict[str, dict[str, str]]`. All three excluded from "Default" reset. For the built-in `default` user, reads fall back to the server settings file when absent (`_DEFAULT_USER_FALLBACK_KEYS` + `vtsearch.settings._read_value`); keeps CLI flat file + single-user deployments working while named users get isolated lists.

Detector access-list model (mirrors datasets): registry entries gain `readers: list[str]` (empty = private, `["*"]` = public) alongside `created_by`; `vtscore.detectors.registry` gains `can_user_access_detector` / `is_detector_owner` / `list_detectors_for_user` / `set_detector_readers`; `GET /api/detectors/registry` filters to the current user; `load`/`delete`/`rename`/`autorun` enforce access; new `PUT /api/detectors/registry/<id>/readers`; dashboard gets `created_by`/`readers` columns (auto-hidden when `provider === 'default'`) + per-row security button.

CLI user identity: `--autodetect` defaults to the `default` user; `--user <name> --api-key <key>` authenticates against `data/api_keys.json` and sets the thread-local user so the run reads that user's list + exporter.

End-to-end wiring:
- **CLI** (`python app.py --autodetect`): with no explicit `--exporter`, falls back to `autofind_exporter` / `autofind_exporter_field_values` (via `CoreConfig`); explicit `--exporter` wins; else legacy `gui` default.
- **Server** (`POST /api/auto-detect`): after scoring, runs the configured `autofind_exporter` and returns an `auto_export` status block, surfaced in the results modal.

Phase 2 — AutoFind rename + set-and-forget hardening (the feature is **AutoFind** / "Auto-Find" in UI, not "AutoRun"):
- **Rename (breaking, no shims):** settings key `autorun_detectors` → `autofind_detectors` (old values ignored; users re-tick once), accessors `get/set_autofind_detectors` + `add/remove/is_autofind_detector`, route `.../autorun` → `.../autofind` (body/response field `autofind`), registry flag `autorun` → `autofind`, `CoreConfig.autorun_detectors` → `autofind_detectors`. The processor-side autorun registries (`vtsearch/autorun_processors.py`, `/api/autorun-extractors`, `/api/autorun-localizers`), the `--autodetect` flag, and the `/api/auto-detect` route intentionally keep their names.
- **Dashboard toggle removed:** the "Auto-Find?" column is gone; membership is curated only in Settings → Auto-Find. Delete confirmation now warns when the detector is on your Auto-Find list.
- **Detector references stay name-based:** an entry is a detector *name* → `data/detectors/<slug>.json` (already the re-derivable form); no new indirection. Lifecycle holes handled instead — rename rewrites the list, delete scrubs+warns, `/api/auto-detect` returns `missing_detectors` (all-missing → 400) instead of silently skipping.
- **Date-named export patterns:** path templates gained `{YYYYMMDD}` / `{YYYY}` / `{MM}` / `{DD}` (UTC) alongside `{YYYYMMDD-HHMMSS}`, via `normalize_field_values` + `resolve_export_filepath`.
