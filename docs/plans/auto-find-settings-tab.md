# Auto-Find settings tab + results exporter wiring

Status: **Shipped.** Adds a dedicated "Auto-Find" settings tab, makes the
autorun-detector list editable and **per-user**, wires a user-chosen "results
exporter" so an Auto-Find run hands results to it automatically, and gives
**detectors the same per-user access-list model as datasets** (owner + readers,
a security button, and a dashboard access column that auto-hides in single-user
mode).

## What shipped

### Icons (frontend)
- `Settings → Browser` tab now uses the **eye** glyph, matching the Browse
  buttons throughout the app (dashboard cards, Find right-panel goods actions).
- `Settings → Data Imports` tab uses a new **import** glyph (rounded box open at
  the upper-left, arrow running from that corner into the centre) — a deliberate
  mirror of the export icon, so import/export read as a matched pair instead of
  the old upload arrow that looked like export.
- New **auto-find** glyph: a small magnifying glass inside a ring of three
  arrows chasing each other clockwise.

### Auto-Find settings tab
New `Settings → Auto-Find` tab (`auto-find` icon), housing:
- **Auto-Find detectors** — editable checklist of the detectors visible to the
  user. Reuses the per-detector autorun toggle
  (`PUT /api/detectors/registry/<id>/autorun`), which persists into the
  caller's **per-user** `autorun_detectors` list. (Moved here from the
  read-only Server tab, renamed from "Auto-run detectors".)
- **Results Exporter** — a dropdown (None + each pickable exporter) with a
  sub-tab per exporter type (Server JSON File, Server CSV File, Send by Email,
  Webhook). Each sub-tab renders that exporter's own fields; values persist
  per-exporter so switching the dropdown keeps each exporter's config.

### Settings model (per-user)
`autorun_detectors` moved from `ServerSettings` to `UserSettings`, joined by two
new per-user keys:
- `autofind_exporter: str` — chosen exporter name (`""` = no auto-export).
- `autofind_exporter_field_values: dict[str, dict[str, str]]` — per-exporter
  field values, keyed by exporter name.

All three are excluded from the "Default" reset. For the built-in **`default`**
user, reads fall back to the *server* settings file when the key is absent from
its own file (`_DEFAULT_USER_FALLBACK_KEYS` + the read-through in
`vtsearch.settings._read_value`), and the destructive legacy migration skips
these keys. This keeps the CLI `--settings` flat file and single-user
deployments working while named multi-user users get a fully isolated list.

### Detector access-list model (mirrors datasets)
- Registry entries gain a `readers: list[str]` field (empty = private to
  creator, `["*"]` = public), alongside the existing `created_by`.
- `vtscore.detectors.registry` gains `can_user_access_detector`,
  `is_detector_owner`, `list_detectors_for_user`, `set_detector_readers`.
- `GET /api/detectors/registry` filters to the current user and returns
  `created_by` / `readers` / `is_owner`; `load` / `delete` / `rename` /
  `autorun` enforce access/ownership; new `PUT /api/detectors/registry/<id>/readers`.
- Dashboard: `created_by` / `readers` columns (auto-hidden when
  `provider === 'default'`) and a per-row security button → access-list prompt.

### CLI user identity
`--autodetect` defaults to the `default` user; `--user <name> --api-key <key>`
authenticates against `data/api_keys.json` (same as the server's `api_key`
login) and sets the thread-local user so the run reads that user's Auto-Find
list + exporter.

### End-to-end wiring
- **CLI** (`python app.py --autodetect`): when no explicit `--exporter` is
  passed, the run falls back to `autofind_exporter` /
  `autofind_exporter_field_values` from settings (threaded through
  `CoreConfig`). An explicit `--exporter` still wins; if neither is set the
  legacy `gui` default applies.
- **Server** (`POST /api/auto-detect`): after scoring, if `autofind_exporter`
  is configured the route runs that exporter on the results and returns an
  `auto_export` status block (`{exporter, success, message?, error?}`), surfaced
  in the auto-detect results modal.

## Open follow-ups
- The `/api/auto-detect` route currently has no first-party UI caller (the
  feature is CLI-driven); the server-side auto-export is wired and tested so it
  works the moment a UI flow calls the route.
- Streaming CLI exports (`--stream-results`) still require an explicit
  `--exporter`; the settings fallback only covers the buffered path. Wiring the
  fallback into the streaming pipeline is deferred.
