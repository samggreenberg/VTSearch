# Auto-Find settings tab + results exporter wiring

Status: **Phase 1 shipped.** Adds a dedicated "Auto-Find" settings tab, makes
the autorun-detector list editable from the UI, and wires a user-chosen
"results exporter" so an Auto-Find (autodetect) run hands its results to the
configured exporter automatically.

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
- **Auto-Find detectors** — editable checklist of registered detectors. Reuses
  the existing per-detector autorun toggle
  (`PUT /api/detectors/registry/<id>/autorun`), which persists into the
  server-tier `autorun_detectors` list. (Moved here from the read-only Server
  tab, renamed from "Auto-run detectors".)
- **Results Exporter** — a dropdown (None + each pickable exporter) with a
  sub-tab per exporter type (Server JSON File, Server CSV File, Send by Email,
  Webhook). Each sub-tab renders that exporter's own fields; values persist
  per-exporter so switching the dropdown keeps each exporter's config.

### Settings model (server-tier, editable)
Two new `ServerSettings` keys (shared across users, like `autorun_detectors`;
editable from the UI now that they live on a user-facing tab):
- `autofind_exporter: str` — chosen exporter name (`""` = no auto-export).
- `autofind_exporter_field_values: dict[str, dict[str, str]]` — per-exporter
  field values, keyed by exporter name.

Both are excluded from the "Default" reset (like `autorun_detectors`).

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
