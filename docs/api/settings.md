# Settings

[← Back to API index](../API.md)

---

The settings model is defined once, in `vtsearch/settings_models.py`
(`ServerSettings` / `UserSettings`: types, defaults, clamps), and exposed
through the marshmallow schemas in `vtsearch/schemas/settings.py`. The
**`AppSettings`** and **`SettingsUpdate`** schemas in the
[OpenAPI spec](../API.md#machine-readable-schema) are the exhaustive key list;
this page covers the behaviour the schemas don't show.

### Get all settings

```
GET /api/settings
```

→ The merged server-tier + per-user settings object (abridged):

```json
{
  "volume": 1.0,
  "theme": "system",
  "beta": 1.0,
  "calibrate_count": 2,
  "calibration_fraction": null,
  "show_animations": "show",
  "autopilot_enabled": true,
  "autopilot_top_greens": 3,
  "autofind_detectors": [],
  "autofind_exporter": "",
  "browse_graphics": "auto",
  "grid_icon_size_left": {"audio": "M"},
  "saved_datasets_dir": "data/saved_datasets",
  "detectors_dir": "data/detectors",
  "solo_media_type": null,
  "effective_solo_embedder_per_media_type": {},
  "hidden_plugins": {}
}
```

Keys fall into these groups:

| Group | Keys | Notes |
|-------|------|-------|
| Appearance & playback | `theme`, `show_animations`, `show_usage_bars`, `volume`, `audio_playing`, `show_metadata`, `label_hint_dismissed`, `enable_achievements`, `hide_left_panel`, `hide_right_panel` | `theme`: `dark` / `light` / `highviz` / `system` (default `system`, which follows the OS `prefers-color-scheme`). `show_animations`: `show` (default) / `hide` / `os`. `show_usage_bars` (the Dashboard's RAM / Disk bars): `default` (each shown only while its probe reports free space `low`; see [Dashboard › Headroom](dashboard.md#headroom)) / `hide` / `view`. `volume` 0–1. Turning `enable_achievements` off wipes the stored achievement counters. `hide_left_panel` / `hide_right_panel` (both default `true`) fold Train's and Test's side panels to a thin strip; the left folds only on the Autopilot tab. |
| Training | `beta`, `calibrate_count`, `calibration_fraction`, `enrich_descriptions` | `beta` is the **balance** the line is drawn at (F-beta's beta; #4413), clamped to 0.25..4 (same value as `POST /api/balance`); `1` by default, with `4` the recall-leaning preset and `0.25` the precision-leaning one (#4448; they were `2` and `0.5`). It seeds each detector's balance on first read. It is never `null`: a `null` in a `PUT` is a 422, and one left in an older settings file reads as the default. The retired `inclusion`, `min_precision` and `line_preference` keys are dropped like any unknown key. `calibration_fraction` `null` = no explicit split; the per-embedder default applies (0.3 single-vector, 0.5 patch). Changing these drops stale thresholds/heads on every loaded detector. |
| Autopilot | `autopilot_enabled`, `autopilot_top_greens`, `autopilot_hard_reds`, `autopilot_resort_interval`, `autopilot_goal_diversity` | Clamped to ≥ 1. |
| AutoFind | `autofind_detectors`, `autofind_exporter`, `autofind_exporter_field_values`, `autofind_on_import`, `autofind_cli_delete_dataset` | `autofind_exporter` must name a pickable exporter (`""` = none); field values are `{exporter: {key: value}}`. `autofind_on_import` (default `true`) is whether a web import runs the AutoFind detectors on the new dataset: the Add Dataset dialog's **Run AutoFind** checkbox starts from it, and an import that sends `autofind` writes it back. `autofind_cli_delete_dataset` (default `false`) is whether a command-line AutoFind deletes the dataset it imported once its detectors have run; only the CLI reads it (see [CLI](../CLI.md#saving-the-dataset-to-the-dashboard---tempimport)). See [below](#detector-autofind-flag). |
| Per-media-type UI state | `focus_mode_{left,right}`, `grid_icon_size_{left,right,popup}`, `panel_pct_{left,right}`, `popup_metadata_shown`, `popup_preview_size`, `bin_details_docked`, `import_defaults_by_media_type`, `browse_colormap`, `browse_icon_size`, `browse_thumbnail_border`, `browse_mouse_zooms_per_level`, `browse_signposts`, `browse_signpost_captioner` | Dicts keyed by media type id, e.g. `{"audio": "M"}`; a missing entry means "use the frontend default". |
| Browse panel sizes | `browse_graphics`, `browse_panel_width`, `browse_details_panel_width`, `browse_details_metadata_width` | `browse_graphics`: `auto` / `full` / `reduced`. Widths are clamped CSS px. |
| Embedders | `solo_embedder_per_media_type`, `last_embedder_per_media_type` | `solo_embedder_per_media_type` locks a media type to one embedder (`""` opts that type out of a CLI-set lock); invalid type/embedder pairs are a 400. `last_embedder_per_media_type` is written by the load pipeline — accepted by `PUT` but ignored. |
| Storage paths | `saved_datasets_dir`, `detectors_dir` | Path-validated; confined to the user's data dir in multi-user deployments. |
| **Read-only** (admin / computed) | `solo_media_type`, `semantic_only`, `autopilot_only`, `hidden_plugins`, `dataset_max_age_days`, `support_email`, `docs_links`, `max_concurrent_dataset_downloads`, `max_concurrent_dataset_embeddings`, `browse_signpost_vocab`, `effective_solo_embedder_per_media_type` | Returned by `GET`, not accepted by `PUT` (dropped like any unknown key). Set by the operator via CLI flags, env vars, or the settings file; each reports the value actually in force (see `vtsearch/admin_overrides.py`). |

### Update settings

```
PUT /api/settings
```

**Body:** a partial object with any writable keys.

```json
{"volume": 0.5, "theme": "light"}
```

→ The full settings object, as `GET` returns it.

- **All-or-nothing.** Every key in the body is validated before any is
  written, so a 400 means nothing changed.
- **422** for anything the schema catches — a wrong type (`"volume": "x"`) or
  an enum value outside its set (`"theme": "pink"`) — with the standard
  `errors` envelope; **400** for a setter-level failure (unknown media type,
  embedder, or exporter; an empty or escaping directory path).
- **Numeric ranges clamp** rather than fail: `{"volume": 5}` stores `1.0`,
  `{"beta": 10}` stores `4.0`.
- **Unknown and read-only keys are silently dropped.**

### Get default settings

```
GET /api/settings/defaults
```

→ Default values for all settings, in the same shape (infrastructure keys like
`autofind_detectors`, `saved_datasets_dir`, `detectors_dir`, and
`settings_source` are absent).

### Detector AutoFind flag

`autofind_detectors` is a flat list of registered detector names that
should run during `/api/auto-detect` and the CLI's `--autodetect` flow.
Toggle a detector via `PUT /api/detectors/registry/{detector_id}/autofind`
(see `docs/api/detectors.md`); the registry endpoint is the source of
truth and writes through to this settings list.

---

## Settings Sources (Sync)

Settings sources provide **bidirectional sync** for settings; when a source
is active, settings changes are automatically exported to the source, and
`/sync` pulls from the source back into the app. At startup, the active
source is auto-imported so it takes precedence over local settings.

Sources are plugins discovered via the `SETTINGS_SOURCE` sentinel.
Field values support `{username}` template (resolved via `get_current_user()`).

### List available settings sources

```
GET /api/settings-sources
```

→ JSON array of source plugin objects (`name`, `display_name`, `description`,
`icon`, `fields`, `ui_mode`, `hidden_from_picker`):

```json
[
  {
    "name": "server_json_file",
    "display_name": "Server JSON File",
    "icon": "🔄",
    "fields": [
      {"key": "filepath", "label": "Server file path", "type": "server_path"}
    ]
  }
]
```

### Get active settings source

```
GET /api/settings-sources/active
```

→ `{"source_name": "server_json_file", "field_values": {"filepath": "data/{username}.settings.json"}, "inherited": false}` or `null`.

`inherited` is `true` when the source in force is the deployment-wide
`default_settings_source` (server settings file) rather than the user's own.
`null` when no source resolves, or the user has opted out.

### Set or clear active settings source

```
PUT /api/settings-sources/active
```

**Body:** `{"source_name": "server_json_file", "field_values": {"filepath": "data/shared.settings.json"}}`

To clear the user's own choice (and fall back to the deployment default, if
any): `{}` or an empty `source_name`. To opt out even when a default exists:
`{"source_name": "none"}`.

→ `{"ok": true, "message": "..."}`

404 if source_name is unknown.

### Force sync from settings source

```
POST /api/settings-sources/sync
```

Imports settings from the active source into the app.

→ `{"ok": true, "message": "Imported 5 setting(s) from source.", "keys": ["volume", "theme", ...]}`

If no source is configured: `{"ok": false, "message": "No settings source configured or source is empty.", "keys": []}` (still 200).

---

## Labelset Sources (Sync)

Labelset sources provide **bidirectional sync** for detector labels.
Each detector can have its own linked source. When votes are cast or
labels imported, the labelset is automatically exported to the source.

Field values support `{detector_id}` and `{detector_name}` templates
(resolved from the active detector context).

The link lives on the **loaded detector's in-memory context** — it is not
persisted, and it is gone once the detector is unloaded. The path segment
below is spelled `{detector_name}` in the spec, but it is looked up as the
loaded context's key: the registry **detector id** (the `X-Detector-Id`
value).

### List available labelset sources

```
GET /api/labelset-sources
```

→ JSON array of source plugin objects:

```json
[
  {
    "name": "server_json_file",
    "display_name": "Server JSON File",
    "icon": "🔄",
    "fields": [
      {"key": "filepath", "label": "Server file path", "type": "server_path"}
    ]
  }
]
```

### Get detector's labelset source

```
GET /api/detectors/{detector_id}/labelset-source
```

→ `{"source_name": "server_json_file", "field_values": {"filepath": "..."}}`, or
`null` when no source is set **or the detector isn't loaded** (not a 404).

### Set or clear detector's labelset source

```
PUT /api/detectors/{detector_id}/labelset-source
```

**Body:** `{"source_name": "server_json_file", "field_values": {"filepath": "labels/{detector_id}.json"}}`

To clear: `{}` or an empty `source_name`.

→ `{"ok": true, "message": "..."}`

404 if the detector isn't loaded or `source_name` is unknown.

### Force sync from labelset source

```
POST /api/detectors/{detector_id}/labelset-source/sync
```

Imports labels from the detector's linked source.

→ `{"ok": true, "message": "Imported 42 label(s) from source."}`

If no source is configured (or the source is empty): `{"ok": false, "message": "..."}`
(still 200). 404 if the detector isn't loaded.

A rename that moves the templated file path is handled by
[`POST /api/detectors/registry/{detector_id}/labelset-source/move-file`](detectors.md#move-an-orphaned-labelset-file).
