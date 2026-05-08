# Settings

[← Back to API index](../API.md)

---

### Get all settings

```
GET /api/settings
```

→ ```json
{
  "volume": 1.0,
  "theme": "dark",
  "inclusion": 0,
  "enrich_descriptions": false,
  "safe_thresholds": false,
  "calibrate_count": 2,
  "calibration_fraction": 0.5,
  "audio_playing": true,
  "swipe_animation": true,
  "show_metadata": true,
  "view_mode_left": {},
  "view_mode_right": {},
  "focus_mode_left": {},
  "focus_mode_right": {},
  "grid_icon_size_left": {},
  "grid_icon_size_right": {},
  "panel_pct_left": {},
  "panel_pct_right": {},
  "autoload_media_embedders": [],
  "autorun_detectors": [],
  "autopilot_enabled": true,
  "hide_autopilot": false,
  "autopilot_top_greens": 3,
  "autopilot_hard_reds": 4,
  "autopilot_resort_interval": 10,
  "autopilot_goal_diversity": 40,
  "saved_datasets_dir": "data/saved_datasets",
  "detectors_dir": "data/detectors"
}
```

Per-media-type settings (`view_mode_*`, `focus_mode_*`, `grid_icon_size_*`,
`panel_pct_*`) use dicts keyed by media type ID (e.g. `{"audio": "list"}`).

### Update settings

```
PUT /api/settings
```

**Body:** partial object with any settings keys to update.

```json
{"volume": 0.5, "theme": "light"}
```

→ Full settings object.

Supported keys: `volume` (number), `theme` (`"dark"` / `"light"` /
`"highviz"`), `inclusion` (int, -10 to +10), `enrich_descriptions` (bool),
`safe_thresholds` (bool), `calibrate_count` (int), `calibration_fraction`
(number), `audio_playing` (bool), `swipe_animation` (bool),
`show_metadata` (bool), `view_mode_left` (dict), `view_mode_right` (dict),
`focus_mode_left` (dict), `focus_mode_right` (dict), `grid_icon_size_left`
(dict), `grid_icon_size_right` (dict), `panel_pct_left` (dict),
`panel_pct_right` (dict), `autoload_media_embedders` (list of strings),
`autopilot_enabled` (bool),
`hide_autopilot` (bool), `autopilot_top_greens` (int),
`autopilot_hard_reds` (int), `autopilot_resort_interval` (int),
`autopilot_goal_diversity` (int), `autorun_detectors` (list of detector names),
`saved_datasets_dir` (string path), `detectors_dir` (string path).

### Get default settings

```
GET /api/settings/defaults
```

→ Default values for all settings (excluding infrastructure keys like
`autorun_detectors`, `saved_datasets_dir`, `detectors_dir`,
and `settings_source`).

### Trainable-model autorun

`autorun_detectors` is a flat list of registered model names that
should run during `/api/auto-detect` and the CLI's `--autodetect` flow.
Toggle a model via `PUT /api/detectors/registry/{model_id}/autorun` (see
`docs/api/models.md`) — the registry endpoint is the source of truth and
writes through to this settings list.

---

## Settings Sources (Sync)

Settings sources provide **bidirectional sync** for settings — when a source
is active, settings changes are automatically exported to the source, and
`/sync` pulls from the source back into the app. At startup, the active
source is auto-imported so it takes precedence over local settings.

Sources are plugins discovered via the `SETTINGS_SOURCE` sentinel.
Field values support `{username}` template (resolved via `get_current_user()`).

### List available settings sources

```
GET /api/settings-sources
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

### Get active settings source

```
GET /api/settings-sources/active
```

→ `{"source_name": "server_json_file", "field_values": {"filepath": "data/{username}.settings.json"}}` or `null`.

### Set or clear active settings source

```
PUT /api/settings-sources/active
```

**Body:** `{"source_name": "server_json_file", "field_values": {"filepath": "data/shared.settings.json"}}`

To clear: `{"source_name": null}`

→ `{"ok": true, "message": "..."}`

404 if source_name is unknown.

### Force sync from settings source

```
POST /api/settings-sources/sync
```

Imports settings from the active source into the app.

→ `{"ok": true, "message": "Imported 5 setting(s) from source.", "keys": ["volume", "theme", ...]}`

If no source is configured: `{"ok": false, "message": "No settings source configured or source is empty."}`.

---

## Labelset Sources (Sync)

Labelset sources provide **bidirectional sync** for detector labels.
Each detector can have its own linked source. When votes are cast or
labels imported, the labelset is automatically exported to the source.

Field values support `{detector_id}` and `{detector_name}` templates
(resolved from the active detector context).

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
GET /api/detectors/{name}/labelset-source
```

→ `{"source_name": "server_json_file", "field_values": {"filepath": "..."}}` or `null`.

404 if detector not found.

### Set or clear detector's labelset source

```
PUT /api/detectors/{name}/labelset-source
```

**Body:** `{"source_name": "server_json_file", "field_values": {"filepath": "labels/{detector_id}.json"}}`

To clear: `{"source_name": null}`

→ `{"ok": true, "message": "..."}`

404 if source_name is unknown. 404 if detector not found.

### Force sync from labelset source

```
POST /api/detectors/{name}/labelset-source/sync
```

Imports labels from the detector's linked source.

→ `{"ok": true, "message": "Imported 42 label(s) from source."}`

If no source configured: `{"ok": false, "message": "..."}`. 404 if detector not found.
