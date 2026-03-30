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
  "autoload_media_types": [],
  "autoload_media_embedders": [],
  "autorun_processors": [],
  "autopilot_enabled": true,
  "hide_autopilot": false,
  "autopilot_top_greens": 3,
  "autopilot_hard_reds": 4,
  "autopilot_resort_interval": 10,
  "autopilot_goal_diversity": 40,
  "autorun_detector_names": [],
  "saved_datasets_dir": "data/saved_datasets",
  "detectors_dir": "data/detectors",
  "trainable_models_dir": "data/trainable_models"
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
`panel_pct_right` (dict), `autoload_media_types` (list of strings),
`autoload_media_embedders` (list of strings), `autopilot_enabled` (bool),
`hide_autopilot` (bool), `autopilot_top_greens` (int),
`autopilot_hard_reds` (int), `autopilot_resort_interval` (int),
`autopilot_goal_diversity` (int), `autorun_detector_names` (list of strings),
`saved_datasets_dir` (string path), `detectors_dir` (string path),
`trainable_models_dir` (string path).

### Get default settings

```
GET /api/settings/defaults
```

→ Default values for all settings (excluding `autorun_processors`).

### Autorun processors

```
GET /api/settings/autorun-processors
```

→ `{"autorun_processors": [{"processor_name": "...", "processor_importer": "...", "field_values": {...}, "settings_json": "..."}]}`

```
POST /api/settings/autorun-processors
```

**Body:** `{"processor_name": "my_detector", "processor_importer": "server_detector_file", "field_values": {"filepath": "/path/to/detector.json"}}`

→ `{"success": true, "processor_name": "...", "processor_importer": "...", "field_values": {...}, "settings_json": "..."}`

```
DELETE /api/settings/autorun-processors/{name}
```

→ `{"success": true}` or 404.

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

→ ```json
{
  "sources": [
    {
      "name": "server_json_file",
      "display_name": "Server JSON File",
      "icon": "🔄",
      "fields": [
        {"key": "filepath", "label": "Server file path", "type": "server_path", ...}
      ]
    }
  ]
}
```

### Get active settings source

```
GET /api/settings-sources/active
```

→ `{"source": {"source_name": "server_json_file", "field_values": {"filepath": "data/{username}.settings.json"}}}` or `{"source": null}`.

### Set or clear active settings source

```
PUT /api/settings-sources/active
```

**Body:** `{"source_name": "server_json_file", "field_values": {"filepath": "data/shared.settings.json"}}`

To clear: `{"source_name": null}`

→ `{"ok": true}`

400 if source_name is unknown.

### Force sync from settings source

```
POST /api/settings-sources/sync
```

Imports settings from the active source into the app.

→ `{"ok": true, "settings": {"volume": 80, "theme": "dark", ...}}`

If no source is configured: `{"error": "No active settings source configured"}` (400).

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

→ ```json
{
  "sources": [
    {
      "name": "server_json_file",
      "display_name": "Server JSON File",
      "icon": "🔄",
      "fields": [
        {"key": "filepath", "label": "Server file path", "type": "server_path", ...}
      ]
    }
  ]
}
```

### Get detector's labelset source

```
GET /api/detectors/{name}/labelset-source
```

→ `{"source": {"source_name": "server_json_file", "field_values": {"filepath": "..."}}}` or `{"source": null}`.

404 if detector not found.

### Set or clear detector's labelset source

```
PUT /api/detectors/{name}/labelset-source
```

**Body:** `{"source_name": "server_json_file", "field_values": {"filepath": "labels/{detector_id}.json"}}`

To clear: `{"source_name": null}`

→ `{"ok": true}`

400 if source_name is unknown. 404 if detector not found.

### Force sync from labelset source

```
POST /api/detectors/{name}/labelset-source/sync
```

Imports labels from the detector's linked source.

→ `{"ok": true, "imported": 42}` (number of labels applied)

400 if no source configured. 404 if detector not found.
