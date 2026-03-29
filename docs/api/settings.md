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
