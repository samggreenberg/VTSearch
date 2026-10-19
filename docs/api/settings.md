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

→ The merged server-tier + per-user settings object. Keys fall into these groups:

| Group | Keys | Notes |
|-------|------|-------|
| Appearance & playback | `theme`, `show_animations`, `show_usage_bars`, `volume`, `audio_playing`, `show_metadata`, `hide_all_hints`, `hidden_hints`, `enable_achievements`, `hide_left_panel`, `hide_right_panel` | `theme`: `dark` / `light` / `highviz` / `system` (default `system`, which follows the OS `prefers-color-scheme`). `show_animations`: `show` (default) / `hide` / `os`. `show_usage_bars` (the Dashboard's RAM / Disk bars): `default` (each shown only while its probe reports free space `low`; see [Dashboard › Headroom](dashboard.md#headroom)) / `hide` / `view`. `volume` 0–1. `hide_all_hints` (default `false`) hides every one of Toasty's hints, the speech bubbles beside the control a new user should click next; `hidden_hints` lists the ids of hints hidden one at a time (blank and repeated ids are dropped). Turning `enable_achievements` off wipes the stored achievement counters. `hide_left_panel` / `hide_right_panel` (both default `true`) fold Train's and Tune's side panels to a thin strip; the left folds only on the Autopilot tab. |
| Training | `beta`, `calibrate_count`, `calibration_fraction`, `enrich_descriptions` | `beta` is the **balance** the line is drawn at (F-beta's beta; #4413), clamped to 0.25..4 (same value as `POST /api/balance`); `1` by default, with `4` the recall-leaning preset and `0.25` the precision-leaning one. It is the user's last pick (#4665): every balance pick and every New
Detector updates it, the New Detector form starts on it, and a detector that
keeps no balance of its own (one made before #4665, or by AutoFind or the CLI)
is seeded from it on first read; a detector that keeps one ignores it. It is never `null`: a `null` in a `PUT` is a 422, and one left in a settings file reads as the default. `calibration_fraction` `null` = no explicit split; the per-embedder default applies (0.3 single-vector, 0.5 patch). Changing these drops stale thresholds/heads on every loaded detector. |
| Autopilot | `autopilot_enabled`, `autopilot_top_greens`, `autopilot_hard_reds`, `autopilot_resort_interval`, `autopilot_goal_diversity` | Clamped to ≥ 1. |
| AutoFind | `autofind_detectors`, `autofind_exporter`, `autofind_exporter_field_values`, `autofind_on_import`, `autofind_cli_delete_dataset` | `autofind_exporter` must name a pickable exporter (`""` = none); field values are `{exporter: {key: value}}`. `autofind_on_import` (default `true`) is whether a web import runs the AutoFind detectors on the new dataset: the Add Dataset dialog's **Run AutoFind** checkbox starts from it, and an import that sends `autofind` writes it back. `autofind_cli_delete_dataset` (default `false`) is whether a command-line AutoFind deletes the dataset it imported once its detectors have run; only the CLI reads it (see [CLI](../CLI.md#saving-the-dataset-to-the-dashboard---tempimport)). See [below](#detector-autofind-flag). |
| Per-media-type UI state | `focus_mode_{left,right}`, `grid_icon_size_{left,right,popup}`, `panel_pct_{left,right}`, `popup_metadata_shown`, `popup_preview_size`, `bin_details_docked`, `import_defaults_by_media_type`, `browse_colormap`, `browse_icon_size`, `browse_thumbnail_border`, `browse_mouse_zooms_per_level`, `browse_signposts`, `browse_signpost_captioner` | Dicts keyed by media type id, e.g. `{"audio": "M"}`; a missing entry means "use the frontend default". |
| Browse panel sizes | `browse_graphics`, `browse_panel_width`, `browse_details_panel_width`, `browse_details_metadata_width` | `browse_graphics`: `auto` / `full` / `reduced`. Widths are clamped CSS px. |
| Embedders | `solo_embedder_per_media_type`, `last_embedder_per_media_type` | `solo_embedder_per_media_type` locks a media type to one embedder (`""` opts that type out of a CLI-set lock); invalid type/embedder pairs are a 400. `last_embedder_per_media_type` is written by the load pipeline — accepted by `PUT` but ignored. |
| **Read-only** (admin / computed) | `saved_datasets_dir`, `detectors_dir`, `solo_media_type`, `semantic_only`, `autopilot_only`, `hidden_plugins`, `dataset_max_age_days`, `support_email`, `docs_links`, `max_concurrent_dataset_downloads`, `max_concurrent_dataset_embeddings`, `browse_signpost_vocab`, `effective_solo_embedder_per_media_type` | Returned by `GET`, not accepted by `PUT` (dropped like any unknown key), and skipped by a settings import or sync. Set by the operator via CLI flags, env vars, or the settings file; each override reports the value actually in force (see `vtsearch/admin_overrides.py`). |

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
  embedder, or exporter).
- **Numeric ranges clamp** rather than fail: `{"volume": 5}` stores `1.0`,
  `{"beta": 10}` stores `4.0`.
- **Unknown and read-only keys are silently dropped.** That includes every
  server-tier key: those are the operator's, never one user's to change.
- **The same rule covers an import and a sync.** A
  [settings import](io.md#run-settings-import) or a settings-source sync
  writes each key through the checks `PUT` runs (`plan_user_write` in
  `vtsearch/settings.py`), so neither can store a value `PUT` would refuse.
  A file's `beta`, `calibrate_count` or `calibration_fraction` drops stale
  thresholds/heads on loaded detectors as `PUT` does, except that an imported
  `beta` is the user's balance, not a balance pick on the active detector.

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
(see [Detectors](detectors.md)); the registry endpoint is the source of
truth and writes through to this settings list.

---

## Settings Sources (Sync)

Settings sources provide **bidirectional sync** for settings; when a source
is active, settings changes are automatically exported to the source, and
`/sync` pulls from the source back into the app. A user's settings are also
pulled from their active source the first time they load in a process, and
again whenever the source reports a new version (probed at most once a
second); a key the user has just edited locally is not overwritten by that
automatic pull.

Sources are plugins discovered via the `SETTINGS_SOURCE` sentinel.
Field values support `{username}` template (resolved via `get_current_user()`).

### List available settings sources

```
GET /api/settings-sources
```

→ JSON array of source plugin objects (`name`, `display_name`, `description`,
`icon`, `fields`, `ui_mode`, `hidden_from_picker`). Operator-hidden plugins
(the `hidden_plugins` setting) are omitted.

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

Imports settings from the active source into the app. Only the user's own
settings are applied, as for a [settings import](io.md#run-settings-import):
server-tier keys and invalid values in the source are skipped, and `keys`
lists what was applied. The automatic re-sync follows the same rule.

→ `{"ok": true, "message": "Imported 5 setting(s) from source.", "keys": ["volume", "theme", ...]}`

If no source is configured: `{"ok": false, "message": "No settings source configured or source is empty.", "keys": []}` (still 200).

---

## Labelset Sources (Sync)

Labelset sources provide **bidirectional sync** for detector labels: a
detector linked to one has its whole labelset pushed there as it changes
(paced: pushes for one detector never overlap and start at least
`labelset_source_push_interval_seconds` apart), and pulled back and merged
when it loads or on a manual sync. See
[EXTENDING-plugins.md § Adding a Labelset Source](../EXTENDING-plugins.md#adding-a-labelset-source).

A detector is linked by a binding stored on its **registry entry**, so the
link survives restarts and works whether or not the detector is loaded. A
detector with no binding of its own follows the deployment-wide
`default_labelset_source` server setting, if any; every non-empty value in
that setting's `field_values` is fixed by the admin and is never echoed or
stored per detector. Field values support `{detector_id}` and
`{detector_name}` templates, resolved for the detector being synced.

The path segment below is spelled `{detector_name}` in the spec; it takes
the registry **detector id**, or the detector's name.

### List available labelset sources

```
GET /api/labelset-sources
```

→ JSON array of source plugin objects, the same shape as the settings-source
list.

### Get the deployment-wide labelset source

```
GET /api/labelset-sources/default
```

→ the `default_labelset_source` server setting's source as a plugin entry
whose `fields` are only those the admin left for users to fill in (the New
Detector form renders these), plus `fixed_keys` (the fields the admin
fixed; their values stay on the server) and `push_interval_seconds`. `null`
when the deployment has no default.

```json
{
  "name": "server_json_file",
  "display_name": "Server JSON File",
  "fields": [],
  "fixed_keys": ["filepath"],
  "push_interval_seconds": 60
}
```

### Get detector's labelset source

```
GET /api/detectors/{detector_ref}/labelset-source
```

`detector_ref` is the detector's id or its name, on all three routes below.

→ `{"source_name", "display_name", "field_values", "inherited", "fixed_keys", "missing_fields"}`,
or `null` when the detector syncs nowhere. `field_values` holds only the
detector's own values, and only for its creator. `inherited` is `true` when
it follows the deployment default. `missing_fields` names required fields
with no value; the detector is not synced until they are filled in. 403 if
the caller cannot access the detector, 404 if it does not exist.

### Set or clear detector's labelset source

```
PUT /api/detectors/{detector_ref}/labelset-source
```

**Body:** `{"source_name": "server_json_file", "field_values": {"filepath": "labels/{detector_name}.json"}}`

Values the deployment default fixes for that source are dropped (they always
come from the server). The rest are validated the way a push uses them -
required fields present, templates resolvable, paths and URLs allowed -
then stored, after which the detector is levelled with the source: what the
source already holds is merged in (the detector's labels win a
disagreement) and the detector's labels are pushed back.

To clear the detector's own binding (it then follows the default again):
`{}` or an empty `source_name`. To opt it out even when a default exists:
`{"source_name": "none"}`.

→ `{"ok": true, "message": "..."}` (the message counts any labels it restored)

400 if a value is missing or not allowed, 403 unless the caller created the
detector, 404 if the detector or `source_name` is unknown.

### Force sync from labelset source

```
POST /api/detectors/{detector_ref}/labelset-source/sync
```

Pulls the source's labels into the detector: labels it lacks are added and,
where the two disagree, the **source's** label wins. Nothing is removed. A
loaded detector's votes are re-derived from the result, and the merged
labels are pushed back.

→ `{"ok": true, "message": "Imported 42 label(s) from source (3 new or changed).", "labels": 42, "changed": 3}`

If no source is configured (or the source is empty): `{"ok": false, "message": "..."}`
(still 200). 403 if the caller cannot access the detector, 404 if it does not
exist.

A rename moves a `{detector_name}`-templated server file to the new name's
path itself; see [Rename detector](detectors.md#rename-detector).
