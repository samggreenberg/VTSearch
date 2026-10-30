# Settings files

What goes in VTSearch's two settings files, key by key. Both live in the data
directory; where that is, and which of its files to back up, is in
[DEPLOYMENT.md § Data directory layout](DEPLOYMENT.md#data-directory-layout).

Settings live in **two tiers, in two different files**. Editing the wrong one
is the most common settings mistake on a fresh deployment, so start here:

| Tier | File | Holds | Model |
|------|------|-------|-------|
| **Server** | `data/settings.json` | Deployment-wide infrastructure and operator locks. Loaded once at startup, before any user logs in. | `ServerSettings` (`vtsearch/settings/models.py`) |
| **Per-user** | `<user data dir>/user_settings.json` | Every user preference — theme, volume, autopilot, AutoFind, panel layout, browse prefs. Resolved per request from the logged-in user. | `UserSettings` (same module) |

The per-user file lives at `data/user_settings.json` in a single-user
deployment and at `data/<username>/user_settings.json` under a multi-user login
provider. Both files are auto-created on first use and auto-saved on every
change. Both models accept extra keys, so an unrecognised key is preserved
rather than dropped.

A key the model *does* know but whose value it rejects — a hand-edit out of
range, or a value written before the field changed shape — is **ignored on
load**: the field reads as unset, so its default applies. Nothing is rewritten,
so the file keeps whatever you put in it; fix the value and it takes effect on
the next start. VTSearch does not migrate old settings shapes forward.

**If you are changing a user preference, edit `user_settings.json`, not
`settings.json`.** A `theme` or `autopilot_enabled` key placed in
`data/settings.json` is simply ignored. The one deliberate exception is the
CLI-facing AutoFind keys (`autofind_detectors`, `autofind_exporter`,
`autofind_exporter_field_values`, `autofind_cli_delete_dataset`): for the
built-in `default` user only, a read
that misses in `user_settings.json` falls through to `data/settings.json`, which
is what lets the CLI's `--settings` flat file and single-user deployments keep
working.

**Only the operator sets the server tier.** No user's own write reaches it:
`PUT /api/settings` drops server-tier keys, and a settings import or a
settings-source sync skips them, so a file a user imports can neither repoint
the storage directories nor lift an operator lock. Change these keys here, or
through their CLI flag or environment variable.

## Server tier — `data/settings.json`

```json
{
  "saved_datasets_dir": "data/saved_datasets",
  "detectors_dir": "data/detectors",
  "max_concurrent_dataset_downloads": 4,
  "max_concurrent_dataset_embeddings": 2,
  "hidden_plugins": {},
  "dataset_max_age_days": null,
  "support_email": "ops@example.org",
  "docs_links": [
    {"label": "Acme plugin guide", "url": "https://docs.example.org/acme-plugin"},
    {"label": "Lab data policy", "url": "/wiki/data-policy"}
  ],
  "semantic_only": false,
  "autopilot_only": false,
  "solo_media_type": null,
  "projection_n_neighbors": 15,
  "projection_min_dist": 0.1,
  "browse_signpost_vocab": {},
  "default_settings_source": null,
  "default_labelset_source": null,
  "labelset_source_push_interval_seconds": 60
}
```

- `saved_datasets_dir`, `detectors_dir`: infrastructure directories
  (overridable here for custom data layouts; the Settings dialog shows them
  read-only).
- `max_concurrent_dataset_downloads` / `max_concurrent_dataset_embeddings`:
  concurrency gates for dataset loading. The download gate covers the
  bandwidth/disk-bound import phase; the embed gate covers CPU/GPU-bound
  embedding plus post-load clipping, dedup, and coverage-atlas construction.
  Unset, they autodetect from hardware on every start, as described under
  [Dataset-ingest concurrency](DEPLOYMENT.md#dataset-ingest-concurrency); the autodetected
  value is **not** persisted. An explicit value here (clamped to `[1, 16]`)
  wins over both the autodetect and the env vars. Changes take effect on
  queued and future loads (running tasks are never preempted).
- `hidden_plugins`, `dataset_max_age_days`, `support_email`, `semantic_only`,
  `solo_media_type`: the persisted half of the operator restrictions whose
  flags are described in [CLI.md § Web server modes](CLI.md#web-server-modes)
  (`--hide-plugin`, `--dataset-max-age-days`, `--support-email`,
  `--semantic-only`, `--solo-media-type`). `hidden_plugins` maps a plugin
  family (`"converters"`, `"embedders"`, … as `--list-plugins` prints them) to
  a list of names and is *unioned* with the flag/env hides;
  `dataset_max_age_days: null` (the default) means datasets never expire;
  `support_email` defaults to the built-in project address. For the others a
  flag or env var overrides the persisted value for the process lifetime
  (`semantic_only` can only be switched *on* that way). None of them is
  editable through `PUT /api/settings`; `GET /api/settings` reports the
  effective values.
- `docs_links`: this deployment's own documentation, for when you add plugins
  or extensions users need to read up on. An ordered list of
  `{"label": ..., "url": ...}` objects; the Help modal lists them, in this
  order, under **Docs for this server** at the bottom of every tab, and each
  opens in a new browser tab. `url` must be an absolute `http://` / `https://`
  URL or a root-relative `/path` on the VTSearch host; an entry whose URL is
  anything else, or whose `label` is blank, is dropped, and the startup log
  prints an `Ignoring docs_links entry …` line for each one (a good entry is
  listed on a `Help-modal docs: …` line). Empty (the default) shows nothing.
  Settings-file only (there is no flag or env var); surfaced read-only at
  `GET /api/settings`, not editable via `PUT`. Read at startup, so restart
  after editing it.
- `semantic_only`: locks the deployment to Semantic embedders, hiding the
  prototype Patch Semantic and Structural types from every picker and rejecting
  them at the dataset-load / detector-create routes. Also settable with
  `--semantic-only` / `VTSEARCH_SEMANTIC_ONLY`.
- `autopilot_only`: keeps **Train** and **Tune** on their Autopilot tabs. Train
  loses its Manual tab and Tune its Review tab, and neither view shows a tab
  bar, so users only see the guided flow. Train starts in Autopilot whatever a
  user's own `autopilot_enabled` preference says. With Tune's Review tab gone,
  its Browse / To Dataset / Export actions for unverified matches are gone too.
  Autopilot needs a first sort to start from, and on a dataset that can't search by
  text that has to be an example item, which the New Detector dialog
  requires. A detector that reaches Train without one (opened on such a
  dataset with only a text description and too few labels to train) shows a
  note instead of the Autopilot panel. A UI simplification, not a permission
  boundary. Also settable with `--autopilot-only` / `VTSEARCH_AUTOPILOT_ONLY`.
- `solo_media_type`: narrows the whole instance to one media type. The importer
  and new-detector flows hide their media-type pickers and lock to it, the
  converter picker filters to converters that output it, media-type steps
  in tabbed UIs are skipped, and fields that could only name it (Find Results,
  the metadata panes, the Dashboard's Type columns) are hidden. `null` shows
  everything. Users cannot change it;
  also settable with `--solo-media-type` / `VTSEARCH_SOLO_MEDIA_TYPE`, either of
  which wins for the process lifetime.
- `projection_n_neighbors`, `projection_min_dist`: UMAP knobs for the VTSBrowse
  map. The persisted projection is keyed on these, so changing one forces a
  recompute rather than serving a layout fit under the old params. Clamped to
  `[2, 200]` and `[0.0, 0.99]`.
- `browse_signpost_vocab`: per-media-type zero-shot tag vocabulary used to name
  map regions in Tags mode, replacing the built-in AudioSet-527 /
  OpenImages-600 lists — the hook for a domain-specific taxonomy (bird species,
  machine faults, product categories). Server-tier and read-only over the API,
  since every user of an instance should read the same region names. Normalized
  on write (trimmed, de-duplicated, capped at 2000 terms); a media type with an
  empty or absent list falls back to the shipped vocabulary. Takes effect on the
  next projection build / Re-project, since signpost texts are cached.
- `default_settings_source`: deployment-wide default for settings sync, same
  `{"source_name": ..., "field_values": ...}` shape as the per-user
  `settings_source`. A user with no explicit source of their own inherits this;
  a user whose `settings_source` is `{"source_name": "none"}` explicitly opts
  out. Template the `field_values` (e.g.
  `{"filepath": "data/user-settings/{username}.json"}`) to personalise it per
  user.
- `default_labelset_source`: auto-export every detector's labels (#4913). Same
  `{"source_name": ..., "field_values": ...}` shape as a detector's own
  labelset-source binding; every detector without one keeps its whole
  labelset synced to this source as it is labelled, and pulls it back (merging)
  when it loads. That makes the source the copy that survives losing this
  server: re-create a detector under the same name and loading it restores its
  labels. Each **non-empty** value in `field_values` is fixed - users never see
  or change it - and the New Detector form asks for the source's other fields,
  storing them on the detector (the Dashboard's **Auto-export…** action edits
  them later, and flags a detector still missing one). For example, to keep
  one JSON file per detector on a mounted backup volume, with nothing for
  users to fill in:
  `{"source_name": "server_json_file", "field_values": {"filepath": "/mnt/backup/vtsearch/{detector_name}.labels.json"}}`.
  In a multi-user deployment a `server_path` value is confined to the
  detector creator's data directory, like every server path a user supplies.
  `null` (the default) auto-exports nothing.
- `labelset_source_push_interval_seconds`: the shortest gap between two pushes
  of one detector's labels to its labelset source (default 60). Pushes for one
  detector never overlap, and changes made meanwhile ride the next push, so a
  slow target or a burst of votes costs one write per window; pending pushes
  are flushed on a normal shutdown. `0` pushes as soon as the previous push
  lands. Clamped to `[0, 86400]`.

## Per-user tier — `user_settings.json`

An abridged example; the full field list is `UserSettings` in
`vtsearch/settings/models.py`.

```json
{
  "volume": 1.0,
  "beta": 1.0,
  "theme": "system",
  "enrich_descriptions": false,
  "calibrate_count": 2,
  "calibration_fraction": null,
  "audio_playing": true,
  "show_animations": "show",
  "show_usage_bars": "default",
  "show_metadata": false,
  "hide_all_hints": false,
  "hidden_hints": [],
  "used_features": [],
  "enable_achievements": true,
  "autopilot_enabled": true,
  "hide_left_panel": true,
  "hide_right_panel": true,
  "autopilot_top_greens": 3,
  "autopilot_hard_reds": 4,
  "autopilot_resort_interval": 10,
  "autopilot_goal_diversity": 0,
  "autofind_detectors": [],
  "autofind_exporter": "",
  "autofind_exporter_field_values": {},
  "autofind_on_import": true,
  "autofind_cli_delete_dataset": false,
  "autofind_export_enabled": true,
  "import_alerts_enabled": true,
  "focus_mode_left": {},
  "focus_mode_right": {},
  "grid_icon_size_left": {},
  "grid_icon_size_right": {},
  "panel_pct_left": {},
  "panel_pct_right": {},
  "browse_graphics": "auto",
  "browse_panel_width": 360,
  "browse_colormap": {},
  "browse_icon_size": {},
  "import_defaults_by_media_type": {},
  "recent_sessions": []
}
```

- `autopilot_goal_diversity`: how many Coverage Atlas regions Autopilot's
  **Explore Diversity** step asks your votes to cover before Done. `0` (the
  default since #5473) skips the step, so Done waits on the Smart and Stable
  lights alone; any positive goal turns it back on (`40` was the default
  before). Skipping it stopped sessions about 20 votes sooner with the same
  F-beta on faces and slightly less on COCO
  ([report](experiments/2026-10-30-skip-diversity-5473/REPORT.md)); what it is
  worth depends on the embedder and the data. Clamped to ≥ 0.
- `theme`: `"system"` (the default — follows the OS `prefers-color-scheme`),
  `"dark"`, `"light"`, or `"highviz"`.
- `autofind_detectors`: detector names to run on each web import, from a
  dataset's **Run AutoFind**, during `/api/auto-detect`, and in the CLI
  `--autodetect` flow, each mapping to a JSON file under `data/detectors/`.
  Every user curates their own list on the Dashboard's AutoFind detector tab
  (`PUT /api/detectors/registry/<id>/autofind`). `autofind_exporter` names the
  results exporter run afterwards (`""` = no auto-export; the CLI then falls
  back to the `gui` exporter), and `autofind_exporter_field_values` keeps each
  exporter's configuration around when the picker switches between them.
  `autofind_cli_delete_dataset` (default `false`) makes a CLI `--autodetect`
  run delete the dataset it imported once its detectors have run (see
  [`docs/CLI.md`](CLI.md#saving-the-dataset-to-the-dashboard---tempimport)).
  These four read through to `data/settings.json` for the `default` user.
  `autofind_on_import` (default `true`) is whether a web import runs them: the
  Add Dataset dialog's **Run AutoFind** checkbox starts from it and each import
  writes it back. `autofind_export_enabled` (default `true`) is whether a web
  run sends its results to `autofind_exporter`; `false` pauses the export
  without forgetting the exporter or its configuration.
- `import_alerts_enabled` (default `true`): whether the user wants to be
  alerted when an import they started finishes. It reaches the
  `--on-dataset-imported` hooks as `DatasetImported.notify`; a hook that emails
  should skip the user when it is `false` (see
  [EXTENDING.md § Dataset-Import Hooks](EXTENDING.md#dataset-import-hooks)).
  Settings › Import offers it only when hooks are configured.
- `grid_icon_size_*`, `focus_mode_*`, `panel_pct_*`, and the `browse_*` maps:
  per-media-type UI preferences, keyed by media-type id, so a user can tune
  audio and image datasets independently. Empty entries fall back to the
  frontend's per-type default.
- `import_defaults_by_media_type`: the embedder, clipper (+ params),
  converter rows, cleanup gates and ingest toggles (`build_projection`,
  `merge_near_duplicates`, each present only when on) to auto-fill into the
  Add Dataset advanced panel for each output media type, plus
  `multi_dataset`: the kinds (media type ids, e.g. `["image", "face"]`) that
  open the dialog in Multi-Dataset mode, filed under the type the import
  started from (a demo's own type, or the importer's media type). Set from
  Settings → Import, or by the **Save as default** offer after an import (the
  only way to save the cleanup gates or a `multi_dataset` list; Settings can
  only clear the latter).
- `settings_source` (not shown; excluded from the defaults endpoint): opt-in
  bidirectional sync. Set it to a plugin name + field values to auto-export
  every settings change and auto-import on the user's first settings access
  (again whenever the source changes). See `vtsearch/settings_io/sources/`.
