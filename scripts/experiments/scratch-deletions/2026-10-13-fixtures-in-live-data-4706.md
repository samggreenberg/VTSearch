# 2026-10-13: 2.9 MB of test fixtures in the live `vtsearch-data` (#4706, owner-approved)

Until #4706's fix (PR #4823), GRID suite runs inherited `VTSEARCH_DATA_DIR` from `~/.bashrc`, and that names the live app's
data. Every test that wrote under `config.DATA_DIR` wrote there, so these were deleted:

- **336 `autodetect_results_*.json`** at the top level. They are the `server_json_file` exporter's default output, and
  every one held fixture results (`cat_meow`, `dog_bark`).
- **192 `local_uploads/local_folder_*`** dirs, each holding exactly `f/a.wav`.
- **2 `staging/stage_*.pkl`**, from a suite run at 03:53 that day.

Why it was safe:
- Nothing reads them. The exporters only write `autodetect_results_*`. No JSON or pickle in `vtsearch-data` names
  `local_uploads` or either stage id. Staging pickles are transient files for the combine flow.
- No app job was running.
- Each file was re-checked as a fixture just before the `rm`.

`local_uploads/`, `staging/` and everything else in `vtsearch-data` stayed. That includes the registries, `detectors/`,
`models/`, the settings files, and the empty `ESC-50-master/` that suite runs create. Nothing to rebuild.
