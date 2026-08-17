# Changelog: `vtsearch`

User-facing changes to the `vtsearch` Flask + Angular application. The
companion `vtscore` library has its own CHANGELOG at
[`vtscore/CHANGELOG.md`](vtscore/CHANGELOG.md).

`vtsearch` is **not** versioned with traditional semver. The `__version__`
attribute is the UTC timestamp of `HEAD`'s commit (ISO 8601, Z-terminated),
computed from git at import time in `vtsearch/__init__.py`. Every commit on
`dev` is effectively a new release; there is no tracked version constant to
bump and no per-release tag.

This CHANGELOG is therefore a **curated** record of notable changes and does
not list every commit. Use `git log` for the full history.

## Unreleased

### Fixed

- **Detector load can no longer hang forever at "Preparing".** Loading a
  detector while the selected dataset was not yet (re)loaded — typical right
  after an app restart, while the dataset was still being read from its
  pickle — left the dashboard row stuck at "Loading detector · Step 1 of 3 ·
  Preparing" indefinitely: Cancel did nothing, every retry reported "Detector
  load already in progress", voting returned 409, and only a restart
  recovered. The load now fails fast with a clean 409 (`dataset_not_loaded`)
  in that case, any other failure before the background worker starts
  releases the load reservation and surfaces an error on the row, and load
  failures are written to the server log. (#3139)

### Changed

- **Audio now defaults to the larger CLAP checkpoint.** New audio datasets and
  text queries use `clap_general` (`laion/larger_clap_general`, shown as "CLAP
  (general, larger)") instead of `clap`. It wins every measured retrieval
  comparison on ESC-50, at roughly 2.1x the embedding time. The old checkpoint
  is still selectable as "CLAP (general, faster)" for large collections where
  ingest speed matters more, and existing datasets and detectors built with it
  keep working. Cached demo-dataset pickles built with `clap` are re-embedded
  the next time they are loaded with the new default.
- **Library extracted.** The reusable core of VTSearch was carved out into a
  separate `vtscore/` package. The user-facing application surface (the Flask
  app, the Angular SPA, the settings system, the auth layer) is unchanged.
  Internally, every library-candidate import path moved from `vtsearch.<lib>`
  to `vtscore.<lib>`; `vtsearch/state/__init__.py` is now a thin app-tier shim
  that re-exports `vtscore.state` and layers the proxy view (`medias`,
  `good_votes`, …) on top. See
  [`vtscore/docs/architecture.md`](vtscore/docs/architecture.md) for the
  seven seams the refactor introduced.
- **Plugin entry-point groups renamed.** Library-tier plugin families now
  register under `vtscore.<family>` instead of `vtsearch.<family>`
  (`vtscore.importers`, `vtscore.label_importers`, `vtscore.labelset_sources`,
  `vtscore.media_sources`, `vtscore.converters`). Settings-related families
  remain under `vtsearch.<family>` because they stay app-side
  (`vtsearch.settings_importers`, `vtsearch.settings_exporters`,
  `vtsearch.settings_sources`). Third-party plugin authors targeting the
  library tier need to update their `pyproject.toml` entry-point group
  names.

### Added

- **Seed importers: a new plugin family for unlabeled seed media.** An
  external package can now contribute its own tab to the New Detector modal's
  **Blank** flow, beside Text and the media picker, by registering a
  `SeedImporter` in the `vtscore.seed_importers` entry-point group. Where a
  label importer imports media that already carry a good/bad verdict, a seed
  importer imports a *batch* of media with **no verdict** — items that are
  "close but not quite" what the user is hunting for. Seeds are stored on the
  detector as `{"type": "media", "value": …, "labeled": false}`: they steer
  the first sort (Autopilot ranks against the centroid of every media
  example) but never become a Good label or vote, so a detector seeded this
  way starts untrained. Nothing ships in-tree, so an install with no such
  plugin looks exactly as before. New endpoints: `GET /api/seed-importers`,
  `POST /api/seed-import/<name>`, `POST /api/seed-import/<name>/options`.
- **Server-side code can raise a toast.** A new `notify()`
  (`vtscore/concurrency/notifications.py`) lets any backend code — most
  usefully a plugin that hit a recoverable problem — tell the user something
  happened *without* failing the operation: "skipped 3 unreadable files",
  "the remote API rate-limited us, results are partial". The message is
  broadcast over a new `notification` channel on `/api/events` and rendered
  as a toast; toasts gained `warning` and `info` levels alongside the
  existing `error` and `success`. Plugin subclasses get `self.notify(...)`
  with their display name attached. Headless runs print the same messages
  (stderr in text mode, `notification` NDJSON records under
  `--progress-format json`). Delivery is live-only — there is no replay for
  a client that connects afterwards. See
  [`docs/EXTENDING-plugins.md`](docs/EXTENDING-plugins.md#notifying-the-user-toasts).

- **The app now tells you when your browser is running an out-of-date build.**
  `static/` is a build artifact that git does not track, so pulling new code and
  restarting the server used to leave the browser loading whichever bundle was
  last built — silently, since the version in Settings is the *server's* and
  looks current regardless. The bundle now carries the commit it was built from,
  and a mismatch raises a toast naming both versions and the rebuild command,
  plus a `⚠ bundle v …` chip beside the version in the Settings footer.

- `vtscore` library distribution with its own [README](vtscore/README.md) and
  [CHANGELOG](vtscore/CHANGELOG.md). See the
  [package reference](vtscore/docs/README.md#package-reference) for the
  documented public surface.
