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

- `vtscore` library distribution with its own [README](vtscore/README.md) and
  [CHANGELOG](vtscore/CHANGELOG.md). See the
  [package reference](vtscore/docs/README.md#package-reference) for the
  documented public surface.
