# Codebase Reorganization Plan

**Status:** Complete

This plan covered the round of organizational changes to the VTSearch
codebase that followed the test-suite reorganization (PRs around the
`claude/reorganize-codebase-structure-sk9Qt` branch). All five
mid-sized refactors tracked here have landed.

The plan was deliberately scoped: it does **not** include the larger
greenfield reshape (splitting `models/` into training/embedding/detectors,
breaking up god route files, redesigning the frontend feature folders).
Those changes are non-trivial and should be planned in their own docs
now that these foundations have landed.

## Completed

- ✅ `vtsearch/medias.py` → `tests/fixtures/medias.py` (commit ce905b4)
- ✅ Test suite bucketed into `tests/<group>/` folders, path-based marker
  derivation, 11 previously-unmapped files now correctly grouped (commit ae6af11)
- ✅ Shared text-embedder stubs extracted from download tests (commit c9ad6aa)
- ✅ #8 — `docs/plans/sync-sources.md` deleted (design absorbed into
  `EXTENDING-plugins.md`); `docs/plans/README.md` index added.
- ✅ #9 — `PatchEmbedOutput` and friends moved out of `vtsearch/models/`
  into `vtsearch/media/` (commit d101f9d).
- ✅ #5 — Routes grouped by domain under `vtsearch/routes/<resource>/`
  (commit 66d3ffe).
- ✅ #6 — `vtsearch/utils/` split into `vtsearch/state`, `vtsearch/plugins`,
  `vtsearch/sync`, `vtsearch/concurrency`, `vtsearch/security`, and
  `vtsearch/media/audio`. `utils/` now only hosts `hits.py` and the
  `synthetic/` media generators. Progress-module merge with
  `vtsearch/models/progress.py` deferred to a follow-up commit.
- ✅ #7 — `docker/`, `requirements/`, and install scripts moved out of
  the repo root. Dockerfiles live under `docker/`, compose files under
  `docker/compose/`, requirements files under `requirements/` (renamed
  to `base.txt` / `gpu.txt` / `plugins.txt` / `labbench.txt` /
  `image-embedders[-gpu].txt`), and install / download scripts moved to
  `scripts/`.

## Explicitly NOT in this plan

- Splitting `models/` into `training/`, `embedding/`, `detectors/`, `sampling/`.
- Splitting the 1000+ LOC route files (datasets.py, detectors.py, sorting.py, medias.py).
- Splitting the monolithic media-type files (image/media_type.py is 1641 LOC).
- Unifying the three plugin-discovery patterns (sentinel scan in
  `media/__init__.py`, PluginRegistry, in-memory JSON catalogs).
- Frontend reorganization (god components, services-by-type, missing
  `shared/` folder, dead `.spec.ts` files).
- CLI package split (`vtsearch/cli.py` → `vtsearch/cli/`).
- Collapsing `vtsearch/auth/` — explicitly NOT recommended. Auth
  expects more login providers (Google, SAML, etc.) and the folder
  is the right home for them following the same plugin pattern used
  in `labels/sources/` and `settings_io/sources/`.

Each of those is worth its own dedicated plan now that the foundations
above are in place.
