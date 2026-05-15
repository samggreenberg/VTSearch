# Codebase Reorganization Plan

**Status:** In Progress

This plan covers the next round of organizational changes to the VTSearch
codebase, following the test-suite reorganization (PRs around the
`claude/reorganize-codebase-structure-sk9Qt` branch). It tracks four
mid-sized refactors that are independent and can land separately.

The plan is deliberately scoped: it does **not** include the larger
greenfield reshape (splitting `models/` into training/embedding/detectors,
breaking up god route files, redesigning the frontend feature folders).
Those changes are non-trivial and should be planned in their own docs
once these foundations land.

## Completed (out of scope here)

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

## Open items

### #7 — Move `docker/`, `requirements/` out of the repo root

**Problem.** Repo root has 5 Dockerfiles, 4 docker-compose files, 6
requirements files. Plus 4 install scripts. Discovery is hard.

**Target.**
```
docker/
  Dockerfile
  Dockerfile.gpu
  Dockerfile.image-embedders
  Dockerfile.image-embedders.gpu
  Dockerfile.labbench
  compose/
    docker-compose.yml
    docker-compose.gpu.yml
    docker-compose.image-embedders.gpu.yml
    docker-compose.labbench.yml
requirements/
  base.txt           # current requirements.txt
  gpu.txt            # current requirements-gpu.txt
  image-embedders.txt
  image-embedders-gpu.txt
  plugins.txt
  labbench.txt
scripts/
  install-cpu.sh
  install-gpu.sh
  install-plugin-deps.sh
  download_models.sh
```

**Steps.**
1. `git mv` the files.
2. Update **inside each Dockerfile**: `COPY requirements.txt ./` →
   `COPY requirements/base.txt ./requirements.txt` (or change the
   working assumption).
3. Update each `docker-compose*.yml`: the `build.dockerfile` path needs
   the new location.
4. Update `README.md`, `docs/DEPLOYMENT.md`, and any CI config (the
   GitHub Actions workflows reference these by path).
5. Update CLAUDE.md commands section.

**Blast radius.** Self-contained but spreads to docs and CI. Test by
running `docker build` against each image variant before merging.

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

Each of those is worth its own dedicated plan once the foundations
above are in place.
