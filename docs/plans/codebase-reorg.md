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

## Open items

### #5 — Group routes by domain (`routes/<resource>/`)

**Problem.** `vtsearch/routes/` is a 27-file flat folder. Files like
`detectors.py`, `detectors_registry.py`, `detector_scoring.py`,
`detector_find.py` look like duplicates at a glance even though each
handles a distinct facet. Suffix grammar is invented per-domain
(`_registry`, `_crud`, `_scoring`, `_find`, `_ui`, `_io`) with no
consistency. `helpers.py` (209 LOC) tells you nothing from its name.

**Target.**
```
vtsearch/routes/
  datasets/    __init__.py  crud.py  registry.py  ui.py  loading.py
  detectors/   __init__.py  store.py  registry.py  scoring.py  find.py
  processors/  __init__.py  crud.py  scoring.py
  media/       __init__.py  list.py  server.py  embed.py
  settings/    __init__.py  api.py  io.py  sources.py
  labels/      __init__.py  vote.py  importers.py  exporters.py
  sorting.py   eval.py  achievements.py  auth.py  main.py  file_browser.py
  _shared.py   # current helpers.py, renamed for clarity
```

**Steps.**
1. Create the six new folders with `__init__.py` re-exports.
2. `git mv` each route file to its new home; rename inside the new folder
   to drop the redundant domain prefix (`detectors_registry.py` →
   `detectors/registry.py`).
3. Update the blueprint imports in `vtsearch/routes/__init__.py` and in
   `app.py`.
4. Rename `helpers.py` → `_shared.py`.

**Blast radius.** `routes/__init__.py`, `app.py`, and any code that
imports a route module directly (tests sometimes do for unit testing).
Estimate ~50 import updates. Pure mv + import-rewrite; no behavior change.

**Why not done yet.** Needs a dedicated PR because the move touches
~30 files and is easier to review on its own.

### #6 — Split `vtsearch/utils/` into purpose-named packages

**Problem.** `utils/` is two unrelated things stapled together:

1. Seven `state_*.py` files + `state.py` (≈1850 LOC) that own the
   `DatasetContext` / `DetectorContext` proxy machinery — this is the
   **core application state**, not utilities.
2. A grab-bag of framework primitives: `registry.py` (the PluginRegistry
   base every plugin system uses), `sync_source.py` (foundation of
   sync sources), `async_jobs.py`, `progress.py`, `memory_budget.py`,
   `audio_generator.py`, `ffmpeg.py`, `hits.py`, `url_validation.py`,
   `paths.py`.

**Target.**
```
vtsearch/state/         # state_*, locks, proxies, contexts
vtsearch/plugins/       # PluginRegistry + sentinel discovery
vtsearch/sync/          # SyncSource base, shared by settings & labels
vtsearch/concurrency/   # async_jobs, ConcurrencyGate, memory_budget, progress
vtsearch/security/      # url_validation, pickle_security (move from datasets/)
                        # path_validation (move from utils/paths.py)
vtsearch/media/audio/   # audio_generator.py, ffmpeg.py (move from utils/)
```

`utils/` shrinks to 2-3 genuinely tiny pure helpers, or disappears
entirely. There are also **two `progress.py` files** today
(`utils/progress.py` 437 LOC and `models/progress.py` 745 LOC) that
should be unified under `concurrency/`.

**Steps.**
1. Create new packages.
2. Move files, leaving thin `vtsearch/utils/__init__.py` re-exports
   only for symbols imported from many places (`medias`, `good_votes`,
   etc. — the proxy attributes the codebase depends on).
3. Update imports across the codebase (~150 sites).
4. Merge the two `progress.py` modules.

**Blast radius.** Larger than #5. The state machinery is imported
broadly. The merge of the two progress modules has the most risk —
both files define overlapping concepts (`ProgressState`, cancel flags,
loading-tasks tracker). Plan to do the move first, the merge in a
follow-up commit.

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

### #8 — Delete `docs/plans/sync-sources.md` (implemented), add `docs/plans/README.md` index

**Problem.** `docs/plans/` has 7 files. They all have explicit
"Status:" headers, but:
- `sync-sources.md` — Status: Implemented. The feature ships. The plan
  is now reference material at best; should be deleted or its useful
  bits absorbed into `docs/EXTENDING.md` (the sync-source pattern).
- The rest are still active (combine-models-ui frontend in progress;
  delete-detectors, extract-library, multi-media-import,
  patch-embedder, RCDatasetImporter all proposed/in-progress).
- There's no index — a user has to read the directory listing to know
  these plans exist.

**Steps.**
1. Audit `sync-sources.md` for content worth preserving; fold any
   still-relevant design notes into `docs/EXTENDING-plugins.md` (which
   already documents the sync-source plugin pattern), then delete the
   plan file.
2. Create `docs/plans/README.md` listing the open plans with a
   one-line status for each.
3. Audit each remaining plan; mark stale ones for follow-up.

**Blast radius.** Docs only. Low risk.

### #9 — Fix `media/` → `models/` backwards import (`PatchEmbedOutput`)

**Problem.** Six files under `vtsearch/media/image/` import
`PatchEmbedOutput` (a dataclass describing patch-embedding output) and
related helpers from `vtsearch/models/patch_regions.py`. This is
backwards layering: `media/` is the lower layer and `models/` should
depend on it, not vice versa.

Specifically:
```
media/image/_dinov3_shared.py       → models/patch_regions.py
media/image/embedder_dinov3_patch.py → models/patch_regions.py
media/image/_dinov2_shared.py       → models/patch_regions.py
media/image/embedder_dinov2_patch.py → models/patch_regions.py
media/image/embedder_eupe_patch.py   → models/patch_regions.py
media/image/_eupe_shared.py          → models/patch_regions.py
media/embedder.py                    → models/patch_regions.py, models/loader.py
```

`media/embedder.py` also pulls in `ensure_torch_configured` from
`models/loader.py`.

**Target.** Move `PatchEmbedOutput` and any related shared types into
`media/` (likely `media/patch_embed.py` or beside `media/embedder.py`),
and move `ensure_torch_configured` into `media/embedder.py` or a small
`media/torch_setup.py`. Update the consumers.

**Steps.**
1. Move the dataclass + helpers to `media/`.
2. Update the 7 import sites.
3. Leave a thin `models/patch_regions.py` re-export only if there are
   external consumers (probably none). Otherwise delete that module
   and update its callers in `models/` (or move it to `media/` if
   `models/` is the only consumer of the *runtime* code, which is
   likely).

**Blast radius.** Small. 7-10 import updates plus moving a single
module. Validate by running `./run-tests.sh detectors` (which covers
the patch embedders) plus the GPU-marked `test_gpu.py` if a GPU is
available.

## Ordering

These four are independent. Suggested merge order:

1. **#8** (docs cleanup) — smallest, lowest risk, fastest win.
2. **#9** (PatchEmbedOutput move) — small, fixes a real layering bug.
3. **#7** (docker/requirements out of root) — touches CI + docs but
   self-contained.
4. **#5** (route-folder bucketing) — bigger PR but pure mv + import
   updates.
5. **#6** (utils split) — last, because it's the most invasive and
   benefits from doing #5 first (route files can move to their new
   homes without the simultaneous utils rename).

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
