# VTSBrowse audit — queued fixes

Status: **planned, not yet applied** (paused behind an unrelated in-flight
`build_pyramid` auto-depth change in the working tree). Apply on a feature
branch off `dev`; the files below are disjoint from the auto-depth WIP, so they
commit cleanly on their own.

This came out of a live-driving + targeted-code audit of VTSBrowse (the hex-tile
UMAP browse view). The feature is well-built; these are the genuinely
worth-fixing items. Two agent claims were investigated and **rejected** (see
bottom) — don't re-raise them.

## Findings & fixes, in priority order

### 1. `clear_medias()` leaks the projection + tile pyramid  (HIGH value)

- **Where:** `vtscore/state/__init__.py` — `clear_medias()` (around lines
  132–138).
- **What:** The function's docstring says it drops caches "so RAM is released
  immediately," and it clears `medias`, `_emb_matrix`, `_emb_matrix_ids`,
  `diversity_tree`, and `dataset_display_name` — but it leaves `ctx._projection`
  (2-D coords) and `ctx._pyramid` (the **entire hex-tile pyramid — the largest
  Browse artifact**) resident.
- **Two consequences, both verified in code:**
  - *Memory:* on the RAM-tight box (~3.7 GB), unloading a dataset never frees
    its tiles.
  - *Correctness (latent):* the build route short-circuits on
    `if ctx._pyramid is not None:` at `vtsearch/routes/projection.py:86` and
    returns `{"status": "ready"}` **before** the signature check at line ~108.
    So if the same `dataset_id` context is reloaded with changed contents and
    the projection isn't rebuilt, the **stale pyramid is served** (hexes for the
    old data). Reload path: `clear_dataset` → `clear_all` → `clear_medias`
    (`vtscore/datasets/load_pipeline.py:152`); `_pyramid` is only re-set on load
    when `build_projection` is requested (`load_pipeline.py:1107–1108`).
- **Fix:** add to `clear_medias()`, inside the `_state_lock` block:
  ```python
  ctx._projection = None
  ctx._pyramid = None
  ```
- **Test:** add a regression test asserting `clear_medias()` nulls both — put it
  in a **state** test file (e.g. `tests_lib/` for the state core), **NOT** in
  `tests_lib/projection/test_pyramid.py` (that file has unrelated WIP). Build a
  context with a non-null `_pyramid`, call `clear_medias`, assert both `None`.

### 2. Hover preview/highlight goes stale on zoom  (MEDIUM)

- **Where:** `frontend/src/app/components/browse-canvas/browse-canvas.component.ts`
  — `zoomBy()` (~line 502), used by the +/- buttons and the mouse wheel.
- **What:** `zoomBy()` changes the transform and re-runs `updateActiveLevel()`
  (which can re-bin to a different hex level) and redraws, but never clears or
  re-evaluates hover state (`hoveredCell` / `hexHover` emit). Hover is only
  updated on `onCanvasMouseMove` (~526) or cleared on `onCanvasMouseLeave`
  (~559). So zooming with the cursor stationary leaves the "N items" tooltip
  pinned at its old screen spot and the hex highlight on stale `q/r` coords —
  which, after a wheel-driven level change, can point at the wrong/nonexistent
  hex. (Reproduced live: the "3 items" tooltip persisted across a button zoom.)
- **Fix (preferred):** track the last cursor position on `onCanvasMouseMove`;
  at the end of `zoomBy()` (after `updateActiveLevel()`), re-run `hitTest()` at
  that position and emit the updated `hexHover` (or `null`). Keeps the preview
  live and correct as you wheel-zoom.
- **Fix (minimal fallback):** clear hover at the end of `zoomBy()` —
  `this.hoveredCell = null; this.ngZone.run(() => this.hexHover.emit(null));`.

### 3. Frontend hardening bundle  (LOW–MED)

- **AbortController on hover-preview text fetch** —
  `frontend/src/app/components/browse-hover-preview/browse-hover-preview.component.ts:111`.
  `loadText()` uses bare `fetch()` with no cancellation; rapid hover can race
  responses (partially guarded by the `rep_id` check). Store an
  `AbortController`, abort the previous request when a new load starts, abort in
  `ngOnDestroy`.
- **`pollBuildStatus` retry hygiene** —
  `frontend/src/app/components/browse-view/browse-view.component.ts:259`. Uses a
  recursive `setTimeout(poll, 2000)` on error with no cap/backoff, not tied to
  `takeUntil(destroy$)`; a scheduled retry can fire after navigation. Convert to
  a `timer()`/`interval()` pipeline with `takeUntil(destroy$)`, add a retry cap
  + backoff.
- **Canvas listener cleanup** — `browse-canvas.component.ts:134–137`. The four
  canvas listeners are added with inline `.bind(this)` (fresh refs) and are not
  removed in `ngOnDestroy` (~158). Store bound handlers as fields (like the
  existing `boundMouseMove`/`boundMouseUp` at 85–86) and remove all four in
  `ngOnDestroy`. (Practical leak is small — the routed canvas DOM is torn down
  with the component — but the inline binds make proper cleanup impossible.)
- **Delete dead `onCanvasClick()`** —
  `browse-hover-preview.component.ts:129`. Confirmed zero callers (grep). Remove
  it (and any now-unused autoplay-unlock scaffolding) unless wiring it up is
  intended.

## Verification / process notes

- Base on `dev`, feature branch, PR → `dev` (repo convention; `main` is
  protected). Auto-open the PR when done.
- Local `./run-tests.sh` is blocked by the codespell/stray-`venv` gotcha; run
  targeted instead: `pytest tests_lib/projection` + the relevant state tests,
  `ruff check <changed>`, `codespell --toml pyproject.toml <changed>`, and
  `cd frontend && npm run build:prod` to typecheck + rebuild `static/`.
- Frontend serves the **pre-built** bundle in `static/`; rebuild after TS edits
  or the running app won't reflect them. The prod build needs ~700 MB free —
  stop the app first (CLAP holds RAM) on the tight box.

## Rejected agent claims (do not re-raise)

- *"Tile request storm / no dedup" (claimed HIGH):* **false.**
  `frontend/src/app/services/tile-cache.service.ts` has an `inflight` map +
  `shareReplay(1)` dedup and a 512-entry LRU (`evict()` by `lastAccess`). The
  ~26 tiles seen on one zoom were distinct visible+prefetch tiles, each fetched
  once. The only real (minor) gap is no **cancellation** of now-offscreen
  in-flight requests on rapid pan/zoom — LOW.
- *"Listener leak is HIGH":* **overstated** — see #3; real but low-impact
  because the routed component's DOM is destroyed with it.
