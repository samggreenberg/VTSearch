# Code-structure review

**Status:** Review complete; acting on **Theme A** (the `vtscore` ↔ `vtsearch`
boundary) first. The other themes are scoped below as a prioritized backlog
for later passes. See "What shipped" + "Open follow-ups" at the bottom for
live status.

This is a systematic, repo-wide structural review: where have design
decisions that were right at small scale been outgrown, and what is worth
streamlining, abstracting, or reorganizing? The codebase is healthy and
unusually well-documented; the findings below are **accretion** problems
(modules that started focused and absorbed adjacent responsibilities), not
rot. Nothing here is urgent. Themes are ordered by leverage-to-effort.

---

## Theme A — The `vtscore` ↔ `vtsearch` boundary has eroded (highest leverage)

The architecture's central promise (`docs/ARCHITECTURE.md`) is a Flask-free
library tier (`vtscore`) under a thin Flask app tier (`vtsearch`). The seam
leaks in **both** directions, which undermines testability, reuse, and the
library-extraction story the whole architecture doc is built around.

### A1. Fat controllers — domain logic living in route handlers

Private helper-function count per route file (proxy for misplaced domain
logic; these are not request-parsing helpers):

| Route file | Lines | Private helpers |
|---|---|---|
| `vtsearch/routes/sorting.py` | 1032 | 19 |
| `vtsearch/routes/detectors/registry.py` | 1220 | 8 |
| `vtsearch/routes/detectors/find.py` | 592 | 18 |
| `vtsearch/routes/detectors/scoring.py` | 835 | 6 |
| `vtsearch/routes/projection.py` | 606 | 18 |

Concrete examples of business logic that belongs in `vtscore`:

- `sorting.py`: `_resolve_labelset_local_state`, `_model_matches_local_votes`,
  `_update_det_ctx_with_trained_model`, `_build_learned_sort_signature` —
  the learned-sort / vote-reconciliation pipeline.
- `detectors/registry.py`: `_maybe_start_label_reembed` (embedder-mismatch
  detection + background re-embed task orchestration).
- `detectors/scoring.py`: `_resolve_or_train_detector` (cold-path
  train-on-demand, model selection, embedder-mismatch defense).

**Fix:** extract per area into the library — `vtscore.detectors.model_loading`
(from `scoring.py`), `vtscore.detectors.embedder_sync` (from `registry.py`),
and a learned-sort orchestration module (from `sorting.py`). Routes become
request↔library glue. Each extraction makes the logic unit-testable without a
Flask client.

### A2. Library importing the app (the inverse leak)

`vtscore/detectors/workflow.py` imports `flask.g`-backed state from
`vtsearch.state` (`good_votes`, `bad_votes`, `apply_label`,
`snapshot_medias`, `override_detector_context`). A library module reaching
back into the request tier is the inverse violation of A1.

**Fix:** parameterize `apply_and_retrain()` — pass vote dicts / media
snapshot in explicitly; move the `override_detector_context` wrapper up into
the route handler (`vtsearch/routes/detectors/labels.py`).

### A3. Two training pipelines

`vtscore/detectors/training.py` and `vtscore/detectors/labelset_training.py`
both implement resolve → build X/y → threshold → train → score.
`labelset_train_and_score` re-implements rather than reuses `train_and_score`,
and patch-region pooling (`_training_vec_for_vote` vs `_pool_box_from_media`)
is ~90% duplicated.

**Fix:** a shared training-data-builder seam (votes vs labelset both yield
`(X_list, y_list)`); one scoring function parameterized by data source;
consolidate patch pooling into one helper.

---

## Theme B — Mega-files mixing unrelated concerns

The common shape: a module that started focused and absorbed adjacent
responsibilities.

- **`app.py` (1408 lines).** ~625 lines (783–1408) are the `__main__`
  argparse block + autodetect dispatch; ~180 lines (596–776) are Linux
  `/proc` port-preflight helpers. Neither relates to the WSGI `app` object
  gunicorn imports. Extract `vtsearch/cli_main.py` (argparse/dispatch) and
  `vtsearch/app/server.py` (port preflight); optionally `hooks.py` +
  `errors.py` for the request lifecycle and error handlers.
- **`vtsearch/settings.py` (1887 lines).** Conflates cache state,
  cross-process file locking, two-tier routing, a one-shot legacy migration,
  bidirectional sync state-machines, and dynamic accessor generation. The
  lock-ordering discipline (`file_lock → settings_lock`, re-entrance guards)
  would be safer encapsulated in a `UserSettingsStore`; the legacy-migration
  path could move to a one-shot script.
- **`vtscore/datasets/load_pipeline.py` (1588 lines).** Six concerns:
  `ConcurrencyGate`, progress/step mapping, clipper-chain fixup, embedding,
  dedup/diversity, registry/migration, background-task orchestration. Move
  `ConcurrencyGate` to `vtscore/concurrency/`; split post-import stages into
  a `stages/` package.
- **`vtscore/datasets/importers/base.py` (1202 lines, 35 methods).** A
  trivial single-API-call importer inherits all the multi-media-spec /
  converter / ingestion machinery. Split a thin `ImporterBase` from a
  `MultiMediaImporter` mixin.
- **`vtscore/state/core.py` — `DetectorContext` (31 fields).** A god-object
  spanning vote state, training artifacts, cached embeddings, and
  Find-session state. Splitting into `VoteState` / `TrainingState` /
  `FindSessionState` is the right shape but touches ~346 call sites —
  expensive; defer until already in that code.
- **Frontend:** `dataset-importer-modal.component.ts` (2162; 17 state slices
  + HTTP + 5 picker views) and `browse-canvas.component.ts` (1881; ~200 lines
  of pure pan/zoom/clamp/rubber-band geometry that should be a framework-free
  `ViewTransformService`).

---

## Theme C — Repetitive boilerplate a declarative/table-driven approach collapses

Recurring smell: the **same fact declared in N parallel places**, kept in
sync by hand.

- **Settings keys live in 3+ places.** 48 keys are each a Pydantic field
  (`settings_models.py`), a dynamic-accessor entry (`settings.py`), and a
  route-dispatch entry (`routes/settings/api.py` `_SCALAR_SETTERS` /
  `_CUSTOM_SETTERS`), plus `TYPE_CHECKING` stubs. A single declarative
  `SettingSpec` registry (name, tier, type, default, route-handler) that
  generates the rest is the highest-value cleanup in this theme. The
  `_make_per_side_setting` factory (left/right variants) is a smaller
  instance of the same fix.
- **Embedder duplication.** The three CLAP audio embedders (`embedder_clap`,
  `_general`, `_music`) are ~90% identical, differing by a model-ID constant
  — a `_ClapBase` mixin (the pattern `_Dinov2Base` already uses) collapses
  them. Image embedders have single/patch pairs that could share a
  `_SinglePatchBase`; SigLIP/SigLIP2/CLIP have no shared base despite
  identical cross-modal load/warm-up.
- **State proxies.** 7 copy-paste `_ProxyDict` / `_ProxyList` declarations in
  `vtsearch/shim/state_proxies.py` could be built from a registry table.
- **Downloaders.** `downloader/{audio,image,video,text,docs}.py` each
  re-implement check → download → extract → post-process (~500 lines of
  duplicated zip/tar iteration); a `_DatasetDownloader` base with
  `post_process()` hooks consolidates it.

---

## Theme D — Schema/type drift across boundaries

The same data shapes are independently re-declared in up to four type
systems with nothing enforcing agreement:

- Backend: **Pydantic** (`settings_models.py`) + **Marshmallow**
  (`vtsearch/schemas/`, 3983 lines) for the *same* settings.
- Frontend: **both** a generated OpenAPI client (`ng-openapi-gen` →
  `src/app/generated/api-client`, imported by 53 files) **and** a
  hand-written `api.models.ts` (27 interfaces, imported by 69 files). The
  hand-written types shadow concepts the generated client could cover, and
  several use `[key: string]: unknown` escape hatches that silence drift.

**Fix:** migrate hand-written frontend types onto generated ones where the
endpoint is covered; consider Pydantic→Marshmallow generation server-side.
*Decline* adding `zod`/`io-ts` runtime validation — heavier than the drift
risk warrants.

---

## Theme E — Naming / conceptual overlaps that mislead readers

- **"shim" is misnamed.** `vtsearch/shim/state_proxies.py` is the canonical
  app-tier state API (the proxy layer), not a thin adapter. Rename to
  `vtsearch/state_proxies.py` and drop "shim" from docstrings.
- **`detectors/` label-module proliferation.** `label_sync`,
  `label_restoration`, `labelset_elements`, `labelset_training`,
  `labelset_rename` (plus `datasets/labelset.py` and the separate `labels/`
  *plugin* package) blur "labels, the detector concept" vs "labels, the
  import/export plugin family." A `labelset_ops` facade gives callers one
  import instead of five.
- **Three overlapping ingestion concepts:** `MediaSource`,
  `DatasetImporter`, and the bare loader functions all mean "get media in,"
  but importers sometimes use a `MediaSource` and sometimes call loaders
  directly, and `FetchedItem.embedding` is bypassed by loader override
  dicts. Draw the boundaries explicitly (or fold `MediaSource` into
  importers).

---

## Theme F — Abstractions that aren't paying for themselves

- **`SyncSource[LoadT, SaveT]`** (`vtscore/sync/`) has two subclasses, each
  with one concrete implementation. Don't add more indirection here until a
  third consumer appears.
- **`PluginField` aliases.** Six no-op aliases (`ImporterField = PluginField`
  …) imply field types differ per family. Collapse to `PluginField`.
- **`loader.py` façade** re-exports its three sibling loaders without adding
  abstraction.

---

## Evaluated and deliberately **not** touching

The plugin auto-discovery core; the dual `flask.g` / thread-local resolution
mechanism (necessary for the library split); the single `_state_lock` (no
contention symptoms); the frozen request-missing sentinels; the
`concurrency` / `eval` / `security` / `utils` packages (well-sized); and the
route `_shared.py` helpers are all sound. Also declining the proposal to
rename every plugin family's `run()`/`export()`/`load()` to a uniform
`execute()` — domain-meaningful names, churn dwarfs benefit.

---

## Prioritized backlog

1. **Theme A** — extract fat-controller logic into `vtscore`; de-couple
   `workflow.py` from `flask.g`. (In progress.)
2. **Theme C quick wins** — settings `SettingSpec` registry; `_ClapBase`
   mixin; per-side settings factory.
3. **Theme E** — "shim" rename; `labelset_ops` facade.
4. **Theme B** — split `app.py`, `load_pipeline.py`, `settings.py`,
   `importers/base.py`.
5. **Theme D** — converge frontend types onto the generated client.
6. **Theme F** — collapse `PluginField` aliases; revisit `SyncSource` if a
   third consumer appears.
7. **Theme B (deferred)** — `DetectorContext` sub-context split (~346 call
   sites; do opportunistically).

---

## What shipped

_(nothing yet)_

## Open follow-ups

All themes above except the slice of Theme A currently in progress.
</content>
