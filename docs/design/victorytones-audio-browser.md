# Design: VictoryTones — a hover-to-hear Audio Browser on `vtscore`

> **Status:** Proposed (design only; no code yet). This doc scopes
> VictoryTones as a **second app tier** in the VTSearch repo, built on
> the existing `vtscore` library alongside `vtsearch`. It records the
> repo-structure decision, what VictoryTones reuses vs. omits, the
> frontend plan, and the `vtscore → vtsearch` back-edges that need
> attention. See *Open follow-ups* at the bottom.

## Problem / Goal

We want an **Audio Browser** ("VictoryTones") for quickly auditioning a
collection of audio. The interaction is the **hover-to-hear** affordance
from VTSearch's Find UI: as the cursor moves over items in the list/grid,
each one plays audibly — but with **no big center panel** (no waveform,
no transport, no detail view). You sweep the grid and hear the dataset.

It reuses the heavy lifting VTSearch already has:

- the **DatasetImporters** (load audio from folders, pickles, archives, …);
- the **audio MediaEmbedders** (LAION-CLAP, CLAP-Music, AST, Whisper) so a
  text query can seed an ordering of the grid, or items can be arranged by
  embedding similarity.

It deliberately **does not** want most of VTSearch's apparatus: no
LabelSets, no labels/votes, no detector training, no MLP, no eval, no
achievements. It is a browser, not a trainable searcher.

## Decision: in-repo second app tier (not a separate repo)

**Add `victorytones/` (an app tier) to this repo, consuming the same
`vtscore`.** Do *not* spin up a separate repository that "references
vtscore" — at least not now. Reasoning below.

### Why in-repo wins

1. **The reused code lives in `vtscore` and changes there.** Importers and
   audio embedders are exactly what VictoryTones leans on, and exactly the
   code most likely to evolve. Same-repo keeps both consumers in lockstep
   under one test run (`./run-tests.sh`); a separate repo guarantees
   version skew on the shared surface.
2. **The repo is already "library tier + app tier."** `vtscore` is the
   Flask-free library; `vtsearch` is the Flask app on top of it. Adding a
   *second* app tier that also consumes `vtscore` is the grain of the
   existing architecture — see `docs/ARCHITECTURE.md` §Directory map and
   §Dependency graph — not a fork of it.
3. **The hover-to-hear UI is not in `vtscore` at all.** It's Angular, in
   the app tier (`frontend/`). Either path reuses or re-implements
   frontend; same-repo lets VictoryTones share Angular components,
   services, and SCSS instead of copy-pasting them into another repo.
4. **Dropping labels/detectors is subtractive and trivial in-repo.** Those
   are per-**detector** concerns (`DetectorContext`: votes, training,
   model, threshold). VictoryTones simply never instantiates a
   `DetectorContext`; it uses only `DatasetContext` (`medias`) +
   embedders + media serving. Nothing has to be removed — it just isn't
   wired up.

### The fact that rules out "separate repo referencing vtscore" today

**`vtscore` is not currently a standalone, distributable package.** It is
Flask-free at *import* time (enforced by `./run-tests.sh vtscore-clean`),
but several `vtscore` modules carry **lazy runtime back-imports into
`vtsearch`**:

| `vtscore` module | imports from `vtsearch` (lazily) |
|------------------|----------------------------------|
| `datasets/load_pipeline.py` | `vtsearch.auth`, `vtsearch.state` |
| `datasets/ingest.py` | `vtsearch.state.next_media_id` |
| `detectors/workflow.py`, `media_seeding.py`, `labelset_elements.py` | `vtsearch.state` |
| `labels/sync.py` | `vtsearch.auth`, `vtsearch.achievements` |
| `concurrency/async_jobs.py` | `vtsearch.auth` |
| `exporters/_template.py` | `vtsearch.auth` |
| `cli.py` | `vtsearch.achievements` |
| `embedding/loader.py` | `vtsearch.logging_config` (optional bridge) |

So "a separate repo that *references* vtscore" presumes an artifact that
doesn't exist yet. To make it real you'd have to either vendor the **whole**
VTSearch repo as a git dependency (you inherit `vtsearch` anyway), or first
**sever those back-edges and publish `vtscore` to PyPI** — a real project
that an audio browser does not by itself justify.

Notably, the paths VictoryTones touches most are in that table:
`datasets/load_pipeline.py` (dataset loading) reaches into `vtsearch.auth`
and `vtsearch.state`. So even the in-repo version benefits from tidying
those edges (see *vtscore back-edges* below), but in-repo it keeps working
as-is because `vtsearch` is present.

### When to revisit (separate repo / published `vtscore`)

Reconsider extraction the day independent distribution becomes a concrete
requirement: a separate release cadence, a separate deploy target, a
different owning team, or external consumers who should `pip install
vtscore`. At that point the right sequence is **decouple + publish
`vtscore` first, then build VictoryTones against the package** — and that
work pays for itself because you'd need it regardless. Until then, in-repo
avoids paying the decoupling tax up front for a benefit you don't yet need.

## What VictoryTones reuses (from `vtscore`)

- **Dataset importers** — `vtscore/datasets/importers/*` (server_folder,
  local_folder, pickle, http_archive, demo, …) via the `PluginRegistry`.
  Unchanged.
- **Audio media type + embedders** — `vtscore/media/audio/*`
  (`embedder_clap`, `embedder_clap_music`, `embedder_ast`,
  `embedder_whisper`) via the media/embedder registries in
  `vtscore/media/__init__.py`. Unchanged.
- **Media loading + embedding** — `vtscore/datasets/loader*.py`,
  `vtscore/embedding/{helpers,matrix,loader}.py`.
- **Per-dataset state** — `vtscore/state/core.py` `DatasetContext`
  (`medias`) and the embedding matrix. No `DetectorContext`.
- **Text-seeded ordering** — `embed_text_query` + cosine against the
  dataset embedding matrix to order the grid by similarity to a text query
  (the "quick stand-alone search" half of VTSearch's semantic sort, minus
  the detector seeding).
- **Concurrency/progress** — `vtscore/concurrency/{progress,async_jobs}.py`
  for background dataset loads.

## What VictoryTones omits (vs. `vtsearch`)

- **No LabelSets / labels / votes** — no `good_votes`/`bad_votes`, no
  `label_history`, no `LabelSet`, no label importers/exporters.
- **No detectors / training** — no `DetectorContext`, no
  `vtscore/training/*` (MLP, thresholds), no `vtscore/detectors/*`
  workflow, no scoring routes.
- **No eval** — no `vtscore/eval`.
- **No achievements**, no detector/labelset persistence.
- **No center panel** — no waveform render, no transport, no detail view.

### Package name

The package is `victorytones/` (not `vtvictorytones/`). The `vt` prefix on
`vtscore`/`vtsearch` (and the `vt-` Angular selectors) plausibly derives
from **V**ictory**T**ones itself, so `vtvictorytones` would read as
"VictoryTones VictoryTones." VictoryTones is its own product, so it takes
the brand name directly rather than a `vt`+role name like `vtsearch`. (If
family-grouping under `vt*` ever matters more than the brand, `vtbrowser`
was the runner-up.)

## App-tier shape (`victorytones/`)

A thin Flask app paralleling `vtsearch`, importing only the `vtscore`
slices above. Minimum surface:

- `GET /api/medias` — list items in the loaded `DatasetContext` (id, name,
  metadata, audio URL).
- `GET /api/medias/<id>/audio` — stream audio bytes (mirror of the
  existing media-serving route, audio-only).
- `POST /api/datasets/load` (or registry load) — load a dataset via an
  importer / pickle, populate a `DatasetContext`.
- `POST /api/order` (optional) — text query → ordered list of media ids by
  embedding similarity. No persistence, no votes.

It can borrow `vtsearch`'s settings/auth tiers if multi-user is wanted, but
the default is single-user (`DefaultLoginProvider`) and a single loaded
dataset — much smaller than VTSearch's multi-dataset/multi-detector
context machinery.

## Frontend plan

The hover-to-hear behavior already exists and is **decoupled from
voting**:

- `frontend/src/app/components/left-panel/media-item/media-item.component.ts`
  — `@Input() focusMode: 'click' | 'hover'`; `onMouseEnter()` emits a
  `select` when `focusMode === 'hover'`. (`media-list.component.ts` and
  `left-panel.component.ts` thread the same `focusMode` input.)
- Audio playback lives in the center panel's player component and is keyed
  off the selected media — **no label/vote coupling**.

VictoryTones reuses the **left panel grid + media-item hover→select**
wiring but **drops the center panel entirely**. Instead of "select →
load center panel → play," the flow is "hover → play the hovered item's
audio directly." Concretely:

- Reuse the media-list/media-item grid (icon-size, view-mode controls).
- Replace the center-panel player with a lightweight hover-driven
  `HTMLAudioElement` (or small Web Audio wrapper) that plays the hovered
  item and stops/replaces on the next hover — no canvas, no transport.
- Keep an optional text-query box that calls `/api/order` to reorder the
  grid.
- Drop label-view, vote controls, right-panel detector UI, achievements,
  dashboard.

This can be a **new Angular app** in `frontend/` (second build target) or
a separate minimal frontend that imports the shared components/services.
Decide at scaffold time; either way the shared bits stay in one repo.

(Per repo policy: desktop-only; no mobile/responsive concerns.)

## `vtscore` back-edges to address

Even in-repo, the dataset-load path VictoryTones depends on
(`datasets/load_pipeline.py`, `datasets/ingest.py`) reaches into
`vtsearch.auth` / `vtsearch.state`. Two acceptable approaches:

1. **Leave as-is for v1.** Because VictoryTones is in the same repo,
   `vtsearch` is importable, so the lazy back-imports resolve. Cheapest;
   defers the cleanup.
2. **Parameterize the seams.** Where `vtscore` reaches back for user
   context (`get_current_user`), id allocation (`next_media_id`), or the
   active context, accept these as injected callables/params instead of
   importing `vtsearch`. This is the same work a future published
   `vtscore` needs, done incrementally and motivated by a real second
   consumer.

Recommendation: start with (1) to ship the browser, but treat each
back-edge VictoryTones actually exercises as a candidate for (2) — it
both de-risks a future extraction and removes a hidden `vtsearch`
dependency from the load path.

## Open questions (resolve before/at scaffold)

- **Second Angular app vs. shared-component minimal frontend?** Affects
  build config and how much SCSS/component sharing is practical.
- **Single dataset vs. multi-dataset?** Single keeps the app tiny;
  multi reuses VTSearch's registry/context headers.
- **Hover playback policy** — debounce/leading-edge, overlap vs.
  hard-cut, volume from settings? (UX detail, not architecture.)

## Open follow-ups

- Nothing shipped yet — this is a proposal. When the first slice lands,
  add a *What shipped* section and update the status header.
- If/when independent distribution is required, open a companion plan for
  **`vtscore` decoupling + PyPI publish** (sever the back-edges in the
  table above) and migrate VictoryTones to depend on the package.
