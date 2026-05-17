# Active-context switcher (top-bar dataset/detector pulldowns)

*Status: Phase 1 + 2 shipped. Phase 1 covered the switcher UI, "+ Add New" footers, incompatible-pair explainer, and the five follow-ups (focus-other-pulldown button, auto-select new item, deleted-item toast, registry-error retry, Dashboard-sort mirroring). Phase 2 made the URL authoritative for the active pair — `/label/:datasetId/:detectorId` and `/find/:datasetId/:detectorId` with an `activeContextGuard` that validates, flips, and loads before the route activates. Phase 3 (in-flight job affordances) and one Phase 1 follow-up (embedder progress message) remain deferred — see "Open follow-ups" below.*

This plan implements [ux-brainstorm.md §6.11](ux-brainstorm.md#611-active-datasetdetector-indicator--s) ("Active-dataset/detector indicator") and largely closes [§8.2](ux-brainstorm.md#82-switching-active-datasetdetector--s) ("Switching active dataset/detector"). It supersedes both entries as the canonical design — when this plan ships those entries will be marked SHIPPED and link here.

## Problem

The top bar at `frontend/src/app/app.component.html:62-63` already shows the active dataset and detector as **read-only text** (`Data: foo` / `Detector: cats`) driven by `ActiveContextService`, which is the same source that drives the `X-Dataset-Id` / `X-Detector-Id` headers sent on every API request.

What it doesn't do:

1. Let the user **switch** the active pair without going back to the Dashboard. Switching today is: navigate to Dashboard → reselect → click Train (or Find). The §6.11 promise was a 2-click pulldown change from the label view.
2. Let the user **create** a new dataset / detector without that same Dashboard round-trip.
3. Let the URL **encode** the active pair. Reloading `/label` doesn't reliably restore the previous pair; pasting `/label` to a colleague doesn't share "Train on DataA / DetectorB" (Phase 2 territory).
4. Let the user **see** that a long-running job is in flight on a non-active pair (Phase 3 territory).

The §6.11 "indicator" half is already shipped via the existing read-only text. This plan delivers the missing **switcher** half plus the two related affordances (URL encoding, in-flight job visibility) that came up while designing it.

## Design summary — rules that govern every behavior in the plan

These came out of the design discussion and are repeated here so a fresh Claude implementing any phase can resolve ambiguity without re-deriving them.

1. **Switcher = Dashboard shortcut, nothing less.** Clicking a pulldown entry is exactly equivalent to "go back to Dashboard, select that item, click Train/Find." Full prep fires — dataset load, detector load, labelset re-resolve against the new embedder, threshold recompute, etc. We **do not** skip any prep just because the user used the shortcut.
2. **Show everything, don't filter.** The pulldown list mirrors the user's Dashboard grid exactly — same items, same order. Incompatible items (different `media_type` than the other half of the pair) are **dimmed/desaturated**, not hidden, so the user's mental map of "what exists" doesn't shift between surfaces.
3. **No auto-swap.** Picking an incompatible item never silently swaps the other half "to fix it." Instead the view below collapses to an explainer and the user picks the fix themselves.
4. **No prep on an incompatible pair.** When the active pair is incompatible, the standard Train/Find prep **does not fire**. We don't want to load a dataset or materialize a labelset just to immediately invalidate it. Prep only runs once both halves are compatible.
5. **Replace, not widen.** The switcher in Find re-launches single-pair Find on the new pair — it does *not* add a dataset to a running multi-dataset Find. (Multi-N Find lives on the Dashboard via `onOldFind` and is out of scope for this plan; the `/find` route itself is always single-pair.)
6. **Reset view-local state on switch.** Scroll position, partial Find query input, etc. all reset. Switching is mentally a re-entry to the view, like going through the Dashboard would be.
7. **Background-continue in-flight jobs (Phase 3).** Jobs run against a `DetectorContext`/`DatasetContext` regardless of which one the UI is looking at, and `JobManager` caches results by signature. So switching away doesn't waste work — the result lives in the cache and the view rehydrates on re-entry. Phase 3 wires the visibility (spinner glyph) + verifies every job-producing view actually rehydrates.
8. **Honest loading state.** A switch that triggers 30s of prep should look like 30s of prep — same loading affordance the Dashboard route shows today. When the new dataset's embedder differs from the previous dataset's, the message reads "Re-resolving labels for X's embedder…" rather than a generic spinner.
9. **Desktop only.** No responsive / mobile design. Truncate-with-ellipsis on the closed state, widen the menu on open. (See `CLAUDE.md` § "Frontend Scope: Desktop Only".)

## Architecture — where each piece lives

- **`frontend/src/app/services/active-context.service.ts`** — Today holds `datasetId` / `modelId` as `BehaviorSubject`s and is read by `activeContextInterceptor`. Phase 1 extends it with `setActivePair(datasetId, detectorId)` so the switcher writes both atomically (avoids a transient mismatched pair fighting through the interceptor). Phase 2 drives the subjects from the URL via a route param.
- **`frontend/src/app/app.component.html` + `.scss` + `.ts`** — Replaces the two read-only `<span class="top-bar-field">` blocks at `app.component.html:62-63` with two instances of a new `<vt-context-pulldown>` component (one for dataset, one for detector).
- **New: `frontend/src/app/components/context-pulldown/`** — The pulldown component itself. One file set, two instances (dataset / detector) parameterized by `kind: 'dataset' | 'detector'`. Owns the dropdown overlay, row rendering (glyph + fade + label), "Add New" footer, placeholder text, ellipsis truncation, click-outside-to-close, keyboard navigation.
- **`frontend/src/app/services/datasets-api.service.ts` + `detectors-api.service.ts`** — Already expose `/api/datasets/registry` and `/api/detectors/registry`. The pulldown subscribes to these for its row list.
- **`frontend/src/app/components/dashboard/dashboard.component.ts`** — Hosts the importer modal and new-detector modal today. Phase 1 hoists their *invocation* into a small service (`NewThingFlowsService` or equivalent) so the pulldown's "Add New" footer can open them in-place over Train/Find without navigating to the Dashboard. The Dashboard's own buttons keep working — they just route through the same service.
- **`vtsearch/routes/detectors/find.py`** — Backend already supports multi-dataset Find; the switcher does not change this. The `/find` GUI route remains single-pair (`onFind` in `dashboard.component.ts`).
- **`frontend/src/app/services/find-session.service.ts`** — Will likely be deletable once Phase 2 lands (the URL becomes the source of truth). Phase 1 leaves it alone.

## Phase 1 — Switcher core

Ship the pulldown UI, click-to-switch, compatibility handling, "Add New" footers, and the loading / cancellation behaviors. After Phase 1 the URL is unchanged; the active pair still lives in `ActiveContextService` exactly as today.

### Files

- **New**: `frontend/src/app/components/context-pulldown/context-pulldown.component.{ts,html,scss,spec.ts}`
- **New**: `frontend/src/app/services/new-thing-flows.service.ts` — invokable launcher for the importer modal and the new-detector modal, decoupled from the Dashboard component. Holds open/close state, completion callbacks.
- **New**: `frontend/src/app/components/context-pulldown/incompatible-pair-explainer.component.{ts,html,scss}` — the explainer rendered in place of Train/Find content when the active pair is incompatible. Owns the two recovery affordances (focus other pulldown / go to Dashboard).
- **Modify**: `frontend/src/app/app.component.html` — replace `app.component.html:62-63` with two `<vt-context-pulldown>` instances. Wire the incompatible-pair explainer at the page level (sibling of `<router-outlet />`) so it can override view content when the pair is incompatible.
- **Modify**: `frontend/src/app/app.component.ts` — drop `datasetDisplayName` / `modelDisplayName` direct bindings (the pulldown owns its own labels now); subscribe to `ActiveContextService` for the "is active pair compatible?" predicate that gates the explainer overlay.
- **Modify**: `frontend/src/app/app.component.scss` — remove `.top-bar-field` / `.top-bar-value` styling (moves into the pulldown). Keep `.top-bar-spacer` etc.
- **Modify**: `frontend/src/app/services/active-context.service.ts` — add `setActivePair(datasetId, detectorId)` that emits both subjects atomically.
- **Modify**: `frontend/src/app/components/dashboard/dashboard.component.ts` — refactor the importer modal and new-detector modal openers to go through `NewThingFlowsService`. Dashboard buttons keep working; the pulldown's "Add New" footer also calls into the same service.
- **Modify**: `frontend/src/app/components/find-view/find-view.component.ts` — handle the pulldown-driven pair change. On a new pair, treat as a fresh entry to `/find` (re-run the same prep that `onFind` runs today). The view should not assume `FindSessionService` was pre-populated — the switcher does not write `FindSessionService`.
- **Modify**: `frontend/src/app/components/label-view/label-view.component.ts` — same: handle pair changes by re-entering the prep path.

### Pulldown behavior

Each pulldown row shows:

- **Glyph (leftmost)** — three states, single character each:
  - `○` (or `·`) — registered but not loaded into RAM. Click costs disk + embedding time.
  - `●` (or `–`) — loaded into RAM. Click is fast (no load needed; detector may still re-resolve labels if embedder differs).
  - `✓` — loaded **and** currently active. Click is a no-op.
- **Label** — the registry item's display name. Append a small `· image` / `· audio` / `· text` / `· video` / `· document` tag in a muted color showing the item's `media_type`.
- **Row state**:
  - **Compatible with the other half**: full opacity, hover highlight, click switches.
  - **Incompatible with the other half**: ~50% opacity, no hover highlight, click still switches (the user explicitly opted in by clicking). Tooltip on hover: `"Detector ImgerB embeds images; cannot score audio dataset AudiosC."` (or the appropriate inversion for dataset rows).
- **Active row marker**: the active row gets the `✓` glyph (already covered above); also use bold or a subtle background tint so it's identifiable at a glance.

Row order: **mirror the Dashboard grid's order exactly.** Don't re-sort compatibles to the top.

Footer (always present, even with empty registry):

- `+ Add New Dataset` / `+ Add New Detector` — visually distinct from the rows above (separator + lighter background). Clicking opens the Dashboard's importer / new-detector modal **in-place over the current view** via `NewThingFlowsService`. On successful completion the new item becomes the active half of the pair (other half unchanged), and the user stays in their current view. If the resulting pair is incompatible, the explainer takes over normally.

Placeholder text when no item is selected for that half (initial state, or after the active item is unloaded/deleted):

- Dataset pulldown: `Select a dataset`
- Detector pulldown: `Select a detector`

Empty-registry behavior: pulldown contains only the "Add New" footer; the placeholder is unchanged.

Truncation: closed state uses CSS ellipsis on the label. Open state widens the menu (e.g. `max-width: 480px` or wider) so long names render fully. No mobile considerations.

### Compatibility predicate

Single helper, single source of truth (proposed: `frontend/src/app/utils/context-compat.ts`):

```ts
export function isPairCompatible(
  dataset: DatasetRegistryEntry | null,
  detector: DetectorRegistryEntry | null,
): boolean {
  if (!dataset || !detector) return false; // half-set pair is "not yet compatible"
  return dataset.media_type === detector.media_type;
}
```

Datasets and detectors each have a single `media_type` (confirmed in design). Strict equality. Implementing this as a predicate function (rather than inlining `===`) means future multi-media-type detectors — if they ever exist — only require updating one place.

### Add-new footer flow (decided)

- The Dashboard's existing importer modal and new-detector modal must be invokable from outside the Dashboard component. Refactor: extract the open/close state and the post-completion callback wiring into `NewThingFlowsService`. The Dashboard's own toolbar buttons re-route through it. The pulldown's "Add New" footer calls the same service methods.
- On successful import/create:
  - Service emits a `created` event with the new ID.
  - The pulldown listens, calls `activeContext.setActivePair(...)` with the new ID in the appropriate half and the other half unchanged.
  - The user stays in their current view (Train/Find). The standard prep fires for the new pair, including the loading state.
  - If the new pair is incompatible, the explainer takes over and the user fixes the other half (which can be another "Add New" round if needed).

### Incompatible-pair explainer

Rendered as a sibling of `<router-outlet />` in `app.component.html`, conditionally shown when `isPairCompatible(activeDataset, activeDetector)` returns false **and** the current route is one that consumes the active pair (`/label`, `/find`).

Text template (copy lifted from design discussion, parameterize names):

> **AudiosC** and **ImgerB** are incompatible because they are different media types (audio vs. image).
>
> Please choose a compatible pair, or go to the dashboard.
>
> [ Pick a compatible detector ]   [ Go to Dashboard ]

- `Pick a compatible detector` (label inverts when the *dataset* half is the one to swap): opens the *other* pulldown with the dropdown menu open and scrolled to the first compatible row.
- `Go to Dashboard`: `router.navigate(['/dashboard'])`.

The standard Train/Find prep **does not run** while the explainer is showing. The view's data state remains whatever it was before the user clicked an incompatible row; on returning to a compatible pair, normal prep fires for the new pair (not the old one).

### Loading state on switch

The switch triggers exactly the prep path that the Dashboard's "Train" / "Find" buttons trigger today. Reuse the existing loading affordance (`datasetState.setLoading(true)` + progress polling in `dashboard.component.ts:1278`'s `loadRegistered` flow). Two cosmetic refinements:

- When the new dataset's embedder differs from the previously-active dataset's embedder, the progress message shows `Re-resolving labels for <embedder name>…` during the labelset re-materialization step (replaces the generic spinner text). This helps the user understand why a same-MediaType, same-detector switch can still take 30s.
- Background load: the progress polling mechanism already exists (`startProgressPolling`). The switcher does not invent a new polling loop.

### Cancel-and-replace on rapid re-click

User clicks DataB, prep starts (load + re-resolve, 30s budget), user clicks DataC 5s in. Behavior:

1. Cancel any cancellable in-flight prep step for DataB. `cancel_dataset_progress()` already exists for the dataset load (`vtsearch/concurrency/progress.py`); use it.
2. For prep steps that aren't gracefully cancellable (most embedding work isn't preemptible), tag each switcher invocation with a monotonic request id. On completion, if the in-flight request id doesn't match `activeContext.currentRequestId`, discard the result (don't write it to the context, don't dismiss the loader).
3. Start the new prep for DataC.

This matches browser back/forward semantics: the user expects "latest click wins."

If cancellation turns out to be infeasible for some intermediate step, that's acceptable — the request-id check is the actual correctness guarantee. Cancellation is a CPU/bandwidth optimization, not a correctness mechanism.

### View-local state on switch

Reset everything view-local:

- Scroll position in the media list.
- Partial Find query text (not yet submitted).
- Any in-progress region-vote draft (rectangle being drawn but not committed).
- Sort mode selection — actually, this is a *user* setting (per-user-tier), so it persists across switches. Don't reset.
- Inclusion toggles — also per-user-tier. Persist.

Rule of thumb: per-user settings (in `vtsearch/settings.py` "Per-user tier") survive a switch by design. Per-view ephemeral UI state resets. Per-`DetectorContext` state (votes, last_learned_scores, etc.) is naturally per-pair; switching shows the new pair's state.

### Switcher placement across views

The top bar is global (rendered in `app.component.html`'s `<header>`), so the pulldowns appear on every page. Behavior per route:

- `/label` (Train), `/find` — full behavior as described.
- `/dashboard` — pulldowns visible and interactive. Picking an item there is silly (the grid is right there) but not disallowed; the active pair updates and the Dashboard's own selection state is independent of the active pair so nothing breaks.
- Anywhere else (e.g. mid-modal flow) — pulldowns visible. Switching doesn't dismiss the modal; if the modal depends on the active pair (rare), it's expected to either re-fetch or warn. Treat this as a v2 polish question; v1 doesn't add explicit guards.

### Edge cases (Phase 1)

1. **Active pair is half-set** (e.g. dataset selected, no detector): The half that's set shows its label; the empty half shows the placeholder. The explainer takes over the view since `isPairCompatible` returns false for a null half.
2. **Both halves unset (first run)**: Both pulldowns show placeholders. Explainer shows: "Pick a dataset and detector to begin." Same two buttons (focus dataset pulldown / go to Dashboard).
3. **Active pair references a deleted/unregistered item**: Detect via subscription to the registry list. Clear that half via `activeContext.setActivePair(...)` with the survivor. Show a toast: `"Dataset 'Foo' was removed. Pick another."`
4. **User opens pulldown while a switch is mid-prep**: Pulldown opens normally; clicking another row cancel-and-replaces as above. The in-flight loader keeps animating.
5. **User clicks the active row**: No-op. Don't dismiss; don't re-fire prep.
6. **Registry endpoint returns an error**: Show pulldown with only the "Add New" footer + a small inline error: `"Couldn't load datasets — retry?"`
7. **`FindSessionService` mismatch**: Today `find-view` reads from `FindSessionService` to know which detector to query against. With switcher-initiated entries, `FindSessionService` may be stale or empty. Phase 1 fix: read primarily from `ActiveContextService`; treat `FindSessionService` as a soft hint only. (Phase 2 may delete `FindSessionService` entirely.)

## Phase 2 — URL-driven active context

Move the active pair into the route so reload, share-link, and browser back/forward all carry the pair correctly. After Phase 2 the switcher just calls `router.navigate(...)`; `ActiveContextService` is driven by the route, not the other way around.

### Files

- **Modify**: `frontend/src/app/app.routes.ts` — Routes become:
  ```ts
  { path: 'label/:datasetId/:detectorId', loadComponent: ... },
  { path: 'find/:datasetId/:detectorId', loadComponent: ... },
  { path: 'label', redirectTo: 'dashboard' }, // legacy bare path → dashboard
  { path: 'find', redirectTo: 'dashboard' },
  ```
- **New**: `frontend/src/app/guards/active-context.guard.ts` — Route guard that:
  1. Validates the URL `datasetId` / `detectorId` against the registry (404 to Dashboard if either is unknown).
  2. Calls `activeContext.setActivePair(datasetId, detectorId)`.
  3. Triggers the standard load if either half isn't already loaded.
  4. Lets the route activate once loading completes (or immediately if both are loaded).
- **Modify**: `frontend/src/app/services/active-context.service.ts` — The setters become **private** (or at least flagged "called only by the guard / router"); the switcher pulldown calls `router.navigate(['/label', dsId, detId])` instead of `activeContext.setActivePair(...)`. Existing code that calls `setDatasetId` / `setModelId` directly (`dashboard.component.ts:1252-1253`, etc.) is migrated to navigation.
- **Modify**: `frontend/src/app/components/dashboard/dashboard.component.ts` — `onFind` and `onTrain` change from `setDatasetId + setModelId + router.navigate(['/find'])` to `router.navigate(['/find', dsId, detId])`. The guard handles the rest.
- **Delete (likely)**: `frontend/src/app/services/find-session.service.ts` — `modelId` / `datasetId` move to route params; `modelName` is derivable from the registry.

### Backwards compatibility

- Bare `/label` and `/find` redirect to Dashboard (no active pair to encode).
- Bookmarks to old `/label` and `/find` URLs land on Dashboard, not a broken view.
- localStorage rehydration of the active pair is removed (the URL is authoritative). If we want "remember last pair across sessions," persist it as a setting and have the Dashboard's Train/Find buttons consult it as a default — but that's separate from Phase 2's correctness goal.

### Edge cases (Phase 2)

1. **Race: guard starts loading DataB, user clicks back to a `/label/DataA/DetA` URL**: The router cancels the in-flight navigation and starts the new one. The Phase 1 cancel-and-replace logic (request-id check on completion) already handles the race correctly — no Phase 2-specific work needed.
2. **Shared URL points to a dataset the recipient doesn't have access to**: Guard returns false, redirects to Dashboard, shows a toast: `"Dataset 'foo' is not available."`
3. **Pair encoded in URL is incompatible**: Guard allows the navigation (incompatibility is a valid UI state); the explainer renders.

## Phase 3 — In-flight job affordances

Make running async jobs visible in the pulldown, and verify each job-producing view actually re-reads from `JobManager`'s signature cache on re-entry.

### Files

- **Modify**: `frontend/src/app/components/context-pulldown/context-pulldown.component.{ts,html,scss}` — Add a spinner glyph to rows whose pair has a job in flight. Subscribe to a new (or existing) endpoint that exposes the set of pairs with running jobs.
- **Modify (likely)**: `frontend/src/app/services/active-context.service.ts` or a new `running-jobs.service.ts` — Holds a `Set<string>` of `${datasetId}::${detectorId}` keys with active jobs. Polls or subscribes via SSE.
- **Verification sweep + wire-up (the harder half of this phase)**: For each job-producing view, confirm that on `ngOnInit` (or context-change subscription) the view queries `JobManager` for the cached result against the current signature, and rehydrates the UI from it if found:
  - **Eval view** — check that re-entering after switching away mid-eval restores the in-progress UI (or finished results) from cache.
  - **Labeling-progress training** (`vtsearch/detectors/labeling_progress.py` — the per-step MLP cache + stability analysis) — check the same.
  - **Learned-sort jobs** — check that scores from a completed background sort are visible on re-entry.
- For any view that doesn't currently rehydrate, wire it. The cache exists (`JobManager` keeps results by signature); the work is making each view subscribe + render.

### Spinner glyph

In addition to the three-state load glyph (`○ / ● / ✓`), append a small animated spinner (`⟳` or similar) at the right edge of the row if any job is running on that pair. Tooltip: `"Eval running on this pair"` or similar.

The spinner is **per-pair**, not per-half. If DataA/DetX has a job and the user is on DataA/DetY, the spinner shows on the DetX row in the detector pulldown (the row that completes the busy pair), not on the DataA row.

### Verification protocol

Implementer should write a short Markdown table in this plan's "What shipped" section listing each job type, which view consumes it, and whether the view rehydrated already vs. needed wiring. This becomes the durable record of "we checked these."

### Edge cases (Phase 3)

1. **Job completes while the user is on a different pair**: The spinner glyph disappears from the row; no toast or other notification (out of scope — the existing app doesn't have a notification system).
2. **Multiple jobs on the same pair**: Single spinner; tooltip lists all running job types.
3. **Job is cancelled by the user from elsewhere**: Spinner disappears.
4. **JobManager has a pending slot AND a running slot for the same manager** (the "latest wins" pre-emption pattern): Show the spinner for the *intended* pair, which is the pending one — the running one is about to be cancelled. (Or both, with a tooltip explaining. Implementer's call; document choice in "What shipped.")

## Out of scope

These came up in design and were deliberately deferred or rejected.

- **Typeahead search inside the dropdown** — Deferred until the registry routinely exceeds ~10 items per side. Adding it later is a self-contained polish item.
- **Mobile / narrow viewports** — Never. See `CLAUDE.md` § "Frontend Scope: Desktop Only."
- **Inline editing of the multi-dataset Find set from the top bar** — Multi-N Find is a Dashboard-launched fire-and-forget flow (`onOldFind`) that renders into a results modal without entering the Find view. Cramming a multi-select into the top-bar pulldown would overload the control (same widget meaning two different things depending on launch path). The Dashboard remains the surface for "Find across N datasets."
- **Compatibility predicate beyond media-type equality** — Datasets and detectors each have a single `media_type`; multi-MediaType detectors don't exist today and adding the abstraction prematurely would be speculative.
- **Cross-session "remember my last pair" persistence** — Not part of the switcher itself. If wanted, layer on as a separate per-user setting that the Dashboard's Train/Find buttons consult for default selection. Out of scope here so Phase 2 stays focused on URL-as-identity.
- **Notification when a backgrounded job completes** — The app has no general notification system; adding one is much bigger than this plan. Phase 3's spinner-disappears-when-done is the affordance for v1.

## What shipped (Phase 1)

- **`utils/context-compat.ts`** with `isPairCompatible()` — single source of truth for the media-type-equality predicate.
- **`services/active-context.service.ts`** gained `setActivePair(datasetId, modelId)`, `pair$` (atomic-pair observable), `pairKey$` (joined-key for switchMap triggers), `nextRequestId()` / `currentRequestId` for cancel-and-replace tracking.
- **`services/new-thing-flows.service.ts`** — singleton openers for the dataset importer + new-detector modals plus emit subjects (`importStarted$`, `demoSelected$`, `created$`) so the modals can be invoked from outside the Dashboard.
- **`services/context-switch.service.ts`** — drives a pulldown-initiated pair change end-to-end (atomic flip, parallel dataset/detector load, request-id guard, best-effort cancellation of stale loads).
- **`components/context-pulldown/context-pulldown.component.*`** — top-bar pulldown with closed-state button + dropdown listbox (one row per registry entry, loaded/active glyph, dimmed-when-incompatible row, ellipsis truncation, "+ Add New" footer). Keyboard: ArrowUp / ArrowDown / Home / End / Enter / Escape.
- **`components/context-pulldown/incompatible-pair-explainer.component.*`** — sibling of `<router-outlet />` that takes over `/label` and `/find` when the active pair is incompatible (mismatch, half-set, or fully empty).
- **`app.component.*`** — replaced the read-only `Data: foo` / `Detector: cats` spans with `<vt-context-pulldown>` instances. Hosts the hoisted importer + new-detector modals via `@defer` blocks so they lazy-load (kept initial bundle at 476 kB, well under the 525 kB budget) and the explainer overlay.
- **`components/dashboard/dashboard.component.*`** — opens its two `+` modals through `NewThingFlowsService` (instead of owning the open-state), and subscribes to the service's `importStarted$` / `demoSelected$` / `created$` so the existing post-action flows (progress polling, train-after-create) still work. Combine modals remain Dashboard-local.
- **`components/find-view`, `components/label-view`** — subscribe to `activeContext.pair$` and re-run their loads when the pair changes mid-route, so a pulldown click on `/label` or `/find` re-prepares against the new pair.
- **Tests**: `./run-tests.sh` green (3615 passed, 1 skipped, 2 xpassed). Dashboard specs updated to drive `NewThingFlowsService` directly for the two affected cases.

## Phase 1 follow-ups shipped

A second pass picked up four of the seven open items from the initial follow-up list:

- **Focus-other-pulldown affordance** — `vt-incompatible-pair-explainer` now renders a primary "Pick a compatible detector" / "Pick a dataset" button alongside `Go to Dashboard`. The button calls a new `PulldownControlService.requestOpen(kind)`, which `ContextPulldownComponent` subscribes to and uses to call `openMenu()` — auto-focusing the first compatible row and scrolling it into view. Default direction: for media-type mismatches, swap the detector (matches the design's "Pick a compatible detector" copy); for missing-half cases, focus the missing half.
- **"Add New" footer auto-selects the new item** — the pulldown tracks an `awaitingNew` flag set when its own `addNew()` runs. Detector flow: subscribes to `NewThingFlowsService.created$` and switches on success. Dataset flow: subscribes to `importStarted$`, snapshots the current registry-id set, and switches when a new id appears via `datasets$`. Dashboard-initiated adds don't touch the pulldown's flag, so they remain a no-op for the pulldown. Modal-dismissal-without-success clears the flag so an unrelated later registry change can't false-trigger.
- **Toast on active-item deletion** — new `ActiveContextWatcherService` (started by `AppComponent`) watches `combineLatest([pair$, datasets$, detectors$])`, remembers the last-seen name per active half, and when an active id disappears from a non-empty registry, clears that half via `setActivePair(...)` and emits a toast (`"Dataset 'Foo' was removed. Pick another from the top-bar pulldown."`). Idempotent `start()` and "only fires after the item has been seen at least once" guarantees protect against initial-load false positives.
- **Empty-registry error retry** — `DatasetStateService` gained `error$` (BehaviorSubject<string | null>) and a `catchError` in the registry-fetch pipeline so a failed forkJoin no longer tears down the outer subscription. The pulldown renders an inline `"Couldn't load … Retry"` banner inside the dropdown when `error$` is set; the Retry button calls `refresh()`. Successful fetches clear the error.
- **Pulldown row order mirrors Dashboard grid sort** — `ManagedColumns` gained a `sortState$` BehaviorSubject-backed observable. New `DashboardColumnsService` owns the two `ManagedColumns` instances (datasets, detectors); `DashboardComponent` now pulls them from the service instead of constructing locally. `ContextPulldownComponent` subscribes to the matching `sortState$` and applies the same comparator the Dashboard's `sortedDatasets` / `sortedDetectors` getters use, so a column-header click in the Dashboard re-orders the pulldown's dropdown the same way.

Drive-by: `settings-state.service.spec.ts` mockSettings.theme widened to `string` rather than the typed union, which broke the `Expected<AppSettings>` typecheck. Fixed with `as const`.

## What shipped (Phase 2)

- **Routes** (`frontend/src/app/app.routes.ts`) — Canonical view paths are now `/label/:datasetId/:detectorId` and `/find/:datasetId/:detectorId`, both gated by `activeContextGuard`. Bare `/label`, `/find`, and any unmatched path redirect to `/dashboard`.
- **`frontend/src/app/guards/active-context.guard.ts`** — Reads `datasetId` / `detectorId` from the URL, waits for the registry to be loaded (`datasetState.loaded$`), validates both ids exist (toast + redirect on miss), then calls `ContextSwitchService.applyActivePair(...)` and holds the route until the returned Observable completes. Incompatible pairs are allowed through (the explainer handles them).
- **`services/context-switch.service.ts`** — Split into two entry points: `applyActivePair(ds, det)` is called only by the guard (returns Observable that completes when prep finishes; uses a `ReplaySubject(1)` so late subscribers still see synchronous fast-path completion); `switchTo(ds, det)` is called by the top-bar pulldowns and now navigates to the new URL when on `/label` or `/find`, otherwise flips imperatively (so `/dashboard` pulldown clicks still work).
- **`services/dataset-state.service.ts`** — Gained `loaded$` / `loaded`. Flips to `true` the first time the registry fetch settles (success or error), so the guard never hangs on a deep-link cold start.
- **`services/active-context.service.ts`** — `setDatasetId` and `setModelId` removed (the guard owns mutation now via `setActivePair`).
- **`services/active-context-watcher.service.ts`** — On detecting an active id was deleted from the registry, it now navigates to `/dashboard` after clearing the half (a half-pair URL is not representable).
- **`components/dashboard/dashboard.component.ts`** — `onLabel` / `onFind` reduced to `router.navigate(['/label', ds, det])` / `['/find', ds, det]`. Per-button `trainLoading` / `findLoading` flags flip on click and reset on the next `NavigationEnd` / `NavigationCancel` / `NavigationError` so the icon waggle still works while the guard waits.
- **`components/find-view/find-view.component.ts`** — Reads `datasetId` / `modelId` from `ActiveContextService` and the detector name from the registry; no longer depends on `FindSessionService`.
- **`components/modals/export-modal/export-modal.component.ts`** — Falls back to a registry lookup against `ActiveContextService.modelId` when both upstream name sources are empty.
- **Deleted**: `frontend/src/app/services/find-session.service.ts`. The active (dataset, detector) pair is the URL; the modelName is derivable from the registry.
- **Tests**: `./run-tests.sh` green (3615 passed, 1 skipped, 2 xpassed). New specs for the guard and the new `loaded$` flag; Dashboard's `onLabel` spec rewritten around the navigation contract (the guard owns the load it used to test inline).

## Open follow-ups

Carry-overs from earlier phases. Phase 2 closed the `FindSessionService` deletion item.

- **"Re-resolving labels for X's embedder…" progress message** (Phase 1 follow-up) — the design called for a custom progress message when the new dataset's embedder differs from the previous one. Phase 1 reuses the generic loading affordance. Implementing this is bigger than a copy change because: (a) `ContextSwitchService.flipAndLoad` only triggers a detector re-load when `!detector_loaded`, so an embedder change on a detector that's already "loaded" against the previous dataset would leave its label embeddings in the old embedder's space — there's no current re-embed path to hang a custom progress message off; (b) the labelset re-resolve in `before_request` (`ensure_votes_match_active_dataset`) only re-keys cids by origin, it doesn't re-embed. To land this properly you need to (1) detect the embedder change on switch, (2) force a detector re-load (or add a new "re-embed labels only" path), (3) thread the new embedder's display name through to the progress tracker in `vtsearch/routes/detectors/registry.py`'s "Embedding labels…" step. Worth a small design pass before implementation.
- **Phase 3 — in-flight job affordances** — Pulldown spinner glyph for rows whose pair has a running async job, plus the rehydration verification sweep across eval / labeling-progress / learned-sort views. See "Phase 3 — In-flight job affordances" above for the design.

## Open questions

(None at draft time — the design was fully resolved in conversation before this plan was written. Implementers should add new questions here as they encounter ambiguity, and ping the user before guessing.)
