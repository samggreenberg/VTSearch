# Find Verification Workflow

**Status:** Design — not yet implemented. Requirements locked with the user; awaiting go-ahead to build.

## The reframe (read this first)

Verification is **not a new mode**. It is a richer label vocabulary layered onto the *existing* Find interface. Today Find is binary: the detector scores every item and the threshold splits them into **good** (≥ threshold) and **bad** (< threshold). We keep that exact machinery and add one distinction on top — **has a human touched this item or not** — yielding four states:

| State | Meaning | How it arises |
|-------|---------|---------------|
| **Unverified Good** | Over threshold, detector's call, human hasn't looked | Default for every above-threshold item after a find run |
| **Unverified Bad** | Under threshold, detector's call, human hasn't looked | Default for every below-threshold item after a find run |
| **Verified Good** | Human confirmed (or rescued) it as good | Human acted on the item with the **Good** button |
| **Verified Bad** | Human confirmed (or culled) it as bad | Human acted on the item with the **Bad** button |

A user who just wants "run the detector and export the hits" never touches a verify button: they export the Unverified sets exactly like today. A user who wants to verify walks the work queue and confirms items, promoting them from Unverified to Verified. **Same screen, no toggle.**

### Why this is mostly a *surfacing* job, not new plumbing

The current find-label flow (`vtsearch/routes/detectors/scoring.py`) already:

- scores every item and applies a `good`/`bad` label by threshold (`scoring.py:344-351`),
- records the detector's original assignment in `find_initial_labels` (`scoring.py:353`) so corrections can be detected,
- sets `find_mode = True` so these ephemeral labels are never persisted to disk.

And the frontend already treats `goodVotes` = (above-threshold items) ∪ (manual good corrections); `onBrowse`/`onToDataset` in `find-view.component.ts:452,475` already scope over the **full** `goodVotes` set. So the "flood-fill" (unverified-good folds into the exported good set) is *already how it works* — we are exposing it, not building it.

What's genuinely missing: a way to tell **Verified** from **Unverified** (a confirm that doesn't change the vote leaves no trace today), the panel UI that shows the four buckets with count-folding headers, the **auto-advance** after a verify, and the **Unverified Export**.

## Locked requirements

1. **Persistence: ephemeral.** Nothing about Find is saved between sessions — not before, not now. The verified set lives in memory on `DetectorContext` for the process lifetime, like every other Find artifact. The **Unverified Export** is the escape hatch: dump the work queue to a file and re-import it as its own dataset before it evaporates.
2. **Inclusion: unchanged for now.** The Inclusion slider keeps today's behavior — each change retrains the MLP (it's a training class-bias knob, `-10..10`) and re-scores. We revisit a pure-threshold-over-frozen-scores version **only if it proves slow** in practice. (Tracked in Open follow-ups.)
3. **Verification is positives-only.** The work queue the user walks is the over-threshold pile (find false positives → flip to Verified Bad; confirm true positives → Verified Good). Under-threshold items stay auto-Bad and are not a primary review surface. **But** the user can always click any item from the left panel into the center and use the big **Good**/**Bad** buttons to label it ActuallyGood / ActuallyBad — those buttons must work, and **after a mark the next item auto-loads into the center.**
4. **One selection-pile, two resolution buttons.** Marking is "select item → resolve as Good or Bad," not two independent toggle types.

## Data model

Everything derives from two pieces of in-memory state on the active `DetectorContext`:

- the **current votes** (`good_votes`, `bad_votes` — already exist, mutated by the Good/Bad buttons), and
- a new **`verified_ids: set[int]`** — the items the human has acted on this session.

The four buckets are *derived*, never stored separately:

```
Verified Good   = good_votes ∩ verified_ids
Verified Bad    = bad_votes  ∩ verified_ids
Unverified Good = good_votes − verified_ids
Unverified Bad  = bad_votes  − verified_ids
```

`verified_ids` is required because confirming an item *without changing its vote* (the common case — detector said good, human agrees) produces no observable change in `good_votes`; only an explicit "I looked at this" set can capture it. It is process-scoped state on `DetectorContext`, which the "No Persisted Vectors or MLPs" rule explicitly blesses (in-memory caches on the context are fine; never written to disk/settings/JSON).

**Re-score semantics (preserve verified work):** the right panel is a *stable accumulation* — the items the human has resolved. The left panel is the churning work queue. Therefore a re-score (initial load, or after an Inclusion change) re-labels **only the unverified items**; it never touches a verified item's vote, and `verified_ids` survives. Concretely:

- Re-scoring computes fresh scores for every item (the model retrained at the new Inclusion), but only **applies** the new threshold labels to items *not* in `verified_ids`.
- A previously Unverified Good item can flip to Unverified Bad (and vice-versa) as the threshold moves — that's expected; it's still unverified.
- A Verified Good / Verified Bad item keeps its human vote no matter where its new score lands.

`verified_ids` is only cleared when the dataset/detector **pair changes** (a genuinely fresh context — `reloadForNewPair` in `find-view.component.ts:146`). Nothing the user does in the left panel (re-score, Inclusion, browse) clears it.

## Backend changes

1. **`DetectorContext`** (`vtscore/state/core.py:322`): add `verified_ids: set[int] = field(default_factory=set)`. **Do not** clear it on re-score or model-cache invalidation — only on a dataset/detector pair change.
1b. **Re-score preserves verified votes.** Change the find-label run (`scoring.py:341-353`) so it stops doing `apply_labels_bulk_with_click_time(..., replace_all=True)` over *all* items. Instead it applies the fresh threshold labels **only to items not in `verified_ids`**, leaving verified items' votes and their `find_initial_labels` entries intact. Scores are still computed for every item (for ordering), but verified items are excluded from re-labeling.
2. **Mark-verified on manual votes in find mode.** When `find_mode` is True and a vote arrives through the manual path (the big buttons, hover-vote, right-panel vote — *not* the bulk `apply_labels_bulk_with_click_time` that find-label uses), add the id to `verified_ids`. Cleanest hook: in `vtscore/state/votes.py` `set_vote`/`toggle_vote`/`apply_label_with_click_time` (the click-time-bearing, single-item paths), guarded by `is_find_mode()`. The bulk find-label path stays unverified by construction.
3. **Expose `verified_ids`** in the `GET /api/votes` payload (a `verified` id array) so the frontend can derive the four buckets without a second request. (`vtsearch/routes/labels/vote.py`.)
4. **Unverified Export.** Extend the label export `label_filter` (`/api/labels/export`, `vote.py:147`) with two new values: `unverified` (emit `good_votes ∪ bad_votes − verified_ids`) and, for symmetry, `verified`. The work-queue dump the user re-imports as a dataset uses `unverified`. Existing `good`/`bad`/`corrections`/`all` are unchanged.

## Frontend changes

1. **`VoteStateService`**: track a `verifiedIds` set sourced from the new `/api/votes` `verified` field; expose `verifiedIds$`. On a manual vote in find mode, optimistically add the id (backend reconciles).
2. **Right panel** (`right-panel.component.ts`): in find mode, replace the single good/bad lists with the four-bucket view:
   - A **Verified Good** list, with a header **"[N] Unverified Good"** where N = `unverifiedGood.length`. Its **Browse / Export / To Dataset** buttons act on *everything below the button* = Verified Good ∪ Unverified Good = the full `goodVotes` set. (This is already what `onBrowse`/`onToDataset` do — they read `goodVotes` — so only the header/labeling is new.)
   - A symmetric **Verified Bad** list with a **"[M] Unverified Bad"** header.
   - Buckets derived from `goodVotes$`, `badVotes$`, `verifiedIds$`.
3. **Left panel** = the work queue: the Unverified Good items (above threshold, untouched), in score order — what the user walks. Add an **Unverified Export** button that calls the export with `label_filter=unverified`.
4. **Center auto-advance** (the explicit ask): today `find-view.component.ts:439 onMediaVoted` only reloads votes — no advance. Change it so, in find mode, after a vote on the centered item it selects the **next item in the work queue** (next Unverified Good in score order) via `mediaState.selectMedia(nextId)`. The center-panel swipe animation already fires `mediaVoted` after ~180ms (`center-panel.component.ts:260`), so the advance rides on that callback. When the queue empties, leave the center on the last item (or show a "queue clear" state).
5. **Big Good/Bad buttons must work in find mode.** Verify the `voting-overlay` is rendered and not `disabled` while in find mode (it's gated by the Find scoring-busy `disabled` input today). Wire its `voted` emit through `castVote` → `submitToggleVoteAndRecord` (already the path) and ensure the find-mode vote marks the item verified (item 1/2 above).

## State-transition summary

- **Find run** → every item Unverified Good or Unverified Bad; `verified_ids` empty; center seeded on the item just above threshold (existing behavior, `find-view.component.ts:237`).
- **User clicks Good on centered item** → vote good + add to `verified_ids` → item becomes Verified Good → auto-advance to next Unverified Good.
- **User clicks Bad on centered item** (a false positive) → vote bad + verified → Verified Bad → auto-advance.
- **Export / Browse / To Dataset (good panel)** → operates on Verified Good ∪ Unverified Good (flood-fill); unreviewed positives ride along on the detector's call.
- **Unverified Export** → dumps only the untouched work queue for re-import as a dataset.
- **Inclusion change** → retrain + re-score → only the **unverified** items get re-labeled at the new threshold; every Verified Good / Verified Bad item (the right-panel accumulation) keeps its human vote. `verified_ids` is preserved.

## Decisions captured

- **Inclusion retrains (not threshold-only) for v1.** Matches current wiring; revisit only if slow.
- **Positives-only review**, with the big buttons available on any centered item for one-off ActuallyGood/ActuallyBad marks + auto-advance.
- **Ephemeral throughout**; Unverified Export is the persistence escape hatch.
- **Re-score preserves verified work (CRITICAL).** Verified votes are a stable right-panel accumulation; re-scoring (incl. every Inclusion change) only re-labels unverified items and never disturbs them. `verified_ids` is cleared only on a dataset/detector pair change. Anything the user does in the left panel leaves the right panel untouched.

## Open follow-ups

- **Pure-threshold Inclusion.** If retrain-per-change is too slow, repurpose the Inclusion knob into a score-threshold slider over frozen scores (run once, slide the green/red line with no retrain/re-embed). Forks the backend (Inclusion stops invalidating the MLP) and the UX ("frozen detector"). Deferred pending real-world latency.
- **Under-threshold (false-negative) review surface.** Currently only reachable item-by-item via the left panel + big buttons. If demand appears, add a dedicated under-threshold work queue (the symmetric "find what the detector missed" flow).
