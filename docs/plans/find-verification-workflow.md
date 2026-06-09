# Find Verification Workflow

**Status:** Design — not yet implemented. Requirements locked with the user; awaiting go-ahead to build.

## The reframe (read this first)

Verification is **not a new mode**. It is a richer label vocabulary layered onto the *existing* Find interface. Find scores every item with the detector; a cutoff line splits them into **good** (≥ cutoff) and **bad** (< cutoff). We keep that split and add one distinction on top — **has a human touched this item or not** — yielding four states:

| State | Meaning | How it arises |
|-------|---------|---------------|
| **Unverified Good** | Above the cutoff, detector's call, human hasn't looked | Default for every above-cutoff item |
| **Unverified Bad** | Below the cutoff, detector's call, human hasn't looked | Default for every below-cutoff item |
| **Verified Good** | Human confirmed (or rescued) it as good | Human acted on the item with the **Good** button |
| **Verified Bad** | Human confirmed (or culled) it as bad | Human acted on the item with the **Bad** button |

A user who just wants "run the detector and export the hits" never touches a verify button: they export the Unverified sets exactly like today. A user who wants to verify walks the work queue and confirms items, promoting them from Unverified to Verified. **Same screen, no toggle.**

## Core mechanic: score once, then slide a cutoff over frozen scores

This is the spine of the whole feature, so it comes first.

**The detector scores every item exactly once**, at its trained configuration, producing a fixed score per item. Those scores are **frozen** for the life of the Find session. The left-panel slider is a **cutoff** over those frozen scores: sliding it moves the green/red line up or down the fixed ranking. It does **not** retrain the MLP, does **not** re-embed, and does **not** reorder items — it only changes *how deep into the ranking the positive cut goes*. (Higher = more inclusive = more greens; this is why the user calls it "Inclusion" — it controls how inclusive the positive set is. Whether the UI keeps that label or renames it "Cutoff" is a minor call noted in follow-ups.)

This is a deliberate change from today, where the Find slider *is* the training-Inclusion knob and every change retrains the MLP (`-10..10`, feeding hidden-layer width + regularization + calibration). **Train mode keeps that training-Inclusion knob unchanged** — it legitimately wants retrain-on-change while you teach a detector. Only Find's slider is decoupled into a pure cutoff. The two are different controls that happen to have shared a name.

Freezing the scores is what makes the rest cheap:

- **Instant slide.** No GPU work on a cutoff change — just re-thresholding fixed numbers.
- **Preserve-verified is automatic.** Verified items carry an explicit human vote that the cutoff never overrides; sliding only reclassifies *unverified* items. Nothing the user does in the left panel can disturb the right panel.
- **"At what cutoff would this item return?" is a trivial inversion** — every item has one fixed score, so the answer is just where the line crosses it (see Stats, below).

## Data model

State on the active `DetectorContext` (all in-memory, ephemeral — blessed by the "No Persisted Vectors or MLPs" rule, which explicitly allows process-scoped caches on the context):

- **`find_scores: dict[int, float]`** — the frozen per-item detector scores from the single scoring pass.
- **`find_cutoff: float`** — the current cutoff line. Defaults to the calibrated threshold the detector produces (`train_and_threshold`); the slider moves it.
- **`good_votes` / `bad_votes`** — already exist. In find mode these hold the items' current good/bad assignment.
- **`verified_ids: set[int]`** (new) — the items the human has explicitly acted on this session.

**Resolved label of any item** = `good` if it's a verified-good vote; `bad` if verified-bad; otherwise `score ≥ find_cutoff`. The four buckets fall out:

```
Verified Good   = verified_ids ∩ good_votes
Verified Bad    = verified_ids ∩ bad_votes
Unverified Good = { i ∉ verified_ids : find_scores[i] ≥ find_cutoff }
Unverified Bad  = { i ∉ verified_ids : find_scores[i] <  find_cutoff }
```

`verified_ids` is required because confirming an item *without changing its side of the line* (the common case — detector says good, human agrees) leaves no trace in the scores; only an explicit "I looked at this" set captures it.

**Cutoff-slide semantics (no re-score, preserve verified):** moving the slider sets `find_cutoff` and re-thresholds **only unverified items** against the frozen `find_scores`. Verified items keep their human vote wherever their score lands. An unverified item can flip Unverified Good ↔ Unverified Bad as the line moves; that's expected. No retrain, no re-embed.

`verified_ids`, `find_scores`, and `find_cutoff` are cleared/recomputed only when the dataset/detector **pair changes** (a genuinely fresh context — `reloadForNewPair` in `find-view.component.ts:146`), which is also the only time the detector re-scores.

## Backend changes

1. **`DetectorContext`** (`vtscore/state/core.py:322`): add `find_scores: dict[int, float]`, `find_cutoff: float`, and `verified_ids: set[int]` (all `field(default_factory=...)`). Cleared only on a pair change, not on a cutoff slide.
2. **Score once, then stop re-scoring.** The find-label run (`vtsearch/routes/detectors/scoring.py`) trains/loads the model and scores all items as it does today, but now **caches the scores in `find_scores`**, sets `find_cutoff` to the calibrated default, applies labels by that default cutoff (the initial good/bad split), and records `find_initial_labels` (the detector's natural call at the default cutoff — the stable eval baseline; see Stats). It no longer reruns on a slider change.
3. **New cutoff endpoint** `POST /api/find/cutoff` (decoupled from the training-Inclusion `POST /api/inclusion` at `sorting.py:760`, which Find stops calling): sets `find_cutoff`, re-thresholds **unverified** items against `find_scores`, reassigns their `good_votes`/`bad_votes`, and leaves verified items and `find_initial_labels` untouched. Cheap — no model work. The Find slider calls this instead of the retrain path.
4. **Mark-verified on manual votes in find mode.** When `find_mode` is True and a vote arrives through the *single-item* manual path (the big buttons, hover-vote, right-panel vote), add the id to `verified_ids`. Cleanest hook: `vtscore/state/votes.py` `set_vote` / `toggle_vote` / `apply_label_with_click_time`, guarded by `is_find_mode()`. The bulk find-label/cutoff paths stay unverified by construction.
5. **Expose `verified_ids`** in the `GET /api/votes` payload (a `verified` id array) so the frontend can derive the four buckets in one request (`vtsearch/routes/labels/vote.py`).
6. **Unverified Export.** Extend the label-export `label_filter` (`/api/labels/export`, `vote.py:147`) with `unverified` (emit `good_votes ∪ bad_votes − verified_ids`) and, symmetrically, `verified`. The work-queue dump the user re-imports as a dataset uses `unverified`.

## Frontend changes

1. **Find slider becomes a cutoff control.** `find-view.component.ts onInclusionChange` (currently `setInclusion` → `runFindLabel`, which retrains) is replaced by a call to `POST /api/find/cutoff`. Because the frontend already has every item's score (`sortState.setSortResults(sorted, threshold)` holds `[{id, score}]`), the green/red display updates **optimistically client-side** the instant the slider moves; the cutoff POST reconciles server votes for export. No scoring spinner on a slide.
2. **`VoteStateService`**: track a `verifiedIds` set from the new `/api/votes` `verified` field; expose `verifiedIds$`. On a manual vote in find mode, optimistically add the id.
3. **Right panel** (`right-panel.component.ts`), find mode — the four-bucket view:
   - A **Verified Good** list with a header **"[N] Unverified Good"** (N = unverified-good count). Its **Browse / Export / To Dataset** buttons act on *everything below the button* = Verified Good ∪ Unverified Good = the full good set. (`onBrowse`/`onToDataset` already scope over `goodVotes` in `find-view.component.ts:452,475`, so the header/labeling is the new part.)
   - A symmetric **Verified Bad** list with a **"[M] Unverified Bad"** header.
   - Buckets derived from scores + cutoff + `verifiedIds$`.
4. **Left panel** = the work queue: Unverified Good items in score order — what the user walks. Add an **Unverified Export** button (`label_filter=unverified`).
5. **Center auto-advance** (explicit ask): today `find-view.component.ts:439 onMediaVoted` only reloads votes. In find mode, after a vote on the centered item, select the **next Unverified Good in score order** via `mediaState.selectMedia(nextId)`. The center-panel swipe already fires `mediaVoted` ~180ms after the vote (`center-panel.component.ts:260`), so the advance rides that callback. Empty queue → rest on the last item (or a "queue clear" state — minor, flagged below).
6. **Big Good/Bad buttons must work in find mode.** Ensure the `voting-overlay` is rendered and not `disabled` outside the brief initial-scoring window; its `voted` → `castVote` → `submitToggleVoteAndRecord` path marks the item verified (item 4 above).

## Detector evaluation: the Stats button

The real reason to verify a test haystack is **"is this detector worth subscribing to on my data?"** The verified set doubles as an evaluation sample, surfaced via a **Stats** button on the right panel (same affordance as the Datasets Dashboard card → `vt-dataset-stats-modal`).

**No new tracked state.** The detector's natural call per item is `find_initial_labels` (its good/bad at the *default calibrated cutoff*, fixed at score time — so the eval baseline is stable no matter where the user later drags the slider). The human's call is the current vote. Crossing them over `verified_ids` gives a 2×2:

| | Detector said **Good** | Detector said **Bad** |
|---|---|---|
| **Human → Good** (Verified Good) | Confirmed positive *(agree)* | Rescued false negative *(correction)* |
| **Human → Bad** (Verified Bad) | Culled false positive *(correction)* | Confirmed negative *(agree)* |

Reported figures:

- **Verified count** — evaluation sample size.
- **Agreements vs. corrections** and the **agreement rate** — `confirmed_good + confirmed_bad` vs. `culled_fp + rescued_fn`.
- **Precision on reviewed positives** — `confirmed_good / (confirmed_good + culled_fp)`: of the detector's above-cutoff hits the human checked, how many held up. The single most decision-relevant number.
- **Misses surfaced** — `rescued_fn`: items the human pulled up from below the line.
- **"At what cutoff would it have returned?"** — now cheap because scores are frozen. For each rescued false-negative (a Verified Good sitting below the line), report the cutoff/score at which it would cross — i.e. how far the user would have to slide to make the detector surface it unaided. A per-item readout *and* an aggregate ("slide to score X to recover all your rescued positives"). This is the headline "how good is it really" signal and is a real column now, not a deferred proxy.
- **Run context** — the default cutoff and the current cutoff.

### Backend
- **`GET /api/find/stats`**: computes the four counts from `verified_ids` × (`good_votes`/`bad_votes`) × `find_initial_labels`, plus per-item recover-cutoffs from `find_scores`. Pure read; no new state.

### Frontend
- **`vt-find-stats-modal`** (mirrors `vt-dataset-stats-modal`): standalone modal on the shared `ModalComponent`, fetches `GET /api/find/stats` on open, renders the 2×2 + rates + recover-cutoff readout.
- **Stats button** in the right panel (find mode), beside Browse / Export / To Dataset, emitting a `stats` event `find-view` handles — same wiring as the dashboard card's `stats` output.

## State-transition summary

- **Find run (pair load)** → score once → `find_scores` frozen, `find_cutoff` = calibrated default, `find_initial_labels` recorded, `verified_ids` empty; center seeded on the item just above the line (`find-view.component.ts:237`).
- **Slide cutoff** → re-threshold unverified items only (instant, client-optimistic + cheap server cutoff POST); verified items unmoved; no retrain.
- **Good on centered item** → vote good + verify → Verified Good → auto-advance to next Unverified Good.
- **Bad on centered item** (false positive) → vote bad + verify → Verified Bad → auto-advance.
- **Export / Browse / To Dataset (good panel)** → Verified Good ∪ Unverified Good (flood-fill); unreviewed positives ride along on the detector's call.
- **Unverified Export** → dumps only the untouched work queue for re-import as a dataset.

## Decisions captured

- **Find slider = pure cutoff over frozen scores (CRITICAL).** Score once at the detector's trained config; the slider only moves the line — no retrain, no re-embed, no reorder. Train mode keeps its separate training-Inclusion knob.
- **Preserve-verified is automatic.** Verified items carry explicit votes the cutoff never overrides; sliding reclassifies only unverified items. The right panel is undisturbed by anything in the left panel. `verified_ids`/`find_scores`/`find_cutoff` reset only on a pair change.
- **Positives-only review**, with the big buttons available on any centered item for one-off ActuallyGood/ActuallyBad marks + auto-advance.
- **Ephemeral throughout**; Unverified Export is the persistence escape hatch.
- **Stats eval baseline = the default calibrated cutoff**, fixed at score time, so corrections are measured against the detector's natural recommendation regardless of later slider position.

## Open follow-ups

- **Slider parameterization & label.** Decide how cutoff maps to slider units (raw score, percentile/quantile of the score distribution, or keep a `-10..10` feel mapped monotonically to cutoff position) and whether the Find control keeps the word "Inclusion" or becomes "Cutoff." Cosmetic-ish; doesn't change the architecture.
- **Hover-vote = verified?** Confirm a quick hover-vote should count as a deliberate verification the same as a center-panel click. Assumed yes for now.
- **Empty work queue UX.** Rest on the last item vs. an explicit "queue clear" state. Minor.
- **Under-threshold (false-negative) review surface.** Currently reachable only item-by-item via the left panel + big buttons. If demand appears, add a dedicated below-the-line work queue (the symmetric "find what the detector missed" flow).
