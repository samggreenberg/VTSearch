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

**The detector scores every item exactly once**, at its trained configuration, producing a fixed score per item. Those scores are **frozen** for the life of the Find session. The **Inclusion** slider is a **cutoff** over those frozen scores: sliding it moves the green/red line up or down the fixed ranking. It does **not** retrain the MLP, does **not** re-embed, and does **not** reorder items — it only changes *how deep into the ranking the positive cut goes*. Higher Inclusion = more inclusive = more greens, which is exactly what the name has always meant; we keep calling it Inclusion.

This is a deliberate change from today, where the Inclusion slider retrains the MLP on every change (`-10..10`, feeding hidden-layer width + regularization + calibration). **Inclusion is unified across both Train and Find as a no-retrain cutoff.** It leaves the MLP entirely: the architecture (hidden width) and regularization become fixed defaults, independent of Inclusion. Calibration computes a neutral default threshold, and Inclusion offsets the cutoff line from there. Training still happens in Train mode **when you vote** (that's the training loop) — but an Inclusion change never retrains, in either mode; it only moves the line over the current model's scores.

> **Backwards-compat break (call out to user):** existing detectors' stored `inclusion` values change meaning (training-bias → cutoff position), and model capacity no longer varies with inclusion. Per repo policy we make the clean change and surface the break rather than shimming old behavior.

Freezing the scores is what makes the rest cheap:

- **Instant slide.** No GPU work on a cutoff change — just re-thresholding fixed numbers.
- **Preserve-verified is automatic.** Verified items carry an explicit human vote that the cutoff never overrides; sliding only reclassifies *unverified* items. Nothing the user does in the left panel can disturb the right panel.
- **"At what cutoff would this item return?" is a trivial inversion** — every item has one fixed score, so the answer is just where the line crosses it (see Stats, below).

## Data model

State on the active `DetectorContext` (all in-memory, ephemeral — blessed by the "No Persisted Vectors or MLPs" rule, which explicitly allows process-scoped caches on the context):

- **`find_scores: dict[int, float]`** (new) — the frozen per-item detector scores from the single scoring pass.
- **`inclusion`** (already exists) — now the cutoff knob: a no-retrain position that resolves to a score-threshold over `find_scores`. Default = the neutral calibrated threshold (`train_and_threshold`); the slider offsets from there. The resolved cutoff lives in the existing **`threshold`** field. ("Cutoff" below = `threshold` resolved from `inclusion`.)
- **`good_votes` / `bad_votes`** — already exist. In find mode these hold the items' current good/bad assignment.
- **`verified_ids: set[int]`** (new) — the items the human has explicitly acted on this session.

**Resolved label of any item** = `good` if it's a verified-good vote; `bad` if verified-bad; otherwise `score ≥ cutoff`. The four buckets fall out:

```
Verified Good   = verified_ids ∩ good_votes
Verified Bad    = verified_ids ∩ bad_votes
Unverified Good = { i ∉ verified_ids : find_scores[i] ≥ cutoff }
Unverified Bad  = { i ∉ verified_ids : find_scores[i] <  cutoff }
```

`verified_ids` is required because confirming an item *without changing its side of the line* (the common case — detector says good, human agrees) leaves no trace in the scores; only an explicit "I looked at this" set captures it.

**Cutoff-slide semantics (no re-score, preserve verified):** moving the slider sets `cutoff` and re-thresholds **only unverified items** against the frozen `find_scores`. Verified items keep their human vote wherever their score lands. An unverified item can flip Unverified Good ↔ Unverified Bad as the line moves; that's expected. No retrain, no re-embed.

`verified_ids`, `find_scores`, and `cutoff` are cleared/recomputed only when the dataset/detector **pair changes** (a genuinely fresh context — `reloadForNewPair` in `find-view.component.ts:146`), which is also the only time the detector re-scores.

## Backend changes

1. **`DetectorContext`** (`vtscore/state/core.py:322`): add `find_scores: dict[int, float]` and `verified_ids: set[int]` (both `field(default_factory=...)`). The cutoff itself reuses the existing `threshold` field (resolved from `inclusion`); no new cutoff field. Both new fields reset only on a pair change, not on an Inclusion slide.
2. **Inclusion leaves the MLP (training-path refactor).** In `vtscore/detectors/training.py`, `train_model` stops taking `inclusion` for hidden-layer width / regularization — the architecture becomes a fixed default. `POST /api/inclusion` (`sorting.py:760`) stops calling `invalidate_loaded_detector_models()` (`vtscore/state/__init__.py:189-200`): an Inclusion change no longer drops the cached MLP. Calibration (`calculate_cross_calibration_threshold`) computes a **neutral** default threshold; `inclusion` is applied as an **offset** that resolves to the actual cutoff (`threshold`) over the current model's scores. This is the unified Train+Find behavior; **it is the backwards-compat break** flagged above.
3. **Score once, then stop re-scoring (Find).** The find-label run (`vtsearch/routes/detectors/scoring.py`) trains/loads the model and scores all items as today, but now **caches the scores in `find_scores`**, sets the cutoff to the neutral-calibrated default, applies labels by that default (the initial split), and records `find_initial_labels` (the detector's natural call at the default cutoff — the stable eval baseline; see Stats). It no longer reruns on a slider change.
4. **Repurposed Inclusion endpoint (shared by Train and Find).** `POST /api/inclusion` no longer retrains; it sets `inclusion`, resolves the new cutoff over the current scores (the frozen `find_scores` in Find; the live model's scores in Train), and re-thresholds the **unverified** items (in Find, leaving verified items and `find_initial_labels` untouched). Cheap — no model work. The Find slider calls this same endpoint; there is no separate `/api/find/cutoff`.
5. **Mark-verified on manual votes in find mode.** When `find_mode` is True and a vote arrives through the *single-item* manual path — the big Good/Bad buttons **and the hover-vote**, which the user has confirmed are equivalent (both pull the item out of the left queue into the right/verified pile) — add the id to `verified_ids`. Cleanest hook: `vtscore/state/votes.py` `set_vote` / `toggle_vote` / `apply_label_with_click_time`, guarded by `is_find_mode()`. The bulk find-label / Inclusion-slide paths stay unverified by construction.
6. **Expose `verified_ids`** in the `GET /api/votes` payload (a `verified` id array) so the frontend can derive the four buckets in one request (`vtsearch/routes/labels/vote.py`).
7. **Unverified Export.** Extend the label-export `label_filter` (`/api/labels/export`, `vote.py:147`) with `unverified` (emit `good_votes ∪ bad_votes − verified_ids`) and, symmetrically, `verified`. The work-queue dump the user re-imports as a dataset uses `unverified`.

## Frontend changes

1. **Find slider becomes a cutoff control.** `find-view.component.ts onInclusionChange` (currently `setInclusion` → `runFindLabel`, which retrains) now calls the repurposed `POST /api/inclusion` (no retrain). Because the frontend already has every item's score (`sortState.setSortResults(sorted, threshold)` holds `[{id, score}]`), the green/red display updates **optimistically client-side** the instant the slider moves; the POST reconciles server votes for export. No scoring spinner on a slide.
2. **`VoteStateService`**: track a `verifiedIds` set from the new `/api/votes` `verified` field; expose `verifiedIds$`. On a manual vote in find mode, optimistically add the id.
3. **Right panel** (`right-panel.component.ts`), find mode — the four-bucket view:
   - A **Verified Good** list with a header **"[N] Unverified Good"** (N = unverified-good count). Its **Browse / Export / To Dataset** buttons act on *everything below the button* = Verified Good ∪ Unverified Good = the full good set. (`onBrowse`/`onToDataset` already scope over `goodVotes` in `find-view.component.ts:452,475`, so the header/labeling is the new part.)
   - A symmetric **Verified Bad** list with a **"[M] Unverified Bad"** header.
   - Buckets derived from scores + cutoff + `verifiedIds$`.
4. **Left panel** = the work queue: Unverified Good items in score order — what the user walks. Add an **Unverified Export** button (`label_filter=unverified`).
5. **Center auto-advance → the marginal positive** (explicit ask): today `find-view.component.ts:439 onMediaVoted` only reloads votes. In find mode, after any vote (button or hover), select the **lowest-scored unverified item still above the cutoff** — i.e. the Unverified Good nearest the line — via `mediaState.selectMedia(nextId)`. This is a cheap active-learning order (always vote the most marginal/uncertain item next, so "just sit and vote" does the most good) without reimplementing Train's Good/Hard ordering. It's intentionally a little jarring — voting the top of the stack throws you to the boundary — but it makes "sit and vote" the optimal default. The initial seed already uses this exact rule (`find-view.component.ts:237` picks the lowest item ≥ threshold), so seed and advance unify. **The queue is empty exactly when no unverified item remains above the cutoff** — that *is* the done state (show a brief "all positives reviewed" rest state). The center-panel swipe fires `mediaVoted` ~180ms after the vote (`center-panel.component.ts:260`), so the advance rides that callback.
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

- **Find run (pair load)** → score once → `find_scores` frozen, `cutoff` = calibrated default, `find_initial_labels` recorded, `verified_ids` empty; center seeded on the item just above the line (`find-view.component.ts:237`).
- **Slide Inclusion** → re-threshold unverified items only (instant, client-optimistic + cheap `POST /api/inclusion`, no retrain); verified items unmoved.
- **Good on centered item** (button or hover) → vote good + verify → Verified Good → auto-advance to the lowest unverified item still above the cutoff.
- **Bad on centered item** (false positive; button or hover) → vote bad + verify → Verified Bad → auto-advance to the lowest unverified item still above the cutoff.
- **Export / Browse / To Dataset (good panel)** → Verified Good ∪ Unverified Good (flood-fill); unreviewed positives ride along on the detector's call.
- **Unverified Export** → dumps only the untouched work queue for re-import as a dataset.

## Decisions captured

- **Inclusion = pure cutoff, unified across Train and Find, no retrain (CRITICAL).** Inclusion leaves the MLP entirely (fixed architecture + regularization); it only moves the line over the current model's scores. In Find the model is scored once and frozen; in Train the model still retrains *when you vote*, but never on an Inclusion change. **Backwards-compat break:** existing `inclusion` values change meaning and model capacity no longer varies with inclusion.
- **Auto-advance = lowest unverified item above the cutoff** (the marginal positive), not next-in-stack. Doubles as the cheap active-learning order and self-defines the empty/done state.
- **Hover-vote ≡ button-vote.** Both verify the item and move it left→right; both trigger auto-advance.
- **Preserve-verified is automatic.** Verified items carry explicit votes the cutoff never overrides; sliding reclassifies only unverified items. The right panel is undisturbed by anything in the left panel. `verified_ids`/`find_scores`/`cutoff` reset only on a pair change.
- **Positives-only review**, with the big buttons available on any centered item for one-off ActuallyGood/ActuallyBad marks + auto-advance.
- **Ephemeral throughout**; Unverified Export is the persistence escape hatch.
- **Stats eval baseline = the default calibrated cutoff**, fixed at score time, so corrections are measured against the detector's natural recommendation regardless of later slider position.

## Open follow-ups

- **Inclusion parameterization.** Decide how the `inclusion` knob maps to a cutoff offset over scores (raw-score offset, percentile/quantile of the score distribution, or keep the `-10..10` feel mapped monotonically). The name stays **Inclusion** (confirmed). New constraint from the unification: the mapping must be **stable across Train's vote-triggered retrains** — `inclusion` is a position re-resolved over whatever the current scores are, so the same value should mean "the same relative cut" before and after a retrain. Doesn't change the architecture.
- **Done-state polish.** The empty queue (no unverified item above the cutoff) is now well-defined; decide how rich the "all positives reviewed" rest state should be (plain message vs. a summary nudge toward Stats/Export).
- **Under-threshold (false-negative) review surface.** Currently reachable only item-by-item via the left panel + big buttons. If demand appears, add a dedicated below-the-line work queue (the symmetric "find what the detector missed" flow).
