# Find Verification Workflow

**Status:** **Phase 1 (backend spine) shipped** — full `./run-tests.sh` green. Phase 2 (frontend) deferred to a session where the UI can be eyeballed (no browser in the build container). See **What shipped / Open follow-ups** at the bottom.

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

This is a deliberate change from today, where the Inclusion slider retrains the MLP on every change (`-10..10`, feeding hidden-layer width + regularization + calibration). **Inclusion is unified across both Train and Find as a no-retrain cutoff.** It leaves the *model* entirely: the architecture (hidden width) and regularization become fixed defaults, independent of Inclusion. What Inclusion keeps doing is its real job — it's the **exponential weight on false-positive vs. false-negative cost in the cross-calibrated min-cost threshold search** (`calculate_cross_calibration_threshold` → `find_optimal_threshold`, `vtscore/training/thresholds.py`), run over the **LabelSet's** scored vectors (the labeled examples that trained the detector — *not* anything in the Find dataset). Warping that cost function up or down slides the threshold more or less inclusive. Crucially, once the model no longer depends on Inclusion, the labelset fold-scores are inclusion-independent and cacheable, so an Inclusion change re-runs only the cheap min-cost search over cached scores — no MLP fit at all. That threshold is then applied to the Find haystack's frozen scores. Training still happens in Train mode **when you vote** (that's the training loop) — but an Inclusion change never retrains, in either mode.

> **Backwards-compat break (call out to user):** existing detectors' stored `inclusion` values change meaning (training-bias → cutoff position), and model capacity no longer varies with inclusion. Per repo policy we make the clean change and surface the break rather than shimming old behavior.

Freezing the scores is what makes the rest cheap:

- **Instant slide.** No GPU work on a cutoff change — just re-thresholding fixed numbers.
- **Preserve-verified is automatic.** Verified items carry an explicit human vote that the cutoff never overrides; sliding only reclassifies *unverified* items. Nothing the user does in the left panel can disturb the right panel.
- **"At what cutoff would this item return?" is a trivial inversion** — every item has one fixed score, so the answer is just where the line crosses it (see Stats, below).

## Data model

State on the active `DetectorContext` (all in-memory, ephemeral — blessed by the "No Persisted Vectors or MLPs" rule, which explicitly allows process-scoped caches on the context):

- **`find_scores: dict[int, float]`** (new) — the frozen per-item detector scores from the single scoring pass.
- **`inclusion`** (already exists) — now the cutoff knob: the FP/FN cost weight in the min-cost threshold search over the LabelSet vectors. It produces the cutoff (the existing **`threshold`** field) with no MLP retrain; that cutoff is applied to `find_scores`. ("Cutoff" below = `threshold`, computed from `inclusion` via the labelset min-cost calibration.)
- **`calibration_cache`** (already exists — **re-key it**) — today it memoizes `(key, threshold)` where `key` fingerprints **inclusion** among its inputs (`core.py:391-399`), so every Inclusion change misses and refits all K fold-MLPs. Repurpose it to memoize the **K fold orderings** — each fold's held-out `(calibrate scores, labels)` — under a key that **excludes inclusion**. The fold scores are inclusion-independent once `train_model` drops inclusion, so an Inclusion change hits the cache and re-runs only `find_optimal_threshold` per fold (an O(n) sweep, no torch). Invalidation is unchanged — the cache already busts on labelset/embedder/calibration-setting changes (`dataset_sync.py:208`, context reset); inclusion is simply no longer in the key.
- **`good_votes` / `bad_votes`** — already exist. In find mode these hold the items' current good/bad assignment.
- **`verified_ids: set[int]`** (new) — the items the human has explicitly acted on this session.

**Resolved label of any item** = `good` if it's a verified-good vote; `bad` if verified-bad; otherwise `score ≥ cutoff`. The four logical states fall out (the **right panel renders the two verified ones**, stacked Good-over-Bad; the **left panel is the two unverified ones** — the work queue):

```
Verified Good   = verified_ids ∩ good_votes
Verified Bad    = verified_ids ∩ bad_votes
Unverified Good = { i ∉ verified_ids : find_scores[i] ≥ cutoff }
Unverified Bad  = { i ∉ verified_ids : find_scores[i] <  cutoff }
```

`verified_ids` is required because confirming an item *without changing its side of the line* (the common case — detector says good, human agrees) leaves no trace in the scores; only an explicit "I looked at this" set captures it.

**Cutoff-slide semantics (no re-score, preserve verified):** moving the slider sets `cutoff` and re-thresholds **only unverified items** against the frozen `find_scores`. Verified items keep their human vote wherever their score lands. An unverified item can flip Unverified Good ↔ Unverified Bad as the line moves; that's expected. No retrain, no re-embed.

**Two scopes, deliberately separated:**

- **Detector-scoped (survives a haystack switch):** the K fold orderings (`calibration_cache`) and the `inclusion`→`threshold` result derived from them. These depend only on the LabelSet + embedder + calibration settings, not the haystack, so pointing the same detector at a different dataset reuses them as-is. They recompute only when the LabelSet changes (Train votes), the embedder changes, or a calibration setting changes — the existing `calibration_cache` invalidation, minus inclusion.
- **Find-session-scoped (tied to the current haystack):** `find_scores`, `good_votes`/`bad_votes`, and `verified_ids`. These reset when the haystack (dataset) or the detector changes — `reloadForNewPair` in `find-view.component.ts:146` — and the haystack is re-scored. `find_scores` is the *only* thing a haystack switch recomputes; the threshold rides along from detector scope.

## Backend changes

1. **`DetectorContext`** (`vtscore/state/core.py:322`): add `find_scores: dict[int, float]` and `verified_ids: set[int]` (both `field(default_factory=...)`). The cutoff itself reuses the existing `threshold` field (resolved from `inclusion`); no new cutoff field. Both new fields reset only on a pair change, not on an Inclusion slide.
2. **Inclusion leaves the model (training-path refactor).** `train_model` (`vtscore/training/mlp.py:110`) drops its `inclusion_value` parameter — hidden width + regularization become fixed defaults — and `calculate_cross_calibration_threshold` (`vtscore/training/thresholds.py:227`) stops threading `inclusion_value` into the per-fold `train_model` calls. `inclusion` stays only where it belongs: in `find_optimal_threshold` (the FP/FN cost warp). `POST /api/inclusion` (`sorting.py:760`) stops calling `invalidate_loaded_detector_models()` (`vtscore/state/__init__.py:189-200`) — an Inclusion change no longer drops the MLP; it re-runs only `find_optimal_threshold` over the cached K fold orderings (`calibration_cache` re-keyed to exclude inclusion — see Data model) to get a new `threshold`. This is the unified Train+Find behavior; **it is the backwards-compat break** flagged above.
3. **Score once, then stop re-scoring (Find).** The find-label run (`vtsearch/routes/detectors/scoring.py`) trains/loads the model and scores all items as today, but now **caches the scores in `find_scores`**, sets the cutoff to the neutral-calibrated default, applies labels by that default (the initial split), and records `find_initial_labels` (the detector's natural call at the default cutoff — the stable eval baseline; see Stats). It no longer reruns on a slider change.
4. **Repurposed Inclusion endpoint (shared by Train and Find).** `POST /api/inclusion` no longer retrains; it sets `inclusion`, recomputes the cutoff via the labelset min-cost search (cheap — fold-scores cached, only `find_optimal_threshold` re-runs), and re-thresholds the **unverified** items against the frozen `find_scores` (in Find, leaving verified items and `find_initial_labels` untouched). No MLP work. The Find slider calls this same endpoint; there is no separate `/api/find/cutoff`.
5. **Mark-verified on manual votes in find mode.** When `find_mode` is True and a vote arrives through the *single-item* manual path — the big Good/Bad buttons **and the hover-vote**, which the user has confirmed are equivalent (both pull the item out of the left queue into the right/verified pile) — add the id to `verified_ids`. Cleanest hook: `vtscore/state/votes.py` `set_vote` / `toggle_vote` / `apply_label_with_click_time`, guarded by `is_find_mode()`. The bulk find-label / Inclusion-slide paths stay unverified by construction.
6. **Expose `verified_ids`** in the `GET /api/votes` payload (a `verified` id array) so the frontend can split verified (right panel) from unverified (left work queue) in one request (`vtsearch/routes/labels/vote.py`).
7. **Unverified Export.** Extend the label-export `label_filter` (`/api/labels/export`, `vote.py:147`) with `unverified` (emit `good_votes ∪ bad_votes − verified_ids`) and, symmetrically, `verified`. The work-queue dump the user re-imports as a dataset uses `unverified`.

## Frontend changes

1. **Find slider becomes a cutoff control.** `find-view.component.ts onInclusionChange` (currently `setInclusion` → `runFindLabel`, which retrains) now calls the repurposed `POST /api/inclusion` (no retrain). Because the frontend already has every item's score (`sortState.setSortResults(sorted, threshold)` holds `[{id, score}]`), the green/red display updates **optimistically client-side** the instant the slider moves; the POST reconciles server votes for export. No scoring spinner on a slide.
2. **`VoteStateService`**: track a `verifiedIds` set from the new `/api/votes` `verified` field; expose `verifiedIds$`. On a manual vote in find mode, optimistically add the id.
3. **Right panel** (`right-panel.component.ts`), find mode — **two verified buckets, stacked**. It lists only verified items; the unverified live on the left.
   - **Verified Good** (top) with a count-folding header **"[N] Unverified Good"** (N = the unverified-good count on the left). Its **Browse / Export / To Dataset** buttons act on *everything the header folds in* = Verified Good ∪ Unverified Good = the full good set. (`onBrowse`/`onToDataset` already scope over `goodVotes` in `find-view.component.ts:452,475`, so the header label + folding the cutoff-derived unverified set into the scope is the new part.)
   - **Verified Bad** (bottom), symmetric, with a **"[M] Unverified Bad"** header and bad-set buttons.
   - The two verified lists come from `good_votes`/`bad_votes` ∩ `verifiedIds$`; the header counts come from scores + cutoff (the unverified sets that live on the left).
4. **Left panel** = the work queue: the **unverified** items (the scored haystack — unverified Good above the cutoff, unverified Bad below), in score order. This is what the user walks and auto-advances through; verifying an item moves it off the left and into the right. Add an **Unverified Export** button (`label_filter=unverified`).
5. **Center auto-advance → the marginal positive** (explicit ask): today `find-view.component.ts:439 onMediaVoted` only reloads votes. In find mode, after any vote (button or hover), select the **lowest-scored unverified item still above the cutoff** — i.e. the Unverified Good nearest the line — via `mediaState.selectMedia(nextId)`. This is a cheap active-learning order (always vote the most marginal/uncertain item next, so "just sit and vote" does the most good) without reimplementing Train's Good/Hard ordering. It's intentionally a little jarring — voting the top of the stack throws you to the boundary — but it makes "sit and vote" the optimal default. The initial seed already uses this exact rule (`find-view.component.ts:237` picks the lowest item ≥ threshold), so seed and advance unify. **The queue is empty exactly when no unverified item remains above the cutoff** — that *is* the done state (show a brief "all positives reviewed" rest state). The center-panel swipe fires `mediaVoted` ~180ms after the vote (`center-panel.component.ts:260`), so the advance rides that callback.
6. **Big Good/Bad buttons must work in find mode.** Ensure the `voting-overlay` is rendered and not `disabled` outside the brief initial-scoring window; its `voted` → `castVote` → `submitToggleVoteAndRecord` path marks the item verified (item 4 above).

## Detector evaluation: the Stats button

The real reason to verify a test haystack is **"is this detector worth subscribing to on my data?"** Stats surfaces that via a **Stats** button on the right panel (same affordance as the Datasets Dashboard card → `vt-dataset-stats-modal`).

**Stats counts ALL items, not just the verified ones.** Exactly like Export / Browse / To-Dataset, it treats unverified items as if verified — adopting the detector's current-cutoff call as their truth. This *risks false confidence* in the detector (an unverified item trivially "agrees" with the detector at the current cutoff), but that's the accepted price of not verifying every item, and the user still wants the totals: skip verifying the obvious top and middle-to-bottom, and those counts still show up. So **truth = the full `good_votes` / `bad_votes` adopted set** (human votes flood-filled with the detector's call on everything untouched); `verified_count` is reported separately as how much was actually human-checked.

**No new tracked state.** The detector's original call per item is `find_initial_labels` (its good/bad at the default calibrated cutoff). Crossing the adopted label against it gives a 2×2:

| | Detector said **Good** | Detector said **Bad** |
|---|---|---|
| **Adopted → Good** | Confirmed positive *(agree)* | Rescued false negative *(correction)* |
| **Adopted → Bad** | Culled false positive *(correction)* | Confirmed negative *(agree)* |

Unverified items adopted the detector's own call, so they fall in the confirmed cells (the false-confidence inflation); corrections come from human overrides.

Reported figures:

- **Totals** — `total_good` / `total_bad` (adopted, all items) and `verified_count` (how many the human actually checked).
- **Agreements vs. corrections** and the **agreement rate** — `confirmed_good + confirmed_bad` vs. `culled_fp + rescued_fn`, over all items.
- **Precision** — `confirmed_good / (confirmed_good + culled_fp)`: of everything the detector originally called good, how much the adopted set keeps (inflated by unverified items — the false-confidence number).
- **FP/FN-vs-Inclusion sweep — the headline chart.** For every inclusion `i ∈ [−10, 10]`, compute the calibrated threshold `t_i` (cheap: re-run the min-cost average over the **cached fold orderings** at `i` — 21 O(n) sweeps, no MLP work), then over **all adopted items** count `FP_i` = adopted-Bad items scoring `≥ t_i` (detector at `i` would wrongly include) and `FN_i` = adopted-Good items scoring `< t_i` (detector at `i` would wrongly drop). Plot both curves against inclusion, current inclusion marked. This is the precision/recall tradeoff *on the user's own data*: as inclusion rises the threshold drops, FP climbs, FN falls — the user reads off the inclusion that best balances the two. At the current inclusion the unverified items contribute nothing (they sit on their own side of the line by construction), so the floor is the verified corrections; moving away from current makes unverified items flip and the curves climb. Subsumes the per-item "at what cutoff would this return?" idea — that's one item's crossing point on this chart.
- **Run context** — the current cutoff/inclusion.

### Backend
- **`GET /api/find/stats`**: returns (a) adopted totals + `verified_count`, (b) the 2×2 confusion of the adopted label (`good_votes`/`bad_votes`, all items) × `find_initial_labels` and the derived rates, and (c) the **21-point sweep** `[{inclusion, threshold, false_pos, false_neg}]` for `inclusion ∈ [−10, 10]` over all adopted items — thresholds from the cached fold orderings, counts from the frozen `find_scores`. Pure read; cheap; no new state.

### Frontend
- **`vt-find-stats-modal`** (mirrors `vt-dataset-stats-modal`): standalone modal on the shared `ModalComponent`, fetches `GET /api/find/stats` on open, renders the 2×2 + rates + the **FP/FN-vs-inclusion line chart** (current inclusion marked). The chart is a small hand-rolled inline SVG in the existing dependency-free viz style (cf. the dashboard card's pie) — no new charting dependency.
- **Stats button** in the right panel (find mode), beside Browse / Export / To Dataset, emitting a `stats` event `find-view` handles — same wiring as the dashboard card's `stats` output.

## State-transition summary

- **Find run (pair load)** → score once → `find_scores` frozen, `cutoff` = calibrated default, `find_initial_labels` recorded, `verified_ids` empty; center seeded on the item just above the line (`find-view.component.ts:237`).
- **Slide Inclusion** → re-threshold unverified items only (instant, client-optimistic + cheap `POST /api/inclusion`, no retrain); verified items unmoved.
- **Good on centered item** (button or hover) → vote good + verify → Verified Good → auto-advance to the lowest unverified item still above the cutoff.
- **Bad on centered item** (false positive; button or hover) → vote bad + verify → Verified Bad → auto-advance to the lowest unverified item still above the cutoff.
- **Export / Browse / To Dataset (good panel)** → Verified Good ∪ Unverified Good (flood-fill); unreviewed positives ride along on the detector's call.
- **Unverified Export** → dumps only the untouched work queue for re-import as a dataset.

## Decisions captured

- **Inclusion = pure cutoff, unified across Train and Find, no retrain (CRITICAL).** Inclusion leaves the model entirely (fixed architecture + regularization); it stays the FP/FN cost weight in the labelset min-cost threshold search, which yields the cutoff with no MLP fit. In Find the haystack is scored once and frozen; in Train the model still retrains *when you vote*, but never on an Inclusion change. **Backwards-compat break:** existing `inclusion` values change meaning and model capacity no longer varies with inclusion.
- **Auto-advance = lowest unverified item above the cutoff** (the marginal positive), not next-in-stack. Doubles as the cheap active-learning order and self-defines the empty/done state.
- **Hover-vote ≡ button-vote.** Both verify the item and move it left→right; both trigger auto-advance.
- **Preserve-verified is automatic.** Verified items carry explicit votes the cutoff never overrides; sliding reclassifies only unverified items. The right panel is undisturbed by anything in the left panel. Find-session state (`verified_ids`/`find_scores`/votes) resets when the haystack or detector changes; the detector-scoped K fold orderings + threshold survive a same-embedder haystack switch (see Data model).
- **Positives-only review**, with the big buttons available on any centered item for one-off ActuallyGood/ActuallyBad marks + auto-advance.
- **Ephemeral throughout**; Unverified Export is the persistence escape hatch.
- **Stats eval baseline = the default calibrated cutoff**, fixed at score time, so corrections are measured against the detector's natural recommendation regardless of later slider position.

## What shipped (Phase 1 — backend spine)

All under `./run-tests.sh` (full suite green). Note one doc-vs-code correction discovered during build: `inclusion` only ever fed the **class-weight bias** in `train_model` (the hidden width was already `_auto_hidden_dim(n_train)` and regularization was fixed), so "fixed architecture" was already true — the refactor just removed the class-weight bias.

- **Inclusion decoupled from the MLP.** `train_model` dropped its `inclusion_value` param + class-weight bias; all production/test callers updated. `inclusion` now lives only in `find_optimal_threshold` (the FP/FN cost warp). The model — and every item's score — is inclusion-independent.
- **Fold-ordering cache.** `calculate_cross_calibration_threshold` split into `compute_fold_orderings` (inclusion-independent) + `threshold_from_fold_orderings` (per-inclusion min-cost). `calibration_cache` re-keyed to drop inclusion and store the K fold orderings, so an Inclusion change reuses the orderings and only re-runs the cheap min-cost search.
- **No-retrain Inclusion.** `set_inclusion` now calls `recompute_detector_thresholds_for_inclusion` (re-derive threshold from the fold cache, leave the model in place) instead of `invalidate_loaded_detector_models`.
- **Verification state.** `DetectorContext.verified_ids` + `find_scores` (in-memory; cleared on pair change). Mark-verified on single-item find-mode votes (`set_vote`/`toggle_vote`), un-verify on un-vote.
- **APIs.** `verified` array on `GET /api/votes`; `unverified`/`verified` `label_filter` on `/api/labels/export` (atomic via `VoteSnapshot.verified_ids`); new read-only `GET /api/find/stats` (adopted-label 2×2 confusion + the −10…10 FP/FN sweep, both over **all** items — unverified flood-filled like Export/Browse, with `verified_count` as context). `find-label` freezes `find_scores`.

## Open follow-ups

- **Phase 2 — frontend (the whole UI).** Find slider → `POST /api/inclusion` (no retrain) with optimistic client-side re-threshold; 2 verified buckets + count-folding headers on the right; left work queue + Unverified Export button; marginal-positive auto-advance; big-button/hover verify wired through; `vt-find-stats-modal` rendering the 2×2 + FP/FN sweep as an inline SVG line chart.
- **Safe-threshold blend on Inclusion slide.** `recompute_detector_thresholds_for_inclusion` re-derives the *raw* cross-cal threshold from the fold orderings; it does not re-apply `calculate_safe_threshold` blending (which needs the haystack score distribution). When `safe_thresholds` is on, an Inclusion slide's threshold will differ slightly from a full retrain's. `find_scores` is available to blend if this matters; deferred (safe_thresholds is opt-in).
- **Done-state polish.** The empty queue (no unverified item above the cutoff) is now well-defined; decide how rich the "all positives reviewed" rest state should be (plain message vs. a summary nudge toward Stats/Export).
- **Under-threshold (false-negative) review surface.** Currently reachable only item-by-item via the left panel + big buttons. If demand appears, add a dedicated below-the-line work queue (the symmetric "find what the detector missed" flow).
