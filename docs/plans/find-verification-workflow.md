# Find Verification Workflow

**Status:** Phases 1–4 all shipped (backend spine, frontend, verified-only colouring, left work-queue actions); full `./run-tests.sh` + `npm run build:prod` green. Built headless (no browser in the container), so the UI has **not been visually eyeballed** — a manual pass on a real screen is still owed. Open follow-ups below; shipped detail collapsed under "What shipped".

## The reframe (essential framing)

Verification is **not a new mode** — it's a richer label vocabulary on the *existing* Find interface. Find scores every item; a cutoff splits **good** (≥ cutoff) from **bad** (< cutoff). We add one distinction on top — **has a human touched this item** — yielding four states:

| State | Meaning | How it arises |
|-------|---------|---------------|
| **Unverified Good** | Above cutoff, detector's call, human hasn't looked | Default above-cutoff |
| **Unverified Bad** | Below cutoff, detector's call, human hasn't looked | Default below-cutoff |
| **Verified Good** | Human confirmed/rescued as good | Human hit **Good** |
| **Verified Bad** | Human confirmed/culled as bad | Human hit **Bad** |

Same screen, no toggle: a user who just wants "run the detector and export the hits" never touches a verify button; a user who verifies walks the left work queue and promotes items into the right verified pile.

**Core mechanic:** the detector scores every item **once**, frozen for the session (`find_scores`). The **Inclusion** slider is a **cutoff** over those frozen scores — it re-thresholds only *unverified* items, never retrains the MLP, never re-embeds, never reorders. Inclusion left the model entirely (fixed architecture + regularization); it stays the FP/FN cost weight in the labelset min-cost threshold search (`find_optimal_threshold`), so a slide re-runs only that cheap search over cached K fold orderings. Verified items carry an explicit human vote the cutoff never overrides. **Backwards-compat break (already surfaced):** existing detectors' stored `inclusion` values changed meaning (training-bias → cutoff position) and model capacity no longer varies with inclusion.

## Open follow-ups

- **UI not visually verified.** Built headless; the right-panel folding headers, the SVG chart, the Unverified Export button placement, and the now-uncoloured left work queue should be eyeballed on a real screen.
- **Safe-threshold blend on Inclusion slide.** `recompute_detector_thresholds_for_inclusion` re-derives the *raw* cross-cal threshold from the fold orderings; it does not re-apply `calculate_safe_threshold` blending (which needs the haystack score distribution). When `safe_thresholds` is on, an Inclusion slide's threshold will differ slightly from a full retrain's. `find_scores` is available to blend if this matters; deferred (safe_thresholds is opt-in).
- **Done-state polish.** The empty queue (no unverified item above the cutoff) is now well-defined; decide how rich the "all positives reviewed" rest state should be (plain message vs. a summary nudge toward Stats/Export).
- **Under-threshold (false-negative) review surface.** _Partially addressed:_ the boundary-walk auto-advance (`advanceToBoundary`) now serves below-the-line items too, alternating above/below the cutoff so a "just sit and vote" pass naturally surfaces marginal false negatives near the boundary. A *dedicated* below-the-line work queue (its own panel/order, rather than interleaving into the centre walk) remains unbuilt; add it if demand appears.

## Key design decisions (reference)

- **Inclusion = pure cutoff, unified across Train and Find, no retrain (CRITICAL).** In Find the haystack is scored once and frozen; in Train the model still retrains *when you vote*, but never on an Inclusion change.
- **Auto-advance = boundary walk, alternating above/below the cutoff** (`advanceToBoundary` + `nextFindSide`). Each vote jumps the centre to the nearest unverified item on the side it's that turn, so "just sit and vote" samples both faces of the decision boundary. A fresh score restarts on the `above` side (the marginal positive seed). Queue empty (done state) only when nothing unverified remains on either side. _(Superseded the original positives-only "lowest unverified above cutoff" rule.)_
- **Hover-vote ≡ button-vote.** Both verify the item, move it left→right, and trigger auto-advance.
- **Preserve-verified is automatic.** Find-session state (`verified_ids`/`find_scores`/votes) resets when the haystack or detector changes; the detector-scoped K fold orderings + threshold survive a same-embedder haystack switch.
- **Stats counts ALL items** (unverified flood-filled with the detector's call, `verified_count` reported separately), with the eval baseline = the default calibrated cutoff fixed at score time. The headline is the FP/FN-vs-Inclusion sweep over `[−10, 10]` on the user's own data.
- **Ephemeral throughout**; Unverified Export (`label_filter=unverified`) is the persistence escape hatch.

## What shipped

Each phase landed green under `./run-tests.sh` (+ `npm run build:prod` for frontend).

- **Phase 1 — backend spine.** Inclusion decoupled from the MLP (`train_model` dropped `inclusion_value` + class-weight bias; `inclusion` lives only in `find_optimal_threshold`). Fold-ordering cache: `calculate_cross_calibration_threshold` split into `compute_fold_orderings` + `threshold_from_fold_orderings`; `calibration_cache` re-keyed to exclude inclusion. No-retrain Inclusion (`set_inclusion` → `recompute_detector_thresholds_for_inclusion`). `DetectorContext.verified_ids` + `find_scores` (in-memory, cleared on pair change; mark-verified on single-item find-mode votes). APIs: `verified` array on `GET /api/votes`, `unverified`/`verified` `label_filter` on `/api/labels/export`, new read-only `GET /api/find/stats` (adopted-label 2×2 + −10…10 FP/FN sweep); `find-label` freezes `find_scores`. (Doc-vs-code note: hidden width was already `_auto_hidden_dim(n_train)`, so the refactor only removed the class-weight bias.)
- **Phase 2 — frontend.** No-retrain slide (`find-view.onInclusionChange` → `POST /api/inclusion`, no `runFindLabel`; backend `rethreshold_unverified_find_items`). `VoteStateService.verifiedIds$` with optimistic add/remove. Right panel = verified pile (`good/bad ∩ verified`, folded "[N] Unverified …" headers; buttons act on the full good/bad set). Boundary-walk auto-advance (`advanceToBoundary` + `nextFindSide`; "All items reviewed" toast). Unverified Export button (shared `vt-export-modal` `initialFilter`). `vt-find-stats-modal` with a dependency-free inline-SVG FP/FN-vs-inclusion chart.
- **Phase 3 — verified-only colouring.** Items colour green/red only once verified (`find-view.unverifiedSortOrder` feeds the left queue the ranking minus verified items; media-list + stripe get empty vote sets). Big Good/Bad buttons verify instead of un-toggling (VoteStateService find-mode flag; `currentState()` reads unverified as `'none'`; `effectiveGood`/`effectiveBad` gate the centre buttons).
- **Phase 4 — left work-queue actions.** Browse / To Dataset / Export beside the Inclusion control, scoped to unverified positives (`onBrowseUnverified` / `onToDatasetUnverified`, UI-only `unverified_good` filter resolved to the server `unverified` partition). Browse-canvas "Remove from Good" became "Verified Good" / "Verified Bad" (`browse-selection-panel`, gated on `canVerify`), marking + verifying + dropping the selection.
- **Bug fixes (post-Phase-4).** Inclusion-slide silent no-op fixed: `train_and_threshold` now takes an optional `det_ctx` and caches the fold orderings (find-label / detector-load cold-train path threaded through), and the Find view derives Browse/To-Dataset ids from the live `sortOrder` + `threshold`; regression `tests/detectors/test_find_inclusion_slide.py`. Verified-items-vanish race fixed: `/api/votes` reads stamped with a monotonic `votesSeq`, stale responses dropped (`applyVotesFresh`, `clear()` advances the watermark); regression `frontend/src/app/services/vote-state.service.spec.ts`.
