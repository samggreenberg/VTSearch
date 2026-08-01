# Cold-start threshold experiment — the `too_few_default` degenerates (issue #2788)

**Status:** Design (pre-registered). This is the measurement spec for the
cold-start fixes issue #2788 evaluates; production changes only after this
study's decision rules pick a winner. Reuses the calibration-study harness
(`docs/plans/calibration-experiment.md` pattern; instrumentation currently
lives on the `claude/calibration-experiment-2781` branch — see Dependencies).

## Question

After PR #2784, the residual degenerate thresholds (a cut above every or below
every score) are dominated by the `too_few_default` = 0.5 fallback in
`vtscore/training/thresholds.py::compute_fold_orderings` (the `n < 4` /
<2-per-class early returns), concentrated at a modal total of **4 votes**
(measured: 750 steps across the #2781 study, unchanged by #2784, ~39%
self-healing).

**The measured incidence does not describe the app's Autopilot flow.** The
harness trains and thresholds on every step from the first `(≥1 good, ≥1 bad)`
pair (`voting_iterations.py`: `if not good_votes or not bad_votes: continue`),
so with Autopilot's goods-first vote order its first trained step is always
3G+1B — one negative, no stratified split, flat 0.5. The **app** does not train
there. Autopilot's `bad` phase stays on the *text/media* sort
(`label-view.component.ts`: `phase === 'bad'` → `setSortMode(earlySortMode)`,
`setSelectMode('hard')`, where `hard` picks the unlabeled item nearest the
current sort's threshold index — the middle of the text ranking). The first
learned sort fires at the `hard` phase, i.e. at quorum: 3 good + 4 bad, which
always clears the ≥2-per-class guard and takes the conformal path.

So the study asks two questions:

1. **How much cold-start degeneracy survives a production-faithful train
   cadence?** Re-baseline with the harness deferring its first train to
   Autopilot's quorum, matching the app. If the `too_few_default` bar
   collapses, the headline number was a harness artifact and the remaining
   work is scoped to the doors below.
2. **For the doors that genuinely reach `too_few`, which rule wins?** Those
   are: manual **Sort → Learned**, enabled at `good > 0 && bad > 0`
   (`VoteStateService.learnedSortAvailable`) so a user can train at 1G+1B;
   Autopilot **`retrainMode`**, which *does* run learned sort during the
   good/bad phases against an existing (possibly tiny) labelset; the
   **Find/labelset** path (`train_and_threshold`); and
   **`train_detector_from_origins`** at detector load. All are small-labelset
   states, where extra calibration work is cheap. The bar is removing the
   degenerates without making the cut *worse* — a non-degenerate but mid-mass
   threshold that admits half the dataset is not a win.

## Arms

All arms share the production-faithful config (inclusion 0, MLP trainer,
`calibration_fraction` 0.5, `safe_thresholds` False) and the same simulated
Autopilot vote order; they differ only in the threshold rule. Threshold arms
run in-process per cell (like the calibration study's remedial re-pools), so
each cell prices one trajectory's training and scores all arms' thresholds on
it.

- **`baseline`** — dev + #2784 as-is: `too_few_default` 0.5, `calibrate_count`
  2, harness train-every-step cadence. Reproduces the #2781 study's numbers;
  the reference the others are read against.
- **`app_cadence`** — `baseline`'s rule, but the harness defers its first
  train to Autopilot's quorum (3 good **and** 4 bad) and trains every step
  after, matching what the app does. Not a production change — it is the
  correction that tells us how much of the measured `too_few_default` bar any
  user ever sees on the Autopilot path. If this arm zeroes the bar, arms
  below are scoped to the manual / retrain / labelset doors only.
- **`gmm_fallback`** — when the fold split cannot form (the `too_few` early
  returns), threshold with `calculate_gmm_threshold(all_scores)` over the full
  population score distribution instead of 0.5. Non-degenerate by construction
  (midpoint of two component means fit to the actual scores always lies inside
  the score range); zero extra training cost; same rule text/cosine sort
  already uses. Applied at the callers (`_train_and_score_xy` /
  `train_and_threshold`) via the provenance surface, since
  `compute_fold_orderings` is a pure (X, y) function that never sees
  population scores.
- **`fold_boost`** — `calibrate_count` = 8 when the vote count is below 10
  (else 2). Cannot touch the modal-4 step (no valid split exists at any fold
  count) but directly widens the pooled calibration support in the 5–15 vote
  regime, where 2 folds pool only ~4–6 held-out scores and the residual
  conformal-provenance degenerates and the 12-vote budget overshoot live.
  Nearly free: fold fits at n ≤ 10 are milliseconds.
- **`combined`** — `gmm_fallback` + `fold_boost` together (they target
  disjoint steps: votes ≤ 4-with-degenerate-class-counts vs votes 5–15).

`gmm_fallback` has a continuity argument beyond the mechanics: during the
`good`/`bad` phases the user is *already* looking at a cutoff drawn by
`calculate_gmm_threshold` over the text-sort score distribution, and the
safe-threshold ramp below 6 labels is pure GMM as well. Falling back to GMM on
the learned scores keeps a manual early Learned sort on the same rule the
surrounding UI is already using, rather than a flat 0.5 that means nothing
against the model's actual score range.

**Pre-registered non-arms** (parked; revisit only if the arms above fail their
decision rules):

- *Population-interpolated cold-start cut* (LOO-held-out positives blended
  against population score quantiles) — design-heavy; `gmm_fallback` is the
  cheap version of the same idea.
- *Raising `goodToStart` / `badToStart`, or interleaving the vote order* —
  Autopilot already defers its first train past both phase targets, so these
  change which items get voted, not whether a degenerate threshold is
  computed. They are not levers on this bug.
- *Gating the manual Sort → Learned control on ≥2 votes per class* — closes
  the largest remaining door by making the app's manual path do what Autopilot
  already does. It is a UX change with no threshold-rule content, so it is not
  an arm here; if `app_cadence` shows the residual exposure is dominated by
  the manual door, this becomes the cheapest fix and belongs with the
  UI-suppression work below.

## Datasets and cells

Trimmed relative to the calibration study — this is a threshold-rule question,
not an embedder/style question, so one representative arm per calibration
regime:

- **`caltech101_m` × `siglip` × `whole_image`** — row-wise calibration, binary
  voting, clean classes (6 prevalence-spread categories).
- **`visual_genome_m` × `siglip` × `whole_image`** — row-wise calibration
  under region voting and class overlap (23 scale-band categories).
- **`visual_genome_m` × `dinov3_patch` × `max_patch`** — grouped (bag
  max-pool) calibration (same 23 categories).

**Trajectories are truncated at 30 votes** (cold start is over by then; ~5×
cheaper per cell than the 150-vote study) and the budget saved goes to
**8 seeds** — the modal-4 step happens once per trajectory, so power comes
from trajectory count, not trajectory length. Cells (one SLURM task each):
23 × 2 × 8 + 6 × 1 × 8 = **416 cells**, each ~1/5 the cost of a
calibration-study cell. Vote trajectories are identical across threshold arms
through at least vote 7 (the Good/Bad phases select by scores, not by
threshold; the threshold first enters selection at the Hard phase), so the
early-step comparisons are tightly paired where it matters — and `app_cadence`
shares its trajectory with `baseline` exactly, differing only in which steps
get a threshold at all.

## Metrics (per step t, on the held-out test split)

Existing harness columns (`threshold`, `threshold_provenance`, `degenerate`,
`cost`/`fpr`/`fnr`, `oracle_*`, `regret`, `threshold_percentile`) plus,
derived in the summary stage:

- **Degenerate incidence by provenance per step**, t ∈ [4, 30] — the headline
  plot; `gmm_fallback` should zero the `too_few_default` bar, and
  `app_cadence` should show how much of that bar exists on the Autopilot path
  in the first place.
- **Autopilot-reachable share** — fraction of `too_few_default` steps that
  survive the `app_cadence` restriction. This is the number that sizes the
  whole issue: if it is ~0, the production fix is scoped to the manual /
  retrain / labelset doors and the correct first deliverable is the harness
  cadence fix, not a threshold-rule change.
- **Time-to-first-non-degenerate** and time-to-first-`conformal`-provenance
  step, per trajectory.
- **Cold-start regret** — mean `cost − oracle_cost` over t ≤ 20, and the
  t = 12 cell specifically (the one `inclusion-calibration-bias.md` flags).
- **Threshold sanity** — `threshold_percentile` at t ∈ {4, 5, 6}: catches the
  `gmm_fallback` failure mode mere non-degeneracy hides (a unimodal cold-start
  score distribution puts the GMM midpoint mid-mass, admitting ~half the
  dataset).
- **Threshold churn** — per-step |Δthreshold|, the user-visible "jumps to the
  top, normal again one click later" instability.
- **Late-window non-inferiority** — for `fold_boost` (whose thresholds differ
  from baseline wherever the boost was active), regret over t ∈ [20, 30] must
  not exceed baseline's. For `gmm_fallback` the check is exact: thresholds
  must be bit-identical to baseline from the first valid-calibration step on.

## Hypotheses (pre-registered, honest priors)

- **`app_cadence` removes nearly all of the `too_few_default` steps.** Under
  Autopilot's quorum the first trained step is 3G+4B, which always has ≥2 per
  class; the only survivors should come from trajectories where the vote
  sequence cannot supply 4 negatives early (tiny or highly imbalanced
  categories). Predicted survival < 10% of the 750.
- `gmm_fallback` zeroes the `too_few_default` degenerates and does not worsen
  cold-start regret in any (dataset, regime) cell. Risk: mid-mass cuts on
  unimodal score distributions show up as `threshold_percentile` near 50 with
  elevated FPR at t = 4 — possible on VG (overlapping classes), unlikely on
  Caltech.
- `fold_boost` cuts the conformal-provenance degenerates at t ∈ [5, 15] by
  half or more and shrinks the 12-vote budget overshoot, at no late-window
  cost.
- `combined` is the best arm overall; the two fixes don't interact (disjoint
  steps).
- No arm moves Caltech's late-window regret (already ≈ 0 post-#2784).

## Decision rules (pre-registered)

1. **Scope first.** If `app_cadence` leaves < 10% of the `too_few_default`
   steps, record that the headline incidence was a harness artifact: correct
   the harness's train cadence (its own change — every study built on it
   over-weights the pre-quorum steps), restate #2788's severity against the
   surviving doors, and judge the arms below on the manual / retrain /
   labelset regime rather than on the raw count.
2. Adopt `gmm_fallback` in production iff `too_few_default` degenerates → ~0
   **and** cold-start regret (t ≤ 20) is not worse than baseline by more than
   0.02 in any (dataset, regime) cell. This holds regardless of rule 1: the
   surviving doors are real, and the fix is cheap and rule-consistent with the
   text-sort cutoff the user is already looking at.
3. Adopt `fold_boost` iff it cuts conformal-provenance degenerate steps at
   t ∈ [5, 15] by ≥ 50% **or** shrinks the t = 12 budget excess, without
   failing late-window non-inferiority. Note this window is *post-quorum*, so
   unlike the `too_few` bar it is fully Autopilot-reachable.
4. If `gmm_fallback` fails on mid-mass cuts (rule-2 regret margin breached
   with `threshold_percentile` mid-range at t = 4), fall back to evaluating
   the parked population-interpolation cut — as a new pre-registered arm, not
   a post-hoc tweak.
5. Each adoption is its own production PR against issue #2788, with the
   plan-file item below pruned when it ships.

## Dependencies

The provenance/degenerate/oracle instrumentation and the runner live on the
unmerged `claude/calibration-experiment-2781` branch
(`vtscore/eval/calibration_metrics.py`, harness extensions in
`voting_iterations.py`, `classify_threshold_provenance` in `thresholds.py`,
`scripts/experiments/calibration/`). This study either runs on that branch or
waits for its merge to `dev`; the arms here additionally need a threshold-rule
hook in the harness (per-arm threshold functions over the shared trajectory,
mirroring how the remedial re-pools shared fold models).

## Independent of this study

Two changes need no measurement and can ship on their own:

- **Suppress the confident Inclusion framing** until a valid calibration
  exists (the third option issue #2788 lists) — a "provisional threshold" flag
  on the training response plus a frontend gate.
- **Gate the manual Sort → Learned control** on ≥2 votes per class, so the
  app's manual path defers the way Autopilot's phases already do
  (`learnedSortAvailable` currently only requires one of each). Whether this
  is the *main* fix depends on rule 1's Autopilot-reachable share, but it is
  independently defensible: a threshold computed from a 1-vs-1 fit is not
  something to show a user with confidence.

## Open work

<!-- item-sep -->

- **Add the threshold-arm hook to the harness** — per-arm threshold rules
  evaluated on the shared per-step trajectory (fold orderings + population
  scores), emitting one row per (step, arm); `baseline` must stay
  byte-for-byte the calibration study's behaviour.

<!-- item-sep -->

- **Add the `app_cadence` train gate to the harness** — an opt-in mode that
  defers the first train to Autopilot's quorum (3 good and 4 bad) instead of
  the first `(≥1 good, ≥1 bad)` pair, so a simulated trajectory visits the
  same trained states the app does. Default-off for byte-for-byte
  reproducibility of the existing studies; if rule 1 fires, promoting it to
  the default is its own change with its own re-baselining.

<!-- item-sep -->

- **Build the runner** — `scripts/experiments/coldstart_threshold/` in the
  standard stage layout (prepare → cells → summarize), reusing the
  calibration study's prepare pickles for the three (dataset, embedder)
  pairs; 30-vote truncation and 8 seeds as env knobs; CPU-smoke-tested before
  Grid submission.

<!-- item-sep -->

- **Run on the Grid + write the report** — owner-gated on Grid access;
  `docs/experiments/coldstart-threshold/REPORT.md` in the standard report
  style; verdict flows through the pre-registered decision rules.

<!-- item-sep -->

- **Production adoption PR(s)** — gated on the report; whichever of
  `gmm_fallback` / `fold_boost` passes its decision rule lands as its own PR
  resolving issue #2788 (the UI-suppression option above may ship
  independently at any time).
