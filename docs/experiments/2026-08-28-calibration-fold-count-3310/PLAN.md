# Does more cross-calibration ever pay? Fold count vs its wall-clock price (#3310)

**Status:** pre-registered before the run. Decision rules below are fixed at
submission time; the report records the verdict they produce.

## The question

Issue #3310 asks whether the old "more calibration folds is WORSE" result can
possibly be right — *"there's no way that it HELPS to have a WORSE estimator of
the threshold mid VTSearch, right?"* — and, if more folds do help, what they
cost and where they pay best.

The answer to the theory half is already on record, split across two studies,
and it dissolves the paradox rather than confirming it:

- **The "ancient experiment" (#2897) was real but it measured the pooled
  combine rule, not the fold count.** Its K axis swept
  `threshold_from_fold_orderings` — pool every fold's held-out scores, take one
  conformal quantile — while the shipped rule's anchored component stayed
  frozen across K (the gap #3116 named). Binary regret rising monotonically in
  K is that rule's signature, not a property of folds.
- **#3115 then factored the mechanism**: averaging per-fold cuts beats pooling
  their scores in *both* voting modes, and the pooling penalty *grows with K*
  on region voting. Pooling K folds builds a mixture of K different models'
  score distributions and reads extreme order statistics of it
  (`conformal_threshold`'s gap midpoint reads `min(pos)`, a statistic that
  drifts down as the pool grows) — so the pooled rule's **target moves with
  K**. More folds give a better estimate of a K-dependently-worse target. No
  paradox.

For the **shipped** rule (`fold_anchored_gmm_threshold`, per-fold anchored
mixtures combined as a `qmean` of fold quantiles) the theory runs the other
way: `calibrate_count` draws **independent repeated splits** at a fixed
per-fold size (`compute_fold_orderings`), so per-fold statistics are i.i.d.
draws and the combined quantile is a mean of K of them — same mean at every K,
variance falling as 1/K. More folds cannot systematically hurt; they buy
variance reduction, saturating fast, worth most where per-fold noise is
largest — **small labelsets**. Which is the issue's own complementarity
observation: the folds that help most (few votes) are also the cheapest to fit
(fit cost grows with the labelset), so an *adaptive* count — more folds early,
decaying to 2 — could take most of the benefit for almost none of the cost.

The laptop bench `scripts/experiments/calibration/scratch_folds_3310.py`
(synthetic 2-class Gaussian embeddings, production code paths end to end,
oracle-referenced regret, 40 replicates per cell; full tables in issue #3310)
confirms both halves and adds a third finding that reshapes the question:

- **Averaging improves with K, most where labelsets are small.** `tmean`
  regret falls near-monotonically in K, by −0.005 to −0.02 at n ≤ 16 votes
  (saturating by K≈6–8) and by less at n = 80 — the variance-reduction
  prediction, including its decay.
- **The pooled rule's target visibly moves with K.** Its mean threshold
  drifts with K (0.502 → 0.475 across K=1→16 at n=8) while the anchored
  rule's does not budge; the drift *helps* at n ≤ 12 and *hurts* at
  n = 24–40 on cleanly separated classes — which is #2897's deep-regime
  binary signature, reproduced and explained.
- **The shipped anchored rule is nearly K-invariant on this bench** — paired
  |Δ| ≤ ~0.002 at every K, no consistent sign. Mechanically sensible: at
  κ = 0.3 each fold's cut is dominated by its ~3000-point haystack fit, so
  per-fold draws barely vary and averaging more of them buys almost nothing.
  On a single-vector geometry, K may simply not be a live knob on the shipped
  path — the population term already bought the stability.

What the bench cannot settle — being synthetic, single-vector, and
acquisition-free — is whether that K-invariance survives real data and
**region voting**, where per-fold variance is far larger and #2897 measured
real (if pooled-rule) K gains; nor the wall-clock exchange rate on real
embeddings, nor anything the acquisition loop does with a different
threshold. That is this study.

**Why the archived runs cannot answer it.** The #3115 run (SLURM 539811) does
carry `folds_k{K}_anchored` rows, but it predates PR #3269: every trajectory
was crop-seeded, which voids exactly the early-trajectory regime the
small-labelset hypothesis lives in (see the seeding caveat banner on both
fold reports). Its deep windows remain usable as a sanity cross-check and
nothing more.

## What is being measured

Two curves over the fold count K, under the **shipped** threshold path and a
text-sort opening, banded by votes:

1. **Benefit**: paired Δcost of the shipped rule at K vs at the production
   K=2.
2. **Price**: the calibration wall clock at K (`fold_seconds`, measured
   per-fold inside the run) against the step's total wall clock — the
   user-facing "how much longer does each retrain make me wait".

And from their ratio, whether any fixed or adaptive K clears the
pre-registered ship rules below.

## Design

### Stage A — the nested screen (one run per geometry)

The fold-count screen is nearly free and exact, because the folds are nested:
`compute_fold_orderings` draws each fold as an independent stratified split
off one `RandomState(42)` stream at a size that does not depend on the count,
so the K folds a live `calibrate_count=K` run would train are byte-for-byte
the first K of the Kmax folds trained here. One run at Kmax therefore
measures every K's regret *and* every K's wall clock, paired within the step
(`CALIB_FOLD_COUNTS`; rows `folds_k{K}_anchored` carry production's rule per
K since #3116, with `fold_seconds` beside them).

| axis | value | why |
|---|---|---|
| live count | `calibrate_count = 2` (production; the trajectory users get) | every counterfactual K is scored on the same votes |
| screen grid | `CALIB_FOLD_COUNTS = 1,2,3,4,6,8` | Kmax=8 caps the extra per-step cost at ~4× the fold fits; #2897 already showed K ≥ 12 is unaffordable on the interactive path (`cal_share` 0.885 at 16) |
| dataset | `vg_scale_any` | 12 hand-checked classes × 300 positives against one shared negative pool → prevalence identical in every cell |
| geometries | `siglip/whole_image`, `dinov3_patch/whole_image`, `dinov3_patch/max_patch` | the middle corner separates voting mode from embedder (#3258's confound, fixed here as in #3287) |
| opening | SigLIP text sort in every cell (`siglip+dinov3_patch` pair, `CALIB_REQUIRE_OPENING=text`) | post-#3269; the crop-seeded archives are void exactly where this study reads |
| seeds | 4 | matches #3287; enough for banded paired SEs over cells |
| steps | 150 | the benefit is predicted to *decay* inside the horizon; the decay has to be in the window |
| cell order | `seed` | a truncated array loses seeds uniformly, not whole categories (#3287's argument) |
| head / path | defaults: linear-SVM head, safe thresholds, per-mode blend schedule | the anchored arms need the per-fold haystacks that `_safe_threshold_for_step` supplies |

The primary read is `folds_k{K}_anchored` vs `folds_k2_anchored`, paired
within the step, banded on votes: **1–25, 26–60, 61–100, 101–150** (the #3287
bands). The pooled-rule rows (`folds_k{K}_xcal`) are kept as the replication
of #2897's monotone worsening under text seeding — a secondary check that the
old result reproduces and stays attributable to the combine rule, not a
decision input.

### Stage B — live A/B, gated

The shipped threshold **drives acquisition** (measured in #3115: 100%
`fold_anchored` provenance on every `app_trained` step), so a screen that
holds the trajectory fixed cannot see the votes a different K would have
collected. This is the same reason #2897 ran its A/B and #3287 ran full runs
per arm — and the opposite of the #3115 conformal-combine case, where the
swept rule never reached acquisition and an A/B would have been a null by
construction.

**Gate (pre-registered):** Stage B is booked only if Stage A shows some K
clearing the benefit margin in some band *within* the cost ceiling (rules
below). If the screen comes back flat everywhere, the answer is "folds buy
nothing worth their price even where they are cheapest", and that is the
finding — the report ships without Stage B.

Arms, three full runs (plus the Stage A K=2 run as the control trajectory):

- `k_best` — the smallest fixed K that cleared the gate, live
  (`CALIB_CALIBRATE_COUNT=K`, the `launch_folds_2897_ab.sh` pattern).
- `k_adaptive` — the schedule the issue proposes: more folds while the
  labelset is small, decaying to production's 2. Family pre-registered as
  step schedules `K(n_votes) = K_early while n_votes < N_cut, else 2`, with
  `K_early ∈ {4, 6, 8}` and `N_cut ∈ {25, 60}`; Stage A's banded table picks
  the single (K_early, N_cut) submitted, mechanically: the largest banded
  benefit whose band-wise cost ratio passes the ceiling.
- The harness change this needs — `calibrate_count` resolved per step from a
  schedule rather than a constant — lands behind an eval-only knob
  (`CALIB_FOLD_COUNT_SCHEDULE`), leaving `CALIBRATE_COUNT` untouched for
  every other study. If the schedule ever ships in the app, the constant
  becomes `production_fold_count_for(n_votes)` beside
  `production_split_for`, and `scripts/check-eval-app-sync.py` gains the
  mirror — same discipline as #3287's closing note.

## Metrics

`cost` (headline) and `regret_honest` per band; `fold_seconds` (per-K
calibration wall clock, measured inside the run so all K share one machine —
**ratios are load-bearing, not absolute seconds**); the step's total wall
clock, from which the calibration share at K is
`cal_share(K) = fold_seconds(K) / (step_seconds − fold_seconds(2) + fold_seconds(K))`;
`sd(threshold)` across seeds per band (the variance-reduction mechanism,
observed directly); `n_folds_used` (degenerate-fold exposure).

## Pre-registered decision rules

A fold count K (fixed, or a schedule's early phase within its band) is a
**ship candidate** when all three hold on `cost`:

1. **Benefit**: it beats K=2 by more than both `MARGIN = 0.005` and 2 SE in
   at least one band, paired within the step (Stage A) / within
   `(category, seed, geometry, band)` (Stage B, which shares no votes across
   arms).
2. **No harm**: it is not worse than K=2 by more than 0.01 (`HARM_TOLERANCE`,
   the PR #2891 margin) in **any** band. Pointwise on purpose: a K can win
   early and lose late, and a schedule exists precisely so it does not have
   to pay the late price — but a *fixed* K that loses a band is out.
3. **Affordable, in user-facing terms**: within the bands where the K
   applies, median per-step wall clock at K stays ≤ **1.5×** the K=2 arm's.
   This replaces #2897's 4×-cost ceiling, which was pre-registered without
   reference to `cal_share` and admitted an arm that spent 76% of every step
   calibrating — the lesson recorded in that report's Limitations. The ratio
   is banded because the adaptive schedule concentrates its extra folds
   exactly where fits are cheapest; a global ratio would hide that.

Standard errors are bootstrapped over **cells**, never steps (#3287's rule:
consecutive steps of one trajectory share a model). Stage A contrasts are
paired within the step; Stage B arms are paired within
`(dataset, category, seed, geometry)` only.

The decomposition trap stays closed: no `rule_inefficiency` /
`calibration_shift` claim is made at any K — the #2897 H4 section documents
why that split is algebra, not evidence, on a K-swept calibration set. Levels
are read off `cost` / `regret_honest` only.

Outcomes, all of which are results:

- **"2 is fine everywhere, here is the price sheet."** The bench predicts
  exactly this for single-vector geometries (the shipped rule was K-invariant
  there even at n=8), and it is worth having measured: `calibrate_count=2` is
  currently defended by a study that swept the wrong rule.
- **A fixed K for region voting** — plausible given #2897's region table
  (benefit through K=6 at 2.68× cost); the new ceiling will price it honestly.
- **The adaptive schedule ships** — the issue's own proposal, and the only
  arm that can win the benefit without paying the deep-regime price.

## Running it

Reuses the #3115/#3287 infrastructure unchanged except for the Stage B
schedule knob: `scripts/experiments/calibration/run_cells.py` already reads
`CALIB_CALIBRATE_COUNT` and `CALIB_FOLD_COUNTS`;
`launch_calfrac_3287.sh` is the template for full-runs-per-arm with a shared
prepare, `launch_folds_2897_ab.sh` for the live-count arms. Follow
`scripts/experiments/preflight.sh` and the grid-experiments skill: one
`CALIB_EXP` per stage, size from a real cell (a Kmax=8 region cell is the
critical path — the screen run's fold fits cost ~4× a production step's, so
no prior grid's seconds transfer), declare `CALIB_REQUIRE_OPENING=text`.

Analysis: `analyze_folds_2897.py` (fold-count axis; already emits the
`folds_k{K}_*` families), extended with the banded cost-ratio table; figures
via `curves.py` (`quality_vs_clicks`) and the `viewer.py` output, both
mandatory per the skill. The report lands in this directory as `REPORT.md`.
