# Calibration fold-count experiment — is 2 still the right number of crosses?

**Status:** the harness, launchers and analyzer are written; the GRID run is
still owed (issue #2897, part 2). Nothing here changes production: the fold
count stays at 2 until the run says otherwise.

## Background

Production cross-calibrates its threshold over `calibrate_count = 2` folds.
That constant has not been re-examined since the calibration stack was rebuilt
around it — cross-partially-labelled GMM, fold-anchored mixtures, blend
schedules, conformal inclusion, the acquisition offset. Every one of those
changes altered how much the *cross-calibration term* contributes to the
shipped threshold, so the number of folds feeding it is worth re-pricing.

Only one prior run touched the fold count: the #2861 addendum
(`analyze_folds.py`, `launch_folds_2861.sh`) ran K=4 against K=2, but it did so
to unlock the `qmean`/`qmedian` combine question (byte-identical at two folds),
on VG × siglip only, as an unpaired run-level A/B whose contrast was each run's
own Δ-vs-`xcal_only`. It could not measure cost, could not band the effect, and
did not cover binary voting. Nothing in it answers "how many folds should we
ship".

### The mechanism the design turns on

The folds are **not a K-fold partition**. `compute_fold_orderings` draws each
fold as an independent stratified train/calibrate split at a *fixed*
`calibration_fraction`, and `threshold_from_fold_orderings` pools every fold's
held-out scores into one conformal quantile. So raising K:

- does **not** shrink each holdout (the usual K-fold data/variance trade-off is
  absent) — it averages more draws of the same-sized holdout;
- grows the pooled calibration set linearly, which is what sharpens the
  quantile;
- costs one extra fold fit per step, so wall clock is linear in K;
- should therefore show **diminishing returns** — the estimator's variance
  falls like 1/K at best, against a bias term (calibration→test shift) that K
  cannot touch at all.

Two predictions follow, and the run is worth doing precisely because they are
falsifiable: regret(K) saturates at some knee, and whatever benefit exists lands
on the `rule_inefficiency` term rather than the `calibration_shift` term.

### Why the screen is exact, not approximate

The same code path makes the study cheap. Folds are drawn sequentially from one
`RandomState(42)`, at a per-fold split size that does not depend on the count,
and `train_model` seeds its own generator per call — so the folds are **nested**:
the K folds a live `calibrate_count=K` run trains are byte-for-byte the *first
K* of the Kmax folds one instrumented run trains.

That buys three things at once:

- **every K's threshold from one run**, exactly equal to what that K's own run
  would compute for the same votes;
- **pairing at the step** — each K re-cuts the same votes, the same final model
  and the same held-out test scores — which removes the trajectory noise that
  dominated the #2861 unpaired comparison;
- **a built-in control**: the arm at `K == calibrate_count` must reproduce the
  step's own conformal cut. It is asserted in
  `tests_lib/detectors/test_fold_count_variant_rows.py`, along with the
  requirement that training the extra folds leaves the live trajectory
  byte-identical — the screen must not have an observer effect.

What the screen **cannot** see is acquisition feedback: the threshold is the
rank position Autopilot's Hard pick samples around, so a run genuinely living
at a different K collects different votes from step one. That is the A/B stage,
and it is not optional — it is the only thing that converts the screen's
per-step deltas into a claim about what a user would experience.

## Arms

Emitted per trainable step, per fold count K (`vtscore/eval/voting_iterations.py`,
`_fold_count_variant_rows`):

| Arm | What it is |
|---|---|
| `folds_k{K}_xcal` | the raw cross-calibration cut — the term K is a knob on |
| `folds_k{K}_blend` | that cut after the shipped safe-threshold mix-in — what a user gets |

Both are needed. K acts on the cross-calibration term, but the blend weight
(a function of the vote counts only, hence identical across K) determines how
much of any gain survives into the shipped threshold. A study that reported
only the raw arm could recommend a fold count whose benefit the blend erases.

Pre-registered grid: `K ∈ {1, 2, 3, 4, 6, 8, 12, 16}`, with K=1 (no crossing at
all) as the lower anchor and 16 well past any plausible knee, so saturation is
demonstrated rather than assumed. Cost is set by the grid's **maximum**, not its
length.

Both voting modes, because the calibrators differ and #2897 asks for both:
`visual_genome_m` region voting on the bag-aware (grouped) calibrator, where a
"calibration point" is a whole bag and the effective calibration set is much
smaller; `caltech101_m` binary voting on the row-wise one.

## Pre-registered decision rules

Constants live in `analyze_folds_2897.py` so the analyzer applies them
mechanically:

- **MARGIN = 0.005** regret (`FPR + FNR` units, inclusion 0 → both weighted 1).
  Half a percentage point of combined error rate. Below it a fold-count change
  is not worth a user's retrain latency however clean its p-value, and two fold
  counts within it are the same answer.
- **COST_CEILING_X = 4.0**. Cross-calibration sits on the interactive retrain
  path — the user waits through it after every vote — so "more is better" is
  not adoptable without a bound. 4× production's calibration budget admits K up
  to 8, since cost is linear in K.
- **Deep regime = windows with upper edge ≥ 100 votes**, where a real search
  spends most of its votes. The cold start is reported separately rather than
  averaged in: it is exactly where the two regimes could disagree.

Verdicts:

- **H1 (benefit)** — some K beats K=2 by more than MARGIN in the deep regime,
  with a paired Wilcoxon over **cell** means (not step means: consecutive steps
  share nearly all their votes, so testing over steps would count one
  trajectory's luck hundreds of times) at p < 0.05.
- **H2 (cost)** — that K's measured calibration wall clock is within
  COST_CEILING_X of production's.
- **H3 (recommendation)** — the smallest K satisfying both, per voting mode.
  **If H1 fails, the answer is 2 and the study ships nothing.** A null result is
  a real outcome here and the analyzer reports it as `h3_kept_production`.
- **H4 (mechanism)** — the benefit lands on `rule_inefficiency`, not
  `calibration_shift`. If it lands on the shift term instead, the variance story
  is wrong and the recommendation should not be trusted even if H1 passes.

The report additionally carries an **exchange rate** (regret bought per extra
second) and a **knee** table, so a win that is statistically real but costs 4×
the wall clock for 0.001 regret reads as the bad trade it is.

## Open work

<!-- item-sep -->

- **Run the screen on the GRID** — `bash launch_folds_2897.sh`. Size it from one
  real cell before submitting the array (Kmax=16 is ~8× production's calibration
  work per step); fall back to `CALIB_FOLD_COUNTS=1,2,3,4,6,8` if 16 does not fit
  the window. Run `selftest_analyze_folds_2897.py` first — it plants a saturating
  benefit curve, a priced-out best K, and a null arm, and checks the analyzer
  recovers exactly those.

<!-- item-sep -->

- **Run the A/B once the screen names a K\*** — `bash launch_folds_2897_ab.sh 2 K*`,
  then re-run the analyzer with the arm dirs as arguments to get the
  `screen_agrees` table. Skipping this leaves the recommendation resting on a
  screen that is structurally blind to acquisition feedback. If the screen and
  the live runs disagree by more than MARGIN, the live number wins and the
  disagreement itself is the finding worth writing up.

<!-- item-sep -->

- **Fold the result back into production** — if H1–H3 recommend a K other than
  2, change `calibrate_count`'s default and re-check the interactive retrain
  latency on a real dataset, not just the harness's measured `fold_seconds`.
  If they recommend 2, record that in the issue and delete this plan; the
  constant will have been re-validated rather than merely inherited.

<!-- item-sep -->
