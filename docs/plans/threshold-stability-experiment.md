# Threshold-stability experiment — step-to-step threshold jumps in the labeling loop (issue #2790)

**Status:** Design (pre-registered). This is the Grid study for #2790 (MLP
threshold instability observed on the COCO stop-sign SigLIP 2 whole-image
sweep). The harness knobs + replay tool are not yet written; this document is
the spec they will be written to. Runs on the HLTCOE Grid; harness code lives
on the `evaluation-framework` branch (`scripts/sod/`, `vtscore/eval/`).

## What #2790 shows

Seed-colored labeling-loop curves on COCO "stop sign" × SigLIP 2 × `whole`
show violent single-step excursions: at seed 0, t=24, the trained threshold
drops 0.49 → 0.35 in one retrain, cost jumps 0.088 → 0.424 and F1 collapses
0.623 → 0.05, then both recover. Seed 1 shows repeated cost/F1 divergences
(one improves while the other degrades) between t=25 and t=40. The **oracle**
cost curve barely moves through all of this, so the *ranking* is stable — the
instability lives entirely in threshold placement. This is the suspected
source of the large cross-seed variance in the bigger sweeps.

## Diagnosis to verify (ranked suspects)

The issue's config takes the **whole-image MLP path** of the realistic loop
(`vtscore/eval/region_curve.py::_train_pool_head`, box-pool/whole branch),
whose threshold rule differs from production in a load-bearing way:

- **S1 — superseded min-cost rule (fidelity gap).** The whole path calls
  `vtscore/eval/xcal.py::cross_calibrated_threshold`: 2 folds, each an
  **unstratified** random half-split, per-fold threshold =
  `find_optimal_threshold` (the min-cost **argmin** over observed cuts on the
  ~t/2 held-out votes), then the **average of the 2 fold argmins**. Production
  replaced exactly this argmin with the pooled conformal quantile rule +
  gap-midpoint cut (issues #2693, #2784) because the argmin over a handful of
  held-out votes is a step function of the data — one changed vote moves the
  optimum to a distant cut. The sweep's **region-voting** path already uses the
  production rule (`cross_calibration_threshold_cached`); the `whole` path
  measures a rule production no longer runs. Prime suspect.
- **S2 — wholesale fold-membership churn.** `cross_calibrated_threshold`
  seeds `default_rng(seed)` fresh each retrain with the *loop* seed, but the
  vote count grows by one per step, so `rng.permutation(n)` vs
  `rng.permutation(n+1)` are uncorrelated partitions: at t and t+1 the fold
  models train on largely different halves. Averaged over only 2 folds, the
  threshold is a 2-sample mean of noisy argmins whose noise re-draws every
  step.
- **S3 — fold→final scale mismatch.** The cut is measured on the fold models'
  sigmoid scale (trained on ~half the votes) but applied to the final model
  (trained on all votes). Half-data MLPs saturate differently per step, so
  even a stable fold-side cut can wander on the final model's scale.
- **S4 — unstratified splits skip folds.** A single-class fold is skipped, so
  some steps average 1 argmin instead of 2 (or fall back to 0.5) — extra
  variance concentrated exactly at low/imbalanced vote counts.
- **S5 — selection feedback.** The threshold feeds `_select_hard` /
  `_select_new`, so a one-step threshold spike changes *which item gets
  labeled next*, compounding across the trajectory and inflating cross-seed
  variance beyond the per-step noise itself.
- **Non-bug: the cost/F1 opposite moves.** At `--neg-multiple 100`
  (prevalence ≈ 1/101), cost is rate-based (FPR+FNR) while F1 is
  precision-based: a downward threshold move trades a small FPR increase
  (cost barely worse, FNR better) for a precision collapse (F1 crushed).
  Opposite trends are the expected geometry of that metric pair under a
  moving threshold, not a separate defect — verified below by showing the
  divergence vanishes at oracle-stable thresholds.

Relation to prior work: `calibration-experiment.md` / #2781 measured
**static regret** (trained vs oracle cut at a fixed t) on the
`voting_iterations` harness. This study measures **temporal stability** —
step-to-step jump size, spike incidence, and how much of the cross-seed
variance is threshold noise vs ranking noise — on the `scripts/sod` realistic
loop, and A/B-tests candidate stabilizers.

## Design — two stages

### Stage A: frozen-trace replay (cheap, CPU, paired)

Record `--labeling-trace` runs of the baseline arm, then **replay** each
trace: at every step t, freeze the vote set and recompute the threshold
under each arm × R = 10 fold-split seeds × 10 trainer seeds (varied one at a
time). Because votes are frozen, arms and seeds are exactly paired, and the
variance decomposes:

- var across fold-split seeds at fixed t → split noise (S2/S4);
- var across trainer seeds at fixed (t, split) → fold-fit noise (part of S3);
- Δthreshold across adjacent t at fixed seeds → label-increment sensitivity;
- rule differences at identical inputs → S1.

Replay needs no selection feedback, so it cannot see S5 — that is Stage B's
job.

### Stage B: live-loop arms (Grid)

Run the full realistic Autopilot loop per arm, so selection feedback (S5) is
included. All arms share config with #2790's repro (COCO, SigLIP 2, `whole`,
`--max-labels 60`, `--neg-multiple 100`, `--min-box-frac 0.03`) except for
seeds (10, not 3 — variance is the measurand) and classes (5, below).

## Arms (both stages)

| Arm | Threshold rule | Isolates |
|---|---|---|
| `argmin-k2` | status quo: per-fold min-cost argmin, mean of 2 folds | baseline |
| `argmin-k8` | same rule, `calibrate_count = 8` | fold count vs rule |
| `conformal-k2` | production rule: stratified folds, pooled held-out scores, conformal quantile + gap-midpoint (dev `thresholds.py`) | S1 |
| `conformal-k8` | production rule, 8 folds | S1 + S2 |
| `conformal-k2-med3` | `conformal-k2`, then threshold_t = median of the last 3 raw thresholds | temporal hysteresis (cheapest possible production fix) |
| `rank-transfer-k2` | `conformal-k2`'s cut converted to a quantile of the pooled fold scores, then applied at that quantile of the **final model's** pool-score distribution | S3 |

All arms use the shipped fused threshold (there is no switch), so the population fit below 20
labels is common to every arm. (Fidelity footnote to record, not sweep: the
whole path blends against *train* scores where production blends against the
full scored set — only relevant at t < 20.)

## Metrics

Per step (new trace columns; also emitted without `--viz`):

- `threshold`, per-fold thresholds pre-average, `n_folds_used` /
  `n_folds_skipped`, `delta_threshold` (vs t−1), `threshold_percentile` (in
  the test score distribution), `calib_mode`.
- Existing `cost`/`fpr`/`fnr`/`f1`/`oracle_cost`/`oracle_f1`; add `regret =
  cost − oracle_cost` and a `spike` flag (`|Δcost| > 0.1` between adjacent t).

Per (arm, class, seed) cell aggregates:

- `sd(Δthreshold)` and spike count over t ≥ 20 (past the GMM ramp);
- mean regret over t ∈ [40, 60]; cost AUC over t ∈ [20, 60];
- and the headline: **across-seed sd of cost at each t**, alongside the same
  for oracle cost (the ranking-variance floor). Threshold noise is the gap
  between the two.

## Hypotheses (pre-registered, honest priors)

- **H1 (S1 dominates):** `conformal-k2` cuts `sd(Δthreshold)` and spike
  incidence ≥ 2× vs `argmin-k2` at equal fold count. Production already
  re-learned this lesson once; the harness's whole path just never picked it
  up.
- **H2 (fold count matters second):** `*-k8` halves the residual jumpiness of
  its `*-k2` counterpart; the effect is larger for `argmin` (mean of 8
  argmins) than for `conformal` (pooled quantiles already share scores).
- **H3 (hysteresis nearly finishes the job):** `conformal-k2-med3` removes
  ≥ 80% of remaining cost spikes at ≤ 0.01 mean-regret penalty — the ranking
  is stable (oracle says so), so smoothing the cut trades almost nothing.
- **H4 (scale mismatch is real but secondary):** `rank-transfer-k2` beats
  `conformal-k2` on spike incidence but by less than the S1 fix; if instead
  it wins big, S3 is the true driver and the production rule inherits the
  same weakness.
- **H5 (threshold noise explains the sweep variance):** across-seed cost sd
  under `argmin-k2` is ≥ 2× the oracle-cost sd at matched t; under the
  winning arm the two nearly coincide.
- **H6 (cost/F1 divergence is metric geometry):** opposite-sign cost/F1
  moves correlate with |Δthreshold| and disappear (rate < 5% of steps) under
  the winning arm; they never appear in oracle-threshold curves.

## Decision rules (pre-registered)

1. **Fix the fidelity gap regardless of outcome:** the `whole`/box-pool path
   switches to the production conformal rule (its own PR on
   `evaluation-framework`), unless the study shows conformal is *worse*
   there — in which case that contradiction is the headline finding and goes
   back to #2781's line of work.
2. Adopt as the harness default the cheapest arm that reduces spike incidence
   ≥ 80% and across-seed cost sd ≥ 50% vs `argmin-k2` without worsening mean
   regret at t ∈ [40, 60] by > 0.01 (paired Wilcoxon over (class, seed)
   cells, Holm-corrected).
3. If `conformal-k2-med3` wins, evaluate median-of-last-3 as a **production**
   candidate (the app retrains per vote too) — file a separate issue with the
   Stage B numbers; production adoption is not part of this study.
4. If no arm reaches the rule-2 bar, S3 is implicated beyond what
   `rank-transfer` fixes: escalate with a follow-up design that calibrates on
   the final model's own scores (e.g. leave-one-out or refit-free conformal),
   and record the negative result in the report.
5. H6 failing (divergences persist at stable thresholds) reopens the metric
   itself: cost weights vs F1 at prevalence 1/101 — report, don't patch.

## Sizing

- **Classes (5, COCO):** stop sign (the reported case), traffic light, fire
  hydrant, parking meter, bus — small-object-skewed with a spread of
  prevalence/difficulty, matching the sweep family the variance was observed
  in.
- **Stage B:** 6 arms × 5 classes × 10 seeds = **300 loops**, each ≤ 60
  retrains of a tiny MLP + its folds — CPU-cheap; one SLURM array task per
  (class, seed) with all arms inside (they share the embedding cache; the
  `whole`/SigLIP 2 cache slug from #2790's run in
  `docs/experiments/sod-sweep/cache` is reusable as-is).
- **Stage A:** replays the 5 × 10 baseline traces at 60 steps × 6 arms ×
  (10 fold seeds + 10 trainer seeds) ≈ 360k tiny fold fits total, batched in
  the same array; a few CPU-hours per class.
- No GPU needed anywhere (embeddings come from cache; MLPs are CPU-sized).
  Env knobs `THRSTAB_*` mirroring the existing `MAXPATCH_*`/`CALIB_*` sets.

## Deliverables

1. Harness knobs on `evaluation-framework`: `--threshold-rule
   {argmin,conformal,rank-transfer}`, `--threshold-smooth {none,med3}`,
   `--calibrate-count` (exists), plus the per-step threshold-diagnostic
   columns above — default-off / default-current so existing sweeps and
   caches are untouched.
2. Replay tool `scripts/sod/replay_thresholds.py` (Stage A) consuming
   recorded `--labeling-trace` output.
3. Runner stage scripts (queue + summarize) in the standard layout.
4. `docs/experiments/threshold-stability/REPORT.md` — BLUF, verdict per
   decision rule, figures: threshold + cost traces per seed (the #2790
   figure, per arm); sd(Δthreshold) and spike-incidence bars; across-seed
   cost sd vs oracle sd fan charts; Stage A variance decomposition; regret
   check.

## Open work

<!-- item-sep -->

- **Build the harness knobs + diagnostics** — threshold-rule / smoothing /
  trace columns on `evaluation-framework`, per Deliverables. Byte-identical
  default behavior.

<!-- item-sep -->

- **Build the replay tool** — `scripts/sod/replay_thresholds.py`, paired
  Stage A decomposition over recorded traces.

<!-- item-sep -->

- **Build the runner + queue scripts** — `THRSTAB_*` env knobs, CPU
  smoke-test on one (class, seed) before Grid submission.

<!-- item-sep -->

- **Run on the Grid + write the report** — owner-gated on Grid access;
  verdict flows through the pre-registered decision rules.

<!-- item-sep -->

## Known limitations (accepted for v1)

- Whole-image proposals only, one embedder (SigLIP 2), COCO only — the
  regime the variance was reported in. If the winning arm's effect is large,
  a confirmation cell on `hac` + region voting (which already uses the
  production rule, so only S2/S3/S5 apply there) is a follow-up, not v1.
- Stage A cannot see selection feedback (S5); Stage B sees it but cannot pair
  trajectories across arms after the first divergent pick. The two stages
  bracket the effect rather than measuring it exactly.
- The oracle uses test labels by definition; it is the ranking-variance
  floor, not an achievable rule.
- `rank-transfer` changes what the threshold *means* (a quantile, not a
  score); it is a diagnostic arm for S3 first, a production candidate only if
  it wins decisively.
