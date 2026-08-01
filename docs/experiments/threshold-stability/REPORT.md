# Threshold-stability study (#2790) — report

The labeling loop calibrates its decision threshold with the split-conformal
inclusion rule (#2784) — the same rule the shipped app runs. This study asks why
the threshold still makes the large single-step cost jumps #2790 reported, and what
reduces them.

**BLUF.** The large cross-seed variance is **threshold placement, not ranking** (the
across-seed cost sd is ≈ 4× the oracle-cost floor, at every setting). Among the knobs
tested, the **fold count** is the one that reduces it: raising `calibrate_count` from
the app default of 2 to 8 cuts the cross-seed cost sd and the spike rate by ~15–25%.
Median smoothing (`med3`) does **not** help; the rank-transfer variant is inert in the
live loop. A residual remains (cost sd still ≈ 3× the oracle floor at `k8`), pointing
at the fold→final-model score-scale mismatch (S3). The spike drill-down (below) shows
the residual jumps are almost entirely the **sparse-positive** case: a Bad vote on a
boundary item when the detector has only ~3 positives to calibrate on.

## Setup

- **Loop:** the realistic Autopilot labeling loop (`scripts/sod/sweep.py`,
  `vtscore/eval/region_curve.py`), whole-image / box-pool path, MLP head, conformal
  calibration.
- **Config (the #2790 repro):** COCO, SigLIP 2 (`siglip2`), `whole`,
  `--max-labels 60`, `--neg-multiple 100` (prevalence ≈ 1/101),
  `--min-box-frac 0.03`, inclusion 0, `--safe-thresholds` on.
- **Classes (5):** stop sign, traffic light, fire hydrant, parking meter, bus.
  **Seeds:** 10.
- **Arms** (all conformal; named by what they vary): **`k2`** = `calibrate_count`
  2 (the app default, the baseline), **`k8`** = 8 folds, **`k2-med3`** = k2 + median-
  of-last-3 threshold smoothing, **`rank-transfer`** = k2 with the cut re-expressed as
  a rank on the final model's scores (an S3 probe, only active in Stage-A replay).
- **Metrics** (per arm, over `t ≥ 20`, from each arm's `results.jsonl` per-`(seed, t)`
  `cost` / `oracle_cost` / `threshold`): across-seed **`cost_sd`** vs **`oracle_sd`**
  (the ranking-variance floor — the gap is threshold noise); **`spike_rate`** (steps
  with `|Δcost| > 0.1`); **`mean_regret`** = `cost − oracle_cost` over `t ∈ [40, 60]`.
  `sd_threshold` (across-seed threshold spread) is reported for continuity but is a
  cross-model quantity (each seed retrains a different MLP), so read `cost_sd` as the
  sound version.

## Stage B results (live loop)

| arm | cost_sd | oracle_sd | spike_rate | mean_regret | sd_Δthreshold | (sd_threshold) |
|---|---|---|---|---|---|---|
| **`k2`** (default) | 0.1412 | 0.0428 | 0.1639 | 0.1533 | 0.0543 | 0.1273 |
| **`k8`** | **0.1222** | 0.0380 | **0.1298** | 0.1578 | **0.0350** | 0.0993 |
| `k2-med3` | 0.1406 | 0.0435 | 0.1902 | 0.1503 | 0.0527 | 0.1309 |
| `rank-transfer` | 0.1412 | 0.0428 | 0.1639 | 0.1533 | 0.0543 | 0.1273 |

## Findings

- **F1 — The instability is the threshold, not the ranking.** Across every arm the
  cross-seed `cost_sd` (0.122–0.141) is ≈ 4× the oracle floor `oracle_sd`
  (0.038–0.044). The sort is stable across seeds; the variance is where the cut lands.

- **F2 — More folds is the lever that helps.** `k8` cuts `cost_sd` 0.141→0.122
  (~14%), `spike_rate` 0.164→0.130 (~20%), and `sd_Δthreshold` 0.054→0.035. Averaging
  the cut over 8 calibration folds instead of 2 is the single change that most reduces
  the jumps — at 4× the fold-fit cost per retrain.

- **F3 — `med3` does not help.** Median-smoothing the blended threshold raised the
  spike rate (0.164→0.190) and left `cost_sd` unchanged. Symmetric smoothing damps the
  corrective moves as much as the bad ones, so it is not the cheap win it looked like.

- **F4 — `rank-transfer` is inert in the live loop.** Its rank-remap needs the final
  model's score pool, which only exists in Stage-A replay, so in the live loop it is
  identical to `k2`. It is retained as a Stage-A diagnostic for S3.

## Verdict

No arm removes the instability: even `k8` leaves `cost_sd` (0.122) ≈ 3× the oracle
floor (0.038). The residual is larger than any tested knob closes, implicating the
fold→final-model score-scale mismatch (S3) — the cut is chosen on half-data fold
models and applied to the full-data final model. The indicated follow-up is a
threshold calibrated on the final model's own scores (leave-one-out / refit-free
conformal).

**Practical recommendation:** raise the app's `calibrate_count` from 2 to 8 (arm
`k8`) — it is the one setting that measurably reduces both the cross-seed cost
variance and the spike rate, for 4× the (cheap) fold-fit work per retrain. `med3` and
rank-transfer are not worth adopting.

## Spike attribution — what vote causes a spike

_Pending the full 5-class conformal trace run (job 437015). Preliminary read on
stop-sign (10 seeds, 49 up-spikes) below; the section is finalized when the run
completes._

Drill-down attributes every up-spike (`Δcost > 0.1`) to the vote added that step
(`scripts/experiments/threshold_stability/spike_analysis.py`), stated on
model-independent metrics (test-set `cost`/`fnr`/`fpr`; "the cut is bad" = far from
*that step's own* oracle cut — never a raw threshold compared across the retrained
models).

Preliminary pattern (stop-sign): a spike is a **Bad vote (92%) on a `hard`-selected
boundary item (86%)** while positives are **sparse — 94%, median n_good = 3**. The
conformal cut's gap-midpoint already caps the cut at that step's lowest calibration
positive, so the catastrophic FNR→1.0 runaways are down (~20% of spikes, ~half the
pre-conformal rate) and ~47% of spikes now recover within 2 steps. What remains is the
sparse-positive regime: with only ~3 positives, even the lowest-positive anchor is
itself unstable, so a boundary Bad vote still moves the operating point sharply.

**Blockable (all within-model / vote-count based, no cross-step threshold compare):**

1. **Defer trusting the trained cut while positives are sparse** (`n_good ≤ ~6`;
   median at spike = 3): stay on the cold-start / GMM (or text) sort, or widen the
   cut's positive-coverage floor, until enough positives exist to pin it. This targets
   the dominant residual directly.
2. **Acquisition guard:** `hard` selection surfaces each model's boundary items, which
   is what re-triggers the jump; bias away from pure boundary sampling in the sparse
   regime.
3. The conformal gap-midpoint coverage floor (already shipped) is what keeps this from
   being worse — it caps each retrained model's cut against its own positives, which is
   why the runaways are the minority and most spikes now recover.

## Caveats / scope

- **Stage B (live loop) only** for the arm comparison. The Stage-A frozen-trace replay
  (the paired split-noise vs fit-noise decomposition, and the only place
  `rank-transfer` is active) is built (`scripts/sod/replay_thresholds.py`, unit-tested)
  and unblocked now that `--no-trace-images` keeps trace output tiny; it has not been
  run at scale.
- Whole-image path only (the #2790 case). Detector training and the region-voting/hac
  path use the same conformal rule after the #2784 backport, but were not swept here.

## Artifacts

Grid run under `/exp/sgreenberg/threshold-stability/results/` (`summary.json`,
`agg/by_arm.csv`, per-cell `cells/<class>/arm_*/results.jsonl`) and
`/exp/sgreenberg/threshold-stability/traces_conformal/` (spike traces). Harness on
branch `claude/threshold-stability-2790` (PRs #2795 merged, #2796).
