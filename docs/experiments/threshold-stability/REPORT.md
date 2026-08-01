# Threshold-stability study (#2790) — report

**BLUF.** On the COCO × SigLIP 2 × `whole` Autopilot labeling loop, the large
cross-seed variance #2790 reported is **threshold-placement noise, not ranking
noise** (across-seed cost sd ≈ 4× the oracle floor, every arm). The biggest lever
on that noise is the **fold count** (`calibrate_count` 2→8 roughly halves the
threshold jitter), not the calibration *rule*. Switching the sweep's historical
min-cost `argmin` rule to production Autopilot's split-conformal rule barely changes
threshold sd at matched fold count — but it **cuts calibration regret ~37%** (the
cut is placed much better). No arm clears the pre-registered adoption bar; the
best practical arm is **`conformal-k8`** (lowest spike rate and near-lowest
threshold sd, with conformal's low regret). The residual points at the
fold→final-model score-scale mismatch (S3), which needs a follow-up.

## Setup

- **Loop:** the realistic Autopilot labeling loop (`scripts/sod/sweep.py`,
  `vtscore/eval/region_curve.py`), whole-image / box-pool path, MLP head.
- **Config (the #2790 repro):** COCO, SigLIP 2 (`siglip2`), `whole`,
  `--max-labels 60`, `--neg-multiple 100` (prevalence ≈ 1/101),
  `--min-box-frac 0.03`, inclusion 0, `--safe-thresholds` on.
- **Classes (5):** stop sign, traffic light, fire hydrant, parking meter, bus.
- **Seeds:** 10. **Cells:** 6 arms × 5 classes = 30.
- **Metrics** (per arm, over the post-GMM-ramp window `t ≥ 20`), derived from each
  arm's `results.jsonl` per-`(seed, t)` `threshold` / `cost` / `oracle_cost`:
  across-seed `sd_threshold`; within-seed step-to-step `sd_dthreshold`; `spike_rate`
  (fraction of steps with `|Δcost| > 0.1`); across-seed `cost_sd` vs `oracle_sd`
  (the ranking-variance floor); `mean_regret` = `cost − oracle_cost` over `t ∈ [40, 60]`.

### Reframe (plan vs. reality — the load-bearing finding)

The pre-registered plan assumed the sweep's region-voting path "already uses the
production conformal rule." It does not: `evaluation-framework` predates the #2784
conformal cut, so **both** the whole and region-voting paths reduced to min-cost
`argmin`. Per the owner's steering ("match a simulation of Autopilot over the plan"),
`argmin` is therefore the *infidelity* and `conformal` (ported verbatim from `dev`'s
`thresholds.py`) is what production Autopilot actually runs. The study measures
whether making the simulation faithful fixes the jumps.

## Results

| arm | sd_threshold | sd_Δthreshold | spike_rate | cost_sd | oracle_sd | mean_regret |
|---|---|---|---|---|---|---|
| `argmin-k2` (baseline) | 0.1583 | 0.0676 | 0.1732 | 0.1471 | 0.0358 | 0.2455 |
| `argmin-k8` | **0.0871** | 0.0390 | 0.1337 | 0.1241 | 0.0348 | 0.2368 |
| `conformal-k2` | 0.1273 | 0.0543 | 0.1639 | 0.1412 | 0.0428 | 0.1533 |
| `conformal-k8` | 0.0993 | **0.0350** | **0.1298** | **0.1222** | 0.0380 | 0.1578 |
| `conformal-k2-med3` | 0.1309 | 0.0527 | 0.1902 | 0.1406 | 0.0435 | 0.1503 |
| `rank-transfer-k2` | 0.1273 | 0.0543 | 0.1639 | 0.1412 | 0.0428 | 0.1533 |

## Findings

- **F1 — The instability is the threshold, confirmed (plan H5).** Across every arm
  the cross-seed `cost_sd` (0.122–0.147) is ~4× the oracle floor `oracle_sd`
  (0.035–0.044). The sort is stable; the variance is where the cut lands. This is
  exactly the effect #2790 observed, now measured across 5 classes × 10 seeds.

- **F2 — Fold count is the dominant stability lever, not the rule.** `k8` roughly
  halves threshold jitter (`argmin` sd_threshold 0.158→0.087; `conformal`
  0.127→0.099; `sd_Δthreshold` and `spike_rate` fall in step). At matched fold
  count, `argmin`→`conformal` moves threshold sd only modestly (and at `k8`,
  slightly the wrong way, 0.087→0.099). The knob that most reduces the jumps is
  averaging over more calibration folds.

- **F3 — Conformal's win is accuracy, not stability (plan S1, refined).**
  `conformal` cuts `mean_regret` ~37% vs `argmin` (0.245→0.153 at k2; 0.237→0.158
  at k8) and holds it — the fidelity-correct rule places a materially better cut.
  So adopting the rule the app actually runs is worth it *for accuracy*; it is not,
  on its own, the fix for the *jumpiness* (that's F2).

- **F4 — med3 did not help; rank-transfer is inconclusive here.**
  `conformal-k2-med3` did not reduce spikes (spike_rate 0.164→0.190) — temporal
  median smoothing of the *blended* threshold is not the cheap win H3 hoped for.
  `rank-transfer-k2` is byte-identical to `conformal-k2` because its rank-remap
  needs the final model's score pool, which is only available in Stage-A replay,
  not the live loop — so this arm carries no signal in Stage B (a wiring
  limitation, documented).

## Verdict (against the pre-registered decision rules)

- **Rule 2 (adopt the cheapest arm cutting spike incidence ≥80% and across-seed
  cost sd ≥50% vs `argmin-k2`, without worsening regret by >0.01): NOT MET by any
  arm.** Best spike reduction ≈ 25% (`conformal-k8`); best `sd_threshold` reduction
  ≈ 45% (`argmin-k8`); best `cost_sd` reduction ≈ 17% (`conformal-k8`). None reaches
  the bar.
- **⇒ Rule 4 fires:** the residual is larger than any single knob (rule or fold
  count) removes, implicating the fold→final-model score-scale mismatch (S3). The
  indicated follow-up is a threshold calibrated on the *final* model's own scores
  (leave-one-out / refit-free conformal), not on half-data fold models.
- **Rule 3 (med3 as a production candidate): dropped** — F4 shows it does not help.

**Practical recommendation:** `conformal-k8`. It is the best available combination —
lowest `spike_rate` (0.130), near-lowest `sd_threshold` (0.099) and `sd_Δthreshold`
(0.035), lowest `cost_sd` (0.122), and conformal's low regret (0.158). It pairs the
fold-count stability lever (F2) with the conformal accuracy win (F3). It does not
clear the strict bar, so it is a "ship the better default while the S3 follow-up is
scoped," not a declared solution.

## Spike attribution — what vote causes a spike

Drill-down on the baseline `argmin-k2` arm (`--labeling-trace --no-trace-images`,
5 classes × 10 seeds), attributing every up-spike (`Δcost > 0.1`) to the vote added
that step (`scripts/experiments/threshold_stability/spike_analysis.py`). **278 spikes.**

**Metric note (important).** The MLP is retrained from scratch every vote, so
`MLP_t` and `MLP_{t+1}` have *different* score scales — a raw threshold value is not
comparable across steps, and "the threshold moved up" is not a well-defined quantity
(it conflates the model changing with the calibration changing). Everything below is
stated in **model-independent** terms: `cost` / `fnr` / `fpr` are measured on a fixed
held-out **test set** (so `Δcost`, `Δfnr` are comparable step-to-step), and "the cut
is bad" means **far from `MLP_{t+1}`'s own oracle cut** — never a comparison of one
model's threshold to another's. (Stage B established this is possible: `oracle_sd ≈
¼ · cost_sd`, so each step's ranking *is* cuttable into a low-cost operating point;
the spikes are the calibration missing it, not the ranking failing.)

| signal (all test-set / within-model) | share |
|---|---|
| culprit is a **Bad** vote | 87% |
| Bad **and** `hard`-selected (surfaced at the boundary) | 79% |
| `hard`-selected (any label) | 91% |
| **FNR**-driven (Δfnr > Δfpr on the test set) | 76% |
| **runaway** (Δfnr > 0.2 — the operating point rejects many test positives) | 37% |
| **sparse positives** at spike (`n_good ≤ 6`; median `n_good` = **4**) | 67% |
| **narrow** (test cost recovers within 2 steps) | 26% (54% still elevated after 5) |

(A raw-threshold-delta was also logged — 81% "up" — but is **not** reported as a
finding: comparing `MLP_t`'s cut to `MLP_{t+1}`'s cut is comparing two models'
calibrations, not a moving quantity. The FNR operating-point move above is the sound
version of the same effect.)

**Mechanism (worked example, stop-sign seed 4), stated on the test set.** At `t≈24`
the operating point is healthy (test FNR 0.12, cost 0.14, 5 good / 19 bad votes). At
`t28` a **Bad** vote on a boundary item (id 140556, surfaced at that model's cut) is
added; `MLP_{28}` is retrained on ~4 positives, and its argmin cut lands far above
`MLP_{28}`'s own test positives — **test FNR 0.12 → 0.58, cost → 0.58**. It does
**not** snap back: `hard`-selection keeps surfacing each new model's boundary items,
and over the next four votes the successive models' cuts keep excluding their own
positives — **test FNR runs to 1.0** (the detector rejects everything). Each step is
a *fresh* model + cut; what compounds is the *acquisition* (boundary selection) plus
the *sparse-positive calibration*, not a single threshold drifting.

**Root pattern:** a Bad vote on a `hard`-selected boundary item, in the
**sparse-positive regime (~4 good votes)**, makes the retrained model's argmin cut
land far from its own oracle — test FNR spikes. A secondary ~24% are the mirror
(Bad vote → the new model's cut admits a flood of test false positives). Both are the
same instability — a handful of calibration positives can't pin *any* single model's
cut, so the argmin lands badly in either direction. The "narrow" intuition holds for
the FPR blips but **not** the FNR runaways (37%), which compound via acquisition.

**Blockable (ranked):**

1. **Within-model positive-coverage floor (proactive, already built): the #2784
   conformal gap-midpoint rule** forbids each retrained model's cut from exceeding
   *its own* lowest calibration positive — a purely within-model constraint (no
   cross-step threshold comparison), so it structurally blocks the "cut lands above
   its own positives" runaway (the damaging 37%). This is why `conformal` cut regret
   −37% in Stage B. Make the `whole`/box-pool loop use it (and confirm the app does).
   Note this is the sound version of "don't let the cut jump up" — it caps the cut
   against the *current* model's positives, not against the previous model's cut.
2. **Defer trusting the trained cut while positives are sparse** (`n_good ≤ ~6`;
   median at spike = 4): 41% of spikes are early and the enabling condition is too
   few positives to pin *any* model's cut — stay on the cold-start / GMM (or text)
   sort longer, or widen the cut's coverage floor until enough positives exist. This
   is a vote-count guard, fully model-independent.
3. **Acquisition guard:** `hard` selection surfaces each model's boundary items,
   which is what compounds the runaway across steps (item 91% `hard`-selected). In
   the sparse-positive regime, biasing away from pure boundary sampling breaks the
   compounding. (This targets the *acquisition* half; it is orthogonal to the
   calibration cap in #1.)

Explicitly **not** recommended: a per-vote clamp on the raw threshold *change*
between steps — that presumes `threshold_t` and `threshold_{t+1}` are on one scale,
but the models are independently retrained, so the delta isn't a meaningful quantity.
The within-model coverage floor (#1) achieves the intended effect soundly.

## Caveats / scope

- **Stage B only.** The Stage-A frozen-trace replay (the paired split-noise vs
  fit-noise decomposition) was dropped from this run: `--labeling-trace` renders
  ~2 PNGs/step (~11k files, multiple GB) and filled the 50 GB `/exp` volume. Stage
  B needs none of it — `results.jsonl` carries the per-step threshold/cost. Stage A
  needs a **PNG-free trace mode** (write `trace.json` only); the replay tool
  (`scripts/sod/replay_thresholds.py`) is built and unit-tested, waiting on that.
- **Whole-image path only** (the #2790 case). The region-voting/`train_rv_head`
  grouped path still calibrates via `argmin`; porting conformal there (grouped
  stratified folds) is separate follow-up work.
- **`rank-transfer` not evaluated** in the live loop (F4).

## Artifacts

Grid run under `/exp/sgreenberg/threshold-stability/results/`: `REPORT.md` (this,
auto-generated), `summary.json`, `agg/by_arm.csv`, and per-cell
`cells/<class>/arm_<name>/results.jsonl`. Harness on branch
`claude/threshold-stability-2790` (PR #2795). Job chain: warm `436641` (GPU embed) →
cells `436934_[0-4]` (CPU, ~30 min/cell) → analyze `436935`.
