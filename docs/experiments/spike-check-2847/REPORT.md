# #2847 — do the MLP-era cost spikes survive today's stack?

**Issue #2847 (MatthewELucio) · base dev `f0519190` · branch
`claude/spike-check-2847` · GRID worktree
`/exp/sgreenberg/projects/vts-spike-2847` · experiment
`/exp/sgreenberg/spike-2847` · SLURM 472617 / 472631 / 472633 / 472635,
608/608 cells, 0 failures, 13m34s**

## BLUF

**Yes, we are better — and it is mostly the *threshold*, not the head.** On COCO
× SigLIP2 whole-image, the configuration #2847 ran (auto-sized MLP head, bare
cross-calibrated conformal cut) deep-spikes in **58.5%** of trajectories;
today's production stack (linear head, fold-anchored GMM cut at κ=0.3) does so
in **12.2%**, and once steps whose *ranking* was already hopeless are excluded,
in **5.4%** — a **7.7× reduction** in the phenomenon the issue is actually
about. The oracle stays smooth in every arm, which is what makes this a
threshold story rather than a ranking one. Independently corroborated: #2847's
**own command on its own branch** still deep-spikes in 35% of runs when given
enough seeds.

Three things to keep straight, all of which cut against a victory lap:

1. **The spikes are rarer, not gone,** and the survivors are *deeper*: median
   cost at a flagged step rises from 0.632 to 0.748. Two categories
   (`toothbrush` 75%, `scissors` 57%) barely improve at all.
2. **The endpoint did not improve.** Final cost is statistically unchanged
   (0.133 → 0.137, p=0.09). What we bought is steadiness along the way, not a
   better detector at t=100.
3. **Production finds half as many positives** (median 9 → 4 in 100 votes,
   p=1e-20). The fused threshold feeds Autopilot's Hard pick, so it changed
   acquisition — and it made it *less* productive. That is a real cost and it
   is not what this study set out to measure.

## What changed since #2847, and why the run had to be a 2×2

The issue's premise is right on both counts, and I confirmed both in code at
dev `f0519190`:

| | #2847 era | today (dev `f0519190`) |
|---|---|---|
| Detector head | auto-sized MLP | **linear** (logistic) — `LINEAR_HEAD`, everywhere (#2790/#2809) |
| Threshold | cross-calibrated conformal quantile | **fold-anchored GMM cut** — `FOLD_ANCHOR_WEIGHT=0.3`, `FOLD_ANCHOR_CUT_RULE="mid_tilt"`, `FOLD_ANCHOR_COMBINE="qmean"` (#2852/#2861/#2865), unconditional since #2863 |

Because *both* changed, measuring only today's stack answers "are the spikes
still there" but not "what fixed them" — and, worse, cannot tell a real fix from
a harness that never showed the phenomenon. So the run is a 2×2:

| arm | head | threshold | role |
|---|---|---|---|
| `A_mlp_xcal` | mlp | conformal only | the #2847 **configuration** — positive control |
| `B_mlp_fused` | mlp | fold-anchored | threshold change alone |
| `C_lin_xcal` | linear | conformal only | head change alone |
| `D_lin_fused` | linear | fold-anchored | **today's production** |

**Arm A is a counterfactual, not a historical reconstruction.** It is today's
`simulate_voting_iterations` with the old head and the fusion switched off — not
the code MatthewELucio ran. The causal claim it licenses is therefore
within-codebase: *holding the harness fixed, turning these two knobs back on is
what removes the spikes.* The historical route corroborates it independently —
see "The literal rerun" below, where #2847's own command on its own branch
spikes in 35% of runs.

### Grid

`coco_val` × `siglip2` × `whole_image` (binary voting), **19 categories × 8
seeds × 100 labels**, `cat` included at the issue's own seed count. Prepare was
reused from the #2841 mixin run (pickles + exemplar crops symlinked), so there
was no GPU stage; 608 single-threaded CPU cells drained in 13m34s.

Both knobs steer acquisition, so the four arms are four different trajectories.
Nothing here is step-paired: every comparison pairs at the `(category, seed)`
cell on one summary number per trajectory, and the statistical unit is a
trajectory, never a step.

### Spike rules (pre-registered)

- **Deep spike** — a step at `t ≥ 20` with `cost ≥ 0.25` **and**
  `cost − oracle_cost ≥ 0.20`. Calibrated to the issue's figure, whose blips are
  0.25 / 0.65 / 0.68 against an oracle near 0.05.
- **Cold start** (`t < 20`) is reported separately. Every arm humps there,
  including in the issue's own figure, and it is a different phenomenon (no
  model yet).
- **Ranking control** — the same max-jump statistic computed on `oracle_cost`.
  If the oracle jumps too, the step is not a threshold failure at all.

## Result

| arm | trajectories | deep-spike runs | deep-spike steps | median worst-step regret | p90 | median max jump (cost / oracle) | median final cost | median positives found |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `A_mlp_xcal` — mlp + conformal | 147 | **58.5%** | 9.30% | 0.239 | 0.573 | 0.135 / 0.028 | 0.133 | 9.0 |
| `B_mlp_fused` — mlp + fold-anchored | 147 | 26.5% | 1.76% | 0.141 | 0.260 | 0.076 / 0.036 | 0.151 | 4.0 |
| `C_lin_xcal` — linear + conformal | 147 | 46.3% | 6.65% | 0.184 | 0.416 | 0.097 / 0.028 | 0.136 | 8.0 |
| `D_lin_fused` — **production** | 147 | **12.2%** | 0.63% | 0.107 | 0.213 | 0.052 / 0.025 | 0.137 | 4.0 |

Paired against the control at the `(category, seed)` cell:

| arm | deep-spike incidence | only control | only arm | p exact | median Δ worst-step regret | p |
|---|---|---:|---:|---:|---:|---:|
| `B_mlp_fused` | 58.5% → 26.5% | 56 | 9 | 2.0e-9 | −0.071 | <1e-5 |
| `C_lin_xcal` | 58.5% → 46.3% | 26 | 8 | 2.9e-3 | −0.020 | 0.00026 |
| `D_lin_fused` | 58.5% → **12.2%** | 71 | 3 | 7.2e-18 | **−0.117** | <1e-5 |

**The threshold is the bigger lever.** Switching the threshold alone (A→B) takes
incidence from 58.5% to 26.5% — a factor of 0.45; switching the head alone (A→C)
only reaches 46.3%, a factor of 0.79. Together they reach 12.2%, which is
*better* than the two marginal factors would predict on their own (0.45 × 0.79 ×
58.5% = 21.0%, observed 12.2%). So the two changes are not redundant and neither
is masking the other — the fused threshold appears to be more robust
specifically on the runs where the linear head has already stabilised the
scores.

**The ranking never was the problem, and still isn't.** Max oracle jump is flat
across all four arms (0.025–0.036) while max cost jump falls 0.135 → 0.052.
Whatever the threshold was doing, it was not tracking a ranking that had
collapsed — which is exactly the observation #2847 makes when it notes the
oracle stays consistent.

### Separating true threshold blips from hopeless rankings

The deep rule fires on some steps where the *oracle* is also terrible — a
category where no threshold could have done well. Splitting on
`oracle_cost > 0.3`:

| arm | flagged steps | ranking-limited | genuine threshold blips | trajectories with a genuine blip |
|---|---:|---:|---:|---:|
| `A_mlp_xcal` | 1102 | 49.0% | 562 | 61 / 147 (**41.5%**) |
| `D_lin_fused` | 75 | 78.7% | **16** | 8 / 147 (**5.4%**) |

This is the sharpest form of the answer: **the #2847 phenomenon proper falls
from 41.5% of runs to 5.4%, a 7.7× reduction**, and 79% of what production has
left is a ranking problem wearing a threshold problem's clothes.

### On `cat` specifically — the issue's own class

![fig1](figures/fig1_cat_arms.png)

Deep-spike steps on `cat`: **8 → 1** (A → D). Per-run incidence 25% → 12%. The
production panel decays smoothly from the cold start to ~0.10 with the oracle at
~0.07–0.09; the control panel carries isolated blips to 0.76, 0.67, 0.50 and
0.40, which is the character of the figure in the issue.

![fig2](figures/fig2_cat_production.png)

![fig3](figures/fig3_incidence_and_magnitude.png)

![fig0](figures/fig0_literal_repro_n20.png)

*#2847's exact command on `evaluation-framework` HEAD at 20 seeds: 7 of 20 runs
deep-spike (rings), against an oracle whose warm-window median is 0.046. The
five seeds the issue's `--iterations 5` runs are all in the quiet majority.*

### Where it is still bad

| category | A | C | B | D |
|---|---:|---:|---:|---:|
| `toothbrush` | 88% | 100% | 75% | **75%** |
| `scissors` | 100% | 71% | 86% | **57%** |
| `clock` | 50% | 25% | 0% | 25% |
| `cell phone` | 100% | 57% | 0% | 14% |
| `microwave`, `oven`, `baseball bat` | 88–100% | 62–88% | 12–50% | 12% |
| `dining table`, `sink`, `refrigerator`, `sports ball`, `bed`, `train`, `giraffe`, `zebra`, `elephant` | 12–100% | 0–88% | 0–71% | **0%** |

Nine of nineteen categories go to zero. The two that do not — `toothbrush` and
`scissors` — are the rarest in the pool, and their surviving spikes are the
ranking-limited kind: the worst single trajectory (`toothbrush` seed 6, 23
flagged steps) sits at `n_good = 2` for its whole length with a **median oracle
cost of 0.944**. Nothing a threshold rule can do would rescue that run; it is
the #2825 positive-starvation failure, not the #2847 threshold blip.

Five `(category, seed)` cells — in *every* arm identically — found **zero
positives in 100 votes**, so they never trained and emit no steps. They are
excluded from all 147-trajectory counts above and are the same regime, taken to
its limit.

## The literal rerun — it reproduces, and the near-miss is the lesson

I also reran **#2847's exact command** on `evaluation-framework` (HEAD
`8f526819`), where `scripts/sod/sweep.py` lives. It was cheap: the feature cache
at `/exp/sgreenberg/threshold-stability/cache` already holds `coco/siglip2/whole`
regions for all 4952 images and 92 `cat` exemplars, so it runs on CPU in ~5
minutes.

At the issue's own `--iterations 5` it produced **zero** deep spikes — worst
warm-window cost 0.199 — and so did a second run with `--no-safe-thresholds`
(worst 0.213). Taken at face value that reads as "the branch already fixed
itself," and I nearly wrote it up that way.

**It is a sampling artefact.** At 20 seeds the same command deep-spikes in
**7 of 20 runs (35%)**, with worst-step costs of 1.00, 1.00, 0.50, 0.44 and 0.39
against an oracle whose warm-window median is 0.046. Seeds 0–4 — precisely the
five that `--iterations 5` runs — happen to be the quiet ones.

| seeds run | runs with a deep spike | worst warm cost |
|---|---:|---:|
| 0–4 (the issue's command) | 0 / 5 | 0.199 |
| 0–4, blend off | 0 / 5 | 0.213 |
| **0–19** | **7 / 20 (35%)** | **1.00** |

35% is the same rate as the figure in the issue, which spikes in 2 of its 5
seeds, and the same character. **The old path reproduces.** That is an
independent confirmation of what arm A asserts — from a different harness, a
different branch and a different split — and it is the strongest single piece of
support for the main result.

> **The methodological point is worth keeping.** A five-seed check has only a
> 76% chance of catching a phenomenon that hits 25% of runs; seeing zero has
> probability 0.24 under "nothing changed". A clean five-seed run is not
> evidence of a fix. Two of them in a row are not either — they are correlated,
> because they share the same five seeds. Anyone re-running #2847's command to
> check whether something helped should raise `--iterations` first; at 20 seeds
> the same null has probability 0.003.

## Why the sod harness could not be pointed at dev

`scripts/sod/` exists only on `evaluation-framework`, which is **110 commits
behind dev**, and it depends on `vtscore/eval/{region_curve,region_sources,
scoring_heads,threshold_rules,xcal,error_metrics}.py` — none of which exist on
dev. Its curve calls `calculate_safe_threshold(threshold, scores, n_votes)`, the
old three-positional-argument signature; dev's takes a `BlendContext`, and the
shipped path is now `fit_fold_anchored_cut`, which needs per-fold models and
anchor orderings the sod curve never builds. Porting it is real work, not a
rebase.

The dev-side calibration harness is the right vehicle instead:
`_safe_threshold_for_step` in `vtscore/eval/voting_iterations.py` calls the same
estimator `vtscore.detectors.training._safe_threshold` ships, with the same
arguments, specifically so the baseline arm cannot drift from the app.

## Recommendations

1. **Close #2847 as substantially fixed on dev, with the caveat that the fix is
   in the threshold, not the head** — and note that the issue's own branch may
   no longer reproduce its figure either.
2. **Follow up on the acquisition regression.** Production finding a median of
   4 positives per 100 votes against the conformal arm's 9 (p=1e-20) is a large
   effect on the axis users actually spend, and it is not obviously a good
   trade for steadiness when the endpoint cost is unchanged. This is the
   positive-seeking acquisition item still unbuilt from #2790/#2825, now with a
   measured price tag.
3. **Do not chase the residual 5.4% with a better cut rule.** Four fifths of
   what is left is ranking-limited, and the worst case is a run stuck at two
   positives. That is #2825's problem, and a threshold rule cannot reach it.
4. **The survivors are deeper than the originals** (median flagged cost 0.632 →
   0.748, max 1.236 — worse than chance). Fewer, worse events may be a worse
   user experience than more, milder ones; if a runtime guard is ever built, it
   should key on the fused threshold's own excursions, not on the conformal
   ones it replaced.

## Reproducing

```bash
# 4 arms x 19 categories x 8 seeds x 100 steps, no GPU stage
cd /exp/$USER/projects/vts-spike-2847/scripts/experiments/calibration
python selftest_analyze_spikes.py          # planted-answer test, run this first
CALIB_N_SEEDS=8 bash launch_spike_2847.sh
python analyze_spikes.py --category cat    # once all four arms drain
```

Analyzer self-test plants known spike counts per arm plus the traps that read as
good news: a cold-start hump that must not count, a ranking collapse that must
not count as a threshold spike, decoy variant rows, a header-only cell, a
zero-byte cell, and the paired-delta sign.
