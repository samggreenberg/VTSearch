# Why region voting and binary voting want different anchor masses

**Open question behind #2864 · proposed experiments · a runnable synthetic
bench (`scripts/experiments/calibration/theory_kappa_bench.py`) with
preliminary results**

## Background — the observation that has no explanation yet

PR #2864's anchor-mass sweep established two things empirically and explained
neither:

- **Fusion pays on region voting and does not on binary voting.** Against the
  blend that ships, `fold_anchored κ=0.3 mid` is −0.026 on region voting and
  −0.0004 (n.s.) on binary; the shipped `κ=1 rate` is *+0.0063 worse* than
  `cap50` on binary.
- **The cut rule flips with κ, and κ\* differs by mode.** `mid` peaks near
  κ=0.3 and `rate` near κ=1, the curves crossing at κ≈1–2.

The report attributes the *rule* flip to mixture weights (`mid` ignores them,
`rate` reads them, and anchor mass lets acquisition-biased prevalence into
them) and attributes the *mode* split to positive counts (24 → −0.093, 8 →
−0.019, 3 → −0.002). The positive-count story is a good predictor but it is
not a mechanism: it says *when* fusion pays, not *why max over a set per image
changes what the estimator wants*. That is the open question.

Note also that #2864's design **confounds pooling with class separation**: its
region environments (Visual Genome) are also its overlap-heavy ones, and its
binary environments (COCO, caltech101) are its near-separable ones — binary
x-cal FNR runs 0.04–0.06 against region's 0.18–0.28. Nothing in that design
can say which of the two axes drives the split.

## The hypothesis: max-pooling is an overlap generator, and overlap is curvature

An image's score is `max` over its `m` region scores. Against the whole-image
(`m=1`) geometry that operator does three things at once:

- **Translates and compresses the negative bulk.** The max of `m` background
  regions concentrates near `sqrt(2 ln m)` with shrinking spread (at `m=24`,
  1.95 ± 0.51 against 0.00 ± 1.00). Negatives move *toward* the positives.
- **Censors the positive lower tail.** A positive image's score is
  `max(object, background)`, so a weak object is replaced by its own
  background max. The positive class inherits the negative class's upper tail.
- **Amplifies FPR `m`-fold.** `FPR(t) = 1 − Φ(t)^m`: every region gets an
  independent chance to fire, so the same per-region threshold error costs `m`
  times as much.

All three push the same quantity up — the **curvature of the rate loss at its
optimum**, `L''(τ*)`. In the closed-form model, `L''` goes 0.39 → 0.74 → 1.00
across `m` = 1 → 6 → 24 at separation 3.

Curvature is the bridge to κ, because regret is locally quadratic in threshold
error:

> **regret ≈ ½ · L''(τ*) · (τ̂ − τ*)²**

So curvature is the exchange rate that converts threshold *imprecision* into
cost. Where it is high, the estimator's **variance** dominates its bias, and
the right move is to lean on the low-variance estimate — the 2000-sample
haystack — and admit only a whisper of the high-variance, acquisition-biased
label evidence. That is a **small κ**. Where curvature is low (`m=1`,
separable), the valley floor is flat, threshold precision is nearly free, and
the labels' *identification* value — knowing which component is which, and
which quantile matters — is worth more than their variance costs. That is a
**large κ**.

This reframes the mode split as not really about voting mode at all. Voting
mode is a *proxy* for where the task sits on the curvature axis, and #2864's
design happens to have both of its axes (pooling, separation) pointing the
same way.

## Preliminary evidence — the synthetic bench, run 1

`theory_kappa_bench.py` sweeps the **production estimators**
(`fold_anchored_gmm_threshold`, `anchored_gmm_fit`,
`threshold_from_fold_orderings`, `rank_transfer`, `blend_gmm_threshold`) over
the #2836 closed-form generative model, so every cut is scored against the
*exact* optimum rather than a held-out sample. Each replicate trains a 1-D
logistic head on the votes plus one per calibration fold, so all three
cross-calibration deficits (#2790/#2799: conformal sample size, fold→final
scale transfer, per-retrain redraw) are present and priced.

**Run 1: 96 configurations × 25 reps = 2400 replicates**, `m` ∈ {1, 6, 24} ×
separation ∈ {2, 3} × prevalence ∈ {0.02, 0.05} × votes ∈ {20, 50, 100, 300} ×
{random, hard} acquisition. Results in
`scripts/experiments/calibration/` output dirs; the headline table is κ\* (fold
family, `mid` rule, random votes):

| κ\* by vote count | 20 | 50 | 100 | 300 |
|---|---:|---:|---:|---:|
| **m = 24** (region-like) | 0.1 | **0.01** | **0.01** | **0.01** |
| **m = 1** (binary-like) | 10 | 1 | 0.3 | 0.03 |

**At every vote count, the pooled geometry wants an anchor mass one to two
decades smaller than the whole-image geometry.** That is the qualitative
answer to "why does max over a set suggest a different κ setup": pooling
raises curvature, curvature punishes variance, and κ is exactly the knob that
buys label information at the price of variance. It also reproduces #2864's
finding 3 (κ\* falls as votes accumulate) independently, in a model with no
acquisition feedback and no real data.

Broken out by separation, with curvature alongside (random votes, pooled):

| config | curvature | κ\* | x-cal excess |
|---|---:|---:|---:|
| m=1, sep=3 | 0.388 | **1** | 0.043 |
| m=1, sep=2 | 0.484 | 0.01 | 0.028 |
| m=6, sep=2 / 3 | 0.739 / 0.740 | 0.01 / 0.03 | 0.031 / 0.040 |
| m=24, sep=2 | 0.787 | 0.01 | 0.028 |
| m=24, sep=3 | 1.003 | 0.01 | 0.035 |

The lone configuration wanting a large κ is the one with the **lowest
curvature**, and it is the most binary-like cell in the design (unpooled,
separable). Consistent with the hypothesis — but see the caveats.

**Three things run 1 did *not* confirm, stated plainly:**

- **The fusion *advantage* did not grow with `m`** (0.032 / 0.034 / 0.030 for
  m = 1 / 6 / 24). The naive form of the prediction — "pooling is what makes
  fusion pay" — is **false** in this model. Fusion beats x-cal everywhere by a
  similar margin; what changes with `m` is *which κ* you must use to get it.
  So `m` moves κ\*, not the size of the prize. Any theory claiming pooling
  explains #2864's *mode split in effect size* has to answer this.
- **κ\* is censored at the grid bottom** for five of six cells (0.01 was the
  lowest κ tested), so the curvature ordering is only partially observable and
  the correlation of curvature with advantage is weak (r = 0.17).
- **Hard acquisition inverted the prediction**: threshold-adjacent votes moved
  κ\* *up* (to 3–10), not down, at every `m`. Under the `mid` rule that is
  coherent — labels near the boundary are highly informative about *where the
  boundary is*, and `mid` never reads the corrupted mixture weights — which
  suggests #2864's acquisition-bias mechanism is specific to the weight-reading
  `rate` rule. Worth testing directly rather than assuming.

**Status: preliminary and not to be cited as a result.** Run 2 (extending κ
down to 0.001 to uncensor the argmin, adding separations 4–5 to break the
pooling/separation confound, adding the `cap50`/`slow_cap50` blend arms that
fusion actually loses to, and a mixed rather than all-hard acquisition mode)
completed 74 of 192 cells before the container running it was recycled. The
bench checkpoints per configuration, so a rerun resumes.

## Open work

<!-- item-sep -->

- **Run 2 of the synthetic bench to completion.** `python
  theory_kappa_bench.py --reps 25 --procs <n>` — 192 configurations, single-
  threaded per cell, ~30 CPU-minutes at 4 procs, no GPU, no dataset, no
  prepare stage. Cells checkpoint to `cells/cfg_NNNN.csv` and resume, so it
  can be chunked or restarted freely. The three questions it settles:
  **(a)** where κ\* actually lands once the grid extends to 0.001, i.e.
  whether the m=24 argmin is interior or the model genuinely wants
  vanishing anchor mass; **(b)** whether **separation or pooling** is the
  real driver, which #2864's design cannot separate and this one can, since
  it crosses `m` ∈ {1, 6, 24} with separation ∈ {2, 3, 4, 5} — the decisive
  contrast is `m=24, sep=5` (pooled but separable) against `m=1, sep=2`
  (unpooled but overlapping); **(c)** whether fusion loses to `cap50` in the
  near-separable cells, which is the real-data result the mechanism has to
  reproduce to be believed at all. Falsifier: if κ\* tracks separation and
  ignores `m`, the max-pooling story is wrong and "curvature" reduces to
  "class overlap", which is a simpler and more useful statement.

<!-- item-sep -->

- **Curvature as the shipped gate, if it survives.** The practical payoff of
  the mechanism is a gate that is *measurable at run time*. `L''(τ*)` is not
  observable, but the fitted mixture is: the two components' separation and
  the density at the cut are both available from `GmmFit1D` at zero extra
  cost, and either is a candidate proxy. This would replace #2864's
  recommendation 3 ("gate on positives") with a gate on the quantity the
  positives were standing in for — and unlike the positive count, it does not
  need labels to evaluate. Only worth designing after run 2 says whether
  curvature is the right axis.

<!-- item-sep -->

- **The causal test on real data.** The synthetic bench can only show the
  mechanism is *sufficient* in a model. To show it operates in production,
  vary `m` while holding the dataset, embedder and votes fixed: re-pool an
  existing patch-embedder region environment at `m` ∈ {1, 4, 24} (top-1 vs
  top-k vs all regions — the harness already has `segment_topk_mean_pool` and
  the `_repool_variants` machinery from #2781) and re-run the κ sweep on each.
  A real-data κ\* that falls with `m` on the *same* images is the causal claim;
  #2864's cross-dataset comparison cannot make it. This is the expensive one
  (GPU prepare + a full cell array) and should wait on run 2.

<!-- item-sep -->

- **Separate the acquisition-bias mechanism from the cut rule.** Run 1's hard
  mode moved κ\* the *opposite* way #2864's mechanism predicts, but only the
  `mid` rule was examined closely. Compare `mid` and `rate` under matched
  acquisition: if the corrupted-weights story is right, `rate` should degrade
  under hard acquisition while `mid` improves, and the κ\* crossover should
  move with it. Cheap — it is a re-analysis of run 2's existing rows, since
  both rules are already emitted per cell.

## Reproduction

The bench is self-contained: no dataset, no embedder, no models, no
`prepare` stage. It needs `numpy`, `scipy`, `scikit-learn`, `pandas` and the
`vtscore` import path that `common.setup_env()` sets up.

```bash
cd scripts/experiments/calibration
VTS_REPO=<worktree> CALIB_EXP=<scratch> \
  python theory_kappa_bench.py --reps 25 --procs 16 --out <results>
```

`--smoke` runs six configurations for sizing. `--no-resume` re-runs cells that
already exist. The self-check (`selfcheck()`, run before every sweep) asserts
the sampled scores reproduce the closed-form `FPR`/`FNR` at three thresholds,
so a broken generative model fails loudly rather than producing a plausible
wrong curve. Outputs: `kappa_raw.csv` (one row per replicate),
`kappa_by_config.csv`, `kappa_curve_{fold,label}_mid.csv`, and
`kappa_summary.json` (the P1–P4 verdicts).
