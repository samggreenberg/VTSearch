# `calibration_fraction`: is a 50/50 Train/Calibrate split optimal? (#3287)

**Status:** pre-registered before the run. Decision rules below are fixed at
submission time; the report records the verdict they produce, including
"the incumbent survives", which is the outcome this study most expects.

## The question

`calibration_fraction` splits each calibration fold's labelset in two: a Train
half that fits the fold model, and a Calibrate half whose held-out scores the
threshold is read from. It has been `0.5` since it was introduced and was never
measured — the obvious default, not a result — and it sits on the shipped
threshold path, so it is priced on every detector anyone trains.

It is a genuine trade-off in both directions:

| more Train | more Calibrate |
|---|---|
| better fold models, so their held-out orderings are a closer proxy for the final model's | more anchors for `fit_fold_anchored_cut`, a finer conformal quantile |
| but fewer anchors and a coarser quantile | but the fold models drift further from the final model, which always trains on **all** votes — so the scale the cut is *read* on is less like the scale it is *applied* on |

## What #3288 did to the premise, before the run

The issue's evidence is #3286's period-4 wave: the whole-image arms were worst
at the parity that spent the odd vote on Train, `max_patch` worst at the parity
that spent it on Calibrate — anti-phase, pointing the same way twice.

That wave was `round`'s round-half-to-even tie-break, and **PR #3288 (merged to
`dev` hours before this run) replaced it with an unbiased dither.** On this code
the wave does not exist, so its anti-phase reading is no longer evidence of
anything. This is recorded here rather than quietly dropped, because a study
that inherits a dissolved premise and reports a result as if it confirmed one is
worse than a study that never ran.

The *question* survives intact: the constant is still unmeasured, and the
trade-off above is a property of the estimator, not of the rounding. The dither
in fact makes the measurement cleaner — pre-dither, an arm at 0.3 and one at 0.5
would have differed both in mean split and in which deterministic seesaw they
rode; now they differ only in the mean.

**One residual to read the results against.** The dither fires whenever
`n × fraction` has a fractional part: most steps at 0.3/0.4/0.6/0.7, and only
odd steps at 0.5. It is unbiased, so it does not move any arm's mean, but it
adds a little within-arm variance to the four off-centre arms. That is why the
headline is a level with a standard error and never a rank ordering of five
point estimates.

## Design

**Five full runs, not five paired arms.** The fraction sets the threshold, the
threshold sets the acquisition cut, and the cut sets which item Autopilot's Hard
pick samples next — so an arm at 0.3 has collected different votes by its second
trained step. This is the same reason `calibrate_count` needed a live A/B
(`launch_folds_2897_ab.sh`) after its cheap screen: a knob upstream of
acquisition cannot be screened inside one trajectory.

| axis | value | why |
|---|---|---|
| arms | `calibration_fraction` ∈ {0.3, 0.4, 0.5, 0.6, 0.7} | the issue's grid; 0.5 is an arm, measured here, not a level quoted from another study |
| dataset | `vg_scale_any` | 12 hand-checked classes × 300 positives against one shared 3900-image negative pool → **prevalence identical in every cell** |
| geometries | `siglip/whole_image`, `dinov3_patch/whole_image`, `dinov3_patch/max_patch` | the middle corner is what separates **voting mode** from **embedder** |
| opening | SigLIP text sort in every cell (`siglip+dinov3_patch` pair) | DINOv3 has no text tower; a bare arm would put a *seeding* contrast inside the mode contrast (#3278) |
| seeds | 4 | the minimum that makes `sd(threshold)` — taken *across* seeds at a fixed step — computable at all |
| steps | 150 | the trade-off is predicted to reverse inside the horizon; the crossing has to be *in* the window |

Uniform prevalence is the instrument, not a nicety. A threshold **is** a
quantile of the calibration set, and this study's subject is how big that set
should be — so a grid whose cells' calibration sets differ 60-fold in size
(`visual_genome_m`'s 25-to-1645 positives) would confound the swept axis with
itself.

Everything else is production: the fused threshold path (`CALIB_SAFE_THRESHOLDS=1`
— `docs/ML.md`: "Every trained threshold fuses the haystack into the cut. There
is no setting for this"), `calibrate_count=2`, the linear-SVM head that has been
production since PR #3198, and the per-mode blend schedule from #2841.

## Metrics

`cost` (headline), `regret_honest` (the cross-fitted reference, for a *level*),
`sd(threshold)` across seeds, and `n_cal_scores`.

## Read across vote bands, not pooled

Bands: 1–25, 26–60, 61–100, 101–150. Few votes should favour spending them on
the model; many votes should favour resolution. If that is real the arm ordering
**reverses** somewhere inside the horizon, and a single pooled winner is an
average across a crossing — the number that hides the finding, and probably the
wrong thing to ship anyway.

## The analysis trap, and what is done about it

`rule_inefficiency` and `calibration_shift` are **not independent effects of
this knob.** `calibration_shift` is measured against `cal_oracle_cost`, which is
estimated *from the calibration set*, so moving the fraction moves the yardstick
itself: the two terms slide in opposite directions with their sum pinned to
`regret` by construction. #2897 read exactly that anti-correlation as a finding
when it was algebra.

The decomposition is therefore computed **only** as `agg/trap_check.csv`, and
only to show the anti-correlation and the pinned sum. No per-term claim is made.
Levels are read off `regret` / `regret_honest`, which are referenced to
something this knob does not move.

## Pre-registered decision rules

An arm is a **candidate** for a mode's default when both hold on `cost`:

1. it beats 0.5 by more than **2 SE**, pooled inverse-variance across the bands;
2. it is not worse than 0.5 by more than **0.01** (`HARM_TOLERANCE`, the margin
   PR #2891 pre-registered) in **any** band.

(2) is pointwise on purpose: an arm can win on average while being worse
everywhere a short session actually lives, and a short session is most sessions.

Standard errors are bootstrapped over **cells**, never steps — consecutive steps
of one trajectory share a model, so testing over steps would count one
trajectory's luck a hundred times. Arms are paired within
`(dataset, category, seed, geometry)`, never on votes, because by construction
they do not share votes.

Outcomes, both of which are results:

- **"0.5 is fine, here is the evidence."** Worth having on record — it is
  currently an unmeasured constant on the shipped threshold path.
- **A per-mode default**, alongside `PRODUCTION_SCHEDULE_BY_MODE`. If that
  lands, the eval default arm has to move with it (`docs/EVAL.md`, "The Eval
  Default Arm IS the App") and `scripts/check-eval-app-sync.py` gains a
  `default`-kind mirror — there is currently no mirror watching this constant.

## Running it

```bash
cd /exp/$USER/projects/vts-calfrac-3287/scripts/experiments/calibration
python selftest_analyze_calfrac.py            # planted answer; before the array
bash launch_calfrac_3287.sh prepare           # once, shared by every arm
bash launch_calfrac_3287.sh size 0            # a binary cell
bash launch_calfrac_3287.sh size 48           # a region cell: it sets mem + the critical path
bash launch_calfrac_3287.sh arms              # five arrays + one cross-arm analyze
```

One prepare, five symlinks, one cell numbering: `run_cells.py --print-cells`
enumerates from `prepare_info.json`, so five independent prepares would be five
chances for array index 37 to mean a different cell in different arms.
