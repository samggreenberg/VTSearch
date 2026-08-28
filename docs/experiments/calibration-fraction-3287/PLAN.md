# What Train/Calibrate split should a detector use? (#3287)

**Status:** pre-registered before the run. Decision rules below are fixed at
submission time; the report records the verdict they produce, including
"the incumbent survives", which is the outcome this study most expects.

## The question

Of the votes a user has cast, what share should **train** each calibration
fold's model, and what share should be held out to **read that fold's
threshold**?

`calibration_fraction` is the knob that decides it. It has been `0.5` since it
was introduced and was never measured — the obvious default, not a result — and
it sits on the shipped threshold path, so it is priced on every detector anyone
trains.

It is a genuine trade-off in both directions:

| more Train | more Calibrate |
|---|---|
| better fold models, so their held-out orderings are a closer proxy for the final model's | more anchors for `fit_fold_anchored_cut`, a finer conformal quantile |
| but fewer anchors and a coarser quantile | but the fold models drift further from the final model, which always trains on **all** votes — so the scale the cut is *read* on is less like the scale it is *applied* on |

## What is being measured

The cost a VTSearch user ends up paying, on average, at each split fraction —
not a mechanism, and not an artifact. Everything except the fraction is held at
the app's own behaviour: the fused threshold path, the production linear-SVM
head, `calibrate_count=2`, the app's per-mode blend schedule, and the text-sort
opening a user gets by typing a query.

"On average, across scenarios" is the shape of the answer, so the grid spends
its cells on scenarios rather than on precision at any one of them: three
geometries, 12 classes at identical prevalence, 4 seeds, and four vote bands
across a 150-click horizon.

One note on provenance, because it changes how a number here should be read but
is not itself the subject. The question surfaced from #3286's period-4 waves in
the learning curves; PR #3288 has since traced those to `round`'s
round-half-to-even tie-break and replaced it with an unbiased dither. So the
waves are gone on this code and are **not** what this run is about — but their
absence is what makes it a clean measurement, because two arms now differ only
in their mean split and not also in which deterministic seesaw they rode.

The dither does leave one residual worth reading the results against: it fires
whenever `n × fraction` has a fractional part — most steps at 0.3/0.4/0.6/0.7,
only odd steps at 0.5. It is unbiased, so it moves no arm's mean, but it adds a
little within-arm variance to the four off-centre arms. That is why the headline
is a level with a standard error and never a rank ordering of five point
estimates.

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
| cell order | `seed`, not the `category` default | the run has a wall-clock deadline. A truncated `category` array loses its last **categories** entirely — whole environments, and the per-mode contrast short at one end; a truncated `seed` array loses its last **seeds**, uniformly, which widens the standard errors and leaves the design intact. With five arms it matters twice over: arms that lost *different* categories would not be comparable at all. |

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
bash launch_calfrac_3287.sh size 12           # a region cell: it sets mem + the critical path
bash launch_calfrac_3287.sh arms              # five arrays + one cross-arm analyze
```

One prepare, five symlinks, one cell numbering: `run_cells.py --print-cells`
enumerates from `prepare_info.json`, so five independent prepares would be five
chances for array index 37 to mean a different cell in different arms.
