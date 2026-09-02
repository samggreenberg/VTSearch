# Does the Inclusion knob still have authority under the linear SVM head? (#3196)

**Status: pre-registered, not yet run.** Decision rules below are fixed before
the first cell lands; the report records the measurement against them.

## The question

PR #3198 made `LINEAR_SVM_HEAD` the head a live detector trains. The shipped
fold-anchored cut rule is
[`FOLD_ANCHOR_CUT_RULE`](../../../vtscore/training/thresholds/anchored.py) `= "mid_tilt"`,
composed as `q_mid + (q_rate(weights) - q_rate(equal weights))`. `rate`'s root
is **invariant to the cost weights while it stays inside the inter-mean
interval** — the prior-odds factor in its `lam` cancels the `w_lo`/`w_hi` in
`_rate_cut`'s offset — so the tilt only materialises once the knob pushes that
root *out* of the interval and the first-order continuation takes over.

#3196 observed that under the SVM head the fitted components keep the root
interior much further along the knob: on the synthetic planted-patch fixture the
tilt moved the threshold at 7/7 grid steps under the logistic head and **0/7**
under the SVM. Synthetic planted data is well-separated by construction, which
is exactly the regime that keeps the root interior, so it is evidence about the
fixture and not about users. This study asks the product question on real data:

> Does the Inclusion slider still change what a user sees, in the fold-anchored
> path, under the head that actually ships?

## Environments

**One dataset, `vg_scale`** (#3156, PR #3163/#3255): 12 hand-checked classes at
three box-size bands (`bus@small`, `bus@medium`, `bus@large`, …) = 36 designated
cells, 100 positives and prevalence **0.0250 in every one**. Chosen over the
issue's COCO + `visual_genome_m` pair on the owner's instruction — the labels are
verified and every cell is already embedded in the pile — and it is the better
fixture for *this* question anyway:

- A threshold is a quantile of a calibration set, so a grid whose cells differ
  60-fold in positives (which `visual_genome_m` does) confounds the swept axis
  with prevalence. `vg_scale` holds prevalence exactly fixed.
- The mechanism under test is **separability**: the tilt dies when the rate root
  stays inside the inter-mean interval, which is what a cleanly separated
  haystack produces. The box-size band **is** a separability ladder — #3255
  measured cost roughly tripling from large targets to small — so the three
  bands sweep the very axis the mechanism runs on, rather than leaving it to
  vary uncontrolled between two unrelated datasets.

Three arms, which separate the embedder from the voting mode (#3115's corner):

| env | voting mode | what it isolates |
|---|---|---|
| `vg_scale/siglip/whole_image` | binary | the shipped default detector |
| `vg_scale/siglip+dinov3_patch/whole_image` | binary | same mode, other embedder |
| `vg_scale/siglip+dinov3_patch/max_patch` | region | same embedder, other mode |

The region arm is the **pair** `siglip+dinov3_patch`, not bare `dinov3_patch`:
DINOv3 has no text tower, so a bare arm opens on three random known-goods while
every SigLIP arm opens on a typed query, hiding a seeding contrast inside the
voting-mode contrast (#3278). `CALIB_REQUIRE_OPENING=text` asserts it per cell.

## Arms

**Head is a run-level A/B, not a paired arm.** The threshold drives Autopilot's
`hard` pick, so a different head collects different votes; the two heads cannot
share a trajectory and therefore cannot share a `CALIB_EXP`.

- `svm` — `CALIB_HEAD` **unset** → `PRODUCTION_HEAD`, the linear SVM users have.
- `linear` — `CALIB_HEAD=linear`, the logistic head the SVM replaced. Declared
  to preflight as `--diverges head`: it is deliberately off-production, and it
  is the reference that says what the knob *used* to do.

Everything else is production: `safe_thresholds` on (the app has no switch),
anchor mass κ=0.3, combine `qmean`, `calibrate_count=2`, the app's per-mode
blend schedule, the per-space calibration fraction (#3287), and the shipped
`ACQUISITION_INCLUSION_OFFSET`.

Inside each run the cut sweep is **eval-only and near-free** — the per-fold
anchored EM does not depend on the cut rule or the inclusion, so one fit per step
serves the whole (rule × k) grid, which is the same no-refit re-cut the app does
when the user drags the slider:

- `CALIB_CUT_INCL_KS=-10..10`, **all 21 stops** (the issue asks for the full
  nominal range; #2865 sampled 13 and could not see where a band begins).
- `CALIB_ANCHORED_RULES=mid,mid_tilt,rate,cross_tilt,q_tilt`.
- `CALIB_CUT_INCL_QTILT_STEPS=0.005,0.01,0.02,0.04,0.08` — `q_tilt`'s step size
  is a free parameter with no derivation, so it is fitted, not assumed.

## Outcomes

All read off the `__cutincl` frame, on the incumbent arm
`fold_anchored_w0.3_mid_tilt_qmean`, per environment and per head. Computed per
*step* — one trajectory point is one realization of the whole slider, and "how
many answers does dragging it produce right now?" is what a user experiences —
then averaged over steps and cells.

- **flat fraction** (`dead_step_rate`): share of adjacent `k` pairs that admit
  the identical set. This is the issue's "fraction of the knob over which
  `admitted_frac` is constant".
- **authority** (`admitted_span`): `admitted_frac` end to end across the knob.
- **inert rate**: share of steps where all 21 stops admit one single set.
- The flat band's **location**: dead-step rate as a function of `k`, which is
  what says whether the band sits where users are (near 0) or only at the ends.

## Pre-registered decision rules

**H1 — the head moved the knob.** Paired by cell (same env, category, seed;
trajectories differ by construction, so the cell is the pairing unit), bootstrap
over cells: `dead_step_rate(svm) - dead_step_rate(linear)` for the incumbent.
Supported if the 95% CI sits entirely above 0 in at least one environment.

**H2 — the knob has gone soft in absolute terms** (the product question, read on
the **shipped** head only, in the deep regime `n_votes >= 100`). Fires in an
environment when either:

- `dead_step_rate >= 0.5` — half the slider does nothing; or
- `admitted_span <= 0.05` — dragging the slider end to end changes fewer than 1
  item in 20.

H2 is what decides whether anything needs fixing. H1 without H2 is a fact about
the head change with no user consequence, and the report says so.

**H3 — ship `q_tilt` only on a measurement.** All three must hold:

1. H2 fires in at least one shipped environment;
2. `q_tilt` at some single step size materially lowers `dead_step_rate` there
   (CI on the paired difference vs the incumbent entirely below 0);
3. that same step size stays inside the **0.01 rate-scale regret tolerance**
   against the incumbent at **every** `k` in **every** environment — the
   non-inferiority bar #2865 and PR #2891 both pre-registered, on
   `cut_regret / 2**abs(k)` so one tolerance means the same thing at both ends
   of a knob whose cost weights double per step.

If no single step size clears (3) everywhere, `q_tilt` does not ship and the
recommendation is to keep `mid_tilt`. A free parameter that only wins at one
hand-picked value is not a result.

**H4 — the acquisition offset (measurement, not a decision).** #2896 records
that `ACQUISITION_INCLUSION_OFFSET` collapses wherever the tilt flattens: the
offset is a *gap across the slider*, so a flat band makes the selector cut and
the reporting cut the same cut. Report, per head and environment, the share of
steps where `admitted_frac(k + offset) - admitted_frac(k) == 0` at the shipped
offset, and the mean gap. A wider flat band predicts a wider collapse.

## Falsifiers and instrument checks

These are read **before** any headline number; a failure here means the
instrument is broken and nothing else in the run is readable.

- **`mid` must come back inert.** It never looks at the cost weights, and #2865
  measured one admitted set across the whole slider in all 65,671 cell-steps. If
  `mid`'s `dead_step_rate` is not ≈1.0, the flatness measure is wrong.
- **`mid_tilt` and `rate` must have identical liveness.** #2865 showed
  `mid_tilt(k) - rate(k) = q_mid - q_rate(0)` **exactly** — a constant offset in
  fold-quantile space — so the two differ by where they sit, never by how much
  they move. A liveness gap between them contradicts the algebra.
- **The region premise is asserted per cell**, not per dataset
  (`--require-region-voting vg_scale:siglip+dinov3_patch`, and `styles=` in the
  harness's own resolution output is the evidence a patch cell is a patch cell).
  This is the #2877/#2897/#2905/#3255 failure and it has cost a full arm twice.
