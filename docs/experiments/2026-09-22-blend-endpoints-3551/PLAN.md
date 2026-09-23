# #3551 — tuning the retired `rare` and `corridor` blend schedules

**Pre-registered before the screen array went in** (2026-09-22). Executes issue
#3551, which follows #2841 / PR #2849.

## The question the issue asked, and the one today's stack asks

#2841 retired `rare` (ramp on the rarer class, 1→8) and `corridor` (clamp the
x-cal cut between the GMM component means) against `cap50` / `slow_cap50`, on
endpoints nobody tuned. At the time the schedule blend **was** the shipped
threshold.

It no longer is. Since #2861/#2863 the shipped threshold is the fold-anchored
fused cut, unconditionally, and the schedule blend runs only as its
**fallback**: `vtscore.detectors.training._fused_threshold` calls
`calculate_safe_threshold` only when the calibration folds are unusable, and
then feeds it `NO_GOOD_THRESHOLD` (2.0, "admit nothing") as the x-cal side.
`PRODUCTION_SCHEDULE_BY_MODE` therefore picks how a **fallback** step combines
the GMM midpoint with that sentinel. Re-running #2841's arms as written would
measure a path the app no longer takes on ~95% of steps.

So the issue splits into two questions, and one screen measures both:

- **Q1, fallback.** On the steps where production falls back, which schedule
  should combine the GMM cut with the sentinel? A fallback step in the harness
  always holds both classes (no row exists before one Good and one Bad coexist)
  with the rarer at **one** vote, or fewer than four votes in total. On those
  steps every `rare` ramp with `lo ≥ 1` is **pure GMM**, and the corridor clamps
  the sentinel to `cut + w·(mu_hi − cut)`.
- **Q2, replacement.** On fused steps, would a *tuned* blend of the raw x-cal cut
  and the GMM midpoint beat the fused cut? #2864 measured `cap50` tying fusion on
  COCO binary and beating it on caltech101 — four threshold changes ago
  (κ, `mid_tilt`, #3308 vote exclusion, #3825/#3585 EM) — so on binary voting a
  tuned `rare` beating today's fusion is not a formality.

## What was changed before measuring

- **The `corridor_ramp` discontinuity is fixed first** (the issue's instruction):
  it released its clamp entirely at `hi`, so a wild x-cal jumped from nearly the
  corridor edge to the raw cut between 19 and 20 labels — on a fallback step,
  from `mu_hi` to "admit nothing". It now holds its full width past `hi`.
- **Corridor width is a parameter**, `w` ∈ [0, 1] = the fraction of the way from
  the GMM cut to each component mean. `w=1` is #2841's corridor; `w=0` is pure GMM.
- **Parametric schedule names** (`rare:lo=1:hi=16`, `corridor:w=0.2`) let the
  grid name points without registering them; malformed names fail the cell.
- **Every schedule row carries `shipped_provenance`** (fused vs `gmm_blend`) and
  `fold_fallback`, so the Q1 and Q2 rows are never pooled.
- **The rows mirror the app's fallback exactly**: the GMM is fitted on the
  scored population, as `calculate_safe_threshold` fits it.

## The grid

| axis | value | why |
|---|---|---|
| environments | region: `visual_genome_m`, `coco_val` × `siglip+dinov3_patch` / `max_patch`; binary: `visual_genome_m`, `coco_val`, `caltech101_m` × `siglip` | the three classic pile datasets, none dependent on vg_scale, DocMarks or coco_quarry. Two region and three binary environments, so no verdict rests on one. The region arm is the pair (#3278). |
| categories | the harness's own selector; caltech at 12 | the study does not pick its environments. |
| cell seeds | 4 | |
| calibration draws | 42, 0 — a **cell axis** | #3796: 70% of cell-to-cell variance is the draw. Two draws make every contrast an average over splits rather than a fact about split 42. |
| horizon | 150 votes | the fallback lives in the cold start; the replacement question needs depth (#2864's binary gap tracked positives). |
| everything else | the app's own | fused threshold on, production linear-SVM head, per-space calibration fraction, acquisition offset −4 (#3319; holding it is what keeps these arms comparable to #2841's, half of whose gain was acquisition feedback), inclusion 0. `--diverges calibration_seed` is the only declared divergence. |

**Schedules scored on every step** (42): references `cap50`, `slow_cap50`,
`prod`, `cap80`, `pure_gmm`, `pure_xcal`; `rare:lo=L:hi=H` for L ∈ {0,1,2,4},
H ∈ {4,8,16,32,64}, H > L (19); `rare:lo=1:hi=H:cap=0.5` for H ∈ {4,8,16,32};
`rare:lo=1:hi=H:cap=0.8` for H ∈ {8,16,32}; `corridor` and `corridor:w=W` for
W ∈ {0.05,0.1,0.2,0.3,0.5,0.75}; `corridor_ramp`, `corridor_ramp:w=0.2`,
`corridor_ramp:w=0.5`. Both endpoint ranges run far enough past #2841's guess
that an optimum on a grid edge is visible as one (#2861's lesson).

## The unit, the metric, and the noise floor

- **Unit: the cell** (environment × category × seed × draw). 150 steps of one
  trajectory are one experiment. Every p-value and SE is on cell means.
- **Metric: `cost`** (FPR + FNR at inclusion 0), averaged over the trajectory —
  the area under the cost-over-clicks curve, per the issue's "score on the
  trajectory" — and banded on votes (1–20, 21–50, 51–150) and on positives found.
- **Reweighting:** every contrast is also scored at `4·FPR + FNR` and
  `FPR + 4·FNR`. #2841 showed a lower cut can win at 1:1 and reverse at 4:1; the
  Inclusion knob is exactly that reweighting, so a winner must survive both.
- **Q1 contrast:** per step, `cost(schedule)` on fallback steps and `cost(base)`
  on fused ones, minus `cost(base)`; reported both **diluted** (over every step,
  what a user's session sees) and **conditional** (over fallback steps only,
  what the schedule does where it acts).
- **Q2 contrast:** `cost(schedule row) − cost(base row)` on every step.
- **Screen noise.** Screen rows are paired within a step — same model, same
  split, same test scores — so the draw does not enter their difference. The
  calseed floor (sd ≈ 0.047 per cell from the draw) applies to the A/B, where
  trajectories diverge: resolving 0.01 at 2 SE then needs ~(2·0.066/0.01)² ≈ 170
  cells per environment, which the `ab` mode's 8 seeds × 2 draws × ~12–24
  categories provides.

## Fidelity gate (run before anything is reported)

The shipped fallback schedule's row (`slow_cap50` region, `cap50` binary) must
equal the base row's threshold on **every** fallback step. The analyzer counts
mismatches and refuses to write a verdict if there is one.

## Decision rules

**Promotion to A/B** (the screen decides who runs, never what ships). A
candidate is promoted for a voting mode if, in **every** environment of that
mode, its Δcost against the shipped path is negative and resolvable
(|Δ| > 2 SE), and at neither reweighting is it resolvably worse. Q2 reads the
trajectory-mean contrast. Q1 reads the **conditional** one (fallback steps
only): its diluted contrast is small by construction, and a fallback step also
sets the acquisition cut, which the screen cannot see and #2841 found to be
half of a schedule's effect — so the A/B, not the screen, measures what the
diluted effect is worth. At most the best two per family per question.

**Ship rule** (A/B, per mode): adopt iff, against the shipped arm, cell-paired
on trajectory-mean cost,

1. the within-mode pooled 95% CI upper bound is below 0; **and**
2. in every environment of that mode the 95% upper bound is below +0.01 (the
   #2877 / #3319 non-inferiority tolerance); **and**
3. neither reweighting reverses the sign resolvably (> 2 SE) in any environment.

A Q2 winner would replace the fused cut, which needs a harness knob to make the
blend the live threshold; that is built only if the screen promotes one.

**If nothing is promoted, nothing ships**, and the report says what the tuned
optimum is worth and where it sits, so the question does not need re-asking.

## What this cannot show

- The screen holds the trajectory fixed, so it cannot see acquisition feedback
  (#2841: ~half its gain). A null screen bounds the threshold-rule effect, not
  the whole-system effect; the A/B exists for anything promoted.
- Issue item 3 (why `pure_gmm` loses at `fpr×4` on region voting) is a
  diagnostic, not a decision; it is filed separately (#4103) rather than run here.
