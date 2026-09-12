# #3796 — the calibration-split noise floor

**Pre-registered before the array went in.** Executes issue #3796, which follows
#3794 / PR #3797: that PR added `calibration_seed` to the voting-iterations eval
and deliberately ran nothing, because the container it was written in had no
dataset and no cluster.

## The question

Production splits each calibration fold's labelset off a fresh
`RandomState(CALIBRATION_SPLIT_SEED)` — 42 — on every fit. The pin is
deliberate: #2934 fixed exactly the opposite state of affairs, where the
uncached threshold paths drew from the unseeded global `np.random` and a
detector's verdicts moved between two runs on identical votes.

So every user gets the same arbitrary draw, and every eval number this repo
quotes is one sample from a distribution nobody has measured. **How wide is that
distribution?**

## What is held, and what moves

The grid is inverted against every other sweep here. Elsewhere a knob is varied
to find a better setting; this knob is already decided, and is varied to find
out what a single sample from it was worth believing. So the **cell seed is
held** — same medias voted, in the same order, against the same held-out test
split — and the **split is redrawn**:

```python
run_voting_iterations_eval({"vg": medias}, seeds=[s], calibration_seed=c)
```

A draw is a whole run, not a re-cut of another draw's run. The loop is closed:
a different split is a different threshold, a different threshold is a different
acquisition rank, and a different rank is a different next vote. That
compounding is part of the measurand, not a confound — it is what a user's
session actually does — and §5 below measures how much of the spread it is.

## The grid

| axis | value | why |
|---|---|---|
| dataset | `vg_scale_any` (18,049 medias, 25 classes available) | identical prevalence in every cell. A threshold **is** a quantile of the calibration set, so cells whose calibration sets differ in size would confound the spread with the thing that drives it. The issue says "vg_scale"; this is the member of that family built for calibration studies (#3115, #3287). |
| classes | 5, prevalence-spread: `fork`, `bicycle`, `knife`, `bird`, `kite` | the issue's "~5 classes", chosen by the harness's own selector so the study does not pick its own environments. |
| geometries | `siglip/whole_image`, `dinov3_patch/whole_image`, `dinov3_patch/max_patch` | the region column is the one the issue asks about; the other two are #3115's confound-breaking corner, so a difference between the modes is separable from the embedder. The region arm is the **pair** `siglip+dinov3_patch` — DINOv3 has no text tower, and a bare arm would open on three random known-goods while the SigLIP arm opens on a typed query (#3278). |
| cell seeds | 5 | the comparison is spread-across-draws against spread-across-**cell seeds**, and the second needs seeds to be taken across. |
| calibration draws | 20, `42` first then `0..18` | 42 is production's own pin, so it is in the grid and drawn first: a truncated array can take the last draws but not the reference point. The rest are arbitrary by construction and consecutive-from-zero says so. |
| horizon | 150 votes | one deliverable is whether the spread shrinks with vote count, which is a curve over the horizon, not a number at its end. |
| everything else | the app's own | `safe_thresholds` on (the app has no switch), production linear-SVM head, the app's per-mode blend schedule, the app's per-space `calibration_fraction` (#3290), inclusion 0. Declared to `preflight.sh --diverges calibration_seed`, which is the only knob off production. |

1,000 cells (5 classes × 5 seeds × 20 draws × 2 embedder arms), one array, one
results dir. The draws are cells rather than arms — the #3287 shape — because
#3797 gave every row a `calibration_seed` column, so an arm per draw would be
twenty arrays whose only difference is already written on every row.

## Deliverables, and what each could show

1. **The spread.** sd and range of `cost` across the 20 draws, per (class,
   geometry, cell seed) block, at fixed vote count, banded 1-25 / 26-60 /
   61-100 / 101-150. Computed on cell-**band means**, never on raw steps: 150
   steps of one trajectory are 150 readings of one draw.
2. **Against what studies already average over.** The same variance split into
   its cell-seed and calibration-draw components — a nested, balanced
   decomposition rather than two unrelated sds, because "is the split noise a
   rounding error next to the data noise, or a fraction of it?" is a question
   about components of one variance and reading it off two sds computed on
   different denominators gets the ratio wrong by the number of draws.
3. **Does it shrink with votes?** The same spread as a curve over the horizon.
   The mechanics predict it should: a bigger labelset makes any one split less
   pivotal. A measured curve says where the shallow end is.
4. **Where does the pin sit?** Draw 42's percentile among its own block's draws.
   A median near 0.5 means the shipped draw is an ordinary sample. This is the
   only part of the question a per-cell bootstrap could never see, because every
   study reuses the *same* 42.
5. **Was it the cut, or the loop?** `auroc` and `oracle_cost` are properties of
   the ranking, so they move only if the trajectories diverged; `threshold` is
   the direct effect. And the pick log — which carries the draw since this study
   — gives the literal click at which two draws first vote on different media.

## Pre-registered readings

- **The spread is not an extra error bar on published numbers.** A study that
  bootstraps over cells at the pinned draw already resamples cells that each
  carry their own realisation of it. Reporting "every number in the repo is
  ±sd" would double-count. This is written down *before* the numbers arrive
  because it is the reading the issue's framing invites.
- **What it is, is a floor on a paired contrast.** Two arms paired on (class,
  seed) are *not* paired on the split — a different knob is a different
  trajectory, so each arm draws its own realisation — and pairing, which is what
  buys those studies their precision, cannot remove it. `sqrt(2)·sd_draw/sqrt(N)`
  is the smallest SE a study of N cells per arm can have from this source alone.
  The report will quote that against #3287's own published SEs.
- **A per-geometry answer, not a pooled one.** If the region and binary
  geometries disagree about the size of the spread, the study reports both;
  one pooled number would sit between them and describe neither (#3287's own
  lesson, and #3115's before it).
- **A null is a result.** If the spread is small everywhere — say under 0.005 in
  cost at every band — then the pin costs nothing measurable, and that is the
  finding: it retires the question rather than leaving it open.

## Not in scope

The head's fit seed. The shipped head is a convex liblinear solve, so reseeding
it is measurably a no-op (0.0000 sd over 20 seeds in #3797's synthetic probe);
the retired MLP head is where a fit seed moved the ranking.

Unpinning the split in production. Nothing here proposes that, and a "best" draw
is not a result — 42 is not better than 7, it is one sample.
