# The one-constant EVT tail-α cut (#2881) — the negative, and the close of the EVT line

> # ⚠️ SEEDING CAVEAT — these runs did not start the way the app does
>
> **Recorded 2026-08-26 (#3156).** Autopilot seeds its first three Good votes from
> a **text sort**: the user types a query and votes down that ranking. Until
> PR #3269 this harness instead ranked every item by cosine to a **crop of one
> boxed positive** — a ranking no user ever produces — and passed it as
> `seed_scores`, the argument that `al_strategies`, `EVAL.md` and
> `voting_iterations` all describe as "similarity to the **typed query**".
>
> **What to distrust here:** anything that depends on *how a run starts* —
> positive starvation, stuck or never-got-going runs, `n_good`, and
> early-trajectory cost. Measured on one cell after the fix, text seeding put the
> first positive at **rank 1** with five in the top 20, while the exemplar that
> crop-seeding made look like the dataset's hardest positive ranked **4006 of
> 7749** for its own class.
>
> **What still holds:** within-study contrasts where every arm seeded identically,
> which is most of what these reports conclude — the seeding is a shared baseline
> shift, not an arm-dependent one.
>
> See [the harness seeded from a crop](../../../scripts/experiments/lessons/2026-08-26-the-harness-seeded-from-a-crop.md).


**Run 2026-08-12 · base dev `07deb9a63` · branch `run/tail-2881` · GRID worktree
`/exp/sgreenberg/projects/vts-calib` · experiment
`/exp/sgreenberg/calibration-tail2881` · SLURM 495382 (552 cells, 0 failures) +
495383 · 338,931 variant rows, 27,306 cut-diagnostic rows · pre-registration:
[`PREREG-2881.md`](PREREG-2881.md)**

## BLUF

**The tail-α rule is significantly *worse* than production, on both arms, and the
EVT cut line closes with it — as pre-registered.**

The prediction was a null: `mean_d_cost` inside ±0.005 with p > 0.05. The measured
result missed that band **on the wrong side** — the rule is worse than the
threshold production already computes, at p = 0.001:

| ramp 6–20, vs the run's own base row | Δ cost | p | n cells |
|---|---:|---:|---:|
| `pooled_tail_a158` — region (`dinov3_patch/max_patch`) | **+0.0069** | 0.00096 | 267 |
| `pooled_tail_a158` — binary control (`siglip/whole_image`) | **+0.0058** | 0.014 | 233 |
| `pooled_priorfree` (region) | +0.0001 | 0.38 | 267 |
| `pooled_sim_oracle` *(reads labels — the family's own bound)* | −0.0055 | 0.071 | 267 |

**All four pre-registered criteria for a positive fail**, including the one that
was expected to pass:

| # | Criterion | Result |
|---|---|---|
| 1 | α = 0.158 beats the base row at p < 0.05 | ✗ — *worse*, +0.0069 at p = 0.001 |
| 2 | Cost curve flat by the ≥2× α-span bar | ✗ — the 1-SE band holds one level, ratio 1.0 |
| 3 | `pooled_tail_a158` in `closest_to_oracle_tied` | ✗ — that is `[pooled_priorfree, pooled_rate]` |
| 4 | No significant regression on the `whole_image` control | ✗ — +0.0058 at p = 0.014 |

The reopen condition — *"a flat curve with a negative mean that misses
significance"* — is decisively absent. Per the pre-registration: **close #2881,
and close the EVT cut line with it.** The `tail_a*` rules stay as measured
variants, exactly as `gumbel_any_*` was kept; they are not promoted.

## What makes this a clean close rather than another ambiguous one

**The fallback prediction landed, and it landed harder than predicted.** Every
previous EVT contrast was diluted by a 20–25 % midpoint-fallback rate, and that
dilution was always the available excuse for a non-result. The tail rule has no
crossing to fail, so it has no orientation decline — and the measurement shows
exactly that, on the ramp window the 24.9 % / 19.6 % figures came from:

| rule (ramp 6–20) | fallback rate, region / binary | reasons seen |
|---|---:|---|
| `pooled_tail_a158` | **0.7 % / 1.5 %** | `fit_gumbel_mle_failed` **only** |
| `pooled_priorfree` / `pooled_rate` | 5.4 % / 6.6 % | — |
| `pooled_gumbel_any_cross` | 47.6 % / 74.3 % | `lo_owns_hi_mode` 77 %, `hi_owns_lo_mode` 21 % |
| `pooled_gumbel_cross` | 49.6 % / 75.3 % | — |

Zero `hi_owns_lo_mode` / `lo_owns_hi_mode` on the tail rules, and the residual is
*below* the 1.3–3.0 % EVT fit-failure floor the pre-registration predicted. So the
rule is measured at essentially full strength on essentially every step — and
still loses. There is no dilution left to blame.

## The stability finding was real, and it was never actionable

#2836 and #2846 both found the oracle cut sits at a stable tail level on the
Gumbel low component (median α 0.158, IQR ratio 2.38 over 511 cells). This run
splits that finding cleanly in two:

- **Location: it transferred.** Across seven levels the argmin is **exactly the
  pre-registered 0.158** (`decisions.tail_alpha_curve.best_alpha`). Nothing was
  fitted to this run; the constant carried over from a different study's cells.
- **Cost: it did not.** The curve around that argmin is steep, not flat. Only
  α = 0.158 itself falls within one standard error of the best, so the flat band
  spans a factor of 1.0 in α against a pre-registered bar of ≥ 2.

| α | cut offset | Δ cost vs base (region, ramp) | p | cells improved |
|---:|---|---:|---:|---:|
| 0.04 | `loc + 3.20·scale` | +0.0392 | 7e−17 | 27 % |
| 0.08 | `loc + 2.48·scale` | +0.0173 | 3e−07 | 37 % |
| 0.11 | `loc + 2.15·scale` | +0.0105 | 0.0004 | 40 % |
| **0.158** | `loc + 1.76·scale` | **+0.0069** | 0.001 | 38 % |
| 0.22 | `loc + 1.39·scale` | +0.0102 | 1e−07 | 36 % |
| 0.30 | `loc + 1.03·scale` | +0.0219 | 6e−18 | 26 % |
| 0.40 | `loc + 0.67·scale` | +0.0443 | 8e−29 | 17 % |

That is the whole answer to why a stable diagnostic never became a good rule:
**"the oracle cut sits at α ≈ 0.16" is a claim about where the optimum is, not
about what it costs to aim there**, and the two come apart when the valley is
steep. Even at the bottom of its own U, the rule is worse than production —
and no level in the sweep is better, so this is not a matter of picking a
different α.

## The successor question is unchanged, and this run re-measured it

Two numbers reproduce the prior reports and both point at #2883:

- **`family_headroom_exhausted: True`.** `pooled_sim_oracle` — the empirical
  rate-loss minimiser over the sim scores read with true labels, which bounds
  *every* rule that picks a threshold from that set — is −0.0055 against
  production at **p = 0.071**. #2879 measured −0.0059 at p = 0.22. The
  label-reading oracle of the whole unanchored cut family still does not clear
  what production already does without labels.
- **`dominant_error_term: transfer`**, at 0.0389 of a 0.0686 total — **57 %**.
  `prior_loss` 0.0111, `misspecification` 0.0129, `identification` 0.0057.

Beating production needs a better *fit*, not a better cut. That is #2883.

## Two reading notes for this output

- **`pooled_mid` production-sanity says `ok: false`; that is expected, not a
  broken run.** That check reconstructs #2836's era production rule, and
  production has since moved to `mid_tilt` at κ = 0.3 — it mismatches on 11,364
  of 11,366 `fold_anchored[2/2]` steps and matches exactly on all 2,158
  `gmm_blend` steps, which is precisely the shape of a stale reconstruction
  rather than a defect. Every contrast in this report is paired within-step
  against the run's **own base row** (`base_provenance` recorded per row), as the
  pre-registration requires. `beats_midpoint` gates nothing.
- **The deeper `pure_gmm_2_5` window is worse still** for the tail rule (+0.0257
  region / +0.0202 binary) while `pooled_priorfree` is *better* there (−0.0075 /
  −0.0113). The tail rule's deficit is not a ramp-window artifact.

## Environment notes

This was the first study to run off the shared pile (#3121):
`VTSEARCH_DATA_DIR=/expscratch/sgreenberg/vts-cache/datadir`, since the
launcher's historical `/exp/$USER/max-patch/datadir` was archived in the
2026-08-12 cleanup. Prepare was reused from the archived
`calibration-safe-linear` run, restaged with its exemplar-crop symlinks
repointed at the archive's real files — the archived symlinks dangle into the
deleted `/exp/max-patch`, and the launcher's `readlink -f` would have recreated
them dangling without error. See `scripts/experiments/LESSONS.md`.

## Reproduction

```bash
CALIB_CONC=32 VTS_REPO=<worktree> \
  source scripts/experiments/pile/pile_env.sh && \
  bash scripts/experiments/calibration/launch_tail_2881.sh
```

Outputs: `agg/cut_contrasts_vs_base.csv` (the contrast table above),
`agg/cut_tail_alpha_curve.csv` (the α sweep), `agg/cut_fallback_reasons.csv`
(`window == "ramp_6_20"` rows), `summary_cut.json` (`decisions`).
