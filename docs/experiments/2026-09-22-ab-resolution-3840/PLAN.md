# Pre-registration: the validation grid for the A/B cells-to-resolution curve (issue #3840)

Written and committed **before** the validation grid was submitted, so the
prediction below cannot have been fitted to the result.

## What is being predicted

The trajectory A/B pairs two whole grids on (environment, category, seed), takes
each cell's mean cost over the app-visible steps, and reports the paired mean Δ
with SE = sd(Δ) / √n. The curve in this study is built from the 114 paired cells
of #3585 and #3825 that already exist. It says the SE falls as σ/√n, with σ the
cell-to-cell sd of Δ **for that pair of arms**.

For #3825's own pair — the parameter rule it replaced (`baseline`) against the
rule it shipped (`ll1e-8`) — the 114 existing cells give **σ = 0.040** (bootstrap
90% interval 0.030–0.048).

The validation grid is the same two arms, the same runner
(`run_cells_arm_3825.py`), the same eight environments, at seeds 2–8: **399 new
paired cells** the curve never saw. 399 is the size the curve says resolves a
Δ of **0.004** at two standard errors (N = (2σ/δ)² = 390, rounded up to a whole
seed).

## Predictions, fixed now

1. **SE of the paired mean Δcost over the 399 new cells = 0.0020**, with a 90%
   interval of **0.0015–0.0024** carried from the uncertainty in σ.
   *Pass:* the observed SE lies in that interval. *Fail:* it lies outside, and
   the curve is not a design tool at this n.
2. The observed sd of Δ over the new cells lies in 0.030–0.048 (same statement,
   per cell rather than per grid).
3. **No prediction about the sign or size of Δ itself.** #3825 measured
   +0.0049 ± 0.0037. If that is the true effect, 399 cells resolve it at about
   2.5 SE; if the true effect is zero, they do not. Either is a result; the
   validation is of the *resolution*, not of the arm.

## Two checks that ride on the same submission

- **Code drift.** Seeds 0–1 are re-run on today's `dev` for both arms. Against
  the 2026-09-13 grids they say whether eleven days of merges moved a
  trajectory. The curve assumes a grid's cells are the same cells whenever they
  are run.
- **Determinism.** `rep` runs `baseline` seeds 0–1 a second time on the same
  code. If any of its cells differs from the first run, part of every A/B's σ is
  harness noise rather than the arm, and the curve would be describing that.

## What is not changed

Environments, steps (100), `CALIB_REQUIRE_OPENING=mixed`, safe thresholds on,
BLAS pinned to one thread, the prepare reused from
`/expscratch/sgreenberg/bench-overview/results`: all as #3825 ran them, via
`launch_3825.sh` unmodified. Results root `/expscratch/sgreenberg/abres-3840`.
