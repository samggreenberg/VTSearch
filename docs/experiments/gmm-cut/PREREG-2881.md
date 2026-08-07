# Pre-registration: the one-constant EVT tail-α cut (issue #2881)

**Written before the run.** Nothing below may be edited after the cells array is
submitted; the run's findings go in a separate `REPORT-2881.md`. The point of
writing it down is that this family has now been mis-sized twice by the theory
bench in the same direction (#2836, #2846), and a prediction recorded afterwards
is not a prediction.

## BLUF — the prediction

**The tail rule will not beat production, and the EVT line closes with it.**

Concretely, at the pre-registered α = 0.158, against the run's own base row on
the production arm's ramp window: `mean_d_cost` **inside ±0.005 with p > 0.05**.
That is the honest prior, and it is the outcome this run is expected to produce.

The reasoning is not about this rule specifically. `pooled_sim_oracle` is the
empirical rate-loss minimiser over the sim scores *read with true labels*, with
no parametric form at all, so it upper-bounds every rule that picks a threshold
from that sim set — a Gumbel tail quantile included. #2846 measured it at
−0.0059 against production, p = 0.22: **the whole unanchored cut family already
has no measurable headroom left.** A tail-α rule is still a cut of the same
unanchored fit.

What makes it worth one run anyway: it is cheap (closed form, no extra fit, it
rides along in the existing cells array), it is the only EVT variant whose
fallback rate is not the thing killing it, and it is the one piece of the family
the data has twice declined to refute.

## The rule

`tail_a<α>` cuts where the fitted **Bad** component still has α of its mass above
the cut — the inverse of the `oracle_lo_sf_evt` diagnostic that #2836 and #2846
both recorded but neither measured *as a rule*. One constant. No `lam`, no tilt,
no crossing, and no orientation question.

Solved on the **logit axis**, where the EVT fit lives, then squashed back
(`vtscore/eval/cut_rules.py`, `tail_cuts`). Closed form in both orientations
(`GumbelNormalFit1D.lo_quantile`):

| the low component is | cut |
|---|---|
| the Gumbel | `loc − scale·ln(−ln(1−α))` — at α = 0.158, `loc + 1.761·scale` |
| the Normal (swapped fit) | `mu + sd·Φ⁻¹(1−α)` |

It branches exactly as `lo_survival` does, so a swapped fit reports the Normal's
tail rather than a plausible number read off the wrong component.

## Why the fallback rate should collapse

Every `hi_owns_lo_mode` / `lo_owns_hi_mode` decline in #2846's fallback table
exists because a *crossing* may not exist between the modes. A tail quantile
always exists for a non-degenerate fit. **Prediction: the tail rules' fallback
rate goes to the EVT fit-failure rate alone (1.3–3.0 %)**, against the 24.9 % /
19.6 % the crossing rules carried.

Read this off `agg/cut_fallback_reasons.csv`, **`window == "ramp_6_20"` rows** —
those 24.9 % / 19.6 % figures are ramp-window rates, and the file also carries an
`all_steps` row set that is a different population.

If the tail rules' fallback rate is *not* near-zero, the implementation is wrong,
not the hypothesis. That is a harness failure, not a finding.

## The α sweep, and what it is for

Seven levels: **0.04, 0.08, 0.11, 0.158, 0.22, 0.30, 0.40**, spaced so the *cut*
moves in near-even steps (`loc + {3.20, 2.48, 2.15, 1.76, 1.39, 1.03, 0.67}·scale`)
rather than α, which is logarithmic in the cut.

The stability finding says the oracle cut sits at a stable *level* (median 0.158,
IQR ratio 2.38 over 511 cells). That is a claim about where the optimum **is**,
not about what it costs to aim there, and the two come apart if the cost curve is
steep. So the claim actually being tested is **flatness**:

> **Pre-registered bar.** The α levels whose cost is within one standard error of
> the best level's must span a factor of **≥ 2** in α. A constant that can be
> wrong by 2× and still land inside the noise will transfer to another dataset;
> one that cannot is a number fitted to this run.

Reported as `decisions.tail_curve_is_flat`, with the curve in
`agg/cut_tail_alpha_curve.csv`. Its sibling
`decisions.tail_preregistered_alpha_in_flat_band` says whether 0.158 is still
*inside* that band — membership, not equality with the argmin, because demanding
the exact argmin would fail on noise alone. If 0.158 falls outside the band, the
constant does not carry across runs, which is the same conclusion as a steep
curve by another route.

## The sweep does not get seven shots at the ship gate

**Only α = 0.158 is a ship candidate.** The other six levels are measured and
reported but are excluded from `best_by_cost`, `best_vs_production`,
`closest_to_oracle` and the ship gate (`analyze_cut.SWEEP_ONLY`, named in
`decisions.sweep_only_variants`).

A sweep varies a free parameter. Handing all seven levels to a two-sided 5 % bar
buys roughly a 30 % chance of a "winner" from noise alone, and this study line has
already paid twice for a wrong-but-plausible read. If the curve's minimum lands
somewhere other than 0.158, that is a finding for the *next* pre-registration —
evidence that the constant does not transfer — not a rule to ship off this run.

## The baseline

Paired **within-step against the run's own base row** — the threshold production
actually used on that step, whatever path it took — from
`agg/cut_contrasts_vs_base.csv`, with `base_provenance` recorded alongside.
`pooled_priorfree` and `pooled_sim_oracle` come out of the same table for
continuity with the two prior reports.

Not `pooled_mid`. That variant *reconstructs* the production rule of #2836's era
and by #2846 it reproduced the shipped threshold on only 16 % of steps; the
`beats_midpoint` column survives as the historical #2836 contrast and gates
nothing.

## What would count as a positive

All four, on the production arm's ramp window:

1. α = 0.158 beats the base row: `mean_d_cost < 0` at p < 0.05.
2. The curve is flat by the bar above — otherwise the constant is this run's, not
   a constant.
3. `pooled_tail_a158` is in `closest_to_oracle_tied`. A rule that is right on
   average and wrong step by step is not a rule.
4. No significant regression on the `whole_image` control arm, which
   `calculate_gmm_threshold` also serves through the cosine/text sort.

Anything less is the negative result predicted above.

## What closes the line

If α = 0.158 lands inside the noise — **close #2881, and close the EVT cut line
with it.** Not "try another α", not "try another tail model": the bound in the
BLUF says the axis is exhausted, and #2883 already carries the successor
question (aim at the *fit*, where `transfer` is +0.041 and two thirds of the
total). Keep the `tail_a*` rules as measured variants, as `gumbel_any_*` was
kept; do not promote them.

The one result that would reopen it is a flat curve with a *negative* mean that
misses significance — that is an underpowered measurement rather than a null, and
it would justify one larger run at α = 0.158 alone.

## Scope

- Rides along inside the existing `CALIB_SAFE_THRESHOLDS=1` cells array
  (`launch_tail_2881.sh`): same arms, sizing and seeds as #2836/#2846, so the
  contrasts stay comparable to the numbers those reports produced. No extra fit
  per step — the tail rules read the EVT fit that is already computed.
- The theory bench is **not** run. Its `CANDIDATE_RULES` list does not include
  the tail family, deliberately: the bench mis-sized this family twice in the
  same direction, and #2846's recommendation was to stop trusting it here.
- At inclusion 0 the cost weights are (1, 1), so `rate` and `priorfree` are the
  same rule in the comparison tables — as in both prior reports. The tail rules
  have no cost-weight dependence at all, which is a difference worth remembering
  when reading them next to the tilted ones.
- Every contrast is within-step, so none of it sees acquisition feedback.
