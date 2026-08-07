# Does keeping the swapped Gumbel fits help? The Grid re-measure (issue #2846)

_The Visual Genome run #2846 asked for and #2875 could not do. Every number comes
from `scripts/experiments/calibration/analyze_cut.py` over
`/exp/$USER/calibration-cut2846`, plus the direct paired contrasts in
`make_cut_2846_fig.py`; the prose is written on top of those numbers. The study
being re-measured is `docs/experiments/gmm-cut/REPORT.md`._

## BLUF

**Do not ship `gumbel_any_*`, and close the Gumbel line of work.** #2875's repair
does exactly what it was built to do — the `modes_swapped` branch is gone and the
crossing fires on more steps — and it buys **nothing that matters**:

| production arm, ramp 6–20 | `gumbel_priorfree` | `gumbel_any_priorfree` |
|---|---|---|
| steps falling back to the midpoint | 24.9 % | **19.6 %** |
| paired Δ cost vs the midpoint | +0.0059 | **+0.0056** |
| paired Δ cost vs `gumbel_priorfree` | — | −0.00037 (p = 0.18, n.s.) |

The repair changes the cut on **5.4 %** of steps. Everywhere else the two
variants are the same number, so a real per-step effect is diluted to nothing.

Three findings, in descending order of how much they should change what anyone
does next:

1. **The premise of #2846 is refuted on real scores.** The issue's case was that
   the misspecification term is +0.0134 and "a Gumbel rule that actually fired
   every step is worth roughly a second helping of what #2836 delivered". The
   term is still there (+0.0130, replicated). Making the rule fire does **not**
   collect it. The whole Gumbel family remains ~0.012 behind the plain Gaussian
   `pooled_priorfree` that #2836 shipped.
2. **Where the repair fires, its sign depends on the tilt** — and the effect is
   real in both directions (pooled over both arms, conditional on the steps it
   changes): `priorfree` **−0.0117** (CI [−0.0187, −0.0045], p = 8e−4),
   `cross` **+0.0219** (CI [+0.0034, +0.0409], p = 0.015). Recovering a crossing
   is only worth having if the loss it is tilted for is the right one; giving the
   wrong-loss rule a real root just lets it be confidently wrong, and the
   midpoint fallback had been protecting it.
3. **The bench mis-sized this a second time.** At full power on the Grid
   (11 520 fits) it puts the repair at **−0.0079** excess. Real VG scores say
   −0.0004 marginal. #2846 warned that "the bench and the data disagreed here
   once already"; they have now disagreed twice, in the same direction, about the
   same family.

**And a finding this run was not looking for, which matters more than any of the
above:** the incumbent moved. See "The baseline is no longer the midpoint".

![Panel A: the share of ramp steps falling back to the midpoint drops for every
tilt and arm when the ordering guard comes off. Panel B: on the steps where the
repair changes the cut, it lowers cost under the prior-free tilt on both arms and
raises it under the cross tilt on both arms.](gumbel_any_2846.png)

## The baseline is no longer the midpoint

The harness's fidelity check — `pooled_mid` must reproduce the run's own
production threshold, the check that licenses every within-step contrast —
**failed**: max abs diff 0.2398 over 13 653 steps, against 0.0 in #2836.

It is not a harness fault and not #2875's. Splitting those steps by the
provenance the base row records:

| production took | steps | `pooled_mid` reproduces it? |
|---|---|---|
| `gmm_blend` (the path #2836 measured) | 2 158 | **yes — max abs diff 0.0** |
| `fold_anchored[*]` | 11 495 | no |

A perfect 1:1 split. `pooled_mid` is still bit-for-bit the old production
threshold; production simply stopped taking that path. Between #2836's run
(2026-08-03) and this one, `d195b004` shipped the fold-anchored threshold at
κ=1 and `196085b5` moved it to **κ=0.3 with the midpoint cut**, and `b03d54e5`
deleted the `safe_thresholds` setting so the fused path is unconditional. The
study's stated incumbent is two ship decisions stale.

So the question "does this rule beat what we ship" cannot be answered by
`vs pooled_mid` any more. Asked directly instead — every variant paired against
the **base row**, i.e. the fold-anchored κ=0.3 threshold the run actually used,
production arm, ramp 6–20, 267 cells:

| variant | mean cost | Δ vs today's production | p |
|---|---|---|---|
| `pooled_sim_oracle` *(reads labels — a bound, not a rule)* | 0.4632 | −0.0059 | 0.22 |
| **production (fold-anchored κ=0.3 mid)** | **0.4683** | — | — |
| `pooled_priorfree` *(#2836's shipped winner, unanchored)* | 0.4709 | +0.0027 | 0.58 |
| `pooled_mid` | 0.4778 | +0.0095 | 0.0012 |
| `pooled_cross` | 0.4828 | +0.0143 | 4e−6 |
| `pooled_gumbel_any_priorfree` | 0.4836 | +0.0151 | 2e−6 |
| `pooled_gumbel_priorfree` | 0.4840 | +0.0154 | 2e−6 |
| `pooled_gumbel_cross` | 0.5095 | +0.0410 | <1e−9 |
| `pooled_gumbel_any_cross` | 0.5099 | +0.0413 | <1e−9 |
| `xcal_only` | 0.5713 | +0.1019 | <1e−9 |

Read the first two rows together. **The label-reading oracle of the whole
unanchored cut-rule family is not significantly better than what production
already does without labels** (−0.0059, p = 0.22). Anchoring the mixture took
the cut-rule axis with it: there is no longer measurable headroom in "which cut
of the unanchored sim-set GMM", which is the axis #2836, #2846 and this run all
work on. That is the strongest argument for closing the line, and it is
independent of anything the Gumbel repair did.

Scope on that claim: `pooled_sim_oracle` is the oracle *of that family* — the
best cut of the unanchored sim-set mixture — not a global bound. The finding is
that beating production now requires a better **fit**, not a better **cut**.

## What the repair actually did

It removed the branch it was aimed at, exactly. Fallback attribution on the
production arm, `priorfree` tilt, **ramp window** (3 921 steps):

| guard that declined | `gumbel_priorfree` | `gumbel_any_priorfree` |
|---|---|---|
| `modes_swapped` | 518 | **0** |
| `hi_owns_lo_mode` | 407 | 407 |
| `lo_owns_hi_mode` | 27 | **335** |
| `fit_gumbel_mle_failed` | 26 | 26 |
| **total fallbacks** | 978 (24.94 %) | 768 (19.59 %) |

Of the 518 fits #2836 discarded outright, **210 now yield a crossing and 308 come
back as `lo_owns_hi_mode`** — the fit is kept, and the crossing still does not
exist in the orientation EM chose. That 210 is independently the number of steps
on which the two variants' cuts differ at all, which is the run's internal check
on this table. #2875's bench estimate of the recovery ("5 of the ~32 missing
points") was right in spirit: the ordering guard was rejecting sound fits, but
most of those fits have no root either way.

> `agg/cut_fallback_reasons.csv` is written over **all 30 steps**, not the ramp
> window — there the same split is 946 `modes_swapped` → 391 crossings + 555
> `lo_owns_hi_mode` (1 795 → 1 404 fallbacks of 7 365 steps). Don't mix its
> counts with the ramp-window rates above.

`gumbel_any_cross` is worth noting as the one place the repair makes a variant's
fallback rate *worse-behaved*: it fires on 2.0 % more steps and loses on them.

### The fit was never the problem

| production arm | image geometry | pooled geometry |
|---|---|---|
| EVT fit succeeds | 97.0 % | 98.7 % |
| EM puts the Gumbel on the **low** mode | 80.1 % | 87.0 % |

So EM orients the Gumbel upward on 13 % of pooled production fits — close to the
14.2 % `modes_swapped` share #2875 measured on the bench, which is a clean
replication of that diagnosis on real data. #2836's ordering premise was wrong,
#2875's reading of why was right, and fixing it still does not pay.

## The theory bench, at full power, still says the opposite

The bench half ran too (`reps=40`, 11 520 fits, 47 min), with #2875's
survivorship fix in place so every rule is scored on every replicate at the
midpoint it would fall back to. Production-like corner, m = 24, prevalence ≤ 0.05
(2 880 fits), mean excess true rate loss:

| rule | fire rate | excess (honest) | excess (conditional on firing) |
|---|---|---|---|
| `mid` | 100 % | 0.1824 | 0.1824 |
| `priorfree` | 100 % | 0.1883 | 0.1883 |
| `cross` | 100 % | 0.2159 | 0.2159 |
| `gumbel_priorfree` | 67.6 % | 0.1547 | 0.1151 |
| **`gumbel_any_priorfree`** | **71.7 %** | **0.1468** | 0.1106 |
| `gumbel_cross` | 37.3 % | 0.1823 | 0.1465 |
| `gumbel_any_cross` | 38.2 % | 0.1811 | 0.1446 |
| `sim_oracle` *(bound)* | 100 % | 0.0397 | 0.0397 |

This replicates #2875's local probe (which had 0.1585 → 0.1487 on 1 440 fits;
0.1547 → 0.1468 here) and reproduces #2836's `mid` at 0.1824 against its
published 0.1824. The bench is measuring what it was measuring before.

**It just keeps being wrong about this family.** In the bench the Gumbel rules
*beat the Gaussian ones* — `gumbel_any_priorfree` 0.1468 against `priorfree`
0.1883 — and the repair is worth −0.0079. On real VG scores the ordering is
reversed (the Gumbel family loses to `priorfree` by ~0.012) and the repair is
worth −0.0004. Both halves are now measured at full power with the survivorship
bug fixed, and they still disagree, so the disagreement is not a measurement
artefact this time: **the generative model of region voting the bench samples
from is not what VG scores look like.** #2836's prediction (4) already said the
Gumbel is not capturing "a max over region nodes"; this is the same conclusion
arriving from the other direction.

One thing the bench does get right: it puts `evt_gumbel_is_low` at 87.8 % in the
corner, against 87.0 % measured on the real pooled production geometry. The
*diagnosis* transfers; the *value of fixing it* does not.

## The pre-registered recomputation, discharged

`REPORT.md` flagged its tail-alpha table as needing recomputation, because both
numbers excluded the ~14 % of fits #2836 discarded. Those fits are now kept and
`lo_survival` reads whichever component is the low one. Over the same 511 cells:

| tail model | median α (#2836 → now) | IQR ratio (#2836 → now) | stable (bar: < 3) |
|---|---|---|---|
| Gaussian low component | 0.069 → **0.070** | 4.93 → **5.54** | no |
| Gumbel low component | 0.165 → **0.158** | 2.22 → **2.38** | **yes** |

**The finding survives.** The EVT tail level is still the stable one, and
"cut the Bad tail at α ≈ 0.16" is still a coherent one-constant rule that needs
no crossing at all — which is now the only part of the EVT work with a future,
given that the crossing rules do not clear the plain Gaussian `priorfree`, let
alone production.

> A reading trap: `summary_cut.json`'s `decisions.tail_alpha_stable: false`
> refers to the **Gaussian** row (`analyze_cut.py` keys it off
> `oracle_lo_sf_gauss`). It is not saying the EVT rule is unstable.

## The decomposition replicates

Production arm, ramp 6–20, excess-cost units, against #2836's run:

| term | #2836 | this run |
|---|---|---|
| prior / loss mismatch | +0.0132 | +0.0127 |
| component identification | −0.0074 | −0.0067 |
| Gaussian misspecification | +0.0134 | +0.0130 |
| finite-sim-set transfer | +0.0406 | +0.0407 |
| total (`cross` → test oracle) | +0.0623 | +0.0623 |

Four independent terms to within 0.0007 on a re-run whose trajectories differ
(the production threshold changed, so the steps are not the same steps). This is
the run's own evidence that the harness and the decomposition are sound, which is
what lets the negative result above be read as a result rather than as breakage.

The misspecification term is real, replicated, and **still unclaimed**. What this
run rules out is one particular way of trying to claim it.

## Scope and what would falsify this

- Base dev `4f267306`, branch `claude/gumbel-any-remeasure-2846`, worktree
  `/exp/$USER/projects/vts-cut2846`, experiment `/exp/$USER/calibration-cut2846`
  (cells array 474359, theory 474358, analyze 474360).
- VG region voting on the production linear head, inclusion 0, 30 votes deep,
  12 seeds, 552 cells (511 non-empty — the ~17 % that emit no rows are rare
  small-object categories on the siglip arm, and the split is pre-vote and
  deterministic, so it is symmetric across variants). 552/552 cells COMPLETED,
  0 failures, 0 zero-byte outputs; nothing was dropped from the analysis.
- At inclusion 0 the cost weights are (1, 1), so `rate` and `priorfree` are the
  **same rule** here — every `*_rate` row in the tables above is identical to its
  `*_priorfree` sibling by construction, not by coincidence.
- Every contrast is within-step, so none of it sees acquisition feedback.
- **The one result that could flip with more data** is finding 2's `priorfree`
  conditional effect: it is genuine (p = 8e−4 pooled) but it fires on 5 % of
  steps, and a variant that helps a little on 5 % of steps and is invisible
  marginally is not a shipping case. If the fallback rate could be driven to
  near zero — which would take fixing `hi_owns_lo_mode`, the collapsed-fit
  branch, not the ordering one — the marginal effect could become measurable.
  That is a different repair from the one #2846 asked for, and given the table
  in "The baseline is no longer the midpoint" it would still be aiming at an
  axis with no headroom left.

## Recommendation

1. **Close #2846.** The question it asked is answered: the guard was wrong,
   removing it is principled, and on real scores it is worth −0.0004 (n.s.).
   Keep `gumbel_any_*` as the measured variant it is — it is strictly the more
   correct of the two implementations — but do not promote it, and do not spend
   more on the crossing.
2. **Update `docs/experiments/gmm-cut/REPORT.md`'s fidelity claim**, which is now
   stale in a way that would mislead the next reader into thinking the harness
   broke.
3. **The live question is no longer "which cut".** Production's fold-anchored
   estimator has closed the gap to that family's own labelled oracle. Anything
   further on this path should target the **fit** — and `transfer`, at +0.041 and
   two thirds of the total, is where the cost still is, exactly as #2836 said.
