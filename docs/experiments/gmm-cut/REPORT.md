# Which cut does the GMM path want? (issue #2836)

_An Autopilot simulation study on the HLTCOE Grid, plus a synthetic bench with a
known ground truth. Every number below comes from
`scripts/experiments/calibration/analyze_cut.py` and `theory_bench.py`; the prose
is written on top of those numbers. Design and pre-registered predictions:
`docs/plans/gmm-cut-theory-experiment.md`._

## BLUF

**Ship the prior-free crossing** — `_weighted_gaussian_crossing` at
`lam = (fnr_weight/fpr_weight)·(w_lo/w_hi)` instead of `lam = 1`. On the
production `dinov3_patch × max_patch` arm over the 6–20-vote ramp, 267 paired
cells:

| vs the shipped midpoint | Δ | p | cells improved |
|---|---|---|---|
| blended cost | **−0.0044** | 1.0e−11 | 67 % |
| raw cut cost (the rule itself, un-blended) | **−0.0084** | 2.6e−08 | — |
| FPR | −0.0024 | — | — |
| FNR | −0.0021 | — | — |

Three things make this a clean recommendation:

1. **It is not a metric trade.** FPR *and* FNR both fall. The recurring failure
   mode in #2790/#2799 was a cut that buys one error with the other; this is the
   opposite of that, and it is why #2798's crossing lost (+0.0034 cost here, from
   −0.0112 FPR against +0.0147 FNR).
2. **It wins as a rule, not via the blend.** The un-blended cut improves
   *twice* as much as the blended threshold (−0.0084 vs −0.0044). A rule that
   only won after blending would be winning by sitting nearer the conformal
   threshold; this one does not.
3. **It captures 60 % of what is achievable.** The label-reading `sim_oracle`
   — which cheats, re-cutting the same fits with ground truth — gets −0.0073.
   The prior-free crossing recovers −0.0044 of that without labels.

It supersedes **both** #2801 and #2833: a principled crossing with the priors
correctly divided out, rather than a heuristic that only approximates it when the
component variances happen to match.

**Fidelity.** `pooled_mid` reproduced the run's own production threshold exactly
over **13 653 steps** (max abs diff 0.0), and `pooled_cross` reproduces #2799's
headline (+0.0034 here vs +0.0036 there). The harness is measuring the shipped
code and the same phenomenon.

> **Stale as of 2026-08-07.** That fidelity claim held against the production
> path *of this run*. Production has since moved to the fold-anchored threshold
> at κ=0.3 (`d195b004`, `196085b5`, `b03d54e5`), so `pooled_mid` is no longer
> what the app computes and the check now fails by construction on the steps that
> take the new path — bit-for-bit exact on the ones that don't. Everything below
> remains a valid comparison *between cut rules*; nothing below still establishes
> that a rule beats **what ships**. See `REMEASURE-2846.md`.

## What the prior-free crossing is

### The variables

The safe-threshold path fits a 2-component Gaussian mixture to the **unlabelled**
sim-set scores — every media in the pool, scored by the current detector, with no
labels involved:

```
f(x) = w_lo · N(x; mu_lo, var_lo)  +  w_hi · N(x; mu_hi, var_hi)
```

| symbol | code | what it is |
|---|---|---|
| `x` | — | one media's score, a sigmoid output in [0, 1] |
| `w_lo`, `w_hi` | `GmmFit1D.w_lo/.w_hi` | mixture weights, summing to 1: the share of the fitted distribution EM assigns to each component |
| `mu_lo`, `mu_hi` | `.mu_lo/.mu_hi` | the component means — `lo` is the Bad (low-score) mode, `hi` the Good one |
| `var_lo`, `var_hi` | `.var_lo/.var_hi` | the component variances |
| `fpr_weight`, `fnr_weight` | `inclusion_weights(k)` | the cost of a false positive vs a missed match. `(1, 1)` at Inclusion 0; raising Inclusion raises `fnr_weight` |
| `lam` | `crossing(lam=…)` | the tilt this study added: it multiplies the Good side of the equation |

Note that `w_lo` and `w_hi` are properties of a **curve fitted to unlabelled
scores**. Nothing in the fit knows which media are true matches.

### The rule we shipped in #2798, and what it optimises

```
w_lo · N_lo(x)  ==  w_hi · N_hi(x)
```

Divide both sides by `f(x)` and this is the textbook Bayes boundary: the score at
which `P(Bad | x) == P(Good | x)`, with `w_lo`/`w_hi` playing **class priors** and
`N_lo`/`N_hi` playing **class-conditional densities**. Cutting there minimises the
expected *number* of mistakes:

```
count loss  =  P(Bad) · FPR  +  P(Good) · FNR
```

Each error is weighted by how *common* its class is. That is a coherent
objective — it is simply not ours.

### What we actually score

```
cost = fpr_weight · FPR + fnr_weight · FNR
```

`FPR` divides by the number of negatives and `FNR` by the number of positives, so
each error is normalised by **its own** class. The objective therefore does not
depend on the class balance at all — deliberately, because that is what makes
Inclusion ("what fraction of true matches am I willing to miss") mean the same
thing on a dataset that is 1 % positive and one that is 30 % positive.

### The derivation

Write the rates as integrals over the true class-conditional densities:

```
FPR(tau) = integral of f_neg from tau to +inf     (negatives landing above the cut)
FNR(tau) = integral of f_pos from -inf to tau     (positives landing below it)
```

Differentiate the cost and set it to zero — the cut sits where moving it stops
helping:

```
dL/dtau  =  -fpr_weight · f_neg(tau)  +  fnr_weight · f_pos(tau)  =  0

    =>   fnr_weight · f_pos(tau)  ==  fpr_weight · f_neg(tau)
```

**No priors appear.** They cannot: they were never in the objective. Contrast the
count loss, whose stationarity condition is `P(Good)·f_pos == P(Bad)·f_neg` —
there the priors appear precisely because they are in the loss.

Now identify the fitted components with the classes (`f_neg = N_lo`,
`f_pos = N_hi`) and ask what `lam` turns the implemented solve into the correct
one. We need `N_lo(tau) == (fnr_weight/fpr_weight) · N_hi(tau)`, while the code
solves `N_lo(tau) == lam · (w_hi/w_lo) · N_hi(tau)`, so

```
lam = (fnr_weight / fpr_weight) · (w_lo / w_hi)
```

The `w_lo/w_hi` factor exists for one purpose: to **cancel** the prior odds the
original equation smuggled in. At Inclusion 0 the cost weights are equal and this
is simply `lam = w_lo / w_hi`.

### Why the old cut was biased, and by how much

With equal variances the family has a closed form:

```
x(lam) = midpoint  +  var · ln( w_lo / (lam · w_hi) ) / (mu_hi - mu_lo)
```

- **`lam = 1`** (#2798): the offset is `var·ln(w_lo/w_hi)/(mu_hi-mu_lo)`. The pool
  is overwhelmingly negative, so `w_lo >> w_hi`, the log is large and positive,
  and the cut sits **well above** the midpoint — admitting too little.
- **`lam = w_lo/w_hi`** (prior-free): the log term becomes `ln(1) = 0` and the cut
  **is exactly the midpoint**.

That is why the historical midpoint was so hard to beat: under equal variances it
*is* the rate-optimal rule, not a heuristic that got lucky. And it is why a third
point exists — with unequal variances the two separate, and the gap is the part of
the correct rule the midpoint cannot express.

### The robustness argument, which turned out to matter

Cancelling the weights does more than fix the loss: it makes the cut **immune to
the weights being wrong**. After substituting `lam`, the equation is
`fnr_weight·N_hi == fpr_weight·N_lo` — `w_lo` and `w_hi` are gone, so the cut
depends only on the four component moments and the two cost weights.

This matters because the mixture weights are *not* class priors, and this study
measured how badly: on the control arm the fitted `w_hi` averaged **0.35 against a
true positive prevalence of 0.09**. EM's high component is "media with a
confidently-scored region", not "true matches". The count-optimal cut is therefore
wrong twice over — the wrong loss, corrected by a prior it has mis-estimated by
4x — while the prior-free cut never consults that number at all.
`test_prior_free_crossing_does_not_depend_on_the_mixture_weights` pins the
invariance to 1e-12.

## Why: the derivation was right, but it is not the dominant error

Decomposing today's gap on the production arm, in **excess-cost** units
(n = 3921 steps):

| term | cost | share |
|---|---|---|
| prior / loss mismatch | +0.0132 | 21 % |
| component identification | −0.0074 | — |
| Gaussian misspecification | +0.0134 | 22 % |
| **finite-sim-set transfer** | **+0.0406** | **65 %** |
| total (`cross` → test oracle) | +0.0623 | |

The prior term is real and worth fixing — that is the shipped change — but
**about two thirds of the remaining gap is sim→test transfer, which no cut rule
can address.** Cut-rule work has roughly 0.026 of headroom, not 0.062. That is
the most useful number in this study for planning purposes.

### The pre-registered predictions

- **(1) The offset identity — CONFIRMED.** `tau_cross − tau_mid` tracks the
  closed form `var·ln(w_lo/w_hi)/(mu_hi−mu_lo)`: on the image geometry
  corr 0.946, mean abs residual 0.0020. On the pooled geometry it degrades
  (corr 0.705, residual 0.0083) — exactly as it must, since the closed form
  assumes equal variances and max-pooling is what breaks that. The algebra in the
  issue is correct.
- **(2) The penalty scales with the offset — FAILED.** Across offset quintiles
  the crossing's penalty goes +0.0004, +0.0035, +0.0051, +0.0016, **−0.0063**:
  non-monotone, and *negative* at the largest offsets. The crossing's damage is
  not proportional to the prior-odds term, so the pooled-vs-image inversion is
  **not** explained quantitatively by offset size.
- **(3) The prior-free crossing beats both incumbents — CONFIRMED.** See BLUF.
- **(4) The EVT gain is geometry-specific — FAILED, in both halves and in
  opposite directions.** On real data the Gumbel's likelihood gain is *larger* on
  the image geometry (+0.0041) than the pooled one (+0.0014). In the synthetic
  bench it *decreases* with region count (m=1: +0.0327, m=6: +0.0196, m=24:
  +0.0128). Whatever the Gumbel low component buys, it is **not** "a max over
  region nodes is an extreme-value statistic". #2798's geometric premise does not
  survive measurement, even though its algebra does.

## The theory bench disagreed with the data because it was scoring a different thing

> **The bench numbers below are superseded.** The reconciliation this section
> originally offered — "the crossing is fragile on real scores and reliable in
> the bench" — was wrong in its second half, and the first half was the wrong
> diagnosis. See "What #2846 found" immediately after. The table is kept because
> the corrected numbers are only meaningful against it.

The bench generates scores from a model of region voting whose class-conditionals
are known in closed form, so each rule's excess loss is measured against **truth**
rather than a held-out sample (11 520 fits).

In the production-like regime (m = 24, prevalence ≤ 0.05), mean excess true rate
loss **as originally reported**:

| rule | bench | | real data (ramp cost) |
|---|---|---|---|
| `gumbel_priorfree` | **0.1151** (wins 56 % of configs) | | 0.5165 (**loses**, +0.0043) |
| `gumbel_cross` | 0.1465 | | 0.5294 (**loses**, +0.0172) |
| `mid` | 0.1824 | | 0.5121 |
| `priorfree` | 0.1883 | | **0.5075 (wins)** |
| `cross` | 0.2159 | | 0.5156 |

The real-data fallback rates are sound: the Gumbel crossing does not exist on
**27 %** of steps (`gumbel_priorfree`) and **53 %** (`gumbel_cross`), where the
rule silently degrades to the midpoint. What was wrong was the claim that the
bench's samples are clean enough that the root essentially always exists.

### What #2846 found

**The bench column is conditional on the rule firing; the real-data column is
not.** `theory_bench.py` wrote `NaN` for a rule with no root, and every
aggregation (`groupby.mean`, `idxmin`) skips NaN by default — so the EVT rules
were scored only on the replicates where they applied, while `mid` and
`priorfree` were scored on all of them. The real-data harness does the honest
thing: it falls back to the midpoint and scores that, flagged in `cut_fallback`.
The two halves were reporting different estimands.

The bench's samples are not clean. Re-running the production-like corner with
every replicate scored (1 440 fits; `mid` reproduces at 0.1830 and the old
conditional `gumbel_priorfree` at 0.1129, so this is the same measurement):

| `gumbel_priorfree`, excess true rate loss | |
|---|---|
| conditional on firing (the number above) | 0.1129 |
| **honest, midpoint fallback scored** | **0.1533** |
| `mid` | 0.1830 |

Roughly **half the bench's margin was survivorship**, and the dropped replicates
are not a random subset: on them `mid` itself scores 0.2563 against 0.1542 where
a root exists. The rule fails on exactly the hard steps. The Gumbel rule still
wins in the bench after the correction — by −0.030 rather than −0.070 — so the
family argument survives, but weakly enough that it no longer contradicts the
real-data result on its own.

### Where the missing roots actually come from

Splitting the failures by which guard fired, on the same corner:

| branch | share | what it means |
|---|---|---|
| `modes_swapped` (fit rejected) | **14.2 %** | EM put the Gumbel on the **high** mode and `fit_gumbel_normal_mixture` discarded the fit |
| `hi_owns_lo_mode` | 17.7 % | the two components collapsed onto each other (mode gap 0.141 logits, against 0.607 where a root exists) |
| `lo_owns_hi_mode` | 2.8 % | the low component's tail swamps the high mode |
| M-step numerical failure | 0.1 % | — |

So the largest single branch is not the crossing at all — it is the fit's
*ordering assumption*. #2836 pinned the Gumbel to the low component from the
region-voting argument, and that premise does not survive the arithmetic: a sim
set is 95–99 % negatives, so the right-skewed max-pooled bulk **is** the negative
class, and EM putting the Gumbel on the upper mode is EM preferring the better
description. (This also explains why prediction (4) failed: the Gumbel is
capturing right-skew wherever it finds it, not "the max over region nodes".)

### The repair, and one that looked obvious but is not

Between the two modes the log-density difference is monotone —
`d/dx [log g − log n] = (e^{−z} − 1)/scale + (x − mu)/var`, both terms
non-positive on `[loc, mu]` — so the existing bisection is exact, a sign change
at the endpoints is necessary *and* sufficient, and the same solver works in
either orientation. Measured on the production-like corner:

| | fire rate | excess |
|---|---|---|
| today | 68.1 % | 0.1585 |
| `sup` clamp (what the Gaussian sibling did at the time, in `_rate_cut_in_interval`) | 100 % | 0.1568 |
| **keep swapped fits, solve in either orientation** | 73.1 % | **0.1487** |

The Gaussian family's own answer to this problem does **not** transfer: clamping
is worse than the midpoint fallback on the `hi_owns_lo_mode` bucket (0.2858 vs
0.2563), which is the larger of the two, and only better on the small
`lo_owns_hi_mode` one. (The Gaussian sibling has since stopped clamping at the
edge too — issue #2896 found the flat step deadened the Inclusion knob — and now
continues past it at the rule's first-order slope. That is a different variant
from the one measured in this row, and it has not been scored on the EVT family.) Dropping the ordering constraint does help, and it is
principled rather than tuned — but it recovers 5 of the ~32 missing points. The
rest are genuinely non-bimodal steps, which is a statement about those steps
rather than about the solver.

All of the above is bench-only. `gumbel_any_*` now runs as a measured variant
beside the incumbent `gumbel_*`, and `cut_fail_reason` records which guard
declined, so the next Visual Genome run settles it on real scores.

> **Settled 2026-08-07 — and the bench over-sold it again.** On real VG scores
> the repair cuts the fallback rate 24.9 % → 19.6 % as designed, but changes the
> cut on only 5.4 % of steps and is worth **−0.0004 (n.s.)** against the bench's
> −0.010. Where it does fire its sign flips with the tilt: `priorfree` −0.0117
> (p = 8e−4), `cross` **+0.0219** (p = 0.015). Recommendation is not to promote
> it. The bench half re-ran at full power on the Grid and reproduced the local
> probe (`gumbel_priorfree` honest 0.1547, `gumbel_any_priorfree` **0.1468**,
> `mid` 0.1824) — so the two halves now disagree with the survivorship bug fixed
> on both sides, which is a statement about the bench's generative model rather
> than about either measurement. See `REMEASURE-2846.md`.

Estimation noise is *not* the story: finite-sample (n=500) excess minus
population excess is ≤ 0.015 in magnitude for every rule, and slightly negative.
Issue hypothesis 3 is refuted.

## A stable alternative rule fell out

The pre-registered fallback was "if the true optimum sits at a stable survival
level of the fitted Bad component, cut the tail at alpha". Measured across 511
cells:

| tail model | median alpha | IQR ratio | stable (bar: < 3) |
|---|---|---|---|
| Gaussian low component | 0.069 | 4.93 | no |
| **Gumbel low component** | **0.165** | **2.22** | **yes** |

The EVT *fit* is well-behaved even where the EVT *crossing* has no root, which
is consistent with the #2846 diagnosis above — the largest failure branch was the
ordering guard rejecting a sound fit, not the fit being bad — and makes
"cut the Bad tail at alpha ≈ 0.165" a coherent one-constant rule worth testing
against the shipped winner. It needs no crossing at all, so it fires wherever the
fit exists.

Both numbers here were computed over fits that #2836 kept, i.e. excluding the
~14 % it discarded as swapped. Those now produce fits, and `lo_survival` reads
whichever component is the low one, so this table needs recomputing on the next
run before it is leaned on.

> **Recomputed 2026-08-07 — the finding survives.** With the swapped fits kept,
> over the same 511 cells: Gaussian median α 0.070, IQR ratio 5.54 (still
> unstable); Gumbel median α **0.158**, IQR ratio **2.38** (still inside the
> pre-registered bar of 3). See `REMEASURE-2846.md`.

## Scope

VG region voting on the production linear head, inclusion 0, 30 votes deep, 552
cells (511 non-empty; ~17 % emit no rows, concentrated in rare small-object
categories on the siglip arm, symmetric across variants because the split is
pre-vote and deterministic). At inclusion 0 the cost weights are (1, 1), so
`rate` and `priorfree` are the **same rule** — the Inclusion-tied form is
untested at non-zero Inclusion and needs its own trajectory arm. Every contrast
is within-step, so none of this can see selection feedback; #2799 showed that
effect is real and worth ~0.02 on its own, so the shipped gain is a lower bound.
