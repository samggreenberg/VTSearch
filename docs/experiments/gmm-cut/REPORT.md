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

## The theory bench disagrees with the data, and the fallback rate says why

The bench generates scores from a model of region voting whose class-conditionals
are known in closed form, so each rule's excess loss is measured against **truth**
rather than a held-out sample (11 520 fits).

In the production-like regime (m = 24, prevalence ≤ 0.05), mean excess true rate
loss:

| rule | bench | | real data (ramp cost) |
|---|---|---|---|
| `gumbel_priorfree` | **0.1151** (wins 56 % of configs) | | 0.5165 (**loses**, +0.0043) |
| `gumbel_cross` | 0.1465 | | 0.5294 (**loses**, +0.0172) |
| `mid` | 0.1824 | | 0.5121 |
| `priorfree` | 0.1883 | | **0.5075 (wins)** |
| `cross` | 0.2159 | | 0.5156 |

The bench says the *family* is the binding constraint — `priorfree` is
statistically tied with `mid` there (slightly worse), and only the EVT rules
help. The real data says the opposite. **The fallback rates reconcile them:** on
real scores the Gumbel crossing does not exist on **27 %** of steps
(`gumbel_priorfree`) and **53 %** (`gumbel_cross`), where the rule silently
degrades to the midpoint. In the bench's clean two-mode samples it essentially
always exists.

So the EVT rule is better in principle and too fragile in practice. Making that
fit robust is the single highest-value follow-up — it is worth up to the
misspecification term (+0.0134), which is comparable to the prior term we just
banked.

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

The EVT *fit* is well-behaved even though the EVT *crossing solve* is fragile —
which is consistent with the fallback-rate diagnosis above, and makes
"cut the Bad tail at alpha ≈ 0.165" a coherent one-constant rule worth testing
against the shipped winner.

## Scope

VG region voting on the production linear head, inclusion 0, 30 votes deep, 552
cells (511 non-empty; ~17 % emit no rows, concentrated in rare small-object
categories on the siglip arm, symmetric across variants because the split is
pre-vote and deterministic). At inclusion 0 the cost weights are (1, 1), so
`rate` and `priorfree` are the **same rule** — the Inclusion-tied form is
untested at non-zero Inclusion and needs its own trajectory arm. Every contrast
is within-step, so none of this can see selection feedback; #2799 showed that
effect is real and worth ~0.02 on its own, so the shipped gain is a lower bound.
