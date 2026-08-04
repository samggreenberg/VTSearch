# Which cut does the GMM path want? (issue #2836)

**Status: pre-registered, not yet run.** This file is the design and the
predictions, written before the numbers exist. Delete it when the study ships
and fold the durable derivation into `docs/ML.md`.

## Background

The safe-threshold path fits a 2-component mixture to the unlabelled sim-set
score distribution and cuts it somewhere. #2798 moved that cut from the
midpoint-of-means to the equal-density crossing on a geometry argument; #2799
measured the two as paired within-step variants and the crossing lost in every
max-pooled window (+0.0036 cost at 6–20 votes, +0.0059 at 2–5), so #2833 reverted
it. The revert is empirically right and theoretically empty: we now ship a rule
with no derivation behind it, having rejected the one rule that had one.

The inversion in the #2799 data is the clue. The crossing *helps* in the
whole-image geometry (−0.0022, p = 1.6e−4) and *hurts* under max-pooling
(+0.0036, p = 1.9e−8) — that is, it overshoots most exactly where #2798's
argument said it should help most.

## The diagnosis

`_weighted_gaussian_crossing` solves

```
w_lo · N(x; mu_lo, var_lo)  ==  w_hi · N(x; mu_hi, var_hi)
```

which is the Bayes boundary **with the mixture weights as class priors** — the
cut that minimises expected misclassification *count*. What the harness scores,
and what the Inclusion knob is defined in terms of, is

```
cost = fpr_weight · FPR + fnr_weight · FNR        # inclusion_weights(0) == (1, 1)
```

a weighted sum of **rates**. FPR normalises by the negatives and FNR by the
positives, so the objective is prevalence-free *on purpose*: that is what makes
"the fraction of true matches I am willing to miss" mean the same thing on every
dataset. Differentiating it,

```
dL/dtau = -fpr_weight · f_neg(tau) + fnr_weight · f_pos(tau) = 0
```

so the rate-optimal cut is the **prior-free** crossing `fnr_weight · f_pos ==
fpr_weight · f_neg`. Under the identification `f_neg = N_lo`, `f_pos = N_hi`,
that is the same solve at a different tilt — hence the one-parameter family

```
w_lo · N_lo(x)  ==  lambda · w_hi · N_hi(x)
```

with `lambda = 1` the count-optimal cut we shipped and reverted, and
`lambda = (fnr_weight/fpr_weight) · (w_lo/w_hi)` the rate-optimal one. In the
equal-variance case the family has a closed form:

```
x(lambda) = midpoint + var · ln(w_lo / (lambda · w_hi)) / (mu_hi - mu_lo)
```

Two things fall out immediately.

**The midpoint is not a heuristic.** At `lambda = w_lo/w_hi` and equal cost
weights the log term vanishes: the prior-free crossing *is* the midpoint of the
means. So the rule we just reverted to is an estimate of the rate-optimal cut
that happens to be exact whenever the two components have equal variance — which
is why it has been hard to beat.

**The bias has a name and a size.** At `lambda = 1` the cut sits above the
rate optimum by `var · ln(w_lo/w_hi) / (mu_hi - mu_lo)`, and the pool is
overwhelmingly negative, so `ln(w_lo/w_hi)` is large. The bias grows with the
component variance and shrinks with the separation, which is a quantitative
prediction about *where* the crossing should hurt most, not just that it should.

Where the midpoint and the prior-free crossing come apart is precisely where the
variances differ — and that is a live region, since max-pooling reshapes the Bad
mode. So the family predicts a third point, and the interesting question is
which side of the midpoint it lands on.

## Competing hypotheses

The prior term is the leading explanation, not the only one, and the study is
built so the alternatives are measured rather than argued.

<!-- item-sep -->

- **Misspecification.** #2798's own premise is that a max-pooled Bad mode is an
  *extreme-value* statistic — the max over ~24 region nodes. We then fit it with
  a Gaussian, so the shape is wrong exactly in the upper tail where the cut
  lands, and the crossing is solved against a fiction. Repaired by fitting the
  low component as a **Gumbel** (`vtscore/training/evt_mixture.py`).

  **The axis matters, and it is not the score axis.** The limit theorem applies
  to the max of the region *logits*; a score is that max pushed through a
  sigmoid, and the squash destroys the shape. Measured before this study ran: a
  Gumbel fitted to sigmoid scores *loses* to a 2-Gaussian mixture (−0.008 mean
  log likelihood at m = 24), and wins on the logit axis. So the EVT fit is done
  in logit space and the cut mapped back — which is exact, because a density
  crossing is invariant under a monotone reparametrisation (both sides pick up
  the same Jacobian). This is a different claim from #2799's dead logit-space
  *Gaussian* variant: that moved a family across axes, which changes little;
  this changes the family to the one the limit theorem names.

<!-- item-sep -->

- **Identification.** The mixture's high component is whatever the upper mode
  is, not "the positives". Under max-pooling even a thoroughly bad image
  contributes one confidently-scored region, so the high component may be
  capturing *images with a strong region* rather than true matches. If so, no
  crossing between fitted components has class semantics and the midpoint wins
  by being less sensitive to the misidentification. Note this is not fully
  independent of the previous item: a right-skewed low component with a heavy
  tail *absorbs* those one-strong-region negatives, which is why the Gumbel arm
  addresses both at once.

<!-- item-sep -->

- **Estimator variance, not population optimum.** The crossing reads both
  variances and both weights; the midpoint reads only the means. At 6–20 votes
  on a few hundred sim scores the crossing may be the better *rule* and the
  worse *estimator*.

<!-- item-sep -->

## Design

Two halves, run together, answering different questions. Neither alone is
enough: the bench can explain a mechanism but not tell us what VG does, and the
data arm can rank rules but not say why one won.

### Part A — the theory bench (`scripts/experiments/calibration/theory_bench.py`)

Scores from a generative model of region voting whose class-conditionals are
known in closed form: `m` region logits per image, negatives all
`N(mu_bad, sd_bad)`, positives one `N(mu_good, sd_good)` object region plus
`m-1` bad ones, image score = sigmoid of the max. Then

```
FPR(t) = 1 - Phi_bad(t)^m
FNR(t) = Phi_good(t) · Phi_bad(t)^(m-1)
```

so the exact rate-optimal cut is a grid minimisation and **every rule's excess
loss is measured against the truth**, with no held-out sample and no noise.
Sweeps `m ∈ {1, 6, 24}` (m = 1 is the whole-image control, where the
extreme-value story must vanish), prevalence, separation, variance ratio, and
sample size, and repeats the whole sweep at n = 50 000 (the GMM's own subsample cap) to
separate a rule's own error from small-sample jitter.

Note what prevalence does *not* do in those two lines: it does not appear. The
truth is prevalence-free, so sweeping it while holding the correct answer fixed
is a direct measurement of what a prior-bearing rule pays.

### Part B — the data arm (the #2799 harness, extended)

Same sizing as #2799 (VG region voting, `dinov3_patch × max_patch` production arm
plus a `siglip × whole_image` single-vector control, linear head, 30 steps, 12
seeds), because every contrast is paired *within a step*: each variant re-cuts
the same per-step model on the same votes and is scored against the same held-out
test set. Only the safe-ON arm needs running.

**Arms** (`_SAFE_GMM_VARIANTS`), each recorded twice — at the blended threshold
(what a user gets) and at the raw cut (what the rule is worth before the
conformal blend damps it):

| variant | rule |
|---|---|
| `pooled_mid` | midpoint — production, the incumbent |
| `pooled_cross` | `lambda = 1`, count-optimal — #2798's cut, the control |
| `pooled_priorfree` | `lambda = w_lo/w_hi`, rate-optimal at equal weights — **the leading candidate** |
| `pooled_rate` | `lambda = (wn/wf)(w_lo/w_hi)` — ties the cut to the Inclusion knob, which the conformal path already respects and this path ignores |
| `pooled_gumbel_{cross,priorfree,rate}` | the same three tilts against a Gumbel low component |
| `image_{mid,cross,priorfree,rate}` | the whole-image geometry, where the #2799 inversion lives |
| `pooled_supervised`, `pooled_sim_oracle` | **diagnostics, not candidates** — they read the sim set's true labels |

At inclusion 0 the cost weights are (1, 1), so `rate` and `priorfree` coincide by
construction; the `rate` arm exists to be exercised, and to be the shipped form
if the winner is the prior-free family (the app's Inclusion knob is not always 0
even though the study runs there).

**The decomposition.** The simulation knows the sim set's true labels, so the gap
between what a rule cuts and where the rate loss is actually minimised splits
into named terms that telescope:

```
tau_cross      - tau_priorfree     prior / loss mismatch
tau_priorfree  - tau_supervised    component identification
tau_supervised - tau_sim_oracle    Gaussian misspecification
tau_sim_oracle - tau_test_oracle   finite sim set / transfer
```

reported per step in threshold units **and** in excess-cost units (threshold
distance is not what anyone pays: a term that moves the cut a long way through a
flat region is cheap, one that moves it across the elbow is not). Written to a
side frame, `task_*__cutdiag.csv`, one row per (step, geometry), along with the
fitted mixture parameters of both families.

## Predictions, pre-registered

1. **The offset identity.** `tau_cross - tau_mid` equals
   `var · ln(w_lo/w_hi) / (mu_hi - mu_lo)` to fit error, with the residual
   growing in `|ln(var_lo/var_hi)|` (the closed form assumes equal variances, so
   that dependence is the check, not noise).
2. **The penalty scales with the offset.** The per-step cost penalty of the
   crossing rises monotonically across quintiles of that offset. If it does, the
   pooled-vs-image inversion is explained quantitatively rather than by story.
3. **The prior-free crossing beats both incumbents**, and beats the midpoint
   specifically where the variances differ most.
4. **The EVT gain is geometry-specific.** `evt_loglik_gain > 0` on the pooled
   geometry and ≈ 0 on the image geometry and at `m = 1` in the bench. A gain
   that is uniform across geometries means the Gumbel is merely more flexible,
   not more correct. (`gmm_logit_loglik` is recorded alongside so a gain is
   attributable to the *family* rather than to the logit axis it is fitted on.)

If (1) holds and (3) fails, the diagnosis is wrong and the decomposition says
which of the alternatives took over.

## Decision rules

<!-- item-sep -->

- **Ship** the rule that is closest to the oracle cut **and** wins on cost, on
  the production `dinov3_patch × max_patch` arm in the 6–20-vote window,
  provided it does not regress the single-vector arm (`calculate_gmm_threshold`
  also backs the cosine/text sort).

<!-- item-sep -->

- **If the prior-free crossing wins**, it supersedes both #2801 and #2833: a
  principled crossing with the priors correctly dropped, rather than a heuristic
  that approximates it only when the variances happen to match.

<!-- item-sep -->

- **If the midpoint still wins** after the derivation is repaired, say so
  explicitly: the fitted mixture is not describing the classes, and the GMM path
  is a heuristic mode-finder, which changes what it should be used for. In that
  case the fallback deliverable is the **Bad-tail rule** — if the true optimum
  sits at a stable survival level of the fitted low component across cells
  (pre-registered bar: interquartile ratio < 3), then "cut the Bad tail at
  alpha" is a one-constant rule with a real justification, and the mixture is
  being used as a tail model instead of as a classifier.

<!-- item-sep -->

- **A rule that wins only on the blended column** and not on the raw cut is
  winning by sitting closer to the conformal threshold, not by being a better
  rule; report it as such and do not ship it on that basis.

<!-- item-sep -->

## Known limitations

The `rate` arm's cut is tilted by the Inclusion weights, but the run's
trajectory is driven by inclusion 0 throughout, so its within-step contrast is a
counterfactual on the cut alone — the conformal threshold it blends with, and
the items Autopilot chose to label, are both still the inclusion-0 ones. A real
answer for non-zero Inclusion needs its own trajectory arm.

More generally, every contrast here is within-step by construction, so none of
them can see selection feedback (a better cut surfaces better items to vote on).
#2799 showed that effect is real and worth ~0.02 cost on its own. If a rule wins
here, the ship decision is still sound — a within-step win is a lower bound on
an A/B win when the mechanism is "the cut is better placed" — but the *size* of
the gain is understated.
