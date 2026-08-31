# Is the 2-component Gaussian mixture a *good* fit? — results (issue #3329)

Findings for the run pre-registered in [`PREREG.md`](PREREG.md). Nothing in the
pre-registration was edited after the array was submitted; where this report
departs from it, it says so and why.

## BLUF

**The mixture is a mildly bad fit, in exactly the place predicted and nowhere
near the predicted size — and the misfit that does exist goes with *lower* cost,
not higher.** The Bad mode really is right-skewed under max-pooling, and the
shipped anchoring really does leave the fit alone, but neither costs anything at
the operating point.

**#3329's answer for this fit: the fit is not the problem.** That is the
outcome H4 was pre-registered to be able to return, and it closes this line
cheaply rather than launching a fit-replacement programme on an aesthetic
objection.

**The sharper answer, which the run was not designed to produce, is in the
[worked-cell figure](#the-worked-cell--the-figure-3329-asked-for):** the arm with
the *worst* distance-to-CDF statistics is the only one whose fitted components
actually land on the true classes, and the arm with the *best* ones never learns
anything at all. A 2-Gaussian mixture fits an unseparated blob beautifully. So
"is it a good fit?" and "is it doing its job?" are close to opposite questions
here, and the absolute goodness-of-fit statistics this issue asked for answer
only the first.

| # | claim | pre-registered bar | measured | verdict |
|---|---|---|---|---|
| H1 | misfit concentrated in the tail, Gaussian **under**-predicts | median `tail_ratio` > 1.5 region / > 1.2 binary | **1.02–1.05** | **refuted** — real, ~30× too small |
| H2 | Bad mode right-skewed under max-pooling | `shape_skew_neg` > 0.5 on `max_patch`; paired diff > 2 SE | **0.85**; paired **+0.78 ± 0.011** | **confirmed** |
| H3 | shipped anchoring is inert | \|`anchored_dmu_lo`\| < 0.01 **and** mass < 1e-3 | **0.0004–0.0023** ✓; **0.0023–0.0038** ✗ | **prediction holds, mechanism refuted** |
| H4 | misfit predicts regret | slope > 0 at > 2 SE, partial R² ≥ 0.05 | partial R² **0.053–0.089**; slope **negative** | **refuted, with the opposite sign** |

## What was run

`vg_scale_any` × {`siglip`, `siglip+dinov3_patch`} × 12 categories × 8 seeds =
**192 cells**, 100 clicks, shipped defaults, `CALIB_SAFE_THRESHOLDS=1`.

| | |
|---|---|
| cells completed | **192 / 192**, 0 zero-byte, 0 unreadable |
| fit-quality rows | **23 064**, 0 dropped |
| `fit_ok` | **100 %** of rows — no degenerate-fit filter to report |
| positives per shape statistic | 132–170 (floor 30), so **0 %** of steps declined |
| prevalence | **7.1 % median, 6.2–8.0 %** on a constant 2100-item test set |

None of PREREG's uninterpretability conditions fired.

**The array was run twice.** The first run could not score H3 — the instrument
was broken in two ways (see [Instrument defects](#instrument-defects-found-and-fixed)).
After the fix, H1, H2 and H4 came back **bit-identical**, which is the evidence
that the fix touched only the anchor columns. All numbers below are from the
second run.

## H1 — the tail misfit is real, and small

Median `tail_ratio` (empirical exceedance ÷ predicted, at the shipped cut):

| scope | `siglip/whole_image` | `dinov3_patch/whole_image` | `dinov3_patch/max_patch` |
|---|---|---|---|
| **`fold`** (the shipped fit) | 1.01 | 0.99 | 0.99 |
| `sim:pooled` (the labelled set) | 1.04 | 1.02 | 1.05 |

Standard errors are 0.0010–0.0016, so every sim figure is resolvably above 1.0 —
roughly 30 SE. The Gaussian does under-predict its own tail. It under-predicts
it by **2–5 %**, against a pre-registered bar of 20–50 %.

**The fold scope is the headline, and it is the first time anyone has looked.**
Nothing in `vtscore/eval/` had ever read `FoldAnchoredCut.fits`; every
measurement in this line has been the unanchored fit on the sim set. The fit the
app *actually cuts from* is calibrated in the tail to within 1 %, and on both
region arms it **over**-predicts — the opposite direction to the one predicted,
and better-behaved than the fit the whole prior line was measuring.

Distance to the fitted CDF on its own data, median:

| scope | arm | KS | CvM | AD |
|---|---|---|---|---|
| `fold0` | `siglip/whole_image` | 0.021 | 0.17 | 1.1 |
| `fold0` | `dinov3_patch/whole_image` | 0.019 | 0.12 | 0.8 |
| `fold0` | `dinov3_patch/max_patch` | 0.030 | 0.43 | 2.6 |
| `sim:pooled` | `siglip/whole_image` | 0.023 | 0.23 | 1.6 |
| `sim:pooled` | `dinov3_patch/whole_image` | 0.021 | 0.19 | 1.4 |
| `sim:pooled` | `dinov3_patch/max_patch` | 0.032 | 0.53 | **3.4** |

The max-pooled arm is the worst-fitting on every statistic, and the gap is
widest on **AD**, which weights the tails — the same direction H2 predicts.
(No p-values, deliberately: at these sample sizes every test rejects every
model, so a p-value would report the sample size. These are effect sizes.)

**The misfit grows as the user clicks**, which a median over the run hides:

| arm | t=5 | t=10 | t=20 | t=50 | t=100 |
|---|---|---|---|---|---|
| `siglip/whole_image` | 1.014 | 1.013 | 1.035 | 1.045 | 1.056 |
| `dinov3_patch/whole_image` | 1.000 | 0.996 | 1.008 | 1.038 | 1.037 |
| `dinov3_patch/max_patch` | 1.003 | 1.040 | 1.057 | 1.062 | 1.044 |

## H2 — confirmed, and it is the pooling, not the embedder

Median skewness of the **true Bad** class on the logit axis:

| arm | `shape_skew_neg` | role |
|---|---|---|
| `dinov3_patch / max_patch` | **0.85 ± 0.014** | region voting |
| `siglip / whole_image` | 0.58 ± 0.013 | binary control |
| `dinov3_patch / whole_image` | 0.33 ± 0.011 | same embedder, no pooling |

Paired `pooled − image` within each `max_patch` run — the same media, the same
model, only the pooling changed: **+0.78 ± 0.011**, about 72 SE. Both the level
contrast and the paired contrast agree, so the pre-registered tie-break (the
paired one governs) never had to be used.

**The third geometry is what makes this a finding rather than an observation
about DINOv3.** DINOv3 under whole-image pooling has the *least* skewed Bad mode
of the three — less than SigLIP's. So the skew is not a property of the
embedder; max-pooling manufactures it, which is what the extreme-value argument
predicts.

**The skew is not a constant of the arm — it changes sign early in the run:**

| arm | t=5 | t=10 | t=20 | t=50 | t=100 |
|---|---|---|---|---|---|
| `siglip/whole_image` | −0.37 | 0.37 | 0.64 | 0.62 | 0.52 |
| `dinov3_patch/whole_image` | **−1.48** | −0.25 | 0.18 | 0.39 | 0.37 |
| `dinov3_patch/max_patch` | 0.71 | **1.12** | 1.00 | 0.86 | 0.73 |

Early fits are *left*-skewed on both whole-image arms. Only `max_patch` is
right-skewed from the first trainable click. Any future work that assumes an
EVD shape from the start would be wrong for the first ~15 clicks of a
whole-image run.

## H3 — the prediction holds; the argument for it does not

The two halves split:

| half | `siglip/whole` | `dinov3/whole` | `dinov3/max_patch` | bar | |
|---|---|---|---|---|---|
| \|`anchored_dmu_lo`\| | 0.0019 | 0.0023 | 0.00044 | < 0.01 | **confirmed** |
| `anchor_mass_frac` | 0.0023 | 0.0036 | 0.0038 | < 1e-3 | **refuted** |

`anchor_mass_frac` runs from 2.9e-4 at click 5 to **7.4e-3 at click 100**,
crossing the pre-registered ceiling at about **click 20**. PREREG computed
1.2e-4 for 20 votes against a **50 000**-point haystack; `vg_scale_any` fits
about **2000**. The denominator, not the votes, was carrying that argument.

So the anchors carry roughly **7× the mass the pre-registration called
negligible, and still barely move the fit**: \|Δμ_lo\| is 0.0021 at click 100,
**1.3 % of the distance between the fitted components** (p90 12.8 %). The
conclusion survives on direct evidence; the mechanism offered for it does not.
What makes the anchors inert is the **one-hot E-step clamp**, not their M-step
mass share.

**They are not inert everywhere.** Median \|Δμ_hi\| is **0.012** — past the same
0.01 ceiling — and median \|Δw_lo\| is **0.032**. The anchors reweight the
mixture and shift the **Good** component; the Bad component, whose tail the cut
sits in, is the one they leave alone. Since the shipped cut is a midpoint rule,
a 0.012 shift in μ_hi is a ~0.006 shift in the cut, which is not nothing.

This also means the κ = 0.3 optimum found by the anchor-mass sweep
([#2861](../2026-08-05-population-anchored-calibration/)) still needs an
explanation, and "mass" is not it.

## H4 — the gate, refuted with the opposite sign

OLS of `regret_honest` on `log(tail_ratio)`, conditioned on `log(n_test_pos)`:

| arm | n | slope | partial R² |
|---|---|---|---|
| `dinov3_patch/max_patch` | 1924 | **−0.16 ± 0.012** | 0.089 |
| `dinov3_patch/whole_image` | 1924 | **−0.38 ± 0.034** | 0.062 |
| `siglip/whole_image` | 1924 | **−0.30 ± 0.029** | 0.053 |

Every arm clears the pre-registered partial-R² bar of 0.05, at 10–14 SE. **The
slope is negative on all three**: the cells where the Gaussian under-predicts
its tail most are the cells with the *lower* cost.

The pre-registered test is one-sided (`slope/SE > 2`), so the analyzer's canned
verdict reads "does not predict regret". That undersells it. Misfit predicts
regret about as well as pre-registered — in the direction opposite to the one
that would license replacing the fit.

The plain reading is that both quantities are downstream of how well the run is
going: a run that has separated its classes has a heavier, better-resolved tail
above the cut *and* a lower cost. `n_test_pos` does not absorb it, so it is not
simply prevalence. **What it is not is evidence that fixing the fit would buy
anything.**

## Figures

Generated by `figures_fitq_3329.py` from the same frames as the tables above.

### The mandatory quality-over-clicks pair

![cost vs clicks](figures/cost_vs_clicks.png)

*Mean over all 96 cells per arm, with an inter-quartile band; click 0 is each
cell's own free text sort, and every curve is anchored there. Lower is better.
Read the distance from the left-hand dot to the right-hand end as what the
clicking bought over what typing already gave. It does **not** license a
per-embedder claim: the arms differ in embedder and pooling together.*

| arm | text sort (0 clicks) | crossover | cost @ 100 | AP @ 100 |
|---|---|---|---|---|
| `dinov3_patch/max_patch` | 0.38 | **click 10** | **0.24** | 0.79 |
| `siglip/whole_image` | 0.38 | click 25 | 0.30 | 0.67 |
| `dinov3_patch/whole_image` | 0.38 | **never** | 0.39 | 0.62 |

**`dinov3_patch/whole_image` never beats the free text sort** — after 100 clicks
it is still worse on cost (0.39 vs 0.38) and on average precision (0.62 vs
0.65). This is the middle geometry, H2's "same voting mode, different embedder"
control. It does not invalidate H2's paired contrast, which is taken *within*
`max_patch` runs, but it does mean the level comparison against that arm is a
comparison against an arm that never learns, and should be read as such.

![cost vs clicks, per run](figures/cost_vs_clicks_runs__vg_scale_any.png)

*Every seed of every category as its own line, one panel per arm. The spread is
the finding the mean hides: within each arm there are cells that never leave the
floor and cells that are excellent by click 20.*

### The four pre-registered statistics over clicks

![fit statistics over clicks](figures/fit_statistics_over_clicks.png)

*Median and inter-quartile band against the axis the user spends. Both H3 panels
are log-scaled. This figure is the argument for not reporting these as single
medians: `anchor_mass_frac` (bottom left) crosses its pre-registered bar around
click 20–25, so "the median is 0.0038" describes an average across a crossing,
and the Bad mode's skew (top right) starts negative on both whole-image arms.*

### The worked cell — the figure #3329 asked for

![worked cell fit overlay](figures/worked_cell_fit_overlay.png)

*One cell per geometry (`backpack`, seed 0), at four click checkpoints. The
histogram is the score distribution **coloured by ground truth** — the colouring
no user and no prior run has seen, because it needs labels the app does not
have. Since the Good class is only ~7 % of the mass, its prior-weighted
class-conditional density is drawn as a step line too, on the same scale as the
fitted components. Over that are the two fitted Gaussians and their sum; the
dashed red line is the threshold the app would ship at that click. It is **one
cell**, fixed before the results were read, and is illustrative — every
quantitative claim above is over all 192.*

**What it shows, and it is not what the summary statistics suggest.** The three
rows behave completely differently, and the difference is not fit quality in the
distance-to-CDF sense:

- **`max_patch` (middle row)** is the only geometry where the mixture becomes
  genuinely bimodal, and by 100 clicks its **fitted high component sits on the
  true Good mass** — the dark orange curve and the light orange step line up.
  The two-component model is describing the two classes.
- **`siglip/whole_image` (top row)** never separates. Its fitted high component
  is far broader than the true Good class and peaks well to the *left* of it: it
  is absorbing the right shoulder of the **Bad** distribution, not finding the
  Good one. The cut still lands in a reasonable place, which is why balanced
  accuracy stays at 0.84 — but for the wrong reason.
- **`dinov3_patch/whole_image` (bottom row)** barely moves at all across 100
  clicks, which is the picture behind its never crossing the text sort.

This is the sharpest thing in the run. The arm with the **worst** distributional
fit statistics (`max_patch`: KS 0.032, AD 3.4) is the one whose components
actually correspond to the classes, and the arm with the **best** ones
(`dinov3_patch/whole_image`: KS 0.021, AD 1.4) is the one that never learns
anything. A 2-Gaussian mixture fits an unseparated blob very well. **Distance to
the fitted CDF is not a measure of whether the model is doing its job**, and
that — more than any single bar — is the answer to "I never look to see if it's
a good fit."

Numerically, across all cells (`sim:pooled`), the components do land on the
classes:

| arm | balanced accuracy | ARI | μ_lo error | μ_hi error |
|---|---|---|---|---|
| `dinov3_patch/max_patch` | **0.87 ± 0.001** | 0.62 | −0.0034 | −0.0041 |
| `siglip/whole_image` | 0.84 ± 0.002 | 0.42 | −0.0085 | −0.025 |
| `dinov3_patch/whole_image` | 0.74 ± 0.003 | 0.49 | −0.0011 | 0.012 |

The fitted low component identifies the true Bad class at 74–87 % balanced
accuracy, and its mean sits within 0.009 of the true class mean. **The mixture
is recovering the classes, not merely fitting the histogram** — which is the
strongest single reason the misfit costs so little.

### Interactive viewer

[`viewer.html`](viewer.html) — filter by category, seed and arm; every cell's
own curve. Built with
`viewer.py --results $EXP --arms results=prod --baseline analysis/text_baseline.csv`.
It carries no supervised-skyline floor (#3322): this grid was not run with
`CALIB_SKYLINE_ARMS`, so there is no learnability ceiling drawn on it.

## Instrument defects found (and fixed)

Three, all of which emitted plausible numbers rather than failing. They are
recorded here because the first run's output was *readable and wrong*.

1. **The baseline could not write.** `text_baseline.py` completes its whole pass
   before handing the path to pandas, so a missing `$CALIB_EXP/analysis` cost
   the entire run and surfaced as a write error 47 s in.
2. **The base-row filter emptied the main frame.** `load_main` filtered *both*
   variant columns to blank — right for `gmm_variant`, wrong for `pool_variant`,
   which the harness stamps with the base pooling's own name, `max`. All 192
   cells were dropped, and **H4 was written out as an empty CSV and scored as a
   null it had never computed.** The selftest passed because its fixture planted
   `pool_variant: ""`, a value nothing in the harness emits.
3. **H3 was unmeasurable, twice over.** `anchored_dmu_*` were filled only when
   handed an `anchored_fit` that no call site ever passed — 0 of 11 520 fold
   rows. And `anchor_mass_frac` was computed from `n_anchored`, which counts
   **folds** (0/1/2), where the formula needs the fold's **vote** count;
   `anchor_n` sat flat at 2.0 from click 5 to click 100.

**The common shape, worth carrying forward:** each of these produced a number a
reader would accept — a flat 2.9e-4, an empty CSV read as a null, NaN read as
"did not move". The existing test for the drift statistic checked only its
**null** (identical fits → zero drift), which passes whether or not the column
is ever computed. A test that a statistic is *finite* is not a test that it was
*measured*. The new tests assert the columns arrive populated and move with a
planted displacement.

## What this licenses, and what it does not

**Do not launch a fit-replacement programme for this fit.** H1 and H4 together
say the Gaussian's tail error at the operating point is a few per cent and is
not associated with higher cost. The #2836 `misspecification` term (+0.0129) is
real as an accounting entry, but this run finds no cost attached to it at the
shipped operating point.

**H2 stands as a mechanism result, not a cost result.** Max-pooling really does
manufacture a right-skewed Bad mode. If a Gumbel-family fit is ever worth
building, this is the evidence for where it belongs — but H4 says the payoff for
building it is not visible here.

**The `max_patch` arm remains the one worth running.** It reaches cost 0.24
against the text sort's 0.38 and crosses at click 10, while also being the
worst-fitting arm on every distributional statistic. That combination is itself
the study's tidiest summary: **fit quality and usefulness point in opposite
directions here.**

## Limits

Recorded so the next study does not over-read this one.

- **One dataset, one prevalence.** `vg_scale_any` holds prevalence at 7.1 % by
  construction — chosen so the statistics are comparable, at the cost of saying
  nothing about how any of this moves with prevalence. Banding is the obvious
  follow-up.
- **One horizon.** 100 clicks. H1's tail ratio is still rising at click 100 on
  two of three arms, so a longer run may not stay under the bar.
- **Two folds.** `calibrate_count=2`, so every fold statistic is a median over
  two fits per cell.
- **H3's mass bar was set for the wrong haystack.** 1e-3 was derived for 50k
  points; at ~2k it is crossed by click 20 on every arm. The bar, not the
  finding, is what needs restating next time.
- **The middle geometry never learns**, so level comparisons against
  `dinov3_patch/whole_image` compare against an arm that does not beat typing.
- **`regret_honest` is a simulated quantity.** H4's negative slope is a
  within-harness association; nothing here measures a real user's regret.
