# Pre-registration: is the 2-component Gaussian mixture a *good* fit? (issue #3329)

**Written before the run.** Nothing below may be edited after the cells array is
submitted; the findings go in a separate `REPORT.md`. This line has already
produced two wrong-but-plausible readings at full power (#2836 and #2846, both
mis-sized in the same direction by the theory bench, both corrected only by a
later run), so a prediction recorded afterwards is not a prediction.

## BLUF — the prediction

**The mixture is a bad fit, in a specific and predictable place, and the
shipped estimator's "anchoring" does nothing to fix it.** Whether that costs
anything is the open question, and H4 is pre-registered to be able to say *no*.

| # | claim | pre-registered bar |
|---|---|---|
| H1 | the misfit is **concentrated in the tail the cut sits in**, and the Gaussian **under**-predicts it | median `tail_ratio` **> 1.5** on the region arm, and **> 1.2** on the binary control, both at > 2 SE from 1.0 |
| H2 | the Bad mode is **right-skewed under max-pooling**, and materially less so without it | `shape_skew_neg` median **> 0.5** on `dinov3_patch/max_patch`; the paired `max_patch − whole_image` difference **> 2 SE** on the *same* embedder |
| H3 | at the shipped κ = 0.3 the anchors **do not move the fit** | median \|`anchored_dmu_lo`\| **< 0.01** score units, and `anchor_mass_frac` median **< 1e-3** |
| H4 | misfit **predicts regret** | OLS slope of `regret_honest` on `log(tail_ratio)` **> 0 at > 2 SE**, and partial R² **≥ 0.05** after conditioning on `n_pos` |

**H4 is the gate, and it is pre-registered to be falsifiable in the useful
direction.** If H1–H3 land and H4 does not, the finding is *"the Gaussian is
the wrong shape, the anchors are inert, and neither costs anything measurable at
the operating point"* — which closes #3329 for this fit cheaply and honestly
rather than launching a fit-replacement programme on an aesthetic objection.
That is a result worth the compute, not a null.

## Why this fit, out of the ~15 the app performs

An inventory of every fit in the shipped workflow is in
[the #3329 thread](https://github.com/samggreenberg/VTSearch/issues/3329). Three
things pick this one out:

- **Reach.** It is the only fit whose output is a user-visible decision on every
  item. The shipped fold-anchored mixture sets every trained detector's
  threshold; the unanchored `fit_score_gmm` sets the text-sort cut and the
  small-labelset fallback.
- **It is where the surviving error is.** #2883 showed the #2836 chain's
  dominant `transfer` term is an artefact of an optimistic reference point
  (+0.041 naive, −0.001 cross-fitted). With it removed, `misspecification`
  (+0.0129 cost) is the largest real term left — and `misspecification` is
  defined in `vtscore/eval/cut_rules.py` as exactly this: *"the classes are not
  Gaussian (a max-pooled Bad mode is an EVD)"*.
- **The prior line hands off to it and then stops.** #2881 closed the EVT *cut*
  line with "beating production needs a better **fit**, not a better cut". No
  run since has measured the fit.

## What is actually missing, and why nothing already answers this

Every fit diagnostic in the tree is **relative**:

| existing diagnostic | what it compares |
|---|---|
| `evt_loglik_gain` | Gumbel+Normal mixture vs 2-Gaussian mixture |
| `gmm_logit_loglik` | the same family on two axes |
| the `tau_*` chain | one cut rule against another |
| `s_mu_neg` / `s_var_neg` | the true classes' **first two moments** |

None compares a fit to **its own data**, so a misspecification both families
share cancels in every one of them. And the one label-supervised reading that
exists stops at two moments — which are precisely the statistics that *cannot*
see skew, i.e. cannot see the very EVD shape the geometry argument predicts.

`vtscore/eval/fit_quality.py` (new) supplies the absolute half: distance to the
fitted CDF (KS / CvM / AD), tail calibration at the cut, per-class shape on the
logit axis, and component-to-class identification. It reads labels, so it is
eval-only; it delegates every fit to the app's own functions, so there is no new
app surface mirrored and nothing for `check-eval-app-sync.py` to pin.

**No p-values are reported for the distributional statistics, deliberately.**
These fits see up to 50 000 points, where every test rejects every model; a
p-value would report the sample size. The bars above are all effect sizes.

## The grid

`vg_scale_any` × {`siglip`, `siglip+dinov3_patch`} × 12 categories × 8 seeds =
**192 cells**, 100 clicks, shipped defaults, `CALIB_SAFE_THRESHOLDS=1`.

**`vg_scale_any`, not `vg_scale`.** Every statistic here is prevalence-sensitive
— a tail ratio, a balanced accuracy, a class-conditional moment — and
`vg_scale_any` holds prevalence at **7.1 % in every cell** by construction (300
positives against a shared 3900-negative pool), on COCO-exhaustive labels with a
human review pass rather than VG free text (whose recall over these classes is
0.76). The band axis is the obvious follow-up once the instrument reads.

**The three geometries are the design**, and they isolate the two axes
independently:

| arm | role |
|---|---|
| `siglip / whole_image` | binary control |
| `dinov3_patch / whole_image` | same voting mode, different embedder |
| `dinov3_patch / max_patch` | same embedder, different voting mode |

H2 needs the third against the second, or a skew difference is just "DINOv3's
scores are shaped differently". On top of that, each `max_patch` run emits
**both poolings of the same media under the same model** (`sim:pooled` and
`sim:image`), which is the exactly-paired form of the same contrast with no
cross-run matching at all. Both readings are pre-registered; if they disagree,
the paired one governs and the disagreement is itself reported.

## The two scopes, and why they are not pooled

| scope | population | carries |
|---|---|---|
| `fold<i>` | fold *i* of the **shipped** `FoldAnchoredCut`, against its own haystack | distances, tail calibration, anchor mass |
| `sim:<geometry>` | the labelled sim set under one pooling | + class shape, identification |

The fold scope is label-free on purpose: a fold haystack is the unlabelled
remainder under *that fold's* model, while the sim labels live on the final
model's score scale, so attaching them would compare two scalings — the exact
mistake #2881's `pooled_mid` sanity check records.

**Nothing in `vtscore/eval/` has ever read `FoldAnchoredCut.fits`.** The fold
rows are the first observation of the mixture the app's threshold is actually
cut from; everything measured in this line so far has been the unanchored fit on
the sim set, which is a different fit on a different population.

## H3's mechanism, stated in advance

The anchored EM clamps labelled points one-hot in the E-step and counts them
`κ` times in the M-step, so the labels' entire influence is their M-step mass
share, `κv / (N + κv)`. At the shipped κ = 0.3 with 20 votes against a
50 000-point haystack sample that is **1.2 × 10⁻⁴** — and the folds anchor on
their *held-out* share of the votes, so the real figure is smaller still.

The prediction is therefore arithmetic, not empirical: the anchors cannot move
the fitted means by more than that share of the distance between the anchor mean
and the component mean. If H3 lands, the semi-supervised mixture is
**unsupervised in practice at production settings**, and the degeneracy guards
(`inverted_means`, `component_collapse`) are doing all the work the anchors were
introduced for — which would also mean the −0.021 identification bias measured
on the unanchored family carries over to the shipped path essentially unchanged.

H3 failing is informative too: it would mean the anchors bite through some route
other than mass, and the κ = 0.3 optimum found by the anchor-mass sweep needs a
different explanation than the one currently written down.

## What would make this run uninterpretable

Recorded in advance so they are not rationalised afterwards:

- **Fewer than 30 labelled items per class in the sim set**, which makes the
  shape statistics decline (they are third and fourth moments). At 7.1 %
  prevalence and this sim fraction the positive class is the binding one; if
  the realised count falls below the guard on a material share of steps, H2 is
  reported as **unresolved**, not as a null.
- **A degenerate-fit rate above ~20 %** on either arm. `fit_ok` is recorded per
  row; if the mixture routinely fails to fit at all, the statistics describe a
  filtered subset and the filter has to be reported beside every table.
- **The `sim:pooled` / `sim:image` pair disagreeing with the cross-arm
  contrast.** Both are pre-registered above; the paired one governs.

## Analysis, fixed in advance

`analyze_fitq_3329.py`, with `selftest_analyze_fitq_3329.py` (planted answers)
green **before** the array is submitted. Every contrast is paired within
(category, seed, step) and carries a standard error; two significant digits
unless a bar turns on a third. The mandatory
[quality-over-clicks pair](../../../scripts/experiments/calibration/curves.py)
and an interactive `viewer.html` ship with the report, plus the figure this
study exists to produce: **the score histogram with the fitted components
overlaid and the true classes coloured underneath**, at click checkpoints, for a
worked cell of each arm — the picture #3329 asks for, and the one no user or
prior run has ever seen.
