# A Gaussian-process head, conditioned on the votes — pilot (issue #3954)

**Question** (issue #3954, from a mathematically minded user): why not a Gaussian
process (GP) conditioned on the user's Good/Bad responses instead of the shipped
head? A GP gives a posterior over the whole haystack, so it answers two things
the app currently answers by other means: *how likely is this item a match*
(a calibrated probability, rather than a max-margin score squashed through a
sigmoid) and *which yes/no question is most informative next* (the posterior
spread, rather than the rank-nearest-cut Hard pick). The issue asks for the two
to be measured against the shipped algorithm on accuracy and calibration.

**What this is:** a CPU-box **pilot**, not the GRID study the question deserves.
One dataset (`caltech101_m`, SigLIP), six categories, five seeds, 150 clicks.
Its job is to build the arms into the harness, find out what breaks, and size
the follow-up — the follow-up is filed as its own issue (see the end). Every
number below is from this grid; every difference is paired on `(category,
seed)` and carries a standard error.

**Verdict — in one paragraph.** On the ranking the GP and the linear SVM are
indistinguishable here (AUROC 1.00 on every cell for every head: SigLIP
separates these six Caltech categories completely, so this dataset cannot rank
heads by ranking). On **probability calibration the GP is clearly better** from
16 labels on — Brier 0.036 vs 0.091 and ECE 0.18 vs 0.30 at 128 labels, paired
differences ten to a hundred times their standard errors — which is the one
claim the issue makes that this pilot can confirm. But **the app's threshold
machinery does not survive a GP's probabilities as they are**: the GP's score
scale moves with every refit (the marginal-likelihood fit re-scales the kernel
amplitude, and each added negative pulls the prior down), so a cut carried from
the calibration folds to the final model as a *raw score* lands under every
negative after ~20 votes and flags the whole haystack — on a ranking that is
perfect. Carrying the cut by *rank* fixes the collapse but under-admits, because
the fold models know fewer positives than the final model. The app's own
fold-less fallback (the schedule blend with a GMM on the final model's scores)
is the one shipped rule that holds. So the head question decomposes into two
that this repo already knows how to ask: the GP's calibration advantage is real
but the threshold that would let a user *see* it has to be fitted on the GP's
own haystack scores, not transferred from a re-fitted model. The
uncertainty-driven Hard picks are measured against the app's on the arms below.

<!-- STAGE_B_VERDICT -->

## What was built

Everything is in the tree and gated by `./run-tests.sh`:

- **Two sweep trainers**, `gp_rbf` and `gp_dot`
  (`vtscore/eval/sweep_trainers.py`): scikit-learn's
  `GaussianProcessClassifier` (a latent GP with a logistic likelihood, Laplace
  approximation) on the vote embeddings. `gp_rbf` is `amplitude · RBF(ℓ)` —
  every embedding is L2-normalised, so the kernel reads `2 − 2·cos`; `gp_dot` is
  `amplitude · DotProduct(σ₀)`, a Bayesian linear classifier and the closest GP
  analogue of the shipped linear head. Hyperparameters are fitted by marginal
  likelihood (ML-II) on every call; `gp_<kernel>@ls=…,amp=…,fixed` pins them.
  Both return `(P(positive), per-item std)` like the MLP ensembles, where the
  std is the posterior standard deviation of `sigmoid(f)` under the latent
  posterior, integrated by Gauss–Hermite quadrature so it is bounded like any
  std of a `[0, 1]` variable. The latent mean and variance are read off the
  fitted estimator with the same identities sklearn's own `predict_proba` uses.
- **A `gp_*` step-trainer path** (`vtscore/eval/step_trainers.py`) for the
  Autopilot voting simulation, exposing the spread as `StepModel.predict_std`,
  and a `standalone_cut="rank"` knob on `simulate_voting_iterations` that
  carries the cross-calibration cut from the fold models to the final one by
  rank (`_rank_transferred_threshold`) instead of as a raw score.
- **Two vote-order strategies** beside `autopilot`
  (`vtscore/eval/al_strategies.py`): `autopilot_uncertainty` keeps every
  Autopilot phase and replaces the Hard pick with the level-set *straddle*
  `argmax 1.96·std − |p − cut|` at the detector's own acquisition cut;
  `autopilot_maxvar` replaces it with `argmax std`. Both refuse to run against
  a trainer that reports no spread, so the app's picks can never be attributed
  to a GP rule.
- An **`ece` column** on every label-curve row (`vtscore/eval/label_curve.py`).
- The runner, [`scripts/experiments/gp_head/`](../../../scripts/experiments/gp_head/README.md).

## Design

**Data.** `caltech101_m` (838 images, the slice the MLP-vs-SVM study ran on),
SigLIP 768-d. Six query categories spanning common → rare, chosen by even rank
over the categories with an eval query and at least 16 positives:

<!-- GRID -->

Natural prevalence only: the slice has too few positives per category to thin
to 1% and keep a measurable test half. Five seeds. Every arm of a cell runs on
the same 50/50 simulation/test split and the same text-sort opening, so arms are
paired.

**Stage A — the head alone.** The label-curve sweep
(`run_label_curve_eval`): each estimator is given N balanced labels drawn from
the simulation half (N ∈ {8, 16, 32, 64, 128}; the sampler caps N at the
category's positives, so the larger grid points are smaller on the rarer
categories — `n_actual_mean` in the table says by how much) and scores the
held-out half. Ranking (AUROC, AP, best F1), the F1 at the cross-calibrated cut
(`f1_at_xcal`, the production threshold path), and two probability-calibration
diagnostics (Brier, ECE with ten equal-width bins). Trainers: `svm_linear` (the
standalone analogue of the shipped head), `gp_rbf`, `gp_dot`, and
`gp_rbf@fixed` (ℓ = 1, amplitude 1, no ML-II — the control for "is it the
kernel or the fit").

**Stage B — the head inside the loop.** The Autopilot voting simulation to 150
clicks, `calibrate_count=2`, the app's per-embedder calibration fraction (0.3),
inclusion 0. Eight arms:

| Arm | Trainer | Hard pick | Threshold rule |
|---|---|---|---|
| `app` | the shipped pipeline (linear-SVM head) | rank-nearest-cut | fold-anchored fusion — **the shipped detector** |
| `app_xcal` | the shipped pipeline | rank-nearest-cut | plain cross-calibration, raw score — the parity control |
| `gp_rbf` | RBF GP, ML-II | rank-nearest-cut | plain cross-calibration, raw score |
| `gp_rbf_rank` | RBF GP | rank-nearest-cut | cross-calibration carried by rank |
| `gp_rbf_blend` | RBF GP | rank-nearest-cut | the app's fold-less fallback: schedule blend of the cut with a GMM on the final model's haystack scores |
| `gp_dot_blend` | dot-product GP | rank-nearest-cut | blend |
| `gp_rbf_blend_straddle` | RBF GP | `autopilot_uncertainty` (straddle) | blend |
| `gp_rbf_blend_maxvar` | RBF GP | `autopilot_maxvar` | blend |

Why three threshold rules on one GP: the standalone arms **cannot** run the
shipped fused threshold — their fold models are not the app's head, so the
fold-anchored mixture has nothing to anchor on (`_safe_threshold_for_step`
lands them on the blend) — and the pilot's first pass found the raw cut
collapsing on the GP. `app_xcal` is the same head as `app` under a rule the GP
arms can share, so `gp_rbf − app_xcal` is the head with the rule held fixed
and `app − app_xcal` is what the shipped rule is worth on the shipped head.

**What is not here.** No rare-prevalence arm, one dataset, one embedder, no
region voting, no calibration-regret decomposition on the Stage B rows (the
harness emits the oracle cut only on the styled app path). The GRID study that
would settle the question is sized in the follow-up issue.

## Stage A — the head alone

<!-- STAGE_A -->

Reading it:

- **Ranking cannot separate the heads here.** AUROC is 1.00 for every trainer
  at every N, AP ≥ 0.997. Whatever the GP is worth, this dataset will not show
  it in the ranking, and neither would any dataset SigLIP separates this well.
- **The GP's probabilities are better calibrated, and it is the ML-II fit that
  makes them so.** From 16 labels on, `gp_rbf` and `gp_dot` beat `svm_linear`
  on both Brier and ECE, paired differences ten to a hundred standard errors
  from zero; at 128 labels Brier 0.036 vs 0.091 and ECE 0.18 vs 0.30. At 8
  labels the GP is *worse* (ECE 0.45 vs 0.42): with three positives ML-II keeps
  the amplitude small and the GP hedges near 0.5 everywhere. `gp_rbf@fixed`
  never catches up on ECE until 64 labels and is the worst arm on `f1_at_xcal`
  at every N ≥ 16, so the fixed-kernel control rules out "any RBF would do".
- **At the production cut the GP is a little better past 16 labels and worse
  at 8.** `f1_at_xcal` favours `gp_rbf` by +0.02 to +0.04 (p ≤ 0.05 at 16, 64,
  128) and `gp_dot` by +0.02 to +0.03, and disfavours both at 8 labels (−0.09
  and −0.13). Note this is the cut applied at a *fixed* N, where the fold and
  final models differ by 30% of the labels; Stage B is where the two drift
  apart.
- ECE is high for every head, because the categories are 2–27% prevalent and a
  score that is not tiny on the negatives is wrong on most of the haystack;
  read the *differences*, not the levels.

## Stage B — the head inside the loop

<!-- STAGE_B -->

## Follow-ups

<!-- FOLLOWUPS -->
