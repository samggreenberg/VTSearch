# `vtscore.training` - Learned-sort training primitives

Generic neural-net training and decision-threshold helpers extracted from
the detector pipeline. Everything in this package operates on raw numpy
arrays and PyTorch tensors - there are no media-type, embedder, vote,
or context dependencies. Library consumers can use it as a stand-alone
learned-sort toolkit: feed in `(N, D)` feature matrices and binary
labels, get back a trained classifier head, a calibrated threshold, and
(optionally) patch-level cosine scoring against a query vector.

The detector-specific glue that resolves votes → origins → embeddings →
training data lives one layer up in [`vtscore.detectors`](detectors.md);
this package is the underlying ML core.

## Contents

| Module                                                                | What it provides                                                |
|-----------------------------------------------------------------------|-----------------------------------------------------------------|
| `vtscore/training/mlp.py`                                             | `build_model`, `build_model_from_weights`, `train_model`        |
| `vtscore/training/thresholds/`                                        | Threshold helpers: `gmm`, `conformal`, `anchored`, `blend`, `costs`, `knobs` (all re-exported from `vtscore.training.thresholds`) |
| `vtscore/training/blend_schedules.py`                                 | Mix-in schedules for the safe-threshold blend                   |
| `vtscore/training/svm.py`                                             | `SVMClassifier`, `train_svm`, and `fit_linear_svm_head` (the production head's fit) |
| `vtscore/training/region_similarity.py`                               | Patch-level cosine scoring with bounding boxes                  |
| `vtscore/training/structural_similarity.py`                           | Stage-2 geometric re-rank: the inlier gate, and the returned-set line on documents (#4367, #4440) |
| `vtscore/training/structural_stage1.py`                               | The tiled Stage 1 for document pages (best tile per page), the shortlist size and the verification cache |
| `vtscore/training/query_sort.py`                                      | External-query sorts of the active dataset (example media, label files): `cosine_sort_active`, `example_sort_from_paths`, `train_and_score_active`, … |

The package `__init__.py` re-exports the head-building names and the eight
threshold functions below; everything else (`text_sort_threshold`, the SVM,
region and structural helpers) is imported from its submodule.

```python
from vtscore.training import (
    build_model, build_model_from_weights, train_model,
    calculate_gmm_threshold, conformal_threshold,
    calculate_cross_calibration_threshold, calculate_safe_threshold,
    calibration_folds, calibration_folds_cached, threshold_from_folds,
    fold_anchored_gmm_threshold,
)
from vtscore.training.thresholds import text_sort_threshold
from vtscore.training.svm import SVMClassifier, train_svm
from vtscore.training.region_similarity import (
    score_against_query, cosine_sort_with_boxes,
)
```

---

## Classifier-head trainer

A classifier that emits raw logits, built and trained from feature
matrices and binary labels. The `hidden_dim` argument selects the head:

### Architecture

**Linear SVM head - the production head.** Selected by the
`hidden_dim=LINEAR_SVM_HEAD` (`-1`) sentinel in `vtscore/training/mlp.py`:

```python
nn.Sequential(
    nn.Linear(input_dim, 1),
)
```

`train_model` routes this sentinel to `fit_linear_svm_head` in
`vtscore/training/svm.py` instead of running the BCE epoch loop. That function
delegates to `train_svm(kernel="linear")` - the very call the eval harness
scores as its `svm_linear` arm, so the shipped head and the measured arm cannot
drift apart - and copies the resulting hyperplane into the `Linear(input_dim, 1)`
module, whose forward pass is then the SVM's decision function. `dropout` is
ignored. Every production fit passes `LINEAR_SVM_HEAD` - in
`vtscore.detectors` for both the final model and the calibration fold models,
so the threshold is always calibrated on the head the final model has.

**Linear (logistic) head - eval harness and tests only.** The `LINEAR_HEAD`
(`0`) sentinel builds the *same* `Linear(input_dim, 1)` but fits it through
`train_model`'s balanced BCE-with-logits loop, which makes it logistic
regression. It is a named eval arm (`head="linear"`, see [eval.md](eval.md)),
not a production path.

**MLP head - eval harness and tests only.** Any `hidden_dim > 0`:

```python
nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, 1),
)
```

With only a handful of labelled positives the MLP is under-determined and its
retrains wobble, so it is an eval arm (see [eval.md](eval.md)), not a
production path. See
[`docs/ML.md`](../../../docs/ML.md#the-three-heads-which-one-is-shipped-and-why)
for why the SVM head ships.

All three are built by
`vtscore/training/mlp.py::build_model(input_dim, hidden_dim=64, dropout=0.0, generator=None)`
(`LINEAR_HEADS = (LINEAR_HEAD, LINEAR_SVM_HEAD)` both build the bare
`Linear`). Pass a seeded `torch.Generator` to
deterministically re-initialise the `Linear` weights (Kaiming uniform
on the weight matrix, uniform on the bias with the standard PyTorch
fan-in bound).

> Note the mismatch between the two defaults: `build_model`'s own
> `hidden_dim=64` builds an MLP, and `train_model`'s `hidden_dim=None`
> auto-sizes one. Neither default is the production head - callers that want
> it must pass `LINEAR_SVM_HEAD` explicitly.

### Auto-sizing the MLP hidden layer

Only the MLP head uses this; neither linear head has a hidden layer.
`_auto_hidden_dim(n_train)` in `vtscore/training/mlp.py` chooses the
hidden width from the number of training examples:

```python
return max(MLP_HIDDEN_MIN, min(MLP_HIDDEN_MAX, n_train // 3))
```

With the default `MLP_HIDDEN_MIN=8` and `MLP_HIDDEN_MAX=32` (from
`vtscore.config`), the heuristic keeps the model small when only a
handful of labels exist - n_train=10 picks 8 (floored), n_train=60 picks 20,
n_train=120 picks 32 (capped). `train_model(hidden_dim=None)` uses it.

### Training

`vtscore/training/mlp.py::train_model(X_train, y_train, input_dim, seed=42, hidden_dim=None, sample_weights=None)`
is the workhorse:

```python
import numpy as np, torch
from vtscore.training import train_model
from vtscore.training.mlp import LINEAR_SVM_HEAD

X = torch.from_numpy(np.random.RandomState(0).standard_normal((60, 512)).astype(np.float32))
y = torch.tensor([1.0] * 30 + [0.0] * 30).unsqueeze(1)

# hidden_dim=LINEAR_SVM_HEAD is what production passes; omitting it auto-sizes an MLP.
model = train_model(X, y, input_dim=512, hidden_dim=LINEAR_SVM_HEAD)
with torch.no_grad():
    scores = torch.sigmoid(model(X)).squeeze(1).cpu().numpy()
```

Behaviour shared by every head:

- **Inclusion does not enter training.** It is the unit cuts are measured
  in, applied later in `vtscore.training.thresholds.conformal_threshold`, so
  the trained model - and therefore every item's score - is independent of
  inclusion.
- **Class balance:** by default the fit balances class frequencies
  (`weight_true = num_false / num_true`, `weight_false = 1.0`, which is what
  `class_weight="balanced"` derives on the SVM path). Pass `sample_weights` to
  replace that balance entirely - that is how the region-flooding path
  expresses per-bag balancing.
- **Device:** the returned module lands on
  `vtscore.embedding.loader.get_torch_device()` via `ensure_torch_configured()`.

#### The SVM head (`LINEAR_SVM_HEAD`, production)

- **Objective:** squared hinge + L2, solved by liblinear via scikit-learn's
  `LinearSVC(C=config.SVM_HEAD_C` (env `VTSEARCH_SVM_HEAD_C`, default `1.0`)`, class_weight="balanced", dual="auto",
  max_iter=5000, random_state=seed)`. One blocking solve, not an epoch loop -
  so `TRAIN_EPOCHS`, `TRAIN_PATIENCE`, `MLP_DROPOUT` and `MLP_LABEL_SMOOTHING`
  do not apply, and a cancelled background job is checked once up front rather
  than per epoch.
- **Backend:** always sklearn on the CPU. The fit is milliseconds at any real
  vote count, and cuML's `LinearSVC` takes neither a seed nor per-row weights,
  so a GPU fit would buy nothing and cost reproducibility.
- **Determinism:** liblinear is deterministic given `random_state`, so the same
  `(X, y, seed)` always yields the same hyperplane - no RNG forking needed.

#### The BCE heads (`LINEAR_HEAD` and the MLP, eval arms)

- **Loss:** weighted `BCEWithLogitsLoss(reduction="none")`, with the per-sample
  weights described above.
- **Label smoothing:** targets are smoothed by `MLP_LABEL_SMOOTHING` after the
  class weights are derived from the hard labels, so a strongly-fit model
  can't saturate every score to an exact 0.0/1.0 sigmoid and collapse the
  conformal rule's quantiles into a single tie.
- **Optimiser:** `Adam(lr=0.001, weight_decay=1e-4)`.
- **Early stop:** trains up to `config.TRAIN_EPOCHS` (default 200) and
  stops after `config.TRAIN_PATIENCE` consecutive epochs with no
  improvement larger than `min_delta = 1e-4`. Read fresh at every call,
  so monkey-patching `vtscore.config.TRAIN_EPOCHS` for tests works.
- **Mixed precision:** enabled automatically on CUDA via `torch.amp` /
  `GradScaler`; CPU and MPS use FP32 so deterministic training is
  bit-for-bit reproducible.

### Thread safety and reproducibility

On the BCE heads, `train_model` deliberately avoids touching PyTorch's global
RNG:

```python
g = torch.Generator()
g.manual_seed(seed)
model = build_model(input_dim, hidden_dim=hidden_dim,
                    dropout=MLP_DROPOUT, generator=g)
...
with torch.random.fork_rng(), torch.enable_grad():
    torch.manual_seed(seed)
    for _ in range(epochs):
        ...
```

The local `torch.Generator` seeds weight initialisation; `fork_rng`
isolates the dropout RNG inside the training loop (neither linear head has
dropout, so that half matters only for the MLP arm). Two concurrent
`train_model` calls in different threads do not interfere with each
other's seed, and either call produces the same model given the same
`(X, y, seed)`. The SVM head gets the same guarantee for free - liblinear
touches no global RNG at all. This matters because cross-calibration trains *k* fold
models in sequence and the eval harness can run multiple seeds in
parallel.

### Reloading from saved weights

`build_model_from_weights(weights)` in `vtscore/training/mlp.py`
reconstructs a model from a dict of lists (the output of
`tensor.tolist()` per state-dict entry). It infers the head from the keys
present: `0.*` alone means a linear head, while a `3.weight` means an MLP
whose hidden width is the length of `0.bias`. The two linear heads are
indistinguishable here by design - they have the same architecture, and which
objective produced the numbers is irrelevant once the numbers are in hand. It also remaps the
legacy 3-layer MLP key format (`0.*`, `2.*`).

> **No persisted model weights.** Detector JSON files store labelsets
> (origins + labels) only; the head is re-derived on every load (see
> [`detectors.md`](detectors.md) and
> [architecture.md](../architecture.md#the-no-persisted-vectors-rule)).
> `build_model_from_weights` is for callers with their own reason to ship
> weights around; the detector pipeline never calls it.

---

## Decision thresholds

All threshold helpers operate on score lists and label lists and return
a `float` (`fold_anchored_gmm_threshold` returns `(threshold, provenance)`).
Detector-specific glue (sourcing the score/label lists from votes, caching
on `DetectorContext`) sits one layer up. The measurements behind each rule
are summarised in
[`docs/ML.md`](../../../docs/ML.md#threshold-calibration).

| Function                                  | When it fires                                                 |
|-------------------------------------------|---------------------------------------------------------------|
| `calculate_gmm_threshold`                 | All-media score distribution - used by the safe blend         |
| `text_sort_threshold`                     | A cosine/text sort's line - the midpoint, or the guarded rule behind `VTSEARCH_TEXT_SORT_CUT` |
| `conformal_threshold`                     | Conformal inclusion rule on one (scores, labels) set          |
| `calculate_cross_calibration_threshold`   | k-fold cross-calibration, in one call                         |
| `calibration_folds` / `calibration_folds_cached` | The inclusion-*independent* half: fit the folds       |
| `threshold_from_folds`                    | The inclusion-*dependent* half: apply the rule to fitted folds |
| `fold_anchored_gmm_threshold`             | The shipped cut - fold mixtures anchored on held-out labels    |
| `calculate_safe_threshold`                | Blends cross-cal with GMM when label counts are low           |
| `precision_floor_cut`                     | The largest set whose #4220-estimated precision clears a floor; off the line's path since #4272, and off the Find Stats chart since #4360 |
| `reporting_line`                          | The estimator's own line at an operating point; the app hands it no estimate any more (see `balance_line`) |
| `balance_schedule` / `SpotCheck` / `likely_range` | The balance's spot check (#4272, #4413): the cap, bands and picks a walk costs, the walk itself, and the likely ranges a checked set carries |
| `LineRanking` / `balance_line` / `balance_state` | The ranking a detector's line keeps a set of, the line the balance draws over it, and the state every response carries |

### `text_sort_threshold(scores, rule=None)`

`vtscore/training/thresholds/gmm.py` (import from `vtscore.training.thresholds`).
The line a **typed-query** sort draws - called by
`vtscore/training/query_sort.py::cosine_sort_active` for `role="text"` and by
the eval harness's Autopilot opening. Example and label-file sorts keep
`calculate_gmm_threshold`. The rule comes from `rule=`, else
`TEXT_SORT_CUT_RULE` (env `VTSEARCH_TEXT_SORT_CUT`):

- `gmm_midpoint` (the default): exactly `calculate_gmm_threshold`.
- `guarded_tail`: `guarded_text_sort_threshold`. If the shipped fit's two
  components are separated (Ashman's D >= `TEXT_SORT_SEPARATION_D` = 2), it
  continues that fit to convergence (`converge_score_gmm`) and cuts at the
  midpoint. Otherwise it cuts at median + `TEXT_SORT_TAIL_K` (3) x the
  lower-half-MAD sigma (`bulk_location_scale`).

`guarded_tail` admits far fewer non-matches on a typical one-mode text sort,
but is off by default because it made Autopilot's opening worse in an A/B
(`docs/experiments/2026-09-23-text-cut-ab-3826/REPORT.md`).

### `calculate_gmm_threshold(scores)`

`vtscore/training/thresholds/gmm.py`. Fits a 2-component 1-D Gaussian
mixture to the score list and returns the **midpoint between the two
component means**. Used to produce a reasonable operating point even when
only a few labels exist - the score distribution still tends to be bimodal
because the embedder space already separates "kind of like X" from "kind of
not like X".

The fit is `fit_score_gmm`: a deterministic 2-means init and EM, stopping
when an iteration improves the mean log-likelihood by less than 1e-3 (the
same criterion as sklearn's `GaussianMixture`, which `fit_score_gmm_sklearn`
keeps available for comparison but which production no longer calls).
`fit_gmm_threshold(scores)` returns the cut together with the `GmmFit1D`.
`_weighted_gaussian_crossing` / `GmmFit1D.crossing_or_midpoint` (an
equal-density-crossing cut) remain as eval variants only.

Falls back to `np.median(scores)` when GMM fitting raises (e.g. degenerate
score distributions), and to `0.5` when fewer than 2 scores are provided.

### The anchored refit

`fit_anchored_score_gmm` initialises from the unanchored fit and then runs
`_anchored_em` with the votes clamped as anchors (`ANCHOR_WEIGHT_DEFAULT`).
It stops when the weighted semi-supervised log-likelihood (anchors included)
improves by less than `_ANCHORED_EM_LOGLIK_TOL` = 1e-8, or at
`_ANCHORED_EM_MAX_ITER`. The tolerance is deliberately far tighter than the
unanchored 1e-3: a loose stop halts before the minority component has
migrated to the high mode
(`docs/experiments/2026-09-13-anchored-em-stop-3825/REPORT.md`).

- **`stats`**, an optional out-dict on `_anchored_em` and
  `fit_anchored_score_gmm`, carries `n_iter`, `converged` and the `loglik` the
  stopping decision was taken on.
- **`FoldAnchoredCut.n_unconverged`** counts folds that hit the cap, and the
  provenance string names it (`fold_anchored_maxiter2[2/2]`). The `[a/k]`
  group stays last because `vtscore.eval.row_metrics.folds_used` parses it
  with an end-anchored regex.

### `conformal_threshold(scores, labels, inclusion_value=0)`

`vtscore/training/thresholds/conformal.py`. Split-conformal quantile rule mapping
`inclusion_value` (integer in [-10, 10]) to a threshold over held-out
calibration scores. For `k = inclusion_value` (with
`CONFORMAL_BASE_BUDGET = 0.25`, `CONFORMAL_QPOS_MAX = 0.75`):

- A false-negative **cap** `alpha = min(1, 0.25 * 2^-k)`: the threshold
  never exceeds the alpha-quantile of the calibration *positive* scores,
  so at most an estimated `alpha` of true matches is missed. The cap is
  an upper bound, not a target - with cleanly separated classes the cut
  drops to the lowest calibration positive and the budget goes unspent.
- A false-positive **guard** for `k <= 0`: the threshold stays at or
  above the `1 - 0.25 * 2^k` quantile of the calibration *negative*
  scores, and above a walk up the positive score distribution
  (`q_pos-level = 0.75 * |k| / 10`; at -10 only the top quartile of
  positives remains).

Monotone non-increasing in `k` by construction, so included sets are
nested as the knob rises. Returns `0.5` when the input is empty or
single-class. (`INCLUSION_MIN` / `INCLUSION_MAX` live in `thresholds/knobs.py`.)

### `calculate_cross_calibration_threshold(...)`

`vtscore/training/thresholds/conformal.py::calculate_cross_calibration_threshold(X_list, y_list, input_dim, inclusion_value=0, rng=None, calibrate_count=2, calibration_fraction=0.5, hidden_dim=None, groups=None, score_rows_by_group=None)`.
Cross-calibration in one call (the interactive path uses the split form
below). For each of `calibrate_count` rounds:

1. Randomly split `(X_list, y_list)` into Train (`1 - calibration_fraction`)
   and Calibrate (`calibration_fraction`).
2. Train a head on Train via `train_model` (the caller passes `hidden_dim`;
   the detector pipeline passes `LINEAR_SVM_HEAD`).
3. Score the held-out Calibrate portion.

Pools every round's held-out (score, label) pairs and applies
`conformal_threshold` once via `threshold_from_fold_orderings` (pooling
rather than per-fold averaging maximises the quantile rule's
resolution). Defaults to 0.5 when `n < 4` or fewer than 2 of either
class; returns `NO_GOOD_THRESHOLD` when the split would leave fewer
than 2 training examples or 1 calibration example.

```python
from vtscore.training import calculate_cross_calibration_threshold
from vtscore.training.mlp import LINEAR_SVM_HEAD

t = calculate_cross_calibration_threshold(
    X_list=[v for v in feature_vectors],
    y_list=[1.0, 0.0, 1.0, 0.0, ...],
    input_dim=512,
    inclusion_value=0,
    calibrate_count=2,
    calibration_fraction=0.5,
    hidden_dim=LINEAR_SVM_HEAD,   # match the full-data model's head
)
```

The fold models honour `hidden_dim` when provided - important for the
detector pipeline, which wants fold thresholds calibrated against a
model fitted the same way as the final full-data model, and therefore
passes `LINEAR_SVM_HEAD` on both sides. `LINEAR_HEAD` or a positive
`hidden_dim` here calibrates against logistic or MLP fold models, which is
what the eval harness's head-sweep arms want and nothing else does.

### `calibration_folds_cached(...)` + `threshold_from_folds(...)`

The interactive path splits the work in two, because only half of it
depends on where the cut goes:

```python
from vtscore.training import calibration_folds_cached, threshold_from_folds

folds = calibration_folds_cached(          # expensive: fits `calibrate_count` fold models
    X_list, y_list, input_dim,
    calibrate_count=2, calibration_fraction=0.5,
    hidden_dim=LINEAR_SVM_HEAD, det_ctx=det_ctx,
)
threshold = threshold_from_folds(folds, inclusion_value=0)   # cheap: a quantile rule
```

`CalibrationFolds` is a `NamedTuple` of `(orderings, fallback, models)`. Pass
`holdout_sink=[]` to either call to also receive, per fold, the training row
behind each held-out score, so a caller can tell which votes a fold held out.
`calibration_folds_cached` memoises it on `det_ctx.calibration_cache` under a
deterministic key built from `X_list`, `y_list`, the calibrate settings,
`hidden_dim`, and any `score_rows_by_group` - so a re-cut at another
inclusion (the acquisition cut, Smart's pricing) re-runs only the cheap rule,
with no fold refits. A real label change produces a different key and falls through to a
fresh calibration; no explicit invalidation is needed.

The key bytes encode the actual training vectors (not just label IDs),
so if the embedder changes and a labelset is re-resolved to different
embeddings, the cache invalidates automatically. The fitted fold *models*
ride along in the tuple because the shipped fold-anchored cut needs to
re-score the haystack with them.

### `calculate_safe_threshold(xcal_threshold, all_scores, ctx, schedule=None)`

`vtscore/training/thresholds/blend.py`. Combines the cross-calibration
threshold with a GMM threshold computed on the full score distribution.
The cross-cal output gets noisy when labels are few, so a **mix-in
schedule** (`vtscore/training/blend_schedules.py`) decides how much of
each to use. `ctx` is a `BlendContext` carrying the vote counts (total,
good, bad — in votes, not flooded rows); a bare `int` is accepted where
only the total is known.

**These schedules are the fused threshold's fallback only**: they run on
steps with no usable calibration folds, where the cross-cal side is the
`NO_GOOD_THRESHOLD` sentinel (in practice a small fraction of early steps).

The shipped schedule depends on the **voting mode**
(`PRODUCTION_SCHEDULE_BY_MODE`; `production_schedule_for(region_voting=...)`
resolves it, falling back to `PRODUCTION_SCHEDULE`):

| mode | schedule | shape |
|---|---|---|
| region (patch dataset) | `slow_cap50` | pure GMM ≤6 labels, ramping to **half** cross-cal at 40 and held there |
| binary (single vector) | `corridor20` | clamp the x-cal cut to 0.2 of the way from the GMM midpoint to each component mean; `cap50` when no fit exists |
| unknown | `cap50` | the one arm that improved both modes under every weighting |

`SAFE_BLEND_SCHEDULES` / `schedule_names()` / `get_schedule(name)` expose
the full registry (the old 6→20 linear ramp is kept as `prod`, a baseline);
entries vary the endpoints, the curve shape, the statistic the ramp reads,
or replace the weighted average with a clamp.

When `xcal_threshold` is `float("inf")` (no valid fold split), falls
back entirely to the GMM threshold.

### `precision_floor_cut(floor, corpus_scores, pool_scores, fold_orderings, fold_haystacks, ...)`

`vtscore/training/thresholds/precision_floor.py`. The #4220 estimator's cut
(#4245), ported from the estimator #4220 measured
([`docs/experiments/2026-09-28-precision-frames-4220/REPORT.md`](../../../docs/experiments/2026-09-28-precision-frames-4220/REPORT.md)).
It returns the largest top-*k* of the corpus whose **lower-bound** estimated
precision is at least `floor`:

- each calibration fold's held-out vote scores are ranked in that fold's own
  haystack (`fold_rank_evidence`), so every fold shares one coordinate;
- `P(positive | percentile)` is fitted on them (`fit_posterior`, logistic by
  default) and applied to the corpus through the pool's percentiles;
- the precision curve is read at the 10th percentile of 30 bootstrap refits
  (`precision_lower_bound_curve`), after an EM re-estimate of the corpus prior
  (`em_prior_shift`) for when the corpus is not the voted pool.

The result is a `PrecisionFloorCut` in one of three `PrecisionFloorStatus`
states: `promised` (with the threshold, the count and the estimate),
`unreachable` (enough evidence, but no cut clears the floor; the best bound is
reported), or `insufficient_evidence` (fewer than `MIN_CALIBRATION_POSITIVES`,
10, positives among the calibration votes). The transfer `coordinate`
(`"percentile"` or `"tail"`), the bound level and the refit count are
parameters because #4221 is still pricing them.

One call is `fit_precision_floor_curve(...)` (the costly half: the bootstrap
refits) followed by `PrecisionFloorCurve.cut(floor)` (a scan), so a caller
cutting one corpus at several floors keeps the curve.
`PrecisionFloorEstimate(corpus_scores, fold_orderings, fold_haystacks)` holds one
detector's inputs: it fits the curve the first time a floor is asked for,
reads it off a seeded sample above 50k scores, and counts `n_returned` on the
whole corpus. The app no longer builds one (#4362); the estimator is library
API for a caller that wants its reading.

**Which row backs each held-out score.** Pass `holdout_sink=[]` to the
calibration (`compute_fold_orderings`, `calibration_folds`,
`calibration_folds_cached`) to learn which training row backs each held-out
score; the sink is read-only. `eligible_fold_orderings(orderings, holdout_rows,
eligible_rows)`, which cut the app's evidence down to the votes
`vtscore.datasets.vote_provenance.calibrates_precision` accepted (#4245), is
deprecated with that filter (#4362): it still answers, with a
`DeprecationWarning`.

### `reporting_line(cut, estimate, *, inclusion_value, min_precision)`

The estimator's own line at an operating point. `min_precision=None` is no
floor: `cut.threshold_at(inclusion_value)`, where *inclusion_value* is the
internal unit, not a user preference (#4269) - the app passes
`PRECISION_FLOOR_FALLBACK_INCLUSION`, and a re-cut passes the acquisition or
Smart inclusion. With a floor it says what the #4220 estimate says: a floor
that is `promised` draws the estimate's own threshold, one that promises
nothing draws the `PRECISION_FLOOR_FALLBACK_INCLUSION` (0) cut. **The app
does not draw its line here** (#4272, #4413): it passes `min_precision=None`
and no estimate, and draws the balance's line with `balance_line` below,
coming here only when there is no balance or no ranking to keep a set of. The
returned
`ReportingLine` carries the verdict, and `line_inclusion` gives the inclusion
Autopilot's acquisition offset starts from - derived from the line itself when
no inclusion drew it.

### The spot check: `balance_schedule`, `SpotCheck`, `likely_range`, `LineRanking`, `balance_line`, `balance_state`

`vtscore/training/thresholds/spot_check.py` (#4272; the band walk of #4388,
stopped at the F-beta peak by #4413 and priced in
[`docs/experiments/2026-10-01-fbeta-line-4411/REPORT.md`](../../../docs/experiments/2026-10-01-fbeta-line-4411/REPORT.md)).
The line's preference is a **balance**: F-beta's *beta*, with the presets
`BALANCE_PRESETS` (0.5, 1, 2; `DEFAULT_BETA` 1) and any value in
`[BETA_MIN, BETA_MAX]` = `[0.25, 4]` accepted. Under a balance the line keeps
a **set**, and a spot check of uniform random picks from it estimates how much
of it is right and how much of the corpus's positives it found; the precision
range comes only from those picks, never from a model.

- `balance_schedule(beta)` is the cap and what a walk from it costs: the
  starting candidate `K` (`CHECK_BASE_CANDIDATE`, the top 32, at beta <= 1;
  `CHECK_RECALL_CANDIDATE`, the top 128, above it), the bands that hold it
  (`rounds_for(K)`: 3 for 32, 5 for 128), and the picks a band
  (`CHECK_MIN_PICKS`, 5). `check_shape(beta)` is how a check treats the line
  (#4427): `CHECK_ADVISORY` at beta <= 1 (the walk informs the line and never
  moves it), `CHECK_TRIM` above (the walk may only step shallower, and the
  line takes its end). `resolve_line_knobs(beta)` resolves an eval arm's knob:
  `None` is `DEFAULT_BETA`, `NO_BALANCE` (`"off"`) is no balance (the
  Inclusion arm), and a number is validated.
- `LineRanking.from_scores(ids, scores, voted)` is the ranking the line is
  drawn over: sorted, unscorable items dropped, the trainer's voted items
  marked. `candidate(count, also_voted)` is the top *count* unvoted ids,
  `threshold_for(count, also_voted)` the line that keeps them (the last
  item's score, floored to the four decimals responses carry - `line_under` -
  so the item clears its own line however it is compared), `above(threshold)`
  the count at or above it, and `fingerprint(count, also_voted)` the set's
  identity for the `stale` flag.
- `mixture_positives(ranking, labels, also_voted)` is the vote-anchored
  mixture's count of positives among the unvoted items (`mixture_posterior`,
  fitted once per ranking and memoised on it), `walk_positives(...)` the
  count a walk reads recall against (the mixture's, else the cap lowered to
  what is unvoted, #4419), and `fbeta_count(ranking, beta, labels,
  also_voted)` the count at which the mixture's F-beta peaks: the unchecked
  line's proposal.
- `SpotCheck.start_balance(ranking_ids, beta, n_pos, *, seed=None,
  start_count=None)` fixes the unvoted ranking and *n_pos*, cuts the ranking
  into bands (`band_edges`: the top 8, the next 8, 16, 32, ...), starts at the
  bands holding the cap, and deals the first band's picks (`draw`);
  `record({id: right})` takes a band's labels and, once the band is audited,
  deals the next band the set under test owes or decides it. The set's F-beta
  estimate (`fbeta_estimate()`) is its band-stratified positives - each
  band's share of right picks (`estimate()` is the set's) times the band's
  size - over *n_pos*. The walk goes one band deeper while the estimate rises
  and ends on the peak the first time it falls; from a start whose first
  deeper step falls, it goes one band shallower while the estimate does not
  fall. Under `trim` it only steps shallower from the start. It ends
  `BALANCE_CHECKED` on the peak's band edge, a tie keeping the smaller set.
  `range()` and `recall_range()` are the set's likely precision and recall
  ranges, its bands' intervals weighted by size (recall over *n_pos*);
  `as_dict()` the state a client sees (the band pending, the set's `bands`,
  the walk's `direction`, the `estimate`, `fbeta`, `recall`);
  `is_stale(ranking, also_voted)` whether the set moved since the check. The
  other keywords (`picks`, `tol`, `fine`, `guard`, `shallow_only`) are
  #4427's harness arms, all off in the app. `CHECK_PROVENANCE` is the
  provenance its votes are recorded with.
- `likely_range(right, labelled, candidate, tail)` is a Clopper-Pearson
  interval with each tail at `range_tail(bands)` = alpha / bands (split over
  the set's bands, so their size-weighted mean holds by the union bound),
  exact once the labels cover the band; `clopper_pearson_lower` / `_upper` are
  the bounds.
- `balance_count(beta, result, proposal, shape)`, `balance_line(ranking,
  beta, result, also_voted, proposal, shape)` and `balance_state(beta,
  result, ranking, also_voted, proposal, shape)` are the rule the app's
  retrain, re-cut and the eval harness's default arm share: the set the
  finished walk ended on where the check's shape lets it move the line
  (`applicable_balance`: a result belongs to the beta it was run at), else the
  unchecked count - the cap, lowered to the *proposal* (`fbeta_count`) - and
  the `BalanceState` every response carries (`status` in `BALANCE_STATES`:
  `unchecked` / `checked`, `count`, `precision`, `recall`, `fbeta`, `stale`,
  `schedule`, `shape`, `audited`). `aim_words(state)` names the balance in a
  message (`at F1`).

`scripts/check-eval-app-sync.py` pins the band schedule, `likely_range` and
`SpotCheck` against the analysis scripts that priced them.

---

## SVM

`vtscore/training/svm.py` holds the general SVM trainer. Its
`fit_linear_svm_head(X, y, input_dim, *, seed=42, sample_weight=None)` is
what `train_model` calls for `LINEAR_SVM_HEAD` - it runs
`train_svm(kernel="linear")` and copies the hyperplane into a
`Linear(input_dim, 1)` module. The rest of the module (RBF kernels,
probability calibration) serves the eval harness's head sweeps (see
[`vtscore.eval.label_curve`](eval.md#label-curve)).

### `SVMClassifier` (`vtscore/training/svm.py`)

Dataclass wrapping a fitted sklearn estimator plus an optional
probability source: `base` (`LinearSVC` or `SVC`), `calibrator`
(`CalibratedClassifierCV` or `None`), `scaler` (optional
`StandardScaler`), plus `kernel` and `calibration` tags.
`predict_proba(X)` returns a 1-D `float32` array in `[0, 1]`. When a
calibrator is fitted, it consults `calibrator.predict_proba(...)`;
otherwise it sigmoids the raw `decision_function` (clipped to ±30).
The sigmoid wrapper is not a true probability, but it is monotone in
the SVM score - which is all the ranker and threshold-finder need.

### `train_svm(X, y, *, kernel="linear", C=1.0, gamma="scale", calibration="decision_sigmoid", inclusion_value=0, seed=42, standardize=False, backend="auto", sample_weight=None, ...)`

Fits a `LinearSVC` (linear, fast) or `SVC` (RBF), translates
`inclusion_value` into a sklearn `class_weight` map, and optionally
wraps the result in `CalibratedClassifierCV`. Calibration modes:
`"decision_sigmoid"` (default; no CV cost), `"sigmoid"` (Platt),
`"isotonic"`, `"auto"` (picks by per-class label counts, degrades
gracefully when CV is infeasible). The default is intentional:
VTSearch picks its operating threshold via cross-calibration, not
from the model's raw probability, so burning training data on k-fold
CV calibration would shrink the final-fit data while inverting
`inclusion_value`. Raises `ValueError` for fewer than 2 samples or
single-class inputs, matching `train_model`, which refuses single-class data
up front for the same reason: BCE has no discriminative signal there.

---

## Region similarity (patch-level scoring)

`vtscore/training/region_similarity.py` is the entry point for
embedding-cosine sort when the dataset's media expose per-patch
embeddings (DINOv2, DINOv3, EUPE). It dispatches to a fast vectorised
numpy path when no patches are present, so SigLIP / CLIP datasets see
zero overhead.

| Function                                              | Behaviour                                                                |
|-------------------------------------------------------|--------------------------------------------------------------------------|
| `score_against_query(media, query_vec, embedder_name=None)` | Returns `(max_cosine_similarity, best_region_box)` for one media. For patch media, scores every row of `media_score_rows` (image-level vector + every raw patch) and returns the max + that row's box. For single-vector media, returns the cosine plus `(0.0, 0.0, 1.0, 1.0)`. `(0.0, None)` on zero-norm or missing embedding. |
| `cosine_sort_with_boxes(snap, query_vec, embedder_name=None, *, region_aware=None)` | Snapshot-level scorer. Per-snapshot dispatch: patch snapshots use the cached flattened float16 score-row matrix + a chunked matvec and segmented max-pool (K = 1 + H·W, 197 on DINOv3); single-vector snapshots use the cached `(N, D)` matrix via `vtscore.embedding.matrix.get_embedding_matrix_for_snap`. Returns `(results_sorted_desc, raw_similarities_in_input_order)`. Result entries are `{"id": cid, "similarity": float, "best_region": [x0, y0, x1, y1]?}`. |

```python
from vtscore.training.region_similarity import cosine_sort_with_boxes

results, sims = cosine_sort_with_boxes(snap, query_vec)
top_ten = results[:10]
```

---

## Invariants worth restating

- **No persisted model weights.** Library callers that load a model from
  disk are expected to re-derive it from a labelset's origins (see
  [`detectors.md`](detectors.md)). `build_model_from_weights` is a
  utility, not a contract.
- **Config read at call time.** `train_model` reads
  `vtscore.config.TRAIN_EPOCHS` / `TRAIN_PATIENCE` / `MLP_HIDDEN_MIN` /
  `MLP_HIDDEN_MAX` / `MLP_DROPOUT` / `SVM_HEAD_C` at call time, so tests
  can monkey-patch them.
- **Thread-safe RNG.** `train_model` uses `torch.random.fork_rng` so
  parallel training calls don't interfere; cross-calibration uses an
  optional `np.random.RandomState` (seeded with 42 by the cached
  wrapper) so two threads sharing the cache still get deterministic
  thresholds.
- **No settings lookups in the math.** Threshold/training inputs are
  function arguments. (`query_sort.py` is the exception by design: it reads
  the active dataset through `vtscore.state`.)
