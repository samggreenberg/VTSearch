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
| `vtscore/training/thresholds.py`                                      | GMM / cross-cal / safe threshold helpers                        |
| `vtscore/training/blend_schedules.py`                                 | Mix-in schedules for the safe-threshold blend                   |
| `vtscore/training/evt_mixture.py`                                     | Gumbel + Normal score mixture - the extreme-value cut           |
| `vtscore/training/svm.py`                                             | `SVMClassifier` + `train_svm` prototype                         |
| `vtscore/training/region_similarity.py`                               | Patch-level cosine scoring with bounding boxes                  |
| `vtscore/training/structural_similarity.py`                           | Stage-2 geometric re-rank + match-statistic verification classifier |

The package `__init__.py` re-exports the head-building and threshold names; SVM
and region-similarity helpers are imported from their submodules.

```python
from vtscore.training import (
    build_model, build_model_from_weights, train_model,
    calculate_gmm_threshold, conformal_threshold,
    calculate_cross_calibration_threshold, calculate_safe_threshold,
    calibration_folds, calibration_folds_cached, threshold_from_folds,
    fold_anchored_gmm_threshold,
)
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
ignored (a bare linear map has nothing to regularise with dropout). Every
production fit passes `LINEAR_SVM_HEAD`: `vtscore/detectors/training.py` does it
for both the final model and the calibration fold models (so the threshold is
always calibrated on the head the final model has), `labeling_progress.py` does
it for the per-step stopping-condition models, and `labelset_training.py`
inherits it by going through `train_and_threshold`.

**Linear (logistic) head - eval harness and tests only.** The `LINEAR_HEAD`
(`0`) sentinel builds the *same* `Linear(input_dim, 1)` but fits it through
`train_model`'s balanced BCE-with-logits loop, which makes it logistic
regression. This was the production head between the threshold-stability work
(#2790) and the switch to the SVM; it survives as a named eval arm
(`head="linear"`, see [eval.md](eval.md)) and in unit tests, and is not
reachable from the app.

**MLP head - eval harness and tests only.** Any `hidden_dim > 0`:

```python
nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, 1),
)
```

This was the production head until the threshold-stability work (#2790): with
only ~3-5 labelled positives the MLP is under-determined, each retrain wobbles
the scores, and the calibrated cut lurches. It survives for the eval harness's
head-sweep arm (see [eval.md](eval.md)) and unit tests, and is not reachable
from the app. See
[`docs/ML.md`](../../../docs/ML.md#the-three-heads-which-one-is-shipped-and-why)
for the measurements behind both moves.

All three are built by
`build_model(input_dim, hidden_dim=64, dropout=0.0, generator=None)` at
`vtscore/training/mlp.py`. Pass a seeded `torch.Generator` to
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
n_train=120 picks 32 (capped). The function is private but stable; the eval
harness's `_resolve_hidden_dim` (`vtscore/eval/voting_iterations.py`) calls it
for the `"mlp"` arm. The detector code no longer does - it passes
`LINEAR_SVM_HEAD` for both the final model and the cross-calibration fold
models, so fold thresholds stay directly comparable to the full-data model.

### Training

`train_model(X_train, y_train, input_dim, seed=42, hidden_dim=None, sample_weights=None)`
in `vtscore/training/mlp.py` is the workhorse:

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

- **Inclusion does not enter training.** It is a pure threshold knob applied
  later in `vtscore.training.thresholds.conformal_threshold`, so the trained
  model - and therefore every item's score - is independent of inclusion.
- **Class balance:** by default the fit balances class frequencies
  (`weight_true = num_false / num_true`, `weight_false = 1.0`, which is what
  `class_weight="balanced"` derives on the SVM path). Pass `sample_weights` to
  replace that balance entirely - that is how the region-flooding path
  expresses per-bag balancing.
- **Device:** the returned module lands on
  `vtscore.embedding.loader.get_torch_device()` via `ensure_torch_configured()`.

#### The SVM head (`LINEAR_SVM_HEAD`, production)

- **Objective:** squared hinge + L2, solved by liblinear via scikit-learn's
  `LinearSVC(C=config.SVM_HEAD_C, class_weight="balanced", dual="auto",
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
objective produced the numbers is irrelevant once the numbers are in hand. It also silently remaps the
legacy 3-layer MLP format (`0.*`, `2.*`) to the current keys, so old detector
files don't have to be migrated.

> **Invariant - no persisted model weights.** In VTSearch proper, detector
> JSON files store labelsets (origin info + per-element labels) only;
> the head is re-derived from those origins on every load. See
> [`detectors.md`](detectors.md) for the detector storage contract.
> `build_model_from_weights` exists for callers (eval harnesses, third-
> party tooling) that have their own reason to ship weights around - the
> detector pipeline never calls it on the production path.

---

## Decision thresholds

All threshold helpers operate on score lists and label lists and return
a single `float`. Detector-specific glue (sourcing the score/label
lists from votes, caching on `DetectorContext`) sits one layer up.

| Function                                  | When it fires                                                 |
|-------------------------------------------|---------------------------------------------------------------|
| `calculate_gmm_threshold`                 | All-media score distribution - used by the safe blend         |
| `conformal_threshold`                     | Conformal inclusion rule on one (scores, labels) set          |
| `calculate_cross_calibration_threshold`   | k-fold cross-calibration, in one call                         |
| `calibration_folds` / `calibration_folds_cached` | The inclusion-*independent* half: fit the folds       |
| `threshold_from_folds`                    | The inclusion-*dependent* half: apply the rule to fitted folds |
| `fold_anchored_gmm_threshold`             | The shipped cut - fold mixtures anchored on held-out labels    |
| `calculate_safe_threshold`                | Blends cross-cal with GMM when label counts are low           |

### `calculate_gmm_threshold(scores)`

`vtscore/training/thresholds.py`. Fits a 2-component
`sklearn.mixture.GaussianMixture` to the score list and returns the
**midpoint between the two component means**. Used to produce a
reasonable operating point even when only a few labels exist - the score
distribution still tends to be bimodal because the embedder space already
separates "kind of like X" from "kind of not like X".

Issue #2798 briefly cut instead at the **equal-density crossing** of the
two weighted components (the root of `w_lo·N(x; μ_lo, σ²_lo) = w_hi·N(x;
μ_hi, σ²_hi)` between the means), which sits above the midpoint under
region voting where the max-pooled Bad component comes out wide and
heavy. Issue #2799 measured that as a small net cost regression and #2833
reverted it; `_weighted_gaussian_crossing` / `GmmFit1D.crossing_or_midpoint`
remain in the module as eval variants only (see issue #2836).

Falls back to `np.median(scores)` when GMM fitting raises (e.g. degenerate
score distributions), and to `0.5` when fewer than 2 scores are provided.

### `conformal_threshold(scores, labels, inclusion_value=0)`

`vtscore/training/thresholds.py`. Split-conformal quantile rule mapping
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
single-class. (Replaced the old min-cost `find_optimal_threshold`
argmin, which provably could not move with inclusion on well-separated
calibration folds - see `docs/experiments/2026-07-27-inclusion-knob/REPORT.md`.)

### `calculate_cross_calibration_threshold(...)`

`vtscore/training/thresholds.py`. The production threshold trainer.
For each of `calibrate_count` rounds:

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
depends on the Inclusion knob:

```python
from vtscore.training import calibration_folds_cached, threshold_from_folds

folds = calibration_folds_cached(          # expensive: fits `calibrate_count` fold models
    X_list, y_list, input_dim,
    calibrate_count=2, calibration_fraction=0.5,
    hidden_dim=LINEAR_SVM_HEAD, det_ctx=det_ctx,
)
threshold = threshold_from_folds(folds, inclusion_value=0)   # cheap: a quantile rule
```

`CalibrationFolds` is a `NamedTuple` of `(orderings, fallback, models)`.
`calibration_folds_cached` memoises it on `det_ctx.calibration_cache` under a
deterministic key built from `X_list`, `y_list`, the calibrate settings,
`hidden_dim`, and any `score_rows_by_group` - so toggling Inclusion during
an interactive sort re-runs only the cheap rule, with no ~200-epoch fold
refits. A real label change produces a different key and falls through to a
fresh calibration; no explicit invalidation is needed.

The key bytes encode the actual training vectors (not just label IDs),
so if the embedder changes and a labelset is re-resolved to different
embeddings, the cache invalidates automatically. The fitted fold *models*
ride along in the tuple because the shipped fold-anchored cut needs to
re-score the haystack with them.

### `calculate_safe_threshold(xcal_threshold, all_scores, ctx, schedule=None)`

`vtscore/training/thresholds.py`. Combines the cross-calibration
threshold with a GMM threshold computed on the full score distribution.
The cross-cal output gets noisy when labels are few, so a **mix-in
schedule** (`vtscore/training/blend_schedules.py`) decides how much of
each to use. `ctx` is a `BlendContext` carrying the vote counts (total,
good, bad — in votes, not flooded rows); a bare `int` is accepted where
only the total is known.

The shipped schedule depends on the **voting mode**, because #2841
measured the two separately and they want different curves
(`PRODUCTION_SCHEDULE_BY_MODE`, resolved per training call by
`vtscore.detectors.training._blend_schedule_for_snap`):

| mode | schedule | shape |
|---|---|---|
| region (patch dataset) | `slow` | pure GMM ≤6 labels → pure cross-cal at **40** |
| binary (single vector) | `cap50` | the old 6→20 ramp, but capped at **half** cross-cal forever |
| unknown | `cap50` | the one arm that improved both modes under every weighting |

The historical rule — a single 6→20 linear ramp — is retained as `prod`,
the baseline every number in the study's report is a delta against.
Other registry entries vary the endpoints, the curve shape, the statistic
the ramp reads (total labels vs the rarer class), or replace the weighted
average with a clamp into the GMM's component means.
See `docs/experiments/2026-08-04-mixin-schedule/REPORT.md`.

When `xcal_threshold` is `float("inf")` (no valid fold split), falls
back entirely to the GMM threshold.

---

## SVM (prototype)

`vtscore/training/svm.py` ships a parallel trainer with the same call
shape as `train_model`. It is **not** wired into the detector pipeline;
its purpose is to let
[`vtscore.eval.label_curve`](eval.md#label-curve) sweep the neural head
vs. SVM head-to-head so the team can decide whether to add a trainer-
selection field on detectors.

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

### `train_svm(...)` (`vtscore/training/svm.py`)

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
| `score_against_query(media, query_vec)` (line 34)     | Returns `(max_cosine_similarity, best_region_box)` for one media. For patch media, scores every row of `media_score_rows` (image-level vector + every raw patch) and returns the max + that row's box. For single-vector media, returns the cosine plus `(0.0, 0.0, 1.0, 1.0)`. `(0.0, None)` on zero-norm or missing embedding. |
| `cosine_sort_with_boxes(snap, query_vec)` (line 92)   | Snapshot-level scorer. Per-snapshot dispatch: patch snapshots use the cached flattened float16 score-row matrix + a chunked matvec and segmented max-pool (K = 1 + H·W, 197 on DINOv3); single-vector snapshots use the cached `(N, D)` matrix via `vtscore.embedding.matrix.get_embedding_matrix_for_snap`. Returns `(results_sorted_desc, raw_similarities_in_input_order)`. Result entries are `{"id": cid, "similarity": float, "best_region": [x0, y0, x1, y1]?}`. |

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
- **No hardcoded paths.** `train_model` reads
  `vtscore.config.TRAIN_EPOCHS` / `TRAIN_PATIENCE` /
  `MLP_HIDDEN_MIN` / `MLP_HIDDEN_MAX` / `MLP_DROPOUT` at call time;
  the only filesystem-aware module in this package is none - `config`
  itself centralises every path via `DATA_DIR`.
- **Thread-safe RNG.** `train_model` uses `torch.random.fork_rng` so
  parallel training calls don't interfere; cross-calibration uses an
  optional `np.random.RandomState` (seeded with 42 by the cached
  wrapper) so two threads sharing the cache still get deterministic
  thresholds.
- **No Flask, no settings.** Every threshold/training input is a
  function argument, not a global lookup.
