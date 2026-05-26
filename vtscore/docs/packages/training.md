# `vtscore.training` - Learned-sort training primitives

Generic neural-net training and decision-threshold helpers extracted from
the detector pipeline. Everything in this package operates on raw numpy
arrays and PyTorch tensors - there are no media-type, embedder, vote,
or context dependencies. Library consumers can use it as a stand-alone
learned-sort toolkit: feed in `(N, D)` feature matrices and binary
labels, get back a trained MLP, a calibrated threshold, and (optionally)
patch-level cosine scoring against a query vector.

The detector-specific glue that resolves votes → origins → embeddings →
training data lives one layer up in [`vtscore.detectors`](detectors.md);
this package is the underlying ML core.

## Contents

| Module                                                                | What it provides                                                |
|-----------------------------------------------------------------------|-----------------------------------------------------------------|
| `vtscore/training/mlp.py`                                             | `build_model`, `build_model_from_weights`, `train_model`        |
| `vtscore/training/thresholds.py`                                      | GMM / cross-cal / safe threshold helpers                        |
| `vtscore/training/svm.py`                                             | `SVMClassifier` + `train_svm` prototype                         |
| `vtscore/training/region_similarity.py`                               | Patch-level cosine scoring with bounding boxes                  |

The package `__init__.py` re-exports the MLP and threshold names; SVM
and region-similarity helpers are imported from their submodules.

```python
from vtscore.training import (
    build_model, build_model_from_weights, train_model,
    calculate_gmm_threshold, find_optimal_threshold,
    calculate_cross_calibration_threshold,
    cross_calibration_threshold_cached, calculate_safe_threshold,
)
from vtscore.training.svm import SVMClassifier, train_svm
from vtscore.training.region_similarity import (
    score_against_query, cosine_sort_with_boxes,
)
```

---

## MLP trainer

A small `Linear → ReLU → Dropout → Linear` classifier that emits raw
logits. Built and trained from feature matrices and binary labels.

### Architecture

```python
nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, 1),
)
```

Built by `build_model(input_dim, hidden_dim=64, dropout=0.0, generator=None)`
at `vtscore/training/mlp.py:35`. Pass a seeded `torch.Generator` to
deterministically re-initialise the `Linear` weights (Kaiming uniform
on the weight matrix, uniform on the bias with the standard PyTorch
fan-in bound).

### Auto-sizing the hidden layer

`_auto_hidden_dim(n_train)` at `vtscore/training/mlp.py:25` chooses the
hidden width from the number of training examples:

```python
return max(MLP_HIDDEN_MIN, min(MLP_HIDDEN_MAX, n_train // 3))
```

With the default `MLP_HIDDEN_MIN=4` and `MLP_HIDDEN_MAX=32` (from
`vtscore.config`), the heuristic keeps the model small when only a
handful of labels exist - n_train=10 picks 4, n_train=60 picks 20,
n_train=120 picks 32 (capped). The function is private but stable; the
detector code in `vtscore/detectors/training.py:182` and
`vtscore/detectors/labelset_training.py:277` use it to ensure
cross-calibration fold models share the same architecture as the final
full-data model, so fold thresholds are directly comparable.

### Training

`train_model(X_train, y_train, input_dim, inclusion_value=0, seed=42, hidden_dim=None)`
at `vtscore/training/mlp.py:110` is the workhorse:

```python
import numpy as np, torch
from vtscore.training import train_model

X = torch.from_numpy(np.random.RandomState(0).standard_normal((60, 512)).astype(np.float32))
y = torch.tensor([1.0] * 30 + [0.0] * 30).unsqueeze(1)

model = train_model(X, y, input_dim=512, inclusion_value=0)
with torch.no_grad():
    scores = torch.sigmoid(model(X)).squeeze(1).cpu().numpy()
```

Key behaviour:

- **Loss:** weighted `BCEWithLogitsLoss(reduction="none")`. Per-sample
  weights are precomputed from `y_train` so the loss balances class
  frequencies (`weight_true = num_false / num_true`, `weight_false = 1.0`)
  even before the inclusion bias is applied.
- **Inclusion bias** (`inclusion_value ∈ [-10, 10]`): multiplies the
  per-class weight by `2 ** abs(inclusion_value)`. Positive values
  inflate the True-class weight so the classifier prefers recall
  (include more); negative values inflate the False-class weight so it
  prefers precision (include fewer). See `vtscore/training/mlp.py:193`.
- **Optimiser:** `Adam(lr=0.001, weight_decay=1e-4)`.
- **Early stop:** trains up to `config.TRAIN_EPOCHS` (default 200) and
  stops after `config.TRAIN_PATIENCE` consecutive epochs with no
  improvement larger than `min_delta = 1e-4`. Read fresh at every call,
  so monkey-patching `vtscore.config.TRAIN_EPOCHS` for tests works.
- **Mixed precision:** enabled automatically on CUDA via `torch.amp` /
  `GradScaler`; CPU and MPS use FP32 so deterministic training is
  bit-for-bit reproducible.
- **Device:** picked up from `vtscore.embedding.loader.get_torch_device()`
  via `ensure_torch_configured()`.

### Thread safety and reproducibility

`train_model` deliberately avoids touching PyTorch's global RNG:

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
isolates the dropout RNG inside the training loop. Two concurrent
`train_model` calls in different threads do not interfere with each
other's seed, and either call produces the same model given the same
`(X, y, seed)`. This matters because cross-calibration trains *k* fold
models in sequence and the eval harness can run multiple seeds in
parallel.

### Reloading from saved weights

`build_model_from_weights(weights)` at `vtscore/training/mlp.py:78`
reconstructs a model from a dict of lists (the output of
`tensor.tolist()` per state-dict entry). It accepts the current 4-layer
format (`0.weight`, `0.bias`, `3.weight`, `3.bias`) and silently
remaps the legacy 3-layer format (`0.*`, `2.*`) to the current keys, so
old detector files don't have to be migrated.

> **Invariant - no persisted MLP weights.** In VTSearch proper, detector
> JSON files store labelsets (origin info + per-element labels) only;
> MLP weights are re-derived from those origins on every load. See
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
| `find_optimal_threshold`                  | F1 maximiser on (scores, labels) for one model                |
| `calculate_cross_calibration_threshold`   | Production threshold - k-fold cross-calibration               |
| `cross_calibration_threshold_cached`      | Memoised wrapper around the cross-cal trainer                 |
| `calculate_safe_threshold`                | Blends cross-cal with GMM when label counts are low           |

### `calculate_gmm_threshold(scores)`

`vtscore/training/thresholds.py:17`. Fits a 2-component
`sklearn.mixture.GaussianMixture` to the score list and returns the
midpoint between the two component means. Used to produce a reasonable
operating point even when only a few labels exist - the score
distribution still tends to be bimodal because the embedder space
already separates "kind of like X" from "kind of not like X".

Falls back to `np.median(scores)` when GMM fitting raises (e.g.
degenerate score distributions), and to `0.5` when fewer than 2 scores
are provided.

### `find_optimal_threshold(scores, labels, inclusion_value=0)`

`vtscore/training/thresholds.py:149`. Vectorised O(n log n) sweep: sorts
by score descending, builds cumulative TP/FP/FN counts, and picks the
threshold minimising `fpr_weight * FPR + fnr_weight * FNR`. The
weights come from `inclusion_value` the same way as in `train_model`:

```python
if inclusion_value >= 0:
    fpr_weight, fnr_weight = 1.0, 2.0 ** inclusion_value
else:
    fpr_weight, fnr_weight = 2.0 ** (-inclusion_value), 1.0
```

Returns `0.5` when the input is empty or single-class.

### `calculate_cross_calibration_threshold(...)`

`vtscore/training/thresholds.py:220`. The production threshold trainer.
For each of `calibrate_count` rounds:

1. Randomly split `(X_list, y_list)` into Train (`1 - calibration_fraction`)
   and Calibrate (`calibration_fraction`).
2. Train an MLP on Train via `train_model`.
3. Score Calibrate, find the optimal threshold via
   `find_optimal_threshold`.

Returns the mean threshold across rounds. Defaults to 0.5 when
`n < 4`; returns `float("inf")` when the split would leave fewer than
2 training examples or 1 calibration example.

```python
from vtscore.training import calculate_cross_calibration_threshold

t = calculate_cross_calibration_threshold(
    X_list=[v for v in feature_vectors],
    y_list=[1.0, 0.0, 1.0, 0.0, ...],
    input_dim=512,
    inclusion_value=0,
    calibrate_count=2,
    calibration_fraction=0.5,
    hidden_dim=8,   # match the full-data model's hidden width
)
```

The fold models honour `hidden_dim` when provided - important for the
detector pipeline, which wants fold thresholds calibrated against a
model with the same capacity as the final full-data model.

### `cross_calibration_threshold_cached(...)`

`vtscore/training/thresholds.py:93`. Same signature plus an optional
`det_ctx` argument. Builds a deterministic cache key from `X_list`,
`y_list`, `inclusion_value`, `calibrate_count`, `calibration_fraction`,
and `hidden_dim`, then stores the resulting threshold on
`det_ctx.calibration_cache`. The next call with matching inputs returns
the cached value without retraining; a real label change produces a
different key and falls through to a fresh calibration.

The key bytes encode the actual training vectors (not just label IDs),
so if the embedder changes and a labelset is re-resolved to different
embeddings, the cache invalidates automatically.

### `calculate_safe_threshold(xcal_threshold, all_scores, n_labels)`

`vtscore/training/thresholds.py:312`. Blends the cross-calibration
threshold with a GMM threshold computed on the full score distribution.
The cross-cal output gets noisy below ~20 labels; this blend ramps
linearly from pure-GMM at 6 labels to pure-cross-cal at 20:

| `n_labels` | Result                                      |
|-----------:|---------------------------------------------|
|     `< 6`  | pure GMM threshold                          |
|    `6..20` | linear interpolation                        |
|   `>= 20`  | pure cross-cal threshold                    |

When `xcal_threshold` is `float("inf")` (no valid fold split), falls
back entirely to the GMM threshold.

---

## SVM (prototype)

`vtscore/training/svm.py` ships a parallel trainer with the same call
shape as `train_model`. It is **not** wired into the detector pipeline;
its purpose is to let
[`vtscore.eval.label_curve`](eval.md#label-curve-sweep) sweep MLP vs.
SVM head-to-head so the team can decide whether to add a trainer-
selection field on detectors.

### `SVMClassifier` (`vtscore/training/svm.py:26`)

Dataclass wrapping a fitted sklearn estimator plus an optional
probability source: `base` (`LinearSVC` or `SVC`), `calibrator`
(`CalibratedClassifierCV` or `None`), `scaler` (optional
`StandardScaler`), plus `kernel` and `calibration` tags.
`predict_proba(X)` returns a 1-D `float32` array in `[0, 1]`. When a
calibrator is fitted, it consults `calibrator.predict_proba(...)`;
otherwise it sigmoids the raw `decision_function` (clipped to ±30).
The sigmoid wrapper is not a true probability, but it is monotone in
the SVM score - which is all the ranker and threshold-finder need.

### `train_svm(...)` (`vtscore/training/svm.py:130`)

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
single-class inputs (the MLP path returns `None` instead; here the
eval harness wants the error to record a skip).

---

## Region similarity (patch-level scoring)

`vtscore/training/region_similarity.py` is the entry point for
embedding-cosine sort when the dataset's media expose per-patch
embeddings (DINOv2, DINOv3, EUPE). It dispatches to a fast vectorised
numpy path when no patches are present, so SigLIP / CLIP datasets see
zero overhead.

| Function                                              | Behaviour                                                                |
|-------------------------------------------------------|--------------------------------------------------------------------------|
| `score_against_query(media, query_vec)` (line 34)     | Returns `(max_cosine_similarity, best_region_box)` for one media. For patch-region media, scores every `RegionVector` and returns the max + its box. For single-vector media, returns the cosine plus `(0.0, 0.0, 1.0, 1.0)`. `(0.0, None)` on zero-norm or missing embedding. |
| `cosine_sort_with_boxes(snap, query_vec)` (line 92)   | Snapshot-level scorer. Per-snapshot dispatch: patch-region snapshots iterate per media (O(N·K), K≈23); single-vector snapshots use the cached `(N, D)` matrix via `vtscore.embedding.matrix.get_embedding_matrix_for_snap`. Returns `(results_sorted_desc, raw_similarities_in_input_order)`. Result entries are `{"id": cid, "similarity": float, "best_region": [x0, y0, x1, y1]?}`. |

```python
from vtscore.training.region_similarity import cosine_sort_with_boxes

results, sims = cosine_sort_with_boxes(snap, query_vec)
top_ten = results[:10]
```

---

## Invariants worth restating

- **No persisted MLP weights.** Library callers that load a model from
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
