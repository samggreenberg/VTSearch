# Machine Learning Details

VTSearch uses a small MLP (multi-layer perceptron) neural network to learn a binary classifier from user votes ("good" vs "bad"). The model operates on embeddings produced by pretrained feature extractors (LAION-CLAP for audio, SigLIP for images, X-CLIP for video, E5-base-v2 for text) and outputs a score in [0, 1] for each item in the dataset.

## Architecture

The MLP is defined in `vtsearch/training/mlp.py` via `build_model()`:

```
Linear(input_dim, hidden_dim) -> ReLU -> Dropout(p) -> Linear(hidden_dim, 1)
```

- **Input dimension**: Dynamic, depends on the embedding model for the current media type (see [Embedding Models](#embedding-models) below).
- **Hidden layer**: Width chosen automatically by `_auto_hidden_dim(n_train)` — `max(MLP_HIDDEN_MIN, min(MLP_HIDDEN_MAX, n_train // 3))`, i.e. 4–32 neurons depending on training set size. Dropout (`MLP_DROPOUT=0.5`) is applied after ReLU during training.
- **Output**: A single logit. `torch.sigmoid` is applied at inference time to produce a probability in [0, 1].

The model outputs raw logits (not probabilities) during training. This allows the use of `BCEWithLogitsLoss`, which fuses the sigmoid and binary cross-entropy computation using the log-sum-exp trick for better numerical stability. At inference time, `torch.sigmoid()` is applied explicitly to convert logits to probabilities.

## Training Configuration

| Setting | Value | Notes |
|---------|-------|-------|
| **Loss function** | `BCEWithLogitsLoss` | Per-sample (unreduced), with manual class weighting |
| **Optimizer** | Adam | `lr=0.001`, `weight_decay=1e-4` |
| **Epochs (cap)** | 200 | Configurable via `TRAIN_EPOCHS` in `config.py` or the `VTSEARCH_TRAIN_EPOCHS` env var |
| **Early-stop patience** | 10 | Training halts when the loss fails to improve for this many consecutive epochs (configurable via `TRAIN_PATIENCE` / `VTSEARCH_TRAIN_PATIENCE`; set 0 to disable) |
| **Hidden layer** | 4–32 neurons | `max(4, min(32, n_train // 3))` via `_auto_hidden_dim()` |
| **Dropout** | 0.5 | Applied after ReLU, active only during training |
| **Batching** | Full-batch | All labeled data in every forward pass |
| **Reproducibility** | Local `torch.Generator` | Per-model seed (default 42) for thread-safe deterministic init |
| **Gradient scoping** | `torch.enable_grad()` | Explicitly enabled during training loop |

### Class Weighting

Training uses inverse-frequency weighting to balance classes, with an additional `inclusion_value` parameter (range [-10, +10]) that lets users bias the model toward recall or precision:

- **Base weights**: `weight_true = num_false / num_true`, `weight_false = 1.0`
- **Inclusion >= 0**: `weight_true *= 2^inclusion_value` (include more items)
- **Inclusion < 0**: `weight_false *= 2^(-inclusion_value)` (exclude more items)

### Threshold Calibration

A decision threshold separating "good" from "bad" predictions is computed via **cross-calibration**:

1. Compute `hidden_dim` once from the **full** label count.
2. Split labeled data into Train (1 − `calibration_fraction`) and Calibrate (`calibration_fraction`).
3. Train a fold model on Train **using the full-data `hidden_dim`**, find the optimal threshold by evaluating on Calibrate.
4. Repeat for `calibrate_count` independent random splits.
5. Return the mean of all thresholds.

The `calibrate_count` setting defaults to `2` but can be lowered (`VTSEARCH_CALIBRATE_COUNT`) to trade calibration quality for latency. When `safe_thresholds` is enabled and fewer than 6 labels exist, the cross-calibration step is skipped entirely — the `calculate_safe_threshold` blender weights the calibrated value at 0 in that regime, so paying for the fold trainings would be pure waste.

**Why fold models use the full-data hidden_dim:** The hidden-layer width is normally auto-sized from the training-set size (`n_train // 3`, clamped to 4–32). Without intervention, each fold model would train on fewer examples and therefore get a smaller hidden layer than the final model. A smaller architecture produces a different score distribution, so thresholds found on fold models would not transfer faithfully to the final model. By forcing fold models to use the same `hidden_dim` as the final model, the architectures match: same capacity, same score distribution shape. The only difference is the fold models see less data, which is the whole point — you want held-out calibration data. Existing regularization (dropout 0.5, weight decay 1e-4) and the small max width (32) prevent the slightly "oversized" fold models from overfitting meaningfully.

The **Calibration Fraction** setting (0–1, default 0.5) controls how much data is reserved for threshold calibration vs. model training in each split. For example, a value of 0.2 means 80% Train / 20% Calibrate. If the fraction is so extreme that a valid Train/Calibrate split cannot be formed (fewer than 2 training examples or fewer than 1 calibration example), the system returns a maximum threshold so that nothing is predicted as Good.

The optimal threshold at each split minimizes a weighted combination of false-positive rate and false-negative rate, governed by the same `inclusion_value`.

For semantic (text/example) sorts, a **GMM-based threshold** is used instead: a 2-component Gaussian Mixture Model is fitted to the score distribution and the midpoint between component means serves as the threshold.

## PyTorch Environment Settings

| Setting | Where | Value |
|---------|-------|-------|
| `OMP_NUM_THREADS` | `app.py` | `1` |
| `MKL_NUM_THREADS` | `app.py` | `1` |
| `torch.set_num_threads` | `vtsearch/embedding/loader.py` | `1` |
| dtype | `training.py` | `torch.float32` |
| Device | default | CPU (GPU supported, see tests) |

Threading is restricted to 1 to minimize memory overhead — the real cost is the embedding models, not the MLP.

## Embedding Models

Each media type uses a different pretrained model to produce fixed-size embedding vectors:

| Media type (`type_id`) | Embedder | Model | Embedding dim |
|------------------------|----------|-------|--------------|
| Audio (`audio`) | `clap` (default) | LAION CLAP (`laion/clap-htsat-unfused`) | 512 |
| Audio (`audio`) | `clap_music` | CLAP Music & Speech (`laion/larger_clap_music_and_speech`) | 512 |
| Image (`image`) | `siglip` (default) | SigLIP (`google/siglip-base-patch16-224`) | 768 |
| Image (`image`) | `dinov2_single` / `dinov2_patch` | DINOv2 ViT-B/14 (`facebook/dinov2-base`) | 768 |
| Image (`image`) | `dinov3_single` / `dinov3_patch` | DINOv3 ViT-B/16 (`facebook/dinov3-vitb16-pretrain-lvd1689m`, gated) | 768 |
| Image (`image`) | `eupe_single` / `eupe_patch` | EUPE ViT-B/16 (`facebookresearch/EUPE`, FAIR Noncommercial) | 768 |
| Video (`video`) | `xclip` (default) | Microsoft X-CLIP (`microsoft/xclip-base-patch32`) | 768 |
| Text (`text`) | `e5` (default) | E5 (`intfloat/e5-base-v2`) | 768 |
| Text (`text`) | `bge` | BGE (`BAAI/bge-base-en-v1.5`) | 768 |
| Document (`document`) | — | None (no embedder) | N/A |

Each embedder lives in its own `embedder_<name>.py` file inside the media-type package and exposes a module-level `EMBEDDER` sentinel; the default for a given media type is whichever embedder overrides `is_default` to return `True` (exactly one per media type).

Audio, image, and text media types each ship alternative embedders alongside the default. The image variants come in **single/patch pairs**: `_single` embedders produce one CLS-pooled vector per image (cheap, same shape as SigLIP); `_patch` embedders additionally produce a hierarchical HAC region tree (~24 region vectors per image) and the raw patch grid, enabling region-level similarity, region-aware MLP scoring, and region voting on yes-votes.  See [`docs/plans/patch-embedder.md`](plans/patch-embedder.md) for the full design.

Embedders carry capability flags consumed by the routes layer and the frontend:

- `supports_text: bool` — whether the embedder can embed text queries. Text-sort returns HTTP 400 + `supports_text: false` when this is false.
- `supports_patch_regions: bool` — set on the `_patch` variants. Loaders that see this flag populate `media["patch_regions"]` (HAC tree) and `media["patch_grid"]` (raw `H × W × D` fp16) in addition to `media["embedding"]`.
- `license_notice: Optional[str]` — non-None for embedders with usage restrictions (e.g. EUPE's FAIR Noncommercial Research Licence). Surfaced as a warning chip on the embedder picker.

The **document** media type has no embedding model of its own. Documents (PDF, DOC, PPT) are intended to be converted to other media types (images or text) via media converters in `vtsearch/converters/` before embedding.

Embeddings are computed once when a dataset is loaded. The full-image vector lands in each clip's `"embedding"` field (`numpy.ndarray`, used by every legacy code path); patch embedders additionally populate `"patch_regions"` (list of `RegionVector`s, fp16-on-disk / fp32-in-RAM) and `"patch_grid"` (`H × W × D` ndarray, fp16). The MLP trains on these pre-computed vectors, so training is fast (typically < 1 second for 200 epochs on a few hundred labeled examples).

### Region-aware training loss (patch-region datasets)

When the dataset's embedder produces `patch_regions`, the MLP's per-vote loss is asymmetric: Good votes train on the full-image `"embedding"` (the user's "this image is good" claim doesn't single out a region); Bad votes apply `BCE(score, 0)` to *every* region in the image's HAC tree (the strictly stronger claim "no region in this image is good") and reduce by mean. At inference time `score_media` max-pools the MLP over regions, so train-time and test-time agree about what "low score" means. For datasets whose embedder doesn't produce regions, the Bad-side mean collapses to today's single-vector BCE — fully backward-compatible. See "Detector MLP → Training loss" in [`docs/plans/patch-embedder.md`](plans/patch-embedder.md) for the rationale.

Yes-votes may additionally carry an optional `region_box` (4-float normalised rectangle) drawn by the user via Shift-drag on the focus pane. When set, the trainer pools the box's patch-grid cells on the fly (uniform mean, L2-normalise) and uses that vector instead of the full-image CLS vector for that Good example.

## Key Files

- `vtsearch/training/mlp.py` — `build_model`, `train_model`, `build_model_from_weights`
- `vtsearch/training/thresholds.py` — `calculate_cross_calibration_threshold`, `calculate_safe_threshold`, `calculate_gmm_threshold`, `find_optimal_threshold`
- `vtsearch/detectors/training.py` — `train_and_score`, `train_and_threshold`, origin-based detector training
- `vtsearch/detectors/labeling_progress.py` — Cached per-step training and stability analysis
- `vtsearch/embedding/loader.py` — Model initialization and thread configuration
- `vtsearch/eval/voting_iterations.py` — Voting simulation evaluation
- `vtsearch/config.py` — `TRAIN_EPOCHS` and model IDs
