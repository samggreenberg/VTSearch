# Machine Learning Details

VTSearch uses a small MLP (multi-layer perceptron) neural network to learn a binary classifier from user votes ("good" vs "bad"). The model operates on embeddings produced by pretrained feature extractors and outputs a score in [0, 1] for each item in the dataset.

## Architecture

The MLP is defined in `vtsearch/models/training.py` via `build_model()`:

```
Linear(input_dim, hidden_dim) -> ReLU -> Dropout(p) -> Linear(hidden_dim, 1)
```

- **Input dimension**: Dynamic, depends on the embedding model for the current media type (see [Embedding Models](#embedding-models) below).
- **Hidden layer**: Width chosen automatically by `_auto_hidden_dim(n_labels)` — `max(MLP_HIDDEN_MIN, min(MLP_HIDDEN_MAX, n_labels // 3))`, i.e. 4–32 neurons depending on label count. Dropout (`MLP_DROPOUT=0.5`) is applied after ReLU during training.
- **Output**: A single logit. `torch.sigmoid` is applied at inference time to produce a probability in [0, 1].

The model outputs raw logits (not probabilities) during training. This allows the use of `BCEWithLogitsLoss`, which fuses the sigmoid and binary cross-entropy computation using the log-sum-exp trick for better numerical stability. At inference time, `torch.sigmoid()` is applied explicitly to convert logits to probabilities.

## Training Configuration

| Setting | Value | Notes |
|---------|-------|-------|
| **Loss function** | `BCEWithLogitsLoss` | Per-sample (unreduced), with manual class weighting |
| **Optimizer** | Adam | `lr=0.001`, `weight_decay=1e-4` |
| **Epochs** | 200 | Configurable via `TRAIN_EPOCHS` in `config.py` |
| **Hidden layer** | 4–32 neurons | `max(4, min(32, n_labels // 3))` via `_auto_hidden_dim()` |
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

**Why fold models use the full-data hidden_dim:** The hidden-layer width is normally auto-sized from the training-set size (`n_labels // 3`, clamped to 4–32). Without intervention, each fold model would train on fewer examples and therefore get a smaller hidden layer than the final model. A smaller architecture produces a different score distribution, so thresholds found on fold models would not transfer faithfully to the final model. By forcing fold models to use the same `hidden_dim` as the final model, the architectures match: same capacity, same score distribution shape. The only difference is the fold models see less data, which is the whole point — you want held-out calibration data. Existing regularization (dropout 0.5, weight decay 1e-4) and the small max width (32) prevent the slightly "oversized" fold models from overfitting meaningfully.

The **Calibration Fraction** setting (0–1, default 0.5) controls how much data is reserved for threshold calibration vs. model training in each split. For example, a value of 0.2 means 80% Train / 20% Calibrate. If the fraction is so extreme that a valid Train/Calibrate split cannot be formed (fewer than 2 training examples or fewer than 1 calibration example), the system returns a maximum threshold so that nothing is predicted as Good.

The optimal threshold at each split minimizes a weighted combination of false-positive rate and false-negative rate, governed by the same `inclusion_value`.

For semantic (text/example) sorts, a **GMM-based threshold** is used instead: a 2-component Gaussian Mixture Model is fitted to the score distribution and the midpoint between component means serves as the threshold.

## PyTorch Environment Settings

| Setting | Where | Value |
|---------|-------|-------|
| `OMP_NUM_THREADS` | `app.py` | `1` |
| `MKL_NUM_THREADS` | `app.py` | `1` |
| `torch.set_num_threads` | `vtsearch/models/loader.py` | `1` |
| dtype | `training.py` | `torch.float32` |
| Device | default | CPU (GPU supported, see tests) |

Threading is restricted to 1 to minimize memory overhead — the real cost is the embedding models, not the MLP.

## Embedding Models

Each media type uses a different pretrained model to produce fixed-size embedding vectors:

| Media type (`type_id`) | Embedder | Model | Embedding dim |
|------------------------|----------|-------|--------------|
| Audio (`audio`) | `clap` (default) | LAION CLAP (`laion/clap-htsat-unfused`) | 512 |
| Audio (`audio`) | `clap_music` | CLAP Music & Speech (`laion/larger_clap_music_and_speech`) | 512 |
| Image (`image`) | `clip` (default) | OpenAI CLIP (`openai/clip-vit-base-patch32`) | 768 |
| Image (`image`) | `siglip` | SigLIP (`google/siglip-base-patch16-224`) | 768 |
| Video (`video`) | `xclip` (default) | Microsoft X-CLIP (`microsoft/xclip-base-patch32`) | 768 |
| Text (`text`) | `e5` (default) | E5 (`intfloat/e5-base-v2`) | 768 |
| Text (`text`) | `bge` | BGE (`BAAI/bge-base-en-v1.5`) | 768 |
| Document (`document`) | — | None (no embedder) | N/A |

Audio, image, and text media types each have an **alternative embedder** in addition to the default. Alternative embedders are registered in `vtsearch/media/__init__.py` and live in files like `embedder_clap_music.py`, `embedder_siglip.py`, and `embedder_bge.py` alongside the primary `embedder.py` in each media type directory.

The **document** media type has no embedding model of its own. Documents (PDF, DOC, PPT) are intended to be converted to other media types (images or text) via media converters in `vtsearch/converters/` before embedding.

Embeddings are computed once when a dataset is loaded and stored as `numpy.ndarray` in each clip's `"embedding"` field. The MLP trains on these pre-computed vectors, so training is fast (typically < 1 second for 200 epochs on a few hundred labeled examples).

## Model Serialization

Trained models are serialized as JSON dictionaries mapping state_dict keys to nested lists:

```json
{
    "0.weight": [[...], ...],
    "0.bias": [...],
    "2.weight": [[...]],
    "2.bias": [...]
}
```

To reconstruct a model from saved weights, use `build_model(input_dim)` followed by `load_state_dict()`. The `input_dim` can be inferred from the first layer weights: `len(weights["0.weight"][0])`.

## Key Files

- `vtsearch/models/training.py` — `build_model`, `train_model`, `train_and_score`, threshold functions
- `vtsearch/models/progress.py` — Cached per-step training and stability analysis
- `vtsearch/models/loader.py` — Model initialization and thread configuration
- `vtsearch/eval/voting_iterations.py` — Voting simulation evaluation
- `vtsearch/config.py` — `TRAIN_EPOCHS` and model IDs
