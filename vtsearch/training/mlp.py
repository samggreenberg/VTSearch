"""MLP training primitives — generic build/train, media-agnostic.

The functions here take feature matrices and labels and return PyTorch
models. They have no knowledge of medias, votes, or detector contexts;
that glue lives under :mod:`vtsearch.detectors`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from vtsearch import config
from vtsearch.config import MLP_DROPOUT, MLP_HIDDEN_MAX, MLP_HIDDEN_MIN

# ``TRAIN_EPOCHS`` and ``TRAIN_PATIENCE`` are read off ``config`` at call time
# (not import time) so env-var / monkey-patched overrides take effect.

if TYPE_CHECKING:
    import torch
    import torch.nn as nn


def _auto_hidden_dim(n_train: int) -> int:
    """Choose hidden-layer width based on training-set size.

    Keeps the model small when few votes are available to reduce
    overfitting, and grows (up to ``MLP_HIDDEN_MAX``) as more labels
    arrive.
    """
    return max(MLP_HIDDEN_MIN, min(MLP_HIDDEN_MAX, n_train // 3))


def build_model(
    input_dim: int,
    hidden_dim: int = 64,
    dropout: float = 0.0,
    generator: torch.Generator | None = None,
) -> nn.Sequential:
    """Construct the MLP architecture (untrained).

    The model outputs raw logits (no sigmoid).  Apply ``torch.sigmoid``
    to the output at inference time to obtain probabilities in [0, 1].

    Args:
        input_dim: Dimensionality of the input embeddings.
        hidden_dim: Number of neurons in the hidden layer (default 64).
        dropout: Dropout probability applied after the hidden layer
            (default 0.0 — no dropout).  Active only during training.
        generator: Optional local RNG for weight initialisation.  When
            provided the weights are re-initialised using this generator
            instead of PyTorch's global RNG, making construction
            thread-safe and deterministic.

    Returns:
        An ``nn.Sequential`` model with layers:
        ``Linear(input_dim, hidden_dim) -> ReLU -> Dropout -> Linear(hidden_dim, 1)``.
    """
    import torch.nn as nn  # noqa: PLC0415

    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )
    if generator is not None:
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5), generator=generator)
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(module.bias, -bound, bound, generator=generator)
    return model


def build_model_from_weights(weights: dict[str, list]) -> nn.Sequential:
    """Reconstruct a trained model from exported weight lists.

    Handles both legacy (3-layer, no dropout) and current (4-layer with
    dropout) weight formats.  Legacy keys ``2.weight`` / ``2.bias`` are
    remapped to ``3.weight`` / ``3.bias`` automatically.

    Args:
        weights: Mapping of state-dict key names to nested Python lists
            (as produced by ``tensor.tolist()``).

    Returns:
        A model in eval mode ready for inference.
    """
    import torch  # noqa: PLC0415

    # Remap legacy 3-layer format (keys 0,2) → 4-layer format (keys 0,3)
    if "2.weight" in weights and "3.weight" not in weights:
        weights = dict(weights)
        weights["3.weight"] = weights.pop("2.weight")
        weights["3.bias"] = weights.pop("2.bias")

    input_dim = len(weights["0.weight"][0])
    hidden_dim = len(weights["0.bias"])

    model = build_model(input_dim, hidden_dim=hidden_dim, dropout=0.0)
    state_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in weights.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def train_model(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    input_dim: int,
    inclusion_value: int = 0,
    seed: int = 42,
    hidden_dim: int | None = None,
) -> nn.Sequential:
    """Train a small MLP classifier and return the trained model.

    The hidden-layer width is chosen automatically based on the number of
    training examples (see :func:`_auto_hidden_dim`) unless explicitly
    provided via ``hidden_dim``.  Dropout is applied during training to
    reduce overfitting.

    Trains using weighted binary cross-entropy loss with logits
    (``BCEWithLogitsLoss``).  Class weights are adjusted based on
    ``inclusion_value`` to bias the classifier toward including more
    (positive) or fewer (positive) items.

    A local ``torch.Generator`` seeded with *seed* is used for model-weight
    initialisation, and ``torch.random.fork_rng`` isolates the global RNG
    (used by ``nn.Dropout``) so concurrent calls don't overwrite each
    other's seed (thread-safe and deterministic).

    Args:
        X_train: Float tensor of shape ``(N, input_dim)`` containing training embeddings.
        y_train: Float tensor of shape ``(N, 1)`` containing binary labels
            (1.0 for good, 0.0 for bad).
        input_dim: Dimensionality of the input embeddings.
        inclusion_value: Integer in ``[-10, 10]`` controlling class-weight bias.
            - 0: balance classes equally (weight_true = num_false / num_true).
            - Positive: increase weight for True samples by ``2 ** inclusion_value``,
              causing the model to include more items.
            - Negative: increase weight for False samples by ``2 ** (-inclusion_value)``,
              causing the model to exclude more items.
        seed: Seed for the local RNG used for weight initialisation (default 42).
        hidden_dim: Number of neurons in the hidden layer.  When ``None``
            (default) the width is chosen automatically via
            :func:`_auto_hidden_dim` based on the training-set size.

    Returns:
        A trained ``nn.Sequential`` model in eval mode.
        The model outputs raw logits — apply ``torch.sigmoid`` at inference.
    """
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415

    from vtsearch.embedding.loader import ensure_torch_configured, get_torch_device

    ensure_torch_configured()
    device = get_torch_device()

    n_train = len(X_train)
    if hidden_dim is None:
        hidden_dim = _auto_hidden_dim(n_train)

    # Use a local Generator for weight init and fork_rng for nn.Dropout
    # so that concurrent training calls don't overwrite each other's
    # global seed (thread-safe and deterministic).
    g = torch.Generator()
    g.manual_seed(seed)

    model = build_model(input_dim, hidden_dim=hidden_dim, dropout=MLP_DROPOUT, generator=g)
    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Calculate class weights based on inclusion
    num_true = y_train.sum().item()
    num_false = len(y_train) - num_true

    # Base weights for balanced classes
    if num_true > 0 and num_false > 0:
        weight_true = num_false / num_true
        weight_false = 1.0
    else:
        weight_true = 1.0
        weight_false = 1.0

    # Adjust weights based on inclusion
    if inclusion_value >= 0:
        # Increase weight for True samples
        weight_true *= 2.0**inclusion_value
    else:
        # Increase weight for False samples
        weight_false *= 2.0 ** (-inclusion_value)

    # Create sample weights
    weights = torch.where(y_train == 1, weight_true, weight_false).squeeze().to(device)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    # Fork the global RNG so the Dropout seed is isolated per call —
    # concurrent training invocations each get their own RNG state.
    model.train()
    epochs = config.TRAIN_EPOCHS
    patience = config.TRAIN_PATIENCE
    best_loss = float("inf")
    epochs_since_improve = 0
    # Require at least this much absolute decrease to count as progress —
    # without it, float noise drifts the loss down forever and the early-stop
    # never fires.
    min_delta = 1e-4
    # Mixed-precision: enable autocast + GradScaler only on CUDA, where
    # tensor cores give a real speedup. CPU/MPS keep the FP32 path so
    # deterministic training stays bit-for-bit reproducible.
    from contextlib import nullcontext  # noqa: PLC0415

    use_amp = device.type == "cuda"
    # Prefer the modern ``torch.amp.GradScaler("cuda", ...)`` API; fall back
    # to the legacy ``torch.cuda.amp.GradScaler`` on torch <2.3.
    grad_scaler_cls = getattr(torch.amp, "GradScaler", None)
    if grad_scaler_cls is not None:
        scaler = grad_scaler_cls("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    autocast_ctx = torch.autocast(device_type="cuda") if use_amp else nullcontext()
    with torch.random.fork_rng(), torch.enable_grad():
        torch.manual_seed(seed)
        for _ in range(epochs):
            optimizer.zero_grad()
            with autocast_ctx:
                logits = model(X_train)
                losses = loss_fn(logits, y_train)
                weighted_loss = (losses.squeeze() * weights).mean()
            scaler.scale(weighted_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            cur = weighted_loss.item()
            if cur < best_loss - min_delta:
                best_loss = cur
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
                if patience > 0 and epochs_since_improve >= patience:
                    break

    model.eval()
    return model
