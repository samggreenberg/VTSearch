"""MLP training primitives - generic build/train, media-agnostic.

The functions here take feature matrices and labels and return PyTorch
models. They have no knowledge of medias, votes, or detector contexts;
that glue lives under :mod:`vtscore.detectors`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from vtscore import config

from vtscore.concurrency.async_jobs import check_job_cancelled
from vtscore.config import MLP_DROPOUT, MLP_HIDDEN_MAX, MLP_HIDDEN_MIN, MLP_LABEL_SMOOTHING

# ``TRAIN_EPOCHS`` and ``TRAIN_PATIENCE`` are read off ``config`` at call time
# (not import time) so env-var / monkey-patched overrides take effect.

if TYPE_CHECKING:
    import torch
    import torch.nn as nn


#: ``hidden_dim`` sentinel selecting the **linear (logistic-regression) head** -
#: a single ``Linear(input_dim, 1)`` with no hidden layer.  This is the
#: **production detector head**: trained through :func:`train_model`'s balanced
#: BCE-with-logits loop it *is* logistic regression, and its linear decision
#: boundary avoids the retrain-to-retrain score wobble a small MLP shows when
#: positives are sparse (the threshold-stability #2790 finding).  Positive
#: ``hidden_dim`` values build the MLP instead; that path survives only for the
#: eval harness / experiments and unit tests, and is not reachable from the app.
LINEAR_HEAD = 0


def _auto_hidden_dim(n_train: int) -> int:
    """Choose MLP hidden-layer width based on training-set size.

    Keeps the model small when few votes are available to reduce
    overfitting, and grows (up to ``MLP_HIDDEN_MAX``) as more labels
    arrive.  The width is floored at ``MLP_HIDDEN_MIN`` (8): below ~8
    neurons the detector underfits and destabilizes on harder tasks.

    Only the MLP head uses this; the production linear head (:data:`LINEAR_HEAD`)
    has no hidden layer.
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
            (default 0.0 - no dropout).  Active only during training.
        generator: Optional local RNG for weight initialisation.  When
            provided the weights are re-initialised using this generator
            instead of PyTorch's global RNG, making construction
            thread-safe and deterministic.

    Returns:
        An ``nn.Sequential`` model.  With ``hidden_dim > 0`` (the MLP) the layers
        are ``Linear(input_dim, hidden_dim) -> ReLU -> Dropout -> Linear(hidden_dim, 1)``.
        With ``hidden_dim == 0`` (:data:`LINEAR_HEAD`, the production head) it is a
        single ``Linear(input_dim, 1)`` - a linear/logistic head with no hidden
        layer; ``dropout`` is ignored there (a bare linear map has nothing to
        regularise with dropout, matching plain logistic regression).
    """
    import torch.nn as nn  # noqa: PLC0415

    if hidden_dim == LINEAR_HEAD:
        model = nn.Sequential(nn.Linear(input_dim, 1))
    else:
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
    if "3.weight" in weights:
        # MLP: the second Linear's bias length is the hidden width.
        hidden_dim = len(weights["0.bias"])
    else:
        # Linear head (:data:`LINEAR_HEAD`): a single ``Linear(input_dim, 1)``,
        # so the only keys are ``0.weight`` / ``0.bias``.
        hidden_dim = LINEAR_HEAD

    model = build_model(input_dim, hidden_dim=hidden_dim, dropout=0.0)
    state_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in weights.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def train_model(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    input_dim: int,
    seed: int = 42,
    hidden_dim: int | None = None,
    sample_weights: torch.Tensor | None = None,
) -> nn.Sequential:
    """Train a small MLP classifier and return the trained model.

    The hidden-layer width is chosen automatically based on the number of
    training examples (see :func:`_auto_hidden_dim`) unless explicitly
    provided via ``hidden_dim``.  Dropout is applied during training to
    reduce overfitting.

    Trains using binary cross-entropy loss with logits (``BCEWithLogitsLoss``).
    By default each sample is weighted by inverse class frequency (balanced
    BCE).  When *sample_weights* is provided (one weight per row of
    *X_train*), those weights are used verbatim instead - this is how the
    region-flooding path expresses **per-bag** balancing: a Bad image's many
    correlated region negatives share one image's worth of weight, so they
    don't count as many independent negatives.  Inclusion does **not** enter
    training: it is a pure threshold/cutoff knob applied later in
    :func:`vtscore.training.thresholds.conformal_threshold`, so the trained
    model (and therefore every item's score) is independent of inclusion.  See
    docs/plans/find-verification-workflow.md.

    Targets are label-smoothed by ``MLP_LABEL_SMOOTHING`` (Good trains toward
    ``1 - eps``, Bad toward ``eps``) after class weights are derived from the
    hard labels.  This bounds the optimal logit so a strongly-fit model can't
    saturate every score to exact 0.0/1.0 sigmoids - which would collapse the
    conformal threshold rule's quantiles (and any score-based ranking) into a
    single tie.  It does not change which side of 0.5 an example is pushed to.

    A local ``torch.Generator`` seeded with *seed* is used for model-weight
    initialisation, and ``torch.random.fork_rng`` isolates the global RNG
    (used by ``nn.Dropout``) so concurrent calls don't overwrite each
    other's seed (thread-safe and deterministic).

    Args:
        X_train: Float tensor of shape ``(N, input_dim)`` containing training embeddings.
        y_train: Float tensor of shape ``(N, 1)`` containing binary labels
            (1.0 for good, 0.0 for bad).
        input_dim: Dimensionality of the input embeddings.
        seed: Seed for the local RNG used for weight initialisation (default 42).
        hidden_dim: Number of neurons in the hidden layer.  When ``None``
            (default) the width is chosen automatically via
            :func:`_auto_hidden_dim` based on the training-set size.
        sample_weights: Optional per-row loss weights of shape ``(N,)`` or
            ``(N, 1)``.  When ``None`` (default) inverse-class-frequency
            weights are computed internally.  When provided they replace the
            frequency weights entirely (the caller owns class balance).

    Returns:
        A trained ``nn.Sequential`` model in eval mode.
        The model outputs raw logits - apply ``torch.sigmoid`` at inference.

    Raises:
        ValueError: If ``y_train`` does not contain at least one positive
            (``y == 1``) and one negative (``y == 0``) example.  BCE has no
            discriminative signal on single-class data - the model would
            saturate to a single constant for every input - so we refuse
            up front instead of returning a useless model.
    """
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415

    from vtscore.embedding.loader import ensure_torch_configured, get_torch_device

    num_true = int(y_train.sum().item())
    num_false = len(y_train) - num_true
    if num_true == 0 or num_false == 0:
        raise ValueError(
            "train_model requires at least one positive (y=1) and one negative "
            f"(y=0) example; got {num_true} positives and {num_false} negatives"
        )

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

    # Sample weights: caller-supplied (per-bag flooding balance) or the default
    # inverse-class-frequency balance.  Inclusion is deliberately absent here
    # (it's applied later as a pure threshold knob in find_optimal_threshold),
    # so the model is inclusion-independent and its scores can be frozen across
    # cutoff slides.
    if sample_weights is not None:
        weights = sample_weights.reshape(-1).to(dtype=torch.float32, device=device)
        if weights.numel() != len(y_train):
            raise ValueError(f"sample_weights length {weights.numel()} does not match training-set size {len(y_train)}")
    else:
        # Balanced class weights - the single-class case was rejected above.
        weight_true = num_false / num_true
        weight_false = 1.0
        weights = torch.where(y_train == 1, weight_true, weight_false).squeeze().to(device)
    # Smooth the targets only after the hard labels have been counted and
    # weighted above; BCE-with-logits accepts soft targets natively.
    y_train = y_train * (1.0 - 2.0 * MLP_LABEL_SMOOTHING) + MLP_LABEL_SMOOTHING
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    # Fork the global RNG so the Dropout seed is isolated per call -
    # concurrent training invocations each get their own RNG state.
    model.train()
    epochs = config.TRAIN_EPOCHS
    patience = config.TRAIN_PATIENCE
    best_loss = float("inf")
    epochs_since_improve = 0
    # Require at least this much absolute decrease to count as progress -
    # without it, float noise drifts the loss down forever and the early-stop
    # never fires.
    min_delta = 1e-4
    # Mixed-precision: enable autocast + GradScaler only on CUDA, where
    # tensor cores give a real speedup. CPU/MPS keep the FP32 path so
    # deterministic training stays bit-for-bit reproducible.
    from contextlib import nullcontext  # noqa: PLC0415

    use_amp = device.type == "cuda"
    # Reading ``weighted_loss.item()`` forces a host-device sync that stalls the
    # CUDA stream every epoch. On GPU, only sync (and run the early-stop check)
    # every few epochs; when no improvement is seen at a checkpoint we advance
    # the patience counter by the whole interval, so early-stop still fires at
    # roughly ``patience`` epochs. On CPU/MPS the read is free, so we keep the
    # per-epoch cadence and the CPU path stays bit-for-bit as before (the seeded
    # early-stop test exercises exactly this path).
    loss_check_interval = 5 if device.type == "cuda" else 1
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
        for epoch in range(epochs):
            # Honour a cancel of the background job that owns this thread at
            # the epoch boundary; a no-op outside a job (see
            # ``async_jobs.check_job_cancelled``).
            check_job_cancelled()
            optimizer.zero_grad()
            with autocast_ctx:
                logits = model(X_train)
                losses = loss_fn(logits, y_train)
                weighted_loss = (losses.squeeze() * weights).mean()
            scaler.scale(weighted_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if (epoch + 1) % loss_check_interval != 0:
                continue
            cur = weighted_loss.item()
            if cur < best_loss - min_delta:
                best_loss = cur
                epochs_since_improve = 0
            else:
                epochs_since_improve += loss_check_interval
                if patience > 0 and epochs_since_improve >= patience:
                    break

    model.eval()
    return model
