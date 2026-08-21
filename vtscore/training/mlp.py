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
#: a single ``Linear(input_dim, 1)`` with no hidden layer, fitted by
#: :func:`train_model`'s balanced BCE-with-logits loop.  Its linear decision
#: boundary avoids the retrain-to-retrain score wobble a small MLP shows when
#: positives are sparse (the threshold-stability #2790 finding).  This was the
#: production head until the linear SVM replaced it; it now survives as a named
#: eval arm and in unit tests.  Positive ``hidden_dim`` values build the MLP
#: instead - the older arm still, likewise, not reachable from the app.
LINEAR_HEAD = 0

#: ``hidden_dim`` sentinel selecting the **linear SVM head** - the
#: **production detector head**.  Architecturally identical to
#: :data:`LINEAR_HEAD` (one ``Linear(input_dim, 1)``, so weight serialisation,
#: :func:`build_model_from_weights`, and every scoring path are shared); what
#: differs is the objective that fits it.  :func:`train_model` routes this
#: sentinel to :func:`vtscore.training.svm.fit_linear_svm_head`, which fits a
#: maximum-margin hyperplane (hinge loss + L2) through the very
#: :func:`~vtscore.training.svm.train_svm` call the eval harness scores as
#: ``svm_linear``, rather than running the BCE epoch loop below.
#:
#: The sentinel is negative because it is *not* a width: both linear heads have
#: no hidden layer, and only their loss tells them apart.  Threading it through
#: the existing ``hidden_dim`` plumbing keeps one "which head" knob reaching the
#: final fit and the calibration folds together, which is the property
#: production depends on (fold models must share the final model's fit, or the
#: calibrated threshold does not transfer).
LINEAR_SVM_HEAD = -1

#: Every ``hidden_dim`` that builds a single ``Linear(D, 1)`` rather than an MLP.
LINEAR_HEADS = (LINEAR_HEAD, LINEAR_SVM_HEAD)


def _auto_hidden_dim(n_train: int) -> int:
    """Choose MLP hidden-layer width based on training-set size.

    Keeps the model small when few votes are available to reduce
    overfitting, and grows (up to ``MLP_HIDDEN_MAX``) as more labels
    arrive.  The width is floored at ``MLP_HIDDEN_MIN`` (8): below ~8
    neurons the detector underfits and destabilizes on harder tasks.

    Only the MLP head uses this; the production linear SVM head
    (:data:`LINEAR_SVM_HEAD`) has no hidden layer.
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
        With a linear-head sentinel (:data:`LINEAR_SVM_HEAD`, the production
        head, or :data:`LINEAR_HEAD`) it is a single ``Linear(input_dim, 1)``
        with no hidden layer; ``dropout`` is ignored there (a bare linear map
        has nothing to regularise with dropout).  The two linear sentinels build
        the *same* architecture - they differ only in how :func:`train_model`
        fits it.
    """
    import torch.nn as nn  # noqa: PLC0415

    if hidden_dim in LINEAR_HEADS:
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
        # A linear head: a single ``Linear(input_dim, 1)``, so the only keys are
        # ``0.weight`` / ``0.bias``.  Which of the two linear sentinels fitted
        # those weights is not recoverable from - and irrelevant to - a
        # reconstruction: the objective only shapes the numbers, and they are
        # already in hand.  Either sentinel builds the same architecture.
        hidden_dim = LINEAR_HEAD

    model = build_model(input_dim, hidden_dim=hidden_dim, dropout=0.0)
    state_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in weights.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _amp_context(use_amp: bool, nullcontext):
    """Return the ``(GradScaler, autocast context)`` pair for a training loop.

    Prefers the modern ``torch.amp.GradScaler("cuda", ...)`` API and falls back
    to the legacy ``torch.cuda.amp.GradScaler`` on torch <2.3.  Both are inert
    when *use_amp* is false (CPU/MPS), where the FP32 path keeps training
    bit-for-bit reproducible.
    """
    import torch  # noqa: PLC0415

    grad_scaler_cls = getattr(torch.amp, "GradScaler", None)
    if grad_scaler_cls is not None:
        scaler = grad_scaler_cls("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    return scaler, (torch.autocast(device_type="cuda") if use_amp else nullcontext())


def _train_svm_head(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    input_dim: int,
    seed: int,
    sample_weights: torch.Tensor | None,
) -> nn.Sequential:
    """Fit the :data:`LINEAR_SVM_HEAD` branch of :func:`train_model`.

    Hands the tensors to :func:`vtscore.training.svm.fit_linear_svm_head` as
    numpy (liblinear is a CPU solver) and returns its ``Linear(input_dim, 1)``.
    """
    # liblinear is one blocking call rather than an epoch loop, so honour a
    # cancelled background job once up front instead of at every epoch
    # boundary; the fit itself is milliseconds at any real vote count.
    check_job_cancelled()
    from vtscore.training.svm import fit_linear_svm_head  # noqa: PLC0415

    weights_np = None if sample_weights is None else sample_weights.reshape(-1).detach().cpu().numpy()
    if weights_np is not None and weights_np.shape[0] != len(y_train):
        raise ValueError(f"sample_weights length {weights_np.shape[0]} does not match training-set size {len(y_train)}")
    return fit_linear_svm_head(
        X_train.detach().cpu().numpy(),
        y_train.reshape(-1).detach().cpu().numpy(),
        input_dim,
        seed=seed,
        sample_weight=weights_np,
    )


def train_model(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    input_dim: int,
    seed: int = 42,
    hidden_dim: int | None = None,
    sample_weights: torch.Tensor | None = None,
) -> nn.Sequential:
    """Train the detector head named by *hidden_dim* and return it.

    ``hidden_dim`` picks the head.  :data:`LINEAR_SVM_HEAD` - what every
    production fit passes - short-circuits everything below and delegates to
    :func:`vtscore.training.svm.fit_linear_svm_head`, which fits a
    maximum-margin hyperplane with liblinear instead of running the BCE epoch
    loop; it returns the same ``Linear(input_dim, 1)`` module, so callers see no
    difference beyond the weights.  Everything from here down describes the
    **torch** heads, :data:`LINEAR_HEAD` and the MLP.

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
    model (and therefore every item's score) is independent of inclusion.

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
        hidden_dim: Which head to fit.  :data:`LINEAR_SVM_HEAD` (production)
            and :data:`LINEAR_HEAD` are the two ``Linear(input_dim, 1)``
            heads - hinge-fitted and BCE-fitted respectively; a positive value
            is an MLP hidden width.  ``None`` (default) auto-sizes an MLP via
            :func:`_auto_hidden_dim`.
        sample_weights: Optional per-row loss weights of shape ``(N,)`` or
            ``(N, 1)``.  When ``None`` (default) inverse-class-frequency
            weights are computed internally.  When provided they replace the
            frequency weights entirely (the caller owns class balance).

    Returns:
        A trained ``nn.Sequential`` model in eval mode.
        The model outputs raw logits (the SVM head's decision function) - apply
        ``torch.sigmoid`` at inference.

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

    if hidden_dim == LINEAR_SVM_HEAD:
        return _train_svm_head(X_train, y_train, input_dim, seed, sample_weights)

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
    scaler, autocast_ctx = _amp_context(use_amp, nullcontext)
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
