"""Standalone estimator registry for the label-curve and timing sweeps.

A **sweep trainer** is a self-contained estimator: a callable
``(X_train, y_train, seed) -> predict_fn`` where the returned ``predict_fn``
maps an ``(N, D)`` embedding matrix to per-row ``P(positive)`` scores in
``[0, 1]`` (ensembles additionally return a per-item std; see
:data:`PredictFn`).  It owns its own fit and its own scoring, and knows nothing
about VTSearch's detector pipeline.  :data:`SWEEP_TRAINERS` holds the
fixed-name entries; :func:`resolve_trainer` additionally parses
**parameterised** SVM names such as ``"svm_rbf@C=3,gamma=scale"`` so the
kernel/hyperparameter screen can sweep the SVM configuration space without a
registry entry per point.

**This is not the voting simulation's ``trainer`` knob.**  The two registries
are deliberately named apart (issue #3764), because they answer different
questions and their names used to collide:

* Here — :mod:`vtscore.eval.label_curve` and :mod:`vtscore.eval.timing_benchmark`
  ask *how does this estimator rank, given N labels?*, so every arm is a bare
  estimator and ``"mlp"`` really is an MLP.  The ``gp_*`` arms (issue #3954) are
  Gaussian-process classifiers conditioned on the votes; like the ensembles they
  return a per-item uncertainty beside the score.
* There — :mod:`vtscore.eval.step_trainers`, driven by
  :mod:`vtscore.eval.voting_iterations`, asks *what does VTSearch do with these
  votes?*, so its ``trainer="app"`` arm is the shipped pipeline (whose head is
  picked separately by ``head=``) and its ``svm_*`` arms are the estimators
  here, wrapped in a threshold rule.

Both sweeps share the parameterised SVM parser and the trainer-agnostic
cross-calibration threshold below, which is why this module is imported from
both.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


PredictFn = Callable[[np.ndarray], "np.ndarray | tuple[np.ndarray, np.ndarray]"]
"""Callable returning per-row P(positive) - optionally with per-item std.

A plain trainer returns just ``scores`` (P(positive) in ``[0, 1]``).  An
*ensemble* trainer returns ``(scores, per_item_std)`` where ``scores`` is
the mean sigmoid across ensemble members and ``per_item_std`` is the
member-to-member standard deviation - a cheap epistemic-uncertainty
signal.  Callers that only need the ranking pull ``scores`` out via
:func:`_as_scores`; callers that want the uncertainty (the diagnostic
``std_mean`` column) unpack the tuple explicitly.
"""

TrainerFn = Callable[[np.ndarray, np.ndarray, int], PredictFn]
"""Trainer signature: ``(X_train, y_train, seed) -> predict_fn``."""


def _as_scores(result: "np.ndarray | tuple[np.ndarray, np.ndarray]") -> np.ndarray:
    """Return the score array from a ``predict()`` result.

    Ensemble trainers return ``(scores, per_item_std)``; every other
    trainer returns a bare score array.  This collapses both to the score
    array so ranking metrics and threshold calibration don't have to care
    which trainer produced the prediction.
    """
    if isinstance(result, tuple):
        return np.asarray(result[0], dtype=np.float64)
    return np.asarray(result, dtype=np.float64)


def _train_mlp(X: np.ndarray, y: np.ndarray, seed: int) -> PredictFn:
    """Adapt :func:`vtscore.training.mlp.train_model` to the sweep API."""
    import torch  # noqa: PLC0415

    from vtscore.training.mlp import train_model

    X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
    y_t = torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(1)
    model = train_model(X_t, y_t, input_dim=X.shape[1], seed=seed)

    def predict(X_test: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X_test, dtype=np.float32)
        with torch.no_grad():
            t = torch.from_numpy(X_arr).to(next(model.parameters()).device)
            return torch.sigmoid(model(t)).squeeze(1).cpu().numpy()

    return predict


def _train_mlp_ensemble_factory(n_seeds: int) -> TrainerFn:
    """Build a trainer that averages *n_seeds* seed-varied MLPs.

    Each member is a full :func:`vtscore.training.mlp.train_model` run on
    the same labels but a different weight-init/dropout seed, so the
    ensemble captures the MLP's epistemic uncertainty (how much the
    decision surface wobbles under reseeding) rather than aleatoric label
    noise.  The returned ``predict`` reports the mean sigmoid as the score
    and the member-to-member standard deviation as ``per_item_std`` - high
    where the members disagree, low where they agree.
    """

    def trainer(X: np.ndarray, y: np.ndarray, seed: int) -> PredictFn:
        import torch  # noqa: PLC0415

        from vtscore.training.mlp import train_model

        X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
        y_t = torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(1)
        input_dim = X.shape[1]
        # Distinct seeds per member: ``seed + k`` keeps the whole ensemble a
        # deterministic function of the cell's ``seed`` while decorrelating
        # the members' weight inits.
        models = [train_model(X_t, y_t, input_dim=input_dim, seed=seed + k) for k in range(n_seeds)]

        def predict(X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            X_arr = np.asarray(X_test, dtype=np.float32)
            member_scores: list[np.ndarray] = []
            for model in models:
                with torch.no_grad():
                    t = torch.from_numpy(X_arr).to(next(model.parameters()).device)
                    member_scores.append(torch.sigmoid(model(t)).squeeze(1).cpu().numpy())
            stacked = np.stack(member_scores, axis=0)  # (n_seeds, n_items)
            return stacked.mean(axis=0), stacked.std(axis=0)

        return predict

    return trainer


# Kernels that :func:`resolve_trainer` accepts as ``gp_<kernel>[@params]``.
_GP_KERNELS = frozenset({"rbf", "dot"})

# Gauss-Hermite nodes/weights for E[g(f)] under f ~ N(mean, std^2):
# E[g(f)] = sum_k w_k g(mean + sqrt(2) std x_k) / sqrt(pi).
_GH_NODES, _GH_WEIGHTS = np.polynomial.hermite.hermgauss(32)


def _sigmoid_posterior_std(mean_f: np.ndarray, std_f: np.ndarray) -> np.ndarray:
    """Posterior std of ``sigmoid(f)`` for ``f ~ N(mean_f, std_f^2)``, per item.

    Exact to quadrature precision, so it is bounded by 0.5 like any std of a
    [0, 1] variable and comparable across items whatever their latent scale -
    which the delta-method ``p (1 - p) std_f`` is not once ML-II pushes the
    amplitude up and the latent std past a few units.
    """
    from scipy.special import expit  # noqa: PLC0415

    f = mean_f[:, np.newaxis] + np.sqrt(2.0) * std_f[:, np.newaxis] * _GH_NODES[np.newaxis, :]
    g = expit(f)
    w = _GH_WEIGHTS / np.sqrt(np.pi)
    m1 = g @ w
    m2 = (g * g) @ w
    return np.sqrt(np.clip(m2 - m1 * m1, 0.0, None))


def _train_gp_factory(
    kernel: str,
    *,
    length_scale: float = 1.0,
    amplitude: float = 1.0,
    optimize: bool = True,
) -> TrainerFn:
    """Build a Gaussian-process-classifier trainer (issue #3954).

    The GP is scikit-learn's :class:`~sklearn.gaussian_process.GaussianProcessClassifier`
    - a latent GP with a logistic likelihood, fitted by the Laplace
    approximation - conditioned on the vote embeddings.  Two kernels:

    * ``"rbf"`` - ``amplitude * RBF(length_scale)``.  Every embedding here is
      L2-normalised, so the squared distance the kernel reads is
      ``2 - 2 cos``: a length scale of 1 puts one e-fold of correlation at a
      cosine of 0.5.
    * ``"dot"`` - ``amplitude * DotProduct(sigma_0=length_scale)``, a Bayesian
      linear classifier: the closest GP analogue of the shipped linear head,
      and the arm that separates "a GP" from "a curved boundary".

    With *optimize* (the default) the kernel hyperparameters are fitted by
    marginal likelihood (ML-II) on every call; ``optimize=False`` pins them at
    the values given, the arm to reach for when a handful of votes lets ML-II
    collapse the amplitude to its bound and score everything at 0.5.

    The returned ``predict`` reports ``(P(positive), per_item_std)`` like the
    ensembles: the score is the GP's predictive probability and the std is the
    posterior standard deviation of ``sigmoid(f)`` - the spread of the
    probability itself under the latent posterior ``N(mean, var)``, integrated
    by Gauss-Hermite quadrature - so it lives on the same [0, 1] scale as the
    score (never above 0.5) and the ``std_mean`` column reads it directly.  The
    latent mean and variance are read off the fitted estimator with the
    identities sklearn's own ``predict_proba`` uses (its ``X_train_`` / ``pi_``
    / ``W_sr_`` / ``L_`` fitted attributes), because the public API exposes
    only the integrated probability and the acquisition rules in
    :mod:`vtscore.eval.al_strategies` need the spread as well.
    """
    if kernel not in _GP_KERNELS:
        raise ValueError(f"Unknown GP kernel {kernel!r}; choices: {sorted(_GP_KERNELS)}")

    def trainer(X: np.ndarray, y: np.ndarray, seed: int) -> PredictFn:
        import warnings  # noqa: PLC0415

        from scipy.linalg import solve_triangular  # noqa: PLC0415
        from sklearn.exceptions import ConvergenceWarning  # noqa: PLC0415
        from sklearn.gaussian_process import GaussianProcessClassifier  # noqa: PLC0415
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct  # noqa: PLC0415

        X64 = np.asarray(X, dtype=np.float64)
        y_int = np.asarray(y).astype(int).ravel()
        if len(set(y_int.tolist())) < 2:
            raise ValueError("gp trainer needs both classes in the training labels")

        fixed: Any = "fixed"
        amp = ConstantKernel(amplitude, constant_value_bounds=(1e-1, 1e2) if optimize else fixed)
        if kernel == "rbf":
            base: Any = RBF(length_scale, length_scale_bounds=(1e-1, 1e1) if optimize else fixed)
        else:
            base = DotProduct(sigma_0=length_scale, sigma_0_bounds=(1e-2, 1e1) if optimize else fixed)
        # ``None`` (no optimizer) is a documented value sklearn's stub types as ``str``.
        optimizer: Any = "fmin_l_bfgs_b" if optimize else None
        clf = GaussianProcessClassifier(kernel=amp * base, optimizer=optimizer, random_state=seed)
        with warnings.catch_warnings():
            # ML-II running into a bound is a property of the vote set the
            # report reads off ``std_mean``, not a fault to print per fold.
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf.fit(X64, y_int)
        # sklearn types ``base_estimator_`` as the union of every estimator
        # shape it can hold; a binary fit always holds the Laplace binary one.
        est: Any = clf.base_estimator_
        pos_col = int(np.searchsorted(clf.classes_, 1))

        def predict(X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            Xt = np.asarray(X_test, dtype=np.float64)
            p = np.asarray(clf.predict_proba(Xt))[:, pos_col]
            # Latent posterior variance (Rasmussen & Williams eq. 3.24), the
            # same computation sklearn's predict_proba runs before it integrates.
            K_star = est.kernel_(est.X_train_, Xt)
            f_star = K_star.T.dot(est.y_train_ - est.pi_)
            v = solve_triangular(est.L_, est.W_sr_[:, np.newaxis] * K_star, lower=True)
            var_f = est.kernel_.diag(Xt) - np.einsum("ij,ij->j", v, v)
            std_f = np.sqrt(np.clip(var_f, 0.0, None))
            std_p = _sigmoid_posterior_std(f_star, std_f)
            return p.astype(np.float64), std_p.astype(np.float64)

        return predict

    return trainer


def _train_svm_factory(kernel: str, **svm_kwargs: Any) -> TrainerFn:
    """Build a sweep-shaped trainer that fits an SVM with the given kernel.

    ``svm_kwargs`` (``C``, ``gamma``, ``gamma_mult``, ``degree``) are forwarded
    to :func:`vtscore.training.svm.train_svm`, so :func:`resolve_trainer` can
    turn a parameterised name into a concrete trainer.
    """

    def trainer(X: np.ndarray, y: np.ndarray, seed: int) -> PredictFn:
        from vtscore.training.svm import train_svm

        clf = train_svm(X, y, kernel=kernel, seed=seed, **svm_kwargs)  # type: ignore[arg-type]
        return clf.predict_proba

    return trainer


# Registry of fixed-name sweep trainers.  Adding a new *named* candidate model
# here is the only place it needs to be plugged in for both sweeps that use this
# registry.  Parameterised SVM configurations do not need an entry - see
# :func:`resolve_trainer`.
#
# ``"mlp"`` here is a genuine MLP: ``vtscore.training.mlp.train_model`` with an
# auto-sized hidden layer, which is the head VTSearch shipped *before* #2790.
# It is emphatically **not** the head the app ships today (the linear SVM,
# ``vtscore.training.mlp.LINEAR_SVM_HEAD``); nothing in this registry is, because
# these arms are bare estimators rather than the app's pipeline.  Read a
# label-curve row's ``trainer`` column with that in mind, and use the voting
# simulation's ``trainer="app"`` arm when the question is about the shipped
# detector.
SWEEP_TRAINERS: dict[str, TrainerFn] = {
    "mlp": _train_mlp,
    "svm_linear": _train_svm_factory("linear"),
    "svm_rbf": _train_svm_factory("rbf"),
    # MLP ensembles: N seed-varied members, mean sigmoid as the score and
    # member disagreement as per-item uncertainty.  Registered at 3/5/7/10
    # members so the sweep can trace how ranking quality and the reported
    # ``std_mean`` move with ensemble size.
    "mlp_ens3": _train_mlp_ensemble_factory(3),
    "mlp_ens5": _train_mlp_ensemble_factory(5),
    "mlp_ens7": _train_mlp_ensemble_factory(7),
    "mlp_ens10": _train_mlp_ensemble_factory(10),
    # Gaussian-process classifiers (issue #3954), hyperparameters fitted by
    # marginal likelihood.  ``gp_<kernel>@ls=..,amp=..,fixed`` pins them instead
    # - see :func:`_parse_gp_spec`.
    "gp_rbf": _train_gp_factory("rbf"),
    "gp_dot": _train_gp_factory("dot"),
}


# Kernels that :func:`resolve_trainer` accepts as ``svm_<kernel>[@params]``.
_SVM_KERNELS = {"linear": "linear", "rbf": "rbf", "poly": "poly", "sigmoid": "sigmoid"}


def _parse_gp_spec(name: str) -> tuple[str, dict[str, Any]]:
    """Split ``"gp_rbf@ls=0.5,amp=2,fixed"`` into ``("rbf", {...factory kwargs})``.

    Tokens: ``ls=<float>`` (the RBF length scale, or ``DotProduct``'s
    ``sigma_0``), ``amp=<float>`` (the kernel amplitude) and the bare flag
    ``fixed`` (skip ML-II and keep the values given).  Raises ``KeyError`` for a
    name that is not a ``gp_<kernel>`` spec and ``ValueError`` for a malformed
    parameter, so a typo fails loudly.
    """
    base, _, param_str = name.partition("@")
    if not base.startswith("gp_"):
        raise KeyError(f"Unknown trainer {name!r}; not a gp_<kernel>[@ls=..,amp=..,fixed] spec")
    kernel = base[len("gp_") :]
    if kernel not in _GP_KERNELS:
        raise KeyError(f"Unknown GP kernel {kernel!r}; choices: {sorted(_GP_KERNELS)}")
    kwargs: dict[str, Any] = {}
    if param_str:
        for token in param_str.split(","):
            if not token:
                continue
            key, sep, value = token.partition("=")
            key = key.strip()
            if key == "fixed" and not sep:
                kwargs["optimize"] = False
                continue
            if not sep:
                raise ValueError(f"Malformed GP trainer parameter {token!r} in {name!r} (expected key=value)")
            if key == "ls":
                kwargs["length_scale"] = float(value)
            elif key == "amp":
                kwargs["amplitude"] = float(value)
            else:
                raise ValueError(f"Unknown GP trainer parameter {key!r} (expected ls, amp, or fixed)")
    return kernel, kwargs


def _coerce_param(key: str, value: str) -> Any:
    """Coerce one ``key=value`` token from a parameterised trainer name.

    ``C`` is a float; ``degree`` an int; ``gamma`` is either the sklearn string
    ``"scale"``/``"auto"``, a bare float, or a ``"<mult>x"`` multiplier of the
    ``scale`` heuristic (e.g. ``"4x"`` / ``"0.25x"``) — the last is returned as
    ``("gamma_mult", <float>)`` so the SVM keeps sklearn's data-driven ``scale``
    and merely rescales it.
    """
    if key == "C":
        return ("C", float(value))
    if key == "degree":
        return ("degree", int(value))
    if key == "gamma":
        if value in ("scale", "auto"):
            return ("gamma", value)
        if value.endswith("x"):
            return ("gamma_mult", float(value[:-1]))
        return ("gamma", float(value))
    raise ValueError(f"Unknown SVM trainer parameter {key!r} (expected C, gamma, or degree)")


def _parse_trainer_spec(name: str) -> tuple[str, dict[str, Any]]:
    """Split ``"svm_rbf@C=3,gamma=4x"`` into ``("rbf", {"C": 3.0, "gamma_mult": 4.0})``.

    Raises ``KeyError`` for a name that is neither a registry entry nor a
    recognised ``svm_<kernel>`` spec, so a typo fails loudly.
    """
    base, _, param_str = name.partition("@")
    if not base.startswith("svm_"):
        raise KeyError(
            f"Unknown trainer {name!r}; choices: {sorted(SWEEP_TRAINERS)}, "
            "svm_<kernel>[@C=..,gamma=..,degree=..] or gp_<kernel>[@ls=..,amp=..,fixed]"
        )
    kernel_key = base[len("svm_") :]
    if kernel_key not in _SVM_KERNELS:
        raise KeyError(f"Unknown SVM kernel {kernel_key!r}; choices: {sorted(_SVM_KERNELS)}")
    kwargs: dict[str, Any] = {}
    if param_str:
        for token in param_str.split(","):
            if not token:
                continue
            key, sep, value = token.partition("=")
            if not sep:
                raise ValueError(f"Malformed trainer parameter {token!r} in {name!r} (expected key=value)")
            k, v = _coerce_param(key.strip(), value.strip())
            kwargs[k] = v
    return _SVM_KERNELS[kernel_key], kwargs


def resolve_trainer(name: str) -> TrainerFn:
    """Return the :class:`TrainerFn` for *name*.

    Accepts a fixed registry key (``"mlp"``, ``"svm_linear"``, an ensemble,
    ``"gp_rbf"``), a parameterised SVM spec (``"svm_rbf@C=3,gamma=scale"``,
    ``"svm_poly@degree=2,C=0.3"``, ``"svm_linear@C=0.03"``) or a parameterised
    GP spec (``"gp_rbf@ls=0.5,fixed"``, see :func:`_parse_gp_spec`).  The bare
    ``"svm_linear"`` / ``"svm_rbf"`` names resolve to the registry entries
    unchanged, so existing callers are byte-identical.
    """
    if name in SWEEP_TRAINERS:
        return SWEEP_TRAINERS[name]
    if name.startswith("gp_"):
        gp_kernel, gp_kwargs = _parse_gp_spec(name)
        return _train_gp_factory(gp_kernel, **gp_kwargs)
    kernel, kwargs = _parse_trainer_spec(name)
    return _train_svm_factory(kernel, **kwargs)


def _cross_calibrated_threshold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    trainer_fn: TrainerFn,
    seed: int,
    *,
    inclusion_value: int = 0,
    calibrate_count: int = 2,
    cal_fraction: float = 0.5,
) -> float:
    """Trainer-agnostic port of ``calculate_cross_calibration_threshold``.

    Mirrors what production does at vote time: split the labels k ways
    into train/cal halves, retrain on each half, score the held-out cal
    half, then pool every fold's (score, label) pairs and apply the
    conformal inclusion rule once.  This is the threshold ``f1_at_xcal``
    is measured at, so a trainer that ranks well but doesn't admit a
    stable cross-validated threshold pays the price here.

    Returns ``0.5`` when the label budget is too small to form valid
    splits (mirrors the production fallback).
    """
    from vtscore.training.thresholds import conformal_threshold

    n = int(y_train.size)
    if n < 4:
        return 0.5
    n_cal = max(1, round(n * cal_fraction))
    n_tr = n - n_cal
    if n_tr < 2 or n_cal < 1:
        return 0.5

    rng = np.random.default_rng(seed)
    pooled_scores: list[float] = []
    pooled_labels: list[float] = []
    for k in range(max(1, calibrate_count)):
        order = rng.permutation(n)
        tr_idx = order[:n_tr]
        cal_idx = order[n_tr:]
        # Single-class splits would crash the trainer or short-circuit
        # the threshold rule; just skip this fold.
        if len({int(v) for v in y_train[tr_idx]}) < 2:
            continue
        if len({int(v) for v in y_train[cal_idx]}) < 2:
            continue
        try:
            predict = trainer_fn(X_train[tr_idx], y_train[tr_idx], seed + k)
        except ValueError:
            continue
        pooled_scores.extend(_as_scores(predict(X_train[cal_idx])).tolist())
        pooled_labels.extend(float(v) for v in y_train[cal_idx])

    if not pooled_scores:
        return 0.5
    return conformal_threshold(pooled_scores, pooled_labels, inclusion_value)
