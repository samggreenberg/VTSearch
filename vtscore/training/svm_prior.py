"""The linear SVM started from the Goods' centroid: the app's head in a single-vector space (#4776, #4785).

Below the label quota Test gives the Goods' centroid; at the quota it hands over
to the trained head, a linear SVM on 3 Goods and 4 Bads.  On FHIBE's face crops
that first head ranks a person worse than the centroid it replaces (AP -0.17 to
-0.22, #4762): with so few labels its direction is mostly the difference
between the Goods and a handful of other people, and the Bads' share of it is
noise.  The centroid is a better prior than the origin.

:func:`fit_linear_svm_prior_head` fits the shipped head's objective (squared
hinge, L2, C = :data:`~vtscore.config.SVM_HEAD_C`, balanced classes) on the
votes plus one feature, :data:`CENTROID_PRIOR_SCALE` times each vector's dot
with the unit Goods' centroid of the very rows it is fitted on.  A weight ``v``
on that feature moves the hyperplane ``v * scale`` along the centroid at a
penalty of ``v ** 2``, so the centroid direction costs ``1 / scale ** 2`` of any
other: the head starts at the centroid and leaves it only as far as the labels
pay for.  The fit folds back into one ``Linear(D, 1)``, ``w + scale * v *
centroid``, so the calibration folds (each with the centroid of its own Goods),
region max-pooling and weight serialisation take it as they take the SVM.

The primal solver: the scaled feature leaves liblinear's dual coordinate
descent crawling to its iteration cap, at the same optimum (#4776's harness
check: direction cosine 1.00000, 5 ms against up to 2.5 s).

It is reached through :func:`vtscore.training.mlp.train_model` via the
:data:`~vtscore.training.mlp.LINEAR_SVM_PRIOR_HEAD` sentinel.  The app trains
it wherever a detector learns in a single-vector space
(:func:`vtscore.detectors.training.production_head_for`, #4785); a patch space
keeps the plain SVM until the Region path is priced.  The eval harness names it
``head="linear_svm_prior"``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: How much cheaper the centroid direction is than any other, as a feature
#: scale (the penalty ratio is its square).  10 ranked the withheld half best of
#: 1, 3 and 10 over votes 1-50 in #4776's shadow pricing, on faces and on COCO.
CENTROID_PRIOR_SCALE: float = 10.0


def centroid_prior_direction(
    X: np.ndarray,
    y: np.ndarray,
    *,
    scale: float = CENTROID_PRIOR_SCALE,
    seed: int = 42,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """``(w, b)`` of the SVM started from the Goods' centroid of *X*'s positive rows.

    Raises :class:`ValueError` without both classes.
    """
    from sklearn.svm import LinearSVC  # noqa: PLC0415

    from vtscore import config  # noqa: PLC0415
    from vtscore.detectors.centroid_head import goods_centroid  # noqa: PLC0415

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).reshape(-1)
    goods = X[y > 0.5]
    if not len(goods) or len(goods) == len(X):
        raise ValueError("the centroid-prior head needs both Goods and Bads")
    centroid = np.asarray(goods_centroid(list(goods)), dtype=np.float64)
    clf = LinearSVC(
        C=config.SVM_HEAD_C,
        class_weight=None if sample_weight is not None else "balanced",
        # The primal solver (see the module docstring).  The stubs type ``dual`` as a
        # string, but scikit-learn takes ``False``; ``"auto"`` would pick the dual here.
        dual=False,  # pyright: ignore[reportArgumentType]
        max_iter=5000,
        random_state=seed,
    )
    clf.fit(np.hstack([X, scale * (X @ centroid)[:, None]]), (y > 0.5).astype(int), sample_weight=sample_weight)
    w = np.asarray(clf.coef_, dtype=np.float64).ravel()
    return w[:-1] + scale * w[-1] * centroid, float(np.asarray(clf.intercept_).ravel()[0])


def fit_linear_svm_prior_head(
    X: np.ndarray,
    y: np.ndarray,
    input_dim: int,
    *,
    seed: int = 42,
    sample_weight: np.ndarray | None = None,
) -> Any:
    """Fit :func:`centroid_prior_direction` and return it as the torch ``Linear(D, 1)`` every head is."""
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415

    from vtscore.embedding.loader import ensure_torch_configured, get_torch_device  # noqa: PLC0415
    from vtscore.training.mlp import LINEAR_SVM_PRIOR_HEAD, build_model  # noqa: PLC0415

    w, b = centroid_prior_direction(X, y, seed=seed, sample_weight=sample_weight)
    if w.shape[0] != input_dim:
        raise ValueError(f"centroid-prior fit produced {w.shape[0]} coefficients but input_dim is {input_dim}")
    ensure_torch_configured()
    model = build_model(input_dim, hidden_dim=LINEAR_SVM_PRIOR_HEAD)
    layer: nn.Linear = model[0]  # type: ignore[assignment]  # a linear head builds exactly this
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(w.astype(np.float32)).unsqueeze(0))
        layer.bias.fill_(b)
    model.eval()
    return model.to(get_torch_device())
