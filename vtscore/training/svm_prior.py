"""The linear SVM started from the Goods' centroid: the app's detector head (#4776, #4785).

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
:data:`~vtscore.training.mlp.LINEAR_SVM_PRIOR_HEAD` sentinel, which every
production fit passes since #4785, in every space: priced as full sessions it
gained on FHIBE's faces and COCO Better's Binary opening, and on the Region path
it ranked slightly better at a hair's cost in F-beta (−0.002 ± 0.002), and the
owner chose one head everywhere.  The eval harness names it
``head="linear_svm_prior"``; ``head="linear_svm"`` is the plain SVM before it.

**A typed-query detector has a second prior, its typed query's direction (#4384,
#4859).**  Started from the centroid alone, the first head ranked the corpus worse
than the typed query it replaces until about vote 25 (AP 0.34 at its first fit
against the query's 0.42), because three Goods off the top of the query's own sort
say less about the class than the words that found them.  Within
:func:`typed_query_prior` every fit adds a second cheap feature, the dot with the
query's unit embedding at :data:`QUERY_PRIOR_SCALE`, so the head starts from the
text sort as well as from the Goods and leaves both as the labels pay for.  Priced
on full sessions in #4384: the head's AP at its first fit 0.43, the objective over
votes 1-150 +0.013 / +0.013 / +0.018 at beta 1/4, 1 and 4.  The training entry
points enter it for a detector made from a typed query on that query's own
single-vector space (:func:`vtscore.detectors.query_head.typed_query_prior_vector`);
the query's feature is dropped in a space of another size, so a region bag's fit
never takes it.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import numpy as np

#: How much cheaper the centroid direction is than any other, as a feature
#: scale (the penalty ratio is its square).  10 ranked the withheld half best of
#: 1, 3 and 10 over votes 1-50 in #4776's shadow pricing, on faces and on COCO.
CENTROID_PRIOR_SCALE: float = 10.0

#: How much cheaper the typed query's direction is than any other, as a feature
#: scale (#4384, #4859).  The query's feature spreads about a third as widely over
#: SigLIP's images as the centroid's, so 10 here pulls less than 10 on the
#: centroid.  Priced against 3 at every preset: 10 ranked better (AP +0.008) and
#: won at beta 4 (+0.007 on the objective); 3 kept about 0.6 more Goods by vote 150.
#: ``0`` turns the prior off (the eval's arm before, :func:`vtscore.eval.arms_query_prior.query_prior_scale`).
QUERY_PRIOR_SCALE: float = 10.0

#: The typed query of the training pass in progress, ``(unit direction, text)`` (:func:`typed_query_prior`).
_TYPED_QUERY: ContextVar[tuple[np.ndarray, str] | None] = ContextVar("vtsearch_typed_query", default=None)

#: The attribute a head fitted under a typed query's prior carries: that query's text.
QUERY_PRIOR_TEXT_ATTR = "vtsearch_query_prior_text"

#: The weight a hard-view Good (a face under its space's Face Quality cut,
#: :func:`vtscore.media.face.imagekit_bridge.is_hard_view`) takes in the head's fit
#: while some Good is clear, where a clear Good weighs 1 (#5413).  On FHIBE's
#: four-photo sessions with one hard starting face it raised F at 50 votes by
#: +0.02 / +0.04 at beta 4 and +0.01 / +0.02 at beta 1 (1024 / 640 px), and
#: changed nothing where no Good is hard.  0.1 did as well; 0 (out of the fit)
#: and weighting hard Bads too did no better
#: (``docs/experiments/2026-10-29-hard-face-train-weight-5413``).
HARD_VIEW_GOOD_WEIGHT: float = 0.25

#: The training pass's hard-view rows, as their float32 bytes (:func:`hard_view_rows`).
_HARD_VIEW_ROWS: ContextVar[frozenset[bytes] | None] = ContextVar("vtsearch_hard_view_rows", default=None)


@contextmanager
def typed_query_prior(direction: "np.ndarray | None", text: str = "") -> Iterator[None]:
    """Within the block, every centroid-prior fit also starts from the typed query *direction*.

    *text* is the query the direction embeds; the heads fitted in the block carry
    it (:func:`fitted_query_prior_text`), so a cache can tell a head fitted from
    this query from one fitted without it.  ``None`` changes nothing.  Scoped
    to the calling context (a ``ContextVar``), so two detectors training on two
    threads never see each other's query; the calibration folds and the final
    fit run in the caller's thread, so all of a training pass's fits take it.
    """
    if direction is None:
        yield
        return
    v = np.asarray(direction, dtype=np.float64).ravel()
    token = _TYPED_QUERY.set((v / np.linalg.norm(v), str(text)))
    try:
        yield
    finally:
        _TYPED_QUERY.reset(token)


def _row_key(row: np.ndarray) -> bytes:
    return np.asarray(row, dtype=np.float32).ravel().tobytes()


@contextmanager
def hard_view_rows(vectors: "Iterable[np.ndarray] | None") -> Iterator[None]:
    """Within the block, a fitted row equal to one of *vectors* is a hard-view face (#5413).

    The caller names the vectors of its hard-view faces; every centroid-prior fit
    in the block, the calibration folds' included, weights such a row when it is
    a Good (:func:`hard_view_good_weights`).  Rows are matched by value, so a fold
    that fits on a subset of the rows needs nothing threaded through.  ``None``
    or no vectors changes nothing.  Scoped to the calling context, as
    :func:`typed_query_prior` is.
    """
    keys = frozenset(_row_key(v) for v in vectors) if vectors is not None else frozenset()
    if not keys:
        yield
        return
    token = _HARD_VIEW_ROWS.set(keys)
    try:
        yield
    finally:
        _HARD_VIEW_ROWS.reset(token)


def hard_view_good_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """Each Good row's weight in the fit (:data:`HARD_VIEW_GOOD_WEIGHT` for a hard view, else 1), or ``None``.

    ``None`` when no Good row is a hard view (:func:`hard_view_rows`), or every
    one is, so there is no clear Good to lean on: the fit is then the plain one.
    The weights are the Goods' alone, in row order; a Bad is never weighted.
    """
    keys = _HARD_VIEW_ROWS.get()
    if keys is None or HARD_VIEW_GOOD_WEIGHT == 1.0:
        return None
    goods = np.asarray(X)[np.asarray(y).reshape(-1) > 0.5]
    hard = np.array([_row_key(row) in keys for row in goods], dtype=bool)
    if not hard.any() or hard.all():
        return None
    return np.where(hard, HARD_VIEW_GOOD_WEIGHT, 1.0)


def active_query_prior() -> tuple[np.ndarray, float] | None:
    """The ``(unit direction, scale)`` the fits in progress take, or ``None`` (no typed query, or the scale is 0)."""
    held = _TYPED_QUERY.get()
    if held is None or not QUERY_PRIOR_SCALE > 0:
        return None
    return held[0], float(QUERY_PRIOR_SCALE)


def fitted_query_prior_text(model: object) -> str | None:
    """The typed query a head was fitted from (:func:`typed_query_prior`), or ``None`` for one fitted without."""
    return getattr(model, QUERY_PRIOR_TEXT_ATTR, None)


def active_query_prior_key() -> bytes | None:
    """A digest of :func:`active_query_prior`, for caches of fitted models (``None`` without one)."""
    prior = active_query_prior()
    if prior is None:
        return None
    import hashlib  # noqa: PLC0415

    q, s = prior
    return hashlib.sha1(np.ascontiguousarray(q, dtype=np.float64).tobytes() + repr(s).encode()).digest()


def centroid_prior_direction(
    X: np.ndarray,
    y: np.ndarray,
    *,
    scale: float = CENTROID_PRIOR_SCALE,
    seed: int = 42,
    sample_weight: np.ndarray | None = None,
    query: tuple[np.ndarray, float] | None = None,
) -> tuple[np.ndarray, float]:
    """``(w, b)`` of the SVM started from the Goods' centroid of *X*'s positive rows.

    *query* is a second prior, ``(direction, scale)`` (#4384); unset, it is the
    typed query of the pass in progress (:func:`active_query_prior`).  A query
    scale of 0 adds nothing, and a *scale* of 0 drops the centroid's feature, so
    the query can be priced alone.  Raises :class:`ValueError` without both classes.
    """
    from sklearn.svm import LinearSVC  # noqa: PLC0415

    from vtscore import config  # noqa: PLC0415
    from vtscore.detectors.centroid_head import goods_centroid  # noqa: PLC0415

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).reshape(-1)
    pos = y > 0.5
    goods = X[pos]
    if not len(goods) or len(goods) == len(X):
        raise ValueError("the centroid-prior head needs both Goods and Bads")
    # A hard-view Good weighs less in the loss and in the centroid the head starts from (#5413).
    good_weights = hard_view_good_weights(X, y)
    if good_weights is not None:
        sample_weight = _weigh_goods(pos, sample_weight, good_weights)
    centroid = np.asarray(goods_centroid(list(goods), good_weights), dtype=np.float64)
    clf = LinearSVC(
        C=config.SVM_HEAD_C,
        class_weight=None if sample_weight is not None else "balanced",
        # The primal solver (see the module docstring).  The stubs type ``dual`` as a
        # string, but scikit-learn takes ``False``; ``"auto"`` would pick the dual here.
        dual=False,  # pyright: ignore[reportArgumentType]
        max_iter=5000,
        random_state=seed,
    )
    priors = [(float(scale), centroid)] if scale > 0 else []
    query = active_query_prior() if query is None else query
    if query is not None and query[1] > 0 and np.asarray(query[0]).shape == (X.shape[1],):
        direction = np.asarray(query[0], dtype=np.float64)
        priors.append((float(query[1]), direction / np.linalg.norm(direction)))
    features = [X, *[(s * (X @ d))[:, None] for s, d in priors]]
    clf.fit(np.hstack(features), (y > 0.5).astype(int), sample_weight=sample_weight)
    coef = np.asarray(clf.coef_, dtype=np.float64).ravel()
    w = coef[: X.shape[1]].copy()
    for k, (s, d) in enumerate(priors):
        w += s * coef[X.shape[1] + k] * d
    return w, float(np.asarray(clf.intercept_).ravel()[0])


def _weigh_goods(pos: np.ndarray, sample_weight: np.ndarray | None, good_weights: np.ndarray) -> np.ndarray:
    """Per-row loss weights with the Goods shared out by *good_weights*, each class keeping its mass.

    Without *sample_weight* each class's mass is what ``class_weight="balanced"``
    gives it (half the rows); with one, it is that class's own total.  Within the
    Goods the mass goes by *good_weights*, so a hard Good counts less than the other
    Goods while the Goods together count as before.
    """
    n = len(pos)
    base = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64).ravel()
    out = base.copy() if sample_weight is not None else np.where(pos, n / (2.0 * pos.sum()), n / (2.0 * (~pos).sum()))
    share = base[pos] * good_weights
    mass = n / 2.0 if sample_weight is None else base[pos].sum()
    out[pos] = mass * share / share.sum()
    return out


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
    held = _TYPED_QUERY.get()
    prior_text = (
        held[1] if held is not None and active_query_prior() is not None and len(held[0]) == input_dim else None
    )
    if w.shape[0] != input_dim:
        raise ValueError(f"centroid-prior fit produced {w.shape[0]} coefficients but input_dim is {input_dim}")
    ensure_torch_configured()
    model = build_model(input_dim, hidden_dim=LINEAR_SVM_PRIOR_HEAD)
    layer: nn.Linear = model[0]  # type: ignore[assignment]  # a linear head builds exactly this
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(w.astype(np.float32)).unsqueeze(0))
        layer.bias.fill_(b)
    model.eval()
    model = model.to(get_torch_device())
    if prior_text is not None:
        setattr(model, QUERY_PRIOR_TEXT_ATTR, prior_text)
    return model
