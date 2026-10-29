"""Calibration folds: the Train/Calibrate splits and the per-fold fits behind them.

Draws the stratified Train/Calibrate splits (row by row, or by voted image on
the bag-aware path), trains a head on each fold's Train side, and returns each
fold's held-out ``(scores, labels)`` (:func:`compute_fold_orderings`), or each
held-out bag's node scores (:func:`compute_grouped_fold_node_scores`).  Where a
cut goes is not decided here: the orderings are inclusion-independent, and
:mod:`~vtscore.training.thresholds.conformal` caches them
(:func:`~vtscore.training.thresholds.conformal.calibration_folds_cached`) and
reads its quantile off them.  This module reads nothing from the conformal
rule.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from vtscore.training.thresholds.knobs import NO_GOOD_THRESHOLD, NON_FINITE_SCORE_SENTINEL, sigmoid_to_finite_scores

# Seed of the Train/Calibrate fold splits when a caller passes no ``rng``.
# The splits must be reproducible: the same labelset has to yield the same fold
# models, hence the same conformal threshold, hence the same Good/Bad verdicts
# on every run and every reload of a detector.  Falling back to the *global*
# ``np.random`` would break that twice over - the splits would differ run to
# run, and the request/background threads that run Find scoring would mutate
# shared global RNG state, making results order-dependent under concurrency.
CALIBRATION_SPLIT_SEED = 42


def _per_bag_fit_weights(
    y_rows: np.ndarray,
    group_rows: list,
) -> np.ndarray:
    """Per-row loss weights that balance Good votes against Bad **bags**.

    Every Good row weighs ``n_bad_bags / n_good``; every Bad row weighs
    ``1 / (rows in its bag)`` so each Bad image contributes exactly one image's
    worth of negative mass regardless of how many region nodes it flooded in.
    Total Good mass (``n_bad_bags``) equals total Bad mass (``n_bad_bags``), so
    the classes are balanced at the same magnitude as :func:`train_model`'s
    default inverse-frequency weights - only the *unit* changes from row to bag.
    """
    from collections import Counter  # noqa: PLC0415

    n_good = sum(1 for lbl in y_rows if lbl == 1.0)
    bad_groups = {g for g, lbl in zip(group_rows, y_rows, strict=True) if lbl == 0.0}
    n_bad_bags = len(bad_groups)
    bag_sizes = Counter(g for g, lbl in zip(group_rows, y_rows, strict=True) if lbl == 0.0)
    good_w = (n_bad_bags / n_good) if n_good else 1.0
    w = np.ones(len(y_rows), dtype=np.float32)
    for i, (g, lbl) in enumerate(zip(group_rows, y_rows, strict=True)):
        w[i] = good_w if lbl == 1.0 else (1.0 / bag_sizes[g])
    return w


def _pooled_group_scores(
    model: Any,
    cal_groups: list,
    rows_by_group: dict,
    X_np: np.ndarray,
    score_rows_by_group: dict | None,
) -> list[float]:
    """Collapse each calibration group to one max-pooled sigmoid score.

    With *score_rows_by_group* each group pools over the rows the scorer will
    max-pool at **inference**; otherwise it pools over the rows it trained on
    (the historical behaviour every production caller takes).
    """
    import torch  # noqa: PLC0415

    device = next(model.parameters()).device
    if score_rows_by_group is not None:
        blocks = [np.asarray(score_rows_by_group[g], dtype=np.float32) for g in cal_groups]
        sizes = [b.shape[0] for b in blocks]
        with torch.no_grad():
            X_cal = torch.tensor(np.concatenate(blocks, axis=0), dtype=torch.float32).to(device)
            flat = sigmoid_to_finite_scores(model(X_cal))
        out: list[float] = []
        offset = 0
        for size in sizes:
            out.append(max(flat[offset : offset + size]))
            offset += size
        return out

    cal_idx = [i for g in cal_groups for i in rows_by_group[g]]
    with torch.no_grad():
        X_cal = torch.tensor(X_np[cal_idx], dtype=torch.float32).to(device)
        row_scores = sigmoid_to_finite_scores(model(X_cal))
    by_row = dict(zip(cal_idx, row_scores, strict=True))
    return [max(by_row[i] for i in rows_by_group[g]) for g in cal_groups]


def _group_node_blocks(
    model: Any,
    cal_groups: list,
    rows_by_group: dict,
    X_np: np.ndarray,
    score_rows_by_group: dict | None,
) -> list[np.ndarray]:
    """Per calibration group, the array of that group's per-node sigmoid scores.

    The un-pooled counterpart of :func:`_pooled_group_scores`: it returns each
    group's full node-score vector rather than its max, so a caller can re-pool
    the bag under an alternative rule (top-k, extreme-value) while reusing the
    exact fold model and node scores.  ``max`` over each returned block
    reproduces :func:`_pooled_group_scores` value-for-value.
    """
    import torch  # noqa: PLC0415

    device = next(model.parameters()).device
    if score_rows_by_group is not None:
        blocks = [np.asarray(score_rows_by_group[g], dtype=np.float32) for g in cal_groups]
        sizes = [b.shape[0] for b in blocks]
        with torch.no_grad():
            X_cal = torch.tensor(np.concatenate(blocks, axis=0), dtype=torch.float32).to(device)
            flat = sigmoid_to_finite_scores(model(X_cal))
        out: list[np.ndarray] = []
        offset = 0
        for size in sizes:
            out.append(np.asarray(flat[offset : offset + size], dtype=np.float64))
            offset += size
        return out

    cal_idx = [i for g in cal_groups for i in rows_by_group[g]]
    with torch.no_grad():
        X_cal = torch.tensor(X_np[cal_idx], dtype=torch.float32).to(device)
        row_scores = sigmoid_to_finite_scores(model(X_cal))
    by_row = dict(zip(cal_idx, row_scores, strict=True))
    return [np.asarray([by_row[i] for i in rows_by_group[g]], dtype=np.float64) for g in cal_groups]


def nearest_holdout(class_total: int, calibration_fraction: float) -> int:
    """How many of a class of *class_total* votes each calibration fold holds out.

    ``calibration_fraction`` of them rounded to the nearest whole vote, halves
    up, and kept between 1 and ``class_total - 1`` so both sides of the split
    keep a vote of the class.  A class of fewer than two votes has no split and
    holds out none; the fold code refuses it before it trains.

    No coin (issue #4864).  The held-out count is a function of the class's own
    count alone, and never falls as the class grows.  The labels line stands
    once the folds hold out ``HEAD_COUNT_GOODS`` Goods between them
    (:func:`~vtscore.training.thresholds.labels_line.head_count_serves`), so it
    takes over at one count of Goods and stays: 9 Goods on a single-vector space
    (0.3 held out), 5 on a patch space (0.5), at the app's two folds.  The rule
    before it rounded stochastically, which made that hand-over a labelset's
    coin flip over a band of Goods counts, drawn again at every vote.
    """
    if class_total < 2:
        return 0
    held = math.floor(class_total * calibration_fraction + 0.5)
    return max(1, min(class_total - 1, held))


def _split_train_counts(
    n: int, n_pos: int, n_neg: int, calibration_fraction: float, labelset: tuple[Any, Any] | None
) -> tuple[int, int] | None:
    """How many Goods and Bads train each fold; ``None`` when the split leaves too few to train.

    *n* counts the votes (rows, or bags on the grouped path) and *n_pos* /
    *n_neg* the two classes.  Each class holds out :func:`nearest_holdout` of
    its own votes, and every fold holds out the same counts.  A labelset whose
    calibration share, rounded the same way, would leave fewer than two votes
    to train is too small to split at all (a degenerate fraction), and gets the
    caller's sentinel.

    The cost, measured (#4864): sizes that step on a fixed cadence in a class's
    count step together across the eval's runs where one class's count tracks
    the vote count, which is #3286's setup.  On a single-vector space the Bads'
    holdout does share a cadence across runs, at an eighth of ``round()``'s
    over the whole labelset, and it does not reach the eval's curves.

    *labelset* (the training vectors and their labels) is not read by this
    rule.  It is passed so a harness arm can seed a rule from it: the one
    place the split's sizes are decided, for the row-wise and grouped folds
    alike, so ``vtscore.eval.arms_split_rounding`` can swap in the dithered
    rule this replaced for a run.
    """
    del labelset
    if n - max(1, math.floor(n * calibration_fraction + 0.5)) < 2:
        return None
    return n_pos - nearest_holdout(n_pos, calibration_fraction), n_neg - nearest_holdout(n_neg, calibration_fraction)


def _grouped_folds(
    X_list: list[np.ndarray],
    y_list: list[float],
    input_dim: int,
    groups: list,
    rng: np.random.RandomState | None,
    calibrate_count: int,
    calibration_fraction: float,
    hidden_dim: int | None,
    seconds_sink: list[float] | None = None,
) -> tuple[list[tuple[Any, list]], float | None, np.ndarray, dict, dict]:
    """Train the bag-aware calibration folds; return the trained fold models.

    The shared core of :func:`_compute_fold_orderings_grouped` and
    :func:`compute_grouped_fold_node_scores`: both need identical fold splits and
    fold models, differing only in how they collapse each calibration group
    (max-pool vs. keep every node).  Returns
    ``(folds, fallback, X_np, rows_by_group, label_by_group)`` where *folds* is a
    list of ``(model, cal_groups)`` and *fallback* is a sentinel threshold when
    calibration is impossible (empty *folds* then).

    *seconds_sink*, when given, receives each fold's split-and-fit wall clock in
    fold order — the per-fold marginal cost of ``calibrate_count`` (issue #2897).
    """
    import torch  # noqa: PLC0415

    from vtscore.training.mlp import train_model  # noqa: PLC0415

    _rng = rng if rng is not None else np.random.RandomState(CALIBRATION_SPLIT_SEED)
    X_np = np.array(X_list)
    y_np = np.array(y_list)
    grp = list(groups)

    # Rows per group, and each group's (single) label.
    order_groups: list = []
    rows_by_group: dict = {}
    label_by_group: dict = {}
    for i, g in enumerate(grp):
        if g not in rows_by_group:
            rows_by_group[g] = []
            order_groups.append(g)
            label_by_group[g] = y_np[i]
        rows_by_group[g].append(i)

    pos_groups = [g for g in order_groups if label_by_group[g] == 1.0]
    neg_groups = [g for g in order_groups if label_by_group[g] == 0.0]
    n = len(order_groups)
    if n < 4:
        return [], 0.5, X_np, rows_by_group, label_by_group
    if len(pos_groups) < 2 or len(neg_groups) < 2:
        return [], 0.5, X_np, rows_by_group, label_by_group

    # Each class holds out its nearest whole share of the calibration fraction (#4864).
    sizes = _split_train_counts(n, len(pos_groups), len(neg_groups), calibration_fraction, (X_np, y_np))
    if sizes is None:
        return [], NO_GOOD_THRESHOLD, X_np, rows_by_group, label_by_group
    n_train_pos, n_train_neg = sizes

    # Index the plain group lists by position - group ids are tuples, and
    # ``np.array(list_of_tuples)`` would build a 2-D array and mangle them.
    folds: list[tuple[Any, list]] = []
    for _ in range(max(1, calibrate_count)):
        t_fold = time.monotonic()
        pos_perm = _rng.permutation(len(pos_groups))
        neg_perm = _rng.permutation(len(neg_groups))
        train_groups = [pos_groups[i] for i in pos_perm[:n_train_pos]] + [neg_groups[i] for i in neg_perm[:n_train_neg]]
        cal_groups = [pos_groups[i] for i in pos_perm[n_train_pos:]] + [neg_groups[i] for i in neg_perm[n_train_neg:]]

        train_idx = [i for g in train_groups for i in rows_by_group[g]]
        X_train = torch.tensor(X_np[train_idx], dtype=torch.float32)
        y_train = torch.tensor(y_np[train_idx], dtype=torch.float32).unsqueeze(1)
        fold_w = torch.tensor(_per_bag_fit_weights(y_np[train_idx], [grp[i] for i in train_idx]), dtype=torch.float32)
        model = train_model(X_train, y_train, input_dim, hidden_dim=hidden_dim, sample_weights=fold_w)
        folds.append((model, cal_groups))
        if seconds_sink is not None:
            seconds_sink.append(time.monotonic() - t_fold)

    return folds, None, X_np, rows_by_group, label_by_group


def _compute_fold_orderings_grouped(
    X_list: list[np.ndarray],
    y_list: list[float],
    input_dim: int,
    groups: list,
    rng: np.random.RandomState | None,
    calibrate_count: int,
    calibration_fraction: float,
    hidden_dim: int | None,
    score_rows_by_group: dict | None = None,
    model_sink: list | None = None,
    seconds_sink: list[float] | None = None,
    holdout_sink: list | None = None,
) -> tuple[list[tuple[list[float], list[float]]], float | None]:
    """Bag-aware variant of :func:`compute_fold_orderings`.

    Splits by *group* (a voted image) instead of by row so a Bad bag's flooded
    region negatives never straddle the Train/Calibrate boundary, sizes the
    split over votes not rows, weight-balances each fold fit per-bag, and
    collapses every calibration group to a single max-pooled score (an image
    scores by its best region, as at inference).

    *score_rows_by_group* overrides which rows a calibration group collapses
    over - see :func:`compute_fold_orderings`.  *holdout_sink* receives each
    fold's held-out bags as the first training row of each, in ordering order.
    """
    folds, fallback, X_np, rows_by_group, label_by_group = _grouped_folds(
        X_list, y_list, input_dim, groups, rng, calibrate_count, calibration_fraction, hidden_dim, seconds_sink
    )
    if fallback is not None:
        return [], fallback
    if model_sink is not None:
        model_sink.extend(model for model, _cal in folds)
    if holdout_sink is not None:
        holdout_sink.extend([rows_by_group[g][0] for g in cal] for _model, cal in folds)

    orderings: list[tuple[list[float], list[float]]] = []
    for model, cal_groups in folds:
        # Collapse each calibration group to one max-pooled score, so a Good
        # bag and a Bad bag are pooled the same way the scorer pools an image.
        group_scores = _pooled_group_scores(model, cal_groups, rows_by_group, X_np, score_rows_by_group)
        group_labels = [float(label_by_group[g]) for g in cal_groups]
        orderings.append((group_scores, group_labels))

    return orderings, None


def compute_grouped_fold_node_scores(
    X_list: list[np.ndarray],
    y_list: list[float],
    input_dim: int,
    groups: list,
    rng: np.random.RandomState | None = None,
    calibrate_count: int = 2,
    calibration_fraction: float = 0.5,
    hidden_dim: int | None = None,
    score_rows_by_group: dict | None = None,
    model_sink: list | None = None,
    seconds_sink: list[float] | None = None,
    holdout_sink: list | None = None,
) -> tuple[list[tuple[list[np.ndarray], list[float]]], float | None]:
    """Bag-aware calibration folds, returning each held-out group's node scores.

    Like :func:`_compute_fold_orderings_grouped` but instead of max-pooling every
    calibration group it returns the group's **full node-score vector**, so a
    caller (the #2781 calibration study) can re-pool the same fold models' scores
    under alternative rules (top-k mean, extreme-value ``pnorm``) to recalibrate
    a threshold for a pooling variant without retraining.  ``max`` over each
    returned block reproduces this arm's production threshold exactly.

    Returns ``(fold_node_data, fallback)`` where *fold_node_data* is a list, one
    entry per fold, of ``(group_node_scores, group_labels)`` - *group_node_scores*
    being a list of 1-D float arrays (one per held-out calibration group).

    *model_sink*, when given, receives each trained fold model in fold order -
    the #2852 fold-anchored eval arm scores the haystack with the same fold
    models the orderings came from, so the anchors and the population it fits
    share one score scale without a retrain.  *holdout_sink* receives each
    fold's held-out bags as :func:`_compute_fold_orderings_grouped` gives them.
    """
    folds, fallback, X_np, rows_by_group, label_by_group = _grouped_folds(
        X_list, y_list, input_dim, groups, rng, calibrate_count, calibration_fraction, hidden_dim, seconds_sink
    )
    if fallback is not None:
        return [], fallback
    if model_sink is not None:
        model_sink.extend(model for model, _cal in folds)
    if holdout_sink is not None:
        holdout_sink.extend([rows_by_group[g][0] for g in cal] for _model, cal in folds)

    fold_node_data: list[tuple[list[np.ndarray], list[float]]] = []
    for model, cal_groups in folds:
        blocks = _group_node_blocks(model, cal_groups, rows_by_group, X_np, score_rows_by_group)
        group_labels = [float(label_by_group[g]) for g in cal_groups]
        fold_node_data.append((blocks, group_labels))
    return fold_node_data, None


def _torch_fold_ordering(
    X_np: np.ndarray,
    y_np: np.ndarray,
    train_idx: np.ndarray,
    cal_idx: np.ndarray,
    input_dim: int,
    hidden_dim: int | None,
    model_sink: list | None,
) -> tuple[list[float], list[float]]:
    """One fold of :func:`compute_fold_orderings` under the app's own head."""
    import torch  # noqa: PLC0415

    from vtscore.training.mlp import train_model  # noqa: PLC0415

    X_train = torch.tensor(X_np[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y_np[train_idx], dtype=torch.float32).unsqueeze(1)
    X_cal = torch.tensor(X_np[cal_idx], dtype=torch.float32)

    model = train_model(X_train, y_train, input_dim, hidden_dim=hidden_dim)
    if model_sink is not None:
        model_sink.append(model)

    with torch.no_grad():
        X_cal = X_cal.to(next(model.parameters()).device)
        # Sanitize non-finite sigmoids (destabilised fold model): the
        # orderings are cached, swept for the Stats chart, and averaged
        # into ``DetectorContext.threshold`` - a NaN here would silently
        # break every downstream ``score >= threshold`` comparison and
        # leak NaN into JSON responses.
        scores = sigmoid_to_finite_scores(model(X_cal))
    return scores, y_np[cal_idx].tolist()


def _fold_fit_ordering(
    fold_fit: "Callable[[np.ndarray, np.ndarray], Callable[[np.ndarray], np.ndarray]]",
    X_np: np.ndarray,
    y_np: np.ndarray,
    train_idx: np.ndarray,
    cal_idx: np.ndarray,
    model_sink: list | None,
) -> tuple[list[float], list[float]]:
    """One fold of :func:`compute_fold_orderings` under a caller-supplied estimator.

    Fits on the fold's training rows, scores its held-out rows, and replaces a
    non-finite score with :data:`~vtscore.training.thresholds.knobs.NON_FINITE_SCORE_SENTINEL`
    as the torch path does, so a degenerate fit cannot leak NaN into the cut.
    """

    predict = fold_fit(X_np[train_idx], y_np[train_idx])
    if model_sink is not None:
        model_sink.append(predict)
    raw = np.asarray(predict(X_np[cal_idx]), dtype=np.float64).ravel()
    s = NON_FINITE_SCORE_SENTINEL
    return np.nan_to_num(raw, nan=s, posinf=s, neginf=s).tolist(), y_np[cal_idx].tolist()


def _row_wise_fold_ordering(
    fold_fit: "Callable[[np.ndarray, np.ndarray], Callable[[np.ndarray], np.ndarray]] | None",
    X_np: np.ndarray,
    y_np: np.ndarray,
    train_idx: np.ndarray,
    cal_idx: np.ndarray,
    input_dim: int,
    hidden_dim: int | None,
    model_sink: list | None,
) -> tuple[list[float], list[float]]:
    """One row-wise fold of :func:`compute_fold_orderings`: *fold_fit*'s estimator, or the app's head without one."""
    if fold_fit is not None:
        return _fold_fit_ordering(fold_fit, X_np, y_np, train_idx, cal_idx, model_sink)
    return _torch_fold_ordering(X_np, y_np, train_idx, cal_idx, input_dim, hidden_dim, model_sink)


def compute_fold_orderings(
    X_list: list[np.ndarray],
    y_list: list[float],
    input_dim: int,
    rng: np.random.RandomState | None = None,
    calibrate_count: int = 2,
    calibration_fraction: float = 0.5,
    hidden_dim: int | None = None,
    groups: list | None = None,
    score_rows_by_group: dict | None = None,
    model_sink: list | None = None,
    seconds_sink: list[float] | None = None,
    fold_fit: "Callable[[np.ndarray, np.ndarray], Callable[[np.ndarray], np.ndarray]] | None" = None,
    holdout_sink: list | None = None,
) -> tuple[list[tuple[list[float], list[float]]], float | None]:
    """Train the K calibration folds and return their held-out orderings.

    Each ordering is a ``(cal_scores, cal_labels)`` pair: the fold model's
    sigmoid scores on its held-out calibration split, and that split's true
    labels.  Because :func:`train_model` is inclusion-independent, these
    orderings do **not** depend on ``inclusion`` - so they can be cached once
    and re-thresholded at any inclusion via
    :func:`~vtscore.training.thresholds.conformal.threshold_from_fold_orderings`
    (and swept across all inclusions for the Stats chart).

    Returns ``(orderings, fallback)``.  When calibration is not possible the
    orderings are empty and ``fallback`` is the sentinel threshold the public
    wrapper must return (mirrors
    :func:`~vtscore.training.thresholds.conformal.calculate_cross_calibration_threshold`'s
    historical early-returns); otherwise ``fallback`` is ``None``.

    *groups* activates the **bag-aware** path used when Bad votes are flooded
    into their region nodes: rows sharing a ``groups`` id belong to one voted
    image (a Bad bag's region negatives, or a single Good row) and
    are kept together on one side of every Train/Calibrate split, split counts
    are taken over *groups* (votes) not rows, fold fits are *weights*-balanced,
    and each calibration group collapses to one max-pooled score - matching how
    inference scores an image by its best region.  When *groups* is ``None``
    (every non-flooded caller) the historical row-wise path runs unchanged.

    *score_rows_by_group* (grouped path only) maps each group id to the row
    stack that group should be **scored** over, decoupling "what the fold model
    trains on" from "what a calibration bag collapses to".  It exists because
    the two are not the same whenever a Good vote contributes fewer rows than a
    Bad vote floods: the Good bag then collapses to a max over 1 row while the
    Bad bag - and every image at inference - collapses to a max over N, and
    ``max`` is an upward-biased order statistic, so the calibrated cut
    lands systematically high and the threshold over-rejects positives.  Passing
    each vote's inference rows here puts both classes in the geometry and at the
    width the scorer will actually use.  The production vote / labelset paths
    supply each voted image's full region-node stack
    (:func:`vtscore.detectors.training.inference_score_rows`); ``None`` keeps
    the "collapse over the training rows" behaviour for callers that have no
    inference geometry to offer.

    *model_sink*, when given, receives each trained fold model in fold order
    (see :func:`compute_grouped_fold_node_scores`); production callers pass
    nothing and the models stay fold-local as before.  *seconds_sink* likewise
    receives each fold's wall clock, which is what makes the *cost* half of the
    fold-count question (issue #2897) measurable without a second run.
    *holdout_sink* likewise receives, per fold, the training row behind each
    held-out score in ordering order (the first row of each bag on the grouped
    path), so a caller can tell which votes a fold held out (the eval
    harness's precision frames name each held-out vote's phase).  Read-only:
    none of these sinks changes a split, a fit or an ordering.

    The folds are **independent repeated splits**, not a partition: every fold
    re-draws a stratified ``calibration_fraction`` holdout from the same labels,
    so raising ``calibrate_count`` averages more draws at a *fixed* per-fold
    calibration size rather than shrinking each holdout.  Two consequences the
    fold-count study rests on: the per-fold work is flat in K (total cost is
    linear in K), and the folds at ``calibrate_count=k`` are exactly the first
    *k* folds at any larger count drawn from the same ``rng`` - the splits are
    nested, so one run at Kmax yields every smaller K's calibration for free.

    *fold_fit* (row-wise path only; eval harness only, issue #3959) replaces the
    torch head with any estimator: ``fold_fit(X_train, y_train)`` returns a
    ``predict(X) -> P(positive)`` callable, which scores the fold's held-out
    split and is what *model_sink* receives.  Everything else - the split
    sizes, the stratified draws, the *rng* stream - is this function's,
    so a non-torch head calibrates on exactly the splits the app's head would
    have drawn from the same labels.  Production callers never pass it.
    """
    if groups is not None and fold_fit is not None:
        raise ValueError("fold_fit applies to the row-wise path only")
    if groups is not None:
        return _compute_fold_orderings_grouped(
            X_list,
            y_list,
            input_dim,
            groups,
            rng=rng,
            calibrate_count=calibrate_count,
            calibration_fraction=calibration_fraction,
            hidden_dim=hidden_dim,
            score_rows_by_group=score_rows_by_group,
            model_sink=model_sink,
            seconds_sink=seconds_sink,
            holdout_sink=holdout_sink,
        )
    n = len(X_list)
    if n < 4:
        return [], 0.5

    _rng = rng if rng is not None else np.random.RandomState(CALIBRATION_SPLIT_SEED)
    X_np = np.array(X_list)
    y_np = np.array(y_list)

    pos_idx = np.where(y_np == 1.0)[0]
    neg_idx = np.where(y_np == 0.0)[0]
    # Each class holds out its nearest whole share of the calibration fraction
    # (#4864).  Sized before the class check, so a split too small to train
    # still returns its sentinel.
    sizes = _split_train_counts(n, len(pos_idx), len(neg_idx), calibration_fraction, (X_np, y_np))
    if sizes is None:
        return [], NO_GOOD_THRESHOLD
    if len(pos_idx) < 2 or len(neg_idx) < 2:
        return [], 0.5

    calibrate_count = max(1, calibrate_count)
    n_train_pos, n_train_neg = sizes

    orderings: list[tuple[list[float], list[float]]] = []
    for _ in range(calibrate_count):
        t_fold = time.monotonic()
        pos_perm = _rng.permutation(pos_idx)
        neg_perm = _rng.permutation(neg_idx)
        train_idx = np.concatenate([pos_perm[:n_train_pos], neg_perm[:n_train_neg]])
        cal_idx = np.concatenate([pos_perm[n_train_pos:], neg_perm[n_train_neg:]])

        orderings.append(
            _row_wise_fold_ordering(fold_fit, X_np, y_np, train_idx, cal_idx, input_dim, hidden_dim, model_sink)
        )
        if holdout_sink is not None:
            holdout_sink.append([int(i) for i in cal_idx])
        if seconds_sink is not None:
            seconds_sink.append(time.monotonic() - t_fold)

    return orderings, None
