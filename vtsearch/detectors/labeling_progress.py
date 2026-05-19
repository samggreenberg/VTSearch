"""Labeling-session analyzer: per-step model cache and stopping-condition metrics.

Caches trained MLPs and stability metrics per labelling step so that
repeated queries (the progress button, the auto-indicator) never retrain
models that have already been computed.

Unrelated to :mod:`vtsearch.concurrency.progress`, which is the
infrastructure for tracking and cancelling long-running operations
(``ProgressTracker`` and the dataset/sort/eval/find singletons).  The
two modules used to share the ``progress.py`` name; this one was
renamed to make the distinction obvious.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from vtsearch.training.mlp import train_model
from vtsearch.training.thresholds import find_optimal_threshold

if TYPE_CHECKING:
    import torch.nn as nn

# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------
# Each entry in ``_cached_steps`` corresponds to one index in ``label_history``
# and stores the model, threshold, label sets, and stability result for that
# step.  ``_cache_good_ids`` / ``_cache_bad_ids`` track the running label sets
# so the next step only needs to apply a single delta.

_cache_inclusion: Optional[int] = None
_cached_steps: list[dict[str, Any]] = []
_cache_good_ids: set[int] = set()
_cache_bad_ids: set[int] = set()
_cache_prev_predictions: Optional[dict[int, int]] = None
_cache_diversity_tree: Any = None  # DiversityTree | None

# Live models injected by `train_and_score` during sorting.  Keyed by
# ``(frozenset(good_ids), frozenset(bad_ids))`` so that ``_ensure_cache``
# can look up the actual model that was used at each label step instead
# of retraining from scratch.
_live_models: dict[tuple[frozenset[int], frozenset[int]], tuple[Any, float]] = {}

# Reentrant lock protecting all module-level cache variables.
# RLock is used because public functions call _ensure_cache which may
# call clear_progress_cache internally when the inclusion value changes.
_progress_lock = threading.RLock()


def clear_progress_cache() -> None:
    """Clear all cached progress data.

    Must be called whenever votes are cleared, medias change, or inclusion
    is altered so that stale models are not reused.
    """
    global _cache_inclusion, _cache_prev_predictions, _cache_diversity_tree
    with _progress_lock:
        _cached_steps.clear()
        _cache_good_ids.clear()
        _cache_bad_ids.clear()
        _cache_prev_predictions = None
        _cache_inclusion = None
        _cache_diversity_tree = None
        _live_models.clear()


def invalidate_progress_cache_from(media_id: int) -> None:
    """Truncate the progress cache to just before *media_id* first appeared.

    Called when a vote switches polarity (good→bad or bad→good).  Steps
    before the media was first labeled are still valid — their models never
    included this media in training data.  Only steps from the first
    appearance onward are discarded so they can be retrained and their
    stability/evaluation metrics recomputed.
    """
    global _cache_prev_predictions, _cache_diversity_tree
    with _progress_lock:
        # Find the first cached step that includes media_id in its training data.
        truncate_at = None
        for i, step in enumerate(_cached_steps):
            if media_id in step["good_ids"] or media_id in step["bad_ids"]:
                truncate_at = i
                break

        if truncate_at is None:
            # Media never appeared in any cached step.  Still need to clear
            # live models — they may have been injected by learned-sort
            # without building the progress cache.
            _live_models.clear()
            return

        if truncate_at == 0:
            # Media was present from the very first step — full clear.
            clear_progress_cache()
            return

        # Keep steps [0, truncate_at); discard the rest.
        del _cached_steps[truncate_at:]

        # Restore running ID sets from the last kept step.
        last = _cached_steps[-1]
        _cache_good_ids.clear()
        _cache_good_ids.update(last["good_ids"])
        _cache_bad_ids.clear()
        _cache_bad_ids.update(last["bad_ids"])

        # Reset the stability prediction chain — it will restart from the
        # truncation point when _ensure_cache replays the remaining history.
        _cache_prev_predictions = None

        # Rebuild the diversity tree on next _ensure_cache call so its label
        # state is re-synced from the truncation point forward.
        _cache_diversity_tree = None

        # Clear live models — some may have been trained with the old label.
        _live_models.clear()


def inject_live_model(
    good_votes: dict[int, None],
    bad_votes: dict[int, None],
    model: nn.Sequential,
    threshold: float,
) -> None:
    """Register a live model from ``train_and_score`` for progress-cache reuse.

    Called by the learned-sort route after each live training run.  The model
    is stored keyed by its label set so ``_ensure_cache`` can look it up
    instead of retraining from scratch.
    """
    key = (frozenset(good_votes), frozenset(bad_votes))
    with _progress_lock:
        _live_models[key] = (model, threshold)


def _build_diversity_tree(clips_dict: dict[int, dict[str, Any]]) -> Any:
    """Build a DiversityTree from clip embeddings, or ``None`` if no embeddings."""
    vectors: dict[int, np.ndarray] = {
        cid: np.asarray(media["embedding"], dtype=np.float32)
        for cid, media in clips_dict.items()
        if media.get("embedding") is not None
    }
    if not vectors:
        return None
    from vtsearch.state.diversity_tree import DiversityTree  # noqa: PLC0415

    return DiversityTree(vectors, k=3)


def _apply_label_event(media_id: int, label: str) -> bool:
    """Update ``_cache_good_ids`` / ``_cache_bad_ids`` for one label event.

    Returns ``True`` if *media_id* was already labeled before this event.
    """
    was_labeled = media_id in _cache_good_ids or media_id in _cache_bad_ids
    if label == "unlabel":
        _cache_good_ids.discard(media_id)
        _cache_bad_ids.discard(media_id)
    elif label == "good":
        _cache_bad_ids.discard(media_id)
        _cache_good_ids.add(media_id)
    else:
        _cache_good_ids.discard(media_id)
        _cache_bad_ids.add(media_id)
    return was_labeled


def _sync_diversity_tree(media_id: int, label: str, was_labeled: bool) -> Optional[dict[str, Any]]:
    """Mirror a label event onto the diversity tree and return level info."""
    if _cache_diversity_tree is None:
        return None
    if label == "unlabel":
        # Only unlabel on the tree when the item is no longer labeled at all
        # (guards against good→bad re-labels going through "unlabel").
        if was_labeled and media_id not in _cache_good_ids and media_id not in _cache_bad_ids:
            if media_id in _cache_diversity_tree.vector_to_leaf:
                _cache_diversity_tree.unlabel(media_id)
    else:
        if media_id in _cache_diversity_tree.vector_to_leaf:
            _cache_diversity_tree.label(media_id)
    return {
        "num_labels": len(_cache_good_ids) + len(_cache_bad_ids),
        "diversity_level": _cache_diversity_tree.diversity_level(),
        "depth": _cache_diversity_tree.total_nodes,
    }


def _collect_training_data(
    clips_dict: dict[int, dict[str, Any]],
) -> tuple[list[np.ndarray], list[float]]:
    """Gather embeddings and labels from the current good/bad ID sets."""
    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    for cid in _cache_good_ids:
        if cid in clips_dict and clips_dict[cid].get("embedding") is not None:
            X_list.append(clips_dict[cid]["embedding"])
            y_list.append(1.0)
    for cid in _cache_bad_ids:
        if cid in clips_dict and clips_dict[cid].get("embedding") is not None:
            X_list.append(clips_dict[cid]["embedding"])
            y_list.append(0.0)
    return X_list, y_list


def _compute_step_stability(
    model: nn.Sequential,
    threshold: float,
    clips_dict: dict[int, dict[str, Any]],
    all_media_ids: list[int],
    t: int,
    num_labels: int,
) -> Optional[dict[str, Any]]:
    """Compute prediction stability by comparing to the previous step's predictions."""
    global _cache_prev_predictions
    import torch  # noqa: PLC0415

    labeled_ids = _cache_good_ids | _cache_bad_ids
    unlabeled_ids = [
        cid for cid in all_media_ids if cid not in labeled_ids and clips_dict.get(cid, {}).get("embedding") is not None
    ]

    if not unlabeled_ids:
        return {"time_index": t, "num_labels": num_labels, "num_flips": 0, "num_unlabeled": 0}

    unlabeled_embs = np.array([clips_dict[cid]["embedding"] for cid in unlabeled_ids])
    X_unlabeled = torch.tensor(unlabeled_embs, dtype=torch.float32)

    with torch.no_grad():
        X_unlabeled = X_unlabeled.to(next(model.parameters()).device)
        scores_unl = torch.sigmoid(model(X_unlabeled)).squeeze(1).cpu().tolist()

    predictions: dict[int, int] = {cid: 1 if score >= threshold else 0 for cid, score in zip(unlabeled_ids, scores_unl)}

    stability: Optional[dict[str, Any]] = None
    if _cache_prev_predictions is not None:
        num_flips = sum(
            1
            for cid in predictions.keys() & _cache_prev_predictions.keys()
            if predictions[cid] != _cache_prev_predictions[cid]
        )
        stability = {
            "time_index": t,
            "num_labels": num_labels,
            "num_flips": num_flips,
            "num_unlabeled": len(unlabeled_ids),
        }
    # else: no prior predictions to compare — leave stability as None.

    _cache_prev_predictions = predictions
    return stability


def _train_step(
    clips_dict: dict[int, dict[str, Any]],
    all_media_ids: list[int],
    t: int,
    num_labels: int,
    inclusion_value: int,
) -> tuple[Optional[nn.Sequential], Optional[float], Optional[dict[str, Any]]]:
    """Train a model for one cache step and compute stability.

    Returns ``(model, threshold, stability)``.  All three are ``None`` when
    training is not possible (e.g. only one label polarity present).
    """
    global _cache_prev_predictions

    if not _cache_good_ids or not _cache_bad_ids:
        # No model possible — clear prediction baseline so the first step
        # after regaining a model doesn't produce a misleading flip count.
        _cache_prev_predictions = None
        return None, None, None

    X_list, y_list = _collect_training_data(clips_dict)
    if len(X_list) < 2:
        return None, None, None

    import torch  # noqa: PLC0415

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)

    model = train_model(X, y, X.shape[1], inclusion_value)

    with torch.no_grad():
        X_dev = X.to(next(model.parameters()).device)
        scores = torch.sigmoid(model(X_dev)).squeeze(1).cpu().tolist()
    threshold = find_optimal_threshold(scores, y_list, inclusion_value)

    stability = _compute_step_stability(model, threshold, clips_dict, all_media_ids, t, num_labels)
    return model, threshold, stability


def _ensure_cache(  # noqa: C901
    clips_dict: dict[int, dict[str, Any]],
    label_history: list[tuple[int, str, float]],
    inclusion_value: int,
) -> None:
    """Bring the cache up to date with *label_history*.

    Only computes steps that are not yet cached.  If *inclusion_value*
    differs from the value used for existing cache entries the entire cache
    is rebuilt.
    """
    global _cache_inclusion, _cache_diversity_tree

    if _cache_inclusion is not None and _cache_inclusion != inclusion_value:
        clear_progress_cache()

    if _cache_inclusion is None:
        _cache_inclusion = inclusion_value

    start = len(_cached_steps)
    if start >= len(label_history):
        return  # already up to date

    all_media_ids = sorted(clips_dict.keys())

    if _cache_diversity_tree is None:
        _cache_diversity_tree = _build_diversity_tree(clips_dict)
        # After a partial invalidation (_cache_diversity_tree was set to None
        # but _cache_good_ids/_cache_bad_ids still contain IDs from kept
        # steps), seed the fresh tree with those pre-existing labels so that
        # diversity_level() is correct for subsequently replayed events.
        if _cache_diversity_tree is not None:
            for mid in _cache_good_ids | _cache_bad_ids:
                if mid in _cache_diversity_tree.vector_to_leaf:
                    _cache_diversity_tree.label(mid)

    for t in range(start, len(label_history)):
        media_id, label, _ = label_history[t]

        was_labeled = _apply_label_event(media_id, label)
        diversity_info = _sync_diversity_tree(media_id, label, was_labeled)

        good_ids = list(_cache_good_ids)
        bad_ids = list(_cache_bad_ids)
        num_labels = len(good_ids) + len(bad_ids)

        # Check whether the training data actually changed compared to the
        # previous step.  If the good/bad ID sets are identical, the model
        # would be the same — skip training and stability recording so the
        # line graph and Stable indicator only reflect genuine model updates.
        prev = _cached_steps[-1] if _cached_steps else None
        training_data_changed = (
            prev is None or set(good_ids) != set(prev["good_ids"]) or set(bad_ids) != set(prev["bad_ids"])
        )

        if training_data_changed:
            # Check whether train_and_score already produced a model for
            # this exact label set during live sorting.  If so, reuse it
            # (correct cross-calibrated threshold, zero compute cost).
            live_key = (frozenset(_cache_good_ids), frozenset(_cache_bad_ids))
            live = _live_models.get(live_key)
            if live is not None:
                model, threshold = live
                stability = _compute_step_stability(model, threshold, clips_dict, all_media_ids, t, num_labels)
            else:
                model, threshold, stability = _train_step(clips_dict, all_media_ids, t, num_labels, inclusion_value)
        else:
            # Reuse previous model — no new training or stability entry.
            model = prev["model"] if prev else None
            threshold = prev["threshold"] if prev else None
            stability = None

        _cached_steps.append(
            {
                "model": model,
                "threshold": threshold,
                "good_ids": good_ids,
                "bad_ids": bad_ids,
                "stability": stability,
                "diversity": diversity_info,
            }
        )


# ---------------------------------------------------------------------------
# Helper: evaluate cached models against a label set
# ---------------------------------------------------------------------------


def _eval_cached_models(  # noqa: C901
    clips_dict: dict[int, dict[str, Any]],
    current_good_votes: dict[int, None],
    current_bad_votes: dict[int, None],
    inclusion_value: int,
    start: int = 0,
    end: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Score cached models against the current labelset (forward passes only).

    Returns a list of error-cost dicts for every cached step in
    ``[start, end)`` that has a trained model.
    """
    if inclusion_value >= 0:
        fpr_weight = 1.0
        fnr_weight = 2.0**inclusion_value
    else:
        fpr_weight = 2.0 ** (-inclusion_value)
        fnr_weight = 1.0

    # Build evaluation set from current votes
    current_labels: dict[int, float] = {}
    for cid in current_good_votes:
        current_labels[cid] = 1.0
    for cid in current_bad_votes:
        current_labels[cid] = 0.0

    if not current_labels:
        return []

    eval_embs: list[np.ndarray] = []
    eval_labels: list[float] = []
    for cid, lbl in current_labels.items():
        if cid in clips_dict and clips_dict[cid].get("embedding") is not None:
            eval_embs.append(clips_dict[cid]["embedding"])
            eval_labels.append(lbl)

    if not eval_embs:
        return []

    import torch  # noqa: PLC0415

    X_eval = torch.tensor(np.array(eval_embs), dtype=torch.float32)
    total_positives = sum(1 for lbl in eval_labels if lbl == 1)
    total_negatives = len(eval_labels) - total_positives

    if end is None:
        end = len(_cached_steps)

    results: list[dict[str, Any]] = []
    for t in range(start, end):
        step = _cached_steps[t]
        if step["model"] is None:
            continue

        with torch.no_grad():
            X_in = X_eval.to(next(step["model"].parameters()).device)
            scores = torch.sigmoid(step["model"](X_in)).squeeze(1).cpu().tolist()

        fp = fn = 0
        for score, true_label in zip(scores, eval_labels):
            predicted = 1 if score >= step["threshold"] else 0
            if predicted == 1 and true_label == 0:
                fp += 1
            elif predicted == 0 and true_label == 1:
                fn += 1

        fpr = fp / total_negatives if total_negatives > 0 else 0.0
        fnr = fn / total_positives if total_positives > 0 else 0.0
        error_cost = fpr_weight * fpr + fnr_weight * fnr

        results.append(
            {
                "time_index": t,
                "num_labels": len(step["good_ids"]) + len(step["bad_ids"]),
                "error_cost": round(error_cost, 4),
                "fpr": round(fpr, 4),
                "fnr": round(fnr, 4),
            }
        )

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def recreate_model_at_time(
    clips_dict: dict[int, dict[str, Any]],
    label_history: list[tuple[int, str, float]],
    time_index: int,
    inclusion_value: int = 0,
) -> tuple[Optional[nn.Sequential], Optional[float], list[int], list[int]]:
    """Return the cached model for a given labelling step, training it if needed.

    Args:
        clips_dict: Mapping of media ID to media data dict with ``"embedding"``.
        label_history: Ordered labelling events.
        time_index: Index into *label_history*.
        inclusion_value: FPR/FNR trade-off in ``[-10, 10]``.

    Returns:
        ``(model, threshold, good_ids, bad_ids)`` — same contract as before.
    """
    if time_index < 0 or time_index >= len(label_history):
        return None, None, [], []

    with _progress_lock:
        _ensure_cache(clips_dict, label_history, inclusion_value)

        step = _cached_steps[time_index]
        return step["model"], step["threshold"], step["good_ids"], step["bad_ids"]


def calculate_error_cost_over_time(
    clips_dict: dict[int, dict[str, Any]],
    label_history: list[tuple[int, str, float]],
    current_good_votes: dict[int, None],
    current_bad_votes: dict[int, None],
    inclusion_value: int = 0,
) -> list[dict[str, Any]]:
    """Calculate classification error cost at each labelling step.

    Uses cached models — no retraining.
    """
    with _progress_lock:
        _ensure_cache(clips_dict, label_history, inclusion_value)
        return _eval_cached_models(clips_dict, current_good_votes, current_bad_votes, inclusion_value)


def calculate_prediction_stability_over_time(
    clips_dict: dict[int, dict[str, Any]],
    label_history: list[tuple[int, str, float]],
    inclusion_value: int = 0,
) -> list[dict[str, Any]]:
    """Return cached prediction-stability metrics for every step."""
    with _progress_lock:
        _ensure_cache(clips_dict, label_history, inclusion_value)
        return [step["stability"] for step in _cached_steps if step["stability"] is not None]


def _compute_smart_status(
    clips_dict: dict[int, dict[str, Any]],
    label_history: list[tuple[int, str, float]],
    current_good_votes: dict[int, None],
    current_bad_votes: dict[int, None],
    inclusion_value: int,
    good: int,
    bad: int,
    total: int,
) -> dict[str, Any]:
    """Compute Smart (error-cost flatness) red/yellow/green status."""
    if good < 5 or bad < 5:
        return {
            "status": "red",
            "reason": f"Need at least 5 good and 5 bad. Currently {good}g, {bad}b.",
        }

    n = len(_cached_steps)
    if n < 3:
        return {"status": "yellow", "reason": "Not enough label history steps to assess trend."}

    start_idx = max(0, n - 10)
    recent_entries = _eval_cached_models(
        clips_dict, current_good_votes, current_bad_votes, inclusion_value, start_idx, n
    )
    recent_error_costs = [e["error_cost"] for e in recent_entries]

    if len(recent_error_costs) < 3:
        return {"status": "yellow", "reason": "Not enough valid model steps in recent history to assess trend."}

    # Linear regression slope over the recent error-cost values
    n_pts = len(recent_error_costs)
    x_vals = list(range(n_pts))
    x_mean = sum(x_vals) / n_pts
    y_mean = sum(recent_error_costs) / n_pts

    numer = sum((x_vals[i] - x_mean) * (recent_error_costs[i] - y_mean) for i in range(n_pts))
    denom = sum((x_vals[i] - x_mean) ** 2 for i in range(n_pts))
    slope = numer / denom if denom != 0 else 0.0
    relative_slope = slope / y_mean if y_mean > 0 else slope

    FLAT_THRESHOLD = -0.015
    if relative_slope < FLAT_THRESHOLD:
        return {
            "status": "yellow",
            "reason": "Error cost is still declining. Keep labeling.",
            "slope": round(relative_slope, 4),
        }
    return {
        "status": "green",
        "reason": "Error cost has leveled off. You can likely stop labeling.",
        "slope": round(relative_slope, 4),
    }


def _compute_stable_status(
    good: int,
    bad: int,
    total: int,
) -> dict[str, Any]:
    """Compute Stable (prediction-flip) red/yellow/green status."""
    if good < 5 or bad < 5:
        return {
            "status": "red",
            "reason": f"Need at least 5 good and 5 bad. Currently {good}g, {bad}b.",
        }

    stability = [step["stability"] for step in _cached_steps if step["stability"] is not None]

    MIN_STABLE_ENTRIES = 5
    if len(stability) < MIN_STABLE_ENTRIES:
        return {"status": "yellow", "reason": "Not enough history to assess prediction stability."}

    recent = stability[-10:]

    # Use flip *rate* (fraction of unlabeled predictions that changed) so the
    # threshold scales with dataset size instead of using a fixed absolute count.
    flip_rates: list[float] = []
    for s in recent:
        n_unlabeled = s.get("num_unlabeled", 0)
        if n_unlabeled > 0:
            flip_rates.append(s["num_flips"] / n_unlabeled)
        else:
            flip_rates.append(0.0)

    avg_flip_rate = sum(flip_rates) / len(flip_rates)
    max_flip_rate = max(flip_rates)

    STABLE_RATE_THRESHOLD = 0.005  # average less than 0.5% of predictions flipping
    STABLE_MAX_THRESHOLD = 0.01  # no single recent step above 1%

    if avg_flip_rate < STABLE_RATE_THRESHOLD and max_flip_rate < STABLE_MAX_THRESHOLD:
        return {"status": "green", "reason": "Predictions have stabilized.", "avg_flip_rate": round(avg_flip_rate, 4)}
    return {
        "status": "yellow",
        "reason": f"Average {avg_flip_rate:.1%} of predictions flipping in recent steps.",
        "avg_flip_rate": round(avg_flip_rate, 4),
    }


def compute_labeling_status(
    clips_dict: dict[int, dict[str, Any]],
    label_history: list[tuple[int, str, float]],
    current_good_votes: dict[int, None],
    current_bad_votes: dict[int, None],
    inclusion_value: int = 0,
    span_info: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Compute per-metric red/yellow/green labeling statuses.

    Returns a dict with ``good_count``, ``bad_count``, ``total_count``, and
    three sub-dicts: ``smart``, ``stable``, and ``span``, each with a
    ``status`` field of ``"red"``, ``"yellow"``, or ``"green"``.
    """
    good = len(current_good_votes)
    bad = len(current_bad_votes)
    total = good + bad

    with _progress_lock:
        _ensure_cache(clips_dict, label_history, inclusion_value)

        smart = _compute_smart_status(
            clips_dict, label_history, current_good_votes, current_bad_votes, inclusion_value, good, bad, total
        )
        stable = _compute_stable_status(good, bad, total)

    # Span status from diversity tree info (passed in from the route).
    # ``level`` is the number of consecutive BFS-order seen nodes and
    # ``depth`` is the total number of nodes (the maximum diversity level).
    #
    # The old metric required 4 full tree levels for green, which in a k=3
    # tree is 1+3+9+27 = 40 nodes.  We preserve that scale: green at 40
    # nodes (capped at total), yellow at 10, red below 10.
    from vtsearch.config import CoreConfig  # noqa: PLC0415

    SPAN_GREEN = CoreConfig.from_settings().autopilot_goal_diversity
    SPAN_YELLOW = 10
    if span_info is None:
        span = {
            "status": "red",
            "reason": "Diversity tree not available.",
            "level": 0,
            "depth": 0,
        }
    else:
        level = span_info["level"]
        tree_total = span_info["depth"]  # total nodes
        green_at = min(SPAN_GREEN, tree_total)
        yellow_at = min(SPAN_YELLOW, green_at)
        if tree_total <= 0:
            span = {"status": "green", "reason": "Degenerate tree.", **span_info}
        elif level >= green_at:
            span = {
                "status": "green",
                "reason": "All tree nodes covered." if level >= tree_total else f"{level}/{tree_total} nodes covered.",
                **span_info,
            }
        elif level >= yellow_at:
            span = {
                "status": "yellow",
                "reason": f"{level}/{tree_total} nodes covered.",
                **span_info,
            }
        else:
            span = {
                "status": "red",
                "reason": "No tree coverage yet." if level == 0 else f"{level}/{tree_total} nodes covered.",
                **span_info,
            }

    return {
        "good_count": good,
        "bad_count": bad,
        "total_count": total,
        "smart": smart,
        "stable": stable,
        "span": span,
    }


def calculate_diversity_level_over_time(
    clips_dict: dict[int, dict[str, Any]],
    label_history: list[tuple[int, str, float]],
    inclusion_value: int = 0,
) -> list[dict[str, Any]]:
    """Return cached per-step diversity levels.

    Diversity levels are computed and stored by :func:`_ensure_cache` as it
    processes each label-history step, so this function ensures the cache is
    current before reading it.
    """
    with _progress_lock:
        _ensure_cache(clips_dict, label_history, inclusion_value)
        return [step["diversity"] for step in _cached_steps if step.get("diversity") is not None]


def analyze_labeling_progress(
    clips_dict: dict[int, dict[str, Any]],
    label_history: list[tuple[int, str, float]],
    current_good_votes: dict[int, None],
    current_bad_votes: dict[int, None],
    inclusion_value: int = 0,
) -> dict[str, Any]:
    """Run a comprehensive analysis of labelling progress.

    Models and stability metrics are read from the per-step cache.  Error
    cost is recomputed cheaply using cached models (forward passes only).
    """
    with _progress_lock:
        _ensure_cache(clips_dict, label_history, inclusion_value)

        error_cost = _eval_cached_models(clips_dict, current_good_votes, current_bad_votes, inclusion_value)

        stability = [step["stability"] for step in _cached_steps if step["stability"] is not None]

        diversity = calculate_diversity_level_over_time(clips_dict, label_history, inclusion_value)

    return {
        "error_cost_over_time": error_cost,
        "stability_over_time": stability,
        "diversity_level_over_time": diversity,
        "total_labels": len(current_good_votes) + len(current_bad_votes),
        "total_medias": len(clips_dict),
    }
