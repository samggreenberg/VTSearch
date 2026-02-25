"""Progress tracking and stopping condition analysis.

Caches trained models and stability metrics per labelling step so that
repeated queries (the progress button, the auto-indicator) never retrain
models that have already been computed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from vtsearch.models.training import find_optimal_threshold, train_model

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


def clear_progress_cache() -> None:
    """Clear all cached progress data.

    Must be called whenever votes are cleared, clips change, or inclusion
    is altered so that stale models are not reused.
    """
    global _cache_inclusion, _cache_prev_predictions
    _cached_steps.clear()
    _cache_good_ids.clear()
    _cache_bad_ids.clear()
    _cache_prev_predictions = None
    _cache_inclusion = None


def _ensure_cache(
    clips_dict: dict[int, dict[str, Any]],
    label_history: list[tuple[int, str, float]],
    inclusion_value: int,
) -> None:
    """Bring the cache up to date with *label_history*.

    Only computes steps that are not yet cached.  If *inclusion_value*
    differs from the value used for existing cache entries the entire cache
    is rebuilt.
    """
    global _cache_inclusion, _cache_prev_predictions

    if _cache_inclusion is not None and _cache_inclusion != inclusion_value:
        clear_progress_cache()

    if _cache_inclusion is None:
        _cache_inclusion = inclusion_value

    start = len(_cached_steps)
    if start >= len(label_history):
        return  # already up to date

    all_clip_ids = sorted(clips_dict.keys())

    for t in range(start, len(label_history)):
        clip_id, label, _ = label_history[t]

        # Incrementally update running label sets
        if label == "unlabel":
            _cache_good_ids.discard(clip_id)
            _cache_bad_ids.discard(clip_id)
        elif label == "good":
            _cache_bad_ids.discard(clip_id)
            _cache_good_ids.add(clip_id)
        else:
            _cache_good_ids.discard(clip_id)
            _cache_bad_ids.add(clip_id)

        good_ids = list(_cache_good_ids)
        bad_ids = list(_cache_bad_ids)

        model: Optional[nn.Sequential] = None
        threshold: Optional[float] = None
        stability: Optional[dict[str, Any]] = None

        if _cache_good_ids and _cache_bad_ids:
            # Build training data
            X_list: list[np.ndarray] = []
            y_list: list[float] = []
            for cid in _cache_good_ids:
                if cid in clips_dict:
                    X_list.append(clips_dict[cid]["embedding"])
                    y_list.append(1.0)
            for cid in _cache_bad_ids:
                if cid in clips_dict:
                    X_list.append(clips_dict[cid]["embedding"])
                    y_list.append(0.0)

            if len(X_list) >= 2:
                import torch  # noqa: PLC0415

                X = torch.tensor(np.array(X_list), dtype=torch.float32)
                y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
                input_dim = X.shape[1]

                model = train_model(X, y, input_dim, inclusion_value)

                with torch.no_grad():
                    scores = torch.sigmoid(model(X)).squeeze(1).tolist()
                threshold = find_optimal_threshold(scores, y_list, inclusion_value)

                # --- Stability ---
                labeled_ids = _cache_good_ids | _cache_bad_ids
                unlabeled_ids = [cid for cid in all_clip_ids if cid not in labeled_ids]

                if not unlabeled_ids:
                    stability = {
                        "time_index": t,
                        "num_labels": len(good_ids) + len(bad_ids),
                        "num_flips": 0,
                        "num_unlabeled": 0,
                    }
                else:
                    unlabeled_embs = np.array([clips_dict[cid]["embedding"] for cid in unlabeled_ids])
                    X_unlabeled = torch.tensor(unlabeled_embs, dtype=torch.float32)

                    with torch.no_grad():
                        scores_unl = torch.sigmoid(model(X_unlabeled)).squeeze(1).tolist()

                    predictions: dict[int, int] = {
                        cid: 1 if score >= threshold else 0 for cid, score in zip(unlabeled_ids, scores_unl)
                    }

                    num_flips = 0
                    if _cache_prev_predictions is not None:
                        common = predictions.keys() & _cache_prev_predictions.keys()
                        for cid in common:
                            if predictions[cid] != _cache_prev_predictions[cid]:
                                num_flips += 1

                    stability = {
                        "time_index": t,
                        "num_labels": len(good_ids) + len(bad_ids),
                        "num_flips": num_flips,
                        "num_unlabeled": len(unlabeled_ids),
                    }

                    _cache_prev_predictions = predictions

        _cached_steps.append(
            {
                "model": model,
                "threshold": threshold,
                "good_ids": good_ids,
                "bad_ids": bad_ids,
                "stability": stability,
            }
        )


# ---------------------------------------------------------------------------
# Helper: evaluate cached models against a label set
# ---------------------------------------------------------------------------


def _eval_cached_models(
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
        if cid in clips_dict:
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
            scores = torch.sigmoid(step["model"](X_eval)).squeeze(1).tolist()

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
        clips_dict: Mapping of clip ID to clip data dict with ``"embedding"``.
        label_history: Ordered labelling events.
        time_index: Index into *label_history*.
        inclusion_value: FPR/FNR trade-off in ``[-10, 10]``.

    Returns:
        ``(model, threshold, good_ids, bad_ids)`` — same contract as before.
    """
    if time_index < 0 or time_index >= len(label_history):
        return None, None, [], []

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
    _ensure_cache(clips_dict, label_history, inclusion_value)
    return _eval_cached_models(clips_dict, current_good_votes, current_bad_votes, inclusion_value)


def calculate_prediction_stability_over_time(
    clips_dict: dict[int, dict[str, Any]],
    label_history: list[tuple[int, str, float]],
    inclusion_value: int = 0,
) -> list[dict[str, Any]]:
    """Return cached prediction-stability metrics for every step."""
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

    # The first stability entry always reports 0 flips because there are no
    # prior predictions to compare against.  Drop it so it doesn't
    # artificially drag the average toward zero.
    if len(stability) > 1:
        stability = stability[1:]

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

    STABLE_RATE_THRESHOLD = 0.02  # average less than 2% of predictions flipping
    STABLE_MAX_THRESHOLD = 0.05  # no single recent step above 5%

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

    _ensure_cache(clips_dict, label_history, inclusion_value)

    smart = _compute_smart_status(
        clips_dict, label_history, current_good_votes, current_bad_votes, inclusion_value, good, bad, total
    )
    stable = _compute_stable_status(good, bad, total)

    # Span status from diversity tree info (passed in from the route)
    if span_info is None:
        span = {"status": "red", "reason": "Diversity tree not available.", "level": -1, "depth": -1,
                "next_level_seen": 0, "next_level_total": 0}
    else:
        level = span_info["level"]
        depth = span_info["depth"]
        nls = span_info["next_level_seen"]
        nlt = span_info["next_level_total"]
        if level >= depth and level >= 0:
            span = {"status": "green", "reason": "All tree levels fully covered.", **span_info}
        elif level >= 4:
            span = {"status": "green",
                    "reason": f"Level {level}/{depth} full. {nls}/{nlt} of next level seen.", **span_info}
        elif level <= 1:
            span = {"status": "red", "reason": "No tree coverage yet." if level < 0
                    else f"Level {level}/{depth} full. {nls}/{nlt} of next level seen.", **span_info}
        else:
            span = {"status": "yellow", "reason": f"Level {level}/{depth} full. {nls}/{nlt} of next level seen.",
                    **span_info}

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
) -> list[dict[str, Any]]:
    """Replay label history against a fresh diversity tree to compute level at each step.

    Returns a list of dicts with ``num_labels``, ``diversity_level`` (fractional),
    and ``depth`` at each label step where the label count changes.
    """
    from vtsearch.models.diversity_tree import DiversityTree

    vectors: dict[int, np.ndarray] = {}
    for cid, clip in clips_dict.items():
        emb = clip.get("embedding")
        if emb is not None:
            vectors[cid] = np.asarray(emb, dtype=np.float32)

    if not vectors:
        return []

    tree = DiversityTree(vectors, k=3)
    d = tree.depth()

    good_ids: set[int] = set()
    bad_ids: set[int] = set()
    results: list[dict[str, Any]] = []

    for clip_id, label, _ in label_history:
        if clip_id not in tree.vector_to_leaf:
            continue

        if label == "unlabel":
            was_labeled = clip_id in good_ids or clip_id in bad_ids
            good_ids.discard(clip_id)
            bad_ids.discard(clip_id)
            if was_labeled and clip_id not in good_ids and clip_id not in bad_ids:
                tree.unlabel(clip_id)
        elif label == "good":
            bad_ids.discard(clip_id)
            good_ids.add(clip_id)
            tree.label(clip_id)
        else:  # bad
            good_ids.discard(clip_id)
            bad_ids.add(clip_id)
            tree.label(clip_id)

        num_labels = len(good_ids) + len(bad_ids)
        results.append({
            "num_labels": num_labels,
            "diversity_level": round(tree.fractional_diversity_level(), 4),
            "depth": d,
        })

    return results


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
    _ensure_cache(clips_dict, label_history, inclusion_value)

    error_cost = _eval_cached_models(clips_dict, current_good_votes, current_bad_votes, inclusion_value)

    stability = [step["stability"] for step in _cached_steps if step["stability"] is not None]

    diversity = calculate_diversity_level_over_time(clips_dict, label_history)

    return {
        "error_cost_over_time": error_cost,
        "stability_over_time": stability,
        "diversity_level_over_time": diversity,
        "total_labels": len(current_good_votes) + len(current_bad_votes),
        "total_clips": len(clips_dict),
    }
