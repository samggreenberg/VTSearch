"""Labeling-session analyzer: per-step model cache and stopping-condition metrics.

Caches trained MLPs and stability metrics per labelling step so that
repeated queries (the progress button, the auto-indicator) never retrain
models that have already been computed.

Unrelated to :mod:`vtscore.concurrency.progress`, which is the
infrastructure for tracking and cancelling long-running operations
(``ProgressTracker`` and the dataset/sort/eval/find singletons).  The
two modules used to share the ``progress.py`` name; this one was
renamed to make the distinction obvious.

Lock ordering (audit M1)
------------------------
``_progress_lock`` is acquired strictly *outside* ``vtscore.state.core._state_lock``.
Every callsite that needs to invalidate or clear the cache after a
state-lock'd mutation must release ``_state_lock`` before invoking a
function in this module.  Conversely, code inside ``_progress_lock`` must
not call into anything that acquires ``_state_lock`` - including helpers
on ``DatasetContext`` / ``DetectorContext`` that take the state lock,
and any of the resolve-context-then-mutate functions in
:mod:`vtscore.state.votes` / :mod:`vtscore.state.coverage`.  Holding
both locks in the opposite order would establish a cross-module cycle
and could deadlock.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from vtscore.concurrency.async_jobs import check_job_cancelled
from vtscore.embedding.media_vectors import media_embedding
from vtscore.training.mlp import LINEAR_HEAD, train_model
from vtscore.training.thresholds import conformal_threshold

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

# ---------------------------------------------------------------------------
# Diversity replay cache (independent of the model cache)
# ---------------------------------------------------------------------------
# The "diverse" series is a pure function of the coverage atlas and the order
# of label events - no MLP is involved.  It therefore gets its own cache and
# its own replay, so that:
#
#   * asking for "smart"/"stable" never pays for a coverage-atlas build (a
#     hierarchical k-means over every embedding - 47 s at 20k medias, 146 s at
#     60k, where the dataset is past ``COVERAGE_ATLAS_AUTO_THRESHOLD`` and so
#     has no load-time atlas to clone), and
#   * asking for "diverse" never pays to train one MLP per label step.
#
# Both series used to be produced by a single ``_ensure_cache`` walk, so each
# metric funded the other's dominant cost.
_cache_coverage_atlas: Any = None  # CoverageAtlas | None
#: One entry per label-history step, ``None`` where no atlas was available.
_cached_diversity: list[Optional[dict[str, Any]]] = []
_cache_div_good_ids: set[int] = set()
_cache_div_bad_ids: set[int] = set()

# Fixed pool the per-step stability forward pass scores, built once per cache
# lifetime and reused by every step (see ``_monitored_pool``).  ``_cache_
# monitored_X`` is a device-resident float tensor of the pool's embeddings;
# ``_cache_monitored_set`` is the id set for O(labels) unlabeled counting.
# In-memory only - never serialized (see the "No Persisted Vectors" rule).
_cache_monitored_ids: Optional[list[int]] = None
_cache_monitored_X: Any = None  # torch.Tensor | None
_cache_monitored_set: Optional[set[int]] = None

# Last fully-computed ``/api/labeling-status`` payload (minus the transient
# ``stale`` flag).  ``compute_labeling_status`` refreshes it on every full
# compute; the route returns it immediately to pollers while a background
# worker advances the per-step cache, so the 2 s poll never blocks on an MLP
# retrain (issue #2397).  Cleared alongside the cache so a stale detector's
# status is never shown after a detector switch / vote clear.
_status_snapshot: Optional[dict[str, Any]] = None

# Models available to ``_ensure_cache`` without retraining, keyed by
# ``(frozenset(good_ids), frozenset(bad_ids))`` so a step whose label set
# matches can reuse the model instead of training from scratch.  Populated
# from two places: :func:`inject_live_model` (the model ``train_and_score``
# actually used during sorting) and :func:`rethreshold_progress_cache` (every
# model of a cache being replayed under a new inclusion value).  Only the model
# is kept - the step's cutoff is always re-derived from the current inclusion,
# so one series never mixes calibration schemes.
_live_models: dict[tuple[frozenset[int], frozenset[int]], Any] = {}

# Reentrant lock protecting all module-level cache variables.
# RLock is used because public functions call _ensure_cache which may
# call clear_progress_cache internally when the inclusion value changes.
_progress_lock = threading.RLock()

# Upper bound on the number of items the per-step stability forward pass
# evaluates.  Advancing this cache (from the ``/api/labeling-status``
# background worker, or the ``/api/eval/train-and-score`` job) runs a forward
# over the whole monitored pool once per label step - O(dataset) per new vote.
# Above this cap we score a deterministic seeded sample of the eligible pool
# instead, holding the per-step cost flat as datasets grow.  The "stable"
# indicator keys off the flip *rate* (num_flips / num_unlabeled), for which a
# fixed random sample is an unbiased estimator; sampling from the *full*
# eligible pool (not the shrinking unlabeled set) keeps the monitored ids
# stable across steps so the step-to-step flip comparison stays meaningful.
# Mirrors ``_GMM_MAX_SAMPLES`` in ``vtscore.training.thresholds``.
_STABILITY_MAX_SAMPLES = 50_000

# How long :func:`cached_indicator_history` will wait for ``_progress_lock``
# before declaring the cache unreadable.  Long enough to ride out the brief
# holds taken by status reads, short enough that a click landing mid-build
# falls through to the async job instead of hanging on it.
_CACHE_READ_LOCK_TIMEOUT = 0.25


def clear_progress_cache() -> None:
    """Clear all cached progress data.

    Must be called whenever votes are cleared or medias change so that stale
    models are not reused.  An *inclusion* change no longer needs this - see
    :func:`rethreshold_progress_cache`, which keeps the (inclusion-independent)
    models and re-derives only the cutoffs.
    """
    global _status_snapshot
    with _progress_lock:
        _clear_model_cache()
        clear_diversity_cache()
        # Drop the status snapshot too: it belonged to the just-cleared
        # detector/labelset and would otherwise be served (stale) for the next
        # one until its first background refresh lands.
        _status_snapshot = None


def _clear_model_cache() -> None:
    """Drop the per-step model cache, leaving the diversity replay intact.

    Split out of :func:`clear_progress_cache` for
    :func:`rethreshold_progress_cache`, which must rebuild the model steps but
    would be defeating its own purpose if it also threw away the coverage atlas
    (inclusion-independent, and the single most expensive artifact here).
    """
    global _cache_inclusion, _cache_prev_predictions
    global _cache_monitored_ids, _cache_monitored_X, _cache_monitored_set
    with _progress_lock:
        _cached_steps.clear()
        _cache_good_ids.clear()
        _cache_bad_ids.clear()
        _cache_prev_predictions = None
        _cache_inclusion = None
        # The monitored pool is derived from ``clips_dict``; medias may have
        # changed under us, so drop it alongside everything else.
        _cache_monitored_ids = None
        _cache_monitored_X = None
        _cache_monitored_set = None
        _live_models.clear()


def clear_diversity_cache() -> None:
    """Drop the coverage atlas and the replayed per-step diversity series."""
    global _cache_coverage_atlas
    with _progress_lock:
        _cache_coverage_atlas = None
        _cached_diversity.clear()
        _cache_div_good_ids.clear()
        _cache_div_bad_ids.clear()


def invalidate_progress_cache_from(media_id: int) -> None:
    """Truncate the progress cache to just before *media_id* first appeared.

    Called when a vote switches polarity (good→bad or bad→good).  Steps
    before the media was first labeled are still valid - their models never
    included this media in training data.  Only steps from the first
    appearance onward are discarded so they can be retrained and their
    stability/evaluation metrics recomputed.

    The diversity cache is deliberately left alone: coverage evidence is
    polarity-agnostic (``CoverageAtlas._covered`` tests ``n_pos + n_neg > 0``),
    so moving a label between the two channels cannot change any step's
    ``coverage_level()``.  The old code rewound and replayed the atlas here for
    a result that was provably identical.
    """
    global _cache_prev_predictions, _status_snapshot
    with _progress_lock:
        # Find the first cached step that includes media_id in its training data.
        truncate_at = None
        for i, step in enumerate(_cached_steps):
            if media_id in step["good_ids"] or media_id in step["bad_ids"]:
                truncate_at = i
                break

        if truncate_at is None:
            # Media never appeared in any cached step.  Still need to clear
            # live models - they may have been injected by learned-sort
            # without building the progress cache.
            _live_models.clear()
            return

        # Keep steps [0, truncate_at); discard the rest.
        del _cached_steps[truncate_at:]

        # Restore the running ID sets to the surviving prefix's final state.
        _cache_good_ids.clear()
        _cache_bad_ids.clear()
        if _cached_steps:
            last = _cached_steps[-1]
            _cache_good_ids.update(last["good_ids"])
            _cache_bad_ids.update(last["bad_ids"])
        else:
            # truncate_at == 0: media was present from the very first step, so
            # the whole prefix is gone and no label survives.  No cached step
            # remains to source the Smart / Stable indicators from, so drop the
            # stale snapshot (parity with the old step-0 full-clear path).
            _status_snapshot = None

        # Reset the stability prediction chain - it will restart from the
        # truncation point when _ensure_cache replays the remaining history.
        _cache_prev_predictions = None

        # Clear live models - some may have been trained with the old label.
        _live_models.clear()


def inject_live_model(
    good_votes: dict[int, None],
    bad_votes: dict[int, None],
    model: nn.Sequential,
) -> None:
    """Register a live model from ``train_and_score`` for progress-cache reuse.

    Called by the learned-sort route after each live training run.  The model
    is stored keyed by its label set so ``_ensure_cache`` can look it up
    instead of retraining from scratch.

    The live *threshold* is intentionally not carried over: it is
    cross-calibrated over held-out folds, whereas every other step of the
    series takes an in-sample :func:`conformal_threshold`.  Mixing the two put
    a step-change into the plotted curve wherever a live model happened to land,
    and made the cutoffs impossible to re-derive when inclusion moved.  The
    cache now always derives its own cutoff (see :func:`_step_threshold`).
    """
    key = (frozenset(good_votes), frozenset(bad_votes))
    with _progress_lock:
        _live_models[key] = model


def _active_context_atlas() -> Any:
    """Return the active dataset context's coverage atlas, or ``None``.

    Read *without* acquiring ``_state_lock``: this runs under ``_progress_lock``
    (see the lock-ordering note at module top), which forbids taking
    ``_state_lock`` while held.  ``get_active_context()`` itself takes no lock,
    and the atlas it returns is only ever *read* here (its structure is
    immutable once built), so the lock-free read is safe - at worst a stale
    reference fails the id-set match below and we fall back to a fresh build.
    In the ``/api/labeling-status`` background worker the dataset context is
    bound thread-locally by ``JobManager``, so this resolves the right atlas.
    """
    try:
        from vtscore.state.core import get_active_context  # noqa: PLC0415

        return get_active_context().coverage_atlas
    except Exception:
        return None


def _build_coverage_atlas(clips_dict: dict[int, dict[str, Any]]) -> Any:
    """Build a CoverageAtlas from clip embeddings, or ``None`` if no embeddings.

    When the active dataset context already holds an atlas over *exactly* this
    id set, its hierarchical-k-means structure is identical to what a rebuild
    would produce, so we :meth:`~CoverageAtlas.structural_clone` it (sharing the
    node table by reference, fresh label overlay) instead of re-fitting under
    ``_progress_lock`` - which otherwise starves the request pool at N in the
    few-thousands on every polarity-flip invalidate.
    """
    vectors: dict[int, np.ndarray] = {
        cid: np.asarray(media_embedding(media), dtype=np.float32)
        for cid, media in clips_dict.items()
        if media_embedding(media) is not None
    }
    if not vectors:
        return None

    ctx_atlas = _active_context_atlas()
    if ctx_atlas is not None and ctx_atlas.vector_to_leaf.keys() == vectors.keys():
        return ctx_atlas.structural_clone()

    from vtscore.state.coverage_atlas import CoverageAtlas, auto_max_depth  # noqa: PLC0415

    # Cap the depth exactly as every other build site does
    # (``build_coverage_atlas`` / ``build_coverage_atlas_for_context``).
    # Omitting it left this fallback on ``COVERAGE_ATLAS_MAX_DEPTH``, so the
    # atlas built here was *deeper* - and cost many more k-means fits - than
    # the context atlas it stands in for.  That is the whole cost of a cold
    # progress-cache build on a dataset large enough to skip the load-time
    # atlas build, and it runs under ``_progress_lock``.
    return CoverageAtlas(vectors, k=3, max_depth=auto_max_depth(len(vectors), k=3))


def _ensure_diversity_cache(
    clips_dict: dict[int, dict[str, Any]],
    label_history: list[tuple[int, str, float]],
    on_step: Optional[Any] = None,
) -> None:
    """Bring the per-step diversity series up to date with *label_history*.

    Replays label events onto the coverage atlas, recording the coverage level
    after each one.  No model is trained and no embedding is scored, so this
    costs one atlas build (usually a free ``structural_clone`` of the dataset
    context's) plus O(1) per step.

    Deliberately independent of the model cache: the two used to share a walk,
    which meant a "diverse" plot trained an MLP per label step it never read.
    """
    global _cache_coverage_atlas

    if _cache_coverage_atlas is None and not _cached_diversity:
        _cache_coverage_atlas = _build_coverage_atlas(clips_dict)

    start = len(_cached_diversity)
    total = len(label_history)
    for t in range(start, total):
        check_job_cancelled()
        media_id, label, _ = label_history[t]
        was_labeled = media_id in _cache_div_good_ids or media_id in _cache_div_bad_ids
        if label == "unlabel":
            _cache_div_good_ids.discard(media_id)
            _cache_div_bad_ids.discard(media_id)
        elif label == "good":
            _cache_div_bad_ids.discard(media_id)
            _cache_div_good_ids.add(media_id)
        else:
            _cache_div_good_ids.discard(media_id)
            _cache_div_bad_ids.add(media_id)
        _cached_diversity.append(_diversity_point(media_id, label, was_labeled))
        if on_step is not None:
            on_step(t + 1, total)


def _diversity_point(media_id: int, label: str, was_labeled: bool) -> Optional[dict[str, Any]]:
    """Mirror one label event onto the coverage atlas and return its level info."""
    atlas = _cache_coverage_atlas
    if atlas is None:
        return None
    if label == "unlabel":
        # Only unlabel on the atlas when the item is no longer labeled at all
        # (guards against good→bad re-labels going through "unlabel").
        if was_labeled and media_id not in _cache_div_good_ids and media_id not in _cache_div_bad_ids:
            if media_id in atlas.vector_to_leaf:
                atlas.unlabel(media_id)
    elif media_id in atlas.vector_to_leaf:
        atlas.label(media_id, good=label == "good")
    return {
        "num_labels": len(_cache_div_good_ids) + len(_cache_div_bad_ids),
        "diversity_level": atlas.coverage_level(),
        "depth": atlas.total_nodes,
    }


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


def _collect_training_data(
    clips_dict: dict[int, dict[str, Any]],
) -> tuple[list[np.ndarray], list[float]]:
    """Gather embeddings and labels from the current good/bad ID sets."""
    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    for cid in _cache_good_ids:
        if cid in clips_dict and media_embedding(clips_dict[cid]) is not None:
            X_list.append(media_embedding(clips_dict[cid]))
            y_list.append(1.0)
    for cid in _cache_bad_ids:
        if cid in clips_dict and media_embedding(clips_dict[cid]) is not None:
            X_list.append(media_embedding(clips_dict[cid]))
            y_list.append(0.0)
    return X_list, y_list


def _monitored_pool(
    clips_dict: dict[int, dict[str, Any]],
    all_media_ids: list[int],
) -> tuple[list[int], Any, set[int]]:
    """Return the fixed ``(ids, X, id_set)`` pool the stability pass scores.

    The pool is the embeddable subset of *all_media_ids*, bounded to a
    deterministic seeded sample of ``_STABILITY_MAX_SAMPLES``.  Sampling the
    full eligible pool (rather than the per-step unlabeled set) keeps the
    monitored ids stable across steps, so the flip comparison against
    ``_cache_prev_predictions`` stays over a consistent id set; the resulting
    flip *rate* is an unbiased estimate of the true rate.

    Built once per cache lifetime and memoised.  It used to be rebuilt inside
    every step - an O(N x D) numpy materialisation per label-history step,
    which dominated the cost of advancing the cache.  The pool depends only on
    *clips_dict*, which cannot change without a ``clear_progress_cache()``,
    so one build per cache lifetime is sound.
    """
    global _cache_monitored_ids, _cache_monitored_X, _cache_monitored_set

    if _cache_monitored_ids is not None:
        return _cache_monitored_ids, _cache_monitored_X, _cache_monitored_set  # type: ignore[return-value]

    import torch  # noqa: PLC0415

    from vtscore.embedding.loader import ensure_torch_configured, get_torch_device  # noqa: PLC0415

    eligible = [cid for cid in all_media_ids if media_embedding(clips_dict.get(cid, {})) is not None]
    if len(eligible) > _STABILITY_MAX_SAMPLES:
        rng = np.random.default_rng(42)
        sampled = set(rng.choice(np.asarray(eligible), size=_STABILITY_MAX_SAMPLES, replace=False).tolist())
        eligible = [cid for cid in eligible if cid in sampled]

    if eligible:
        ensure_torch_configured()
        embs = np.array([media_embedding(clips_dict[cid]) for cid in eligible])
        # Park the tensor on the training device once so per-step scoring is a
        # pure forward pass with no host->device copy.
        _cache_monitored_X = torch.tensor(embs, dtype=torch.float32).to(get_torch_device())
    else:
        _cache_monitored_X = None

    _cache_monitored_ids = eligible
    _cache_monitored_set = set(eligible)
    return _cache_monitored_ids, _cache_monitored_X, _cache_monitored_set


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

    monitored_ids, X_monitored, monitored_set = _monitored_pool(clips_dict, all_media_ids)

    labeled_ids = _cache_good_ids | _cache_bad_ids
    # Labels are few relative to the pool, so count the overlap from the
    # labelset rather than rescanning the pool.
    num_unlabeled = len(monitored_ids) - sum(1 for cid in labeled_ids if cid in monitored_set)

    if num_unlabeled <= 0 or X_monitored is None:
        return {"time_index": t, "num_labels": num_labels, "num_flips": 0, "num_unlabeled": 0}

    # Score the whole monitored pool in one pass and drop the currently-labeled
    # ids afterwards.  Scoring the handful of extra (labeled) rows is far
    # cheaper than re-materialising a per-step tensor of the unlabeled subset.
    with torch.no_grad():
        X_in = X_monitored.to(next(model.parameters()).device)
        scores_unl = torch.sigmoid(model(X_in)).squeeze(1).cpu().tolist()

    predictions: dict[int, int] = {
        cid: 1 if score >= threshold else 0
        for cid, score in zip(monitored_ids, scores_unl, strict=True)
        if cid not in labeled_ids
    }

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
            "num_unlabeled": num_unlabeled,
        }
    # else: no prior predictions to compare - leave stability as None.

    _cache_prev_predictions = predictions
    return stability


def _step_threshold(
    model: nn.Sequential,
    X: Any,
    y_list: list[float],
    inclusion_value: int,
) -> float:
    """Derive one step's cutoff from its own training-set scores.

    Training-set scores, not held-out ones: this cache only needs a rough
    per-step cutoff for the stability curve, so the optimistic (tighter) band
    from in-sample quantiles is acceptable here.  Every step in a series - freshly
    trained, reused-live, or replayed under a new inclusion - goes through this
    one function, so a single curve never mixes calibration schemes.
    """
    import torch  # noqa: PLC0415

    with torch.no_grad():
        X_dev = X.to(next(model.parameters()).device)
        scores = torch.sigmoid(model(X_dev)).squeeze(1).cpu().tolist()
    return conformal_threshold(scores, y_list, inclusion_value)


def _resolve_step_model(
    clips_dict: dict[int, dict[str, Any]],
    all_media_ids: list[int],
    t: int,
    num_labels: int,
    inclusion_value: int,
    good_ids: list[int],
    bad_ids: list[int],
    prev: Optional[dict[str, Any]],
) -> tuple[Optional[nn.Sequential], Optional[float], Optional[dict[str, Any]]]:
    """Resolve the model, threshold, and stability for one cache step.

    Reuses the previous step's model when the training data is unchanged,
    otherwise reuses a model already available for this exact label set (see
    :data:`_live_models`), or trains a fresh one.  Returns ``(model, threshold,
    stability)``; all three are ``None`` when no model is possible (e.g. only
    one label polarity present).
    """
    global _cache_prev_predictions

    # Check whether the training data actually changed compared to the
    # previous step.  If the good/bad ID sets are identical, the model
    # would be the same - skip training and stability recording so the
    # line graph and Stable indicator only reflect genuine model updates.
    training_data_changed = (
        prev is None or set(good_ids) != set(prev["good_ids"]) or set(bad_ids) != set(prev["bad_ids"])
    )

    if not training_data_changed:
        # Reuse previous model - no new training or stability entry.
        model = prev["model"] if prev else None
        threshold = prev["threshold"] if prev else None
        return model, threshold, None

    if not _cache_good_ids or not _cache_bad_ids:
        # No model possible - clear prediction baseline so the first step
        # after regaining a model doesn't produce a misleading flip count.
        _cache_prev_predictions = None
        return None, None, None

    X_list, y_list = _collect_training_data(clips_dict)
    if len(X_list) < 2:
        return None, None, None

    import torch  # noqa: PLC0415

    X = torch.tensor(np.array(X_list), dtype=torch.float32)

    # A model for this exact label set may already exist - injected by
    # ``train_and_score`` during live sorting, or carried over by
    # ``rethreshold_progress_cache``.  Reusing it skips the step's dominant
    # cost; the cutoff is re-derived either way.
    live_key = (frozenset(_cache_good_ids), frozenset(_cache_bad_ids))
    model = _live_models.get(live_key)
    if model is None:
        y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
        # Linear (logistic) head, matching the production detector this previews.
        # This is the walk's remaining cost - ~105 ms per step, essentially all
        # of it per-epoch autograd/Adam dispatch rather than arithmetic.  Warm-
        # starting from the previous step's weights was tried and reverted: the
        # early-stop plateau never fires either way, so it saved no epochs, and
        # pairing it with a shorter budget left every step inheriting its
        # predecessor's under-converged weights - measurably *further* from the
        # converged fit than a cold run.  Cutting this further means changing
        # what the curve previews, so it stays.
        model = train_model(X, y, X.shape[1], hidden_dim=LINEAR_HEAD)

    threshold = _step_threshold(model, X, y_list, inclusion_value)
    stability = _compute_step_stability(model, threshold, clips_dict, all_media_ids, t, num_labels)
    return model, threshold, stability


def _ensure_cache(
    clips_dict: dict[int, dict[str, Any]],
    label_history: list[tuple[int, str, float]],
    inclusion_value: int,
    on_step: Optional[Any] = None,
) -> None:
    """Bring the per-step *model* cache up to date with *label_history*.

    Only computes steps that are not yet cached.  If *inclusion_value*
    differs from the value used for existing cache entries the entire cache
    is rebuilt - :func:`rethreshold_progress_cache` exists so that the normal
    inclusion-slider path never reaches that branch.

    *on_step*, when given, is called as ``on_step(done, total)`` after each
    step so a background job can report real progress; the walk is the only
    slow thing in the request, and without this the caller's progress bar sat
    at 0 % for its entire duration.

    Does **not** touch the coverage atlas or the diversity series - see
    :func:`_ensure_diversity_cache`.
    """
    global _cache_inclusion

    if _cache_inclusion is not None and _cache_inclusion != inclusion_value:
        clear_progress_cache()

    if _cache_inclusion is None:
        _cache_inclusion = inclusion_value

    start = len(_cached_steps)
    total = len(label_history)
    if start >= total:
        return  # already up to date

    all_media_ids = sorted(clips_dict.keys())

    for t in range(start, total):
        # Each step retrains a model; honour a cancel of the owning eval job
        # here so a long history doesn't run to completion after cancel.  The
        # partially-built cache is a valid prefix (steps 0..t-1), so the next
        # run resumes cleanly from ``len(_cached_steps)``.  No-op outside a
        # job (see ``async_jobs.check_job_cancelled``).
        check_job_cancelled()
        media_id, label, _ = label_history[t]

        _apply_label_event(media_id, label)

        good_ids = list(_cache_good_ids)
        bad_ids = list(_cache_bad_ids)
        num_labels = len(good_ids) + len(bad_ids)

        prev = _cached_steps[-1] if _cached_steps else None
        model, threshold, stability = _resolve_step_model(
            clips_dict, all_media_ids, t, num_labels, inclusion_value, good_ids, bad_ids, prev
        )

        _cached_steps.append(
            {
                "model": model,
                "threshold": threshold,
                "good_ids": good_ids,
                "bad_ids": bad_ids,
                "stability": stability,
            }
        )
        if on_step is not None:
            on_step(t + 1, total)


def rethreshold_progress_cache(inclusion_value: int) -> None:
    """Re-key the per-step cache to a new *inclusion_value* without retraining.

    Inclusion is a pure cutoff knob: :func:`~vtscore.training.mlp.train_model`
    never sees it, so every cached step's model is still exactly the model that
    step would train under the new value.  Only the derived cutoffs - and the
    prediction flips that depend on them - change.

    Rather than reconstruct those in place (which would have to re-derive which
    steps recorded stability, and why), this hands every trained model back to
    :data:`_live_models` and clears the step list.  The next :func:`_ensure_cache`
    replays the history through the ordinary code path and finds a ready model
    at every step, so the rebuild costs one stability forward pass per step and
    trains nothing.  Measured on 20k medias / 200 votes that is ~1.7 s instead of
    the ~21 s a full retrain took, which is what dragging the inclusion slider
    used to cost.
    """
    global _cache_inclusion, _status_snapshot
    with _progress_lock:
        if _cache_inclusion is None or _cache_inclusion == inclusion_value:
            _cache_inclusion = inclusion_value
            return

        preserved = {
            (frozenset(step["good_ids"]), frozenset(step["bad_ids"])): step["model"]
            for step in _cached_steps
            if step["model"] is not None
        }
        # The monitored pool depends only on ``clips_dict`` (unchanged by an
        # inclusion move), so carry it over rather than re-materialising it.
        pool = (_cache_monitored_ids, _cache_monitored_X, _cache_monitored_set)
        snapshot = _status_snapshot

        # Model side only: the diversity replay and its atlas are untouched by
        # a cutoff change.
        _clear_model_cache()

        _live_models.update(preserved)
        _restore_monitored_pool(*pool)
        # The snapshot is the stale-poll bridge, not a computed answer; keeping
        # it avoids flashing "Computing indicators..." across a slider drag.
        # The next poll recomputes it against the new cutoffs.
        _status_snapshot = snapshot
        _cache_inclusion = inclusion_value


def _restore_monitored_pool(ids: Optional[list[int]], X: Any, id_set: Optional[set[int]]) -> None:
    """Reinstate a monitored pool carried across a cache clear."""
    global _cache_monitored_ids, _cache_monitored_X, _cache_monitored_set
    _cache_monitored_ids = ids
    _cache_monitored_X = X
    _cache_monitored_set = id_set


# ---------------------------------------------------------------------------
# Helper: evaluate cached models against a label set
# ---------------------------------------------------------------------------


def _score_step(
    step: dict[str, Any],
    X_eval: Any,
    eval_labels: list[float],
    total_positives: int,
    total_negatives: int,
    fpr_weight: float,
    fnr_weight: float,
    t: int,
) -> dict[str, Any]:
    """Score one cached step against the evaluation set (forward pass only).

    Returns an error-cost dict for the step.  The caller guarantees
    ``step["model"]`` is not ``None``.
    """
    import torch  # noqa: PLC0415

    with torch.no_grad():
        X_in = X_eval.to(next(step["model"].parameters()).device)
        scores = torch.sigmoid(step["model"](X_in)).squeeze(1).cpu().tolist()

    fp = fn = 0
    for score, true_label in zip(scores, eval_labels, strict=True):
        predicted = 1 if score >= step["threshold"] else 0
        if predicted == 1 and true_label == 0:
            fp += 1
        elif predicted == 0 and true_label == 1:
            fn += 1

    fpr = fp / total_negatives if total_negatives > 0 else 0.0
    fnr = fn / total_positives if total_positives > 0 else 0.0
    error_cost = fpr_weight * fpr + fnr_weight * fnr

    return {
        "time_index": t,
        "num_labels": len(step["good_ids"]) + len(step["bad_ids"]),
        "error_cost": round(error_cost, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
    }


def _build_eval_set(
    clips_dict: dict[int, dict[str, Any]],
    current_good_votes: dict[int, None],
    current_bad_votes: dict[int, None],
) -> Optional[tuple[Any, list[float], int, int]]:
    """Build the evaluation tensor and label set from the current votes.

    Returns ``(X_eval, eval_labels, total_positives, total_negatives)`` or
    ``None`` when there are no usable labeled medias to evaluate against.
    """
    # Build evaluation set from current votes
    current_labels: dict[int, float] = {}
    for cid in current_good_votes:
        current_labels[cid] = 1.0
    for cid in current_bad_votes:
        current_labels[cid] = 0.0

    if not current_labels:
        return None

    eval_embs: list[np.ndarray] = []
    eval_labels: list[float] = []
    for cid, lbl in current_labels.items():
        if cid in clips_dict and media_embedding(clips_dict[cid]) is not None:
            eval_embs.append(media_embedding(clips_dict[cid]))
            eval_labels.append(lbl)

    if not eval_embs:
        return None

    import torch  # noqa: PLC0415

    X_eval = torch.tensor(np.array(eval_embs), dtype=torch.float32)
    total_positives = sum(1 for lbl in eval_labels if lbl == 1)
    total_negatives = len(eval_labels) - total_positives
    return X_eval, eval_labels, total_positives, total_negatives


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

    eval_set = _build_eval_set(clips_dict, current_good_votes, current_bad_votes)
    if eval_set is None:
        return []
    X_eval, eval_labels, total_positives, total_negatives = eval_set

    if end is None:
        end = len(_cached_steps)

    results: list[dict[str, Any]] = []
    for t in range(start, end):
        step = _cached_steps[t]
        if step["model"] is None:
            continue

        results.append(
            _score_step(step, X_eval, eval_labels, total_positives, total_negatives, fpr_weight, fnr_weight, t)
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
        ``(model, threshold, good_ids, bad_ids)`` - same contract as before.
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
    on_step: Optional[Any] = None,
) -> list[dict[str, Any]]:
    """Calculate classification error cost at each labelling step.

    Advances the per-step model cache if needed, then scores every cached model
    against the current labelset (forward passes only).  Pass *on_step* to
    receive ``(done, total)`` progress from the cache walk.
    """
    with _progress_lock:
        _ensure_cache(clips_dict, label_history, inclusion_value, on_step=on_step)
        return _eval_cached_models(clips_dict, current_good_votes, current_bad_votes, inclusion_value)


def calculate_prediction_stability_over_time(
    clips_dict: dict[int, dict[str, Any]],
    label_history: list[tuple[int, str, float]],
    inclusion_value: int = 0,
    on_step: Optional[Any] = None,
) -> list[dict[str, Any]]:
    """Return cached prediction-stability metrics for every step."""
    with _progress_lock:
        _ensure_cache(clips_dict, label_history, inclusion_value, on_step=on_step)
        return [step["stability"] for step in _cached_steps if step["stability"] is not None]


def partial_indicator_series(
    metric: str,
    clips_dict: dict[int, dict[str, Any]],
    current_good_votes: dict[int, None],
    current_bad_votes: dict[int, None],
    inclusion_value: int = 0,
) -> list[dict[str, Any]]:
    """Return *metric*'s series over the steps cached **so far**.

    Intended to be called from an ``on_step`` callback, so a long cache walk can
    publish a curve that fills in as it goes instead of showing nothing until it
    finishes.  ``_progress_lock`` is an ``RLock`` and the callback runs on the
    thread already holding it, so re-entering here is free.
    """
    with _progress_lock:
        if metric == "smart":
            return _eval_cached_models(clips_dict, current_good_votes, current_bad_votes, inclusion_value)
        if metric == "stable":
            return [step["stability"] for step in _cached_steps if step["stability"] is not None]
        return [point for point in _cached_diversity if point is not None]


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


def _compute_span_status(span_info: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Compute the Span (diversity-coverage) red/yellow/green status.

    Depends only on the coverage-atlas ``span_info`` passed in from the route
    (``level`` = consecutive BFS-order seen nodes, ``depth`` = total nodes),
    not on the per-step MLP cache, so it stays cheap and is reused verbatim by
    the pending-status placeholder.
    """
    # The old metric required 4 full tree levels for green, which in a k=3
    # tree is 1+3+9+27 = 40 nodes.  We preserve that scale: green at 40
    # nodes (capped at total), yellow at 10, red below 10.
    from vtscore.config import CoreConfig  # noqa: PLC0415

    SPAN_GREEN = CoreConfig.from_settings().autopilot_goal_diversity
    SPAN_YELLOW = 10
    if span_info is None:
        return {
            "status": "red",
            "reason": "Diversity tree not available.",
            "level": 0,
            "depth": 0,
        }

    level = span_info["level"]
    tree_total = span_info["depth"]  # total nodes
    green_at = min(SPAN_GREEN, tree_total)
    yellow_at = min(SPAN_YELLOW, green_at)
    if tree_total <= 0:
        return {"status": "green", "reason": "Degenerate tree.", **span_info}
    if level >= green_at:
        return {
            "status": "green",
            "reason": "All tree nodes covered." if level >= tree_total else f"{level}/{tree_total} nodes covered.",
            **span_info,
        }
    if level >= yellow_at:
        return {
            "status": "yellow",
            "reason": f"{level}/{tree_total} nodes covered.",
            **span_info,
        }
    return {
        "status": "red",
        "reason": "No tree coverage yet." if level == 0 else f"{level}/{tree_total} nodes covered.",
        **span_info,
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

    Advancing the per-step cache (``_ensure_cache``) can retrain MLPs and run a
    forward pass over every unlabeled media, so this is the *heavy* path.  The
    result is stashed in ``_status_snapshot`` so the ``/api/labeling-status``
    route can serve it immediately (marked ``stale``) on subsequent polls while
    a background worker calls this to advance the cache off the request thread
    (issue #2397).
    """
    global _status_snapshot

    good = len(current_good_votes)
    bad = len(current_bad_votes)
    total = good + bad

    with _progress_lock:
        _ensure_cache(clips_dict, label_history, inclusion_value)

        smart = _compute_smart_status(
            clips_dict, label_history, current_good_votes, current_bad_votes, inclusion_value, good, bad, total
        )
        stable = _compute_stable_status(good, bad, total)

    # Span status from coverage atlas info (passed in from the route).
    span = _compute_span_status(span_info)

    result = {
        "good_count": good,
        "bad_count": bad,
        "total_count": total,
        "smart": smart,
        "stable": stable,
        "span": span,
    }
    # Refresh the snapshot the poll route hands back while the cache is behind.
    # Store a copy so a caller that mutates the returned dict (e.g. adding the
    # ``stale`` flag) doesn't retroactively corrupt the snapshot.
    with _progress_lock:
        _status_snapshot = dict(result)
    return result


def cached_indicator_history(
    metric: str,
    clips_dict: dict[int, dict[str, Any]],
    label_history: list[tuple[int, str, float]],
    current_good_votes: dict[int, None],
    current_bad_votes: dict[int, None],
    inclusion_value: int = 0,
) -> tuple[list[dict[str, Any]], bool]:
    """Read *metric*'s per-step history **without advancing the cache**.

    Returns ``(history, complete)``.  ``complete`` is ``False`` - with an empty
    history - whenever the per-step cache does not already cover the whole of
    *label_history*; the caller is expected to fall back to the async
    ``/api/eval/train-and-score`` job, which does the same work on a background
    thread with progress and cancellation.

    This is the counterpart to the ``calculate_*_over_time`` functions, which
    call :func:`_ensure_cache` and therefore retrain an MLP per uncached step
    on the calling thread.  Doing that inline is exactly what
    ``/api/labeling-status`` refuses to do (issue #2397), so the read path that
    backs the progress-plot modal must not do it either.

    When the cache *is* complete every branch is cheap: ``smart`` runs forward
    passes of the cached models over the (small) labeled set, and ``stable`` /
    ``diverse`` are plain reads of values recorded during the cache build.

    ``diverse`` reads a *different* cache from the other two (see
    :func:`_ensure_diversity_cache`), so its freshness is checked separately;
    nothing advances it in the background, so it is normally cold until the
    async job builds it once.
    """
    # Reading the cache needs ``_progress_lock``, but a background worker holds
    # that lock for the *entire* duration of a cache build - which is exactly
    # the multi-second work this function exists to avoid waiting on.  Blocking
    # here would reintroduce the hang whenever the click lands mid-refresh, so
    # give up quickly and report the cache as unavailable: the caller falls back
    # to the async job, which is the right answer in that state anyway.
    if not _progress_lock.acquire(timeout=_CACHE_READ_LOCK_TIMEOUT):
        return [], False
    try:
        if metric == "diverse":
            if len(_cached_diversity) < len(label_history):
                return [], False
            return [point for point in _cached_diversity if point is not None], True

        if not is_status_cache_fresh(label_history, inclusion_value):
            return [], False

        if metric == "smart":
            data = _eval_cached_models(clips_dict, current_good_votes, current_bad_votes, inclusion_value)
        else:
            data = [step["stability"] for step in _cached_steps if step["stability"] is not None]
        return data, True
    finally:
        _progress_lock.release()


def is_status_cache_fresh(label_history: list[tuple[int, str, float]], inclusion_value: int) -> bool:
    """Return ``True`` when the per-step cache already covers *label_history*.

    A fresh cache means ``compute_labeling_status`` will not retrain any model,
    so the route can compute the status inline instead of deferring to a
    background worker.  A mismatched ``inclusion_value`` counts as not-fresh
    because :func:`_ensure_cache` would rebuild the cache from scratch.
    """
    with _progress_lock:
        if _cache_inclusion is not None and _cache_inclusion != inclusion_value:
            return False
        return len(_cached_steps) >= len(label_history)


def _pending_labeling_status(
    current_good_votes: dict[int, None],
    current_bad_votes: dict[int, None],
    span_info: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build a status from only the cheap fields (counts + Span).

    The MLP-derived Smart / Stable indicators show a transient "computing"
    state; :func:`stale_labeling_status` overlays the real ones from the last
    snapshot when one exists.
    """
    good = len(current_good_votes)
    bad = len(current_bad_votes)
    computing = {"status": "yellow", "reason": "Computing indicators..."}
    return {
        "good_count": good,
        "bad_count": bad,
        "total_count": good + bad,
        "smart": dict(computing),
        "stable": dict(computing),
        "span": _compute_span_status(span_info),
    }


def stale_labeling_status(
    current_good_votes: dict[int, None],
    current_bad_votes: dict[int, None],
    span_info: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build the poll response served while a background cache refresh is pending.

    Counts and the coverage-atlas Span status are recomputed live (both cheap),
    so the panel's counters and diversity chip stay accurate the instant a vote
    lands.  Only the expensive Smart / Stable indicators lag: they come from the
    last ``compute_labeling_status`` snapshot, or - when none exists yet (first
    poll after a detector switch / session start) - a transient "computing"
    placeholder.  The caller stamps ``stale = True`` on the result.
    """
    status = _pending_labeling_status(current_good_votes, current_bad_votes, span_info)
    with _progress_lock:
        if _status_snapshot is not None:
            status["smart"] = dict(_status_snapshot["smart"])
            status["stable"] = dict(_status_snapshot["stable"])
    return status


def calculate_diversity_level_over_time(
    clips_dict: dict[int, dict[str, Any]],
    label_history: list[tuple[int, str, float]],
    on_step: Optional[Any] = None,
) -> list[dict[str, Any]]:
    """Return per-step diversity levels, replaying the history if needed.

    Trains nothing: the series depends only on the coverage atlas and the order
    of label events.  It used to ride along on the model-cache walk, which meant
    plotting it cost one MLP per label step.
    """
    with _progress_lock:
        _ensure_diversity_cache(clips_dict, label_history, on_step=on_step)
        return [point for point in _cached_diversity if point is not None]


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
    This is the one caller that genuinely wants all three series, so it pays
    for both walks.
    """
    with _progress_lock:
        _ensure_cache(clips_dict, label_history, inclusion_value)

        error_cost = _eval_cached_models(clips_dict, current_good_votes, current_bad_votes, inclusion_value)

        stability = [step["stability"] for step in _cached_steps if step["stability"] is not None]

        diversity = calculate_diversity_level_over_time(clips_dict, label_history)

    return {
        "error_cost_over_time": error_cost,
        "stability_over_time": stability,
        "diversity_level_over_time": diversity,
        "total_labels": len(current_good_votes) + len(current_bad_votes),
        "total_medias": len(clips_dict),
    }
