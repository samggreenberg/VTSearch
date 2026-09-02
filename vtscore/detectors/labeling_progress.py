"""Labeling-session analyzer: per-step model cache and stopping-condition metrics.

Caches trained MLPs and stability metrics per labelling step so that
repeated queries (the progress button, the auto-indicator) never retrain
models that have already been computed.

Cache shape
-----------
All cache state lives in :class:`_ProgressCache` instances held in ``_caches``,
an LRU-bounded map keyed by ``(dataset_id, detector_id)``, plus the
per-*dataset* :class:`_MonitoredPool` tensors those caches share.  Every entry
point opens with ``cache = _active_cache()`` (or ``_ensure_cache``, which
returns one) under ``_progress_lock`` and works through that object.  Keying by
the pair is a correctness requirement, not a convenience: without it one
detector's history gets replayed onto another's accumulated label sets, and one
detector's models get served as another's Smart / Stable indicators (issue
#2914).

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
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from vtscore.concurrency.async_jobs import check_job_cancelled
from vtscore.embedding.media_vectors import media_embedding
from vtscore.training.mlp import LINEAR_SVM_HEAD, train_model
from vtscore.training.thresholds import conformal_threshold

if TYPE_CHECKING:
    import torch.nn as nn

# ---------------------------------------------------------------------------
# Per-(dataset, detector) cache
# ---------------------------------------------------------------------------


@dataclass
class _MonitoredPool:
    """The fixed pool the per-step stability forward pass scores.

    Built once per *dataset* and reused by every step of every cache over that
    dataset.  ``X`` is a device-resident float tensor of the pool's embeddings;
    ``id_set`` is the id set for O(labels) unlabeled counting.  In-memory only -
    never serialized (see the "No Persisted Vectors" rule).

    Keyed by ``dataset_id`` rather than by the full cache key because the pool
    is a pure function of ``clips_dict``: two detectors over the same dataset
    would build byte-identical tensors, and the tensor is by far the largest
    thing this module holds (up to ``_STABILITY_MAX_SAMPLES`` x embedding-dim
    floats).  Sharing it is what keeps caching several pairs at once from
    multiplying peak memory.
    """

    ids: list[int]
    X: Any  # torch.Tensor | None
    id_set: set[int]


@dataclass
class _ProgressCache:
    """Everything cached for one ``(dataset_id, detector_id)`` pair.

    Each entry in ``steps`` corresponds to one index in ``label_history`` and
    stores the model, threshold, label sets, and stability result for that step.
    ``good_ids`` / ``bad_ids`` track the running label sets so the next step
    only needs to apply a single delta.

    Every input a cache is built from is resolved *per request* from the
    ``X-Dataset-Id`` / ``X-Detector-Id`` headers (``label_history``,
    ``good_votes`` / ``bad_votes`` via the detector context; ``clips_dict`` and
    the coverage atlas via the dataset context).  Multiple detectors stay loaded
    at once and the UI switches between them freely - re-selecting an *already
    loaded* detector never goes through ``register_detector_context`` /
    ``unregister_detector_context``, so those clears do not cover the switch.
    Keying the cache by the pair is what stops one detector's history being
    replayed on top of another's accumulated label sets, or one detector's
    models being served as another's Smart / Stable indicators (issue #2914).
    """

    key: tuple[str, str]

    #: Inclusion value every cached step was trained under.  A different value
    #: rebuilds the cache in place (see :func:`_ensure_cache`).
    inclusion: Optional[int] = None

    steps: list[dict[str, Any]] = field(default_factory=list)
    good_ids: set[int] = field(default_factory=set)
    bad_ids: set[int] = field(default_factory=set)
    prev_predictions: Optional[dict[int, int]] = None
    coverage_atlas: Any = None  # CoverageAtlas | None

    #: Last fully-computed ``/api/labeling-status`` payload (minus the transient
    #: ``stale`` flag).  ``compute_labeling_status`` refreshes it on every full
    #: compute; the route returns it immediately to pollers while a background
    #: worker advances the per-step cache, so the 2 s poll never blocks on an
    #: MLP retrain (issue #2397).
    status_snapshot: Optional[dict[str, Any]] = None

    #: Live models injected by ``train_and_score`` during sorting.  Keyed by
    #: ``(frozenset(good_ids), frozenset(bad_ids))`` so that ``_ensure_cache``
    #: can look up the actual model that was used at each label step instead of
    #: retraining from scratch.  Per-pair because the lookup is by labelset
    #: alone, so a model must not outlive the detector it was trained for.
    live_models: dict[tuple[frozenset[int], frozenset[int]], tuple[Any, float]] = field(default_factory=dict)

    def reset(self) -> None:
        """Drop everything derived from labels, keeping the pair identity.

        Used by the in-place rebuild (an inclusion change) that keeps the cache
        bound to the same pair.  Callers that want the cache gone entirely
        should use :func:`clear_progress_cache`.
        """
        self.steps.clear()
        self.good_ids.clear()
        self.bad_ids.clear()
        self.prev_predictions = None
        self.inclusion = None
        self.coverage_atlas = None
        self.live_models.clear()
        # Drop the status snapshot too: it belonged to the just-cleared
        # labelset and would otherwise be served (stale) for the rebuild until
        # its first background refresh lands.
        self.status_snapshot = None


# Caches keyed by ``(dataset_id, detector_id)``, most-recently-used last.
# Bounded so that a session cycling through many detectors cannot grow without
# limit; the LRU victim is simply rebuilt on demand if it is selected again.
_caches: OrderedDict[tuple[str, str], _ProgressCache] = OrderedDict()

# Stability pools keyed by ``dataset_id``, shared by every cache over that
# dataset (see :class:`_MonitoredPool`).  Pruned whenever the last cache
# referencing a dataset goes away.
_monitored_pools: dict[str, _MonitoredPool] = {}

# How many ``(dataset, detector)`` pairs stay warm at once.  Small on purpose:
# each cache holds one trained MLP per label-history step plus a
# ``prev_predictions`` map over the monitored pool, so the point is to keep an
# A-to-B-and-back detector toggle from throwing away work, not to cache
# everything a long session ever touched.
_MAX_CACHED_PAIRS = 3

# Reentrant lock protecting ``_caches``, ``_monitored_pools``, and every field
# of every cache in them.  RLock is used because public functions call
# _ensure_cache which may call ``_ProgressCache.reset`` internally (inclusion
# change) while already holding the lock.
_progress_lock = threading.RLock()

# Upper bound on the number of items the per-step stability forward pass
# evaluates.  Advancing a cache (from the ``/api/labeling-status``
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


def _active_cache_key() -> tuple[str, str]:
    """Return ``(dataset_id, detector_id)`` for the current execution context.

    Read *without* acquiring ``_state_lock``: this runs under ``_progress_lock``
    (see the lock-ordering note at module top), which forbids taking
    ``_state_lock`` while held.  Neither ``get_active_context()`` nor
    ``get_active_detector_context()`` takes a lock - both resolve from the
    request-scoped resolver or a thread-local - so the read is safe.  In the
    ``/api/labeling-status`` background worker both contexts are bound
    thread-locally by ``JobManager``, so the worker resolves the same key the
    request thread did.

    Falls back to ``("", "")`` when resolution fails (e.g. a request naming an
    unloaded detector, whose resolver raises); such a caller cannot reach a
    usable ``label_history`` anyway.  That fallback is a key like any other, so
    two such callers share one (empty) cache rather than corrupting a real
    detector's.
    """
    try:
        from vtscore.state.core import get_active_context, get_active_detector_context  # noqa: PLC0415

        return (get_active_context().dataset_id, get_active_detector_context().detector_id)
    except Exception:
        return ("", "")


def _prune_monitored_pools() -> None:
    """Drop stability pools no live cache refers to any more."""
    live = {key[0] for key in _caches}
    for dataset_id in [d for d in _monitored_pools if d not in live]:
        del _monitored_pools[dataset_id]


def _active_cache() -> _ProgressCache:
    """Return the cache for the active ``(dataset, detector)`` pair.

    Creates it on first use and marks it most-recently-used, evicting the LRU
    victim once more than :data:`_MAX_CACHED_PAIRS` pairs are warm.  Must be
    called with ``_progress_lock`` held.

    This is what makes the old identity-stamp invariant structural.  Cache
    state used to live in module globals guarded by a ``_bind_cache_identity()``
    call that every entry point had to remember to make first, or it would
    serve detector A's models as detector B's (issue #2914) - an invariant
    nothing enforced.  Now the only way to reach cache state at all is through
    the key, so a new entry point cannot forget.
    """
    key = _active_cache_key()
    cache = _caches.get(key)
    if cache is None:
        cache = _ProgressCache(key=key)
        _caches[key] = cache
    _caches.move_to_end(key)
    while len(_caches) > _MAX_CACHED_PAIRS:
        _caches.popitem(last=False)
    _prune_monitored_pools()
    return cache


def clear_progress_cache() -> None:
    """Drop every cached pair's progress data.

    Must be called whenever votes are cleared, medias change, or inclusion
    is altered so that stale models are not reused.

    Deliberately global rather than scoped to the active pair: the callers
    (``clear_votes``, ``clear_medias``, ``set_inclusion``,
    ``register_detector_context`` / ``unregister_detector_context``) each
    invalidate *at least* the active pair, and some - a dataset's medias
    changing, the global inclusion knob moving - invalidate every pair over
    that dataset or every pair outright.  Clearing everything is the
    conservative reading and costs only a rebuild.  What the per-pair keying
    buys is the path that does *not* come through here: switching between two
    already-loaded detectors, which no longer throws either one's work away.
    """
    with _progress_lock:
        _caches.clear()
        _monitored_pools.clear()


def invalidate_progress_cache_from(media_id: int) -> None:
    """Truncate the active pair's progress cache to just before *media_id* first appeared.

    Called when a vote switches polarity (good→bad or bad→good).  Steps
    before the media was first labeled are still valid - their models never
    included this media in training data.  Only steps from the first
    appearance onward are discarded so they can be retrained and their
    stability/evaluation metrics recomputed.

    Scoped to the active pair: a polarity flip on one detector says nothing
    about another's cache.
    """
    with _progress_lock:
        cache = _active_cache()

        # Find the first cached step that includes media_id in its training data.
        truncate_at = None
        for i, step in enumerate(cache.steps):
            if media_id in step["good_ids"] or media_id in step["bad_ids"]:
                truncate_at = i
                break

        if truncate_at is None:
            # Media never appeared in any cached step.  Still need to clear
            # live models - they may have been injected by learned-sort
            # without building the progress cache.
            cache.live_models.clear()
            return

        # Keep steps [0, truncate_at); discard the rest.
        del cache.steps[truncate_at:]

        # Restore the running ID sets to the surviving prefix's final state.
        cache.good_ids.clear()
        cache.bad_ids.clear()
        if cache.steps:
            last = cache.steps[-1]
            cache.good_ids.update(last["good_ids"])
            cache.bad_ids.update(last["bad_ids"])
        else:
            # truncate_at == 0: media was present from the very first step, so
            # the whole prefix is gone and no label survives.  No cached step
            # remains to source the Smart / Stable indicators from, so drop the
            # stale snapshot (parity with the old step-0 full-clear path).
            cache.status_snapshot = None

        # Reset the stability prediction chain - it will restart from the
        # truncation point when _ensure_cache replays the remaining history.
        cache.prev_predictions = None

        # Clear live models - some may have been trained with the old label.
        cache.live_models.clear()

        # Rewind the coverage-atlas overlay and replay the surviving labels
        # rather than nulling the atlas (which would force a full hierarchical
        # k-means rebuild on the next /api/labeling-status poll, starving the
        # request pool at scale).  The structure is unchanged - only labels
        # moved - so the atlas object identity survives the invalidate.
        if cache.coverage_atlas is not None:
            cache.coverage_atlas.reset_labeled()
            for mid in cache.good_ids | cache.bad_ids:
                if mid in cache.coverage_atlas.vector_to_leaf:
                    cache.coverage_atlas.label(mid, good=mid in cache.good_ids)


def inject_live_model(
    good_votes: dict[int, None],
    bad_votes: dict[int, None],
    model: nn.Sequential,
    threshold: float,
) -> None:
    """Register a live model from ``train_and_score`` for progress-cache reuse.

    Called by the learned-sort route after each live training run.  The model
    is stored on the active pair's cache, keyed by its label set, so
    ``_ensure_cache`` can look it up instead of retraining from scratch.
    """
    key = (frozenset(good_votes), frozenset(bad_votes))
    with _progress_lock:
        _active_cache().live_models[key] = (model, threshold)


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

    from vtscore.coverage.atlas import CoverageAtlas, auto_max_depth  # noqa: PLC0415

    # Cap the depth exactly as every other build site does
    # (``build_coverage_atlas`` / ``build_coverage_atlas_for_context``).
    # Omitting it left this fallback on ``COVERAGE_ATLAS_MAX_DEPTH``, so the
    # atlas built here was *deeper* - and cost many more k-means fits - than
    # the context atlas it stands in for.  That is the whole cost of a cold
    # progress-cache build on a dataset large enough to skip the load-time
    # atlas build, and it runs under ``_progress_lock``.
    return CoverageAtlas(vectors, k=3, max_depth=auto_max_depth(len(vectors), k=3))


def _apply_label_event(cache: _ProgressCache, media_id: int, label: str) -> bool:
    """Update *cache*'s running good/bad ID sets for one label event.

    Returns ``True`` if *media_id* was already labeled before this event.
    """
    was_labeled = media_id in cache.good_ids or media_id in cache.bad_ids
    if label == "unlabel":
        cache.good_ids.discard(media_id)
        cache.bad_ids.discard(media_id)
    elif label == "good":
        cache.bad_ids.discard(media_id)
        cache.good_ids.add(media_id)
    else:
        cache.good_ids.discard(media_id)
        cache.bad_ids.add(media_id)
    return was_labeled


def _sync_coverage_atlas(
    cache: _ProgressCache, media_id: int, label: str, was_labeled: bool
) -> Optional[dict[str, Any]]:
    """Mirror a label event onto the coverage atlas and return level info."""
    atlas = cache.coverage_atlas
    if atlas is None:
        return None
    if label == "unlabel":
        # Only unlabel on the atlas when the item is no longer labeled at all
        # (guards against good→bad re-labels going through "unlabel").
        if was_labeled and media_id not in cache.good_ids and media_id not in cache.bad_ids:
            if media_id in atlas.vector_to_leaf:
                atlas.unlabel(media_id)
    else:
        if media_id in atlas.vector_to_leaf:
            atlas.label(media_id, good=label == "good")
    return {
        "num_labels": len(cache.good_ids) + len(cache.bad_ids),
        "diversity_level": atlas.coverage_level(),
        "depth": atlas.total_nodes,
    }


def _collect_training_data(
    cache: _ProgressCache,
    clips_dict: dict[int, dict[str, Any]],
) -> tuple[list[np.ndarray], list[float]]:
    """Gather embeddings and labels from *cache*'s running good/bad ID sets."""
    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    for cid in cache.good_ids:
        if cid in clips_dict and media_embedding(clips_dict[cid]) is not None:
            X_list.append(media_embedding(clips_dict[cid]))
            y_list.append(1.0)
    for cid in cache.bad_ids:
        if cid in clips_dict and media_embedding(clips_dict[cid]) is not None:
            X_list.append(media_embedding(clips_dict[cid]))
            y_list.append(0.0)
    return X_list, y_list


def _monitored_pool(
    dataset_id: str,
    clips_dict: dict[int, dict[str, Any]],
    all_media_ids: list[int],
) -> _MonitoredPool:
    """Return the fixed pool the stability pass scores for *dataset_id*.

    The pool is the embeddable subset of *all_media_ids*, bounded to a
    deterministic seeded sample of ``_STABILITY_MAX_SAMPLES``.  Sampling the
    full eligible pool (rather than the per-step unlabeled set) keeps the
    monitored ids stable across steps, so the flip comparison against
    ``cache.prev_predictions`` stays over a consistent id set; the resulting
    flip *rate* is an unbiased estimate of the true rate.

    Built once per dataset and memoised.  It used to be rebuilt inside every
    step - an O(N x D) numpy materialisation per label-history step, which
    dominated the cost of advancing the cache.  The pool depends only on
    *clips_dict*, which cannot change without a ``clear_progress_cache()``, so
    one build per dataset is sound - and is why the memo is keyed by dataset
    rather than by the full ``(dataset, detector)`` pair.
    """
    pool = _monitored_pools.get(dataset_id)
    if pool is not None:
        return pool

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
        X = torch.tensor(embs, dtype=torch.float32).to(get_torch_device())
    else:
        X = None

    pool = _MonitoredPool(ids=eligible, X=X, id_set=set(eligible))
    _monitored_pools[dataset_id] = pool
    return pool


def _compute_step_stability(
    cache: _ProgressCache,
    model: nn.Sequential,
    threshold: float,
    clips_dict: dict[int, dict[str, Any]],
    all_media_ids: list[int],
    t: int,
    num_labels: int,
) -> Optional[dict[str, Any]]:
    """Compute prediction stability by comparing to the previous step's predictions."""
    import torch  # noqa: PLC0415

    pool = _monitored_pool(cache.key[0], clips_dict, all_media_ids)

    labeled_ids = cache.good_ids | cache.bad_ids
    # Labels are few relative to the pool, so count the overlap from the
    # labelset rather than rescanning the pool.
    num_unlabeled = len(pool.ids) - sum(1 for cid in labeled_ids if cid in pool.id_set)

    if num_unlabeled <= 0 or pool.X is None:
        return {"time_index": t, "num_labels": num_labels, "num_flips": 0, "num_unlabeled": 0}

    # Score the whole monitored pool in one pass and drop the currently-labeled
    # ids afterwards.  Scoring the handful of extra (labeled) rows is far
    # cheaper than re-materialising a per-step tensor of the unlabeled subset.
    with torch.no_grad():
        X_in = pool.X.to(next(model.parameters()).device)
        scores_unl = torch.sigmoid(model(X_in)).squeeze(1).cpu().tolist()

    predictions: dict[int, int] = {
        cid: 1 if score >= threshold else 0
        for cid, score in zip(pool.ids, scores_unl, strict=True)
        if cid not in labeled_ids
    }

    stability: Optional[dict[str, Any]] = None
    if cache.prev_predictions is not None:
        prev = cache.prev_predictions
        num_flips = sum(1 for cid in predictions.keys() & prev.keys() if predictions[cid] != prev[cid])
        stability = {
            "time_index": t,
            "num_labels": num_labels,
            "num_flips": num_flips,
            "num_unlabeled": num_unlabeled,
        }
    # else: no prior predictions to compare - leave stability as None.

    cache.prev_predictions = predictions
    return stability


def _train_step(
    cache: _ProgressCache,
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
    if not cache.good_ids or not cache.bad_ids:
        # No model possible - clear prediction baseline so the first step
        # after regaining a model doesn't produce a misleading flip count.
        cache.prev_predictions = None
        return None, None, None

    X_list, y_list = _collect_training_data(cache, clips_dict)
    if len(X_list) < 2:
        return None, None, None

    import torch  # noqa: PLC0415

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)

    # Linear SVM head, matching the production detector this previews.
    model = train_model(X, y, X.shape[1], hidden_dim=LINEAR_SVM_HEAD)

    with torch.no_grad():
        X_dev = X.to(next(model.parameters()).device)
        scores = torch.sigmoid(model(X_dev)).squeeze(1).cpu().tolist()
    # Training-set scores, not held-out ones: this cache only needs a rough
    # per-step cutoff for the stability curve, so the optimistic (tighter)
    # band from in-sample quantiles is acceptable here.
    threshold = conformal_threshold(scores, y_list, inclusion_value)

    stability = _compute_step_stability(cache, model, threshold, clips_dict, all_media_ids, t, num_labels)
    return model, threshold, stability


def _resolve_step_model(
    cache: _ProgressCache,
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
    otherwise reuses a live model injected by ``train_and_score`` for this
    exact label set, or trains a fresh one.  Returns ``(model, threshold,
    stability)``.
    """
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

    # Check whether train_and_score already produced a model for
    # this exact label set during live sorting.  If so, reuse it
    # (correct cross-calibrated threshold, zero compute cost).
    live_key = (frozenset(cache.good_ids), frozenset(cache.bad_ids))
    live = cache.live_models.get(live_key)
    if live is not None:
        model, threshold = live
        stability = _compute_step_stability(cache, model, threshold, clips_dict, all_media_ids, t, num_labels)
        return model, threshold, stability

    return _train_step(cache, clips_dict, all_media_ids, t, num_labels, inclusion_value)


def _ensure_cache(
    clips_dict: dict[int, dict[str, Any]],
    label_history: list[tuple[int, str, float]],
    inclusion_value: int,
) -> _ProgressCache:
    """Bring the active pair's cache up to date with *label_history*.

    Only computes steps that are not yet cached.  If *inclusion_value*
    differs from the value used for existing cache entries the entire cache
    is rebuilt.  Returns the cache, so callers never have to re-resolve it.

    Must be called with ``_progress_lock`` held.
    """
    cache = _active_cache()

    if cache.inclusion is not None and cache.inclusion != inclusion_value:
        # Same pair, different inclusion: rebuild in place.  The monitored pool
        # is a pure function of ``clips_dict`` and so survives, but medias may
        # have changed under us, so drop it alongside the rest for parity with
        # the full clear.
        cache.reset()
        _monitored_pools.pop(cache.key[0], None)

    if cache.inclusion is None:
        cache.inclusion = inclusion_value

    start = len(cache.steps)
    if start >= len(label_history):
        return cache  # already up to date

    all_media_ids = sorted(clips_dict.keys())

    if cache.coverage_atlas is None:
        cache.coverage_atlas = _build_coverage_atlas(clips_dict)
        # A freshly built (or cloned) atlas starts with an empty label overlay.
        # Defensively seed it with any labels already accumulated in the
        # running ID sets so coverage_level() is correct before the history
        # replay below runs; normally these sets are empty at first build
        # (invalidate rewinds and replays its atlas in place rather than
        # nulling it, so this branch no longer runs mid-history).
        if cache.coverage_atlas is not None:
            for mid in cache.good_ids | cache.bad_ids:
                if mid in cache.coverage_atlas.vector_to_leaf:
                    cache.coverage_atlas.label(mid, good=mid in cache.good_ids)

    for t in range(start, len(label_history)):
        # Each step retrains a model; honour a cancel of the owning eval job
        # here so a long history doesn't run to completion after cancel.  The
        # partially-built cache is a valid prefix (steps 0..t-1), so the next
        # run resumes cleanly from ``len(cache.steps)``.  No-op outside a
        # job (see ``async_jobs.check_job_cancelled``).
        check_job_cancelled()
        media_id, label, _ = label_history[t]

        was_labeled = _apply_label_event(cache, media_id, label)
        diversity_info = _sync_coverage_atlas(cache, media_id, label, was_labeled)

        good_ids = list(cache.good_ids)
        bad_ids = list(cache.bad_ids)
        num_labels = len(good_ids) + len(bad_ids)

        prev = cache.steps[-1] if cache.steps else None
        model, threshold, stability = _resolve_step_model(
            cache, clips_dict, all_media_ids, t, num_labels, inclusion_value, good_ids, bad_ids, prev
        )

        cache.steps.append(
            {
                "model": model,
                "threshold": threshold,
                "good_ids": good_ids,
                "bad_ids": bad_ids,
                "stability": stability,
                "diversity": diversity_info,
            }
        )

    return cache


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
    cache: _ProgressCache,
    clips_dict: dict[int, dict[str, Any]],
    current_good_votes: dict[int, None],
    current_bad_votes: dict[int, None],
    inclusion_value: int,
    start: int = 0,
    end: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Score *cache*'s models against the current labelset (forward passes only).

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
        end = len(cache.steps)

    results: list[dict[str, Any]] = []
    for t in range(start, end):
        step = cache.steps[t]
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
        cache = _ensure_cache(clips_dict, label_history, inclusion_value)

        step = cache.steps[time_index]
        return step["model"], step["threshold"], step["good_ids"], step["bad_ids"]


def calculate_error_cost_over_time(
    clips_dict: dict[int, dict[str, Any]],
    label_history: list[tuple[int, str, float]],
    current_good_votes: dict[int, None],
    current_bad_votes: dict[int, None],
    inclusion_value: int = 0,
) -> list[dict[str, Any]]:
    """Calculate classification error cost at each labelling step.

    Uses cached models - no retraining.
    """
    with _progress_lock:
        cache = _ensure_cache(clips_dict, label_history, inclusion_value)
        return _eval_cached_models(cache, clips_dict, current_good_votes, current_bad_votes, inclusion_value)


def calculate_prediction_stability_over_time(
    clips_dict: dict[int, dict[str, Any]],
    label_history: list[tuple[int, str, float]],
    inclusion_value: int = 0,
) -> list[dict[str, Any]]:
    """Return cached prediction-stability metrics for every step."""
    with _progress_lock:
        cache = _ensure_cache(clips_dict, label_history, inclusion_value)
        return [step["stability"] for step in cache.steps if step["stability"] is not None]


def _compute_smart_status(
    cache: _ProgressCache,
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

    n = len(cache.steps)
    if n < 3:
        return {"status": "yellow", "reason": "Not enough label history steps to assess trend."}

    start_idx = max(0, n - 10)
    recent_entries = _eval_cached_models(
        cache, clips_dict, current_good_votes, current_bad_votes, inclusion_value, start_idx, n
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
    cache: _ProgressCache,
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

    stability = [step["stability"] for step in cache.steps if step["stability"] is not None]

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
    result is stashed in the pair's ``status_snapshot`` so the
    ``/api/labeling-status`` route can serve it immediately (marked ``stale``)
    on subsequent polls while a background worker calls this to advance the
    cache off the request thread (issue #2397).
    """
    good = len(current_good_votes)
    bad = len(current_bad_votes)
    total = good + bad

    with _progress_lock:
        cache = _ensure_cache(clips_dict, label_history, inclusion_value)

        smart = _compute_smart_status(
            cache, clips_dict, label_history, current_good_votes, current_bad_votes, inclusion_value, good, bad, total
        )
        stable = _compute_stable_status(cache, good, bad, total)

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
    # ``stale`` flag) doesn't retroactively corrupt the snapshot.  The lock was
    # released for the (settings-reading) Span computation above, so another
    # thread may have dropped this pair's cache meanwhile (a vote clear, a
    # detector unload, an LRU eviction); the identity check republishes only
    # onto the very object these indicators were computed from, never onto a
    # successor that has been rebuilt or belongs to someone else.
    with _progress_lock:
        if _caches.get(cache.key) is cache:
            cache.status_snapshot = dict(result)
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
        if not is_status_cache_fresh(label_history, inclusion_value):
            return [], False

        cache = _active_cache()
        if metric == "smart":
            data = _eval_cached_models(cache, clips_dict, current_good_votes, current_bad_votes, inclusion_value)
        elif metric == "stable":
            data = [step["stability"] for step in cache.steps if step["stability"] is not None]
        else:
            data = [step["diversity"] for step in cache.steps if step.get("diversity") is not None]
        return data, True
    finally:
        _progress_lock.release()


def is_status_cache_fresh(label_history: list[tuple[int, str, float]], inclusion_value: int) -> bool:
    """Return ``True`` when the per-step cache already covers *label_history*.

    A fresh cache means ``compute_labeling_status`` will not retrain any model,
    so the route can compute the status inline instead of deferring to a
    background worker.  A mismatched ``inclusion_value`` counts as not-fresh
    because :func:`_ensure_cache` would rebuild the cache from scratch.  The
    length comparison is against the *active pair's* cache, so another
    detector's longer history can never be read as covering this one's.
    """
    with _progress_lock:
        cache = _active_cache()
        if cache.inclusion is not None and cache.inclusion != inclusion_value:
            return False
        return len(cache.steps) >= len(label_history)


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
        # Reading the snapshot off the *active pair's* cache is what stops one
        # detector's indicators being handed to another; a detector with no
        # snapshot of its own shows the "computing" placeholder instead.
        snapshot = _active_cache().status_snapshot
        if snapshot is not None:
            status["smart"] = dict(snapshot["smart"])
            status["stable"] = dict(snapshot["stable"])
    return status


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
        cache = _ensure_cache(clips_dict, label_history, inclusion_value)
        return [step["diversity"] for step in cache.steps if step.get("diversity") is not None]


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
        cache = _ensure_cache(clips_dict, label_history, inclusion_value)

        error_cost = _eval_cached_models(cache, clips_dict, current_good_votes, current_bad_votes, inclusion_value)

        stability = [step["stability"] for step in cache.steps if step["stability"] is not None]

        diversity = calculate_diversity_level_over_time(clips_dict, label_history, inclusion_value)

    return {
        "error_cost_over_time": error_cost,
        "stability_over_time": stability,
        "diversity_level_over_time": diversity,
        "total_labels": len(current_good_votes) + len(current_bad_votes),
        "total_medias": len(clips_dict),
    }
