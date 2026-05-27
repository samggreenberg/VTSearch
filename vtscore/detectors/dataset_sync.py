"""Keep a loaded detector's cid-keyed vote state aligned with the active dataset.

A ``DetectorContext``'s ``good_votes`` / ``bad_votes`` (and related per-vote
state) are dicts keyed by integer media id; media ids are only meaningful
within a single dataset.  When the same detector is used across two datasets
(e.g. trained on dataset A, then reused with dataset B), the in-memory cids
from A leak into B's space, so unrelated B-medias whose ids happen to coincide
with A's voted ids appear as voted in the labeling UI.  Worse, the next vote
in B feeds those stale cids into ``sync_labels_to_loaded_detector``, which
silently writes corrupt entries back to the on-disk labelset.

The on-disk labelset is the canonical source of truth (dataset-agnostic; its
elements are keyed by origin / md5).  This module re-derives the cid dicts
from that labelset whenever the active dataset has changed.

The companion helper :func:`validated_vote_snapshot` extends that defense to
read/write paths: it atomically copies the active dataset's medias and the
active detector's vote dicts under a single ``_state_lock`` acquisition, so
a concurrent rehydrate on the same detector against a different dataset
can't slip in between the (request-boundary) rehydrate and the composition.
"""

from __future__ import annotations

from typing import Any, NamedTuple


def _detector_file_mtime(path) -> float:
    """Return the mtime of ``path`` in seconds, or 0.0 if it doesn't exist."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def ensure_votes_match_active_dataset() -> None:
    """Rehydrate the active detector's cid-keyed vote state against the active dataset.

    No-op unless all of these hold:

    * a detector is loaded,
    * a dataset is loaded (and its id is non-empty),
    * the detector has a registry entry that points at a real on-disk file
      (so we have a labelset to rehydrate from), and
    * either the detector's recorded ``votes_dataset_id`` differs from the
      active dataset id, or the on-disk detector file's mtime has changed
      since the cached labelset was loaded.

    When triggered, clears the per-dataset detector state (good_votes,
    bad_votes, label_history, vote_click_times, click_counter,
    last_learned_scores, find_initial_labels, training_medias) and replays
    ``restore_labels_from_detector`` against the active dataset's medias,
    then stamps ``votes_dataset_id`` and caches the parsed labelset +
    file mtime on the detector context so subsequent requests within the
    same (dataset, file-mtime) tuple are no-ops.
    """
    from vtscore.state.core import (
        _state_lock,
        get_active_context,
        get_active_detector_context,
    )

    det_ctx = get_active_detector_context()
    if not det_ctx.detector_id:
        return
    ds_ctx = get_active_context()
    if not ds_ctx.dataset_id:
        # No active dataset; preserve whatever the detector last saw so a
        # request that happens to omit the dataset header doesn't wipe state.
        return

    from vtscore.detectors.registry import get_detector
    from vtscore.detectors.store import _detector_path, _read_detector
    from vtscore.detectors.label_restoration import restore_labels_from_detector

    entry = get_detector(det_ctx.detector_id)
    if not entry or not entry.get("name"):
        # No registry entry (typical in unit tests that build DetectorContext
        # directly); nothing to rehydrate from.
        return

    path = _detector_path(entry["name"])
    current_mtime = _detector_file_mtime(path)

    if (
        det_ctx.votes_dataset_id == ds_ctx.dataset_id
        and det_ctx.cached_labelset is not None
        and det_ctx.cached_labelset_mtime == current_mtime
        and current_mtime != 0.0
    ):
        return

    data = _read_detector(path)
    if data is None:
        # Detector file missing. Still mark the dataset transition so we
        # don't keep stale cids around.
        with _state_lock:
            det_ctx.good_votes.clear()
            det_ctx.bad_votes.clear()
            det_ctx.label_history.clear()
            det_ctx.vote_click_times.clear()
            det_ctx.click_counter = 0
            det_ctx.last_learned_scores.clear()
            det_ctx.find_initial_labels.clear()
            det_ctx.training_medias.clear()
            det_ctx.votes_dataset_id = ds_ctx.dataset_id
            det_ctx.cached_labelset = None
            det_ctx.cached_labelset_mtime = 0.0
            det_ctx.cached_labelset_media_type = ""
            if ds_ctx.diversity_tree is not None:
                ds_ctx.diversity_tree.reset_seen()
        return

    from vtscore.datasets.labelset import LabelSet
    from vtscore.state.diversity import resync_diversity_tree_to_detector

    with _state_lock:
        # Re-check inside the lock; another request may have rehydrated us.
        refreshed_mtime = _detector_file_mtime(path)
        if (
            det_ctx.votes_dataset_id == ds_ctx.dataset_id
            and det_ctx.cached_labelset is not None
            and det_ctx.cached_labelset_mtime == refreshed_mtime
            and refreshed_mtime != 0.0
        ):
            return
        det_ctx.good_votes.clear()
        det_ctx.bad_votes.clear()
        det_ctx.label_history.clear()
        det_ctx.vote_click_times.clear()
        det_ctx.click_counter = 0
        det_ctx.last_learned_scores.clear()
        det_ctx.find_initial_labels.clear()
        det_ctx.training_medias.clear()
        restore_labels_from_detector(data)
        det_ctx.votes_dataset_id = ds_ctx.dataset_id
        det_ctx.cached_labelset = LabelSet.from_dict(data.get("labelset") or {})
        det_ctx.cached_labelset_mtime = refreshed_mtime
        det_ctx.cached_labelset_media_type = data.get("media_type", "") or ""
        # The dataset's diversity tree was either built for a previous
        # detector on this dataset (so its ``seen`` set reflects that
        # detector's votes) or built before any votes existed.  Re-derive
        # it against the freshly-restored votes so ``next_sample`` /
        # ``diversity_level`` track the now-active detector.  Restored
        # labels above are applied with ``silent=True`` (see
        # ``label_restoration.py``) and therefore skip the per-vote tree
        # update; this is where the equivalent bulk update lands.
        resync_diversity_tree_to_detector(ds_ctx, det_ctx)


def invalidate_detector_model_on_embedder_mismatch(det_ctx, new_embedder: str) -> bool:
    """Drop *det_ctx*'s cached MLP when *new_embedder* differs.

    ``DetectorContext.model`` is trained against a specific embedding space -
    ``det_ctx.embedder`` records which.  Scoring with a cross-space MLP
    either crashes (different dim → ``nn.Linear`` size-mismatch) or
    silently produces garbage labels (same dim, different space).  When
    the dataset about to be scored uses a different embedder than the one
    the cached MLP was trained on, clear the embedder-tagged scoring
    caches (``model``, ``threshold``, ``last_learned_scores``,
    ``training_medias``, ``calibration_cache``) so the next scoring /
    learned-sort call rebuilds against *new_embedder*.

    Deliberately leaves ``label_embeddings`` and ``embedder`` alone:

    * The per-label embedding cache is invalidated lazily by
      :func:`~vtscore.detectors.labelset_training._maybe_clear_cache_on_embedder_switch`
      inside :func:`~vtscore.detectors.labelset_training.populate_label_embeddings`,
      which is the only consumer.  Clearing it here would just shift the
      reset and risk surprising any caller that reads the cache directly.
    * ``det_ctx.embedder`` is the "what was the last training pass aligned
      with" marker.  Leaving it stamped as the old value lets the load
      endpoint's :func:`~vtsearch.routes.detectors.registry._maybe_start_label_reembed`
      still detect the mismatch and schedule its progress-tracked
      re-embed task.  The marker is restamped to *new_embedder* by the
      next training pass (via
      :func:`~vtscore.detectors.registry.record_detector_embedder`
      in :func:`~vtscore.detectors.labelset_training.populate_label_embeddings`
      and the workflow path).

    Returns ``True`` when invalidation happened, ``False`` otherwise.
    Idempotent: a second call without intervening training simply finds
    ``model`` already cleared and returns ``False``.
    """
    from vtscore.state.core import _state_lock

    if det_ctx is None or not det_ctx.embedder or not new_embedder:
        return False
    if new_embedder == det_ctx.embedder:
        return False
    if det_ctx.model is None:
        # Already invalidated by an earlier request in this dataset
        # transition; nothing to clear.
        return False
    with _state_lock:
        if det_ctx.model is None or new_embedder == det_ctx.embedder:
            return False
        det_ctx.model = None
        det_ctx.threshold = 0.5
        det_ctx.last_learned_scores.clear()
        det_ctx.training_medias.clear()
        det_ctx.calibration_cache = None
    return True


def _embedder_of_active_dataset() -> str:
    """Return the embedder name recorded on the active dataset's medias, or ``""``."""
    from vtscore.state.core import get_active_context

    ds_ctx = get_active_context()
    if not ds_ctx.dataset_id or not ds_ctx.medias:
        return ""
    first = next(iter(ds_ctx.medias.values()), {})
    return first.get("embedder", "") or ""


def ensure_detector_model_matches_active_embedder() -> None:
    """Active-context wrapper for :func:`invalidate_detector_model_on_embedder_mismatch`.

    Called from ``before_request`` so a dataset switch can't leave the
    active detector pointing at an MLP trained against the old embedder
    (silent garbage scores at best, ``nn.Linear`` size-mismatch crash at
    worst).  Vote-rehydrate in :func:`ensure_votes_match_active_dataset`
    handles cid-keyed votes; this handles the embedder-specific caches
    that the on-disk labelset doesn't carry.

    The scoring fast-paths
    (:func:`vtsearch.routes.detectors.scoring._resolve_or_train_detector`
    and the find dispatcher) defensively repeat the check per-detector,
    since autorun / multi-dataset Find iterate detectors that aren't the
    active one.
    """
    from vtscore.state.core import get_active_detector_context

    det_ctx = get_active_detector_context()
    if not det_ctx.detector_id:
        return
    new_embedder = _embedder_of_active_dataset()
    invalidate_detector_model_on_embedder_mismatch(det_ctx, new_embedder)


class VoteSnapshot(NamedTuple):
    """Atomic, internally-consistent (dataset, detector) state snapshot.

    Holds shallow copies of the active dataset's ``medias`` and the active
    detector's cid-keyed vote dicts, all captured under a single
    ``_state_lock`` acquisition.  Composing the four fields is safe because
    they were taken together; subsequent mutations to the live contexts
    cannot leak into a held snapshot.

    ``safe`` reports whether the vote dicts could be proved to be keyed in
    the snapshot's medias' cid space.  When ``safe`` is False, ``good_votes``
    / ``bad_votes`` / ``vote_region_boxes`` are returned empty (so read paths
    that just compose with ``medias`` degrade to an empty result instead of
    leaking cross-dataset cids).  Write paths that would replace on-disk
    state must check ``safe`` and skip the write; otherwise an empty
    composition would erase legitimate labels for the active dataset.
    """

    medias: dict[int, dict[str, Any]]
    good_votes: dict[int, None]
    bad_votes: dict[int, None]
    vote_region_boxes: dict[int, tuple[float, float, float, float]]
    safe: bool


def validated_vote_snapshot() -> VoteSnapshot:
    """Return an atomic, internally-consistent (dataset, detector) snapshot.

    Takes shallow copies of medias + vote dicts under a single ``_state_lock``
    acquisition so the returned vote dicts are guaranteed to be derived
    against the same dataset whose medias are in the returned snap.  Callers
    should operate on the copies; the live contexts can be mutated by other
    requests in the meantime.

    The rehydrate that aligns ``votes_dataset_id`` with the active dataset
    is the responsibility of :func:`ensure_votes_match_active_dataset` in
    ``before_request``; calling it again here would double-clear vote dicts
    that test code or in-flight mutations have already populated.

    Refuses to compose when the snapshot cannot be made safe.  Returns
    ``safe=False`` (with empty vote dicts) when the detector's vote state is
    stamped for a different dataset than the one whose medias the snapshot
    is taken from:

    * A concurrent request on the same detector rehydrated against a
      different dataset between ``before_request`` and this lock acquisition
      (the H14 concurrency race).
    * The detector has no registry entry / on-disk file, so the rehydrate
      bailed early and left stale cids from a previous dataset in place.
    * The active dataset has empty id (no ``X-Dataset-Id`` header on the
      request, or an id that refers to an unloaded dataset) while the
      detector's vote state was previously stamped for a real dataset.

    The ``medias`` field is always populated from whatever the active
    dataset context resolves to (empty when the header is missing).  When no
    detector is loaded at all, or when ``votes_dataset_id`` is empty (no
    rehydrate has ever stamped it, typical for a brand-new detector with
    no votes), the snapshot is considered safe by construction.
    """
    from vtscore.state.core import (
        _state_lock,
        get_active_context,
        get_active_detector_context,
    )

    with _state_lock:
        ds_ctx = get_active_context()
        det_ctx = get_active_detector_context()
        snap = dict(ds_ctx.medias)
        # Compose votes only when we can prove they're keyed in the active
        # dataset's cid space.  Two cases are safe to compose:
        #   1. ``votes_dataset_id == ds_ctx.dataset_id``: the post-rehydrate
        #      invariant holds; the cid dicts are derived against the active
        #      dataset.
        #   2. ``votes_dataset_id == ""``: the detector has never been
        #      stamped against any dataset, which in production means no
        #      votes have ever been cast (vote-casting goes through paths
        #      that stamp the id).  Composing empty vote dicts is harmless,
        #      and short-circuiting here lets save / sync endpoints work
        #      against newly-created detectors that haven't been loaded.
        # The remaining case (non-empty ``votes_dataset_id`` that doesn't
        # match the active dataset) is the race / stale-state scenario and
        # is the only one we refuse to compose for.
        if det_ctx.detector_id and det_ctx.votes_dataset_id and det_ctx.votes_dataset_id != ds_ctx.dataset_id:
            return VoteSnapshot(snap, {}, {}, {}, safe=False)
        return VoteSnapshot(
            snap,
            dict(det_ctx.good_votes),
            dict(det_ctx.bad_votes),
            dict(det_ctx.vote_region_boxes),
            safe=True,
        )
