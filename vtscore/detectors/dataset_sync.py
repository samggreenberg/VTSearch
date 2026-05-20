"""Keep a loaded detector's cid-keyed vote state aligned with the active dataset.

A ``DetectorContext``'s ``good_votes`` / ``bad_votes`` (and related per-vote
state) are dicts keyed by integer media id — but media ids are only meaningful
within a single dataset.  When the same detector is used across two datasets
(e.g. trained on dataset A, then reused with dataset B), the in-memory cids
from A leak into B's space, so unrelated B-medias whose ids happen to coincide
with A's voted ids appear as voted in the labeling UI.  Worse, the next vote
in B feeds those stale cids into ``sync_labels_to_loaded_detector``, which
silently writes corrupt entries back to the on-disk labelset.

The on-disk labelset is the canonical source of truth (dataset-agnostic — its
elements are keyed by origin / md5).  This module re-derives the cid dicts
from that labelset whenever the active dataset has changed.

The companion helper :func:`validated_vote_snapshot` extends that defense to
read/write paths: it runs the rehydrate and then atomically copies the active
dataset's medias and the active detector's vote dicts under a single
``_state_lock`` acquisition, so a concurrent rehydrate on the same detector
against a different dataset can't slip in between the rehydrate and the
composition.
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
        # Detector file missing — still mark the dataset transition so we
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
        return

    from vtscore.datasets.labelset import LabelSet

    with _state_lock:
        # Re-check inside the lock — another request may have rehydrated us.
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


class VoteSnapshot(NamedTuple):
    """Atomic, internally-consistent (dataset, detector) state snapshot.

    Holds shallow copies of the active dataset's ``medias`` and the active
    detector's cid-keyed vote dicts, all captured under a single
    ``_state_lock`` acquisition.  Composing the four fields is safe because
    they were taken together — subsequent mutations to the live contexts
    cannot leak into a held snapshot.

    ``safe`` reports whether the vote dicts could be proved to be keyed in
    the snapshot's medias' cid space.  When ``safe`` is False, ``good_votes``
    / ``bad_votes`` / ``vote_region_boxes`` are returned empty (so read paths
    that just compose with ``medias`` degrade to an empty result instead of
    leaking cross-dataset cids).  Write paths that would replace on-disk
    state must check ``safe`` and skip the write — otherwise an empty
    composition would erase legitimate labels for the active dataset.
    """

    medias: dict[int, dict[str, Any]]
    good_votes: dict[int, None]
    bad_votes: dict[int, None]
    vote_region_boxes: dict[int, tuple[float, float, float, float]]
    safe: bool


def validated_vote_snapshot() -> VoteSnapshot:
    """Return an atomic, internally-consistent (dataset, detector) snapshot.

    Runs :func:`ensure_votes_match_active_dataset` and then takes shallow
    copies of medias + vote dicts under ``_state_lock`` so the returned vote
    dicts are guaranteed to have been derived against the same dataset whose
    medias are in the returned snap.  Callers should operate on the copies;
    the live contexts can be mutated by other requests in the meantime.

    Refuses to compose when the snapshot cannot be made safe.  Returns
    ``safe=False`` (with empty vote dicts) in any of these cases (all of
    which would otherwise allow cross-dataset cid leakage):

    * A concurrent request on the same detector rehydrated against a
      different dataset between our rehydrate call and this lock acquisition
      (the H14 concurrency race).
    * The detector has no registry entry / on-disk file, so the rehydrate
      bailed early and left stale cids from a previous dataset in place.
    * The active dataset has empty id (no ``X-Dataset-Id`` header on the
      request, or an id that refers to an unloaded dataset) while the
      detector's vote state was previously stamped for a real dataset.

    The ``medias`` field is always populated from whatever the active
    dataset context resolves to (empty when the header is missing).  When no
    detector is loaded at all, the snapshot is considered safe (the empty
    fallback detector has empty votes by construction, so there's nothing
    to leak).
    """
    from vtscore.state.core import (
        _state_lock,
        get_active_context,
        get_active_detector_context,
    )

    ensure_votes_match_active_dataset()
    with _state_lock:
        ds_ctx = get_active_context()
        det_ctx = get_active_detector_context()
        snap = dict(ds_ctx.medias)
        # Compose votes only when we can prove they're keyed in the active
        # dataset's cid space.  ``votes_dataset_id == ds_ctx.dataset_id`` is
        # the post-rehydrate invariant; if it doesn't hold here, something
        # bypassed or raced past the rehydrate and the cid dicts are not
        # safe to pair with ``snap``.
        if det_ctx.detector_id and det_ctx.votes_dataset_id != ds_ctx.dataset_id:
            return VoteSnapshot(snap, {}, {}, {}, safe=False)
        return VoteSnapshot(
            snap,
            dict(det_ctx.good_votes),
            dict(det_ctx.bad_votes),
            dict(det_ctx.vote_region_boxes),
            safe=True,
        )
