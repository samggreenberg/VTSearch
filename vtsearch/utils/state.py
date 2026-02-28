"""Global state management for medias and votes."""

from __future__ import annotations

import gc
import json
import threading
from typing import Any

# Reentrant lock protecting all mutable state in this module.
# RLock is used because some public functions call other public functions
# (e.g. clear_all -> clear_medias + clear_votes).
_state_lock = threading.RLock()

# Clips storage: id -> {id, type, duration, file_size, embedding, media_bytes, media_string, ...}
medias: dict[int, dict[str, Any]] = {}

# Optional display-name override for the loaded dataset.  When set, the
# dashboard shows this instead of the name derived from origin info.
_dataset_display_name: str | None = None

# Diversity tree: built from media embeddings after a dataset loads.
# ``None`` until a dataset is loaded and the tree is constructed.
_diversity_tree: Any = None  # DiversityTree | None

# Voting storage (OrderedDict behavior via dict in Python 3.7+)
good_votes: dict[int, None] = {}
bad_votes: dict[int, None] = {}

# Combined label history: [(media_id, label, timestamp), ...]
# Tracks the order of all labels across both categories
label_history: list[tuple[int, str, float]] = []

# Click-time tracking: media_id -> click order (1-indexed).
# Assigned when a vote is cast via the API; labels loaded via import get no entry
# (the frontend treats missing entries as time=-1).
vote_click_times: dict[int, int] = {}
_click_counter: int = 0

# Last learned-sort scores: media_id -> score (float in [0, 1]).
# Updated each time /api/learned-sort completes.
last_learned_scores: dict[int, float] = {}

# Inclusion setting: -10 to +10, default 0.
# ``None`` means "not yet loaded"; on first access the value is read from the
# persisted settings file so that it survives restarts.
inclusion: int | None = None

# Text-sort suggestions: text queries that received a Good vote, most recent last.
textsort_suggestions: list[str] = []

# Autorun detectors: name -> {name, media_type, weights, threshold, created_at}
autorun_detectors: dict[str, dict[str, Any]] = {}

# Autorun extractors: name -> {name, extractor_type, media_type, config, created_at}
autorun_extractors: dict[str, dict[str, Any]] = {}

# Autorun localizers: name -> {name, localizer_type, media_type, config, created_at}
autorun_localizers: dict[str, dict[str, Any]] = {}


def clear_votes() -> None:
    """Clear all votes and the full label history.

    Removes all entries from ``good_votes``, ``bad_votes``, and
    ``label_history`` in place. Does not affect the ``medias`` dict.
    Also clears the progress model cache and click-time / score tracking.
    """
    global _click_counter
    from vtsearch.models.progress import clear_progress_cache

    with _state_lock:
        good_votes.clear()
        bad_votes.clear()
        label_history.clear()
        textsort_suggestions.clear()
        vote_click_times.clear()
        _click_counter = 0
        last_learned_scores.clear()
        clear_progress_cache()


def clear_medias() -> None:
    """Clear all loaded medias from memory.

    Removes all entries from the ``medias`` dict in place. Does not affect
    votes or label history. Also clears the progress model cache since
    cached models reference media embeddings.  Also clears the diversity tree
    and the dataset display name override.
    """
    global _diversity_tree, _dataset_display_name
    from vtsearch.models.progress import clear_progress_cache

    with _state_lock:
        medias.clear()
        _diversity_tree = None
        _dataset_display_name = None
        clear_progress_cache()
    gc.collect()


def clear_all() -> None:
    """Clear all medias, votes, and label history.

    Convenience wrapper that calls :func:`clear_medias` followed by
    :func:`clear_votes`.
    """
    with _state_lock:
        clear_medias()
        clear_votes()


def get_inclusion() -> int:
    """Return the current inclusion setting.

    On first call the value is loaded from the persisted settings file so
    that it survives app restarts.

    Returns:
        An integer in ``[-10, 10]`` representing the inclusion bias. Positive
        values cause the learned sort model to include more items (higher
        recall); negative values cause it to include fewer (higher precision).
    """
    global inclusion
    with _state_lock:
        if inclusion is None:
            from vtsearch import settings

            inclusion = settings.get_inclusion()
        return inclusion


def set_inclusion(value: int) -> None:
    """Set the global inclusion value and persist it to the settings file.

    Also clears the progress model cache since cached models were trained
    with the old inclusion value.

    Args:
        value: New inclusion setting. Should be an integer in ``[-10, 10]``.
            Values outside this range are accepted but may produce unexpected
            results in model training weight calculations.
    """
    global inclusion
    with _state_lock:
        if value != inclusion:
            from vtsearch.models.progress import clear_progress_cache

            clear_progress_cache()
        inclusion = value

    from vtsearch import settings

    settings.set_inclusion(value)


def get_dataset_display_name() -> str | None:
    """Return the current dataset display name override, or ``None``."""
    with _state_lock:
        return _dataset_display_name


def set_dataset_display_name(name: str | None) -> None:
    """Set (or clear) the dataset display name override."""
    global _dataset_display_name
    with _state_lock:
        _dataset_display_name = name


def get_calibrate_count() -> int:
    """Return the number of calibration splits from settings."""
    from vtsearch import settings

    return settings.get_calibrate_count()


def set_calibrate_count(value: int) -> None:
    """Set the calibrate count and persist it to the settings file."""
    from vtsearch import settings

    settings.set_calibrate_count(value)


def get_calibration_fraction() -> float:
    """Return the calibration fraction from settings."""
    from vtsearch import settings

    return settings.get_calibration_fraction()


def set_calibration_fraction(value: float) -> None:
    """Set the calibration fraction and persist it to the settings file."""
    from vtsearch import settings

    settings.set_calibration_fraction(value)


def get_safe_thresholds() -> bool:
    """Return whether safe thresholds blending is enabled.

    Reads the value from the persisted settings file.

    Returns:
        ``True`` if safe thresholds is enabled, ``False`` otherwise.
    """
    from vtsearch import settings

    return settings.get_safe_thresholds()


def set_safe_thresholds(value: bool) -> None:
    """Set the safe thresholds flag and persist it to the settings file.

    Args:
        value: Whether to enable safe threshold blending.
    """
    from vtsearch import settings

    settings.set_safe_thresholds(value)


def assign_click_time(media_id: int) -> int:
    """Assign the next click-time ordinal to a media and return it.

    Each call increments the global counter so click-times are unique and
    monotonically increasing.
    """
    global _click_counter
    with _state_lock:
        _click_counter += 1
        vote_click_times[media_id] = _click_counter
        return _click_counter


def remove_click_time(media_id: int) -> None:
    """Remove the click-time entry for a media (e.g. when unlabelling)."""
    with _state_lock:
        vote_click_times.pop(media_id, None)


def get_vote_click_times() -> dict[int, int]:
    """Return a copy of the click-time mapping."""
    with _state_lock:
        return vote_click_times.copy()


def update_learned_scores(scores: dict[int, float]) -> None:
    """Replace the stored learned-sort scores with *scores*."""
    with _state_lock:
        last_learned_scores.clear()
        last_learned_scores.update(scores)


def get_learned_scores() -> dict[int, float]:
    """Return a copy of the last learned-sort scores."""
    with _state_lock:
        return last_learned_scores.copy()


def add_label_to_history(media_id: int, label: str) -> None:
    """Append a labelling event to the global label history with a timestamp.

    Args:
        media_id: Integer ID of the media that was labelled.
        label: The assigned label; should be ``"good"`` or ``"bad"``.
    """
    import time

    with _state_lock:
        label_history.append((media_id, label, time.time()))


def add_textsort_suggestion(text: str) -> None:
    """Record a text-sort query as a suggested detector/labelset name.

    Duplicates are moved to the end so the most-recently-voted query is last.

    Args:
        text: The text-sort query string to store.
    """
    with _state_lock:
        # Remove existing occurrence so it moves to the end
        try:
            textsort_suggestions.remove(text)
        except ValueError:
            pass
        textsort_suggestions.append(text)


def get_textsort_suggestions() -> list[str]:
    """Return stored text-sort suggestions, most recent last."""
    with _state_lock:
        return list(textsort_suggestions)


def add_autorun_detector(
    name: str,
    media_type: str,
    weights: dict[str, Any] | None = None,
    threshold: float = 0.5,
    *,
    autodetect: bool = False,
    examples: list[dict[str, str]] | None = None,
    num_labels: int = 0,
) -> None:
    """Add or overwrite a named autorun detector in the global store.

    If a detector with the same ``name`` already exists it is replaced.

    Args:
        name: Unique human-readable name for the detector (e.g. ``"dog barks"``).
        media_type: The media type the detector was trained on (``"audio"``,
            ``"video"``, ``"image"``, or ``"paragraph"``).
        weights: Dict mapping layer-parameter names (e.g. ``"0.weight"``) to
            lists of float values, representing the serialised MLP state dict.
            May be ``None`` for an untrained detector stub.
        threshold: Decision boundary score in ``[0, 1]``. Clips scoring at or
            above this value are classified as positive.  Defaults to ``0.5``.
        autodetect: Whether this detector is a "favorite" included when
            running autodetect.  Defaults to ``False``.
        examples: Optional list of example dicts, each with ``"type"``
            (``"text"``, ``"media"``, or ``"detector"``) and ``"value"`` (str).
        num_labels: Number of training labels used when this detector was last
            trained.  Defaults to ``0`` for untrained stubs.
    """
    import time

    with _state_lock:
        autorun_detectors[name] = {
            "name": name,
            "media_type": media_type,
            "weights": weights,
            "threshold": threshold,
            "created_at": time.time(),
            "autodetect": autodetect,
            "examples": examples or [],
            "num_labels": num_labels,
        }


def remove_autorun_detector(name: str) -> bool:
    """Remove a named autorun detector from the global store.

    Args:
        name: Name of the detector to remove.

    Returns:
        ``True`` if the detector was found and removed; ``False`` if no
        detector with that name exists.
    """
    with _state_lock:
        if name in autorun_detectors:
            del autorun_detectors[name]
            return True
        return False


def rename_autorun_detector(old_name: str, new_name: str) -> bool:
    """Rename a autorun detector, updating its internal ``"name"`` field.

    The operation is atomic with respect to the dict: the old entry is removed
    and a new entry is created in a single step (no window where neither exists).

    Args:
        old_name: Current name of the detector to rename.
        new_name: Desired new name for the detector.

    Returns:
        ``True`` if the rename succeeded (old name existed and new name was not
        already taken); ``False`` otherwise (no changes are made).
    """
    with _state_lock:
        if old_name in autorun_detectors and new_name not in autorun_detectors:
            autorun_detectors[new_name] = autorun_detectors[old_name].copy()
            autorun_detectors[new_name]["name"] = new_name
            del autorun_detectors[old_name]
            return True
        return False


def get_autorun_detectors() -> dict[str, dict[str, Any]]:
    """Return a shallow copy of all autorun detectors.

    Returns:
        A dict mapping detector name to its data dict (with keys ``"name"``,
        ``"media_type"``, ``"weights"``, ``"threshold"``, ``"created_at"``).
        The returned dict is a copy; mutations to it do not affect the global store.
    """
    with _state_lock:
        return autorun_detectors.copy()


def get_autorun_detectors_by_media(media_type: str) -> dict[str, dict[str, Any]]:
    """Return all autorun detectors matching a given media type.

    Args:
        media_type: Media type to filter by (``"audio"``, ``"video"``,
            ``"image"``, or ``"paragraph"``).

    Returns:
        A dict mapping detector name to its data dict, containing only
        detectors whose ``"media_type"`` field equals ``media_type``.
        The returned dict is a new dict object; mutations do not affect the
        global store.
    """
    with _state_lock:
        return {name: det for name, det in autorun_detectors.items() if det["media_type"] == media_type}


def set_autorun_detector_autodetect(name: str, autodetect: bool) -> bool:
    """Set the autodetect flag on a named autorun detector.

    Args:
        name: Name of the detector to update.
        autodetect: Whether this detector should be included in autodetect.

    Returns:
        ``True`` if the detector was found and updated; ``False`` otherwise.
    """
    with _state_lock:
        if name in autorun_detectors:
            autorun_detectors[name]["autodetect"] = autodetect
            return True
        return False


def set_autorun_detector_examples(name: str, examples: list[dict[str, str]]) -> bool:
    """Set the examples list on a named autorun detector.

    Args:
        name: Name of the detector to update.
        examples: List of example dicts, each with ``"type"`` and ``"value"``.

    Returns:
        ``True`` if the detector was found and updated; ``False`` otherwise.
    """
    with _state_lock:
        if name in autorun_detectors:
            autorun_detectors[name]["examples"] = examples
            return True
        return False


def get_autorun_detector_examples(name: str) -> list[dict[str, str]]:
    """Return the examples list for a named autorun detector.

    Returns an empty list if the detector is not found or has no examples.
    """
    with _state_lock:
        det = autorun_detectors.get(name)
        if det is None:
            return []
        return list(det.get("examples") or [])


def get_autodetect_detectors_by_media(media_type: str) -> dict[str, dict[str, Any]]:
    """Return autorun detectors matching a media type with autodetect enabled.

    Like :func:`get_autorun_detectors_by_media` but also filters to only
    include detectors whose ``"autodetect"`` flag is ``True``.

    Args:
        media_type: Media type to filter by.

    Returns:
        A dict mapping detector name to its data dict.
    """
    with _state_lock:
        return {
            name: det
            for name, det in autorun_detectors.items()
            if det["media_type"] == media_type and det.get("autodetect", True) and det.get("weights")
        }


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------


def add_autorun_extractor(name: str, extractor_type: str, media_type: str, config: dict[str, Any]) -> None:
    """Add or overwrite a named autorun extractor in the global store.

    Args:
        name: Unique human-readable name for the extractor (e.g. ``"license plates"``).
        extractor_type: The extractor class identifier (e.g. ``"image_class"``).
        media_type: The media type the extractor operates on (``"image"``, etc.).
        config: Extractor-specific configuration dict (class name, threshold, etc.).
    """
    import time

    with _state_lock:
        autorun_extractors[name] = {
            "name": name,
            "extractor_type": extractor_type,
            "media_type": media_type,
            "config": config,
            "created_at": time.time(),
        }


def remove_autorun_extractor(name: str) -> bool:
    """Remove a named autorun extractor from the global store.

    Returns:
        ``True`` if the extractor was found and removed; ``False`` otherwise.
    """
    with _state_lock:
        if name in autorun_extractors:
            del autorun_extractors[name]
            return True
        return False


def rename_autorun_extractor(old_name: str, new_name: str) -> bool:
    """Rename a autorun extractor.

    Returns:
        ``True`` if the rename succeeded; ``False`` otherwise.
    """
    with _state_lock:
        if old_name in autorun_extractors and new_name not in autorun_extractors:
            autorun_extractors[new_name] = autorun_extractors[old_name].copy()
            autorun_extractors[new_name]["name"] = new_name
            del autorun_extractors[old_name]
            return True
        return False


def get_autorun_extractors() -> dict[str, dict[str, Any]]:
    """Return a shallow copy of all autorun extractors."""
    with _state_lock:
        return autorun_extractors.copy()


def get_autorun_extractors_by_media(media_type: str) -> dict[str, dict[str, Any]]:
    """Return all autorun extractors matching a given media type."""
    with _state_lock:
        return {name: ext for name, ext in autorun_extractors.items() if ext["media_type"] == media_type}


# ---------------------------------------------------------------------------
# Localizers
# ---------------------------------------------------------------------------


def add_autorun_localizer(name: str, localizer_type: str, media_type: str, config: dict[str, Any]) -> None:
    """Add or overwrite a named autorun localizer in the global store."""
    import time

    with _state_lock:
        autorun_localizers[name] = {
            "name": name,
            "localizer_type": localizer_type,
            "media_type": media_type,
            "config": config,
            "created_at": time.time(),
        }


def remove_autorun_localizer(name: str) -> bool:
    """Remove a named autorun localizer. Returns True if found."""
    with _state_lock:
        if name in autorun_localizers:
            del autorun_localizers[name]
            return True
        return False


def rename_autorun_localizer(old_name: str, new_name: str) -> bool:
    """Rename a autorun localizer. Returns True if succeeded."""
    with _state_lock:
        if old_name in autorun_localizers and new_name not in autorun_localizers:
            autorun_localizers[new_name] = autorun_localizers[old_name].copy()
            autorun_localizers[new_name]["name"] = new_name
            del autorun_localizers[old_name]
            return True
        return False


def get_autorun_localizers() -> dict[str, dict[str, Any]]:
    """Return a shallow copy of all autorun localizers."""
    with _state_lock:
        return autorun_localizers.copy()


def get_autorun_localizers_by_media(media_type: str) -> dict[str, dict[str, Any]]:
    """Return all autorun localizers matching a given media type."""
    with _state_lock:
        return {name: loc for name, loc in autorun_localizers.items() if loc["media_type"] == media_type}


# ---------------------------------------------------------------------------
# Diversity Tree
# ---------------------------------------------------------------------------


def build_diversity_tree(media_dict: dict[int, dict[str, Any]] | None = None) -> None:
    """Build a 3-Diversity Tree from media embeddings.

    Uses the global ``medias`` dict by default, or an explicit *media_dict*
    if provided.  Existing labels in ``good_votes`` and ``bad_votes`` are
    replayed into the new tree so the seen state stays accurate.
    """
    global _diversity_tree
    import numpy as np

    from vtsearch.models.diversity_tree import DiversityTree

    with _state_lock:
        source = media_dict if media_dict is not None else medias
        vectors: dict[int, np.ndarray] = {}
        for cid, media in source.items():
            emb = media.get("embedding")
            if emb is not None:
                vectors[cid] = np.asarray(emb, dtype=np.float32)

        if not vectors:
            _diversity_tree = None
            return

        _diversity_tree = DiversityTree(vectors, k=3)

        # Replay existing labels so the tree reflects the current vote state.
        for cid in good_votes:
            if cid in _diversity_tree.vector_to_leaf:
                _diversity_tree.label(cid)
        for cid in bad_votes:
            if cid in _diversity_tree.vector_to_leaf:
                _diversity_tree.label(cid)


def get_diversity_tree():
    """Return the current DiversityTree instance, or ``None``."""
    with _state_lock:
        return _diversity_tree


def diversity_tree_next_sample(scores: dict[int, float] | None = None) -> int | None:
    """Return the next diverse sample ID, or ``None`` if unavailable.

    When *scores* is provided, the highest-scored element in the next
    unseen node is returned (so the sort mode influences selection).
    """
    with _state_lock:
        if _diversity_tree is None:
            return None
        return _diversity_tree.next_sample(scores=scores)


def diversity_tree_label(media_id: int) -> None:
    """Mark *media_id* as labeled in the diversity tree."""
    with _state_lock:
        if _diversity_tree is not None and media_id in _diversity_tree.vector_to_leaf:
            _diversity_tree.label(media_id)


def diversity_tree_unlabel(media_id: int) -> None:
    """Remove *media_id*'s label from the diversity tree."""
    with _state_lock:
        if _diversity_tree is not None and media_id in _diversity_tree.vector_to_leaf:
            _diversity_tree.unlabel(media_id)


# ---------------------------------------------------------------------------
# Clip matching helpers (origin+origin_name union with MD5)
# ---------------------------------------------------------------------------


def _origin_key(origin: dict[str, Any], origin_name: str) -> str:
    """Return a hashable string key for an (origin, origin_name) pair."""
    return json.dumps(origin, sort_keys=True) + "\0" + origin_name


def build_media_lookup(
    media_dict: dict[int, dict[str, Any]],
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """Build lookup tables for matching label entries to medias.

    Returns ``(origin_lookup, md5_lookup)`` where:

    * **origin_lookup** maps ``_origin_key(origin, origin_name)`` to a list of
      media IDs that share that origin+name pair.
    * **md5_lookup** maps an MD5 hex string to a list of media IDs whose
      content hash matches.

    Both lookups map to *lists* because the same key can match multiple medias
    (e.g. duplicate files with the same MD5).
    """
    origin_lookup: dict[str, list[int]] = {}
    md5_lookup: dict[str, list[int]] = {}

    for media in media_dict.values():
        cid = media["id"]

        origin = media.get("origin")
        origin_name = media.get("origin_name", "")
        if origin is not None and origin_name:
            key = _origin_key(origin, origin_name)
            origin_lookup.setdefault(key, []).append(cid)

        md5 = media.get("md5", "")
        if md5:
            md5_lookup.setdefault(md5, []).append(cid)

    return origin_lookup, md5_lookup


def resolve_media_ids(
    entry: dict[str, Any],
    origin_lookup: dict[str, list[int]],
    md5_lookup: dict[str, list[int]],
) -> list[int]:
    """Resolve a label entry to matching media ID(s).

    Returns the **union** of medias matched by ``origin`` + ``origin_name``
    and medias matched by ``md5``.  Both lookups are always attempted so that
    a label is applied to every element in the dataset that corresponds to
    the entry, regardless of whether it was matched by provenance or by
    content hash.  Duplicate IDs are removed.
    """
    matched: dict[int, None] = {}

    origin = entry.get("origin")
    origin_name = entry.get("origin_name", "")

    if origin is not None and origin_name:
        key = _origin_key(origin, origin_name)
        for cid in origin_lookup.get(key, []):
            matched[cid] = None

    md5 = entry.get("md5", "")
    if md5:
        for cid in md5_lookup.get(md5, []):
            matched[cid] = None

    return list(matched)


def find_missing_entries(
    label_entries: list[dict[str, Any]],
    origin_lookup: dict[str, list[int]],
    md5_lookup: dict[str, list[int]],
) -> list[dict[str, Any]]:
    """Return label entries that do not match any media by origin+name or md5.

    Only entries with a valid label (``"good"`` or ``"bad"``) are considered;
    entries with invalid labels are silently excluded (they are already counted
    as "skipped" by the caller).
    """
    missing: list[dict[str, Any]] = []
    for entry in label_entries:
        label = entry.get("label", "")
        if label not in ("good", "bad"):
            continue
        cids = resolve_media_ids(entry, origin_lookup, md5_lookup)
        if not cids:
            missing.append(entry)
    return missing


# ---------------------------------------------------------------------------
# Compound operations (atomic vote toggle / label apply)
# ---------------------------------------------------------------------------


def toggle_vote(media_id: int, vote: str) -> None:
    """Atomically toggle a good/bad vote for a media item.

    Implements the same toggle semantics as the ``/api/medias/<id>/vote``
    endpoint: if the media already has the requested vote it is removed
    (unlabelled); otherwise the vote is applied (overriding any existing
    opposite vote).

    This function acquires ``_state_lock`` so that the entire check-then-modify
    sequence is atomic with respect to concurrent requests.

    Args:
        media_id: Integer ID of the media to vote on.
        vote: ``"good"`` or ``"bad"``.
    """
    with _state_lock:
        if vote == "good":
            if media_id in good_votes:
                good_votes.pop(media_id, None)
                remove_click_time(media_id)
                add_label_to_history(media_id, "unlabel")
                if media_id not in bad_votes:
                    diversity_tree_unlabel(media_id)
            else:
                bad_votes.pop(media_id, None)
                good_votes[media_id] = None
                assign_click_time(media_id)
                add_label_to_history(media_id, "good")
                diversity_tree_label(media_id)
        else:
            if media_id in bad_votes:
                bad_votes.pop(media_id, None)
                remove_click_time(media_id)
                add_label_to_history(media_id, "unlabel")
                if media_id not in good_votes:
                    diversity_tree_unlabel(media_id)
            else:
                good_votes.pop(media_id, None)
                bad_votes[media_id] = None
                assign_click_time(media_id)
                add_label_to_history(media_id, "bad")
                diversity_tree_label(media_id)


def apply_label(media_id: int, label: str) -> None:
    """Atomically apply a label to a media (for imports).

    Unlike :func:`toggle_vote`, this always sets the label without toggling.
    No click-time is assigned (imported labels have no click-time).

    Args:
        media_id: Integer ID of the media to label.
        label: ``"good"`` or ``"bad"``.
    """
    with _state_lock:
        if label == "good":
            bad_votes.pop(media_id, None)
            good_votes[media_id] = None
            add_label_to_history(media_id, "good")
        else:
            good_votes.pop(media_id, None)
            bad_votes[media_id] = None
            add_label_to_history(media_id, "bad")
        diversity_tree_label(media_id)


def apply_label_with_click_time(media_id: int, label: str) -> None:
    """Atomically apply a label with click-time assignment (for fill-from-sort).

    Same as :func:`apply_label` but also assigns a click-time ordinal so the
    label appears in the frontend's click-time timeline.

    Args:
        media_id: Integer ID of the media to label.
        label: ``"good"`` or ``"bad"``.
    """
    with _state_lock:
        if label == "good":
            bad_votes.pop(media_id, None)
            good_votes[media_id] = None
            add_label_to_history(media_id, "good")
        else:
            good_votes.pop(media_id, None)
            bad_votes[media_id] = None
            add_label_to_history(media_id, "bad")
        assign_click_time(media_id)
        diversity_tree_label(media_id)


def collapse_duplicates(media_dict: dict[int, dict[str, Any]]) -> int:
    """Collapse duplicate medias (same MD5) into single representative items.

    For each group of medias sharing the same MD5, the first media becomes
    the representative.  Its ``"origin"`` is replaced with a ``"dupe_set"``
    origin whose ``"members"`` list records the original provenance of every
    duplicate (including the representative itself).  All other medias in the
    group are removed from *media_dict*.

    Args:
        media_dict: The mutable medias dict.  Modified in place.

    Returns:
        The number of duplicate groups collapsed (i.e. groups of size >= 2).
    """
    md5_groups: dict[str, list[int]] = {}
    for cid, media in media_dict.items():
        md5 = media.get("md5", "")
        if md5:
            md5_groups.setdefault(md5, []).append(cid)

    dupe_count = 0
    for md5, cids in md5_groups.items():
        if len(cids) < 2:
            continue
        dupe_count += 1

        rep_id = cids[0]
        rep = media_dict[rep_id]

        # Build members list with each duplicate's provenance
        members = []
        for cid in cids:
            media = media_dict[cid]
            members.append({
                "origin": media.get("origin"),
                "origin_name": media.get("origin_name", ""),
                "filename": media.get("filename", ""),
                "category": media.get("category", ""),
            })

        first_name = rep.get("origin_name", rep.get("filename", ""))
        rep["origin"] = {
            "importer": "dupe_set",
            "params": {"name": first_name},
            "members": members,
        }
        rep["origin_name"] = first_name

        # Remove the other duplicates
        for cid in cids[1:]:
            del media_dict[cid]

    return dupe_count


def get_dupe_count(media_dict: dict[int, dict[str, Any]] | None = None) -> int:
    """Return the number of duplicate groups in the media dict.

    Each media whose origin is ``"dupe_set"`` represents one group.
    """
    with _state_lock:
        source = media_dict if media_dict is not None else medias
        return sum(
            1
            for m in source.values()
            if isinstance(m.get("origin"), dict) and m["origin"].get("importer") == "dupe_set"
        )


def next_media_id(media_dict: dict[int, dict[str, Any]]) -> int:
    """Return the next available media ID (one past the current maximum)."""
    if not media_dict:
        return 1
    return max(media_dict) + 1
