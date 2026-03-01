"""Autorun processor (detector, extractor, localizer) CRUD operations."""

from __future__ import annotations

from typing import Any

from vtsearch.utils.state_core import (
    _state_lock,
    autorun_detectors,
    autorun_extractors,
    autorun_localizers,
)


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


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
        autodetect: Whether this detector is included when running
            autodetect.  Defaults to ``False``.
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
    """Return a shallow copy of all autorun detectors."""
    with _state_lock:
        return autorun_detectors.copy()


def get_autorun_detectors_by_media(media_type: str) -> dict[str, dict[str, Any]]:
    """Return all autorun detectors matching a given media type."""
    with _state_lock:
        return {name: det for name, det in autorun_detectors.items() if det["media_type"] == media_type}


def set_autorun_detector_autodetect(name: str, autodetect: bool) -> bool:
    """Set the autodetect flag on a named autorun detector.

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
