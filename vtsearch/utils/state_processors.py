"""Autorun extractor / localizer CRUD operations."""

from __future__ import annotations

from typing import Any

from vtsearch.utils.state_core import (
    _state_lock,
    autorun_extractors,
    autorun_localizers,
)


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
