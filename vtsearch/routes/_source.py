"""Shared pieces of the routes that reach a converter output's source item.

``GET /api/medias/<id>/source`` and its batch form (#4749) name one face's
photo; ``GET /api/sort/collapsed`` (#4751) groups a whole ranking by photo.
Both locate against the requesting dataset, confined to what the current user
may read, and both answer an unloaded sibling with the same 409, so the client
handles it in one place.
"""

from __future__ import annotations

from typing import NoReturn

from flask_smorest import abort

from vtscore.datasets import source_item


def source_locator() -> source_item.SourceLocator:
    """A locator for the requesting dataset, confined to what the current user may read."""
    from vtsearch.auth import get_current_user  # noqa: PLC0415
    from vtsearch.state import get_active_context  # noqa: PLC0415

    return source_item.SourceLocator(get_active_context().dataset_id, user=get_current_user())


def abort_source_not_loaded(dataset_id: str | None) -> NoReturn:
    """The 409 for a source dataset that is registered but not loaded; ``dataset_id`` names it."""
    abort(
        409,
        message="Load the source dataset to resolve this item's source.",
        error_code="source_not_loaded",
        dataset_id=dataset_id,
    )
    raise RuntimeError("unreachable (abort() raises)")  # for type narrowing
