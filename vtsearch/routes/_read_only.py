"""Refuse a change to a received, read-only detector before it has any side effect.

A received (``linked``) detector mirrors a container someone else owns, so its
labelset and its name change only through a refresh.  The store refuses such
a write (:class:`vtscore.detectors.store.DetectorReadOnlyError`), but a vote
lands in the detector's in-memory votes before its file follows, so a route
that changes a labelset or an identity checks first, with
:func:`refuse_linked`, and answers 409 with
``error_code: "detector_read_only"`` before touching anything.  The frontend
keys its "fork it instead" prompt on that code.

Not every write to a detector is refused: thresholds, the balance, test
verdicts, the AutoFind flag, readers, loading, Find and deleting the local
copy all stay open (``docs/ARCHITECTURE.md``, "Detector sharing").  A received detector
may be a combine *source*; combine only ever creates a new detector, so it
can never be a target.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, NoReturn

from flask_smorest import abort

#: The ``alt_response(409, ...)`` description of the refusal, for a route
#: that has no other 409.  A route with its own 409 names both.
READ_ONLY_DESCRIPTION = (
    "The detector is a received, read-only copy (``error_code: detector_read_only``); fork it to change it."
)

#: The targets :func:`refuse_linked` resolves: the request's active detector,
#: or a view argument naming one.
TARGETS = ("active", "name", "detector_id", "detector_ref")


def _entry_for(target: str, view_args: dict[str, Any]) -> dict[str, Any] | None:
    """The registry entry *target* names in this request, or ``None``."""
    from vtscore.detectors.registry import find_by_name, get_detector  # noqa: PLC0415

    if target == "active":
        from vtscore.state.core import DetectorNotLoadedError, get_active_detector_context  # noqa: PLC0415

        try:
            detector_id = get_active_detector_context().detector_id
        except DetectorNotLoadedError:
            return None  # the route body answers that one
        return get_detector(detector_id) if detector_id else None
    value = view_args.get(target)
    if not isinstance(value, str) or not value:
        return None
    if target == "name":
        return find_by_name(value)
    if target == "detector_id":
        return get_detector(value)
    return get_detector(value) or find_by_name(value)


def _in_find_mode(detector_id: str) -> bool:
    """Whether the loaded context of *detector_id* is in Find mode, where votes never reach the labelset."""
    from vtscore.state.core import get_detector_context  # noqa: PLC0415

    ctx = get_detector_context(detector_id)
    return bool(ctx is not None and ctx.find_mode)


def abort_read_only(entry: dict[str, Any]) -> NoReturn:
    """Answer 409 ``detector_read_only`` for the received detector *entry*, naming its container's owner.

    What :func:`refuse_linked` answers; also for a route that refuses a
    received detector only after checks of its own (the sharing routes, which
    answer ``sharing_disabled`` first).
    """
    from vtscore.detectors.store import DetectorReadOnlyError  # noqa: PLC0415
    from vtsearch.errors import DETECTOR_READ_ONLY  # noqa: PLC0415

    refusal = DetectorReadOnlyError.for_entry(entry)
    abort(
        409,
        message=str(refusal),
        error_code=DETECTOR_READ_ONLY,
        detector_id=refusal.detector_id,
        remote_owner=refusal.owner,
    )


def refuse_linked(target: str, *, find_mode_ok: bool = False) -> Callable[[Callable], Callable]:
    """Route decorator: answer 409 ``detector_read_only`` when the targeted detector is a received copy.

    Apply it innermost (directly above the ``def``), so it runs after the
    request's context headers are checked and before the body does anything.
    The route declares the 409 with ``alt_response`` (using
    :data:`READ_ONLY_DESCRIPTION`, or naming it beside its own 409).

    Args:
        target: Which detector the route changes: ``"active"`` (the
            request's active detector, for the vote and import routes) or the
            view argument that names it - ``"name"``, ``"detector_id"`` or
            ``"detector_ref"`` (an id, else a name).
        find_mode_ok: Let the request through while that detector is in Find
            mode: Find votes are verification, kept out of the labelset, so a
            received detector can still be used to find.
    """
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, not {target!r}")

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            from vtscore.detectors.registry import is_linked  # noqa: PLC0415

            entry = _entry_for(target, kwargs)
            if entry is not None and is_linked(entry) and not (find_mode_ok and _in_find_mode(entry["id"])):
                abort_read_only(entry)
            return fn(*args, **kwargs)

        wrapper.refuses_linked_detector = target  # type: ignore[attr-defined]
        return wrapper

    return decorator
