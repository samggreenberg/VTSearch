"""The blueprint every detector-registry route module registers on, and the helpers they share.

:func:`_abort_if_name_taken` guards both create routes and the rename;
:func:`_creation_beta` and :func:`_keep_creation_beta` are how the create
routes handle the balance a new detector starts at.
"""

from __future__ import annotations

from pathlib import Path

from flask_smorest import Blueprint, abort

from vtscore.detectors.store import _detector_path

detectors_registry_bp = Blueprint(
    "detectors_registry",
    __name__,
    description="Register, load, unload, rename, and toggle AutoFind on detectors.",
)


def _abort_if_name_taken(name: str, *, own_path: Path | None = None, exclude_id: str = "") -> None:
    """Abort 409 when *name* would collide with an existing detector.

    Two detectors collide as soon as their names resolve to the same labelset
    file: ``_slug`` lowercases and strips punctuation, so "My Cat" and "my cat"
    share one ``data/detectors/*.json``.  Sharing a file is never benign —
    whichever detector writes last destroys the other's labelset, and deleting
    either one unlinks the file out from under the survivor.

    Both halves are checked because either can outlive the other: the on-disk
    labelset file (a detector created through the non-registry ``/api/detectors``
    route has a file but no registry entry) and the registry entries (an entry
    whose file was deleted still owns the name).

    Args:
        name: The proposed detector name.
        own_path: The labelset path the caller already owns, if any.  A rename
            that only re-spells its own name ("My Cat" -> "my cat") resolves to
            this same path and is allowed through.
        exclude_id: Registry id to skip, so a rename doesn't collide with itself.
    """
    from vtscore.detectors.registry import list_detectors

    new_path = _detector_path(name)
    if own_path is not None and new_path == own_path:
        return
    if new_path.exists() or any(
        entry.get("id") != exclude_id and _detector_path(entry.get("name", "")) == new_path
        for entry in list_detectors()
    ):
        abort(409, message=f"A detector named '{name}' already exists")


def _creation_beta(raw) -> float | None:
    """The balance a new detector is created at (#4665), or ``None`` when the request names none.

    Validated by the same ``settings.validate_beta`` the Threshold control's
    ``POST /api/balance`` uses, so the two cannot disagree about the range;
    an unparseable value is a 400, a boolean included, as there.  A
    form-encoded create (a label importer with a file field) sends it as a
    string, and an empty one means none.
    """
    from vtsearch import settings  # noqa: PLC0415

    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    if isinstance(raw, bool):
        abort(400, message="beta: must be a number")
    try:
        return float(settings.validate_beta(raw))
    except (TypeError, ValueError) as exc:
        abort(400, message=f"beta: {exc}")


def _keep_creation_beta(detector_data: dict, beta: float | None) -> None:
    """Keep *beta* on a new detector's JSON, and as the user's balance (their last pick, #4665).

    The user's balance is what the next New Detector form starts on and what a
    detector that keeps none takes.  With no *beta* the detector keeps none,
    and takes the user's balance when it is loaded.
    """
    from vtscore.detectors.balance import BETA_KEY  # noqa: PLC0415
    from vtsearch import settings  # noqa: PLC0415

    if beta is None:
        return
    detector_data[BETA_KEY] = beta
    settings.set_beta(beta)
