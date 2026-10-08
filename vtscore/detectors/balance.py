"""A detector's balance, kept on the detector (#4665).

The balance is F-beta's *beta* (#4413): which way the detector's line leans
between false positives and false negatives.  It is the detector's own
preference, asked for when the detector is created and changed from then on
by the Threshold control in Train's Manual tab and in Test.  Autopilot has no
Threshold control of its own, so this is how it runs at the balance the user
wants: Autopilot's line, its Hard / New picks and its spot checks all read the
detector's balance.

**What persists.**  One number under :data:`BETA_KEY` at the top of the
detector's JSON, beside its labelset.  A detector with none (one written
before #4665, or by a path that asks nobody: AutoFind, the CLI) takes the
user's balance, the per-user setting that is the user's last pick; see
:func:`vtscore.state.detector_beta`.  A stored value outside
``[BETA_MIN, BETA_MAX]`` or that is not a number reads as none.

**In memory.**  :class:`~vtscore.state.core.DetectorContext` caches the value
(``beta`` / ``beta_seeded``) the first time anything reads it;
:func:`vtscore.state.set_beta` writes both the cache and this file.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from pathlib import Path

    from vtscore.state.core import DetectorContext

log = logging.getLogger(__name__)

#: The detector JSON's key for its balance.
BETA_KEY = "beta"


def valid_beta(raw: Any) -> float | None:
    """*raw* as a balance, or ``None`` when it is not a finite number in ``[BETA_MIN, BETA_MAX]``."""
    from vtscore.training.thresholds import BETA_MAX, BETA_MIN  # noqa: PLC0415

    if isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or not BETA_MIN <= value <= BETA_MAX:
        return None
    return value


def stored_beta(data: Mapping[str, Any] | None) -> float | None:
    """The balance a parsed detector JSON keeps, or ``None`` when it keeps none."""
    if not data:
        return None
    return valid_beta(data.get(BETA_KEY))


def _registered_detector_file(detector_id: str) -> Path | None:
    """The detector JSON a registry id names, or ``None`` with no registry entry to name it."""
    from vtscore.detectors.registry import get_detector  # noqa: PLC0415
    from vtscore.detectors.store import _detector_path  # noqa: PLC0415

    entry = get_detector(detector_id) if detector_id else None
    if not entry or not entry.get("name"):
        return None
    return _detector_path(entry["name"])


def detector_stored_beta(detector_id: str) -> float | None:
    """The balance the registered detector *detector_id* keeps, or ``None``.

    ``None`` too for an id the registry does not know (a throwaway context's
    empty id, the CLI's name-keyed context): those callers read the detector
    JSON they already hold through :func:`stored_beta` instead.
    """
    from vtscore.detectors.store import _read_detector  # noqa: PLC0415

    path = _registered_detector_file(detector_id)
    return None if path is None else stored_beta(_read_detector(path))


def keep_beta(det_ctx: DetectorContext, value: float) -> bool:
    """Write *value* as *det_ctx*'s detector's balance; ``True`` when the file changed.

    Read, merged and written under
    :data:`~vtscore.detectors.label_sync.label_sync_write_lock`, as every
    detector-JSON read-modify-write is, so a concurrent label sync can neither
    lose the balance nor be lost to it.  The cached labelset is re-pointed at
    the file afterwards, so the write does not read as an outside edit.  A
    context with no registered detector behind it has nowhere to keep one,
    and an unchanged value is not rewritten.

    Never call this holding :data:`vtscore.state.core._state_lock`: the label
    sync takes that lock while it holds the write lock.
    """
    from vtscore.detectors.dataset_sync import _repoint_labelset_cache  # noqa: PLC0415
    from vtscore.detectors.label_sync import label_sync_write_lock  # noqa: PLC0415
    from vtscore.detectors.store import _read_detector, _write_detector  # noqa: PLC0415

    beta = valid_beta(value)
    path = _registered_detector_file(getattr(det_ctx, "detector_id", "") or "")
    if beta is None or path is None:
        return False
    with label_sync_write_lock:
        data = _read_detector(path)
        if data is None or stored_beta(data) == beta:
            return False
        data[BETA_KEY] = beta
        _write_detector(path, data)
    _repoint_labelset_cache(det_ctx, path)
    log.info("kept balance %g on detector %s", beta, det_ctx.detector_id)
    return True
