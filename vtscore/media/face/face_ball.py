"""The face ball's floors per face space, read from ``face_ball.json`` beside the face pipeline config (#5497).

#5478's face line keeps every face whose cosine to the Goods' centroid clears a floor, and that
floor is a cosine in one embedder's space: buffalo_s's floors fit buffalo_l by coincidence and lose
0.65 to 0.76 F on FaceNet (#5486).  So each space (the ``Face Compat`` tag the face provider
stamps on every face) brings its own entry, fitted from a labelled face set the server's provider
embedded (:mod:`vtscore.media.face.face_ball_fit`, ``scripts/data/fit_face_ball.py``).  The file
is a calculated asset: it lives beside the pipeline config the app reads (``imagekit/face_ball.json``
in the models folder by default, or beside the config ``IMAGEKIT_FACE_CONFIG`` names), never in the
repo, so a pack directory carries its floors with it (#5621).

An entry holds a floor and a slope per balance, the spread they are centred on, whether the ball
is used in that space at all (``use``, the fit's go/no-go check from the space's own data), and
how it was fitted.  :func:`face_ball_for` is the one reader.  A space with no entry, or a no-go
one, keeps the face crops' line from before the ball and says so once
(:func:`warn_uncalibrated`).

The file::

    {
      "version": 1,
      "spaces": {
        "oss-arcface-mbf:1": {
          "use": true,
          "betas": {"1": {"floor": 0.4425, "slope": 0.25}, "4": {"floor": 0.385, "slope": 0.25}},
          "centre": 0.6371,
          "check": {...},
          "provenance": {...}
        }
      }
    }
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np

logger = logging.getLogger(__name__)

#: The file's name.  It sits beside the face pipeline config in use (:func:`face_ball_path`), so a
#: pack directory (config, models and floors) moves as one unit.
FACE_BALL_NAME = "face_ball.json"

#: Where, under the models folder, the file is when ``IMAGEKIT_FACE_CONFIG`` is unset: beside the
#: default ``imagekit/face.json``.
FACE_BALL_RELPATH = Path("imagekit") / FACE_BALL_NAME

#: The file's format.  A file of another version is not read.
FILE_VERSION = 1

#: How long a read of the file is reused.  The line is drawn on every Find and every click of an
#: eval session, so the file is not re-read each time; a refit shows up within a couple of seconds.
_TTL_S = 2.0


@dataclass(frozen=True)
class FaceBall:
    """One face space's ball: a cosine floor and slope per balance, centred on a spread.

    The floor at *beta* for Goods of *spread* (:meth:`floor`) is the balance's floor moved by its
    slope times how far the spread sits from :attr:`centre`.  Between the fitted balances (and past
    the highest) both are read off a line in log beta; below the lowest they hold at it.
    """

    compat: str
    use: bool
    #: ``{beta: (floor, slope)}``, at least one balance.
    betas: Mapping[float, tuple[float, float]]
    centre: float
    check: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def floor(self, beta: float, spread: float) -> float:
        """The ball's cosine floor at *beta* for Goods of *spread*."""
        lo, hi = min(self.betas), max(self.betas)
        (f_lo, s_lo), (f_hi, s_hi) = self.betas[lo], self.betas[hi]
        if hi == lo:
            return float(f_lo + s_lo * (spread - self.centre))
        x = (np.log(max(beta, lo)) - np.log(lo)) / (np.log(hi) - np.log(lo))
        floor = f_lo + x * (f_hi - f_lo)
        slope = s_lo + x * (s_hi - s_lo)
        return float(floor + slope * (spread - self.centre))

    def to_json(self) -> dict[str, Any]:
        """The entry as the file stores it."""
        return {
            "use": self.use,
            "betas": {f"{b:g}": {"floor": f, "slope": s} for b, (f, s) in sorted(self.betas.items())},
            "centre": self.centre,
            "check": dict(self.check),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_json(cls, compat: str, entry: Mapping[str, Any]) -> "FaceBall":
        """An entry read from the file; raises ``ValueError`` on a malformed one."""
        try:
            betas = {float(b): (float(v["floor"]), float(v["slope"])) for b, v in dict(entry["betas"]).items()}
            ball = cls(
                compat=compat,
                use=bool(entry["use"]),
                betas=betas,
                centre=float(entry["centre"]),
                check=dict(entry.get("check") or {}),
                provenance=dict(entry.get("provenance") or {}),
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"face ball entry {compat!r} is malformed: {exc!r}") from exc
        if not betas or any(b <= 0 or not all(map(math.isfinite, fs)) for b, fs in betas.items()):
            raise ValueError(f"face ball entry {compat!r} needs finite floors at positive balances")
        return ball


def face_ball_path() -> Path:
    """The file the entries are read from: :data:`FACE_BALL_NAME` beside the face pipeline config in use.

    That is the config ``IMAGEKIT_FACE_CONFIG`` names, else the models folder's ``imagekit/face.json``
    (:func:`~vtscore.media.face.imagekit_bridge.config_path`), whether or not either exists.  So the
    floors travel with the pack they were fitted on: a server pointed at a pack elsewhere reads that
    pack's entry, not whatever the models folder's ``imagekit`` directory holds (#5621).
    """
    from vtscore.media.face.imagekit_bridge import config_path  # noqa: PLC0415

    return config_path().parent / FACE_BALL_NAME


_lock = threading.Lock()
#: ``(read at, path, mtime_ns, entries, error)`` of the last read.
_cache: Optional[tuple[float, Path, int, dict[str, FaceBall], str]] = None


def _read(path: Path) -> tuple[dict[str, FaceBall], str]:
    """Every entry in *path*, and why the file could not be read (``""`` when it could, or is absent)."""
    if not path.is_file():
        return {}, ""
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or data.get("version") != FILE_VERSION:
            raise ValueError(
                f"expected version {FILE_VERSION}, got {data.get('version') if isinstance(data, dict) else data!r}"
            )
        spaces = data.get("spaces") or {}
        return {str(c): FaceBall.from_json(str(c), e) for c, e in dict(spaces).items()}, ""
    except Exception as exc:  # a broken file is a deployment error: reported, and the ball is off
        logger.error("face ball file %s is not readable: %s", path, exc)
        return {}, f"{path} is not readable: {exc}"


def _entries() -> tuple[dict[str, FaceBall], str]:
    global _cache
    path = face_ball_path()
    now = time.monotonic()
    with _lock:
        cached = _cache
        if cached is not None and cached[1] == path and now - cached[0] < _TTL_S:
            return cached[3], cached[4]
        try:
            mtime = path.stat().st_mtime_ns
        except OSError:
            mtime = -1
        if cached is not None and cached[1] == path and cached[2] == mtime:
            _cache = (now, path, mtime, cached[3], cached[4])
            return cached[3], cached[4]
        entries, error = _read(path)
        _cache = (now, path, mtime, entries, error)
        return entries, error


def face_ball_entry(compat: str | None) -> FaceBall | None:
    """*compat*'s entry, go or no-go, or ``None`` when the file has none for it."""
    if not compat:
        return None
    return _entries()[0].get(compat)


def face_ball_for(compat: str | None) -> FaceBall | None:
    """The ball the face crops' line is drawn by in *compat*'s space, or ``None`` where it is not used.

    ``None`` for no tag, a tag with no entry, and a no-go entry: there the line is the face crops'
    count from before the ball.
    """
    ball = face_ball_entry(compat)
    return ball if ball is not None and ball.use else None


def uncalibrated_detail(compat: str | None) -> str:
    """Why *compat*'s space has no ball, for the warning's second line."""
    entries, error = _entries()
    if error:
        reason = f"The face ball file {error}"
    elif compat and compat in entries:
        reason = f"Its fit in {face_ball_path()} found the ball no better than that line in this space"
    else:
        reason = f"{face_ball_path()} has no entry for it (scripts/data/fit_face_ball.py fits one)"
    return f"Find and Test keep the face line drawn without a calibrated cosine floor. {reason}."


#: The feature the warning names (:func:`~vtscore.detectors.centroid_head.face_uncalibrated_message`).
FACE_BALL_FEATURE = "The face search line"

_warned: set[str] = set()


def warn_uncalibrated(compat: str | None) -> None:
    """Say, once per space and process, that the face crops' line is uncalibrated for *compat*.

    A warning toast (it stays until dismissed), so the user sees why a face search keeps the
    line from before the ball.  The eval harness meets it too, where it is one log line.
    """
    key = compat or ""
    with _lock:
        if key in _warned:
            return
        _warned.add(key)
    from vtscore.concurrency.notifications import notify  # noqa: PLC0415
    from vtscore.detectors.centroid_head import face_uncalibrated_message  # noqa: PLC0415

    notify(
        face_uncalibrated_message(FACE_BALL_FEATURE, compat),
        level="warning",
        detail=uncalibrated_detail(compat),
        source="Face",
    )


def face_compat_of(medias: Iterable[Any] | Mapping[Any, Any] | None) -> str | None:
    """The one ``Face Compat`` tag *medias* carry, or ``None`` when they carry none or more than one.

    *medias* is a ``{id: media}`` dict or the media dicts themselves.  The face embedder refuses
    to mix spaces in a dataset, so a mixed set has no space a floor could be in.
    """
    from vtscore.media.face.imagekit_bridge import COMPAT_METADATA_KEY  # noqa: PLC0415

    if not medias:
        return None
    values = medias.values() if isinstance(medias, Mapping) else medias
    tags: set[str] = set()
    for media in values:
        if not isinstance(media, Mapping):
            continue
        tag = (media.get("custom_metadata") or {}).get(COMPAT_METADATA_KEY)
        if tag:
            tags.add(str(tag))
            if len(tags) > 1:
                return None
    return next(iter(tags)) if tags else None


def write_entry(ball: FaceBall, path: Path | None = None) -> Path:
    """Put *ball*'s entry in the file (*path*, default :func:`face_ball_path`), keeping every other space's.

    Written to a temporary file beside it and renamed, so a server reading the file never sees half
    of it.  Returns the path written.
    """
    target = Path(path) if path is not None else face_ball_path()
    data: dict[str, Any] = {"version": FILE_VERSION, "spaces": {}}
    if target.is_file():
        old = json.loads(target.read_text())
        if not isinstance(old, dict) or old.get("version") != FILE_VERSION:
            raise ValueError(f"{target} is not a version {FILE_VERSION} face ball file; not overwriting it")
        data["spaces"] = dict(old.get("spaces") or {})
    data["spaces"][ball.compat] = ball.to_json()
    data["spaces"] = dict(sorted(data["spaces"].items()))
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n")
    tmp.replace(target)
    _forget_read()
    return target


def _forget_read() -> None:
    global _cache
    with _lock:
        _cache = None


def reset_for_tests() -> None:
    """Forget the cached read and the spaces already warned about."""
    _forget_read()
    with _lock:
        _warned.clear()
