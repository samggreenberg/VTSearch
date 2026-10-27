"""The index-compat guard: one embedding space per index, enforced, not assumed.

Two embedders produce two spaces, and a cosine between them is noise with a
confident-looking value.  Every :class:`~imagekit.face.records.FaceRecord`
carries the tag of the space it lives in (``compat``: ``"oss-magface-ir100:1"``,
an external backend's own tag, ...), and this module is where an index or a
dataset turns that tag into a rule:

* :func:`stamp_compat` writes the tag into the index's metadata when the index
  is **created**;
* :func:`check_compat` is called before every **add** and every **query**, and
  raises :class:`CompatibilityError` when the vectors' tag is not the index's.

The metadata is any mutable mapping the index already persists (a JSON
sidecar, a dataset header); the guard only owns one key in it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from typing import Any

#: The metadata key the guard owns.
COMPAT_KEY = "compat"


class CompatibilityError(ValueError):
    """Vectors from one embedding space met an index built for another."""


def require_same_compat(expected: str, actual: str, *, what: str = "vectors") -> None:
    """Raise :class:`CompatibilityError` unless *actual* equals *expected*."""
    if str(actual) != str(expected):
        raise CompatibilityError(
            f"{what} carry compat {actual!r} but the index was built for {expected!r}; "
            "re-embed with the index's model, or build a new index"
        )


def stamp_compat(metadata: MutableMapping[str, Any], compat: str) -> MutableMapping[str, Any]:
    """Record *compat* in *metadata* at index creation; returns *metadata*.

    Stamping twice with the same tag is a no-op; stamping an index already
    tagged with a different one raises, because that index already holds
    vectors of the other space.
    """
    if not compat:
        raise ValueError("compat must be a non-empty string")
    existing = metadata.get(COMPAT_KEY)
    if existing is not None:
        require_same_compat(str(existing), compat, what="the new stamp's vectors")
    metadata[COMPAT_KEY] = str(compat)
    return metadata


def check_compat(metadata: Mapping[str, Any], compat: str, *, what: str = "vectors") -> None:
    """Refuse *compat* unless it is the tag *metadata* was stamped with.

    An unstamped index is refused too: a guard that passed silently on a
    missing stamp would pass every mixed index ever created before the stamp
    existed.
    """
    stamped = metadata.get(COMPAT_KEY)
    if stamped is None:
        raise CompatibilityError(
            f"the index carries no {COMPAT_KEY!r} stamp, so {what} (compat {compat!r}) cannot be checked against it; "
            "stamp the index at creation with stamp_compat()"
        )
    require_same_compat(str(stamped), compat, what=what)


def single_compat(compats: Iterable[str], *, what: str = "records") -> str:
    """The one tag all of *compats* share; raises when they disagree or there are none."""
    seen = sorted({str(c) for c in compats})
    if not seen:
        raise CompatibilityError(f"no {what} to take a compat tag from")
    if len(seen) > 1:
        raise CompatibilityError(f"{what} mix embedding spaces: {seen}; one index holds one space")
    return seen[0]
