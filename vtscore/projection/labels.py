"""Region signpost labels for the VTSBrowse map — the "street sign" layer.

The data contract between the labeling pipeline (Toponymy — see
``docs/plans/vtsbrowse-toponymy.md``) and the browse canvas.  A
:class:`RegionLabel` is one named region: a **text** sign anchored at a 2-D
point of the frozen projection layout, tagged with the pyramid zoom **level**
it belongs to (0 = the coarsest layer — "continents"; deeper = finer —
"countries", then "states").  The canvas shows each sign only while the user's
zoom is near its level, small just below it, full-size at it, enlarged above
it (``sign-layout.ts`` owns those bands).

Labels are **derived text**: names + 2-D anchors + scalar scores.  No
embeddings, keyphrase vectors, or model state belong here (the No-Persisted-
Vectors rule); the labeling pipeline keeps those build-time-only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Iterable


@dataclass(frozen=True)
class RegionLabel:
    """One signpost: a named region of the projection map."""

    #: Pyramid zoom level the sign belongs to (0 = coarsest).  May be
    #: fractional — the canvas interpolates visibility on a continuous axis.
    level: float
    #: Anchor in projection space (the frozen 2-D layout's coordinates),
    #: typically the projected medoid of the named cluster.
    x: float
    y: float
    text: str
    #: Naming confidence; the canvas uses it as the de-clutter tiebreak.
    score: float = 1.0
    #: Which namer produced the sign (e.g. ``"keyphrase"``, ``"llm"``).
    source: str = ""


@dataclass(frozen=True)
class RegionLabelSet:
    """All signposts computed for one frozen projection layout.

    ``projection_id`` pins the set to the layout it was computed from: anchors
    are coordinates in that specific (unseeded, non-reproducible) UMAP fit, so
    a set must never be served against any other layout.  Consumers compare it
    against the active pyramid's id and treat a mismatch as "no labels".
    """

    projection_id: str
    labels: tuple[RegionLabel, ...]

    def payload(self) -> list[dict]:
        """The labels as JSON-ready dicts (the ``/api/projection/labels`` body)."""
        return [asdict(label) for label in self.labels]


def make_label_set(projection_id: str, labels: Iterable[RegionLabel]) -> RegionLabelSet:
    """Build a :class:`RegionLabelSet`, normalising ``labels`` to a tuple."""
    return RegionLabelSet(projection_id=projection_id, labels=tuple(labels))


__all__ = ["RegionLabel", "RegionLabelSet", "make_label_set"]
