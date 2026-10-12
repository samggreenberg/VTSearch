"""Group a ranking of converter outputs by the source item each was cut from (#4751).

A face detector ranks face crops, but a person looking for a face wants the
photo.  :func:`group_by_source` turns a ranking of outputs into a ranking of
their sources: every output whose source item resolves
(:class:`~vtscore.datasets.source_item.SourceLocator`) joins the group of that
item, a group scores as its best output, and the groups keep the order of
their best outputs.  ``{1A, 1B, 2C, 2D}`` becomes ``{1, 2}``, each carrying
the faces that matched in it.

This is presentation over a ranking, never a new ranking: no score is
recomputed, the outputs stay the unit a vote lands on, and an output whose
source cannot be named (not derived, no sibling, no match) stays a group of
its own, so every output of the ranking is in exactly one group.

Pure apart from the locator it is handed: the caller decides which dataset
the outputs belong to and which user the siblings are confined to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from vtscore.datasets.source_item import NOT_DERIVED, RESOLVED, SIBLING_NOT_LOADED, SourceLocation
from vtscore.state.sort_results_cache import result_score

__all__ = [
    "SourceGroup",
    "SourceGroupMember",
    "SourceNotLoadedError",
    "group_by_source",
]


class SourceNotLoadedError(LookupError):
    """An output's source dataset is registered but not loaded.

    ``dataset_id`` names it, so the caller can load it and group again.  The
    grouping is all or nothing: a half-grouped ranking would show some photos
    and leave their siblings' faces loose, which reads as a different result.
    """

    def __init__(self, dataset_id: str | None) -> None:
        super().__init__(f"source dataset {dataset_id!r} is not loaded")
        self.dataset_id = dataset_id


@dataclass(frozen=True)
class SourceGroupMember:
    """One output of the ranking inside its group: its id, score and place in the source."""

    media_id: int
    score: float
    #: Where the output sat in its source, normalised ``[x0, y0, x1, y1]``
    #: (``source_box``), or ``None`` when its converter records none.
    box: list[float] | None


@dataclass(frozen=True)
class SourceGroup:
    """A source item and the outputs of the ranking cut from it.

    ``source_dataset_id`` / ``source_media_id`` name the source item in its
    sibling dataset, or are both ``None`` for an output whose source cannot be
    named, which is then the group's only member.  ``score`` is the best
    member's.  ``members`` are in ranking order, best first.
    """

    source_dataset_id: str | None
    source_media_id: int | None
    score: float
    members: tuple[SourceGroupMember, ...]

    @property
    def resolved(self) -> bool:
        return self.source_media_id is not None


def group_by_source(
    results: Iterable[dict[str, Any]],
    medias: Mapping[int, dict[str, Any]],
    locate: Callable[[dict[str, Any]], SourceLocation],
) -> list[SourceGroup]:
    """Group a descending ranking of outputs by their source items.

    *results* are ranking rows (``{"id", "score"}`` or ``{"id",
    "similarity"}``), best first, as every sort route stores them.  *medias*
    maps their ids to media dicts, and *locate* answers where one came from:
    :meth:`SourceLocator.locate <vtscore.datasets.source_item.SourceLocator.locate>`
    for the dataset the ranking is over.

    Returns the groups in the order of their best members, which for a
    descending ranking is descending group score.  A row whose id is not in
    *medias*, or whose source does not resolve, is a group of its own.

    Raises :class:`SourceNotLoadedError` when any output's source dataset is
    registered but not loaded.
    """
    groups: list[tuple[str | None, int | None, list[SourceGroupMember]]] = []
    index: dict[tuple[str, int], int] = {}
    for row in results:
        media_id = row["id"]
        score = float(result_score(row))
        media = medias.get(media_id)
        found = locate(media) if media is not None else SourceLocation(NOT_DERIVED)
        if found.status == SIBLING_NOT_LOADED:
            raise SourceNotLoadedError(found.dataset_id)
        member = SourceGroupMember(media_id, score, found.box)
        if found.status != RESOLVED or found.dataset_id is None or found.media_id is None:
            groups.append((None, None, [member]))
            continue
        key = (found.dataset_id, found.media_id)
        at = index.get(key)
        if at is None:
            index[key] = len(groups)
            groups.append((found.dataset_id, found.media_id, [member]))
        else:
            groups[at][2].append(member)
    return [
        SourceGroup(dataset_id, media_id, max(m.score for m in members), tuple(members))
        for dataset_id, media_id, members in groups
    ]
