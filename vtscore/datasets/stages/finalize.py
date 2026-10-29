"""Finalisation stages: drop failed embeds, collapse duplicates, coverage atlas.

These run after embedding to hand the registry/projection stages a clean
``medias`` dict: media that finished without an embedding are dropped,
exact-duplicate media are collapsed, and the coverage atlas is built.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, TypedDict

from vtscore.embedding.matrix import invalidate_embedding_matrix
from vtscore.embedding.media_vectors import media_embedder_names, media_embedding
from vtscore.media.near_dupes import follow_source_merges
from vtscore.state import (
    build_coverage_atlas_for_context,
    collapse_duplicates,
    collapse_near_duplicates,
    should_auto_build_coverage_atlas,
)

from vtscore.datasets.stages._common import _TOTAL_LOAD_STEPS

if TYPE_CHECKING:
    from vtscore.datasets.stages.embedding import EmbedFailures
    from vtscore.state import DatasetContext

#: Reason given for a dropped item the embed stage recorded nothing about -
#: e.g. one that reached the finalize stage without a media type to embed by.
_UNKNOWN_DROP_REASON = "No embedder produced a vector"

#: How many dropped items' names a dataset's registry entry keeps (#3798).  The
#: entry lives in the registry manifest, which is read and rewritten whole, so
#: an import that drops 100k items must not write 100k names into it.  Counts
#: stay exact past the cap; only the names stop.
MAX_RECORDED_DROPS = 1000


class DroppedGroup(TypedDict):
    """The items one load dropped for one reason, as its registry entry keeps them."""

    #: Why these items have no vector, as the toast said it.
    reason: str
    #: How many items were dropped for it; exact, even past the name cap.
    count: int
    #: Their names (see :func:`_dropped_item_label`), at most
    #: :data:`MAX_RECORDED_DROPS` across all of a load's groups.
    items: list[str]


def _dropped_item_label(mid: int, media: dict[str, Any]) -> str:
    """Name a dropped media the way the user would look for it.

    The file it came from, not the internal id (which the user cannot look up,
    and which is gone the moment the item is dropped).  A clip adds where in
    that file it sits, since every clip of one file shares its name.
    """
    name = str(media.get("origin_name") or media.get("filename") or media.get("media_path") or f"item {mid}")
    start, end = media.get("clip_start"), media.get("clip_end")
    if start is not None and end is not None:
        try:
            return f"{name} [{float(start):g}–{float(end):g}s]"
        except (TypeError, ValueError):
            pass
    index = media.get("clip_index")
    if index is not None:
        return f"{name} (clip {index})"
    return name


def _drop_summary(n_dropped: int, n_total: int, reasons: Counter[str]) -> str:
    """The toast's detail line: how many were dropped, and why, grouped by reason."""
    lead = f"{n_dropped:,} of {n_total:,} imported item(s) had no vector after the embed step, so they were left out."
    if len(reasons) == 1:
        return f"{lead} {next(iter(reasons))}."
    return " ".join([lead, *(f"{reason} ({count:,} item(s))." for reason, count in reasons.most_common())])


def _dropped_groups(labels: list[str], reasons: list[str]) -> list[DroppedGroup]:
    """Group *labels* by their parallel *reasons*, commonest reason first.

    Names are kept until :data:`MAX_RECORDED_DROPS` is reached across every
    group, so the commonest reason fills first; each group's ``count`` is exact.
    """
    groups: dict[str, DroppedGroup] = {}
    for reason, _count in Counter(reasons).most_common():
        groups[reason] = {"reason": reason, "count": 0, "items": []}
    kept = 0
    for label, reason in zip(labels, reasons):
        group = groups[reason]
        group["count"] += 1
        if kept < MAX_RECORDED_DROPS:
            group["items"].append(label)
            kept += 1
    return list(groups.values())


def merge_dropped_groups(recorded: list[DroppedGroup], new: list[DroppedGroup]) -> list[DroppedGroup]:
    """Add *new* to the *recorded* groups of a registry entry, keeping the name cap.

    A group whose reason is already recorded is folded into it; the others are
    appended.  Names stop at :data:`MAX_RECORDED_DROPS` across every group,
    counts never do.
    """
    merged: list[DroppedGroup] = [
        {"reason": g["reason"], "count": int(g["count"]), "items": list(g["items"])} for g in recorded
    ]
    by_reason = {g["reason"]: g for g in merged}
    kept = sum(len(g["items"]) for g in merged)
    for group in new:
        target = by_reason.get(group["reason"])
        if target is None:
            target = {"reason": group["reason"], "count": 0, "items": []}
            merged.append(target)
            by_reason[group["reason"]] = target
        target["count"] += int(group["count"])
        room = max(0, MAX_RECORDED_DROPS - kept)
        target["items"].extend(group["items"][:room])
        kept += min(room, len(group["items"]))
    return merged


def drop_incomplete_medias(medias: dict[int, dict[str, Any]]) -> list[DroppedGroup]:
    """Remove the media missing a vector most of the dataset carries; return them grouped by reason.

    The repair for a dataset saved before the embed stage cleared such items
    (:func:`~vtscore.datasets.stages.embedding._clear_incomplete_vectors`): an
    item one embedder could not read kept another's vector, so it was saved,
    and every request for the missing embedder's space - Browse, sort, Train -
    has raised ``has no embedding for embedder`` on that dataset since
    (issue #3798).

    A saved dataset does not record which embedders its load ran, so the
    dataset's spaces are read off its media: an embedder whose vector more than
    half of them carry is one of them, and a media missing any of those is
    removed.  A vector only a few media carry (one an importer shipped for some
    items) marks nobody as incomplete.  When the media to remove would be half
    the dataset or more, nothing is removed: a dataset that broken is not one
    whose majority can be trusted to say what it should look like.
    """
    from vtscore.datasets.stages.embedding import embedder_display_name  # noqa: PLC0415

    if not medias:
        return []
    carried: Counter[str] = Counter()
    for media in medias.values():
        carried.update(media_embedder_names(media))
    spaces = [name for name, count in carried.items() if count * 2 > len(medias)]
    absent_of = {
        mid: absent
        for mid, media in medias.items()
        if (absent := [n for n in spaces if media_embedding(media, n) is None])
    }
    if not absent_of:
        return []

    import logging  # noqa: PLC0415

    log = logging.getLogger(__name__)
    if len(absent_of) * 2 >= len(medias):
        log.warning(
            "%d of %d media are missing a vector the rest of the dataset carries; too many to repair, "
            "so none were removed",
            len(absent_of),
            len(medias),
        )
        return []

    reason_of = {
        mid: f"Missing the {embedder_display_name(absent[0])} vector the rest of the dataset has"
        for mid, absent in absent_of.items()
    }
    rank = {reason: i for i, (reason, _) in enumerate(Counter(reason_of.values()).most_common())}
    ordered = sorted(absent_of, key=lambda mid: rank[reason_of[mid]])
    dropped = _dropped_groups(
        [_dropped_item_label(mid, medias[mid]) for mid in ordered], [reason_of[mid] for mid in ordered]
    )
    for mid in ordered:
        del medias[mid]
    log.warning("Removed %d media missing a vector the rest of the dataset carries", len(ordered))
    return dropped


def _drop_none_embeddings_stage(
    ctx: DatasetContext, tracker, failures: EmbedFailures | None = None
) -> list[DroppedGroup]:
    """Drop any media that finished the clipper stage without an embedding.

    ``_fixup_clip_md5_and_embeddings`` is best-effort: when its bulk
    re-embed call fails (no embedder, vector ``None``, exception) the
    clip is left in ``ctx.medias`` with no vector.  Letting
    those through poisons every downstream consumer; the matrix builder
    in ``vtscore/embedding/matrix.py`` raises (M11 fix); sort/score
    aggregations get wrong-length lists.  Drop them here so the rest of
    the load pipeline (dedup, coverage atlas, registry) sees a clean
    dict, and surface the count to the progress tracker so the user
    knows N is lower than the importer reported.

    *failures* is what the embed stage recorded about why each item has no
    vector (see :func:`~vtscore.datasets.stages.embedding._embed_missing_stage`).
    The warning toast states those reasons and lists every dropped item by
    name, since a GUI user has no other way to find out which ones (#4232).

    Returns the dropped items grouped by reason (empty when none were), for
    the registry stage to keep on the dataset's entry: the toast is a one-off
    message that a page reload or a missed glance loses, and the Stats window
    reads the list back from the entry (#3798).
    """
    none_ids = [cid for cid, media in ctx.medias.items() if media_embedding(media) is None]
    if not none_ids:
        return []

    failures = failures or {}
    reason_of = {cid: failures.get(cid, _UNKNOWN_DROP_REASON) for cid in none_ids}
    reasons = Counter(reason_of.values())
    # Grouped by reason (commonest first) so a mixed failure reads as blocks.
    rank = {reason: i for i, (reason, _) in enumerate(reasons.most_common())}
    ordered = sorted(none_ids, key=lambda cid: rank[reason_of[cid]])
    labels = [_dropped_item_label(cid, ctx.medias[cid]) for cid in ordered]
    dropped = _dropped_groups(labels, [reason_of[cid] for cid in ordered])
    if len(reasons) > 1:
        # One reason is already the detail line; several need saying per item.
        labels = [f"{label} — {reason_of[cid]}" for label, cid in zip(labels, ordered)]

    for cid in none_ids:
        del ctx.medias[cid]

    import logging  # noqa: PLC0415

    from vtscore.concurrency.notifications import notify  # noqa: PLC0415

    logging.getLogger(__name__).warning(
        "Dropped %d media item(s) with no embedding (importer or re-embed step failed)",
        len(none_ids),
    )
    # The tracker message below is overwritten by the next stage within
    # seconds, so a load that silently shrank was indistinguishable from one
    # that didn't (issue #3798).  A warning toast stays until dismissed.
    notify(
        f"Dropped {len(none_ids)} item(s) whose embedding failed",
        level="warning",
        # The toast is gone on a reload; the dataset's Stats window keeps the list.
        detail=_drop_summary(len(none_ids), len(none_ids) + len(ctx.medias), reasons)
        + " The dataset's Stats window keeps the list.",
        items=labels,
        source="Dataset import",
    )
    tracker.update(
        "loading",
        f"Dropped {len(none_ids)} item(s) with failed embedding…",
        current=0,
        total=0,
        step=_TOTAL_LOAD_STEPS,
        total_steps=_TOTAL_LOAD_STEPS,
    )
    invalidate_embedding_matrix(ctx)
    return dropped


def _collapse_duplicates_stage(ctx: DatasetContext, tracker) -> None:
    def _progress(current: int, total: int) -> None:
        tracker.check_cancelled()
        tracker.update(
            "loading",
            "Removing duplicates…",
            current=current,
            total=total,
            step=_TOTAL_LOAD_STEPS,
            total_steps=_TOTAL_LOAD_STEPS,
        )

    _progress(0, 0)
    collapse_duplicates(ctx.medias, on_progress=_progress)
    invalidate_embedding_matrix(ctx)


def _collapse_near_duplicates_stage(ctx: DatasetContext, tracker) -> None:
    """Opt-in pass: collapse near-duplicate images/text after exact dedup.

    Gated on the transient ``ctx.merge_near_duplicates`` create-time flag
    (set from the importer modal's "Merge near-duplicates" checkbox).  No-op
    on every reload path, where the flag defaults off and the grouping is
    already baked into origins.
    """
    if not getattr(ctx, "merge_near_duplicates", False):
        return

    def _progress(current: int, total: int) -> None:
        tracker.check_cancelled()
        tracker.update(
            "loading",
            "Merging near-duplicates…",
            current=current,
            total=total,
            step=_TOTAL_LOAD_STEPS,
            total_steps=_TOTAL_LOAD_STEPS,
        )

    _progress(0, 0)
    collapse_near_duplicates(ctx.medias, on_progress=_progress)
    invalidate_embedding_matrix(ctx)


def _follow_source_merges_stage(ctx: DatasetContext, source_siblings: list[dict[int, dict[str, Any]]]) -> None:
    """Collapse converter outputs whose sources a sibling dataset merged.

    *source_siblings* are the finished medias of the datasets this one's
    converter rows read from, in the same multi-dataset import (the Image
    dataset, for a Face dataset); empty for every other load, where this is a
    no-op.  Runs whether or not near-duplicate merging was asked for: it
    follows whatever the sibling merged, exact duplicates included.  See
    :func:`~vtscore.media.near_dupes.follow_source_merges`.
    """
    collapsed = 0
    for source_medias in source_siblings:
        collapsed += follow_source_merges(ctx.medias, source_medias)
    if collapsed:
        invalidate_embedding_matrix(ctx)


def _build_coverage_atlas_stage(ctx: DatasetContext, tracker) -> None:
    # An upstream step (e.g. a pickle restore) may have already populated the
    # tree; skip the expensive hierarchical k-means rebuild when so.
    if ctx.coverage_atlas is not None:
        return

    # Past the auto-build threshold the tree is deferred: the build would cost
    # minutes/GBs and the autopilot degrades gracefully without it.  The user
    # can trigger it later via POST /api/datasets/registry/<id>/coverage-atlas.
    if not should_auto_build_coverage_atlas(len(ctx.medias)):
        import logging  # noqa: PLC0415

        logging.getLogger(__name__).info(
            "Skipping automatic coverage-atlas build for %d items (> threshold); "
            "build on demand via the coverage-atlas endpoint.",
            len(ctx.medias),
        )
        return

    def _progress(current: int, total: int) -> None:
        tracker.check_cancelled()
        tracker.update(
            "loading",
            "Building diversity index…",
            current=current,
            total=total,
            step=_TOTAL_LOAD_STEPS,
            total_steps=_TOTAL_LOAD_STEPS,
        )

    _progress(0, 0)
    build_coverage_atlas_for_context(ctx, on_progress=_progress)
