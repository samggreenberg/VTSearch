"""Converter routing for CLI autodetect scoring.

A detector declares the embedding space it needs (its ``media_type``); it does
**not** store a converter.  The converter registry already knows every route
from one media type to another, so one detector can score a dataset of mixed
source types: media whose type already matches the detector are scored
directly, and media of another type are routed through a one-hop converter
(``video2image``, ``document2image``, …) before embedding.  See
``docs/plans/cli-detector-converter.md``.

This module owns the source-type → target-type routing and the per-detector
"prepare a scoring snapshot" step the CLI pipeline runs before scoring.  It is
library-tier (pure ``vtscore``): no Flask, settings, or app imports.
"""

from __future__ import annotations

from typing import Any

from vtscore.embedding.media_vectors import media_embedding


def converter_route_for(source_type: str, target_type: str):
    """Return a converter mapping *source_type* → *target_type*, or ``None``.

    A direct type match needs no converter, so callers test ``source_type ==
    target_type`` first; this only resolves the cross-type hop.  When more than
    one converter produces *target_type* from *source_type* the first
    registered one wins, matching every other one-hop converter lookup in the
    codebase.
    """
    from vtscore.converters import list_converters_for_target  # noqa: PLC0415

    for conv in list_converters_for_target(target_type):
        if conv.source_type == source_type:
            return conv
    return None


def detector_can_score(target_type: str, source_types: set[str]) -> bool:
    """True when a detector needing *target_type* can score a dataset whose
    media carry the types in *source_types*.

    A detector matches when at least one source type is either the target type
    itself (direct) or reachable via a one-hop converter route.  A detector
    with no ``media_type`` (legacy, ``target_type == ""``) matches anything -
    the pre-converter behaviour of scoring whatever embeddings the dataset
    already holds.
    """
    if not target_type:
        return True
    return any(st == target_type or converter_route_for(st, target_type) is not None for st in source_types)


def _partition_by_route(
    source_medias: dict[int, dict[str, Any]],
    target_type: str,
) -> tuple[dict[int, dict[str, Any]], list[tuple[int, dict[str, Any]]]]:
    """Split *source_medias* into direct matches and converter outputs.

    Returns ``(direct, converted)`` where *direct* maps a source id to a media
    already of *target_type*, and *converted* is a list of ``(source_id,
    output_media)`` pairs, one per converter output (a video fans out into
    several frames).  Media of a type with no route to *target_type* are
    omitted from both.
    """
    direct: dict[int, dict[str, Any]] = {}
    converted: list[tuple[int, dict[str, Any]]] = []
    for sid, media in source_medias.items():
        st = media.get("media_type") or ""
        if st == target_type:
            direct[sid] = media
            continue
        conv = converter_route_for(st, target_type)
        if conv is None:
            continue
        for out in conv.convert_normalized(media, {}):
            out["media_type"] = target_type
            converted.append((sid, out))
    return direct, converted


def route_and_embed(
    source_medias: dict[int, dict[str, Any]],
    target_type: str,
    embedder_name: str,
) -> tuple[dict[int, dict[str, Any]], dict[int, int]]:
    """Prepare a *target_type* scoring snapshot from mixed *source_medias*.

    Returns ``(scoring_medias, scoring_to_source)`` where *scoring_medias* maps
    a fresh 1-based id to a target-typed media carrying an embedding under
    *embedder_name*, and *scoring_to_source* maps each of those ids back to the
    source media id it descends from (identity for a direct type match, the
    parent id for every converter output).

    Media whose type matches *target_type* are embedded in place (idempotent -
    a no-op when they already carry the vector).  Media of another type are
    routed through a one-hop converter, and each converter output is embedded.
    Media with no route, and any media the embedder can't embed, are dropped;
    the caller scores only what comes back.

    *embedder_name* may be empty, meaning "the dataset's primary embedder": the
    embed pass then resolves the default and the presence check reads the
    primary vector.
    """
    from vtscore.datasets.stages.embedding import embed_missing  # noqa: PLC0415

    # Resolve the vector under the detector's own embedder; an empty name means
    # "primary", which ``media_embedding(media, None)`` reads.
    emb_key = embedder_name or None

    direct, converted = _partition_by_route(source_medias, target_type)

    if direct:
        embed_missing(direct, embedder_name)
    if converted:
        # Embed the converter outputs in one bulk pass; the throwaway integer
        # keys just satisfy the ``dict[int, media]`` shape ``embed_missing``
        # iterates - only the media values matter.
        embed_missing({i: out for i, (_sid, out) in enumerate(converted)}, embedder_name)

    scoring: dict[int, dict[str, Any]] = {}
    scoring_to_source: dict[int, int] = {}
    next_id = 1
    dropped_direct = 0
    for sid, media in direct.items():
        if media_embedding(media, emb_key) is not None:
            scoring[next_id] = media
            scoring_to_source[next_id] = sid
            next_id += 1
        else:
            # A media of the right type that still has no vector after the embed
            # pass (unreadable/corrupt source, thin media with no resolvable
            # path). Drop it rather than crashing a long CLI run, mirroring the
            # load pipeline's drop-none stage, but count it so the caller can log.
            dropped_direct += 1
    for sid, out in converted:
        if media_embedding(out, emb_key) is not None:
            scoring[next_id] = out
            scoring_to_source[next_id] = sid
            next_id += 1

    if dropped_direct:
        import logging  # noqa: PLC0415

        logging.getLogger(__name__).warning(
            "converter routing: dropped %d %s media with no embedding under %r",
            dropped_direct,
            target_type,
            embedder_name or "(primary)",
        )
    return scoring, scoring_to_source
