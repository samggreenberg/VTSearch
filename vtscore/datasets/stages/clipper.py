"""Clipper / converter chain stage and per-clip MD5 + embedding fixup.

Applies a clipper (or full clipper-chain) to every media in a context, then
repairs the derived sub-items: a fresh content-based MD5 so dedup keeps
distinct clips apart, refreshed thumbnails for audio/video tiles, and a fresh
embedding computed from the clipped content instead of the parent's.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from vtscore.embedding.matrix import invalidate_embedding_matrix
from vtscore.embedding.media_vectors import media_embedding

from vtscore.datasets.stages._common import _STATUS_TO_STEP, _TOTAL_LOAD_STEPS
from vtscore.utils.hashing import content_md5

if TYPE_CHECKING:
    from vtscore.state import DatasetContext


def _stamp_default_clipper(clips_dict: dict, clipper_name: str, clipper_params: dict | None) -> None:
    """Stamp legacy ``clipper`` / ``clipper_<param>`` keys for a ``*_default`` clipper.

    A single ``*_default`` clipper is a no-op on the data but records the
    clipper name and its resolved parameters in every media's
    ``origin.params``.  We copy the ``origin`` (and its ``params``) before
    mutating so we never write through to a shared dict.  Unknown clipper
    names are a no-op (legacy semantics).
    """
    from vtscore.datasets.clipper_chain import CLIPPER_BASE_KEYS  # noqa: PLC0415
    from vtscore.media import get_clipper  # noqa: PLC0415

    try:
        clipper = get_clipper(clipper_name)
    except KeyError:
        return
    if clipper_params:
        clipper = clipper.with_params(clipper_params)
    resolved_dict = clipper.to_dict()
    effective_params = {k: v for k, v in resolved_dict.items() if k not in CLIPPER_BASE_KEYS}
    for media in clips_dict.values():
        orig = media.get("origin")
        if isinstance(orig, dict):
            media["origin"] = dict(orig)
            media["origin"]["params"] = dict(orig.get("params", {}))
            media["origin"]["params"]["clipper"] = clipper.name
            for pk, pv in effective_params.items():
                media["origin"]["params"][f"clipper_{pk}"] = str(pv)


def _prepend_legacy_clipper(
    clips_dict: dict,
    clipper_name: str,
    clipper_params: dict | None,
    steps: list[dict],
) -> list[dict]:
    """Fold the legacy single-clipper args into *steps*, returning the new list.

    Called only when *steps* carries no clipper / converter step of its own, so
    the legacy args are the clipper choice for this load.  Three outcomes:

    * a ``*_default`` clipper stamps the legacy ``clipper`` / ``clipper_<key>``
      origin keys and adds no step (a default clipper is a data no-op, and
      keeping it out of the chain is what stops ``clipper_chain`` from being
      written for plain loads);
    * an unknown clipper name adds no step, matching the legacy no-op
      semantics - any cleaner steps in *steps* still run, because a stale
      clipper name is no reason to silently skip the user's cleanup gates;
    * anything else leads the chain as a single clipper step.
    """
    if clipper_name.endswith("_default"):
        _stamp_default_clipper(clips_dict, clipper_name, clipper_params)
        return steps

    from vtscore.media import get_clipper  # noqa: PLC0415

    try:
        get_clipper(clipper_name)
    except KeyError:
        return steps
    return [{"kind": "clipper", "name": clipper_name, "params": dict(clipper_params or {})}, *steps]


def _apply_clipper(
    clips_dict: dict,
    clipper_name: str,
    clipper_params: dict | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    chain_steps: list[dict] | None = None,
    embedder=None,
) -> None:
    """Apply a clipper (or full chain) to all medias in *clips_dict*, in place.

    Either *clipper_name* (single-step legacy path) or *chain_steps*
    (ordered list of converter/clipper/cleaner steps, see
    ``docs/plans/clipper-chain.md`` and ``docs/plans/media-cleaners.md``)
    may be supplied.  When *chain_steps* already carries a clipper or
    converter step it wins outright; when it carries only *cleaner* steps
    (the shape :func:`~vtscore.datasets.clipper_chain.append_cleaner_steps`
    produces for an import with cleanup gates but no clipper chain) the
    legacy single-clipper args still describe the clipper choice and are
    folded in ahead of the cleaners.

    After running the chain, each clip gets:
    - A recomputed MD5 based on its actual content (so that dedup doesn't
      collapse distinct clips from the same parent).
    - The full chain trail in ``origin.params['clipper_chain']`` plus
      legacy single-clipper keys (``clipper``, ``clipper_<param>``,
      ``clip_start``/``clip_end``/``clip_box``/``clip_index``) describing
      the last clipper step.
    - A fresh embedding computed from the clipped content (audio/image/text)
      instead of inheriting the parent's embedding.

    Any importer-provided MD5 or embedding on the *parent* media is
    discarded for clips produced by a non-trivial chain, since those
    values describe the full media item, not the sub-item.

    Args:
        on_progress: Optional callback ``(current, total, phase)`` invoked
            during clipping and re-embedding so callers can report progress.
        embedder: Optional embedder used to re-embed the produced clips.
            When ``None`` (the default), the first registered embedder for
            the final media type is used.  Callers that loaded the parent
            media with a specific embedder (e.g. the demo loader honouring a
            user-chosen embedder) pass it here so clips are embedded with the
            same model the parents were.
    """
    from vtscore.datasets.clipper_chain import apply_chain_to_clips, normalise_chain

    # Resolve the effective step list. A `chain_steps` carrying a real
    # transform (clipper / converter) takes precedence; otherwise build a
    # length-1 clipper chain from the legacy single-clipper args (if any) and
    # keep any cleaner steps behind it.
    steps = normalise_chain(chain_steps)
    transforms = any(s["kind"] in ("clipper", "converter") for s in steps)
    if not transforms and clipper_name:
        steps = _prepend_legacy_clipper(clips_dict, clipper_name, clipper_params, steps)
    if not steps:
        return

    # Reference (thin) parents carry no bytes, so audio/image clippers - which
    # read ``media_bytes`` to compute boundaries and slice - would early-return
    # the media unchanged.  Transiently hydrate those parents from their source
    # files so clipping (and the per-clip MD5/embed/thumbnail fixup below) runs
    # exactly as it does in full mode.  Each hydrated parent is tagged with a
    # ``_lazy_source`` marker that rides into its clips; those clips are
    # re-lazified by ``_relazify_reference_clips_stage`` after the embed stages
    # so the dataset never stores duplicated clip bytes.
    _hydrate_reference_parents(clips_dict, steps)

    result = apply_chain_to_clips(clips_dict, steps, on_progress=on_progress)
    if result is None:
        return
    final_type, needs_recompute = result

    clips_list = list(clips_dict.values())
    _fixup_clip_md5_and_embeddings(clips_list, needs_recompute, final_type, on_progress=on_progress, embedder=embedder)
    _regenerate_clip_thumbnails(clips_list, needs_recompute, final_type)

    # Update `media_type` on every clip so downstream code sees the final type
    # (converter chains change the media_type of the carriers).
    for clip in clips_dict.values():
        clip["media_type"] = final_type


def _hydrate_reference_parents(clips_dict: dict, steps: list[dict]) -> bool:
    """Materialize bytes on reference (thin) parents so clippers can run.

    A *reference* media carries ``media_path`` but no ``media_bytes`` /
    ``media_string`` (it was imported with ``reference_files`` / ``thin``).
    Audio and image clippers need the actual bytes to compute clip boundaries
    and slice, so we load the source via the media type's ``load_media_data``
    (which also fills ``duration`` / ``width`` / ``height`` / ``thumbnail``,
    skipped at thin-import time) and tag the parent with a ``_lazy_source``
    marker.  The marker rides along into every derived clip via the clipper's
    ``dict(media)`` copy, so :func:`_relazify_reference_clips` can later strip
    those clips back to references.

    Lazy clips only apply to pure same-type clipper chains; a chain containing
    a converter changes media type, so the clip's bytes are no longer a slice
    of the original source file.  Such chains are left fully materialized
    (lazy converter output is out of scope - see
    ``docs/plans/server-dedup-references.md``).

    Returns ``True`` if at least one parent was hydrated.
    """
    from vtscore.media.lazy_clip import LAZY_CLIP_TYPES  # noqa: PLC0415

    if any(step.get("kind") == "converter" for step in steps):
        return False

    import vtscore.media as media_registry  # noqa: PLC0415

    any_hydrated = False
    for media in clips_dict.values():
        if media.get("media_bytes") is not None or media.get("media_string") is not None:
            continue
        media_type = media.get("media_type")
        if media_type not in LAZY_CLIP_TYPES:
            continue
        path = media.get("media_path")
        if not path or not Path(path).exists():
            continue
        try:
            mt = media_registry.get(media_type)
            media.update(mt.load_media_data(Path(path)))
        except Exception:
            import logging as _logging  # noqa: PLC0415

            _logging.getLogger(__name__).warning(
                "lazy clip: failed to hydrate reference parent %s; clipping it as-is", path, exc_info=True
            )
            continue
        media["_lazy_source"] = path
        any_hydrated = True
    return any_hydrated


def _relazify_reference_clips_stage(ctx: DatasetContext, tracker=None) -> None:
    """Strip materialized bytes from reference-derived media (clips + converter output).

    Two producers tag media with the ``_lazy_source`` marker carrying the
    source file path:

    * **clips** descended from a hydrated reference parent (see
      :func:`_hydrate_reference_parents`) - audio slices, image crops.
    * **converter output** emitted in reference mode by
      :func:`~vtscore.converters.runner.run_converters_on_folder` - rendered
      PDF pages, extracted video frames.

    For each, we drop its ``media_bytes`` / ``media_string`` and point
    ``media_path`` back at the source file; ``_resolve_media_bytes`` reproduces
    the bytes on demand from the recipe in ``origin.params`` (a clip slice, a
    converter re-render) or by reading the whole file (a pass-through clip the
    clipper returned unchanged).  The per-item MD5, embedding, and thumbnail
    were already computed from the real bytes, so the reference media is
    byte-for-byte equivalent without the stored payload.

    Runs after the embed / drop-none stages so the embed-missing safety net
    sees real bytes: a derived item's ``media_path`` points at the *whole*
    source file (the parent recording, the source PDF), not the slice/page, so
    embedding it lazily would embed the wrong content.

    **Cleaned items are never re-lazified.**  A cleaner rewrites the payload
    (re-encoded bytes, trimmed text) rather than slicing it, and
    :mod:`vtscore.media.lazy_clip` has no recipe that reproduces that from the
    source file, so dropping the bytes would silently serve and re-embed the
    *uncleaned* original.  Those items keep their payload materialized; a thin
    import therefore loses its byte-savings for exactly the items some cleaner
    actually changed.
    """
    from vtscore.datasets.clipper_chain import has_original_payload  # noqa: PLC0415

    for clip in ctx.medias.values():
        source = clip.pop("_lazy_source", None)
        if source is None:
            continue
        if has_original_payload(clip):
            continue
        clip["media_path"] = source
        clip["media_bytes"] = None
        clip["media_string"] = None


def _thumb_for(clip: dict, media_type: str) -> bytes | None:
    """Compute a refreshed thumbnail for one clip, or ``None`` to leave it.

    For ``audio`` clips, render a waveform thumbnail from the clip's
    ``media_bytes`` (or its ``media_path`` when bytes aren't held).  For
    ``video`` clips, render a frame at the midpoint of
    ``clip_start``/``clip_end``; a clip missing either boundary yields
    ``None``.  Other media types yield ``None``.
    """
    if media_type == "audio":
        from vtscore.media.audio.media_type import (
            generate_waveform_thumbnail,
            generate_waveform_thumbnail_from_file,
        )

        wav = clip.get("media_bytes")
        if wav is not None:
            return generate_waveform_thumbnail(wav)
        path = clip.get("media_path")
        if path:
            return generate_waveform_thumbnail_from_file(Path(path))
        return None

    if media_type == "video":
        from vtscore.media.video.media_type import (
            generate_video_thumbnail_at,
            generate_video_thumbnail_from_file_at,
        )

        t0 = clip.get("clip_start")
        t1 = clip.get("clip_end")
        if t0 is None or t1 is None:
            return None
        mid = (float(t0) + float(t1)) / 2.0
        video_bytes = clip.get("media_bytes")
        if video_bytes is not None:
            return generate_video_thumbnail_at(video_bytes, mid)
        path = clip.get("media_path")
        if path:
            return generate_video_thumbnail_from_file_at(Path(path), mid)
        return None

    return None


def _regenerate_clip_thumbnails(
    clips: list[dict],
    needs_recompute: list[bool],
    media_type: str,
) -> None:
    """Refresh ``thumbnail_bytes`` on clipped audio/video sub-items.

    Audio and video clippers copy ``thumbnail_bytes`` verbatim from the parent
    media; without this fixup, every sub-item would render the parent's
    waveform or middle-frame thumbnail in the find/label list.

    Image clips don't go through this path; their thumbnail is the cropped
    ``media_bytes`` itself, served directly by the media-image route.
    """
    for clip, recompute in zip(clips, needs_recompute):
        if not recompute:
            continue
        thumb = _thumb_for(clip, media_type)
        if thumb is not None:
            clip["thumbnail_bytes"] = thumb


def _fixup_clip_md5_and_embeddings(  # noqa: C901
    clips: list[dict],
    needs_recompute: list[bool],
    media_type: str,
    on_progress: Callable[[int, int, str], None] | None = None,
    embedder=None,
) -> None:
    """Recompute MD5 and embeddings for clips that need it.

    A clip needs recomputation when:
    - ``needs_recompute`` is ``True`` (genuine sub-item from a multi-output
      clipper (the parent's MD5 and embedding are stale), **or**
    - the clip has no embedding at all (import phase was skipped because a
      clipper was going to re-embed anyway).

    For any clip reaching this fixup with new content bytes (audio/image/
    text), the MD5 is always rehashed from those final bytes (including
    single-output clippers that copy the parent dict via ``dict(media)``
    and would otherwise carry the parent's stale MD5 forward.

    Without the MD5 fix, clips from the same parent would share the
    parent's MD5 (causing ``collapse_duplicates`` to merge them).

    Embeddings are batched through ``embed_media_bulk`` in a single call
    per invocation so GPU-backed embedders can fuse the forward pass.
    Failures fall back to keeping the clip's existing embedding.
    """

    total_clips = len(clips)
    embed_indices: list[int] = []
    embed_inputs: list[dict] = []
    for clip_idx, (clip, recompute) in enumerate(zip(clips, needs_recompute)):
        # Also embed clips that have no embedding (e.g. when the import
        # phase skipped embedding because a clipper was specified).
        needs_embed = recompute or media_embedding(clip) is None
        if not needs_embed:
            continue

        content_bytes = _clip_content_bytes(clip, media_type)
        metadata_only = content_bytes is None and media_type == "video"
        if content_bytes is None and not metadata_only:
            continue

        # Always recompute MD5 from the final clip bytes whenever we're
        # also recomputing the embedding. Any single-output clipper that
        # rewrites media_bytes (e.g. ImageObjectClipper with one detection,
        # ImageBboxClipper) would otherwise inherit the parent's MD5 via
        # ``dict(media)`` and cause dedup to merge distinct crops.
        if content_bytes is not None:
            clip["md5"] = content_md5(content_bytes)
        elif recompute:
            # Metadata-only clips (e.g. video): bytes unchanged but
            # boundaries differ; create a unique MD5 by hashing the
            # parent bytes + clip boundaries so dedup doesn't collapse
            # distinct clips.
            parent_bytes = clip.get("media_bytes", b"")
            boundary_tag = f"|clip_start={clip.get('clip_start')}|clip_end={clip.get('clip_end')}"
            combined = content_md5(parent_bytes) + boundary_tag
            clip["md5"] = content_md5(combined.encode())
        embed_indices.append(clip_idx)
        embed_inputs.append(_build_clip_embed_input(clip, media_type))

    if not embed_indices:
        return

    if on_progress:
        on_progress(0, total_clips, "embedding")

    embedder = embedder or _resolve_clip_embedder(media_type)
    if embedder is None:
        return

    def _clip_progress(status: str, message: str = "", current: int = 0, total: int = 0) -> None:
        if not on_progress:
            return
        if status == "loading":
            # The model is loaded lazily on the first embed call, *after*
            # the "Embedding clips…" line is already on screen. While it
            # loads (and on first run downloads weights + JIT-warms up the
            # pipeline) the embedder emits descriptive "loading" messages
            # ("Loading … model weights…", "Warming up audio pipeline…").
            # Forward those verbatim — with the embedder's own current/total —
            # so the user sees what's actually happening instead of a frozen
            # "Embedding clips… (0/N)". Scaling these load sub-steps against
            # the clip list would be meaningless.
            on_progress(current, total, message or "Loading model…")
            return
        # Real per-clip progress: map the embedder's batch-level progress back
        # to clip-list coordinates so the existing on_progress(current, total,
        # phase) contract keeps reporting against total_clips.
        scaled = min(total_clips, int(current * len(embed_indices) / max(1, total)))
        on_progress(scaled, total_clips, "embedding")

    original_cb = embedder._on_progress
    embedder._on_progress = _clip_progress
    try:
        vectors = embedder.embed_media_bulk(embed_inputs)
    except Exception:
        import logging as _logging

        _logging.getLogger(__name__).exception(
            "Bulk clip re-embed failed for media_type=%s (%d clips)", media_type, len(embed_indices)
        )
        return
    finally:
        embedder._on_progress = original_cb

    for slot, vec in zip(embed_indices, vectors):
        if vec is not None:
            # Assign a *fresh* embeddings dict rather than mutating in place:
            # clips are shallow copies (``dict(media)``) that share the parent's
            # ``embeddings`` dict by reference, so an in-place write would corrupt
            # the parent's vector.  A re-embedded clip is single-embedder by
            # construction (the resolved clip embedder), so a one-entry dict is
            # the complete store.
            clip = clips[slot]
            clip["embeddings"] = {embedder.name: vec}
            clip["embedder"] = embedder.name

    if on_progress:
        on_progress(total_clips, total_clips, "embedding")


def _clip_content_bytes(clip: dict, media_type: str) -> bytes | None:
    """Return the embeddable content bytes for a clip.

    For media types where the clipper produces new bytes (audio, image) or
    a new string (text), return those bytes so the caller can hash and
    re-embed them.  For metadata-only clips (video), return ``None``;
    the caller will use a boundary-based hash instead.
    """
    if media_type == "video":
        # Video clippers store boundaries as metadata without slicing the
        # underlying bytes, so there is nothing new to hash/embed - unless a
        # cleaner rewrote the payload (it carries a pre-clean snapshot), in
        # which case the bytes really are this unit's own content.
        from vtscore.datasets.clipper_chain import has_original_payload  # noqa: PLC0415

        if has_original_payload(clip) and clip.get("media_bytes") is not None:
            return clip["media_bytes"]
        return None
    if clip.get("media_bytes") is not None and media_type != "text":
        return clip["media_bytes"]
    if clip.get("media_string") is not None and media_type == "text":
        # A blank / whitespace-only clip has no embeddable content. Treat it
        # as "no content" so the caller skips embedding it: the clip keeps
        # ``embedding=None`` and is removed by ``_drop_none_embeddings_stage``
        # rather than reaching the embedding-matrix builder (M21). This also
        # guards against an embedder that returns a non-None garbage vector
        # for empty input, which the None-drop net would otherwise miss.
        text = clip["media_string"]
        if not text.strip():
            return None
        return text.encode("utf-8")
    return None


def _build_clip_embed_input(clip: dict, media_type: str) -> dict:
    """Build the minimal media dict a bulk embedder needs for a clip.

    Hands the embedder the in-memory ``media_bytes`` (audio/image/video) or
    ``media_string`` (text) so the bulk surface never has to round-trip
    the content through a tempfile.  Preserves ``origin_name`` and
    ``filename`` so embedders that surface diagnostic paths still log
    something useful.

    Video clips additionally carry ``clip_start`` / ``clip_end`` (and
    ``media_path`` when available) because the underlying parent bytes
    are shared across every tile; the embedder uses the boundary
    metadata to sample distinct frame ranges per tile.
    """
    base: dict = {
        "origin_name": clip.get("origin_name", ""),
        "filename": clip.get("filename", ""),
    }
    if media_type == "text":
        base["media_string"] = clip.get("media_string", "")
    else:
        base["media_bytes"] = clip.get("media_bytes")
    if media_type == "video":
        if clip.get("media_path"):
            base["media_path"] = clip["media_path"]
        if "clip_start" in clip:
            base["clip_start"] = clip["clip_start"]
        if "clip_end" in clip:
            base["clip_end"] = clip["clip_end"]
    return base


def _resolve_clip_embedder(media_type: str):
    """Pick the default embedder for *media_type* used by clip re-embed.

    Mirrors the legacy ``embed_file`` fallback chain: first registered
    embedder for the media type, or ``None`` if none are registered.
    """
    try:
        from vtscore.media import embedders_for_type  # noqa: PLC0415
    except ImportError:
        return None
    avail = embedders_for_type(media_type)
    if not avail:
        import logging as _logging

        _logging.getLogger(__name__).warning(
            "clip re-embed: no embedders registered for media_type=%r; skipping bulk call",
            media_type,
        )
        return None
    return avail[0]


def _apply_clipper_stage(
    ctx: DatasetContext, tracker, clipper: str, clipper_params: dict | None, chain_steps: list[dict] | None
) -> None:
    """Run the clipper / chain stage with tracker-routed progress."""
    if not (clipper or chain_steps):
        return

    def _clipper_progress(current: int, total: int, phase: str) -> None:
        tracker.check_cancelled()
        if phase == "clipping":
            msg = "Clipping media…"
        elif phase == "converting":
            msg = "Converting media…"
        elif phase == "cleaning":
            msg = "Cleaning media…"
        elif phase == "embedding":
            msg = "Embedding clips…"
        else:
            # A loading/warmup message forwarded verbatim from the embedder
            # (e.g. "Loading CLAP model weights…", "Warming up audio
            # pipeline…") while the model loads on the first embed call.
            msg = phase
        # Clipping is the embed phase for clipped datasets: it cuts segments
        # and embeds the resulting clips (the standard `_embed_missing_stage`
        # is a no-op afterwards). So it reports the *embedding* step, not the
        # finalize step. Reporting finalize here ran the bar up to ~100% before
        # embedding even began and then — because clipping runs *before* the
        # embed step — tripped the "step went backwards = new job" reset, which
        # visibly slammed the unified bar from ~100% back to mid-job. Keeping it
        # on the embed step makes the whole-job fraction monotonic.
        tracker.update(
            "loading",
            msg,
            current=current,
            total=total,
            step=_STATUS_TO_STEP["embedding"],
            total_steps=_TOTAL_LOAD_STEPS,
        )

    _clipper_progress(0, 0, "clipping")
    _apply_clipper(ctx.medias, clipper, clipper_params, on_progress=_clipper_progress, chain_steps=chain_steps)
    # Clipping reassigns media IDs starting from 1 and rewrites embeddings
    # in place; the cached matrix's id list can collide with the new one
    # while pointing at stale rows. Drop the cache so the next reader
    # rebuilds against the live medias dict.
    invalidate_embedding_matrix(ctx)
