"""Demo dataset loader.

Wraps each media type's :meth:`MediaType.load_demo_source` with pickle
caching and origin-stamping.  Split out from
:mod:`vtscore.datasets.loader` for navigability.

To keep the existing test patches working
(``patch("vtscore.datasets.loader.load_dataset_from_pickle", ...)`` and
``patch("vtscore.datasets.loader.EMBEDDINGS_DIR", ...)``), the call sites
go through the parent ``loader`` module so the lookup happens at call time.
"""

from __future__ import annotations

import pickle
from typing import Any, Optional

import numpy as np

from vtscore.datasets.config import DEMO_DATASETS
from vtscore.datasets.loader import (
    ProgressCallback,
    _default_progress,
)


def _effective_clipper(clipper_name: str, media_type_id: str) -> str:
    """Return *clipper_name* if it is a real (non-default) clipper, else ``""``.

    The demo UI always sends *a* clipper name (the first one in the
    media-type's list is pre-selected), but the conventional "default"
    choice is a no-op that should not split the media.  Mirrors the
    frontend's ``isDefaultClipper`` rule so that the two sides agree on
    when clipping actually happens: a clipper is a no-op when it is empty,
    ends with ``_default``, or is the first clipper registered for the
    media type (the pre-selected default).  Normalising to ``""`` here also
    means legacy pickles that recorded the pre-selected default name (e.g.
    ``"sound_tiling"``) compare equal to "no clipper" and don't trigger a
    needless re-embed.
    """
    if not clipper_name or clipper_name.endswith("_default"):
        return ""
    from vtscore.media import clippers_for_type  # noqa: PLC0415

    clippers = clippers_for_type(media_type_id)
    if clippers and clippers[0].name == clipper_name:
        return ""
    return clipper_name


def _stamp_demo_origin(
    medias: dict[int, dict[str, Any]],
    dataset_name: str,
    converter_name: str = "",
) -> None:
    """Stamp the demo origin on all medias (fresh dict per media).

    Ensures every media has ``origin = {"importer": "demo", "params": {"name": ...}}``.
    """
    demo_origin_params: dict[str, str] = {"name": dataset_name}
    if converter_name:
        demo_origin_params["converter"] = converter_name
    for media in medias.values():
        media["origin"] = {"importer": "demo", "params": dict(demo_origin_params)}


def _stamp_media_type(
    medias: dict[int, dict[str, Any]],
    source_type_id: str,
    converter_name: str = "",
) -> None:
    """Fill in missing ``media_type`` on cached medias.

    Old pkl files may have been created before ``media_type`` was stored
    per-media, causing ``load_dataset_from_pickle`` to fall back to the
    ``"audio"`` default for every item.  This corrects any media whose
    ``media_type`` is absent or empty so the dataset registers with the
    right type.
    """
    expected = source_type_id
    if converter_name:
        from vtscore.converters import get_converter  # noqa: PLC0415

        conv = get_converter(converter_name)
        if conv is not None:
            expected = conv.target_type
    for media in medias.values():
        if not media.get("media_type"):
            media["media_type"] = expected


def _try_load_cached(
    pkl_file: Any,
    dataset_name: str,
    media_type_id: str,
    medias: dict[int, dict[str, Any]],
    on_progress: ProgressCallback,
    embedder_name: str,
    converter_name: str,
    clipper_name: str,
    clipper_params: dict[str, Any] | None,
) -> bool:
    """Try to satisfy the load from the cached ``.pkl`` file.

    Returns ``True`` when the cache was a hit: *medias* has been populated,
    origins/media-types stamped, and the final "Loaded" progress emitted, so
    the caller should return immediately.  Returns ``False`` when the cache
    was missing, stale (embedder/clipper mismatch), or empty on load; in that
    case the stale pickle (if any) has been unlinked and the caller should
    rebuild from scratch.
    """
    from vtscore.datasets import loader as _loader

    if not pkl_file.exists():
        return False

    # If the caller explicitly requested an embedder, verify the cached
    # pickle was produced by the same one.  When *embedder_name* is empty
    # (meaning "use default"), accept whatever is cached.
    from vtscore.datasets.container import read_meta

    cached_meta = read_meta(pkl_file)
    cached_embedder = cached_meta.get("embedder") or ""
    embedder_mismatch = bool(embedder_name) and bool(cached_embedder) and embedder_name != cached_embedder
    # A cached pickle built with a different clipper (or different clipper
    # params) no longer reflects the requested split, so rebuild it.  Both
    # names are normalised through _effective_clipper so the pre-selected
    # default ("" vs e.g. "sound_tiling") never counts as a mismatch.
    requested_clipper = _effective_clipper(clipper_name, media_type_id)
    cached_clipper = _effective_clipper(cached_meta.get("clipper") or "", media_type_id)
    requested_params = clipper_params or {}
    cached_params = cached_meta.get("clipper_params") or {}
    clipper_mismatch = requested_clipper != cached_clipper or (
        bool(requested_clipper) and requested_params != cached_params
    )
    if embedder_mismatch or clipper_mismatch:
        reason = f"with {embedder_name}" if embedder_mismatch else "with new clipper"
        on_progress("loading", f"Re-embedding {dataset_name} {reason}...", 0, 0)
        pkl_file.unlink()
        return False

    on_progress("loading", f"Loading {dataset_name} dataset...", 0, 0)
    _loader.load_dataset_from_pickle(pkl_file, medias)

    # Check if any medias were actually loaded
    if len(medias) == 0:
        # Pickle file exists but media files are missing, delete and re-embed
        on_progress("loading", f"Media files missing, re-embedding {dataset_name}...", 0, 0)
        pkl_file.unlink()
        return False

    # Stamp demo origin on cached medias so that cross-dataset
    # resolution always has the dataset name in the origin params.
    # Old pickles (created before origin stamping) may have empty
    # params - this ensures they are corrected on load.
    _stamp_demo_origin(medias, dataset_name, converter_name)
    # Fill in missing media_type: old pkls may lack the field,
    # causing every item to fall back to the "audio" default in
    # load_dataset_from_pickle and the dataset to register with
    # the wrong type.
    _stamp_media_type(medias, media_type_id, converter_name)
    on_progress("idle", f"Loaded {dataset_name} dataset", 0, 0)
    return True


def _resolve_demo_embedder(embedder_name: str, media_type_id: str) -> Any:
    """Resolve the embedder to use for a demo load.

    When *embedder_name* is given, look it up (raising ``ValueError`` for an
    unknown name).  Otherwise fall back to the first registered embedder for
    *media_type_id*, or ``None`` when the media type has none.
    """
    from vtscore.media import embedders_for_type, get_embedder

    if embedder_name:
        try:
            return get_embedder(embedder_name)
        except KeyError:
            raise ValueError(f"Unknown embedder: {embedder_name}")
    avail = embedders_for_type(media_type_id)
    return avail[0] if avail else None


def load_demo_dataset(
    dataset_name: str,
    medias: dict[int, dict[str, Any]],
    on_progress: Optional[ProgressCallback] = None,
    embedder_name: str = "",
    converter_name: str = "",
    clipper_name: str = "",
    clipper_params: dict[str, Any] | None = None,
) -> None:
    """Load a named demo dataset into the medias dict, downloading and embedding as needed.

    Checks for a cached ``.pkl`` file in ``EMBEDDINGS_DIR``; if found, loads
    from that file. If the cache is missing or the media bytes it references can
    no longer be found on disk, the raw data is re-downloaded and re-embedded.

    Each media type implements its own
    :meth:`~vtscore.media.base.MediaType.load_demo_source` method that
    handles downloading, embedding, and populating clips for its demo sources.
    This function simply orchestrates pickle caching around that delegation.

    When *converter_name* is given (e.g. ``"video2image"``), the demo data is
    loaded using its original media type, then each media is converted via the
    named converter.  The resulting dataset contains the *target* type and
    is cached under a separate pickle key.

    Progress throughout the operation is reported via :func:`update_progress`.

    Args:
        dataset_name: Key into ``DEMO_DATASETS`` identifying which demo dataset
            to load.  Raises ``ValueError`` if the key is not found.
        medias: Dict to populate in-place. Existing entries are removed before
            loading. Keys are integer media IDs; values are media data dicts.
        embedder_name: Optional name of a registered embedder to use.
            When empty, the first registered embedder for the media type
            is used.
        converter_name: Optional name of a converter (e.g. ``"video2image"``).
            When given, the demo is loaded in its native type and then
            converted.
        clipper_name: Optional name of a registered clipper.  When it names
            a real (non-default) clipper, every loaded media is split into
            sub-clips via the shared clipper machinery and the clips are
            re-embedded; the clipped clips inherit their parent's category
            (and other metadata) by construction.  Recorded in the container
            metadata so a later load with a different clipper re-derives.
        clipper_params: Optional parameter overrides for *clipper_name*
            (e.g. ``{"duration": 5.0}`` for a tiling clipper).

    Raises:
        ValueError: If ``dataset_name`` is not in ``DEMO_DATASETS``, or if the
            media type does not support the requested demo source.
    """
    # Look up via the parent module so `patch("vtscore.datasets.loader.X")`
    # in tests continues to take effect at call time.
    from vtscore.datasets import loader as _loader

    if on_progress is None:
        on_progress = _default_progress()

    if dataset_name not in DEMO_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset_info = DEMO_DATASETS[dataset_name]
    media_type_id = dataset_info.get("media_type", "audio")

    # When a converter is specified, use a separate pickle cache key.
    cache_key = f"{dataset_name}__{converter_name}" if converter_name else dataset_name

    # Check if already embedded
    pkl_file = _loader.EMBEDDINGS_DIR / f"{cache_key}.pkl"
    if _try_load_cached(
        pkl_file,
        dataset_name,
        media_type_id,
        medias,
        on_progress,
        embedder_name,
        converter_name,
        clipper_name,
        clipper_params,
    ):
        return

    # Resolve the embedder
    from vtscore.media import get as media_get

    embedder = _resolve_demo_embedder(embedder_name, media_type_id)

    mt = media_get(media_type_id)

    source = dataset_info.get("source", "")
    categories = dataset_info["categories"]
    slice_start = dataset_info.get("slice_start", 0)
    slice_end = dataset_info.get("slice_end")
    slice_frac_start = dataset_info.get("slice_frac_start")
    slice_frac_end = dataset_info.get("slice_frac_end")

    medias.clear()

    # When a real (non-default) clipper is configured, the clipper stage below
    # splits every loaded media into sub-clips and re-embeds each clip from its
    # own bytes. Embedding the full parent during load would then be wasted work
    # whose result is immediately discarded - and for audio it actively breaks:
    # a parent recording longer than the embedder's window (e.g. CLAP's 10 s)
    # could fail to embed and be dropped before the clipper ever cuts it down to
    # an embeddable clip. So defer embedding to the clipper: load the parents
    # with a deferred-embed placeholder and let _apply_clipper produce the
    # vectors. Mirrors the regular import pipeline, which skips parent embedding
    # whenever a clipper is specified.
    clipper_applied = bool(_effective_clipper(clipper_name, media_type_id))
    external_dir = mt.load_demo_source(
        source=source,
        categories=categories,
        slice_start=slice_start,
        slice_end=slice_end,
        clips=medias,
        on_progress=on_progress,
        embedder=embedder,
        slice_frac_start=slice_frac_start,
        slice_frac_end=slice_frac_end,
        skip_embedding=clipper_applied,
    )

    # Stamp the demo origin on all medias
    _stamp_demo_origin(medias, dataset_name, converter_name)

    # --- Apply converter if requested ---
    if converter_name:
        from vtscore.converters.runner import apply_converter_to_demo

        apply_converter_to_demo(
            converter_name=converter_name,
            dataset_name=dataset_name,
            medias=medias,
            embedder_name=embedder_name,
            on_progress=on_progress,
        )

    # --- Apply clipper if requested ---
    # Splits each loaded media into sub-clips (re-embedding them) via the same
    # machinery the regular import pipeline uses, so clips inherit their
    # parent's category/metadata.  Skipped for the pre-selected default
    # clipper (a no-op).  Runs after any converter so it operates on the
    # final media type.
    if clipper_applied:
        from vtscore.datasets.stages.clipper import _apply_clipper

        def _clip_progress(current: int, total: int, phase: str) -> None:
            if phase == "clipping":
                msg = "Clipping media…"
            elif phase == "converting":
                msg = "Converting media…"
            elif phase == "embedding":
                msg = "Embedding clips…"
            else:
                # A loading/warmup message forwarded verbatim from the embedder.
                msg = phase
            on_progress("loading", msg, current, total)

        _clip_progress(0, 0, "clipping")
        _apply_clipper(
            medias,
            _effective_clipper(clipper_name, media_type_id),
            clipper_params,
            on_progress=_clip_progress,
            embedder=embedder,
        )

    _write_demo_cache(
        pkl_file=pkl_file,
        dataset_name=dataset_name,
        media_type_id=media_type_id,
        medias=medias,
        mt=mt,
        embedder=embedder,
        external_dir=external_dir,
        converter_name=converter_name,
        clipper_name=clipper_name,
        clipper_params=clipper_params,
        clipper_applied=clipper_applied,
    )

    on_progress("idle", f"Loaded {dataset_name} dataset", 0, 0)


def _write_demo_cache(
    *,
    pkl_file: Any,
    dataset_name: str,
    media_type_id: str,
    medias: dict[int, dict[str, Any]],
    mt: Any,
    embedder: Any,
    external_dir: Any,
    converter_name: str,
    clipper_name: str,
    clipper_params: dict[str, Any] | None,
    clipper_applied: bool,
) -> None:
    """Serialize the freshly built demo dataset to its pickle container.

    For types with external media dirs (audio, video), excludes media_bytes
    from the pickle and stores the dir path so reloading can find the files.
    When a converter was applied, external_dir is no longer relevant (the
    converted medias carry their own bytes/strings).  Likewise when a clipper
    split the media: each clip is a *slice* of its source file, not the whole
    file the external dir resolves by name, so its bytes must ride in the
    pickle (dataset pickles are the one place persisted bytes are allowed).
    """
    from vtscore.datasets import loader as _loader

    # Local import avoids a circular import at module load (loader imports
    # loader_demo before this helper is defined).
    from vtscore.datasets.loader import _embeddings_dict_for_pickle

    store_external = external_dir is not None and not converter_name and not clipper_applied

    def _pickle_media(media: dict[str, Any]) -> dict[str, Any]:
        return {
            k: _embeddings_dict_for_pickle(v) if k == "embeddings" else (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in media.items()
            if not (store_external and k in ("media_bytes", "thumbnail_bytes"))
        }

    pkl_data: dict[str, Any] = {
        "name": dataset_name,
        "medias": {cid: _pickle_media(media) for cid, media in medias.items()},
    }
    if store_external:
        pkl_data[mt.dir_key] = external_dir

    _loader.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    resolved_name = getattr(embedder, "name", "") if embedder is not None else ""
    extra_pickle_keys: dict[str, Any] = {}
    if store_external:
        extra_pickle_keys[mt.dir_key] = external_dir

    medias_pkl_bytes = pickle.dumps(pkl_data, protocol=5)
    meta = {
        "format_version": 1,
        "embedder": resolved_name,
        # Store the *effective* clipper (normalised default -> "") plus its
        # params so a later load can tell whether the cache still matches the
        # requested split.
        "clipper": _effective_clipper(clipper_name, media_type_id),
        "clipper_params": (clipper_params or {}) if clipper_applied else {},
        "media_type": media_type_id,
        "name": dataset_name,
    }
    from vtscore.datasets.container import write_container

    write_container(pkl_file, medias_pkl_bytes, meta, extra_pickle_keys=extra_pickle_keys)
