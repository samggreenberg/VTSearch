"""Demo dataset loader.

Wraps each media type's :meth:`MediaType.load_demo_source` with pickle
caching and origin-stamping.  Split out from
:mod:`vtsearch.datasets.loader` for navigability.

To keep the existing test patches working
(``patch("vtsearch.datasets.loader.load_dataset_from_pickle", ...)`` and
``patch("vtsearch.datasets.loader.EMBEDDINGS_DIR", ...)``), the call sites
go through the parent ``loader`` module so the lookup happens at call time.
"""

from __future__ import annotations

import pickle
from typing import Any, Optional

import numpy as np

from vtsearch.datasets.config import DEMO_DATASETS
from vtsearch.datasets.loader import (
    ProgressCallback,
    _default_progress,
)


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


def load_demo_dataset(  # noqa: C901
    dataset_name: str,
    medias: dict[int, dict[str, Any]],
    on_progress: Optional[ProgressCallback] = None,
    embedder_name: str = "",
    converter_name: str = "",
    clipper_name: str = "",
) -> None:
    """Load a named demo dataset into the medias dict, downloading and embedding as needed.

    Checks for a cached ``.pkl`` file in ``EMBEDDINGS_DIR``; if found, loads
    from that file. If the cache is missing or the media bytes it references can
    no longer be found on disk, the raw data is re-downloaded and re-embedded.

    Each media type implements its own
    :meth:`~vtsearch.media.base.MediaType.load_demo_source` method that
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
        clipper_name: Optional name of a registered clipper.  Recorded in
            a ``.clipper`` sidecar next to the pickle for status tracking.

    Raises:
        ValueError: If ``dataset_name`` is not in ``DEMO_DATASETS``, or if the
            media type does not support the requested demo source.
    """
    # Look up via the parent module so `patch("vtsearch.datasets.loader.X")`
    # in tests continues to take effect at call time.
    from vtsearch.datasets import loader as _loader

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
    if pkl_file.exists():
        # If the caller explicitly requested an embedder, verify the cached
        # pickle was produced by the same one.  When *embedder_name* is empty
        # (meaning "use default"), accept whatever is cached.
        cached_embedder = _loader.read_pkl_embedder(pkl_file)
        if embedder_name and cached_embedder and embedder_name != cached_embedder:
            # Embedder mismatch — discard stale cache and re-embed below.
            on_progress("loading", f"Re-embedding {dataset_name} with {embedder_name}...", 0, 0)
            pkl_file.unlink()
            pkl_file.with_suffix(".embedder").unlink(missing_ok=True)
            pkl_file.with_suffix(".clipper").unlink(missing_ok=True)
        else:
            on_progress("loading", f"Loading {dataset_name} dataset...", 0, 0)
            _loader.load_dataset_from_pickle(pkl_file, medias)

            # Check if any medias were actually loaded
            if len(medias) == 0:
                # Pickle file exists but media files are missing, delete and re-embed
                on_progress("loading", f"Media files missing, re-embedding {dataset_name}...", 0, 0)
                pkl_file.unlink()
                pkl_file.with_suffix(".embedder").unlink(missing_ok=True)
                pkl_file.with_suffix(".clipper").unlink(missing_ok=True)
            else:
                # Stamp demo origin on cached medias so that cross-dataset
                # resolution always has the dataset name in the origin params.
                # Old pickles (created before origin stamping) may have empty
                # params — this ensures they are corrected on load.
                _stamp_demo_origin(medias, dataset_name, converter_name)
                on_progress("idle", f"Loaded {dataset_name} dataset", 0, 0)
                return

    # Resolve the embedder
    from vtsearch.media import embedders_for_type, get as media_get, get_embedder

    embedder = None
    if embedder_name:
        try:
            embedder = get_embedder(embedder_name)
        except KeyError:
            raise ValueError(f"Unknown embedder: {embedder_name}")
    else:
        avail = embedders_for_type(media_type_id)
        if avail:
            embedder = avail[0]

    mt = media_get(media_type_id)

    source = dataset_info.get("source", "")
    categories = dataset_info["categories"]
    slice_start = dataset_info.get("slice_start", 0)
    slice_end = dataset_info.get("slice_end")
    slice_frac_start = dataset_info.get("slice_frac_start")
    slice_frac_end = dataset_info.get("slice_frac_end")

    medias.clear()
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
    )

    # Stamp the demo origin on all medias
    _stamp_demo_origin(medias, dataset_name, converter_name)

    # --- Apply converter if requested ---
    if converter_name:
        from vtsearch.converters.runner import apply_converter_to_demo

        apply_converter_to_demo(
            converter_name=converter_name,
            dataset_name=dataset_name,
            medias=medias,
            embedder_name=embedder_name,
            on_progress=on_progress,
        )

    # Build the pickle cache payload
    # For types with external media dirs (audio, video), exclude media_bytes
    # from the pickle and store the dir path so reloading can find the files.
    # When a converter was applied, external_dir is no longer relevant (the
    # converted medias carry their own bytes/strings).
    if external_dir is not None and not converter_name:
        pkl_data: dict[str, Any] = {
            "name": dataset_name,
            "medias": {
                cid: {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in media.items()
                    if k not in ("media_bytes", "thumbnail_bytes")
                }
                for cid, media in medias.items()
            },
            mt.dir_key: external_dir,
        }
    else:
        pkl_data = {
            "name": dataset_name,
            "medias": {
                cid: {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in media.items()}
                for cid, media in medias.items()
            },
        }

    _loader.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(pkl_file, "wb") as f:
        pickle.dump(pkl_data, f)

    # Write a lightweight sidecar that records which embedder produced this pkl.
    resolved_name = getattr(embedder, "name", "") if embedder is not None else ""
    _loader._write_embedder_sidecar(pkl_file, resolved_name)

    # Write a clipper sidecar so the demo list can check readiness.
    _loader._write_clipper_sidecar(pkl_file, clipper_name)

    on_progress("idle", f"Loaded {dataset_name} dataset", 0, 0)
