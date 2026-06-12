"""Build an ephemeral browse context over a detector's positive labels.

VTSBrowse is a *dataset* browser: it projects a :class:`DatasetContext`'s
embedding matrix to 2-D and previews items from that context's ``medias``.
A trained detector, however, is a labelset whose positives can come from
**mixed sources** that aren't all present in any one loaded dataset — and
whose embedder is the detector's own, not whatever dataset happens to be
selected on the dashboard.

This module bridges the two by materialising the detector's positives as an
**in-memory, throwaway** :class:`DatasetContext`:

* each positive is origin-resolved to its file (read once for preview bytes),
* embedded with the *detector's* embedder (or reused from the loaded
  detector's cache when it's already in that space),
* and the resulting ``(embedding, media_bytes)`` media dicts are projected
  with the normal UMAP + hex-pyramid pipeline.

The whole context — vectors and bytes — lives only in process memory and is
never persisted (no ``pkl_path``), satisfying the "No Persisted Vectors"
rule. The browse stack then works unchanged: the canvas reads tiles and
previews via the standard ``/api/medias/<id>/...`` endpoints, resolving them
from this context's ``media_bytes``.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from vtscore.datasets.labelset import LabelSet
from vtscore.detectors.labelset_elements import stable_element_id
from vtscore.state.core import DatasetContext

#: ``dataset_id`` prefix for ephemeral detector-positives browse contexts.
#: The frontend keys off this to skip the dataset-registry check in the
#: browse route guard and to release the context when leaving the view.
DETPOS_PREFIX = "__detpos__"

#: Progress callback: ``(current, total, message)``.
ProgressCb = Callable[[int, int, str], None]


def detpos_dataset_id(detector_id: str) -> str:
    """The synthetic browse ``dataset_id`` for *detector_id*'s positives."""
    return f"{DETPOS_PREFIX}{detector_id}"


def build_positives_browse_context(
    detector_data: dict[str, Any],
    dataset_id: str,
    *,
    embedder_name: str,
    cached_embeddings: dict[str, np.ndarray] | None = None,
    display_name: str = "",
    bin_shape: str = "hex",
    on_progress: ProgressCb | None = None,
) -> DatasetContext:
    """Build a ready-to-browse ephemeral context over the positives.

    *embedder_name* is the **detector's** embedder; every uncached positive is
    embedded with it, regardless of any loaded dataset. *cached_embeddings*
    (typically the loaded detector's ``label_embeddings``) is reused when it
    already holds a vector for an element built in that same space, so a
    detector loaded against its own dataset isn't re-embedded.

    Builds the UMAP projection + *bin_shape* pyramid so the browse view finds
    a ready projection on arrival. Raises :class:`ValueError` when no positive
    could be resolved + embedded (nothing to browse).
    """
    from vtscore.detectors.resolver import embed_file, resolve_file_context

    media_type = detector_data.get("media_type", "") or ""
    labelset = LabelSet.from_dict(detector_data.get("labelset") or {})
    positives = [el for el in labelset.elements if el.label == "good"]

    medias: dict[int, dict[str, Any]] = {}
    total = len(positives)
    next_cid = 0
    for idx, elem in enumerate(positives):
        if on_progress is not None:
            on_progress(idx, total, "Resolving positives…")

        # Resolve the backing file once: bytes power the preview, and (when we
        # have no cached vector) the same path is embedded with the detector's
        # embedder. The ``with`` block keeps temp-materialised sources alive.
        emb: np.ndarray | None = None
        if cached_embeddings is not None:
            cached = cached_embeddings.get(stable_element_id(elem))
            if cached is not None:
                emb = np.asarray(cached, dtype=np.float32)

        media_bytes: bytes | None = None
        with resolve_file_context(elem.origin, elem.origin_name, elem.filename) as file_path:
            if file_path is not None and file_path.is_file():
                try:
                    media_bytes = file_path.read_bytes()
                except OSError:
                    media_bytes = None
                if emb is None and media_bytes is not None:
                    emb = embed_file(file_path, media_type, embedder_name)

        if emb is None or media_bytes is None:
            # Origin couldn't be located or embedded → it simply won't appear
            # on the map. Skipping keeps a partially-resolvable detector usable.
            continue

        medias[next_cid] = {
            "id": next_cid,
            "media_type": media_type,
            "embedding": np.asarray(emb, dtype=np.float32),
            "media_bytes": media_bytes,
            "filename": elem.filename or elem.origin_name or "",
            "origin_name": elem.origin_name or "",
            "origin": elem.origin or {},
            "md5": elem.md5 or "",
            "embedder": embedder_name,
        }
        next_cid += 1

    if not medias:
        raise ValueError(
            "None of this detector's positive labels could be resolved and embedded — nothing to browse."
        )

    ctx = DatasetContext(dataset_id)
    ctx.medias = medias
    ctx.dataset_display_name = display_name or None

    # Project + tile up front so the browse view lands on a ready layout.
    from vtscore.embedding.matrix import get_embedding_matrix
    from vtscore.projection import build_pyramid, fit_projection

    if on_progress is not None:
        on_progress(total, total, "Building projection…")
    sorted_ids, matrix = get_embedding_matrix(ctx)
    proj = fit_projection(matrix, sorted_ids)
    pyr = build_pyramid(proj, bin_shape=bin_shape)
    ctx._projection = proj
    ctx._pyramids[bin_shape] = pyr
    return ctx
