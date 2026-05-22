"""Shared bulk helpers for image embedders.

Every image embedder shares the same input pipeline — PIL-load each
file, convert to RGB, pass a list of PIL images through the model's
processor, run the forward in chunks, then split the resulting tensor
back per-item.  This module factors that out so each concrete embedder
only supplies its model-specific forward callable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING

import numpy as np

from vtscore.media.base import ProgressCallback

if TYPE_CHECKING:
    from PIL import Image

_log = logging.getLogger(__name__)


def _load_pil(source: Path | bytes) -> Optional["Image.Image"]:
    """Open *source* and return an RGB PIL Image, or ``None`` on failure.

    *source* may be a filesystem path or an in-memory ``bytes`` blob (used
    by the clip re-embed path so it can hand the bulk surface the
    clipped image bytes directly without a tempfile detour).
    """
    try:
        from PIL import Image  # noqa: PLC0415
        import io  # noqa: PLC0415

        if isinstance(source, (bytes, bytearray)):
            with Image.open(io.BytesIO(bytes(source))) as img:
                return img.convert("RGB")
        with Image.open(source) as img:
            return img.convert("RGB")
    except Exception:
        _log.exception("Error decoding image %r", source if isinstance(source, Path) else "<bytes>")
        return None


def _pil_source_for(media: dict) -> Path | bytes | None:
    """Pick the in-memory or on-disk source to decode for *media*.

    Prefers ``media_bytes`` (set by clip re-embed) over ``media_path`` so
    the bulk path never has to round-trip through a tempfile.
    """
    blob = media.get("media_bytes")
    if isinstance(blob, (bytes, bytearray)) and blob:
        return bytes(blob)
    path_str = media.get("media_path")
    if path_str:
        return Path(path_str)
    return None


def bulk_embed_image_files(
    medias: list[dict],
    forward_pil_batch: Callable[[list["Image.Image"]], np.ndarray],
    batch_size: int,
    on_progress: ProgressCallback,
    label: str,
) -> list[Optional[np.ndarray]]:
    """Embed images from media dicts in chunks of *batch_size*.

    *forward_pil_batch* is a callable that runs the model's preprocess +
    forward pass on a list of PIL images and returns an ``(N, D)``
    :class:`numpy.ndarray`.

    Files that fail to decode are reported as ``None`` in the output at
    the matching index.  A failing GPU forward fails the whole batch
    (those positions are filled with ``None``) but does not abort the
    overall call — subsequent batches still run.
    """
    total = len(medias)
    results: list[Optional[np.ndarray]] = [None] * total
    if total == 0:
        return results

    batch_size = max(1, int(batch_size))
    start = 0
    while start < total:
        end = min(start + batch_size, total)
        chunk_indices: list[int] = []
        chunk_images: list[Image.Image] = []
        for idx in range(start, end):
            media = medias[idx]
            source = _pil_source_for(media)
            if source is None:
                continue
            img = _load_pil(source)
            if img is None:
                continue
            chunk_indices.append(idx)
            chunk_images.append(img)

        on_progress("embedding", f"Embedding {label} {end}/{total}...", end, total)

        if not chunk_images:
            start = end
            continue

        try:
            vectors = forward_pil_batch(chunk_images)
        except Exception:
            _log.exception("Bulk %s forward failed for indices %s", label, chunk_indices)
            start = end
            continue

        if vectors is None or len(vectors) != len(chunk_indices):
            _log.warning(
                "Bulk %s forward returned %d vectors for %d inputs — skipping batch",
                label,
                0 if vectors is None else len(vectors),
                len(chunk_indices),
            )
            start = end
            continue

        for slot, vec in zip(chunk_indices, vectors):
            results[slot] = np.asarray(vec)

        start = end

    return results


def bulk_patch_forward_image_files(
    medias: list[dict],
    forward_pil_batch: Callable[[list["Image.Image"]], list[Any]],
    batch_size: int,
    on_progress: ProgressCallback,
    label: str,
) -> list[Optional[Any]]:
    """Run a batched patch-forward over PIL-decoded images.

    Same per-file decode + per-batch GPU call shape as
    :func:`bulk_embed_image_files`, but *forward_pil_batch* returns a
    list of :class:`PatchEmbedOutput`-typed objects (one per input image)
    so callers can attach side-channel state per-image.
    """
    total = len(medias)
    results: list[Optional[Any]] = [None] * total
    if total == 0:
        return results

    batch_size = max(1, int(batch_size))
    start = 0
    while start < total:
        end = min(start + batch_size, total)
        chunk_indices: list[int] = []
        chunk_images: list[Image.Image] = []
        for idx in range(start, end):
            media = medias[idx]
            source = _pil_source_for(media)
            if source is None:
                continue
            img = _load_pil(source)
            if img is None:
                continue
            chunk_indices.append(idx)
            chunk_images.append(img)

        on_progress("embedding", f"Patch-embedding {label} {end}/{total}...", end, total)

        if not chunk_images:
            start = end
            continue

        try:
            outputs = forward_pil_batch(chunk_images)
        except Exception:
            _log.exception("Bulk %s patch-forward failed for indices %s", label, chunk_indices)
            start = end
            continue

        if outputs is None or len(outputs) != len(chunk_indices):
            _log.warning(
                "Bulk %s patch-forward returned %d outputs for %d inputs — skipping batch",
                label,
                0 if outputs is None else len(outputs),
                len(chunk_indices),
            )
            start = end
            continue

        for slot, out in zip(chunk_indices, outputs):
            results[slot] = out

        start = end

    return results
