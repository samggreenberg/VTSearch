"""Shared bulk helpers for image embedders.

Every image embedder shares the same input pipeline - PIL-load each
file, convert to RGB, pass a list of PIL images through the model's
processor, run the forward in chunks, then split the resulting tensor
back per-item.  This module factors that out so each concrete embedder
only supplies its model-specific forward callable.

The decode is **threaded and one batch ahead** of the forward
(:func:`_iter_decoded_batches`).  Decoding a batch inline, running the
forward, then decoding the next left the GPU idle for the whole decode:
measured over 384 real Visual Genome images on a V100, base SigLIP spent
54% of its wall clock in PIL and 28% in the processor, leaving the GPU
idle 82% of the time.  Pillow's decode releases the GIL in its C
extension, so a plain :class:`~concurrent.futures.ThreadPoolExecutor` both
parallelises a batch's decode and overlaps it with the previous batch's
forward, with no process-level machinery to pay for.

Nothing about the result changes: the same decoder sees the same bytes and
each image lands in the same slot, so this is bit-identical to decoding
serially.  The cost is memory - at most two batches of decoded images are
live at once instead of one, which is why the lookahead is exactly one
batch and the pool is capped (see
:func:`~vtscore.config.resolve_decode_workers`).
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, TYPE_CHECKING

import numpy as np

from vtscore.config import resolve_decode_workers
from vtscore.media.base import ProgressCallback

if TYPE_CHECKING:
    from PIL import Image

_log = logging.getLogger(__name__)


def _load_pil(source: Path | bytes) -> Optional["Image.Image"]:
    """Open *source* and return an RGB PIL Image, or ``None`` on failure.

    *source* may be a filesystem path or an in-memory ``bytes`` blob (used
    by the clip re-embed path so it can hand the bulk surface the
    clipped image bytes directly without a tempfile detour).

    The decode is bounded (see :mod:`vtscore.media.image.decode`): an
    enormous source is downsampled rather than refused or materialised at
    full resolution.  Nothing downstream notices — every image model's
    processor resizes to its own fixed input size, and patch grids /
    region boxes are expressed in normalised coordinates.
    """
    try:
        from vtscore.media.image.decode import decode_bounded_rgb  # noqa: PLC0415

        # Bounded: every image model's processor resizes to a few hundred pixels
        # anyway, so decoding a gigapixel source at full resolution would only
        # buy a huge transient bitmap — and a whole batch of them at once.
        img, _scale = decode_bounded_rgb(source)
        return img
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


def _batch_bounds(total: int, batch_size: int) -> list[tuple[int, int]]:
    """``[(start, end), ...]`` half-open slices of *total* items."""
    return [(s, min(s + batch_size, total)) for s in range(0, total, batch_size)]


def _decode_range(medias: list[dict], start: int, end: int) -> tuple[list[int], list["Image.Image"]]:
    """Decode ``medias[start:end]`` on this thread, dropping failures."""
    indices: list[int] = []
    images: list[Image.Image] = []
    for idx in range(start, end):
        source = _pil_source_for(medias[idx])
        if source is None:
            continue
        img = _load_pil(source)
        if img is None:
            continue
        indices.append(idx)
        images.append(img)
    return indices, images


def _submit_range(
    pool: ThreadPoolExecutor,
    medias: list[dict],
    start: int,
    end: int,
) -> list[tuple[int, "Future[Optional[Image.Image]]"]]:
    """Queue ``medias[start:end]`` for decode, returning ``(slot, future)`` pairs.

    Media with no decodable source are dropped here rather than becoming a
    future that resolves to ``None``, so the pool only ever holds real work.
    """
    jobs: list[tuple[int, Future[Optional[Image.Image]]]] = []
    for idx in range(start, end):
        source = _pil_source_for(medias[idx])
        if source is None:
            continue
        jobs.append((idx, pool.submit(_load_pil, source)))
    return jobs


def _collect(
    jobs: list[tuple[int, "Future[Optional[Image.Image]]"]],
) -> tuple[list[int], list["Image.Image"]]:
    """Wait on a batch's decodes, keeping submission order and dropping failures.

    Order comes from *jobs*, not from completion, so a slow decode cannot
    shuffle images against their slots.
    """
    indices: list[int] = []
    images: list[Image.Image] = []
    for idx, fut in jobs:
        img = fut.result()  # _load_pil swallows its own errors and returns None
        if img is None:
            continue
        indices.append(idx)
        images.append(img)
    return indices, images


def _iter_decoded_batches(
    medias: list[dict],
    batch_size: int,
    workers: Optional[int] = None,
) -> Iterator[tuple[int, list[int], list["Image.Image"]]]:
    """Yield ``(end, indices, images)`` per batch, decoded one batch ahead.

    *end* is the exclusive index of the batch's last media (what the progress
    callback reports), *indices* the slots that decoded successfully, and
    *images* their decoded RGB images, positionally aligned with *indices*.
    Media that carry no source, and those whose decode fails, are simply
    absent - exactly as when the decode ran inline.

    Each batch's decodes are submitted to a pool *before* the previous batch is
    yielded, so the pool works through the next batch while the caller runs the
    current one's forward.  *workers* defaults to
    :func:`~vtscore.config.resolve_decode_workers`; ``0`` decodes inline on
    this thread, which is the pre-threading behaviour.

    A caller that abandons this generator - the progress callback raising
    ``CancelledError`` is the usual reason - drops the prefetched batch rather
    than waiting it out, so a cancel is not held up by a decode nobody wants.
    """
    bounds = _batch_bounds(len(medias), batch_size)
    if not bounds:
        return

    if workers is None:
        workers = resolve_decode_workers()
    if workers < 1:
        for start, end in bounds:
            indices, images = _decode_range(medias, start, end)
            yield end, indices, images
        return

    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vts-decode")
    try:
        jobs = _submit_range(pool, medias, *bounds[0])
        for i, (_, end) in enumerate(bounds):
            # Queue the next batch *before* yielding this one, so the pool works
            # through it while the caller runs this batch's forward.
            ahead = _submit_range(pool, medias, *bounds[i + 1]) if i + 1 < len(bounds) else []
            indices, images = _collect(jobs)
            jobs = ahead
            yield end, indices, images
    finally:
        # ``cancel_futures`` drops the queued prefetch on an early exit; the
        # few decodes already running are left to finish on their own threads
        # (their results are simply discarded) rather than blocking the caller.
        pool.shutdown(wait=False, cancel_futures=True)


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
    overall call - subsequent batches still run.

    Decoding is threaded and runs one batch ahead of *forward_pil_batch*;
    see the module docstring for why and what it costs.
    """
    total = len(medias)
    results: list[Optional[np.ndarray]] = [None] * total
    if total == 0:
        return results

    batch_size = max(1, int(batch_size))
    for end, chunk_indices, chunk_images in _iter_decoded_batches(medias, batch_size):
        on_progress("embedding", f"Embedding {label}...", end, total)

        if not chunk_images:
            continue

        try:
            vectors = forward_pil_batch(chunk_images)
        except Exception:
            _log.exception("Bulk %s forward failed for indices %s", label, chunk_indices)
            continue

        if vectors is None or len(vectors) != len(chunk_indices):
            _log.warning(
                "Bulk %s forward returned %d vectors for %d inputs - skipping batch",
                label,
                0 if vectors is None else len(vectors),
                len(chunk_indices),
            )
            continue

        for slot, vec in zip(chunk_indices, vectors):
            results[slot] = np.asarray(vec)

    return results


def bulk_patch_forward_image_files(
    medias: list[dict],
    forward_pil_batch: Callable[[list["Image.Image"]], list[Any]],
    batch_size: int,
    on_progress: ProgressCallback,
    label: str,
) -> list[Optional[Any]]:
    """Run a batched patch-forward over PIL-decoded images.

    Same threaded decode + per-batch GPU call shape as
    :func:`bulk_embed_image_files`, but *forward_pil_batch* returns a
    list of :class:`PatchEmbedOutput`-typed objects (one per input image)
    so callers can attach side-channel state per-image.
    """
    total = len(medias)
    results: list[Optional[Any]] = [None] * total
    if total == 0:
        return results

    batch_size = max(1, int(batch_size))
    for end, chunk_indices, chunk_images in _iter_decoded_batches(medias, batch_size):
        on_progress("embedding", f"Patch-embedding {label}...", end, total)

        if not chunk_images:
            continue

        try:
            outputs = forward_pil_batch(chunk_images)
        except Exception:
            _log.exception("Bulk %s patch-forward failed for indices %s", label, chunk_indices)
            continue

        if outputs is None or len(outputs) != len(chunk_indices):
            _log.warning(
                "Bulk %s patch-forward returned %d outputs for %d inputs - skipping batch",
                label,
                0 if outputs is None else len(outputs),
                len(chunk_indices),
            )
            continue

        for slot, out in zip(chunk_indices, outputs):
            results[slot] = out

    return results
