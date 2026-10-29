"""Scoring a snapshot's medias with a trained detector head.

:func:`scoring_rows_for_snap` builds the ``(media, score row)`` matrix a
snapshot is scored over, :func:`score_rows_with_model` forwards it through a
head and max-pools it to one score per media, and
:func:`score_media_with_model` sorts and formats the result.  Their
composition, :func:`_score_all_media`, is the one definition of what a
detector scores a media at.  Part of the detector training pipeline
(:mod:`vtscore.detectors.training`); reads only
:mod:`vtscore.detectors.training_snap`.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from vtscore.detectors.training_snap import _score_embedder_for_snap, _scores_in_patch_space
from vtscore.utils.scores import sigmoid_to_finite_array

if TYPE_CHECKING:
    import torch.nn as nn


log = logging.getLogger(__name__)


def _forward_sigmoid_chunked(
    model: nn.Sequential, matrix: np.ndarray, on_rows: Callable[[int], None] | None = None
) -> np.ndarray:
    """``sigmoid(model(matrix))`` as float64, upcasting a float16 matrix chunk-wise.

    The flattened patch matrix is stored float16 (see
    :func:`vtscore.embedding.matrix._build_region_arrays`); torch has no
    float16 CPU linear kernel and a whole-matrix upcast would allocate a
    float32 copy twice the size of the matrix - gigabytes on a large patch
    dataset, where MaxPatch already stacks ~197 rows per image.  Chunking
    bounds that copy at ``ROW_CHUNK`` rows.

    :func:`~vtscore.utils.scores.sigmoid_to_finite_array` replaces NaN/±Inf
    with the ``NON_FINITE_SCORE_SENTINEL`` (-1.0) so a destabilised head cannot
    leak non-finite floats into the JSON response.  The downstream segmented
    max-pool then incidentally drops sentinels in favour of any real score (in
    ``[0, 1]``) for the same media.

    *on_rows*, when given, is called with the number of rows scored so far
    after each chunk.
    """
    import torch  # noqa: PLC0415

    from vtscore.embedding.matrix import ROW_CHUNK  # noqa: PLC0415

    device = next(model.parameters()).device
    out = np.empty(matrix.shape[0], dtype=np.float64)
    with torch.no_grad():
        for start in range(0, matrix.shape[0], ROW_CHUNK):
            block = np.ascontiguousarray(matrix[start : start + ROW_CHUNK])
            chunk = torch.from_numpy(block).to(device=device, dtype=torch.float32)
            out[start : start + chunk.shape[0]] = sigmoid_to_finite_array(model(chunk))
            if on_rows is not None:
                on_rows(start + chunk.shape[0])
    return out


class ScoringRows(NamedTuple):
    """The ``(media, row)`` matrix one snapshot is scored over.

    ``ids`` are the media ids in sorted order; ``matrix`` holds one row per
    ``(media, score row)`` pair; ``media_index`` maps each row to its media's
    index in ``ids`` and ``region_index`` to its position within that media's
    :func:`vtscore.embedding.matrix.media_score_rows` stack.  Built once by
    :func:`scoring_rows_for_snap` and reusable across every head that scores
    the same snapshot in the same space (:func:`score_rows_with_model`).
    """

    ids: list[int]
    matrix: np.ndarray
    media_index: np.ndarray
    region_index: np.ndarray


def scoring_rows_for_snap(
    clips_dict: dict[int, dict[str, Any]],
    embedder_name: str | None = None,
    *,
    region_pooling: bool | None = None,
) -> ScoringRows:
    """Build the rows every media in *clips_dict* is scored over.

    Patch datasets (those whose media expose a ``patch_grid``) contribute every
    media's :func:`vtscore.embedding.matrix.media_score_rows` stack - image-level
    vector + all ``H*W`` raw patches - so the head's score for a media is the max
    over its rows.  Plain datasets contribute one row per media, from the cached
    embedding matrix.

    *embedder_name* is the detector's primary embedder (the space the head was
    trained in).  When it is given, the patch rows are used **only** if that
    primary is the dataset's patch-slot embedder - a detector scoring in the
    text or structural space of a multi-embedder dataset must score against
    that space's full-image vectors, not the patch grid.  When ``None`` (the
    pre-per-detector behaviour) any media carrying a ``patch_grid`` takes the
    patch path, matching the dataset-level score precedence.

    Split out from :func:`_score_all_media` so a caller with several heads over
    one snapshot - the CLI's per-group detector scoring - builds the rows once,
    and so no caller has to re-derive the patch gate (or the skip policy below)
    for itself.

    *region_pooling* overrides that gate for a caller whose **head** is not a
    MaxPatch head.  ``None`` (every ordinary caller) resolves it as described
    above.  ``False`` forces one image-level row per media even on a patch
    dataset - the geometry a whole-image head has to be scored at, because it
    was never shown a patch as a negative and would fire on distractors the
    max-pool then promotes to the media's score (see ``docs/ML.md``, region
    flooding).  ``True`` is accepted for symmetry and still yields one row per
    media on a grid-less dataset, which has no patch rows to pool.

    **No shipped caller passes ``False`` today.**  Cross-dataset Find's cold
    path used to, and was the case this override was written for; #3525 replaced
    the whole-image head it needed the override for with the app's own
    region-flooded labelset training, so the pin went with the head.  The knob
    stays because it is the correct answer for any caller that does build a
    whole-image head, and because it is the only thing separating "this head
    is not MaxPatch" from the dataset-level patch gate.

    Media that carry no usable vector in that space are **skipped**, not fatal:
    the builders raise on one (no vector, or a row of the wrong width), and this
    catches that, drops the offending media via
    :func:`~vtscore.embedding.matrix.scoreable_snapshot`, and rebuilds from what
    is left.  A snapshot is not the dataset's own medias dict - the CLI scores
    importer output that never went through the load pipeline's drop-none stage,
    and one bound embedder of a multi-embedder dataset can have failed on media
    another succeeded on - so a single unembeddable image must cost one skipped
    item, not the whole run (issue #3179).  Every caller reads the media list
    back off :attr:`ScoringRows.ids`, so a shorter list flows through unchanged
    - which is also how a caller names what was skipped, without this having to
    report it.

    The filter runs **on the failure path only**, not as a pre-pass: this is the
    per-vote retrain path over the whole dataset (and ``_fused_threshold`` calls
    it once more per calibration fold), where an unconditional O(N) scan would
    tax every vote on a 300k-media dataset to catch a case that the load
    pipeline has already made impossible there.  A clean snapshot pays nothing
    and keeps its cached-matrix fast path.
    """
    from vtscore.embedding.matrix import (  # noqa: PLC0415
        get_embedding_matrix_for_snap,
        get_region_matrix_for_snap,
        scoreable_snapshot,
    )

    resolved = embedder_name if embedder_name is not None else _score_embedder_for_snap(clips_dict)
    # Explicit per-detector primary: patch-pool only when scoring in the patch
    # space (the grid lives in the patch embedder's vectors).  Same gate
    # ``_build_vote_xy`` trains under - see :func:`_scores_in_patch_space`.
    has_regions = any(clips_dict[cid].get("patch_grid") is not None for cid in clips_dict) and _scores_in_patch_space(
        clips_dict, embedder_name
    )
    if region_pooling is not None:
        has_regions = has_regions and region_pooling

    def _build(snapshot: dict[int, dict[str, Any]]) -> ScoringRows:
        if has_regions:
            # One row per (media, score row) pair, built once and cached on the
            # dataset context (the patch vectors never change between votes -
            # only the head's weights do), so online retraining no longer rebuilds
            # a multi-million-row matrix on every vote.
            return ScoringRows(*get_region_matrix_for_snap(snapshot))
        ids, matrix = get_embedding_matrix_for_snap(snapshot, resolved)
        n = len(ids)
        return ScoringRows(ids, matrix, np.arange(n, dtype=np.int64), np.zeros(n, dtype=np.int64))

    try:
        rows = _build(clips_dict)
    except ValueError:
        # Both refusals land here: the missing-vector ``ValueError`` and the
        # wrong-width ``MismatchedVectorError`` that subclasses it.  Filter
        # against the same key the chosen builder reads - the patch-slot
        # embedder for the region path (``_build_region_arrays`` requires an
        # image-level row per media in that space), the resolved score embedder
        # otherwise - then rebuild.  A rebuild that still refuses is a different
        # problem and propagates.
        survivors, skipped = scoreable_snapshot(clips_dict, resolved, region_rows=has_regions)
        log.warning(
            "Scoring skipped %d media with no usable vector under %r: %s%s",
            len(skipped),
            resolved or "(primary)",
            skipped[:10],
            "…" if len(skipped) > 10 else "",
        )
        if not survivors:
            empty_rows = np.empty((0,), dtype=np.int64)
            return ScoringRows([], np.empty((0, 0), dtype=np.float32), empty_rows, empty_rows)
        rows = _build(survivors)

    if int(rows.matrix.shape[0]) != int(rows.media_index.size):
        # Deliberately outside the retry above: a builder that hands back an id
        # list and a matrix of different lengths is not the "this media has no
        # vector" case that filtering fixes, and rebuilding would only hide it
        # for one round. Left alone it surfaces as an out-of-bounds `reduceat`
        # deep in the max-pool, or - worse, under a plain zip - as silently
        # truncated scores (audit M11). Fail here, naming both counts.
        raise ValueError(
            f"scoring rows for {len(rows.ids)} media claim {rows.media_index.size} rows "
            f"but the matrix holds {rows.matrix.shape[0]}; the embedding matrix and its id list disagree."
        )
    return rows


_scoring_progress = threading.local()


@contextmanager
def scoring_progress(callback: Callable[[int, int], None] | None) -> Iterator[None]:
    """Report every scoring pass this thread makes to *callback* ``(media_done, media_total)``.

    Each :func:`score_rows_with_model` call inside the block is one pass over
    its rows: *callback* fires with ``(0, n)`` as the pass starts and again
    after each ``ROW_CHUNK`` block, ``media_done`` counting the media whose
    rows have all been scored, ending at ``(n, n)``.  A detector's train scores
    its whole haystack several times over (the final head, then each
    calibration fold model) before its own scoring pass, so this is how a
    caller can count those passes in media rather than only see them as time
    gone by (issue #5101).

    The hook is per thread, so detectors scored in parallel each report to
    their own callback; ``None`` reports nothing, and the previous callback is
    restored when the block exits.
    """
    previous = getattr(_scoring_progress, "callback", None)
    _scoring_progress.callback = callback
    try:
        yield
    finally:
        _scoring_progress.callback = previous


def score_rows_with_model(model: nn.Sequential, rows: ScoringRows) -> tuple[list[float], list[int]]:
    """Forward *rows* through *model* and max-pool them down to one score per media.

    Returns ``(scores_per_media, best_row_index_per_media)``; the winning row's
    index is what surfaces the best-match overlay in the UI.  The pass reports
    to the thread's :func:`scoring_progress` callback, if one is set.
    """
    from vtscore.embedding.matrix import segmented_max_pool  # noqa: PLC0415

    if not rows.ids:
        return [], []
    report = getattr(_scoring_progress, "callback", None)
    on_rows: Callable[[int], None] | None = None
    if report is not None:
        n_media = len(rows.ids)
        n_rows = int(rows.matrix.shape[0])

        def _on_rows(done: int) -> None:
            # A media's rows are contiguous, so the media of the next unscored
            # row is the count of media finished before it.
            report(n_media if done >= n_rows else int(rows.media_index[done]), n_media)

        on_rows = _on_rows
        report(0, n_media)
    flat_scores = _forward_sigmoid_chunked(model, rows.matrix, on_rows)
    return segmented_max_pool(flat_scores, rows.media_index, rows.region_index, len(rows.ids))


def _score_all_media(
    model: nn.Sequential,
    clips_dict: dict[int, dict[str, Any]],
    embedder_name: str | None = None,
) -> tuple[list[int], list[float], list[int]]:
    """Score every media in *clips_dict* with the trained detector head.

    The composition of :func:`scoring_rows_for_snap` and
    :func:`score_rows_with_model`, and **the** definition of what a detector
    scores a media at: every path that turns a head into per-media scores -
    online voting, Find, the population estimator behind the threshold, and CLI
    autodetect - goes through it, so none of them can score a media at a
    different geometry than the threshold was cut on.

    Returns ``(all_ids, scores_per_media, best_row_index_per_media)``.
    """
    rows = scoring_rows_for_snap(clips_dict, embedder_name)
    scores, best_region = score_rows_with_model(model, rows)
    return rows.ids, scores, best_region


def _format_results(
    all_ids: list[int],
    scores: list[float],
    best_region: list[int],
    clips_dict: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Sort by score (descending) and produce JSON-serialisable result dicts.

    Raw float scores are used for sorting so tiny differences still
    affect ordering; only the response ``score`` field is rounded.
    Patch media gain a ``best_region`` key holding the winning row's box - the
    whole image for row 0, a single grid cell otherwise
    (:func:`vtscore.embedding.matrix.media_row_box`).
    """
    from vtscore.embedding.matrix import media_row_box  # noqa: PLC0415

    paired = sorted(
        zip(all_ids, scores, best_region, strict=True),
        key=lambda t: t[1],
        reverse=True,
    )
    results: list[dict[str, Any]] = []
    for cid, s, bri in paired:
        entry: dict[str, Any] = {"id": cid, "score": round(s, 4)}
        box = media_row_box(clips_dict[cid], bri)
        if box is not None:
            entry["best_region"] = box
        results.append(entry)
    return results


def score_media_with_model(
    model: nn.Sequential,
    clips_dict: dict[int, dict[str, Any]],
    embedder_name: str | None = None,
) -> list[dict[str, Any]]:
    """Score every media in *clips_dict* with an already-trained *model*.

    Returns sorted (descending by score) result dicts of the same shape the
    vote-driven training path produces: ``{"id", "score"}`` plus a
    ``best_region`` box for patch-region-aware media (the argmax region that
    drove the media's score).  Use this from any route that scores with a
    pre-trained detector - e.g. the Find / detector-scoring path - so the
    best-match highlight is populated regardless of which entry point ran the
    scoring.  Plain single-vector datasets are scored via the cached embedding
    matrix and gain no ``best_region`` field, exactly as before.

    *embedder_name* is the detector's primary embedder, so the scoring space
    matches the one the *model* was trained in (the per-detector primary).
    """
    all_ids, scores, best_region = _score_all_media(model, clips_dict, embedder_name)
    return _format_results(all_ids, scores, best_region, clips_dict)
