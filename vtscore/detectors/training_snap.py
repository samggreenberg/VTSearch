"""Which embedding space a detector trains and scores in, for a snapshot of medias.

The bottom layer of the detector training pipeline
(:mod:`vtscore.detectors.training`): resolving a snapshot's score and
patch-slot embedders, the one patch gate training and scoring share
(:func:`_scores_in_patch_space`), and the good/bad label check.  Every other
``training_*`` module reads this one; it reads none of them.
"""

from __future__ import annotations

from typing import Any


def _score_embedder_for_snap(snap: dict[int, dict[str, Any]] | None) -> str | None:
    """Resolve the score embedder (structural ▸ patch ▸ text) for the medias in *snap*.

    The detector head trains and scores against this embedder's vectors (the v3
    routing table).  Derived from the embedder names *those* medias carry
    rather than the active context, since :func:`_score_all_media` /
    :func:`_build_vote_xy` may be handed an arbitrary snapshot (a test fixture,
    a cross-dataset dict).  Returns ``None`` for a slot-less single-vector
    dataset (e.g. ``dinov2_single``) or medias whose embedder is unregistered,
    so the matrix layer falls back to each media's primary vector.  For a
    single-embedder dataset the resolved name equals the primary, which the
    matrix layer collapses to the cached primary path.

    This is deliberately **not** :func:`~vtscore.embedding.binding.score_marker_embedder_for_snap`:
    that one falls back to the media's primary *name* and returns ``""``, where
    the matrix layer here wants ``None`` to mean "no bound slot, read each
    media's primary vector".
    """
    from vtscore.embedding.binding import slot_embedders_for_snap  # noqa: PLC0415

    text, patch, structural = slot_embedders_for_snap(snap)
    return structural or patch or text


def detector_score_embedder(det_ctx: Any, snap: dict[int, dict[str, Any]] | None) -> str | None:
    """Embedder a *detector* trains and scores in.

    The concrete embedder of the detector's locked type
    (``det_ctx.embedder_type``) that the snap supplies wins; otherwise the
    dataset score precedence - the legacy-migration default and the
    cross-dataset portability fallback (a detector pointed at a dataset that
    lacks its type re-embeds against that dataset's space).

    This is :func:`~vtscore.embedding.binding.keying_embedder_for_snap` with the
    empty marker normalised to ``None``, which is the *only* reason the name
    exists: the training and scoring paths read ``None`` as "no explicit
    primary, fall back to each media's own vector", so the resolver's ``""``
    for an empty snap has to become ``None`` before it reaches them.  Kept as a
    public name because out-of-tree code may import it.  See
    ``docs/ARCHITECTURE.md`` → "Per-detector embedder type".
    """
    from vtscore.embedding.binding import keying_embedder_for_snap  # noqa: PLC0415

    return keying_embedder_for_snap(det_ctx, snap) or None


def _patch_embedder_for_snap(snap: dict[int, dict[str, Any]] | None) -> str | None:
    """Resolve the patch-slot embedder name for *snap*'s medias, or ``None``."""
    from vtscore.embedding.binding import slot_embedders_for_snap  # noqa: PLC0415

    return slot_embedders_for_snap(snap)[1]


def _scores_in_patch_space(snap: dict[int, dict[str, Any]] | None, embedder_name: str | None) -> bool:
    """Whether a detector whose primary is *embedder_name* lives in *snap*'s patch space.

    **The single definition of the patch gate**, so training and scoring cannot
    disagree about which geometry a detector is in:

    * An explicit primary (the per-detector embedder) is in the patch space
      **only** when it *is* the snap's patch-slot embedder.  A detector locked to
      the text or structural space of a multi-embedder dataset scores against
      that space's full-image vectors (:func:`_score_all_media`), so it must
      train on those vectors too - flooding it with patch rows would mix two
      unrelated embedding spaces into one model (or, when their dimensions
      differ, fail outright with a
      :class:`~vtscore.embedding.precomputed.MismatchedVectorError`).
    * ``None`` (no per-detector primary) keeps the dataset-level score
      precedence, where any media carrying a ``patch_grid`` takes the patch
      path - the pre-per-detector behaviour.

    Callers pair this with "does any media actually carry a grid?"; on a
    grid-less dataset both sides collapse to the single image-level vector
    regardless.
    """
    if embedder_name is None:
        return True
    patch = _patch_embedder_for_snap(snap)
    return patch is not None and embedder_name == patch


def validate_good_bad_split(y_list: list[float]) -> tuple[int, int]:
    """Check that *y_list* contains at least one good and one bad label.

    Returns ``(num_good, num_bad)``.
    Raises ``ValueError`` when either count is zero.
    """
    num_good = sum(1 for y in y_list if y == 1.0)
    num_bad = len(y_list) - num_good
    if num_good == 0 or num_bad == 0:
        raise ValueError("Need at least one good and one bad labeled example")
    return num_good, num_bad
