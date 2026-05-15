"""Detector training helpers: validate, train, threshold, serialise.

Consolidates the repeated train → calibrate → safe-threshold → serialise
pipeline used by detector route handlers and test helpers.

This module also holds the vote-aware detector training entry points —
:func:`train_and_score` (online, called every time the user toggles a
vote) and :func:`train_detector_from_origins` (load-time, called when
re-deriving an MLP from a saved labelset). Both build on the generic
:mod:`vtsearch.training` primitives but layer in the patch-region max-
pool and origin-based file resolution that are detector-specific.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch.nn as nn


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


def train_and_threshold(
    X_list: list,
    y_list: list[float],
    snap: dict | None = None,
) -> tuple[Any, float]:
    """Train an MLP and compute a calibrated threshold.

    This is the canonical training pipeline used by all detector routes:

    1. Cross-calibration threshold (respects ``calibrate_count`` /
       ``calibration_fraction`` settings).
    2. Full-data model training (respects ``inclusion`` setting).
    3. Optional safe-threshold blending when ``get_safe_thresholds()`` is
       enabled and *snap* is provided.

    Args:
        X_list: Embedding vectors (list of numpy arrays).
        y_list: Binary labels (1.0 = good, 0.0 = bad).
        snap: Optional media snapshot for safe-threshold scoring.

    Returns:
        ``(model, threshold)``
    """
    import torch

    from vtsearch.state import (
        get_calibrate_count,
        get_calibration_fraction,
        get_inclusion,
        get_safe_thresholds,
    )
    from vtsearch.training import (
        calculate_cross_calibration_threshold,
        calculate_safe_threshold,
        train_model,
    )

    X = torch.from_numpy(np.stack(X_list).astype(np.float32, copy=False))
    y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
    input_dim = X.shape[1]

    safe = bool(get_safe_thresholds() and snap)
    # Below the safe-threshold ramp floor the cross-cal output is blended
    # away (pure GMM), so don't pay for the fold trainings.
    if safe and len(X_list) < 6:
        threshold = float("inf")
    else:
        threshold = calculate_cross_calibration_threshold(
            X_list,
            y_list,
            input_dim,
            get_inclusion(),
            calibrate_count=get_calibrate_count(),
            calibration_fraction=get_calibration_fraction(),
        )

    model = train_model(X, y, input_dim, get_inclusion())

    if safe:
        from vtsearch.embedding.matrix import get_embedding_matrix_for_snap

        _all_ids, all_embs = get_embedding_matrix_for_snap(snap)
        X_all = torch.from_numpy(all_embs)
        with torch.no_grad():
            X_all = X_all.to(next(model.parameters()).device)
            all_scores = torch.sigmoid(model(X_all)).squeeze(1).cpu().tolist()
        threshold = calculate_safe_threshold(threshold, all_scores, len(y_list))

    return model, threshold


def serialize_weights(model) -> dict[str, list]:
    """Convert a PyTorch model's state dict to JSON-serialisable nested lists."""
    return {key: value.cpu().tolist() for key, value in model.state_dict().items()}


# ---------------------------------------------------------------------------
# Vote-aware detector training (online, called from sort/vote handlers)
# ---------------------------------------------------------------------------


def _training_vec_for_vote(
    media: dict[str, Any],
    region_box: tuple[float, float, float, float] | None,
) -> np.ndarray:
    """Return the training vector for one vote on *media*.

    When *region_box* is set **and** *media* has a stored ``patch_grid``,
    pool the box on-the-fly via
    :func:`vtsearch.media.patch_embed.box_to_vote_vector`.  Otherwise
    fall back to ``media["embedding"]`` — the v1/legacy image-level vector.
    Patch-embedder v2.
    """
    if region_box is not None:
        grid = media.get("patch_grid")
        if grid is not None:
            from vtsearch.media.patch_embed import box_to_vote_vector  # noqa: PLC0415

            return box_to_vote_vector(np.asarray(grid), region_box)
    return media["embedding"]


def train_and_score(
    clips_dict: dict[int, dict[str, Any]],
    good_votes: dict[int, None],
    bad_votes: dict[int, None],
    inclusion_value: int = 0,
    safe_thresholds: bool = False,
    calibrate_count: int = 2,
    calibration_fraction: float = 0.5,
    vote_region_boxes: dict[int, tuple[float, float, float, float]] | None = None,
    det_ctx: Any = None,
) -> tuple[list[dict[str, Any]], float, nn.Sequential | None]:
    """Train a small MLP on voted media embeddings and score every media.

    Uses k-fold calibration to determine an appropriate decision threshold,
    then trains a final model on all labelled data and scores every media in
    ``clips_dict``.

    Args:
        clips_dict: Mapping of media ID to media data dict. Each value must contain
            an ``"embedding"`` key with a ``numpy.ndarray`` embedding vector.
        good_votes: Dict whose keys are media IDs labelled as good (values are ``None``).
        bad_votes: Dict whose keys are media IDs labelled as bad (values are ``None``).
        inclusion_value: Integer in ``[-10, 10]`` passed to the training and
            threshold-finding functions to control the inclusion/exclusion bias.
        safe_thresholds: When ``True``, blend the cross-calibration threshold with
            a GMM-based threshold for robustness when few labels are available.
        calibrate_count: Number of random Train/Calibrate splits for threshold
            calibration (default 2).
        calibration_fraction: Fraction of labelled data reserved for calibration
            in each split (default 0.5).  For example, 0.2 means 80% Train /
            20% Calibrate.
        vote_region_boxes: Optional ``media_id -> (x0, y0, x1, y1)`` map from
            yes-votes that designated a region.  When set and the source
            media has a stored ``patch_grid``, the training vector for that
            vote is pooled on-the-fly via
            :func:`vtsearch.media.patch_embed.box_to_vote_vector` instead
            of using ``media["embedding"]``.  Falls back to the full-image
            vector when the media lacks a patch grid (legacy datasets,
            single-vector embedders) or when the box is missing.  Patch-
            embedder v2.

    Returns:
        A tuple ``(results, threshold, model)`` where:

        - ``results`` is a list of ``{"id": int, "score": float}`` dicts, sorted
          by score in descending order (highest confidence first).
        - ``threshold`` is the decision boundary as a float (cross-calibrated,
          or blended with GMM when ``safe_thresholds`` is ``True``).
        - ``model`` is the trained ``nn.Sequential`` model (``None`` when
          training was not possible).
    """
    import torch  # noqa: PLC0415

    from vtsearch.training.mlp import _auto_hidden_dim, train_model
    from vtsearch.training.thresholds import (
        calculate_safe_threshold,
        cross_calibration_threshold_cached,
    )

    region_boxes = vote_region_boxes or {}

    X_list = []
    y_list = []
    for cid in good_votes:
        if cid in clips_dict:
            X_list.append(_training_vec_for_vote(clips_dict[cid], region_boxes.get(cid)))
            y_list.append(1.0)
    for cid in bad_votes:
        if cid in clips_dict:
            X_list.append(clips_dict[cid]["embedding"])
            y_list.append(0.0)

    # Guard against empty or single-class training data after filtering
    num_good = sum(1 for v in y_list if v == 1.0)
    num_bad = len(y_list) - num_good
    if len(X_list) < 2 or num_good == 0 or num_bad == 0:
        return [], 0.5, None

    X = torch.from_numpy(np.stack(X_list).astype(np.float32, copy=False))
    y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)

    input_dim = X.shape[1]

    # Compute hidden_dim once from the *full* label count so that fold
    # models use the same architecture as the final model.  This makes
    # cross-calibration thresholds directly comparable to final-model
    # scores (same capacity, same score distribution shape).
    hidden_dim = _auto_hidden_dim(len(X_list))

    # Calculate threshold using k-fold calibration.  Skip when the label
    # count is below the ``calculate_safe_threshold`` ramp floor: the
    # calibration trainings would be expensive (two 200-epoch fits) and
    # the result is either discarded (safe_thresholds=True blends with
    # label_weight=0 → pure GMM) or unreliable (safe_thresholds=False with
    # so few labels).  Use 0.5 as the neutral default; it blends to pure
    # GMM under safe_thresholds and is a sensible mid-point otherwise.
    if len(X_list) < 6:
        threshold = 0.5
    else:
        threshold = cross_calibration_threshold_cached(
            X_list,
            y_list,
            input_dim,
            inclusion_value,
            calibrate_count=calibrate_count,
            calibration_fraction=calibration_fraction,
            hidden_dim=hidden_dim,
            det_ctx=det_ctx,
        )

    # Train final model on all data.  Training is image-level in v1 — the
    # MLP only ever sees one vector per voted media (``media["embedding"]``,
    # which equals the patch-region full-image vector for patch datasets),
    # mirroring the v1 vote rule of "vote on whole images".  Region-level
    # training examples are a phase-2 concern.
    model = train_model(X, y, input_dim, inclusion_value, hidden_dim=hidden_dim)

    # Score every media — region-aware max-pool over regions when the
    # dataset is patch-region-aware, plain single-vector scoring when not.
    # We build one flat tensor of (media, region) rows so the MLP runs
    # in a single forward pass; the per-media max is computed after.
    has_regions = any(clips_dict[cid].get("patch_regions") for cid in clips_dict)
    if has_regions:
        all_ids = sorted(clips_dict.keys())
        flat_vecs: list[np.ndarray] = []
        media_index_per_row: list[int] = []
        region_index_per_row: list[int] = []
        for mi, cid in enumerate(all_ids):
            media = clips_dict[cid]
            regions = media.get("patch_regions")
            if regions:
                for ri, r in enumerate(regions):
                    flat_vecs.append(np.asarray(r.vec, dtype=np.float32))
                    media_index_per_row.append(mi)
                    region_index_per_row.append(ri)
            else:
                flat_vecs.append(np.asarray(media["embedding"], dtype=np.float32))
                media_index_per_row.append(mi)
                region_index_per_row.append(0)
        X_all = torch.from_numpy(np.stack(flat_vecs).astype(np.float32, copy=False))
    else:
        from vtsearch.embedding.matrix import get_embedding_matrix_for_snap  # noqa: PLC0415

        all_ids, all_embs = get_embedding_matrix_for_snap(clips_dict)
        X_all = torch.from_numpy(all_embs)
        n = len(all_ids)
        media_index_per_row = list(range(n))
        region_index_per_row = [0] * n
    with torch.no_grad():
        X_all = X_all.to(next(model.parameters()).device)
        flat_scores = torch.sigmoid(model(X_all)).squeeze(1).cpu().numpy()

    # Max-pool per media; remember the winning region index so we can
    # surface ``best_region.box`` to the UI for patch-region media.
    scores: list[float] = [-1.0] * len(all_ids)
    best_region: list[int] = [0] * len(all_ids)
    for s, mi, ri in zip(flat_scores, media_index_per_row, region_index_per_row):
        if s > scores[mi]:
            scores[mi] = float(s)
            best_region[mi] = ri

    if safe_thresholds:
        n_labels = len(X_list)
        threshold = calculate_safe_threshold(threshold, scores, n_labels)

    # Sort by raw scores (full precision) so that tiny differences still
    # affect ordering.  Round only for the JSON response values.
    paired = sorted(
        zip(all_ids, scores, best_region),
        key=lambda t: t[1],
        reverse=True,
    )
    results: list[dict[str, Any]] = []
    for cid, s, bri in paired:
        entry: dict[str, Any] = {"id": cid, "score": round(s, 4)}
        media = clips_dict[cid]
        regions = media.get("patch_regions")
        if regions and 0 <= bri < len(regions):
            entry["best_region"] = list(regions[bri].box)
        results.append(entry)
    return results, threshold, model


# ---------------------------------------------------------------------------
# Origin-based helpers (for weight-free detector serialisation)
# ---------------------------------------------------------------------------


def collect_media_origins(
    media_ids: dict[int, None] | list[int],
    snap: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collect origin info for a set of media IDs from a medias snapshot.

    Each returned dict contains ``origin``, ``origin_name``, ``filename``,
    and ``md5`` — enough to re-resolve the original file later.

    Args:
        media_ids: Media IDs (keys of a votes dict, or a plain list).
        snap: Snapshot of all loaded medias (from :func:`snapshot_medias`).

    Returns:
        A list of origin dicts, one per matched media.
    """
    origins: list[dict[str, Any]] = []
    for cid in media_ids:
        if cid not in snap:
            continue
        media = snap[cid]
        origins.append(
            {
                "origin": media.get("origin"),
                "origin_name": media.get("origin_name", ""),
                "filename": media.get("filename", ""),
                "md5": media.get("md5", ""),
            }
        )
    return origins


def train_detector_from_origins(
    good_origins: list[dict[str, Any]],
    bad_origins: list[dict[str, Any]],
    inclusion: int,
    media_type: str,
    calibrate_count: int = 2,
    calibration_fraction: float = 0.5,
) -> tuple[dict[str, list] | None, float]:
    """Resolve origin entries to files, embed them, and train a detector MLP.

    This is the load-time counterpart of file-based detector export: given
    the origin lists that were saved to disk, it re-derives the MLP weights
    by resolving the original media files, embedding them, and training.

    Args:
        good_origins: Origin dicts for media labelled Good.
        bad_origins: Origin dicts for media labelled Bad.
        inclusion: The inclusion value to use for training.
        media_type: Media type string (e.g. ``"audio"``, ``"image"``).
        calibrate_count: Number of k-fold calibration splits.
        calibration_fraction: Fraction reserved for calibration.

    Returns:
        A ``(weights, threshold)`` tuple.  ``weights`` is ``None`` if
        resolution/embedding failed for too many entries (need at least
        one good and one bad).
    """
    import torch  # noqa: PLC0415

    from vtsearch.detectors.resolver import embed_file, resolve_file_context
    from vtsearch.training.mlp import train_model
    from vtsearch.training.thresholds import calculate_cross_calibration_threshold

    X_list: list = []
    y_list: list[float] = []

    for entry in good_origins:
        with resolve_file_context(
            entry.get("origin"),
            entry.get("origin_name", ""),
            entry.get("filename", ""),
        ) as file_path:
            if file_path is None:
                continue
            emb = embed_file(file_path, media_type)
        if emb is None:
            continue
        X_list.append(emb)
        y_list.append(1.0)

    for entry in bad_origins:
        with resolve_file_context(
            entry.get("origin"),
            entry.get("origin_name", ""),
            entry.get("filename", ""),
        ) as file_path:
            if file_path is None:
                continue
            emb = embed_file(file_path, media_type)
        if emb is None:
            continue
        X_list.append(emb)
        y_list.append(0.0)

    num_good = sum(1 for v in y_list if v == 1.0)
    num_bad = len(y_list) - num_good
    if len(X_list) < 2 or num_good == 0 or num_bad == 0:
        return None, 0.5

    X = torch.from_numpy(np.stack(X_list).astype(np.float32, copy=False))
    y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
    input_dim = X.shape[1]

    if len(X_list) < 6:
        threshold = 0.5
    else:
        threshold = calculate_cross_calibration_threshold(
            X_list,
            y_list,
            input_dim,
            inclusion,
            calibrate_count=calibrate_count,
            calibration_fraction=calibration_fraction,
        )
    model = train_model(X, y, input_dim, inclusion)

    state_dict = model.state_dict()
    weights = {k: v.cpu().tolist() for k, v in state_dict.items()}
    return weights, threshold
