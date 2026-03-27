"""Detector weight normalisation.

Detector JSON files may contain two weight sources:

- **Origin-based** (``good_origins`` / ``bad_origins``): re-derives weights
  by resolving the original media files, embedding, and training.
- **Pre-computed** (``weights`` / ``threshold``): serialised weight matrices
  used when the original media files are not available on disk.

The :func:`normalize_detector_weights` helper tries origin-based training
first, falling back to pre-computed weights.

Shared by ``detectors_crud.py``, ``cli.py``, and the
``server_detector_file`` importer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class NormalizedWeights:
    """Result of normalising detector weights from a JSON file."""

    weights: Any
    threshold: float
    good_origins: list | None = None
    bad_origins: list | None = None
    inclusion: int = 0
    origin_derived: bool = False
    """True when weights were successfully re-derived from origins."""


def normalize_detector_weights(
    detector_data: dict,
    *,
    media_type: str = "audio",
) -> NormalizedWeights:
    """Resolve detector weights from *detector_data*, trying origins first.

    1. If ``good_origins`` and ``bad_origins`` are present, attempt to
       re-derive weights via :func:`train_detector_from_origins`.
    2. Fall back to the ``weights`` key if origin resolution fails.
    3. Raise :class:`ValueError` if neither source provides weights.

    Args:
        detector_data: Parsed detector JSON dict.
        media_type: Media type for origin-based training (overridden by
            ``detector_data["media_type"]`` when present in the dict for
            callers that don't supply it separately).

    Returns:
        A :class:`NormalizedWeights` with the resolved weights and metadata.
    """
    good_origins = detector_data.get("good_origins")
    bad_origins = detector_data.get("bad_origins")
    precomputed_weights = detector_data.get("weights")
    file_threshold = detector_data.get("threshold", 0.5)
    inclusion = detector_data.get("inclusion", 0)
    media_type = detector_data.get("media_type", "") or media_type

    weights = None
    threshold = file_threshold
    origin_derived = False

    if good_origins and bad_origins:
        from vtsearch.models.training import train_detector_from_origins

        weights, threshold = train_detector_from_origins(
            good_origins,
            bad_origins,
            inclusion,
            media_type,
        )
        if weights is not None:
            origin_derived = True

    if weights is None and precomputed_weights:
        # Fall back to pre-computed weights (origin files not on disk)
        weights = precomputed_weights
        threshold = file_threshold
        good_origins = None
        bad_origins = None

    if weights is None:
        raise ValueError("Detector file missing 'weights' or origin fields.")

    return NormalizedWeights(
        weights=weights,
        threshold=threshold,
        good_origins=good_origins if origin_derived else None,
        bad_origins=bad_origins if origin_derived else None,
        inclusion=inclusion,
        origin_derived=origin_derived,
    )
