"""Detector weight normalisation.

Detector JSON files must contain origin-based data (``good_origins`` /
``bad_origins``).  Weights are re-derived by resolving the original media
files, embedding, and training.

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


def normalize_detector_weights(
    detector_data: dict,
    *,
    media_type: str = "audio",
) -> NormalizedWeights:
    """Resolve detector weights from *detector_data* by training from origins.

    Requires ``good_origins`` and ``bad_origins`` in *detector_data*.
    Raises :class:`ValueError` if origins are missing or training fails.

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
    inclusion = detector_data.get("inclusion", 0)
    media_type = detector_data.get("media_type", "") or media_type

    if not good_origins or not bad_origins:
        raise ValueError(
            "Detector file must contain 'good_origins' and 'bad_origins' fields."
        )

    from vtsearch.models.training import train_detector_from_origins

    weights, threshold = train_detector_from_origins(
        good_origins,
        bad_origins,
        inclusion,
        media_type,
    )

    if weights is None:
        # Origin resolution failed (files not available on disk).  If the
        # detector data already contains pre-computed weights, use them as a
        # fallback so the detector is still usable without the original media.
        fallback_weights = detector_data.get("weights")
        fallback_threshold = detector_data.get("threshold")
        if fallback_weights and fallback_threshold is not None:
            weights = fallback_weights
            threshold = fallback_threshold
        else:
            raise ValueError(
                "Could not derive weights from origins — original media files "
                "may not be available on disk."
            )

    return NormalizedWeights(
        weights=weights,
        threshold=threshold,
        good_origins=good_origins,
        bad_origins=bad_origins,
        inclusion=inclusion,
    )
