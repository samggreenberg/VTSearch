"""Detector route blueprints."""

from vtsearch.routes.detectors.find import detector_find_bp
from vtsearch.routes.detectors.registry import detectors_registry_bp
from vtsearch.routes.detectors.scoring import detector_scoring_bp
from vtsearch.routes.detectors.store import detectors_bp

__all__ = [
    "detector_find_bp",
    "detector_scoring_bp",
    "detectors_bp",
    "detectors_registry_bp",
]
