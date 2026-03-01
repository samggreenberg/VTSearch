"""Blueprint for detector, extractor, and localizer routes.

This module is a re-export facade.  The actual routes are split across:

* ``detectors_crud`` — CRUD for autorun detectors/extractors/localizers, pregen processors
* ``detectors_training`` — Vote-based training, label-based training, multi-find
* ``detectors_scoring`` — Detector scoring, extractor/localizer execution

All three sub-blueprints are registered under ``detectors_bp`` so
existing code that imports ``detectors_bp`` continues to work.
"""

from __future__ import annotations

from flask import Blueprint

from vtsearch.routes.detectors_crud import (  # noqa: F401
    SERVER_DETECTOR_DIR,
    _EXTRACTOR_FACTORIES,
    _LOCALIZER_FACTORIES,
    _build_extractor,
    _build_localizer,
    _ensure_extractor_factories,
    _ensure_localizer_factories,
    detectors_crud_bp,
)
from vtsearch.routes.detectors_scoring import detectors_scoring_bp  # noqa: F401
from vtsearch.routes.detectors_training import detectors_training_bp  # noqa: F401

detectors_bp = Blueprint("detectors", __name__)

# Register sub-blueprints so all routes appear under the single detectors_bp.
detectors_bp.register_blueprint(detectors_crud_bp)
detectors_bp.register_blueprint(detectors_training_bp)
detectors_bp.register_blueprint(detectors_scoring_bp)
