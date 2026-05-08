"""Blueprint for extractor, localizer, and pregen-processor routes."""

from __future__ import annotations

from flask import Blueprint

from vtsearch.routes.processors_crud import (  # noqa: F401
    _EXTRACTOR_FACTORIES,
    _LOCALIZER_FACTORIES,
    _build_extractor,
    _build_localizer,
    _ensure_extractor_factories,
    _ensure_localizer_factories,
    processors_crud_bp,
)
from vtsearch.routes.processors_scoring import processors_scoring_bp  # noqa: F401

processors_bp = Blueprint("processors", __name__)

processors_bp.register_blueprint(processors_crud_bp)
processors_bp.register_blueprint(processors_scoring_bp)
