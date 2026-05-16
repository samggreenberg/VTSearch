"""Blueprint exports for extractor, localizer, and pregen-processor routes."""

from __future__ import annotations

from vtsearch.routes.processors.crud import (  # noqa: F401
    _EXTRACTOR_FACTORIES,
    _LOCALIZER_FACTORIES,
    _build_extractor,
    _build_localizer,
    _ensure_extractor_factories,
    _ensure_localizer_factories,
    processors_crud_bp,
)
from vtsearch.routes.processors.scoring import processors_scoring_bp  # noqa: F401

__all__ = [
    "processors_crud_bp",
    "processors_scoring_bp",
]
