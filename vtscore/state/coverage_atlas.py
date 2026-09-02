"""Deprecated alias for :mod:`vtscore.coverage.atlas`.

The Coverage Atlas is a pure algorithm — a hierarchical k-means partition over
embeddings — and never touched :class:`~vtscore.state.core.DatasetContext` or
``_state_lock``.  It lived here only because the *wiring* that builds it for the
active dataset (:mod:`vtscore.state.coverage`) does, leaving a near-homograph
pair in which only one of the two was state.  The algorithm has moved to
:mod:`vtscore.coverage.atlas`; the wiring stays put.

This module re-exports the new location so existing imports keep working, and
warns on import.  Import from :mod:`vtscore.coverage` instead::

    from vtscore.coverage import CoverageAtlas, auto_max_depth, domain_shift_report
"""

from __future__ import annotations

import warnings
from typing import Any

from vtscore.coverage import atlas as _atlas
from vtscore.coverage.atlas import (  # noqa: F401
    COVERAGE_ATLAS_DEFAULT_K,
    COVERAGE_ATLAS_MAX_DEPTH,
    COVERAGE_ATLAS_MIN_NODE_SIZE,
    CoverageAtlas,
    auto_max_depth,
    domain_shift_report,
)

warnings.warn(
    "vtscore.state.coverage_atlas has moved to vtscore.coverage.atlas; "
    "import from vtscore.coverage instead. This alias will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "COVERAGE_ATLAS_DEFAULT_K",
    "COVERAGE_ATLAS_MAX_DEPTH",
    "COVERAGE_ATLAS_MIN_NODE_SIZE",
    "CoverageAtlas",
    "auto_max_depth",
    "domain_shift_report",
]


def __getattr__(name: str) -> Any:
    """Forward anything not re-exported above (module privates, new symbols)."""
    return getattr(_atlas, name)
