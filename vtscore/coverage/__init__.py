"""Coverage Atlas: the pure-algorithm half of the coverage machinery.

This package holds the *structure* — a hierarchical k-means partition with
per-class evidence channels, von Mises-Fisher moments and calibrated
typicality ranks (see :mod:`vtscore.coverage.atlas` and
``docs/plans/coverage-atlas.md``).  It touches no
:class:`~vtscore.state.core.DatasetContext` and takes no lock: everything
here is a function of the embeddings handed to it.

The *wiring* that builds an atlas for the active dataset, replays a
detector's votes into it and caches it on a context lives separately, in
:mod:`vtscore.state.coverage`.  Keeping the two apart is the point of this
package: they used to sit side by side as ``state/coverage.py`` and
``state/coverage_atlas.py``, a near-homograph pair in which only one of the
two was actually state.
"""

from __future__ import annotations

from vtscore.coverage.atlas import (
    COVERAGE_ATLAS_DEFAULT_K,
    COVERAGE_ATLAS_MAX_DEPTH,
    COVERAGE_ATLAS_MIN_NODE_SIZE,
    CoverageAtlas,
    auto_max_depth,
    domain_shift_report,
)

__all__ = [
    "COVERAGE_ATLAS_DEFAULT_K",
    "COVERAGE_ATLAS_MAX_DEPTH",
    "COVERAGE_ATLAS_MIN_NODE_SIZE",
    "CoverageAtlas",
    "auto_max_depth",
    "domain_shift_report",
]
