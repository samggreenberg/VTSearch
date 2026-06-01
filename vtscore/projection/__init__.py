"""VictoryTones projection backend: UMAP layout + hex-tile pyramid.

The Flask-free core of the VictoryTones browse canvas (see
``docs/design/victorytones-audio-browser.md``).  Two stages:

- :func:`fit_projection` (Stage 1) reduces a dataset's ``(N, d)`` embedding
  matrix to a frozen ``(N, 2)`` :class:`Projection`.
- :func:`build_pyramid` (Stage 2) aggregates that projection into a
  multi-resolution :class:`Pyramid` of hex :class:`Tile` / :class:`HexCell`
  records the canvas streams while panning and zooming.

Persistence of the projection + pyramid (the carve-out from the
"No Persisted Vectors or MLPs" rule) and the HTTP endpoints live in the
VictoryTones app tier, not here.
"""

from __future__ import annotations

from vtscore.projection.hexbin import hex_center, hexbin_assign
from vtscore.projection.pyramid import (
    HexCell,
    LevelMeta,
    Pyramid,
    Tile,
    build_pyramid,
    max_useful_levels,
)
from vtscore.projection.umap_projection import Projection, fit_projection

__all__ = [
    "Projection",
    "fit_projection",
    "Pyramid",
    "Tile",
    "HexCell",
    "LevelMeta",
    "build_pyramid",
    "max_useful_levels",
    "hexbin_assign",
    "hex_center",
]
