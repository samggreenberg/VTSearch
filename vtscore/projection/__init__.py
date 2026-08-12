"""VTSBrowse projection backend: UMAP layout + hex-tile pyramid.

The Flask-free core of the VTSBrowse browse canvas (see
``docs/plans/vtsbrowse.md``).  Two stages:

- :func:`fit_projection` (Stage 1) reduces a dataset's ``(N, d)`` embedding
  matrix to a frozen ``(N, 2)`` :class:`Projection`.
- :func:`build_pyramid` (Stage 2) aggregates that projection into a
  multi-resolution :class:`Pyramid` of :class:`Tile` / :class:`HexCell`
  records the canvas streams while panning and zooming.  The pyramid can tile
  the projection as hexagons or squares — see ``BIN_SHAPES`` and the
  ``bin_shape`` argument.  The shape is chosen per media type, not by the user:
  :func:`bin_shape_for_media_type` returns squares for browsable-thumbnail
  media (image/video/document) and hexes for the rest (audio/text).

Persistence of the projection + pyramid (the carve-out from the
"No Persisted Vectors or MLPs" rule) and the HTTP endpoints live in the
VTSearch Browse routes, not here.
"""

from __future__ import annotations

from vtscore.projection.compaction import compact_layout
from vtscore.projection.hexbin import hex_center, hexbin_assign
from vtscore.projection.labels import RegionLabel, RegionLabelSet, make_label_set
from vtscore.projection.params import (
    ProjectionParams,
    projection_embedder_for,
    resolve_projection_params,
)
from vtscore.projection.pyramid import (
    BIN_SHAPES,
    DEFAULT_BIN_SHAPE,
    HexCell,
    LevelMeta,
    Pyramid,
    Tile,
    bin_shape_for_media_type,
    build_pyramid,
    max_useful_levels,
    rebin_like,
    tile_member_ids,
)
from vtscore.projection.squarebin import square_center, squarebin_assign
from vtscore.projection.umap_projection import Projection, fit_projection, remove_ids

__all__ = [
    "Projection",
    "ProjectionParams",
    "RegionLabel",
    "RegionLabelSet",
    "compact_layout",
    "fit_projection",
    "make_label_set",
    "projection_embedder_for",
    "resolve_projection_params",
    "Pyramid",
    "Tile",
    "HexCell",
    "LevelMeta",
    "build_pyramid",
    "max_useful_levels",
    "rebin_like",
    "remove_ids",
    "tile_member_ids",
    "BIN_SHAPES",
    "DEFAULT_BIN_SHAPE",
    "bin_shape_for_media_type",
    "hexbin_assign",
    "hex_center",
    "squarebin_assign",
    "square_center",
]
