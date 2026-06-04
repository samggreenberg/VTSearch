"""Projection serialization helpers.

Provides :func:`_pyramid_to_meta` and :func:`_rebuild_from_npz_arrays`,
shared by the ZIP container module (``vtscore.datasets.container``) which
handles all persistence.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from vtscore.projection.pyramid import HexCell, LevelMeta, Pyramid, Tile
from vtscore.projection.umap_projection import Projection


def _pyramid_to_meta(projection: Projection, pyramid: Pyramid) -> dict[str, Any]:
    """Convert a Projection + Pyramid to a JSON-serializable dict."""
    return {
        "projection_id": projection.projection_id,
        "method": projection.method,
        "bin_shape": pyramid.bin_shape,
        "base_radius": pyramid.base_radius,
        "tile_span": pyramid.tile_span,
        "point_count": pyramid.point_count,
        "bounds": list(pyramid.bounds),
        "levels": [{"level": lm.level, "radius": lm.radius, "n_cells": lm.n_cells} for lm in pyramid.levels],
        "tiles": {
            f"{k[0]},{k[1]},{k[2]}": [
                {
                    "q": c.q,
                    "r": c.r,
                    "cx": c.cx,
                    "cy": c.cy,
                    "count": c.count,
                    "rep_id": c.rep_id,
                }
                for c in tile.cells
            ]
            for k, tile in pyramid.tiles.items()
        },
    }


def _rebuild_from_npz_arrays(
    coords: np.ndarray,
    ids: list[int],
    meta_bytes: bytes,
) -> tuple[Projection, Pyramid]:
    """Reconstruct a Projection + Pyramid from raw npz components."""
    meta = json.loads(meta_bytes.decode("utf-8"))

    projection = Projection(
        projection_id=meta["projection_id"],
        ids=ids,
        coords=coords,
        method=meta["method"],
    )

    levels = [LevelMeta(level=lm["level"], radius=lm["radius"], n_cells=lm["n_cells"]) for lm in meta["levels"]]

    tiles: dict[tuple[int, int, int], Tile] = {}
    for key_str, cell_dicts in meta["tiles"].items():
        parts = key_str.split(",")
        level, tx, ty = int(parts[0]), int(parts[1]), int(parts[2])
        cells = [
            HexCell(q=c["q"], r=c["r"], cx=c["cx"], cy=c["cy"], count=c["count"], rep_id=c["rep_id"])
            for c in cell_dicts
        ]
        tiles[(level, tx, ty)] = Tile(level=level, tx=tx, ty=ty, cells=cells)

    pyramid = Pyramid(
        projection_id=meta["projection_id"],
        bounds=tuple(meta["bounds"]),
        base_radius=meta["base_radius"],
        tile_span=meta["tile_span"],
        point_count=meta["point_count"],
        levels=levels,
        tiles=tiles,
        # Containers written before the hex/square toggle have no bin_shape; they
        # are hex by construction, so default accordingly.
        bin_shape=meta.get("bin_shape", "hex"),
    )

    return projection, pyramid
