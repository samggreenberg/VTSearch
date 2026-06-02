"""Persist and restore a Projection + Pyramid to a sidecar ``.projection`` file.

Uses NumPy's ``.npz`` container: the ``(N, 2)`` coordinate array is stored as a
native NumPy array (``coords``), the id vector as ``ids``, and all remaining
metadata (projection_id, method, pyramid config, levels, tiles/cells) is packed
as a single UTF-8 JSON blob (``meta``).

This is the persistence half of the carve-out from the "No Persisted Vectors or
MLPs" rule (see ``docs/design/vtsbrowse.md``).  Only 2-D projection coordinates
and the derived hex pyramid are stored — never embeddings, never MLP weights.
"""

from __future__ import annotations

import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from vtscore.projection.pyramid import HexCell, LevelMeta, Pyramid, Tile
from vtscore.projection.umap_projection import Projection

logger = logging.getLogger(__name__)


def projection_sidecar_path(pkl_path: str | Path) -> Path:
    """Return the ``.projection`` sidecar path for a dataset pickle."""
    return Path(pkl_path).with_suffix(".projection")


def save_projection(
    pkl_path: str | Path,
    projection: Projection,
    pyramid: Pyramid,
) -> Path:
    """Serialize *projection* + *pyramid* to a ``.projection`` sidecar next to *pkl_path*."""
    sidecar = projection_sidecar_path(pkl_path)

    meta: dict[str, Any] = {
        "projection_id": projection.projection_id,
        "method": projection.method,
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

    meta_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")

    buf = BytesIO()
    np.savez_compressed(
        buf,
        coords=projection.coords,
        ids=np.asarray(projection.ids, dtype=np.int64),
        meta=np.frombuffer(meta_bytes, dtype=np.uint8),
    )

    sidecar.parent.mkdir(parents=True, exist_ok=True)
    import os
    import tempfile

    fd, tmp = tempfile.mkstemp(dir=sidecar.parent, suffix=".projection.tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(buf.getvalue())
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(sidecar))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    logger.info("Saved projection sidecar: %s", sidecar)
    return sidecar


def load_projection(pkl_path: str | Path) -> tuple[Projection, Pyramid] | None:
    """Load a Projection + Pyramid from the ``.projection`` sidecar, or ``None``."""
    sidecar = projection_sidecar_path(pkl_path)
    if not sidecar.exists():
        return None

    try:
        with np.load(str(sidecar), allow_pickle=False) as npz:
            coords = np.ascontiguousarray(npz["coords"], dtype=np.float32)
            ids = npz["ids"].tolist()
            meta_bytes = npz["meta"].tobytes()

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
                HexCell(
                    q=c["q"],
                    r=c["r"],
                    cx=c["cx"],
                    cy=c["cy"],
                    count=c["count"],
                    rep_id=c["rep_id"],
                )
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
        )

        logger.info("Loaded projection sidecar: %s", sidecar)
        return projection, pyramid

    except Exception:
        logger.warning("Failed to load projection sidecar %s", sidecar, exc_info=True)
        return None
