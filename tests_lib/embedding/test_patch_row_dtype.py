"""``PATCH_ROW_DTYPE`` is the one dtype every patch-row site follows (#3159).

Three places hold patch rows: the ``patch_grid`` ingest stores, the flattened
region matrix the live scorer max-pools over, and the eval harness's MaxPatch
stack.  Each used to spell ``np.float16`` itself.  #3159 measured the cast by
flipping all three in-process; a site that kept its own literal would have
quietly re-quantised the float32 arm, and the study would have compared float16
against float16 and reported a clean null.  These tests pin both halves: the
shipped value, and that every site actually reads the name.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from vtscore.datasets.stages.embedding import _attach_patch_grid_to_media
from vtscore.embedding import matrix
from vtscore.eval.patch_styles import MaxPatchStyle

DIM = 8
GRID = 2


def _media(media_id: int) -> dict:
    rng = np.random.default_rng(media_id)
    cls = rng.standard_normal(DIM).astype(np.float32)
    grid = rng.standard_normal((GRID, GRID, DIM)).astype(np.float32)
    # A value float16 cannot hold exactly, so a silent re-cast is visible.
    grid[0, 0, 0] = 0.1
    return {
        "id": media_id,
        "media_type": "image",
        "embedder": "dinov3_patch",
        "embeddings": {"dinov3_patch": cls},
        "patch_grid": grid,
    }


def test_shipped_dtype_is_float16():
    # The memory tradeoff #3159 measured.  Changing it re-sizes every patch
    # dataset's resident matrix; do it on purpose, with the report in hand.
    assert matrix.PATCH_ROW_DTYPE is np.float16


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_every_site_follows_the_name(monkeypatch, dtype):
    monkeypatch.setattr(matrix, "PATCH_ROW_DTYPE", dtype)

    # 1. ingest storage
    media: dict[str, Any] = {"id": 1}
    raw = np.full((GRID, GRID, DIM), 0.1, dtype=np.float32)
    _attach_patch_grid_to_media(media, SimpleNamespace(patch_grid=raw))
    assert media["patch_grid"].dtype == dtype

    # 2. the live scorer's flattened region matrix
    snap = {1: _media(1), 2: _media(2)}
    region_matrix, _, _ = matrix._build_region_arrays(snap, [1, 2])
    assert region_matrix.dtype == dtype
    assert region_matrix.shape == (2 * (1 + GRID * GRID), DIM)

    # 3. the eval harness's MaxPatch stack
    style = MaxPatchStyle()
    assert style._rows_for_media(snap[1]).dtype == dtype
    _ids, flat, _starts = style._flattened(snap)
    assert flat.dtype == dtype

    # And the float32 path is not quantised anywhere on the way through.
    got = float(region_matrix[1, 0])
    if dtype is np.float32:
        assert got == np.float32(0.1)
    else:
        assert got == float(np.float16(0.1))
