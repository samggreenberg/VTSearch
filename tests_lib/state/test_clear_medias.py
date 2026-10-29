"""Regression test: ``clear_medias()`` releases the projection + pyramids.

``clear_medias()`` must null ``ctx._projection`` and empty ``ctx._pyramids`` so
that:

1. **Memory** — the tile pyramids (the largest Browse artifact, one per bin
   shape) are freed on dataset unload, not left resident.
2. **Correctness** — the projection build route serves a cached pyramid via a
   fast-path that does *not* re-check the media-id signature, so a stale pyramid
   left over a reload-with-changed-contents would otherwise be returned for the
   new data.
"""

from __future__ import annotations

import numpy as np

from vtscore.state import clear_medias
from vtscore.state.core import (
    DatasetContext,
    get_active_context,
    thread_dataset_context,
)


def test_clear_medias_releases_projection_and_pyramid():
    ctx = DatasetContext("test_clear_projection")
    with thread_dataset_context(ctx):
        ctx.medias[1] = {"id": 1, "embedding": np.ones(4, dtype=np.float32)}
        ctx._emb_matrix = np.ones((1, 4), dtype=np.float32)
        ctx._emb_matrix_ids = [1]
        ctx._projection = object()  # stand-in for a Projection
        ctx._pyramids = {"hex": object(), "square": object()}  # stand-ins for Pyramids

        assert get_active_context() is ctx
        clear_medias()

        # the new guard: projection + every cached pyramid are released
        assert ctx._projection is None
        assert ctx._pyramids == {}
        # and the pre-existing clears still hold
        assert ctx._emb_matrix is None
        assert ctx._emb_matrix_ids is None
        assert ctx.medias == {}
