"""Pickle loads normalize stored embeddings (L2-at-ingest invariant).

A ``vtsearch`` pickle stores whatever vectors it was written with - which,
for datasets produced before the app-wide L2-normalization change, may be
raw (non-unit) embeddings.  The pickle loader is one of the ingest
chokepoints that re-establishes the "every stored embedding is unit-norm"
invariant, so a legacy pickle opens with unit vectors without re-embedding.

See ``docs/design/vtsbrowse.md`` §Prerequisite
(Chokepoint placement: the pickle/import write path must be covered).
"""

from __future__ import annotations

import numpy as np

from vtscore.datasets.loader_pickle import (
    _build_pickle_full_media,
    _build_pickle_thin_media,
)


def _raw_info() -> dict:
    # A deliberately non-unit stored embedding (norm 5).
    return {
        "filename": "clip.wav",
        "embedding": [3.0, 4.0],
        "media_type": "audio",
        "embedder": "clap",
    }


def test_full_media_embedding_is_unit_norm():
    media = _build_pickle_full_media(
        new_id=1,
        media_info=_raw_info(),
        media_type="audio",
        media_bytes=b"abc",
        media_string=None,
        media_path=None,
        extra_fields=[],
    )
    np.testing.assert_allclose(np.linalg.norm(media["embedding"]), 1.0, atol=1e-6)
    assert media["embedding"].dtype == np.float32


def test_thin_media_embedding_is_unit_norm():
    media = _build_pickle_thin_media(
        new_id=1,
        media_info=_raw_info(),
        media_type="audio",
        media_path=None,
        extra_fields=[],
    )
    np.testing.assert_allclose(np.linalg.norm(media["embedding"]), 1.0, atol=1e-6)
    assert media["embedding"].dtype == np.float32
