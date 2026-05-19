"""Shared stub builders for download tests.

Each demo-text dataset (AG News, BBC News, IMDB) follows the same pattern
for testing ``TextMediaType.load_demo_source``: stub the media type's
embedding model, stub a fake text embedder, then call ``load_demo_source``
with patched ``download_<dataset>``. Centralise the stubs here so the
per-dataset test files stay focused on their dataset-specific fixtures
and assertions.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock


def make_text_media_type_stub(embedding_dim: int = 768):
    """Return a ``TextMediaType`` instance with its embedding model stubbed."""
    from vtsearch.media.text.media_type import TextMediaType

    # _model is a runtime-only attr set by load_models on the subclass; not
    # declared on the ABC, so cast to Any to assign it directly.
    mt = cast(Any, TextMediaType())
    stub_model = MagicMock()
    stub_model.encode.return_value = [0.1] * embedding_dim
    mt._model = stub_model
    return mt


def make_text_embedder_stub(embedding_dim: int = 768, name: str = "e5"):
    """Return a ``MagicMock`` text embedder usable as the ``embedder=`` kwarg."""
    import numpy as np

    emb = MagicMock()
    emb.name = name
    emb.media_type_id = "text"
    emb._model = True
    emb._on_progress = lambda *a: None
    emb.embed_text_passage.return_value = np.zeros(embedding_dim)
    return emb
