"""Deterministic stand-ins for the real embedders, shared by both test tiers.

Tests don't need semantically meaningful embeddings; they need deterministic
unit-norm vectors of the right dimension.  Stubbing every registered embedder
avoids loading the ~600 MB CLAP model, the ~100-200 MB librosa/numba stack, and
the CLIP / X-CLIP / E5 / SigLIP weights a stray ``/api/sort`` would otherwise
download — roughly 700-800 MB of RSS per worker.
"""

from __future__ import annotations

import hashlib
from unittest.mock import patch

import numpy as np
import pytest

from vtscore.utils.hashing import content_md5

#: Dimension of every fake vector (CLAP audio's real width).
EMBEDDING_DIM = 512


def fake_embed_audio(arg):
    """Deterministic fake media embedding derived from the media's content.

    Accepts either a path (from the legacy ``embed_audio_file`` wrapper) or a
    media dict (from ``MediaEmbedder.embed_media``).  Uses the first 1000 bytes
    of the resolved file as a seed so that different files — even when written
    to the same temp path — produce distinct vectors.

    In-memory medias (e.g. clipper output) carry ``media_bytes`` but no
    ``media_path``; seed off those bytes so distinct clips still get distinct
    vectors, falling back to the media id only when neither is present.  Every
    seed is content-derived: nothing here may fall back to ``hash()``, which is
    salted per process and would hand different xdist workers different vectors
    for the same media.
    """
    data = None
    if isinstance(arg, dict):
        path = arg.get("media_path")
        if path:
            try:
                with open(path, "rb") as f:
                    data = f.read(1000)
            except Exception:
                data = None
        if data is None:
            raw = arg.get("media_bytes")
            data = bytes(raw[:1000]) if isinstance(raw, (bytes, bytearray)) else str(arg.get("id", arg)).encode()
    else:
        try:
            with open(arg, "rb") as f:
                data = f.read(1000)
        except Exception:
            data = str(arg).encode()
    seed = int(content_md5(data), 16) % 2**31
    rng = np.random.RandomState(seed)
    vec = rng.randn(EMBEDDING_DIM).astype(np.float32)
    # Real embedders L2-normalize at ingest, so the fakes must too:
    # region_similarity scores by dot product on the unit-norm assumption.
    return vec / np.linalg.norm(vec)


def fake_embed_text(text):
    """Deterministic fake text embedding derived from the query string."""
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % 2**31
    rng = np.random.RandomState(seed)
    vec = rng.randn(EMBEDDING_DIM).astype(np.float32)
    return vec / np.linalg.norm(vec)


def make_stub_embedding_models_fixture(embed_audio_file_target: str):
    """Build the session-scoped autouse fixture that stubs every embedder.

    *embed_audio_file_target* is the tier's own
    ``<tree>.fixtures.medias.embed_audio_file`` patch target — the one thing
    that genuinely differs between the two suites.

    Session-scoped: the ~40 patches are applied once and held for the whole run
    instead of being torn down and re-applied around each of the thousands of
    tests.  Tests needing different stub behavior can layer their own
    ``patch.object`` on top; it overrides the session patch and restores it on
    exit.  The media-type and embedder registries are read inside the fixture
    rather than at import time, so this factory can be called from anywhere in
    a conftest.
    """

    @pytest.fixture(scope="session", autouse=True)
    def _stub_embedding_models():
        """Prevent any embedder from loading real model weights during tests."""
        from contextlib import ExitStack

        from vtscore.media import all_embedders, all_types

        stack = ExitStack()
        stack.enter_context(patch(embed_audio_file_target, side_effect=fake_embed_audio))
        # Every registered media type and embedder gets stubbed, not just
        # audio: a test that accidentally touches the image/video/text/document
        # embedders (e.g. via ``/api/sort`` on an image dataset) would
        # otherwise try to download real weights.
        for mt in all_types():
            stack.enter_context(patch.object(mt, "embed_text", side_effect=fake_embed_text))
            stack.enter_context(patch.object(mt, "load_models"))
        for emb in all_embedders():
            stack.enter_context(patch.object(emb, "embed_media", side_effect=fake_embed_audio))
            stack.enter_context(patch.object(emb, "embed_text", side_effect=fake_embed_text))
            stack.enter_context(patch.object(emb, "load_models"))
        yield
        stack.close()

    return _stub_embedding_models
