"""Behavioural pins for the fake embedders both test tiers run on.

The structural gate in ``test_test_tier_helpers.py`` stops the two conftests
re-growing their own copies; this file pins what the surviving copy must *do*.
The library tier's old copy failed both invariants below (issue #3424): it read
``arg["media_path"]`` outside its ``try``, so a media dict without that key
raised ``KeyError``, and it seeded off ``hash(str(path))`` — salted per process
by ``PYTHONHASHSEED``, so the same media embedded differently in different
xdist workers, and every in-memory media collapsed onto one vector.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from tests_shared.embedding_stubs import EMBEDDING_DIM, fake_embed_audio, fake_embed_text

_REPO_ROOT = Path(__file__).resolve().parents[2]


class TestFakeEmbedAudio:
    def test_unit_norm_vector_of_the_right_width(self):
        vec = fake_embed_audio({"id": 1, "media_bytes": b"abc"})
        assert vec.shape == (EMBEDDING_DIM,)
        assert vec.dtype == np.float32
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)

    def test_in_memory_medias_get_distinct_vectors(self):
        """Clipper output carries ``media_bytes`` and no ``media_path``."""
        a = fake_embed_audio({"id": 1, "media_bytes": b"clip-one"})
        b = fake_embed_audio({"id": 2, "media_bytes": b"clip-two"})
        assert not np.allclose(a, b)

    def test_absent_media_path_key_is_not_an_error(self):
        assert fake_embed_audio({"id": 7}).shape == (EMBEDDING_DIM,)

    def test_none_media_path_falls_through_to_the_bytes(self):
        a = fake_embed_audio({"id": 1, "media_path": None, "media_bytes": b"one"})
        b = fake_embed_audio({"id": 2, "media_path": None, "media_bytes": b"two"})
        assert not np.allclose(a, b)

    def test_unreadable_path_falls_through_to_the_bytes(self, tmp_path):
        missing = str(tmp_path / "gone.wav")
        a = fake_embed_audio({"id": 1, "media_path": missing, "media_bytes": b"one"})
        b = fake_embed_audio({"id": 2, "media_path": missing, "media_bytes": b"two"})
        assert not np.allclose(a, b)

    def test_seeds_off_file_contents_not_the_path(self, tmp_path):
        """Two files written to the same path must still differ."""
        path = tmp_path / "clip.wav"
        path.write_bytes(b"first payload")
        first = fake_embed_audio(str(path))
        path.write_bytes(b"second payload")
        assert not np.allclose(first, fake_embed_audio(str(path)))

    def test_media_dict_and_bare_path_agree(self, tmp_path):
        path = tmp_path / "clip.wav"
        path.write_bytes(b"payload")
        assert np.allclose(fake_embed_audio(str(path)), fake_embed_audio({"id": 1, "media_path": str(path)}))


class TestSeedsAreProcessStable:
    """Every seed must be content-derived, never ``hash()``.

    ``hash()`` on a str is salted per interpreter, so a ``hash()``-seeded stub
    hands each xdist worker a different vector for the same media — a
    non-determinism that only shows up as a rare cross-worker failure. Compare
    against a *fresh* interpreter (with a deliberately different
    ``PYTHONHASHSEED``) rather than re-calling in-process, which cannot detect
    it.
    """

    def test_vectors_match_a_separate_interpreter(self):
        script = (
            "import json,sys;"
            f"sys.path.insert(0, {str(_REPO_ROOT)!r});"
            "from tests_shared.embedding_stubs import fake_embed_audio, fake_embed_text;"
            "print(json.dumps(["
            "  fake_embed_audio({'id': 3}).tolist()[:8],"
            "  fake_embed_audio({'id': 4, 'media_bytes': b'xyz'}).tolist()[:8],"
            "  fake_embed_text('a query').tolist()[:8],"
            "]))"
        )
        import json
        import os

        # Inherit the environment (the venv, HOME, and the numba/matplotlib
        # cache dirs some imports want) and override only the hash seed.
        env = {**os.environ, "PYTHONHASHSEED": "12345"}
        out = subprocess.run(  # noqa: S603
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env=env,
            cwd=_REPO_ROOT,
        )
        other = json.loads(out.stdout)
        assert np.allclose(other[0], fake_embed_audio({"id": 3})[:8])
        assert np.allclose(other[1], fake_embed_audio({"id": 4, "media_bytes": b"xyz"})[:8])
        assert np.allclose(other[2], fake_embed_text("a query")[:8])


class TestFakeEmbedText:
    def test_unit_norm_and_query_dependent(self):
        a = fake_embed_text("dogs barking")
        b = fake_embed_text("cars honking")
        assert a.shape == (EMBEDDING_DIM,)
        assert np.isclose(np.linalg.norm(a), 1.0, atol=1e-5)
        assert not np.allclose(a, b)
        assert np.allclose(a, fake_embed_text("dogs barking"))
