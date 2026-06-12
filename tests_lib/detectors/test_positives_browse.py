"""Tests for the ephemeral detector-positives browse context builder.

These exercise the library-tier logic (no Flask): resolving + embedding a
detector's positives into a throwaway ``DatasetContext`` with a ready
projection. Origin resolution and embedding are monkeypatched so the test
stays fast and offline.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pytest

import vtscore.detectors.resolver as resolver
from vtscore.detectors.positives_browse import (
    build_positives_browse_context,
    detpos_dataset_id,
)


def _detector_data(labels):
    return {"media_type": "text", "labelset": {"labels": labels}}


def _patch_resolve_and_embed(monkeypatch, tmp_path, *, resolvable=True):
    """Make every element resolve to a temp file and embed to a seeded vector."""
    f = tmp_path / "item.txt"
    f.write_text("hello")

    @contextmanager
    def fake_resolve(origin, origin_name="", filename=""):
        yield (f if resolvable else None)

    counter = {"n": 0}

    def fake_embed(file_path, media_type, embedder_name=""):
        # Seed per-call so the matrix has distinct, deterministic rows.
        counter["n"] += 1
        rng = np.random.default_rng(counter["n"])
        return rng.standard_normal(16).astype(np.float32)

    monkeypatch.setattr(resolver, "resolve_file_context", fake_resolve)
    monkeypatch.setattr(resolver, "embed_file", fake_embed)


def test_detpos_dataset_id_prefix():
    assert detpos_dataset_id("abc123").startswith("__detpos__")
    assert detpos_dataset_id("abc123").endswith("abc123")


def test_builds_context_over_positives_only(monkeypatch, tmp_path):
    _patch_resolve_and_embed(monkeypatch, tmp_path)
    data = _detector_data(
        [
            {"md5": "a" * 32, "label": "good"},
            {"md5": "b" * 32, "label": "good"},
            {"md5": "c" * 32, "label": "good"},
            {"md5": "d" * 32, "label": "bad"},  # excluded
        ]
    )

    ctx = build_positives_browse_context(
        data,
        detpos_dataset_id("det1"),
        embedder_name="e5",
        display_name="Det — positives",
    )

    # Only the 3 positives are materialised; the negative is dropped.
    assert len(ctx.medias) == 3
    for media in ctx.medias.values():
        assert media["media_type"] == "text"
        assert isinstance(media["embedding"], np.ndarray)
        assert media["media_bytes"] == b"hello"
        assert media["embedder"] == "e5"
    # A ready projection + hex pyramid is built up front.
    assert ctx._projection is not None
    assert "hex" in ctx._pyramids
    assert ctx.dataset_display_name == "Det — positives"


def test_reuses_cached_embeddings(monkeypatch, tmp_path):
    _patch_resolve_and_embed(monkeypatch, tmp_path)

    from vtscore.detectors.labelset_elements import stable_element_id
    from vtscore.datasets.labelset import LabeledElement

    elem = LabeledElement.from_dict({"md5": "a" * 32, "label": "good"})
    eid = stable_element_id(elem)
    cached = {eid: np.full(16, 0.5, dtype=np.float32)}

    data = _detector_data([{"md5": "a" * 32, "label": "good"}])
    ctx = build_positives_browse_context(
        data,
        detpos_dataset_id("det2"),
        embedder_name="e5",
        cached_embeddings=cached,
    )

    (media,) = ctx.medias.values()
    # The cached vector is used verbatim (not re-embedded).
    assert np.allclose(media["embedding"], 0.5)


def test_raises_when_nothing_resolves(monkeypatch, tmp_path):
    _patch_resolve_and_embed(monkeypatch, tmp_path, resolvable=False)
    data = _detector_data([{"md5": "a" * 32, "label": "good"}])
    with pytest.raises(ValueError, match="nothing to browse"):
        build_positives_browse_context(data, detpos_dataset_id("det3"), embedder_name="e5")
