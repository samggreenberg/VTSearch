"""Tests for the dict-keyed per-media embedding accessor (Phase 2c).

``media["embeddings"]`` (a dict keyed by embedder name) is the *only* per-media
vector store; there is no singular ``media["embedding"]`` on a live media.  The
accessor reads only the dict, ``set_media_embedding`` writes only the dict, and
``ensure_embeddings_dict`` re-keys a legacy singular vector into the dict and
then drops the singular key (the on-load migration for old pickles).
"""

from __future__ import annotations

import numpy as np

from vtscore.embedding.media_vectors import (
    ensure_embeddings_dict,
    media_embedder_names,
    media_embedding,
    primary_embedder_name,
    set_media_embedding,
)


def _vec(seed: int, dim: int = 4) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


class TestMediaEmbedding:
    def test_singular_is_not_read(self):
        # A legacy-shaped media (singular only, no dict) is never read by the
        # accessor: there is no fallback to media["embedding"].
        media = {"embedding": _vec(1), "embedder": "siglip"}
        assert media_embedding(media) is None
        assert media_embedding(media, "siglip") is None

    def test_dict_entry_by_name(self):
        a, b = _vec(1), _vec(2)
        media = {"embeddings": {"siglip": a, "dinov3_patch": b}, "embedder": "siglip"}
        assert media_embedding(media, "siglip") is a
        assert media_embedding(media, "dinov3_patch") is b

    def test_primary_prefers_recorded_embedder(self):
        a, b = _vec(1), _vec(2)
        media = {"embeddings": {"siglip": a, "dinov3_patch": b}, "embedder": "dinov3_patch"}
        assert media_embedding(media) is b

    def test_primary_single_entry_without_recorded_embedder(self):
        a = _vec(1)
        media = {"embeddings": {"siglip": a}}
        assert media_embedding(media) is a

    def test_ambiguous_multiple_entries_returns_none(self):
        a, b = _vec(1), _vec(2)
        media = {"embeddings": {"x": a, "y": b}}
        # No recorded primary + >1 entry → no unambiguous primary vector.
        assert media_embedding(media) is None

    def test_named_lookup_misses_for_unbound_embedder(self):
        a = _vec(1)
        media = {"embeddings": {"siglip": a}, "embedder": "siglip"}
        assert media_embedding(media, "dinov3_patch") is None
        assert media_embedding(media, "siglip") is a

    def test_missing_everything_returns_none(self):
        assert media_embedding({}) is None
        assert media_embedding({}, "siglip") is None
        assert media_embedding({"embeddings": {}}) is None


class TestEnsureEmbeddingsDict:
    def test_materializes_from_singular_and_drops_it(self):
        v = _vec(1)
        media = {"embedding": v, "embedder": "siglip"}
        ensure_embeddings_dict(media)
        assert media["embeddings"] == {"siglip": v}
        assert "embedding" not in media

    def test_idempotent_drops_stray_singular(self):
        a, stray = _vec(1), _vec(2)
        media = {"embeddings": {"siglip": a}, "embedding": stray, "embedder": "siglip"}
        ensure_embeddings_dict(media)
        # The dict is authoritative; a stray singular is removed, not merged.
        assert media["embeddings"] == {"siglip": a}
        assert "embedding" not in media

    def test_no_dict_when_embedder_unknown(self):
        media = {"embedding": _vec(1)}
        ensure_embeddings_dict(media)
        assert "embeddings" not in media
        assert "embedding" not in media

    def test_no_op_without_vector(self):
        media = {"embedder": "siglip"}
        ensure_embeddings_dict(media)
        assert "embeddings" not in media


class TestSetMediaEmbedding:
    def test_writes_dict_only(self):
        v = _vec(1)
        media: dict = {}
        set_media_embedding(media, "siglip", v)
        assert media["embeddings"] == {"siglip": v}
        assert "embedding" not in media
        assert media["embedder"] == "siglip"

    def test_second_embedder_adds_entry_keeps_primary_name(self):
        a, b = _vec(1), _vec(2)
        media: dict = {}
        set_media_embedding(media, "siglip", a)
        set_media_embedding(media, "dinov3_patch", b)
        assert media["embeddings"]["siglip"] is a
        assert media["embeddings"]["dinov3_patch"] is b
        assert "embedding" not in media
        # The recorded primary stays the first embedder written.
        assert media["embedder"] == "siglip"


def test_primary_embedder_name():
    assert primary_embedder_name({"embedder": "siglip"}) == "siglip"
    assert primary_embedder_name({"embeddings": {"clap": _vec(1)}}) == "clap"
    assert primary_embedder_name({}) is None


class TestMediaEmbedderNames:
    def test_empty(self):
        assert media_embedder_names({}) == []

    def test_recorded_embedder_without_dict(self):
        # No dict yet (un-embedded media): fall back to the recorded name.
        assert media_embedder_names({"embedder": "siglip"}) == ["siglip"]

    def test_dict_keys(self):
        media = {"embeddings": {"siglip": _vec(1), "dinov3_patch": _vec(2)}}
        assert set(media_embedder_names(media)) == {"siglip", "dinov3_patch"}

    def test_recorded_primary_ordered_first(self):
        media = {
            "embeddings": {"siglip": _vec(1), "dinov3_patch": _vec(2)},
            "embedder": "dinov3_patch",
        }
        assert media_embedder_names(media) == ["dinov3_patch", "siglip"]

    def test_dict_takes_precedence_over_recorded_embedder(self):
        media = {"embeddings": {"a": _vec(1)}, "embedder": "siglip"}
        # "siglip" isn't a dict key, so it doesn't get prepended; the dict wins.
        assert media_embedder_names(media) == ["a"]
