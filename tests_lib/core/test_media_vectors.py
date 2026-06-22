"""Tests for the dict-keyed per-media embedding accessor (Phase 2 substrate).

The accessor resolves a media's embedding from ``media["embeddings"]`` (the
dict-keyed form) with a fallback to the legacy singular ``media["embedding"]``,
so the two representations coexist during the migration.
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
    def test_legacy_singular_only(self):
        v = _vec(1)
        media = {"embedding": v, "embedder": "siglip"}
        assert media_embedding(media) is v
        assert media_embedding(media, "siglip") is v

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

    def test_ambiguous_multiple_entries_falls_back_to_singular(self):
        a, b, s = _vec(1), _vec(2), _vec(3)
        media = {"embeddings": {"x": a, "y": b}, "embedding": s}
        # No recorded primary + >1 entry → legacy singular wins.
        assert media_embedding(media) is s

    def test_named_lookup_misses_when_embedder_differs(self):
        s = _vec(1)
        media = {"embedding": s, "embedder": "siglip"}
        # Singular belongs to siglip; a request for a different embedder misses.
        assert media_embedding(media, "dinov3_patch") is None
        assert media_embedding(media, "siglip") is s

    def test_missing_everything_returns_none(self):
        assert media_embedding({}) is None
        assert media_embedding({}, "siglip") is None


class TestEnsureEmbeddingsDict:
    def test_materializes_from_singular(self):
        v = _vec(1)
        media = {"embedding": v, "embedder": "siglip"}
        ensure_embeddings_dict(media)
        assert media["embeddings"] == {"siglip": v}

    def test_idempotent(self):
        a = _vec(1)
        media = {"embeddings": {"siglip": a}, "embedder": "siglip"}
        ensure_embeddings_dict(media)
        assert media["embeddings"] == {"siglip": a}

    def test_no_dict_when_embedder_unknown(self):
        media = {"embedding": _vec(1)}
        ensure_embeddings_dict(media)
        assert "embeddings" not in media

    def test_no_op_without_vector(self):
        media = {"embedder": "siglip"}
        ensure_embeddings_dict(media)
        assert "embeddings" not in media


class TestSetMediaEmbedding:
    def test_writes_dict_and_primary_mirror(self):
        v = _vec(1)
        media: dict = {}
        set_media_embedding(media, "siglip", v)
        assert media["embeddings"] == {"siglip": v}
        assert media["embedding"] is v
        assert media["embedder"] == "siglip"

    def test_second_embedder_keeps_primary_mirror(self):
        a, b = _vec(1), _vec(2)
        media: dict = {}
        set_media_embedding(media, "siglip", a)
        set_media_embedding(media, "dinov3_patch", b)
        # Both stored; the singular mirror stays on the primary (first/recorded).
        assert media["embeddings"]["siglip"] is a
        assert media["embeddings"]["dinov3_patch"] is b
        assert media["embedding"] is a
        assert media["embedder"] == "siglip"


def test_primary_embedder_name():
    assert primary_embedder_name({"embedder": "siglip"}) == "siglip"
    assert primary_embedder_name({"embeddings": {"clap": _vec(1)}}) == "clap"
    assert primary_embedder_name({}) is None


class TestMediaEmbedderNames:
    def test_empty(self):
        assert media_embedder_names({}) == []

    def test_singular_only(self):
        assert media_embedder_names({"embedding": _vec(1), "embedder": "siglip"}) == ["siglip"]

    def test_dict_keys(self):
        media = {"embeddings": {"siglip": _vec(1), "dinov3_patch": _vec(2)}}
        assert set(media_embedder_names(media)) == {"siglip", "dinov3_patch"}

    def test_recorded_primary_ordered_first(self):
        media = {
            "embeddings": {"siglip": _vec(1), "dinov3_patch": _vec(2)},
            "embedder": "dinov3_patch",
        }
        assert media_embedder_names(media) == ["dinov3_patch", "siglip"]

    def test_dict_takes_precedence_over_singular_embedder(self):
        media = {"embeddings": {"a": _vec(1)}, "embedder": "siglip"}
        # "siglip" isn't a dict key, so it doesn't get prepended; the dict wins.
        assert media_embedder_names(media) == ["a"]
