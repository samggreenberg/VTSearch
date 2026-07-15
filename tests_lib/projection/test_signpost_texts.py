"""Library-tier tests for the signpost text provider layer.

Covers ``vtscore.projection.signpost_texts``: the zero-shot tag provider, the
content provider, the per-media-type registry, and the cache-aware
``ensure_signpost_texts`` entry point (stamping, signature invalidation,
reuse).  No Flask, no models, no toponymy — pure numpy + fakes.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.projection import signpost_texts as st


class FakeTextEmbedder:
    """Maps "<template> <term>" strings onto fixed unit basis vectors."""

    def __init__(self, terms: list[str], name: str = "fake_clap", dim: int = 8):
        self.name = name
        self.terms = terms
        self.dim = dim
        self.calls = 0

    def embed_text(self, text: str):
        self.calls += 1
        for i, term in enumerate(self.terms):
            if term in text:
                vec = np.zeros(self.dim, dtype=np.float32)
                vec[i] = 1.0
                return vec
        return None


@pytest.fixture(autouse=True)
def _fresh_vocab_cache():
    """Isolate the process-scoped vocabulary-vector cache between tests."""
    saved = dict(st._vocab_cache)
    st._vocab_cache.clear()
    try:
        yield
    finally:
        st._vocab_cache.clear()
        st._vocab_cache.update(saved)


@pytest.fixture(autouse=True)
def _fresh_provider_registry():
    """Restore the provider registry after tests that register fakes."""
    saved = dict(st._PROVIDERS)
    try:
        yield
    finally:
        st._PROVIDERS.clear()
        st._PROVIDERS.update(saved)


def _basis_matrix(rows: list[int], dim: int = 8) -> np.ndarray:
    matrix = np.zeros((len(rows), dim), dtype=np.float32)
    for r, term_idx in enumerate(rows):
        matrix[r, term_idx] = 1.0
    return matrix


class TestZeroShotTagProvider:
    def _provider(self, monkeypatch, terms, top_k=2):
        monkeypatch.setattr(st, "_load_vocab", lambda asset: terms)
        return st.ZeroShotTagProvider(
            name="tags:test", vocab_asset="unused.txt", template="The sound of {}", top_k=top_k
        )

    def test_top_tags_ranked_by_similarity(self, monkeypatch):
        terms = ["dog", "rain", "car"]
        provider = self._provider(monkeypatch, terms)
        embedder = FakeTextEmbedder(terms)
        medias = {1: {}, 2: {}}
        matrix = _basis_matrix([2, 0])  # media 1 ≈ "car", media 2 ≈ "dog"

        texts = provider.build_texts([1, 2], medias, matrix, embedder)
        assert texts[1].split(", ")[0] == "car"
        assert texts[2].split(", ")[0] == "dog"
        assert len(texts[1].split(", ")) == 2  # top_k

    def test_vocab_embedded_once_per_embedder(self, monkeypatch):
        terms = ["dog", "rain", "car"]
        provider = self._provider(monkeypatch, terms)
        embedder = FakeTextEmbedder(terms)
        medias = {1: {}}
        matrix = _basis_matrix([0])

        provider.build_texts([1], medias, matrix, embedder)
        first_pass = embedder.calls
        assert first_pass == len(terms)
        provider.build_texts([1], medias, matrix, embedder)
        assert embedder.calls == first_pass  # cache hit, no re-embedding

    def test_unembeddable_terms_are_dropped(self, monkeypatch):
        # "storm" never embeds; the term list and matrix must stay aligned.
        terms = ["dog", "storm", "car"]
        provider = self._provider(monkeypatch, terms, top_k=3)
        embedder = FakeTextEmbedder(["dog", "car"])  # knows only two
        texts = provider.build_texts([1], {1: {}}, _basis_matrix([0]), embedder)
        assert "storm" not in texts[1]

    def test_signature_includes_embedder(self, monkeypatch):
        provider = self._provider(monkeypatch, ["dog"])
        assert provider.signature(FakeTextEmbedder([], name="a")) != provider.signature(FakeTextEmbedder([], name="b"))


class TestContentTextProvider:
    def test_returns_truncated_content(self):
        provider = st.ContentTextProvider(max_chars=11)
        medias = {1: {"content": "  hello world this is long  "}, 2: {"content": ""}, 3: {}}
        texts = provider.build_texts([1, 2, 3], medias, np.empty((3, 0)), embedder=None)
        assert texts == {1: "hello world"}


class TestRegistry:
    def test_defaults_cover_audio_image_text(self):
        audio, image = st.provider_for("audio"), st.provider_for("image")
        assert audio is not None and audio.name == "tags:audioset527"
        assert image is not None and image.name == "tags:openimages600"
        assert isinstance(st.provider_for("text"), st.ContentTextProvider)
        assert st.provider_for("weird_type") is None

    def test_register_replaces(self):
        provider = st.ContentTextProvider(field="caption")
        st.register_signpost_text_provider("video", provider)
        assert st.provider_for("video") is provider

    def test_shipped_vocab_assets_load(self):
        assert len(st._load_vocab("audioset527_labels.txt")) == 527
        assert len(st._load_vocab("openimages600_labels.txt")) == 601


class CountingProvider:
    """Registry-pluggable provider that counts build calls."""

    name = "counting"

    def __init__(self, signature="counting:v1"):
        self._signature = signature
        self.build_calls = 0

    def signature(self, embedder):
        return self._signature

    def build_texts(self, ids, medias, matrix, embedder, on_progress=None):
        self.build_calls += 1
        return {mid: f"text-{mid}" for mid in ids}


class TestEnsureSignpostTexts:
    def _medias(self, n=4, media_type="fake"):
        return {i: {"id": i, "media_type": media_type} for i in range(1, n + 1)}

    def test_computes_and_stamps_missing(self):
        provider = CountingProvider()
        st.register_signpost_text_provider("fake", provider)
        medias = self._medias()
        ids = sorted(medias)

        texts = st.ensure_signpost_texts(medias, ids, np.zeros((4, 2), dtype=np.float32), None)
        assert texts == {i: f"text-{i}" for i in ids}
        assert medias[1][st.TEXT_FIELD] == "text-1"
        assert medias[1][st.SOURCE_FIELD] == "counting:v1"

    def test_cached_texts_are_reused(self):
        provider = CountingProvider()
        st.register_signpost_text_provider("fake", provider)
        medias = self._medias()
        ids = sorted(medias)
        matrix = np.zeros((4, 2), dtype=np.float32)

        st.ensure_signpost_texts(medias, ids, matrix, None)
        st.ensure_signpost_texts(medias, ids, matrix, None)
        assert provider.build_calls == 1  # second call served from the stamps

    def test_signature_change_invalidates_cache(self):
        provider = CountingProvider()
        st.register_signpost_text_provider("fake", provider)
        medias = self._medias()
        ids = sorted(medias)
        matrix = np.zeros((4, 2), dtype=np.float32)

        st.ensure_signpost_texts(medias, ids, matrix, None)
        provider._signature = "counting:v2"  # e.g. a different embedder/vocab
        st.ensure_signpost_texts(medias, ids, matrix, None)
        assert provider.build_calls == 2
        assert medias[1][st.SOURCE_FIELD] == "counting:v2"

    def test_partial_cache_computes_only_misses(self):
        provider = CountingProvider()
        st.register_signpost_text_provider("fake", provider)
        medias = self._medias()
        ids = sorted(medias)
        medias[2][st.TEXT_FIELD] = "pre-cached"
        medias[2][st.SOURCE_FIELD] = "counting:v1"

        seen: list[list[int]] = []
        original = provider.build_texts

        def spying(ids, *args, **kwargs):
            seen.append(list(ids))
            return original(ids, *args, **kwargs)

        provider.build_texts = spying
        texts = st.ensure_signpost_texts(medias, ids, np.zeros((4, 2), dtype=np.float32), None)
        assert seen == [[1, 3, 4]]
        assert texts is not None and texts[2] == "pre-cached"

    def test_no_provider_returns_none(self):
        medias = self._medias(media_type="no_such_type")
        assert st.ensure_signpost_texts(medias, sorted(medias), np.zeros((4, 2)), None) is None

    def test_empty_ids_returns_none(self):
        assert st.ensure_signpost_texts({}, [], np.empty((0, 0)), None) is None


class _DictProvider:
    """Primary-like provider that returns a fixed subset of texts, ignores matrix."""

    def __init__(self, texts: dict[int, str], name: str = "caption:test"):
        self._texts = texts
        self.name = name

    def signature(self, embedder):
        return self.name

    def build_texts(self, ids, medias, matrix, embedder, on_progress=None):
        return {mid: self._texts[mid] for mid in ids if mid in self._texts}


class _RaisingProvider:
    name = "caption:broken"

    def signature(self, embedder):
        return self.name

    def build_texts(self, ids, medias, matrix, embedder, on_progress=None):
        raise RuntimeError("model failed to load")


class _RecordingFallback:
    name = "tags:rec"

    def __init__(self):
        self.seen_ids: list[int] | None = None
        self.seen_rows: list[list[float]] | None = None

    def signature(self, embedder):
        return self.name

    def build_texts(self, ids, medias, matrix, embedder, on_progress=None):
        self.seen_ids = list(ids)
        self.seen_rows = [list(map(float, matrix[i])) for i in range(len(ids))]
        return {mid: f"tag-{mid}" for mid in ids}


class TestFallbackTextProvider:
    def test_primary_covers_all_skips_fallback(self):
        fallback = _RecordingFallback()
        fp = st.FallbackTextProvider(_DictProvider({1: "cap1", 2: "cap2"}), fallback)
        texts = fp.build_texts([1, 2], {1: {}, 2: {}}, np.zeros((2, 3), dtype=np.float32), None)
        assert texts == {1: "cap1", 2: "cap2"}
        assert fallback.seen_ids is None  # nothing missing → fallback untouched

    def test_partial_fills_missing_from_fallback(self):
        fallback = _RecordingFallback()
        fp = st.FallbackTextProvider(_DictProvider({1: "cap1"}), fallback)
        texts = fp.build_texts([1, 2], {1: {}, 2: {}}, np.zeros((2, 3), dtype=np.float32), None)
        assert texts == {1: "cap1", 2: "tag-2"}
        assert fallback.seen_ids == [2]  # only the miss went to the fallback

    def test_primary_raises_all_from_fallback(self):
        fallback = _RecordingFallback()
        fp = st.FallbackTextProvider(_RaisingProvider(), fallback)
        texts = fp.build_texts([1, 2], {1: {}, 2: {}}, np.zeros((2, 3), dtype=np.float32), None)
        assert texts == {1: "tag-1", 2: "tag-2"}
        assert fallback.seen_ids == [1, 2]

    def test_fallback_gets_aligned_matrix_rows(self):
        fallback = _RecordingFallback()
        fp = st.FallbackTextProvider(_DictProvider({1: "cap1"}), fallback)  # id 2,3 miss
        matrix = np.array([[1, 1], [2, 2], [3, 3]], dtype=np.float32)
        fp.build_texts([1, 2, 3], {1: {}, 2: {}, 3: {}}, matrix, None)
        # The fallback must receive exactly the missing ids' rows, in order.
        assert fallback.seen_ids == [2, 3]
        assert fallback.seen_rows == [[2.0, 2.0], [3.0, 3.0]]

    def test_signature_composes_both_sides(self):
        fp = st.FallbackTextProvider(_DictProvider({}, name="cap:x"), _RecordingFallback())
        assert fp.signature(None) == "cap:x|fallback=tags:rec"


class TestCaptionerSelection:
    def test_default_returns_base_provider(self):
        # No captioner opt-in (CoreConfig default {}) → the tag/content base.
        assert st.provider_for("image").name == "tags:openimages600"
        assert st.provider_for("audio").name == "tags:audioset527"

    def test_enabled_wraps_captioner_over_base(self, monkeypatch):
        monkeypatch.setattr(st, "_captioner_enabled", lambda mt: mt == "image")
        provider = st.provider_for("image")
        assert isinstance(provider, st.FallbackTextProvider)
        assert provider.primary.name == "caption:qwen2.5-vl-3b"
        assert provider.fallback.name == "tags:openimages600"
        # A type without opt-in still gets its bare base.
        assert st.provider_for("audio").name == "tags:audioset527"

    def test_enabled_but_no_captioner_returns_base(self, monkeypatch):
        # text has no captioner registered; opting it in is a no-op.
        monkeypatch.setattr(st, "_captioner_enabled", lambda mt: True)
        assert isinstance(st.provider_for("text"), st.ContentTextProvider)
