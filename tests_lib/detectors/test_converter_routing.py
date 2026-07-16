"""Unit tests for the CLI converter-routing helpers.

Library-tier: these touch only ``vtscore.detectors.converter_routing`` and the
converter registry, no app modules.  The registry-lookup tests use the real
built-in converters; the ``route_and_embed`` tests stub the embed pass and (for
the fan-out case) the route so no models load.
"""

from __future__ import annotations

import numpy as np

from vtscore.detectors import converter_routing as cr


class TestRouteLookup:
    def test_converter_route_for_direct_and_missing(self):
        # video2image is a built-in route; the reverse (image→video) is not.
        assert cr.converter_route_for("video", "image") is not None
        assert cr.converter_route_for("image", "video") is None
        # audio2image exists (waveform render); image→audio does not.
        assert cr.converter_route_for("audio", "image") is not None
        assert cr.converter_route_for("image", "audio") is None

    def test_detector_can_score_direct(self):
        assert cr.detector_can_score("image", {"image"}) is True
        assert cr.detector_can_score("audio", {"audio"}) is True

    def test_detector_can_score_via_converter(self):
        # An image detector reaches a video/document/audio dataset by a one-hop
        # converter route.
        assert cr.detector_can_score("image", {"video"}) is True
        assert cr.detector_can_score("image", {"document"}) is True
        assert cr.detector_can_score("image", {"audio"}) is True

    def test_detector_cannot_score_without_route(self):
        # No image→audio converter, so an audio detector can't score images.
        assert cr.detector_can_score("audio", {"image"}) is False

    def test_legacy_empty_target_matches_anything(self):
        assert cr.detector_can_score("", {"image"}) is True
        assert cr.detector_can_score("", set()) is True

    def test_mixed_source_types_match_if_any_reachable(self):
        # image reachable from video even though "text" alone is not.
        assert cr.detector_can_score("image", {"text", "video"}) is True


def _fake_embed_factory(dim: int = 4):
    """Return an ``embed_missing`` stub that stamps a deterministic vector."""

    def _fake_embed(medias, name="", on_progress=None):
        vec = np.ones(dim, dtype=np.float32)
        for m in medias.values():
            m["embeddings"] = {name: vec}
            m["embedder"] = name

    return _fake_embed


class TestRouteAndEmbed:
    def test_identity_route_preserves_source_ids(self, monkeypatch):
        monkeypatch.setattr("vtscore.datasets.stages.embedding.embed_missing", _fake_embed_factory())
        src = {1: {"media_type": "image"}, 2: {"media_type": "image"}}
        scoring, mapping = cr.route_and_embed(src, "image", "clip")

        assert len(scoring) == 2
        # Each scoring id maps back to a distinct source id (identity).
        assert set(mapping.values()) == {1, 2}
        for m in scoring.values():
            assert "clip" in m["embeddings"]

    def test_converter_output_maps_back_to_source(self, monkeypatch):
        class _FakeConv:
            source_type = "video"
            target_type = "image"

            def convert_normalized(self, media, params):
                return [{"media_bytes": b"frame-a"}, {"media_bytes": b"frame-b"}]

        monkeypatch.setattr(
            cr, "converter_route_for", lambda s, t: _FakeConv() if (s, t) == ("video", "image") else None
        )
        monkeypatch.setattr("vtscore.datasets.stages.embedding.embed_missing", _fake_embed_factory())
        src = {5: {"media_type": "video"}}
        scoring, mapping = cr.route_and_embed(src, "image", "clip")

        # One video fanned out into two frames, both attributed to source id 5.
        assert len(scoring) == 2
        assert set(mapping.values()) == {5}
        for m in scoring.values():
            assert m["media_type"] == "image"

    def test_unroutable_media_are_dropped(self, monkeypatch):
        monkeypatch.setattr("vtscore.datasets.stages.embedding.embed_missing", _fake_embed_factory())
        # A text media with no route to image is silently omitted.
        src = {1: {"media_type": "image"}, 2: {"media_type": "text"}}
        scoring, mapping = cr.route_and_embed(src, "image", "clip")

        assert len(scoring) == 1
        assert set(mapping.values()) == {1}

    def test_direct_media_without_embedding_dropped(self, monkeypatch):
        # Embed pass leaves the vector unset (unreadable source): drop, not crash.
        def _no_embed(medias, name="", on_progress=None):
            return None

        monkeypatch.setattr("vtscore.datasets.stages.embedding.embed_missing", _no_embed)
        src = {1: {"media_type": "image"}}
        scoring, mapping = cr.route_and_embed(src, "image", "clip")

        assert scoring == {}
        assert mapping == {}
