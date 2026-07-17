"""Integration tests for CLI converter routing in the multi-detector scorer.

Exercises ``_score_medias_with_detectors`` on a dataset of mixed source types:
a detector targeting ``image`` scores native images directly and reaches videos
through a (stubbed) ``video2image`` route, with the per-frame scores aggregated
back to a single hit on the source video.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vtscore.cli import _score_medias_with_detectors
import vtscore.detectors.converter_routing as cr


def _constant_mlp() -> torch.nn.Module:
    """A Linear(4,1) that maps the all-ones vector to a high positive logit."""
    mlp = torch.nn.Linear(4, 1)
    with torch.no_grad():
        mlp.weight.fill_(1.0)
        mlp.bias.fill_(0.0)
    return mlp


def _stub_embed(monkeypatch, dim: int = 4) -> None:
    def _fake_embed(medias, name="", on_progress=None):
        vec = np.ones(dim, dtype=np.float32)
        for m in medias.values():
            m["embeddings"] = {name: vec}
            m["embedder"] = name

    monkeypatch.setattr("vtscore.datasets.stages.embedding.embed_missing", _fake_embed)


class TestCliConverterRouting:
    def test_video_routed_to_image_aggregates_to_one_hit(self, client, monkeypatch):
        class _FakeVideo2Image:
            source_type = "video"
            target_type = "image"

            def convert_normalized(self, media, params):
                # One video fans out into two frames.
                return [{"media_bytes": b"frame-a"}, {"media_bytes": b"frame-b"}]

        monkeypatch.setattr(
            cr, "converter_route_for", lambda s, t: _FakeVideo2Image() if (s, t) == ("video", "image") else None
        )
        _stub_embed(monkeypatch)

        medias = {
            1: {"id": 1, "media_type": "image", "filename": "pic.png"},
            2: {"id": 2, "media_type": "video", "filename": "clip.mp4"},
        }
        detector_mlps = {
            "img-det": {"mlp": _constant_mlp(), "threshold": 0.5, "media_type": "image", "embedder": "fake"},
        }

        results = _score_medias_with_detectors(medias, detector_mlps)

        assert "img-det" in results
        det = results["img-det"]
        # Two source media scored (image + video); the video's two frames
        # collapse to a single hit, so total hits + negatives == 2, keyed by
        # the *source* ids, not the throwaway per-frame ids.
        all_ids = {h["id"] for h in det["hits"]} | {h["id"] for h in det["negative_hits"]}
        assert all_ids == {1, 2}
        assert len(det["hits"]) + len(det["negative_hits"]) == 2

    def test_detector_targeting_unreachable_type_produces_no_results(self, client, monkeypatch):
        # No route from image → audio, so an audio detector scores nothing on an
        # image-only dataset (and doesn't crash).
        monkeypatch.setattr(cr, "converter_route_for", lambda s, t: None)
        _stub_embed(monkeypatch)

        medias = {1: {"id": 1, "media_type": "image", "filename": "pic.png"}}
        detector_mlps = {
            "aud-det": {"mlp": _constant_mlp(), "threshold": 0.5, "media_type": "audio", "embedder": "fake"},
        }

        results = _score_medias_with_detectors(medias, detector_mlps)
        assert results == {}

    def test_reclip_fanout_aggregates_to_one_hit(self, client, monkeypatch):
        # A detector carrying a re-clip spec splits each media into clips; the
        # per-clip scores collapse back to a single hit on the source media.
        def _fake_apply_clipper(clips_dict, name, params, on_progress=None, chain_steps=None, embedder=None):
            clips_dict.clear()
            for i in (1, 2):
                clips_dict[i] = {
                    "media_type": "audio",
                    "embeddings": {"fake": np.ones(4, dtype=np.float32)},
                    "embedder": "fake",
                    "filename": f"clip{i}",
                }

        monkeypatch.setattr("vtscore.datasets.stages.clipper._apply_clipper", _fake_apply_clipper)

        medias = {9: {"id": 9, "media_type": "audio", "filename": "rec.wav"}}
        detector_mlps = {
            "det": {
                "mlp": _constant_mlp(),
                "threshold": 0.5,
                "media_type": "audio",
                "embedder": "fake",
                "clipper": "sound_tiling",
                "clipper_params": {"duration": "2.0"},
            },
        }

        results = _score_medias_with_detectors(medias, detector_mlps)
        det = results["det"]
        all_ids = {h["id"] for h in det["hits"]} | {h["id"] for h in det["negative_hits"]}
        # Two clips of source media 9 collapse to one hit on 9.
        assert all_ids == {9}
        assert len(det["hits"]) + len(det["negative_hits"]) == 1

    def test_homogeneous_direct_scoring_one_hit_per_media(self, client, monkeypatch):
        _stub_embed(monkeypatch)
        medias = {
            1: {"id": 1, "media_type": "image", "filename": "a.png"},
            2: {"id": 2, "media_type": "image", "filename": "b.png"},
        }
        detector_mlps = {
            "img-det": {"mlp": _constant_mlp(), "threshold": 0.5, "media_type": "image", "embedder": "fake"},
        }

        results = _score_medias_with_detectors(medias, detector_mlps)
        det = results["img-det"]
        all_ids = {h["id"] for h in det["hits"]} | {h["id"] for h in det["negative_hits"]}
        assert all_ids == {1, 2}
        assert len(det["hits"]) + len(det["negative_hits"]) == 2


@pytest.mark.parametrize(
    "target, source_types, expected",
    [
        ("image", {"image"}, True),
        ("image", {"video"}, True),
        ("audio", {"image"}, False),
        ("", {"image"}, True),
    ],
)
def test_detector_can_score_matrix(target, source_types, expected):
    assert cr.detector_can_score(target, source_types) is expected
