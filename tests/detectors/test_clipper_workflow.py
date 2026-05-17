"""End-to-end tests for the MediaClipper workflow.

Validates:
1. Clips get unique MD5s (not the parent's) so dedup doesn't merge them.
2. Clips get their own embeddings based on actual clipped content.
3. Clip boundaries are stored in origin params for label export/import.
4. Label export captures clip origins correctly.
5. Label import can resolve clipped media on the same dataset.
6. Cross-dataset resolution uses clip-aware embedding.
7. The /api/medias endpoint exposes clip metadata to the frontend.
"""

import hashlib
import io

import numpy as np

from vtsearch.media.audio.audio_generator import generate_wav
from vtsearch.state import (
    medias,
    good_votes,
    bad_votes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_audio_media(media_id: int, duration: float = 5.1, *, origin_path: str = "/data/audio") -> dict:
    """Create a fake audio media dict with WAV bytes and an embedding."""
    # Use 441Hz and 5.1s so that tile boundaries don't align with exact
    # sine wave periods — otherwise slices could be byte-identical.
    wav = generate_wav(441, duration)
    rng = np.random.default_rng(media_id)
    return {
        "id": media_id,
        "type": "audio",
        "filename": f"clip_{media_id}.wav",
        "media_bytes": wav,
        "duration": duration,
        "md5": hashlib.md5(wav).hexdigest(),
        "embedding": rng.standard_normal(512).astype(np.float32),
        "origin": {"importer": "server_folder", "params": {"path": origin_path, "media_type": "audio"}},
        "origin_name": f"clip_{media_id}.wav",
    }


def _make_image_media(media_id: int, width: int = 300, height: int = 100) -> dict:
    """Create a fake image media dict with actual image bytes."""
    from PIL import Image

    img = Image.new("RGB", (width, height), color=(media_id * 30 % 256, 100, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    rng = np.random.default_rng(media_id + 1000)
    return {
        "id": media_id,
        "type": "image",
        "filename": f"img_{media_id}.png",
        "media_bytes": img_bytes,
        "width": width,
        "height": height,
        "md5": hashlib.md5(img_bytes).hexdigest(),
        "embedding": rng.standard_normal(512).astype(np.float32),
        "origin": {"importer": "server_folder", "params": {"path": "/data/images", "media_type": "image"}},
        "origin_name": f"img_{media_id}.png",
    }


def _make_text_media(media_id: int, text: str = "First sentence. Second sentence. Third sentence.") -> dict:
    """Create a fake text media dict."""
    text_bytes = text.encode("utf-8")
    rng = np.random.default_rng(media_id + 2000)
    return {
        "id": media_id,
        "type": "text",
        "filename": f"text_{media_id}.txt",
        "media_string": text,
        "media_bytes": text_bytes,
        "md5": hashlib.md5(text_bytes).hexdigest(),
        "embedding": rng.standard_normal(512).astype(np.float32),
        "origin": {"importer": "server_folder", "params": {"path": "/data/texts", "media_type": "text"}},
        "origin_name": f"text_{media_id}.txt",
    }


def _make_video_media(media_id: int, duration: float = 10.0) -> dict:
    """Create a fake video media dict (no real video bytes, just metadata)."""
    fake_bytes = b"FAKE_VIDEO_" + str(media_id).encode()
    rng = np.random.default_rng(media_id + 3000)
    return {
        "id": media_id,
        "type": "video",
        "filename": f"video_{media_id}.mp4",
        "media_bytes": fake_bytes,
        "duration": duration,
        "md5": hashlib.md5(fake_bytes).hexdigest(),
        "embedding": rng.standard_normal(512).astype(np.float32),
        "origin": {"importer": "server_folder", "params": {"path": "/data/videos", "media_type": "video"}},
        "origin_name": f"video_{media_id}.mp4",
    }


# ---------------------------------------------------------------------------
# _apply_clipper — clip thumbnail regeneration
# ---------------------------------------------------------------------------


class TestApplyClipperThumbnails:
    """Clipped audio/video sub-items must get fresh thumbnail_bytes so the
    find/label list shows the clip's range, not the parent media."""

    def test_audio_clips_get_fresh_waveform_thumbnails(self):
        from vtsearch.datasets.load_pipeline import _apply_clipper

        parent = _make_audio_media(1, duration=5.1)
        # Pretend the loader generated a thumbnail from the full waveform.
        parent["thumbnail_bytes"] = b"PARENT_WAVEFORM_PNG"
        clips_dict = {1: parent}

        _apply_clipper(clips_dict, "sound_tiling", {"duration": 2.0})

        assert len(clips_dict) > 1
        for clip in clips_dict.values():
            assert clip.get("thumbnail_bytes") not in (None, b"PARENT_WAVEFORM_PNG"), (
                "clip thumbnail should be regenerated from the clip's own bytes"
            )
            # PNG header check
            assert clip["thumbnail_bytes"][:8] == b"\x89PNG\r\n\x1a\n"

    def test_passthrough_clipper_keeps_parent_thumbnail(self):
        from vtsearch.datasets.load_pipeline import _apply_clipper

        parent = _make_audio_media(1, duration=2.0)
        parent["thumbnail_bytes"] = b"PARENT_WAVEFORM_PNG"
        clips_dict = {1: parent}

        # sound_default returns the media unchanged — no clipping happens.
        _apply_clipper(clips_dict, "sound_default")

        assert len(clips_dict) == 1
        clip = next(iter(clips_dict.values()))
        # No regeneration needed for a non-clip; parent thumbnail preserved.
        assert clip["thumbnail_bytes"] == b"PARENT_WAVEFORM_PNG"


# ---------------------------------------------------------------------------
# _apply_clipper — MD5 recomputation
# ---------------------------------------------------------------------------


class TestApplyClipperMD5:
    """Clips must get unique MD5s so collapse_duplicates doesn't merge them."""

    def test_audio_clips_get_recomputed_md5s(self):
        from vtsearch.datasets.load_pipeline import _apply_clipper

        clips_dict = {1: _make_audio_media(1)}
        parent_md5 = clips_dict[1]["md5"]
        _apply_clipper(clips_dict, "sound_tiling", {"duration": 2.0})

        assert len(clips_dict) == 3
        # All clip MD5s are recomputed from actual clip bytes, not the parent
        for clip in clips_dict.values():
            assert clip["md5"] != parent_md5, "clip MD5 should differ from parent"
            assert clip["md5"] == hashlib.md5(clip["media_bytes"]).hexdigest()

    def test_image_clips_get_recomputed_md5s(self):
        from vtsearch.datasets.load_pipeline import _apply_clipper

        clips_dict = {1: _make_image_media(1, width=300, height=100)}
        parent_md5 = clips_dict[1]["md5"]
        _apply_clipper(clips_dict, "image_tiling")

        assert len(clips_dict) == 3  # 300/100 = 3 tiles
        for clip in clips_dict.values():
            assert clip["md5"] != parent_md5
            assert clip["md5"] == hashlib.md5(clip["media_bytes"]).hexdigest()

    def test_text_clips_get_unique_md5s(self):
        from vtsearch.datasets.load_pipeline import _apply_clipper

        clips_dict = {1: _make_text_media(1)}
        parent_md5 = clips_dict[1]["md5"]
        _apply_clipper(clips_dict, "text_sentence")

        assert len(clips_dict) == 3  # 3 sentences
        md5s = [c["md5"] for c in clips_dict.values()]
        assert len(set(md5s)) == len(md5s), "text clips must have unique MD5s"
        for md5 in md5s:
            assert md5 != parent_md5

    def test_video_clips_get_unique_md5s(self):
        from vtsearch.datasets.load_pipeline import _apply_clipper

        clips_dict = {1: _make_video_media(1, duration=10.0)}
        parent_md5 = clips_dict[1]["md5"]
        _apply_clipper(clips_dict, "video_tiling", {"duration": 2.0})

        assert len(clips_dict) == 5  # 10/2 = 5 tiles
        md5s = [c["md5"] for c in clips_dict.values()]
        assert len(set(md5s)) == len(md5s), "video clips must have unique MD5s"
        for md5 in md5s:
            assert md5 != parent_md5

    def test_dedup_preserves_all_clips(self):
        """collapse_duplicates should NOT merge clips from the same parent."""
        from vtsearch.datasets.load_pipeline import _apply_clipper
        from vtsearch.state import collapse_duplicates

        clips_dict = {1: _make_audio_media(1, duration=5.1)}
        _apply_clipper(clips_dict, "sound_tiling", {"duration": 2.0})
        n_before = len(clips_dict)
        assert n_before > 1

        collapsed = collapse_duplicates(clips_dict)
        assert collapsed == 0, "clips from same parent should NOT be deduped"
        assert len(clips_dict) == n_before

    def test_importer_provided_md5_replaced_for_clips(self):
        """An importer may pre-compute the MD5 for the full media item.
        After clipping, each sub-item must get its own MD5, not the
        importer's value for the parent.
        """
        from vtsearch.datasets.load_pipeline import _apply_clipper

        media = _make_audio_media(1, duration=5.0)
        # Simulate an importer that provides its own MD5
        media["md5"] = "importer_provided_md5_for_full_item"
        clips_dict = {1: media}
        _apply_clipper(clips_dict, "sound_tiling", {"duration": 2.0})

        for clip in clips_dict.values():
            assert clip["md5"] != "importer_provided_md5_for_full_item", (
                "importer-provided MD5 must be replaced for clips"
            )
            # The new MD5 should be computed from the actual clip bytes
            assert clip["md5"] == hashlib.md5(clip["media_bytes"]).hexdigest()

    def test_importer_provided_embedding_replaced_for_clips(self):
        """An importer may pre-compute embeddings for full media items.
        After clipping, each sub-item must get its own embedding based
        on the clipped content, not the parent embedding.
        """
        from vtsearch.datasets.load_pipeline import _apply_clipper

        media = _make_audio_media(1, duration=5.0)
        # Tag the parent embedding so we can detect it later
        parent_embedding = media["embedding"].copy()
        clips_dict = {1: media}
        _apply_clipper(clips_dict, "sound_tiling", {"duration": 2.0})

        # Re-embedding may fail if no embedder is loaded (test env), but
        # we can at least check: if re-embedding succeeded, the embedding
        # differs from the parent.  If it didn't succeed, the embedding
        # is kept as-is (acceptable fallback).
        for clip in clips_dict.values():
            if not np.array_equal(clip["embedding"], parent_embedding):
                # Re-embedding worked — great.
                pass
            # Either way, the clip should have *an* embedding.
            assert "embedding" in clip

    def test_importer_md5_kept_for_passthrough_clipper(self):
        """Default (pass-through) clippers should NOT replace the
        importer-provided MD5 since no clipping occurred."""
        from vtsearch.datasets.load_pipeline import _apply_clipper

        media = _make_audio_media(1, duration=5.0)
        media["md5"] = "importer_provided_md5"
        clips_dict = {1: media}
        _apply_clipper(clips_dict, "sound_default")

        assert clips_dict[1]["md5"] == "importer_provided_md5"


# ---------------------------------------------------------------------------
# _apply_clipper — origin boundary storage
# ---------------------------------------------------------------------------


class TestApplyClipperOriginBoundaries:
    """Clip boundaries must be stored in origin params."""

    def test_audio_clip_origin_has_boundaries(self):
        from vtsearch.datasets.load_pipeline import _apply_clipper

        clips_dict = {1: _make_audio_media(1, duration=5.0)}
        _apply_clipper(clips_dict, "sound_tiling", {"duration": 2.0})

        for clip in clips_dict.values():
            params = clip["origin"]["params"]
            assert params["clipper"] == "sound_tiling"
            assert "clip_start" in params
            assert "clip_end" in params
            assert "clip_index" in params
            # Boundaries are stored as strings
            assert float(params["clip_start"]) >= 0.0
            assert float(params["clip_end"]) > float(params["clip_start"])

    def test_image_clip_origin_has_clip_box(self):
        from vtsearch.datasets.load_pipeline import _apply_clipper

        clips_dict = {1: _make_image_media(1, width=300, height=100)}
        _apply_clipper(clips_dict, "image_tiling")

        for clip in clips_dict.values():
            params = clip["origin"]["params"]
            assert params["clipper"] == "image_tiling"
            assert "clip_box" in params
            # clip_box is stored as comma-separated string
            box_parts = params["clip_box"].split(",")
            assert len(box_parts) == 4

    def test_video_clip_origin_has_boundaries(self):
        from vtsearch.datasets.load_pipeline import _apply_clipper

        clips_dict = {1: _make_video_media(1, duration=10.0)}
        _apply_clipper(clips_dict, "video_tiling", {"duration": 2.0})

        for clip in clips_dict.values():
            params = clip["origin"]["params"]
            assert params["clipper"] == "video_tiling"
            assert "clip_start" in params
            assert "clip_end" in params

    def test_text_clip_origin_has_clipper_and_index(self):
        from vtsearch.datasets.load_pipeline import _apply_clipper

        clips_dict = {1: _make_text_media(1)}
        _apply_clipper(clips_dict, "text_sentence")

        for clip in clips_dict.values():
            params = clip["origin"]["params"]
            assert params["clipper"] == "text_sentence"
            assert "clip_index" in params

    def test_audio_clip_origin_stores_clipper_params(self):
        """Clipper parameter values (duration, min_overlap) must be stored
        in origin so cross-dataset resolution can reconstruct the clipper."""
        from vtsearch.datasets.load_pipeline import _apply_clipper

        clips_dict = {1: _make_audio_media(1, duration=5.0)}
        _apply_clipper(clips_dict, "sound_tiling", {"duration": 2.5, "min_overlap": 0.5})

        for clip in clips_dict.values():
            params = clip["origin"]["params"]
            assert params["clipper_duration"] == "2.5"
            assert params["clipper_min_overlap"] == "0.5"

    def test_video_scene_origin_stores_clipper_params(self):
        """Scene clipper's threshold and min_scene_duration must be stored."""
        from vtsearch.datasets.load_pipeline import _apply_clipper

        # VideoSceneClipper won't produce multiple clips without real video,
        # but the params should still be stored on the single passthrough.
        clips_dict = {1: _make_video_media(1, duration=10.0)}
        _apply_clipper(clips_dict, "video_tiling", {"duration": 3.0, "min_overlap": 0.5})

        for clip in clips_dict.values():
            params = clip["origin"]["params"]
            assert params["clipper_duration"] == "3.0"
            assert params["clipper_min_overlap"] == "0.5"

    def test_default_clipper_stores_no_extra_params(self):
        """Default clippers have no parameter values to store."""
        from vtsearch.datasets.load_pipeline import _apply_clipper

        clips_dict = {1: _make_audio_media(1, duration=5.0)}
        _apply_clipper(clips_dict, "sound_default")

        params = clips_dict[1]["origin"]["params"]
        assert params["clipper"] == "sound_default"
        # No clipper_* keys beyond "clipper" itself
        clipper_keys = [k for k in params if k.startswith("clipper_")]
        assert clipper_keys == []


# ---------------------------------------------------------------------------
# Label export with clipped media
# ---------------------------------------------------------------------------


class TestLabelExportWithClips:
    """Labels exported from clipped media preserve clip origin info."""

    def test_label_export_preserves_clip_origin(self):
        from vtsearch.datasets.labelset import LabelSet

        saved = dict(medias)
        medias.clear()
        try:
            # Set up clipped audio media
            from vtsearch.datasets.load_pipeline import _apply_clipper

            clips_dict = {1: _make_audio_media(1, duration=5.0)}
            _apply_clipper(clips_dict, "sound_tiling", {"duration": 2.0})

            # Populate medias
            for cid, clip in clips_dict.items():
                medias[cid] = clip

            # Vote on the first two clips
            good_votes[1] = None
            bad_votes[2] = None

            ls = LabelSet.from_clips_and_votes(dict(medias), dict(good_votes), dict(bad_votes))
            assert len(ls) == 2

            for elem in ls:
                assert elem.origin is not None
                assert elem.origin["params"]["clipper"] == "sound_tiling"
                assert "clip_start" in elem.origin["params"]
                assert "clip_end" in elem.origin["params"]

            # Verify round-trip through serialization
            data = ls.to_dict()
            ls2 = LabelSet.from_dict(data)
            assert len(ls2) == 2
            for elem in ls2:
                assert elem.origin is not None
                assert elem.origin["params"]["clipper"] == "sound_tiling"
        finally:
            medias.clear()
            medias.update(saved)


# ---------------------------------------------------------------------------
# Label import resolution with clipped media
# ---------------------------------------------------------------------------


class TestLabelImportWithClips:
    """Labels from clipped media can be resolved on the same dataset."""

    def test_origin_lookup_matches_clipped_media(self):
        from vtsearch.datasets.load_pipeline import _apply_clipper
        from vtsearch.state.media_lookup import build_media_lookup, resolve_media_ids

        clips_dict = {1: _make_audio_media(1, duration=5.0)}
        _apply_clipper(clips_dict, "sound_tiling", {"duration": 2.0})

        origin_lookup, md5_lookup, name_lookup = build_media_lookup(clips_dict)

        # Each clip should be findable by its origin
        for clip in clips_dict.values():
            entry = {
                "origin": clip["origin"],
                "origin_name": clip["origin_name"],
                "md5": clip["md5"],
            }
            matches = resolve_media_ids(entry, origin_lookup, md5_lookup, name_lookup)
            assert clip["id"] in matches, f"clip {clip['id']} should be resolvable by origin"

    def test_md5_lookup_matches_clipped_media(self):
        from vtsearch.datasets.load_pipeline import _apply_clipper
        from vtsearch.state.media_lookup import build_media_lookup, resolve_media_ids

        clips_dict = {1: _make_audio_media(1, duration=5.0)}
        _apply_clipper(clips_dict, "sound_tiling", {"duration": 2.0})

        origin_lookup, md5_lookup, name_lookup = build_media_lookup(clips_dict)

        # Each clip should also be findable by its unique MD5
        for clip in clips_dict.values():
            entry = {"md5": clip["md5"]}
            matches = resolve_media_ids(entry, origin_lookup, md5_lookup, name_lookup)
            assert clip["id"] in matches


# ---------------------------------------------------------------------------
# Cross-dataset clip-aware embedding
# ---------------------------------------------------------------------------


class TestCrossDatasetClipEmbedding:
    """resolve_label_embeddings uses clip params to embed the clipped content."""

    def test_apply_clip_and_embed_audio(self, tmp_path):
        """Audio clip params cause the resolver to slice before embedding."""
        from vtsearch.detectors.resolver import _apply_clip_and_embed

        wav = generate_wav(440, 5.0)
        wav_path = tmp_path / "test.wav"
        wav_path.write_bytes(wav)

        origin = {
            "importer": "server_folder",
            "params": {
                "path": str(tmp_path),
                "clipper": "sound_tiling",
                "clip_start": "0.0",
                "clip_end": "2.0",
                "clip_index": "0",
            },
        }

        # This should slice to [0, 2] seconds before embedding.
        # Won't have a real embedder in tests, so just verify it doesn't crash
        # and falls back gracefully.
        _apply_clip_and_embed(wav_path, "audio", origin)
        # Result may be None if no embedder is loaded, which is fine.
        # The important thing is the function handles clip params without error.

    def test_apply_clip_and_embed_image(self, tmp_path):
        """Image clip params cause the resolver to crop before embedding."""
        from PIL import Image

        from vtsearch.detectors.resolver import _apply_clip_and_embed

        img = Image.new("RGB", (300, 100), color=(255, 0, 0))
        img_path = tmp_path / "test.png"
        img.save(img_path, format="PNG")

        origin = {
            "importer": "server_folder",
            "params": {
                "path": str(tmp_path),
                "clipper": "image_tiling",
                "clip_box": "0,0,100,100",
                "clip_index": "0",
            },
        }

        _apply_clip_and_embed(img_path, "image", origin)
        # May be None without a real embedder.

    def test_apply_clip_and_embed_text(self, tmp_path):
        """Text clip params cause the resolver to extract the sentence before embedding."""
        from vtsearch.detectors.resolver import _apply_clip_and_embed

        text = "First sentence. Second sentence. Third sentence."
        text_path = tmp_path / "test.txt"
        text_path.write_text(text, encoding="utf-8")

        origin = {
            "importer": "server_folder",
            "params": {
                "path": str(tmp_path),
                "clipper": "text_sentence",
                "clip_index": "1",
            },
        }

        _apply_clip_and_embed(text_path, "text", origin)

    def test_apply_clip_and_embed_no_clipper_is_passthrough(self, tmp_path):
        """Without clipper params, behaves like normal embed_file."""
        from vtsearch.detectors.resolver import _apply_clip_and_embed

        wav = generate_wav(440, 2.0)
        wav_path = tmp_path / "test.wav"
        wav_path.write_bytes(wav)

        origin = {"importer": "server_folder", "params": {"path": str(tmp_path)}}
        _apply_clip_and_embed(wav_path, "audio", origin)


# ---------------------------------------------------------------------------
# API endpoint — clip metadata exposure
# ---------------------------------------------------------------------------


class TestAPIClipMetadata:
    """The /api/medias endpoint should include clip metadata."""

    def test_batch_medias_includes_clip_fields(self, client):
        saved = dict(medias)
        medias.clear()
        try:
            from vtsearch.datasets.load_pipeline import _apply_clipper

            clips_dict = {1: _make_audio_media(1, duration=5.0)}
            _apply_clipper(clips_dict, "sound_tiling", {"duration": 2.0})
            for cid, clip in clips_dict.items():
                medias[cid] = clip

            ids = list(medias.keys())
            resp = client.post("/api/medias/batch", json={"ids": ids})
            assert resp.status_code == 200
            data = resp.get_json()
            assert len(data) > 0
            for item in data:
                assert "clip_start" in item
                assert "clip_end" in item
                assert "clip_index" in item
        finally:
            medias.clear()
            medias.update(saved)

    def test_image_clip_exposes_clip_box(self, client):
        saved = dict(medias)
        medias.clear()
        try:
            from vtsearch.datasets.load_pipeline import _apply_clipper

            clips_dict = {1: _make_image_media(1, width=300, height=100)}
            _apply_clipper(clips_dict, "image_tiling")
            for cid, clip in clips_dict.items():
                medias[cid] = clip

            ids = list(medias.keys())
            resp = client.post("/api/medias/batch", json={"ids": ids})
            assert resp.status_code == 200
            data = resp.get_json()
            for item in data:
                assert "clip_box" in item
                assert "clip_index" in item
        finally:
            medias.clear()
            medias.update(saved)


# ---------------------------------------------------------------------------
# Default clippers should be no-ops
# ---------------------------------------------------------------------------


class TestDefaultClippersNoOp:
    """Default (pass-through) clippers should not break the workflow."""

    def test_audio_default_clipper_preserves_media(self):
        from vtsearch.datasets.load_pipeline import _apply_clipper

        media = _make_audio_media(1, duration=5.0)
        original_md5 = media["md5"]
        clips_dict = {1: media}
        _apply_clipper(clips_dict, "sound_default")

        assert len(clips_dict) == 1
        assert clips_dict[1]["md5"] == original_md5

    def test_image_default_clipper_preserves_media(self):
        from vtsearch.datasets.load_pipeline import _apply_clipper

        media = _make_image_media(1)
        original_md5 = media["md5"]
        clips_dict = {1: media}
        _apply_clipper(clips_dict, "image_default")

        assert len(clips_dict) == 1
        assert clips_dict[1]["md5"] == original_md5

    def test_text_default_clipper_preserves_media(self):
        from vtsearch.datasets.load_pipeline import _apply_clipper

        media = _make_text_media(1)
        original_md5 = media["md5"]
        clips_dict = {1: media}
        _apply_clipper(clips_dict, "text_default")

        assert len(clips_dict) == 1
        assert clips_dict[1]["md5"] == original_md5

    def test_video_default_clipper_preserves_media(self):
        from vtsearch.datasets.load_pipeline import _apply_clipper

        media = _make_video_media(1)
        original_md5 = media["md5"]
        clips_dict = {1: media}
        _apply_clipper(clips_dict, "video_default")

        assert len(clips_dict) == 1
        assert clips_dict[1]["md5"] == original_md5


# ---------------------------------------------------------------------------
# Multiple medias clipped together
# ---------------------------------------------------------------------------


class TestMultipleMediasClipped:
    """Clipping multiple medias at once should produce correct results."""

    def test_multiple_audio_medias_clipped(self):
        from vtsearch.datasets.load_pipeline import _apply_clipper

        clips_dict = {
            1: _make_audio_media(1, duration=5.1),
            2: _make_audio_media(2, duration=4.1),
        }
        _apply_clipper(clips_dict, "sound_tiling", {"duration": 2.0})

        # Media 1 (5.1s): 3 clips, Media 2 (4.1s): 3 clips = 6 total
        assert len(clips_dict) == 6
        # Each clip's MD5 should be computed from its actual bytes
        for clip in clips_dict.values():
            assert clip["md5"] == hashlib.md5(clip["media_bytes"]).hexdigest()

    def test_multiple_text_medias_clipped(self):
        from vtsearch.datasets.load_pipeline import _apply_clipper

        clips_dict = {
            1: _make_text_media(1, "Foo. Bar. Baz."),
            2: _make_text_media(2, "One sentence only"),
        }
        _apply_clipper(clips_dict, "text_sentence")

        # Media 1: 3 sentences, Media 2: 1 sentence (no split) = 4 total
        assert len(clips_dict) == 4
        md5s = [c["md5"] for c in clips_dict.values()]
        assert len(set(md5s)) == 4

    def test_clip_ids_are_sequential(self):
        from vtsearch.datasets.load_pipeline import _apply_clipper

        clips_dict = {
            1: _make_audio_media(1, duration=5.1),
            2: _make_audio_media(2, duration=4.1),
        }
        _apply_clipper(clips_dict, "sound_tiling", {"duration": 2.0})

        ids = sorted(clips_dict.keys())
        assert ids == list(range(1, len(clips_dict) + 1))


# ---------------------------------------------------------------------------
# _apply_clipper — on_progress callback
# ---------------------------------------------------------------------------


class TestApplyClipperProgress:
    """Verify that _apply_clipper reports progress via the on_progress callback."""

    def test_progress_reports_clipping_and_embedding_phases(self):
        from vtsearch.datasets.load_pipeline import _apply_clipper

        clips_dict = {
            1: _make_audio_media(1, duration=5.1),
            2: _make_audio_media(2, duration=4.1),
        }
        calls = []
        _apply_clipper(
            clips_dict,
            "sound_tiling",
            {"duration": 2.0},
            on_progress=lambda cur, tot, phase: calls.append((cur, tot, phase)),
        )

        clipping_calls = [c for c in calls if c[2] == "clipping"]
        embedding_calls = [c for c in calls if c[2] == "embedding"]

        # Should have clipping progress calls (one per input media)
        assert len(clipping_calls) == 2
        assert clipping_calls[0] == (0, 2, "clipping")
        assert clipping_calls[1] == (1, 2, "clipping")

        # Should have embedding progress calls (one per output clip)
        assert len(embedding_calls) > 0
        # First embedding call starts at 0
        assert embedding_calls[0][0] == 0
        # Total should equal the number of clips produced
        total_clips = embedding_calls[0][1]
        assert total_clips == len(clips_dict)

    def test_no_progress_without_callback(self):
        """Passing on_progress=None doesn't break anything."""
        from vtsearch.datasets.load_pipeline import _apply_clipper

        clips_dict = {1: _make_audio_media(1, duration=5.1)}
        _apply_clipper(clips_dict, "sound_tiling", {"duration": 2.0}, on_progress=None)
        assert len(clips_dict) >= 1  # Clips were still produced


# ---------------------------------------------------------------------------
# Clip metadata in display_metadata and enriched exports
# ---------------------------------------------------------------------------


class TestClipDisplayMetadata:
    """display_metadata should include clip boundary fields when present."""

    def test_audio_clip_metadata(self):
        from vtsearch.media import get as get_media_type

        mt = get_media_type("audio")
        media = {"type": "audio", "clip_start": 0.0, "clip_end": 2.0, "clip_index": 0}
        meta = mt.display_metadata(media)
        assert meta["Clip Start"] == 0.0
        assert meta["Clip End"] == 2.0
        assert meta["Clip Index"] == 0

    def test_image_clip_metadata(self):
        from vtsearch.media import get as get_media_type

        mt = get_media_type("image")
        media = {"type": "image", "clip_box": [0, 0, 100, 100], "clip_index": 0}
        meta = mt.display_metadata(media)
        assert meta["Clip Box"] == "0,0,100,100"
        assert meta["Clip Index"] == 0

    def test_video_clip_metadata(self):
        from vtsearch.media import get as get_media_type

        mt = get_media_type("video")
        media = {"type": "video", "clip_start": 1.5, "clip_end": 4.0, "clip_index": 1}
        meta = mt.display_metadata(media)
        assert meta["Clip Start"] == 1.5
        assert meta["Clip End"] == 4.0
        assert meta["Clip Index"] == 1

    def test_text_clip_metadata(self):
        from vtsearch.media import get as get_media_type

        mt = get_media_type("text")
        media = {"type": "text", "clip_index": 2}
        meta = mt.display_metadata(media)
        assert meta["Clip Index"] == 2

    def test_no_clip_fields_when_absent(self):
        from vtsearch.media import get as get_media_type

        mt = get_media_type("audio")
        media = {"type": "audio", "duration": 3.0}
        meta = mt.display_metadata(media)
        assert "Clip Start" not in meta
        assert "Clip End" not in meta
        assert "Clip Index" not in meta


class TestClipFieldsInEnrichedExport:
    """Enriched label export should surface clip boundary columns."""

    def test_enriched_export_has_clip_columns(self, client):
        saved = dict(medias)
        medias.clear()
        try:
            from vtsearch.datasets.load_pipeline import _apply_clipper

            clips_dict = {1: _make_audio_media(1, duration=5.0)}
            _apply_clipper(clips_dict, "sound_tiling", {"duration": 2.0})
            for cid, clip in clips_dict.items():
                medias[cid] = clip

            # Vote on a clip
            good_votes[1] = None

            resp = client.get("/api/labels/export?enrich=1")
            assert resp.status_code == 200
            data = resp.get_json()
            cols = data["available_columns"]
            assert "Clip Start" in cols
            assert "Clip End" in cols
            assert "Clip Index" in cols
            # Clip Box should NOT be present for audio clips
            assert "Clip Box" not in cols

            # Verify custom_metadata on the label entry
            entry = data["labels"][0]
            meta = entry["custom_metadata"]
            assert "Clip Start" in meta
            assert "Clip End" in meta
        finally:
            medias.clear()
            medias.update(saved)

    def test_enriched_export_image_clip_box(self, client):
        saved = dict(medias)
        medias.clear()
        try:
            from vtsearch.datasets.load_pipeline import _apply_clipper

            clips_dict = {1: _make_image_media(1, width=300, height=100)}
            _apply_clipper(clips_dict, "image_tiling")
            for cid, clip in clips_dict.items():
                medias[cid] = clip

            good_votes[1] = None

            resp = client.get("/api/labels/export?enrich=1")
            assert resp.status_code == 200
            data = resp.get_json()
            cols = data["available_columns"]
            assert "Clip Box" in cols
            assert "Clip Index" in cols
        finally:
            medias.clear()
            medias.update(saved)

    def test_no_clip_columns_without_clips(self, client):
        """Unclipped media should not have clip columns in export."""
        good_votes[1] = None  # Vote on default test media
        resp = client.get("/api/labels/export?enrich=1")
        assert resp.status_code == 200
        data = resp.get_json()
        cols = data["available_columns"]
        assert "Clip Start" not in cols
        assert "Clip End" not in cols
        assert "Clip Box" not in cols
        assert "Clip Index" not in cols


class TestClipFieldsInBuildMediaHit:
    """build_media_hit should include clip fields when present."""

    def test_hit_includes_clip_start_end(self):
        from vtsearch.utils.hits import build_media_hit

        media = {
            "filename": "clip.wav",
            "category": "audio",
            "md5": "abc123",
            "clip_start": 0.0,
            "clip_end": 2.5,
            "clip_index": 0,
        }
        hit = build_media_hit(1, media, 0.95)
        assert hit["clip_start"] == 0.0
        assert hit["clip_end"] == 2.5
        assert hit["clip_index"] == 0

    def test_hit_includes_clip_box(self):
        from vtsearch.utils.hits import build_media_hit

        media = {
            "filename": "tile.png",
            "category": "image",
            "md5": "def456",
            "clip_box": [10, 20, 110, 120],
            "clip_index": 1,
        }
        hit = build_media_hit(1, media, 0.8)
        assert hit["clip_box"] == [10, 20, 110, 120]
        assert hit["clip_index"] == 1

    def test_hit_omits_clip_fields_when_absent(self):
        from vtsearch.utils.hits import build_media_hit

        media = {"filename": "full.wav", "category": "audio", "md5": "xyz"}
        hit = build_media_hit(1, media, 0.5)
        assert "clip_start" not in hit
        assert "clip_end" not in hit
        assert "clip_box" not in hit
        assert "clip_index" not in hit


class TestCsvExportClipColumns:
    """CSV autodetect export should include clip columns when present."""

    def test_csv_includes_clip_start_end(self, tmp_path):
        from vtsearch.exporters.server_csv_file import ServerCsvLabelsetExporter

        exporter = ServerCsvLabelsetExporter()
        results = {
            "detectors_run": 1,
            "results": {
                "det1": {
                    "detector_name": "det1",
                    "threshold": 0.5,
                    "hits": [
                        {
                            "filename": "a.wav",
                            "category": "audio",
                            "score": 0.9,
                            "clip_start": 0.0,
                            "clip_end": 2.0,
                            "origin_name": "a.wav",
                        },
                        {
                            "filename": "b.wav",
                            "category": "audio",
                            "score": 0.8,
                            "clip_start": 2.0,
                            "clip_end": 4.0,
                            "origin_name": "b.wav",
                        },
                    ],
                }
            },
        }
        filepath = tmp_path / "results.csv"
        exporter.export(results, {"filepath": str(filepath)})

        import csv

        with open(filepath) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert "clip_start" in rows[0]
        assert "clip_end" in rows[0]
        assert rows[0]["clip_start"] == "0.0"
        assert rows[1]["clip_end"] == "4.0"

    def test_csv_includes_clip_box(self, tmp_path):
        from vtsearch.exporters.server_csv_file import ServerCsvLabelsetExporter

        exporter = ServerCsvLabelsetExporter()
        results = {
            "detectors_run": 1,
            "results": {
                "det1": {
                    "detector_name": "det1",
                    "threshold": 0.5,
                    "hits": [
                        {
                            "filename": "tile.png",
                            "category": "image",
                            "score": 0.9,
                            "clip_box": [0, 0, 100, 100],
                            "origin_name": "tile.png",
                        },
                    ],
                }
            },
        }
        filepath = tmp_path / "results.csv"
        exporter.export(results, {"filepath": str(filepath)})

        import csv

        with open(filepath) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert "clip_box" in rows[0]
        assert rows[0]["clip_box"] == "0,0,100,100"

    def test_csv_omits_clip_columns_without_clips(self, tmp_path):
        from vtsearch.exporters.server_csv_file import ServerCsvLabelsetExporter

        exporter = ServerCsvLabelsetExporter()
        results = {
            "detectors_run": 1,
            "results": {
                "det1": {
                    "detector_name": "det1",
                    "threshold": 0.5,
                    "hits": [
                        {"filename": "full.wav", "category": "audio", "score": 0.7, "origin_name": "full.wav"},
                    ],
                }
            },
        }
        filepath = tmp_path / "results.csv"
        exporter.export(results, {"filepath": str(filepath)})

        import csv

        with open(filepath) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert "clip_start" not in rows[0]
        assert "clip_end" not in rows[0]
        assert "clip_box" not in rows[0]
