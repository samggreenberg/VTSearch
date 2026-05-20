"""Tests for label import ingestion of missing media elements.

Covers:
- POST /api/label-importers/ingest-missing endpoint
- ingest_missing_medias unit tests (_group_by_origin, _media_type_from_origin, _ingest_via_resolver)
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# API – POST /api/label-importers/ingest-missing
# ---------------------------------------------------------------------------


class TestIngestMissingEndpoint:
    def test_empty_entries_returns_422(self, client):
        # Schema-level validation (entries Length >= 1) → 422.
        res = client.post(
            "/api/label-importers/ingest-missing",
            json={"entries": []},
        )
        assert res.status_code == 422

    def test_missing_entries_key_returns_422(self, client):
        # Schema-level validation (required ``entries``) → 422.
        res = client.post(
            "/api/label-importers/ingest-missing",
            json={},
        )
        assert res.status_code == 422

    def test_ingest_with_unknown_origin_returns_zero(self, client):
        """Entries whose origin importer doesn't exist are gracefully skipped."""
        entries = [
            {
                "md5": "fake_md5",
                "label": "good",
                "origin": {"importer": "nonexistent_importer", "params": {}},
                "origin_name": "file.wav",
            }
        ]
        res = client.post(
            "/api/label-importers/ingest-missing",
            json={"entries": entries},
        )
        assert res.status_code == 200
        result = res.get_json()
        assert result["ingested"] == 0

    def test_ingest_with_no_origin_returns_zero(self, client):
        """Entries without origin cannot be ingested."""
        entries = [{"md5": "fake_md5", "label": "good"}]
        res = client.post(
            "/api/label-importers/ingest-missing",
            json={"entries": entries},
        )
        assert res.status_code == 200
        result = res.get_json()
        assert result["ingested"] == 0


# ---------------------------------------------------------------------------
# ingest_missing_medias (unit tests)
# ---------------------------------------------------------------------------


class TestIngestMissingClips:
    def test_groups_by_origin(self):
        from vtscore.datasets.ingest import _group_by_origin

        entries = [
            {"origin": {"importer": "a", "params": {}}, "origin_name": "x", "md5": "1", "label": "good"},
            {"origin": {"importer": "a", "params": {}}, "origin_name": "y", "md5": "2", "label": "bad"},
            {"origin": {"importer": "b", "params": {}}, "origin_name": "z", "md5": "3", "label": "good"},
        ]
        groups = _group_by_origin(entries)
        assert len(groups) == 2
        # Each group should have the correct number of entries
        counts = sorted(len(es) for _, es in groups.values())
        assert counts == [1, 2]

    def test_entries_without_origin_skipped(self):
        from vtscore.datasets.ingest import _group_by_origin

        entries = [{"md5": "abc", "label": "good"}]
        groups = _group_by_origin(entries)
        assert len(groups) == 0

    def test_media_type_from_origin_folder(self):
        """Folder origins resolve media type from params."""
        from vtscore.datasets.ingest import _media_type_from_origin

        origin = {"importer": "server_folder", "params": {"path": "/tmp/x", "media_type": "text"}}
        assert _media_type_from_origin(origin) == "text"

    def test_media_type_from_origin_demo(self):
        """Demo origins resolve media type from DEMO_DATASETS config."""
        from vtscore.datasets.config import DEMO_DATASETS
        from vtscore.datasets.ingest import _media_type_from_origin

        # Find a real demo dataset name from the config
        for name, info in DEMO_DATASETS.items():
            expected = info.get("media_type", "")
            if expected:
                origin = {"importer": "demo", "params": {"name": name}}
                assert _media_type_from_origin(origin) == expected
                break

    def test_media_type_from_origin_unknown(self):
        """Unknown origins return empty string."""
        from vtscore.datasets.ingest import _media_type_from_origin

        origin = {"importer": "unknown_xyz", "params": {}}
        assert _media_type_from_origin(origin) == ""

    def test_ingest_via_resolver_with_demo_origin(self, tmp_path):
        """_ingest_via_resolver resolves demo origin files item-by-item."""
        import numpy as np

        from vtscore.datasets.ingest import _ingest_via_resolver

        rng = np.random.default_rng(42)

        # Create a fake image file
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        cat_dir = img_dir / "cat"
        cat_dir.mkdir()
        # Create a minimal valid JPEG (just enough bytes for PIL to open)
        from PIL import Image

        img = Image.new("RGB", (32, 32), color=(255, 0, 0))
        img_path = cat_dir / "test_001.jpg"
        img.save(img_path, format="JPEG")

        origin = {"importer": "demo", "params": {"name": "test_demo_dataset"}}
        entries = [
            {
                "origin": origin,
                "origin_name": "cat/test_001.jpg",
                "md5": "some_md5",
                "label": "good",
            }
        ]

        existing: dict = {}

        def noop_progress(status, message, current, total):
            pass

        fake_embedding = rng.standard_normal(512).astype(np.float32)

        # Patch resolve_file_context to yield our test file, embed_file to
        # return a fake embedding, and _media_type_from_origin to return
        # "image".  The resolver imports are lazy (inside the function), so
        # patch at the source module.
        from contextlib import contextmanager
        from unittest.mock import patch

        @contextmanager
        def _fake_resolve_ctx(*_args, **_kwargs):
            yield img_path

        with (
            patch("vtscore.datasets.ingest._media_type_from_origin", return_value="image"),
            patch("vtscore.detectors.resolver.resolve_file_context", _fake_resolve_ctx),
            patch("vtscore.detectors.resolver.embed_file", return_value=fake_embedding),
        ):
            result = _ingest_via_resolver(origin, entries, existing, noop_progress)

        assert result == 1
        assert 1 in existing
        assert existing[1]["origin_name"] == "cat/test_001.jpg"
        assert existing[1]["origin"] == origin
        assert existing[1]["embedding"] is fake_embedding

    def test_ingest_via_resolver_returns_neg1_for_unknown_media_type(self):
        """_ingest_via_resolver returns -1 when media type can't be determined."""
        from vtscore.datasets.ingest import _ingest_via_resolver

        origin = {"importer": "unknown_xyz", "params": {}}
        entries = [{"origin": origin, "origin_name": "x", "md5": "m", "label": "good"}]

        def noop_progress(status, message, current, total):
            pass

        result = _ingest_via_resolver(origin, entries, {}, noop_progress)
        assert result == -1

    def test_ingest_with_folder_importer(self, tmp_path):
        """Ingest missing medias from a real folder origin."""
        import hashlib

        import numpy as np

        from vtscore.datasets.ingest import ingest_missing_medias

        # Create a folder with a text file to simulate a media source
        text_dir = tmp_path / "texts"
        text_dir.mkdir()
        (text_dir / "hello.txt").write_text("Hello world, this is a test paragraph for embedding.")
        (text_dir / "goodbye.txt").write_text("Goodbye world, this is another test paragraph.")

        origin = {"importer": "server_folder", "params": {"path": str(text_dir), "media_type": "text"}}

        # Start with an existing medias dict
        existing_clips: dict = {
            1: {
                "id": 1,
                "type": "text",
                "duration": 0,
                "file_size": 10,
                "md5": "existing_md5",
                "embedding": np.zeros(768),
                "media_bytes": None,
                "media_string": "existing",
                "filename": "existing.txt",
                "category": "test",
                "origin": None,
                "origin_name": "existing.txt",
            }
        }

        missing_entries = [
            {
                "md5": hashlib.md5(b"Hello world, this is a test paragraph for embedding.").hexdigest(),
                "label": "good",
                "origin": origin,
                "origin_name": "hello.txt",
            },
        ]

        def noop_progress(status, message, current, total):
            pass

        ingested = ingest_missing_medias(missing_entries, existing_clips, on_progress=noop_progress)
        assert ingested == 1
        # New media should have id=2 (next after existing)
        assert 2 in existing_clips
        assert existing_clips[2]["origin_name"] == "hello.txt"
        assert existing_clips[2]["embedding"] is not None


# ---------------------------------------------------------------------------
# Clipped-origin re-ingest (H10): MD5 / media_bytes / file_size must
# describe the clip, not the parent — otherwise distinct clips of the same
# parent collide on collapse_duplicates after a save+reload round-trip.
# ---------------------------------------------------------------------------


class TestClippedReingest:
    def test_audio_clip_md5_is_clip_not_parent(self, tmp_path):
        """A re-ingested audio clip's MD5 hashes the *clip* bytes, not the parent file."""
        import hashlib

        import numpy as np

        from vtscore.datasets.ingest import _ingest_via_resolver
        from vtscore.media.audio.audio_generator import generate_wav
        from vtscore.media.audio.clipper import _wav_slice

        rng = np.random.default_rng(42)

        wav = generate_wav(440, 5.0)
        wav_path = tmp_path / "test.wav"
        wav_path.write_bytes(wav)

        clip_start, clip_end = 0.0, 2.0
        clip_bytes = _wav_slice(wav, clip_start, clip_end)
        expected_clip_md5 = hashlib.md5(clip_bytes).hexdigest()
        parent_md5 = hashlib.md5(wav).hexdigest()
        assert expected_clip_md5 != parent_md5, "clip and parent MD5s must differ"

        origin = {
            "importer": "server_folder",
            "params": {
                "path": str(tmp_path),
                "media_type": "audio",
                "clipper": "sound_tiling",
                "clip_start": str(clip_start),
                "clip_end": str(clip_end),
                "clip_index": "0",
            },
        }
        entries = [{"origin": origin, "origin_name": "test.wav", "md5": expected_clip_md5, "label": "good"}]

        fake_embedding = rng.standard_normal(512).astype(np.float32)

        from contextlib import contextmanager
        from unittest.mock import patch

        @contextmanager
        def _fake_resolve_ctx(*_args, **_kwargs):
            yield wav_path

        def _fake_embed_via_tempfile(*_args, **_kwargs):
            return fake_embedding

        # Patch the embed sink so _apply_clip_and_embed completes without a
        # real audio embedder; the clip-to-bytes step still runs against the
        # real WAV file.
        with (
            patch("vtscore.datasets.ingest._media_type_from_origin", return_value="audio"),
            patch("vtscore.detectors.resolver.resolve_file_context", _fake_resolve_ctx),
            patch("vtscore.detectors.resolver._embed_via_tempfile", _fake_embed_via_tempfile),
        ):
            ingested = _ingest_via_resolver(origin, entries, {}, lambda *a, **k: None)

        assert ingested == 1

    def test_two_clips_of_same_parent_get_distinct_md5s(self, tmp_path):
        """Re-ingesting two audio clips of the same parent yields two distinct MD5s.

        Without the H10 fix, both clips inherit the parent's MD5 and
        ``collapse_duplicates`` would merge them on the next save+reload.
        """
        import numpy as np

        from vtscore.datasets.ingest import _ingest_via_resolver
        from vtscore.media.audio.audio_generator import generate_wav

        rng = np.random.default_rng(42)

        wav = generate_wav(440, 5.0)
        wav_path = tmp_path / "test.wav"
        wav_path.write_bytes(wav)

        def _origin(start: float, end: float, idx: int) -> dict:
            return {
                "importer": "server_folder",
                "params": {
                    "path": str(tmp_path),
                    "media_type": "audio",
                    "clipper": "sound_tiling",
                    "clip_start": str(start),
                    "clip_end": str(end),
                    "clip_index": str(idx),
                },
            }

        entries_a = [{"origin": _origin(0.0, 2.0, 0), "origin_name": "test.wav", "md5": "", "label": "good"}]
        entries_b = [{"origin": _origin(2.0, 4.0, 1), "origin_name": "test.wav", "md5": "", "label": "bad"}]

        fake_embedding = rng.standard_normal(512).astype(np.float32)

        from contextlib import contextmanager
        from unittest.mock import patch

        @contextmanager
        def _fake_resolve_ctx(*_args, **_kwargs):
            yield wav_path

        existing: dict = {}
        with (
            patch("vtscore.datasets.ingest._media_type_from_origin", return_value="audio"),
            patch("vtscore.detectors.resolver.resolve_file_context", _fake_resolve_ctx),
            patch("vtscore.detectors.resolver._embed_via_tempfile", return_value=fake_embedding),
        ):
            assert _ingest_via_resolver(entries_a[0]["origin"], entries_a, existing, lambda *a, **k: None) == 1
            assert _ingest_via_resolver(entries_b[0]["origin"], entries_b, existing, lambda *a, **k: None) == 1

        assert len(existing) == 2
        md5s = {m["md5"] for m in existing.values()}
        assert len(md5s) == 2, f"expected distinct MD5s for distinct clips, got {md5s}"

    def test_clip_media_bytes_and_file_size_describe_the_clip(self, tmp_path):
        """media_bytes and file_size on a re-ingested clip describe the clip, not the parent."""
        import numpy as np

        from vtscore.datasets.ingest import _ingest_via_resolver
        from vtscore.media.audio.audio_generator import generate_wav
        from vtscore.media.audio.clipper import _wav_slice

        rng = np.random.default_rng(42)

        wav = generate_wav(440, 5.0)
        wav_path = tmp_path / "test.wav"
        wav_path.write_bytes(wav)

        clip_start, clip_end = 0.0, 2.0
        expected_clip_bytes = _wav_slice(wav, clip_start, clip_end)
        assert len(expected_clip_bytes) < len(wav), "clip should be smaller than parent"

        origin = {
            "importer": "server_folder",
            "params": {
                "path": str(tmp_path),
                "media_type": "audio",
                "clipper": "sound_tiling",
                "clip_start": str(clip_start),
                "clip_end": str(clip_end),
                "clip_index": "0",
            },
        }
        entries = [{"origin": origin, "origin_name": "test.wav", "md5": "", "label": "good"}]

        from contextlib import contextmanager
        from unittest.mock import patch

        @contextmanager
        def _fake_resolve_ctx(*_args, **_kwargs):
            yield wav_path

        fake_embedding = rng.standard_normal(512).astype(np.float32)

        existing: dict = {}
        with (
            patch("vtscore.datasets.ingest._media_type_from_origin", return_value="audio"),
            patch("vtscore.detectors.resolver.resolve_file_context", _fake_resolve_ctx),
            patch("vtscore.detectors.resolver._embed_via_tempfile", return_value=fake_embedding),
        ):
            assert _ingest_via_resolver(origin, entries, existing, lambda *a, **k: None) == 1

        media = next(iter(existing.values()))
        assert media["media_bytes"] == expected_clip_bytes, "media_bytes must be clip bytes, not parent"
        assert media["file_size"] == len(expected_clip_bytes), "file_size must reflect clip, not parent"

    def test_video_metadata_only_clips_get_unique_md5_via_boundary_tag(self, tmp_path):
        """Two video clips of the same parent (metadata-only) still get distinct MD5s.

        Video clippers don't slice the underlying bytes — they only stamp
        ``clip_start`` / ``clip_end``. Without the boundary-tag fallback,
        all clips of one parent collide on the parent's MD5 and
        ``collapse_duplicates`` merges them after a save+reload.
        """
        import numpy as np

        from vtscore.datasets.ingest import _ingest_via_resolver

        rng = np.random.default_rng(42)

        # _apply_clip_and_embed's video branch falls through to embed_file
        # (legacy _clip_to_bytes returns None for video).  We just need
        # embed_file to return a non-None embedding so the function
        # reaches the boundary-tag MD5 path.
        fake_video = tmp_path / "movie.mp4"
        fake_video.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"x" * 1024)

        def _origin(start: float, end: float, idx: int) -> dict:
            return {
                "importer": "server_folder",
                "params": {
                    "path": str(tmp_path),
                    "media_type": "video",
                    "clipper": "video_tiling",
                    "clip_start": str(start),
                    "clip_end": str(end),
                    "clip_index": str(idx),
                },
            }

        fake_embedding = rng.standard_normal(512).astype(np.float32)

        from contextlib import contextmanager
        from unittest.mock import patch

        @contextmanager
        def _fake_resolve_ctx(*_args, **_kwargs):
            yield fake_video

        existing: dict = {}
        with (
            patch("vtscore.datasets.ingest._media_type_from_origin", return_value="video"),
            patch("vtscore.detectors.resolver.resolve_file_context", _fake_resolve_ctx),
            patch("vtscore.detectors.resolver.embed_file", return_value=fake_embedding),
        ):
            assert (
                _ingest_via_resolver(
                    _origin(0.0, 2.0, 0),
                    [{"origin": _origin(0.0, 2.0, 0), "origin_name": "movie.mp4", "md5": "", "label": "good"}],
                    existing,
                    lambda *a, **k: None,
                )
                == 1
            )
            assert (
                _ingest_via_resolver(
                    _origin(2.0, 4.0, 1),
                    [{"origin": _origin(2.0, 4.0, 1), "origin_name": "movie.mp4", "md5": "", "label": "bad"}],
                    existing,
                    lambda *a, **k: None,
                )
                == 1
            )

        assert len(existing) == 2
        md5s = {m["md5"] for m in existing.values()}
        assert len(md5s) == 2, f"video clips of same parent must get distinct MD5s, got {md5s}"
