"""Tests for thin (lazy-loading) media reference support.

Verifies that datasets can be loaded in thin mode (storing media_path
instead of media_bytes) and that lazy loading correctly resolves media
content when needed.
"""

import hashlib
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from vtscore.datasets.loader import (
    _streaming_md5,
    load_dataset_from_folder,
    load_dataset_from_pickle,
)


from helpers import make_wav_bytes as _make_wav_bytes, make_wav_file as _make_wav_file  # noqa: F401


class TestStreamingMD5:
    def test_matches_regular_md5(self, tmp_path):
        content = b"hello world test data"
        p = tmp_path / "test.bin"
        p.write_bytes(content)
        assert _streaming_md5(p) == hashlib.md5(content).hexdigest()

    def test_large_file(self, tmp_path):
        # File larger than the 8192 chunk size
        content = b"x" * 20000
        p = tmp_path / "large.bin"
        p.write_bytes(content)
        assert _streaming_md5(p) == hashlib.md5(content).hexdigest()


class TestThinLoadFromFolder:
    """Test load_dataset_from_folder with thin=True."""

    def test_thin_clips_have_media_path(self, tmp_path):
        _make_wav_file(tmp_path, "test1.wav")
        _make_wav_file(tmp_path, "test2.wav")
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_folder(tmp_path, "audio", medias, thin=True)
        assert len(medias) == 2
        for media in medias.values():
            assert media["media_path"] is not None
            assert Path(media["media_path"]).exists()

    def test_thin_clips_have_no_bytes(self, tmp_path):
        _make_wav_file(tmp_path, "test.wav")
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_folder(tmp_path, "audio", medias, thin=True)
        media = medias[1]
        assert media["media_bytes"] is None
        assert media["media_string"] is None

    def test_thin_clips_leave_embedding_none(self, tmp_path):
        """The loader doesn't embed; framework ``embed_missing`` fills these in."""
        _make_wav_file(tmp_path, "test.wav")
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_folder(tmp_path, "audio", medias, thin=True)
        media = medias[1]
        assert media["embedding"] is None

    def test_thin_clips_have_correct_file_size(self, tmp_path):
        wav_path = _make_wav_file(tmp_path, "test.wav")
        expected_size = wav_path.stat().st_size
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_folder(tmp_path, "audio", medias, thin=True)
        assert medias[1]["file_size"] == expected_size

    def test_thin_clips_have_correct_md5(self, tmp_path):
        wav_path = _make_wav_file(tmp_path, "test.wav")
        expected_md5 = hashlib.md5(wav_path.read_bytes()).hexdigest()
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_folder(tmp_path, "audio", medias, thin=True)
        assert medias[1]["md5"] == expected_md5

    def test_thin_no_duration(self, tmp_path):
        """Thin mode skips load_media_data, so duration stays at default 0."""
        _make_wav_file(tmp_path, "test.wav")
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_folder(tmp_path, "audio", medias, thin=True)
        assert medias[1]["duration"] == 0

    def test_full_mode_has_bytes(self, tmp_path):
        """Full mode (thin=False) should still load bytes as before."""
        _make_wav_file(tmp_path, "test.wav")
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_folder(tmp_path, "audio", medias, thin=False)
        assert medias[1]["media_bytes"] is not None
        assert isinstance(medias[1]["media_bytes"], bytes)

    def test_full_mode_also_has_media_path(self, tmp_path):
        """Full mode should also store media_path for potential future use."""
        _make_wav_file(tmp_path, "test.wav")
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_folder(tmp_path, "audio", medias, thin=False)
        assert medias[1]["media_path"] is not None


class TestThinLoadFromPickle:
    """Test load_dataset_from_pickle with thin=True."""

    def _make_pickle(self, tmp_path, inline_bytes=True, audio_dir=None):
        """Create a test pickle with one audio media."""
        wav_bytes = _make_wav_bytes()
        media_data: dict[str, Any] = {
            "id": 1,
            "media_type": "audio",
            "duration": 0.1,
            "file_size": len(wav_bytes),
            "md5": hashlib.md5(wav_bytes).hexdigest(),
            "embedding": np.zeros(512).tolist(),
            "filename": "test.wav",
            "category": "test",
        }
        if inline_bytes:
            media_data["media_bytes"] = wav_bytes

        pkl_data: dict[str, Any] = {"medias": {1: media_data}}
        if audio_dir:
            pkl_data["audio_dir"] = str(audio_dir)
            # Write the actual file
            audio_dir.mkdir(exist_ok=True)
            (audio_dir / "test.wav").write_bytes(wav_bytes)

        pkl_path = tmp_path / "test.pkl"
        from vtscore.datasets.container import write_container

        write_container(pkl_path, pickle.dumps(pkl_data), {"format_version": 1})
        return pkl_path

    def test_thin_pickle_skips_inline_bytes(self, tmp_path):
        pkl_path = self._make_pickle(tmp_path, inline_bytes=True)
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, medias, thin=True)
        assert len(medias) == 1
        assert medias[1]["media_bytes"] is None

    def test_thin_pickle_has_embedding(self, tmp_path):
        pkl_path = self._make_pickle(tmp_path, inline_bytes=True)
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, medias, thin=True)
        assert isinstance(medias[1]["embedding"], np.ndarray)

    def test_thin_pickle_resolves_media_path_from_audio_dir(self, tmp_path):
        audio_dir = tmp_path / "audio"
        pkl_path = self._make_pickle(tmp_path, inline_bytes=False, audio_dir=audio_dir)
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, medias, thin=True)
        assert medias[1]["media_path"] is not None
        assert Path(medias[1]["media_path"]).exists()

    def test_thin_pickle_preserves_metadata(self, tmp_path):
        pkl_path = self._make_pickle(tmp_path, inline_bytes=True)
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, medias, thin=True)
        assert medias[1]["media_type"] == "audio"
        assert medias[1]["filename"] == "test.wav"
        assert medias[1]["category"] == "test"

    def test_full_pickle_still_works(self, tmp_path):
        pkl_path = self._make_pickle(tmp_path, inline_bytes=True)
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, medias, thin=False)
        assert medias[1]["media_bytes"] is not None


class TestPickleNullEmbedding:
    """Skip-on-None pickle entries (audit M8 / M12).

    ``np.array(None)`` returns a 0-d ``dtype=object`` array that survives
    every ``is None`` guard in the codebase, so the pickle loader must
    drop entries whose ``embedding`` field is missing or ``None`` before
    they enter the medias dict.  Mirrors the folder loader, which
    already returns ``None`` for failed embeds.
    """

    def _wav_pickle(
        self,
        tmp_path: Path,
        *,
        embedding: Any,
        embedding_key_present: bool = True,
        include_bytes: bool = True,
    ) -> Path:
        wav_bytes = _make_wav_bytes()
        media: dict[str, Any] = {
            "id": 1,
            "media_type": "audio",
            "duration": 0.1,
            "file_size": len(wav_bytes),
            "md5": hashlib.md5(wav_bytes).hexdigest(),
            "filename": "test.wav",
            "category": "test",
        }
        if embedding_key_present:
            media["embedding"] = embedding
        if include_bytes:
            media["media_bytes"] = wav_bytes
        pkl_path = tmp_path / "test.pkl"
        from vtscore.datasets.container import write_container

        write_container(pkl_path, pickle.dumps({"medias": {1: media}}), {"format_version": 1})
        return pkl_path

    def test_full_mode_skips_explicit_none_embedding(self, tmp_path, capsys):
        pkl_path = self._wav_pickle(tmp_path, embedding=None)
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, medias, thin=False)
        assert medias == {}
        out = capsys.readouterr().out
        assert "1 media files missing" in out

    def test_full_mode_skips_missing_embedding_key(self, tmp_path, capsys):
        pkl_path = self._wav_pickle(tmp_path, embedding=None, embedding_key_present=False)
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, medias, thin=False)
        assert medias == {}
        out = capsys.readouterr().out
        assert "1 media files missing" in out

    def test_thin_mode_skips_explicit_none_embedding(self, tmp_path, capsys):
        """Regression for M8: prior code only checked key absence, not None."""
        pkl_path = self._wav_pickle(tmp_path, embedding=None, include_bytes=False)
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, medias, thin=True)
        assert medias == {}
        out = capsys.readouterr().out
        assert "1 media files missing" in out

    def test_thin_mode_skips_missing_embedding_key(self, tmp_path, capsys):
        pkl_path = self._wav_pickle(tmp_path, embedding=None, embedding_key_present=False, include_bytes=False)
        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, medias, thin=True)
        assert medias == {}
        out = capsys.readouterr().out
        assert "1 media files missing" in out

    def test_full_mode_mixed_keeps_good_drops_null(self, tmp_path):
        wav_bytes = _make_wav_bytes()
        good = {
            "id": 1,
            "media_type": "audio",
            "duration": 0.1,
            "file_size": len(wav_bytes),
            "md5": hashlib.md5(wav_bytes).hexdigest(),
            "embedding": np.zeros(512).tolist(),
            "filename": "good.wav",
            "category": "test",
            "media_bytes": wav_bytes,
        }
        bad = {
            "id": 2,
            "media_type": "audio",
            "duration": 0.1,
            "file_size": len(wav_bytes),
            "md5": hashlib.md5(wav_bytes + b"x").hexdigest(),
            "embedding": None,
            "filename": "bad.wav",
            "category": "test",
            "media_bytes": wav_bytes,
        }
        pkl_path = tmp_path / "mixed.pkl"
        from vtscore.datasets.container import write_container

        write_container(pkl_path, pickle.dumps({"medias": {1: good, 2: bad}}), {"format_version": 1})

        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, medias, thin=False)
        assert list(medias.keys()) == [1]
        # No poisoned 0-d object array snuck through.
        assert medias[1]["embedding"].ndim >= 1
        assert medias[1]["embedding"].dtype != object

    def test_chunked_loader_skips_null_embedding(self, tmp_path):
        from vtscore.datasets.loader import load_dataset_from_pickle_chunked

        wav_bytes = _make_wav_bytes()
        pkl_data = {
            "medias": {
                1: {
                    "id": 1,
                    "media_type": "audio",
                    "duration": 0.1,
                    "file_size": len(wav_bytes),
                    "md5": hashlib.md5(wav_bytes).hexdigest(),
                    "embedding": np.zeros(512).tolist(),
                    "filename": "a.wav",
                    "category": "test",
                    "media_bytes": wav_bytes,
                },
                2: {
                    "id": 2,
                    "media_type": "audio",
                    "duration": 0.1,
                    "file_size": len(wav_bytes),
                    "md5": hashlib.md5(wav_bytes + b"x").hexdigest(),
                    "embedding": None,
                    "filename": "b.wav",
                    "category": "test",
                    "media_bytes": wav_bytes,
                },
            }
        }
        pkl_path = tmp_path / "chunked.pkl"
        from vtscore.datasets.container import write_container

        write_container(pkl_path, pickle.dumps(pkl_data), {"format_version": 1})

        chunks = list(load_dataset_from_pickle_chunked(pkl_path, chunk_size=10))
        loaded = {cid: m for chunk in chunks for cid, m in chunk.items()}
        assert len(loaded) == 1
        only = next(iter(loaded.values()))
        assert only["filename"] == "a.wav"


class TestPickleMD5Preservation:
    """Test that load_dataset_from_pickle uses pre-existing MD5 from pickle data."""

    def test_full_mode_uses_md5_from_pickle_when_present(self, tmp_path):
        """Full mode should use the MD5 stored in the pickle instead of recalculating."""
        wav_bytes = _make_wav_bytes()
        pre_md5 = "a" * 32  # A fake MD5 that differs from the real hash
        pkl_data = {
            "medias": {
                1: {
                    "id": 1,
                    "media_type": "audio",
                    "duration": 0.1,
                    "file_size": len(wav_bytes),
                    "md5": pre_md5,
                    "embedding": np.zeros(512).tolist(),
                    "filename": "test.wav",
                    "category": "test",
                    "media_bytes": wav_bytes,
                }
            }
        }
        pkl_path = tmp_path / "test.pkl"
        from vtscore.datasets.container import write_container

        write_container(pkl_path, pickle.dumps(pkl_data), {"format_version": 1})

        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, medias, thin=False)
        assert medias[1]["md5"] == pre_md5

    def test_full_mode_computes_md5_when_missing_from_pickle(self, tmp_path):
        """Full mode should compute the MD5 if the pickle doesn't have one."""
        wav_bytes = _make_wav_bytes()
        pkl_data = {
            "medias": {
                1: {
                    "id": 1,
                    "media_type": "audio",
                    "duration": 0.1,
                    "file_size": len(wav_bytes),
                    # no "md5" key
                    "embedding": np.zeros(512).tolist(),
                    "filename": "test.wav",
                    "category": "test",
                    "media_bytes": wav_bytes,
                }
            }
        }
        pkl_path = tmp_path / "test.pkl"
        from vtscore.datasets.container import write_container

        write_container(pkl_path, pickle.dumps(pkl_data), {"format_version": 1})

        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, medias, thin=False)
        assert medias[1]["md5"] == hashlib.md5(wav_bytes).hexdigest()

    def test_thin_mode_uses_md5_from_pickle(self, tmp_path):
        """Thin mode should also preserve the MD5 from the pickle."""
        wav_bytes = _make_wav_bytes()
        pre_md5 = "b" * 32
        pkl_data = {
            "medias": {
                1: {
                    "id": 1,
                    "media_type": "audio",
                    "duration": 0.1,
                    "file_size": len(wav_bytes),
                    "md5": pre_md5,
                    "embedding": np.zeros(512).tolist(),
                    "filename": "test.wav",
                    "category": "test",
                    "media_bytes": wav_bytes,
                }
            }
        }
        pkl_path = tmp_path / "test.pkl"
        from vtscore.datasets.container import write_container

        write_container(pkl_path, pickle.dumps(pkl_data), {"format_version": 1})

        medias: dict[int, dict[str, Any]] = {}
        load_dataset_from_pickle(pkl_path, medias, thin=True)
        assert medias[1]["md5"] == pre_md5


class TestThinImporters:
    """Test that importers pass thin parameter through correctly."""

    def test_folder_importer_thin(self, tmp_path):
        _make_wav_file(tmp_path, "test.wav")
        from vtscore.datasets.importers.server_folder import ServerFolderDatasetImporter

        importer = ServerFolderDatasetImporter()
        medias: dict[int, dict[str, Any]] = {}
        importer.run({"path": str(tmp_path), "media_type": "audio"}, medias, thin=True)
        assert len(medias) > 0
        assert medias[1]["media_bytes"] is None
        assert medias[1]["media_path"] is not None

    def test_folder_importer_run_cli_thin(self, tmp_path):
        _make_wav_file(tmp_path, "test.wav")
        from vtscore.datasets.importers.server_folder import ServerFolderDatasetImporter

        importer = ServerFolderDatasetImporter()
        medias: dict[int, dict[str, Any]] = {}
        importer.run_cli({"path": str(tmp_path), "media_type": "audio"}, medias, thin=True)
        assert len(medias) > 0
        assert medias[1]["media_bytes"] is None

    def test_pickle_importer_thin(self, tmp_path):
        # Create a pickle first
        wav_bytes = _make_wav_bytes()
        pkl_data = {
            "medias": {
                1: {
                    "id": 1,
                    "media_type": "audio",
                    "duration": 0.1,
                    "file_size": len(wav_bytes),
                    "md5": hashlib.md5(wav_bytes).hexdigest(),
                    "embedding": np.zeros(512).tolist(),
                    "filename": "test.wav",
                    "category": "test",
                    "media_bytes": wav_bytes,
                }
            }
        }
        pkl_path = tmp_path / "test.pkl"
        from vtscore.datasets.container import write_container

        write_container(pkl_path, pickle.dumps(pkl_data), {"format_version": 1})

        from vtscore.datasets.importers.pickle import PickleDatasetImporter

        importer = PickleDatasetImporter()
        medias: dict[int, dict[str, Any]] = {}
        importer.run_cli({"file": str(pkl_path)}, medias, thin=True)
        assert len(medias) == 1
        assert medias[1]["media_bytes"] is None


class TestLazyLoadingMediaType:
    """Test that MediaType._resolve_media_bytes/string lazy-loads from media_path."""

    def test_resolve_bytes_from_preloaded(self):
        from vtscore.media.audio.media_type import AudioMediaType

        mt = AudioMediaType()
        media = {"media_bytes": b"hello", "media_path": None}
        assert mt._resolve_media_bytes(media) == b"hello"

    def test_resolve_bytes_from_media_path(self, tmp_path):
        from vtscore.media.audio.media_type import AudioMediaType

        content = b"lazy loaded content"
        p = tmp_path / "test.wav"
        p.write_bytes(content)

        mt = AudioMediaType()
        media = {"media_bytes": None, "media_path": str(p)}
        assert mt._resolve_media_bytes(media) == content

    def test_resolve_bytes_missing_file(self):
        from vtscore.media.audio.media_type import AudioMediaType

        mt = AudioMediaType()
        media = {"media_bytes": None, "media_path": "/nonexistent/file.wav"}
        assert mt._resolve_media_bytes(media) is None

    def test_resolve_bytes_no_path(self):
        from vtscore.media.audio.media_type import AudioMediaType

        mt = AudioMediaType()
        media = {"media_bytes": None, "media_path": None}
        assert mt._resolve_media_bytes(media) is None

    def test_resolve_string_from_preloaded(self):
        from vtscore.media.text.media_type import TextMediaType

        mt = TextMediaType()
        media = {"media_string": "hello world", "media_path": None}
        assert mt._resolve_media_string(media) == "hello world"

    def test_resolve_string_from_media_path(self, tmp_path):
        from vtscore.media.text.media_type import TextMediaType

        content = "lazy loaded text content"
        p = tmp_path / "test.txt"
        p.write_text(content, encoding="utf-8")

        mt = TextMediaType()
        media = {"media_string": None, "media_path": str(p)}
        assert mt._resolve_media_string(media) == content


class TestClipResponseLazyLoading:
    """Test that media_response works with lazy-loaded media."""

    def test_audio_media_response_lazy(self, tmp_path):
        from vtscore.media.audio.media_type import AudioMediaType

        wav_bytes = _make_wav_bytes()
        p = tmp_path / "test.wav"
        p.write_bytes(wav_bytes)

        mt = AudioMediaType()
        media = {"id": 1, "media_bytes": None, "media_path": str(p), "filename": "test.wav"}
        resp = mt.media_response(media)
        assert resp.data == wav_bytes
        assert resp.mimetype == "audio/wav"

    def test_image_media_response_lazy(self, tmp_path):
        from vtscore.media.image.media_type import ImageMediaType

        # Create a minimal PNG
        from PIL import Image as PILImage
        import io

        img = PILImage.new("RGB", (2, 2), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        p = tmp_path / "test.png"
        p.write_bytes(png_bytes)

        mt = ImageMediaType()
        media = {"id": 1, "media_bytes": None, "media_path": str(p), "filename": "test.png"}
        resp = mt.media_response(media)
        assert resp.data == png_bytes
        assert resp.mimetype == "image/png"

    def test_text_media_response_lazy(self, tmp_path):
        from vtscore.media.text.media_type import TextMediaType

        content = "lazy loaded paragraph"
        p = tmp_path / "test.txt"
        p.write_text(content, encoding="utf-8")

        mt = TextMediaType()
        media = {
            "id": 1,
            "media_string": None,
            "media_path": str(p),
            "word_count": 0,
            "character_count": 0,
        }
        resp = mt.media_response(media)
        assert isinstance(resp.data, dict)
        assert resp.data["content"] == content
        assert resp.data["word_count"] == 3  # "lazy loaded paragraph"
        assert resp.data["character_count"] == len(content)

    def test_audio_media_response_no_data(self):
        """media_response returns empty bytes when no data is available."""
        from vtscore.media.audio.media_type import AudioMediaType

        mt = AudioMediaType()
        media = {"id": 1, "media_bytes": None, "media_path": None, "filename": "test.wav"}
        resp = mt.media_response(media)
        assert resp.data == b""
