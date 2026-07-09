"""Tests for Clotho download, .7z extraction, and load_demo_source integration."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# .7z archive handling in the downloader core (validation + extraction dispatch)
# ---------------------------------------------------------------------------


_SEVENZIP_HEADER = b"7z\xbc\xaf\x27\x1c" + b"\x00" * 20


class TestSevenZipValidation:
    def test_accepts_genuine_7z_magic(self, tmp_path):
        """A file starting with the 7z signature validates cleanly."""
        from vtscore.datasets.downloader import core

        archive = tmp_path / "clotho_audio_evaluation.7z"
        archive.write_bytes(_SEVENZIP_HEADER)

        core._validate_archive(archive, "clotho_audio_evaluation.7z", "Clotho")

        assert archive.exists()

    def test_rejects_html_error_page(self, tmp_path):
        """An HTML/text body saved with a .7z suffix is rejected and deleted."""
        from vtscore.datasets.downloader import core

        archive = tmp_path / "clotho_audio_evaluation.7z"
        archive.write_bytes(b"<!DOCTYPE html><html>404</html>")

        with pytest.raises(RuntimeError, match="invalid file"):
            core._validate_archive(archive, "clotho_audio_evaluation.7z", "Clotho")

        assert not archive.exists()


class TestSevenZipExtraction:
    def test_extract_archive_dispatches_to_7z(self, tmp_path):
        """_extract_archive routes a .7z name to _extract_7z."""
        from vtscore.datasets.downloader import core

        called: dict = {}
        with patch.object(core, "_extract_7z", lambda *a: called.setdefault("hit", True)):
            core._extract_archive(tmp_path / "a.7z", "a.7z", tmp_path, "Clotho", lambda *a: None)

        assert called.get("hit")

    def test_missing_py7zr_raises_actionable_error(self, tmp_path):
        """When py7zr is not importable, a helpful RuntimeError is raised."""
        from vtscore.datasets.downloader import core

        # sys.modules[name] = None makes `import name` raise ImportError.
        with patch.dict(sys.modules, {"py7zr": None}):
            with pytest.raises(RuntimeError, match="py7zr"):
                core._extract_7z(tmp_path / "a.7z", tmp_path, "Clotho", lambda *a: None)

    def test_extracts_via_py7zr(self, tmp_path):
        """_extract_7z opens the archive and calls extractall into dest_dir."""
        from vtscore.datasets.downloader import core

        fake_archive = MagicMock()
        fake_archive.getnames.return_value = ["evaluation/clip_0001.wav"]
        fake_mod = MagicMock()
        fake_mod.SevenZipFile.return_value.__enter__.return_value = fake_archive

        with patch.dict(sys.modules, {"py7zr": fake_mod}):
            core._extract_7z(tmp_path / "a.7z", tmp_path, "Clotho", lambda *a: None)

        fake_archive.extractall.assert_called_once()

    def test_rejects_path_traversal_member(self, tmp_path):
        """A member escaping dest_dir is rejected before extraction."""
        from vtscore.datasets.downloader import core

        fake_archive = MagicMock()
        fake_archive.getnames.return_value = ["../evil.wav"]
        fake_mod = MagicMock()
        fake_mod.SevenZipFile.return_value.__enter__.return_value = fake_archive

        with patch.dict(sys.modules, {"py7zr": fake_mod}):
            with pytest.raises(ValueError, match="traversal"):
                core._extract_7z(tmp_path / "a.7z", tmp_path, "Clotho", lambda *a: None)

        fake_archive.extractall.assert_not_called()


# ---------------------------------------------------------------------------
# download_clotho
# ---------------------------------------------------------------------------


class TestDownloadClotho:
    def test_returns_clotho_directory(self, tmp_path):
        """download_clotho downloads, extracts, and returns the clotho/ dir."""
        from vtscore.datasets import downloader as dl_module

        def fake_download(url, dest, size, cb):
            Path(dest).write_bytes(_SEVENZIP_HEADER)

        def fake_extract(archive_path, archive_name, dest_dir, dataset_name, on_progress):
            # Stand in for the real .7z extraction: lay down evaluation/*.wav.
            eval_dir = Path(dest_dir) / "evaluation"
            eval_dir.mkdir(parents=True, exist_ok=True)
            (eval_dir / "clip_0001.wav").write_bytes(b"RIFF" + b"\x00" * 40)

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(dl_module.core, "download_file_with_progress", fake_download),
            patch.object(dl_module.core, "_extract_archive", fake_extract),
        ):
            result = dl_module.download_clotho(on_progress=lambda *a: None)

        assert result == tmp_path / "clotho"
        assert result.exists()
        assert any(result.rglob("*.wav"))

    def test_cached_extraction_skips_download(self, tmp_path):
        """If the clotho directory already exists, no download is triggered."""
        from vtscore.datasets import downloader as dl_module

        eval_dir = tmp_path / "clotho" / "evaluation"
        eval_dir.mkdir(parents=True)
        (eval_dir / "clip_0001.wav").write_bytes(b"RIFF" + b"\x00" * 40)

        download_called = []

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(
                dl_module.core,
                "download_file_with_progress",
                lambda *a, **kw: download_called.append(True),
            ),
        ):
            result = dl_module.download_clotho(on_progress=lambda *a: None)

        assert not download_called, "download should be skipped when cache exists"
        assert result == tmp_path / "clotho"


# ---------------------------------------------------------------------------
# load_demo_source: clotho
# ---------------------------------------------------------------------------


class TestLoadDemoSourceClotho:
    """AudioMediaType.load_demo_source with source='clotho'."""

    def _make_mock_embedder(self):
        import numpy as np

        mock_emb = MagicMock()
        mock_emb.name = "clap"
        mock_emb.media_type_id = "audio"
        mock_emb._model = True
        mock_emb.embed_media = MagicMock(return_value=np.zeros(512))
        return mock_emb

    def _make_recordings(self, tmp_path, n):
        audio_dir = tmp_path / "clotho"
        eval_dir = audio_dir / "evaluation"
        eval_dir.mkdir(parents=True)
        for i in range(n):
            (eval_dir / f"clip_{i:04d}.wav").write_bytes(b"RIFF" + b"\x00" * 40)
        return audio_dir

    def test_single_sound_bucket_populates_clips(self, tmp_path):
        """All recordings load under one 'sound' category."""
        from vtscore.datasets import downloader as dl_module
        from vtscore.media.audio.media_type import AudioMediaType

        audio_dir = self._make_recordings(tmp_path, 7)

        mt = AudioMediaType()
        mt.load_media_data = MagicMock(return_value={"media_bytes": b"", "duration": 1.0})
        mock_emb = self._make_mock_embedder()
        clips: dict = {}

        with patch.object(dl_module, "download_clotho", return_value=audio_dir):
            mt.load_demo_source(
                source="clotho",
                categories=["sound"],
                slice_start=0,
                slice_end=None,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=mock_emb,
            )

        assert len(clips) == 7
        assert {c["category"] for c in clips.values()} == {"sound"}

    def test_fractional_slice_is_applied(self, tmp_path):
        """slice_frac bounds limit how many recordings load (the S/M/L/A axis)."""
        from vtscore.datasets import downloader as dl_module
        from vtscore.media.audio.media_type import AudioMediaType

        audio_dir = self._make_recordings(tmp_path, 7)

        mt = AudioMediaType()
        mt.load_media_data = MagicMock(return_value={"media_bytes": b"", "duration": 1.0})
        mock_emb = self._make_mock_embedder()
        clips: dict = {}

        with patch.object(dl_module, "download_clotho", return_value=audio_dir):
            mt.load_demo_source(
                source="clotho",
                categories=["sound"],
                slice_start=0,
                slice_end=None,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=mock_emb,
                slice_frac_start=0.0,
                slice_frac_end=1 / 7,
            )

        # int(7 * 1/7) == 1 recording in the small slice.
        assert len(clips) == 1

    def test_skip_embedding_defers_to_clipper(self, tmp_path):
        """skip_embedding=True keeps every media with a deferred-embed placeholder."""
        from vtscore.datasets import downloader as dl_module
        from vtscore.media.audio.media_type import AudioMediaType

        audio_dir = self._make_recordings(tmp_path, 3)

        mt = AudioMediaType()
        mt.load_media_data = MagicMock(return_value={"media_bytes": b"", "duration": 1.0})
        mock_emb = self._make_mock_embedder()
        mock_emb.embed_media = MagicMock(return_value=None)
        mock_emb.load_models = MagicMock()
        clips: dict = {}

        with patch.object(dl_module, "download_clotho", return_value=audio_dir):
            mt.load_demo_source(
                source="clotho",
                categories=["sound"],
                slice_start=0,
                slice_end=None,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=mock_emb,
                skip_embedding=True,
            )

        assert len(clips) == 3
        mock_emb.embed_media.assert_not_called()
        for clip in clips.values():
            assert clip["embeddings"] == {}
            assert clip["embedder"] == "clap"
