"""Tests for GTZAN dataset download and load_demo_source integration."""

import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gtzan_tar(tmp_path: Path) -> Path:
    """Create a minimal GTZAN tar.gz fixture with 3 files per genre."""
    tar_path = tmp_path / "genres.tar.gz"

    tree_root = tmp_path / "tar_staging" / "genres"
    for genre in ("blues", "classical", "rock"):
        d = tree_root / genre
        d.mkdir(parents=True)
        for i in range(3):
            # Create tiny WAV-like files (just enough content to exist).
            (d / f"{genre}.{i:05d}.wav").write_bytes(b"RIFF" + b"\x00" * 40)

    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(tree_root, arcname="genres")

    return tar_path


# ---------------------------------------------------------------------------
# download_gtzan
# ---------------------------------------------------------------------------


class TestDownloadGtzan:
    def test_returns_genres_directory(self, tmp_path):
        """download_gtzan returns the genres/ directory path."""
        from vtscore.datasets import downloader as dl_module

        tar_path = _make_gtzan_tar(tmp_path)

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(
                dl_module.core,
                "download_file_with_progress",
                lambda url, dest, size, cb: tar_path.rename(dest),
            ),
        ):
            result = dl_module.download_gtzan(on_progress=lambda *a: None)

        assert result.name == "genres"
        assert result.exists()
        assert (result / "blues").is_dir()
        assert (result / "rock").is_dir()

    def test_cached_extraction_skips_download(self, tmp_path):
        """If the genres directory already exists, no download is triggered."""
        from vtscore.datasets import downloader as dl_module

        genres_dir = tmp_path / "gtzan" / "genres" / "blues"
        genres_dir.mkdir(parents=True)
        (genres_dir / "blues.00000.wav").write_bytes(b"RIFF" + b"\x00" * 40)

        download_called = []

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(
                dl_module.core,
                "download_file_with_progress",
                lambda *a, **kw: download_called.append(True),
            ),
        ):
            result = dl_module.download_gtzan(on_progress=lambda *a: None)

        assert not download_called, "download should be skipped when cache exists"
        assert result.exists()


# ---------------------------------------------------------------------------
# download_speech_commands_v2
# ---------------------------------------------------------------------------


class TestDownloadSpeechCommandsV2:
    def test_returns_extract_directory(self, tmp_path):
        """download_speech_commands_v2 returns the extract directory."""
        from vtscore.datasets import downloader as dl_module

        tar_path = tmp_path / "speech_commands_v0.02.tar.gz"

        tree_root = tmp_path / "tar_staging"
        for keyword in ("yes", "no", "stop"):
            d = tree_root / keyword
            d.mkdir(parents=True)
            for i in range(3):
                (d / f"utterance_{i}.wav").write_bytes(b"RIFF" + b"\x00" * 40)

        with tarfile.open(tar_path, "w:gz") as tf:
            for keyword in ("yes", "no", "stop"):
                for f in (tree_root / keyword).iterdir():
                    tf.add(f, arcname=f"{keyword}/{f.name}")

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(
                dl_module.core,
                "download_file_with_progress",
                lambda url, dest, size, cb: tar_path.rename(dest),
            ),
        ):
            result = dl_module.download_speech_commands_v2(on_progress=lambda *a: None)

        assert result.name == "speech_commands_v2"
        assert result.exists()

    def test_cached_extraction_skips_download(self, tmp_path):
        """If the extract directory already exists, no download is triggered."""
        from vtscore.datasets import downloader as dl_module

        extract_dir = tmp_path / "speech_commands_v2"
        (extract_dir / "yes").mkdir(parents=True)
        (extract_dir / "yes" / "u1.wav").write_bytes(b"RIFF" + b"\x00" * 40)

        download_called = []

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(
                dl_module.core,
                "download_file_with_progress",
                lambda *a, **kw: download_called.append(True),
            ),
        ):
            result = dl_module.download_speech_commands_v2(on_progress=lambda *a: None)

        assert not download_called
        assert result.exists()


# ---------------------------------------------------------------------------
# download_urbansound8k
# ---------------------------------------------------------------------------


class TestDownloadUrbansound8k:
    def test_returns_extract_directory(self, tmp_path):
        """download_urbansound8k returns the UrbanSound8K directory."""
        from vtscore.datasets import downloader as dl_module

        tar_path = tmp_path / "UrbanSound8K.tar.gz"

        tree_root = tmp_path / "tar_staging" / "UrbanSound8K"
        audio_dir = tree_root / "audio" / "fold1"
        audio_dir.mkdir(parents=True)
        (audio_dir / "100032-3-0-0.wav").write_bytes(b"RIFF" + b"\x00" * 40)
        meta_dir = tree_root / "metadata"
        meta_dir.mkdir(parents=True)
        (meta_dir / "UrbanSound8K.csv").write_text(
            "slice_file_name,fsID,start,end,salience,fold,classID,class\n100032-3-0-0.wav,100032,0,1,1,1,3,dog_bark\n"
        )

        with tarfile.open(tar_path, "w:gz") as tf:
            tf.add(tree_root, arcname="UrbanSound8K")

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(
                dl_module.core,
                "download_file_with_progress",
                lambda url, dest, size, cb: tar_path.rename(dest),
            ),
        ):
            result = dl_module.download_urbansound8k(on_progress=lambda *a: None)

        assert result.name == "UrbanSound8K"
        assert result.exists()
        assert (result / "audio").is_dir()
        assert (result / "metadata").is_dir()

    def test_cached_extraction_skips_download(self, tmp_path):
        """If the extract directory already exists, no download is triggered."""
        from vtscore.datasets import downloader as dl_module

        extract_dir = tmp_path / "UrbanSound8K"
        (extract_dir / "audio" / "fold1").mkdir(parents=True)
        (extract_dir / "metadata").mkdir(parents=True)

        download_called = []

        with (
            patch.object(dl_module.core, "DATA_DIR", tmp_path),
            patch.object(
                dl_module.core,
                "download_file_with_progress",
                lambda *a, **kw: download_called.append(True),
            ),
        ):
            result = dl_module.download_urbansound8k(on_progress=lambda *a: None)

        assert not download_called
        assert result.exists()


# ---------------------------------------------------------------------------
# load_audio_metadata_from_folders
# ---------------------------------------------------------------------------


class TestLoadAudioMetadataFromFolders:
    def test_scans_category_subdirectories(self, tmp_path):
        """Finds WAV files in matching category subdirectories."""
        from vtscore.datasets.loader import load_audio_metadata_from_folders

        for genre in ("blues", "rock"):
            d = tmp_path / genre
            d.mkdir()
            (d / f"{genre}.00001.wav").write_bytes(b"RIFF")
            (d / f"{genre}.00002.wav").write_bytes(b"RIFF")

        # Unrelated folder should be ignored.
        other = tmp_path / "jazz"
        other.mkdir()
        (other / "jazz.00001.wav").write_bytes(b"RIFF")

        metadata = load_audio_metadata_from_folders(tmp_path, ["blues", "rock"])

        assert len(metadata) == 4
        categories = {m["category"] for m in metadata.values()}
        assert categories == {"blues", "rock"}

    def test_ignores_non_audio_files(self, tmp_path):
        """Non-audio files (like .txt) are not included."""
        from vtscore.datasets.loader import load_audio_metadata_from_folders

        d = tmp_path / "blues"
        d.mkdir()
        (d / "blues.00001.wav").write_bytes(b"RIFF")
        (d / "readme.txt").write_text("not audio")

        metadata = load_audio_metadata_from_folders(tmp_path, ["blues"])
        assert len(metadata) == 1


# ---------------------------------------------------------------------------
# load_urbansound8k_metadata
# ---------------------------------------------------------------------------


class TestLoadUrbansound8kMetadata:
    def test_loads_csv_metadata(self, tmp_path):
        """Reads UrbanSound8K.csv and maps filenames to metadata."""
        from vtscore.datasets.loader import load_urbansound8k_metadata

        meta_dir = tmp_path / "metadata"
        meta_dir.mkdir()
        audio_dir = tmp_path / "audio" / "fold1"
        audio_dir.mkdir(parents=True)
        (audio_dir / "100032-3-0-0.wav").write_bytes(b"RIFF")

        (meta_dir / "UrbanSound8K.csv").write_text(
            "slice_file_name,fsID,start,end,salience,fold,classID,class\n"
            "100032-3-0-0.wav,100032,0,1,1,1,3,dog_bark\n"
            "200032-0-0-0.wav,200032,0,2,1,2,0,air_conditioner\n"
        )

        metadata = load_urbansound8k_metadata(tmp_path)

        assert len(metadata) == 2
        assert metadata["100032-3-0-0.wav"]["category"] == "dog_bark"
        assert metadata["100032-3-0-0.wav"]["fold"] == 1
        assert metadata["100032-3-0-0.wav"]["class_id"] == 3
        assert metadata["200032-0-0-0.wav"]["category"] == "air_conditioner"


# ---------------------------------------------------------------------------
# load_demo_source - audio (gtzan, speech_commands_v2, urbansound8k)
# ---------------------------------------------------------------------------


class TestLoadDemoSourceGtzan:
    """AudioMediaType.load_demo_source with source='gtzan'."""

    def _make_mock_embedder(self):
        import numpy as np

        mock_emb = MagicMock()
        mock_emb.name = "clap"
        mock_emb.media_type_id = "audio"
        mock_emb._model = True
        mock_emb.embed_media = MagicMock(return_value=np.zeros(512))
        return mock_emb

    def test_gtzan_source_populates_clips(self, tmp_path):
        """load_demo_source with source='gtzan' fills the clips dict."""
        from vtscore.datasets import downloader as dl_module
        from vtscore.datasets import loader as loader_module
        from vtscore.media.audio.media_type import AudioMediaType

        fake_metadata = {
            "blues/blues.00001.wav": {"category": "blues", "path": tmp_path / "blues.00001.wav"},
            "rock/rock.00001.wav": {"category": "rock", "path": tmp_path / "rock.00001.wav"},
        }

        # Create stub audio files.
        for meta in fake_metadata.values():
            meta["path"].write_bytes(b"RIFF" + b"\x00" * 40)

        mt = AudioMediaType()
        mt.load_media_data = MagicMock(return_value={"media_bytes": b"", "duration": 1.0})
        mock_emb = self._make_mock_embedder()
        clips: dict = {}

        with (
            patch.object(dl_module, "download_gtzan", return_value=tmp_path),
            patch.object(loader_module, "load_audio_metadata_from_folders", return_value=fake_metadata),
        ):
            mt.load_demo_source(
                source="gtzan",
                categories=["blues", "rock"],
                slice_start=0,
                slice_end=10,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=mock_emb,
            )

        assert len(clips) == 2
        categories_seen = {c["category"] for c in clips.values()}
        assert categories_seen == {"blues", "rock"}

    def test_gtzan_slice_is_applied(self, tmp_path):
        """slice_start/slice_end limits files per category."""
        from vtscore.datasets import downloader as dl_module
        from vtscore.datasets import loader as loader_module
        from vtscore.media.audio.media_type import AudioMediaType

        fake_metadata = {}
        for i in range(10):
            p = tmp_path / f"blues.{i:05d}.wav"
            p.write_bytes(b"RIFF" + b"\x00" * 40)
            fake_metadata[f"blues/blues.{i:05d}.wav"] = {"category": "blues", "path": p}

        mt = AudioMediaType()
        mt.load_media_data = MagicMock(return_value={"media_bytes": b"", "duration": 1.0})
        mock_emb = self._make_mock_embedder()
        clips: dict = {}

        with (
            patch.object(dl_module, "download_gtzan", return_value=tmp_path),
            patch.object(loader_module, "load_audio_metadata_from_folders", return_value=fake_metadata),
        ):
            mt.load_demo_source(
                source="gtzan",
                categories=["blues"],
                slice_start=2,
                slice_end=5,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=mock_emb,
            )

        assert len(clips) == 3


class TestLoadDemoSourceSpeechCommands:
    """AudioMediaType.load_demo_source with source='speech_commands_v2'."""

    def _make_mock_embedder(self):
        import numpy as np

        mock_emb = MagicMock()
        mock_emb.name = "clap"
        mock_emb.media_type_id = "audio"
        mock_emb._model = True
        mock_emb.embed_media = MagicMock(return_value=np.zeros(512))
        return mock_emb

    def test_speech_commands_source_populates_clips(self, tmp_path):
        """load_demo_source with source='speech_commands_v2' fills the clips dict."""
        from vtscore.datasets import downloader as dl_module
        from vtscore.datasets import loader as loader_module
        from vtscore.media.audio.media_type import AudioMediaType

        fake_metadata = {
            "yes/u1.wav": {"category": "yes", "path": tmp_path / "u1.wav"},
            "no/u2.wav": {"category": "no", "path": tmp_path / "u2.wav"},
        }
        for meta in fake_metadata.values():
            meta["path"].write_bytes(b"RIFF" + b"\x00" * 40)

        mt = AudioMediaType()
        mt.load_media_data = MagicMock(return_value={"media_bytes": b"", "duration": 1.0})
        mock_emb = self._make_mock_embedder()
        clips: dict = {}

        with (
            patch.object(dl_module, "download_speech_commands_v2", return_value=tmp_path),
            patch.object(loader_module, "load_audio_metadata_from_folders", return_value=fake_metadata),
        ):
            mt.load_demo_source(
                source="speech_commands_v2",
                categories=["yes", "no"],
                slice_start=0,
                slice_end=10,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=mock_emb,
            )

        assert len(clips) == 2
        categories_seen = {c["category"] for c in clips.values()}
        assert categories_seen == {"yes", "no"}


class TestLoadDemoSourceUrbansound8k:
    """AudioMediaType.load_demo_source with source='urbansound8k'."""

    def _make_mock_embedder(self):
        import numpy as np

        mock_emb = MagicMock()
        mock_emb.name = "clap"
        mock_emb.media_type_id = "audio"
        mock_emb._model = True
        mock_emb.embed_media = MagicMock(return_value=np.zeros(512))
        return mock_emb

    def test_urbansound8k_source_populates_clips(self, tmp_path):
        """load_demo_source with source='urbansound8k' fills the clips dict."""
        from vtscore.datasets import downloader as dl_module
        from vtscore.datasets import loader as loader_module
        from vtscore.media.audio.media_type import AudioMediaType

        f1 = tmp_path / "100032-3-0-0.wav"
        f2 = tmp_path / "200032-8-0-0.wav"
        f1.write_bytes(b"RIFF" + b"\x00" * 40)
        f2.write_bytes(b"RIFF" + b"\x00" * 40)

        fake_metadata = {
            "100032-3-0-0.wav": {"category": "dog_bark", "fold": 1, "class_id": 3, "path": f1},
            "200032-8-0-0.wav": {"category": "siren", "fold": 2, "class_id": 8, "path": f2},
        }

        mt = AudioMediaType()
        mt.load_media_data = MagicMock(return_value={"media_bytes": b"", "duration": 1.0})
        mock_emb = self._make_mock_embedder()
        clips: dict = {}

        with (
            patch.object(dl_module, "download_urbansound8k", return_value=tmp_path),
            patch.object(loader_module, "load_urbansound8k_metadata", return_value=fake_metadata),
        ):
            mt.load_demo_source(
                source="urbansound8k",
                categories=["dog_bark", "siren"],
                slice_start=0,
                slice_end=10,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=mock_emb,
            )

        assert len(clips) == 2
        categories_seen = {c["category"] for c in clips.values()}
        assert categories_seen == {"dog_bark", "siren"}

    def test_unsupported_source_still_raises(self):
        """Non-existent sources still raise ValueError."""
        from vtscore.media.audio.media_type import AudioMediaType

        mt = AudioMediaType()
        import pytest

        with pytest.raises(ValueError, match="Unsupported audio source"):
            mt.load_demo_source(
                source="unknown_source",
                categories=[],
                slice_start=0,
                slice_end=10,
                clips={},
                on_progress=lambda *a: None,
            )
