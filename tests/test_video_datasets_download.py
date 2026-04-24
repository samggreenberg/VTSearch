"""Tests for video dataset downloaders: HMDB51, UCF-101 full, and KTH Actions."""

import shutil
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


# ---------------------------------------------------------------------------
# HMDB51
# ---------------------------------------------------------------------------


class TestDownloadHmdb51:
    """Verify download_hmdb51 downloads, extracts nested RARs, and organises."""

    def test_extracts_into_category_dirs(self, tmp_path):
        """download_hmdb51 should create per-category dirs with .avi files."""
        from vtsearch.datasets.downloader import video as vid_module

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        video_dir = data_dir / "video"

        # Simulate what _extract_rar does: create category dirs with .avi files.
        call_log = []

        def fake_extract_rar(rar_path, extract_to):
            call_log.append(rar_path.name)
            if "hmdb51_org" in rar_path.name:
                # Outer RAR: create inner .rar stubs
                extract_to.mkdir(parents=True, exist_ok=True)
                for cat in ("brush_hair", "cartwheel", "catch"):
                    (extract_to / f"{cat}.rar").write_bytes(b"Rar!")
            else:
                # Inner RAR: create .avi files in the target dir
                extract_to.mkdir(parents=True, exist_ok=True)
                cat = rar_path.stem
                for i in range(3):
                    (extract_to / f"{cat}_video_{i}.avi").write_bytes(
                        b"RIFF" + b"\x00" * 20 + b"AVI "
                    )

        with (
            patch.object(vid_module._core, "DATA_DIR", data_dir),
            patch.object(vid_module._core, "VIDEO_DIR", video_dir),
            patch.object(
                vid_module._core,
                "download_file_with_progress",
                lambda url, dest, size, cb: dest.write_bytes(b"Rar!"),
            ),
            patch.object(vid_module, "_extract_rar", fake_extract_rar),
        ):
            result = vid_module.download_hmdb51(on_progress=lambda *a: None)

        assert result.exists()
        assert result.name == "hmdb51"
        assert (result / "brush_hair").is_dir()
        assert (result / "cartwheel").is_dir()
        assert (result / "catch").is_dir()
        assert len(list((result / "brush_hair").glob("*.avi"))) == 3

    def test_skips_if_already_present(self, tmp_path):
        """If hmdb51/ already has videos, skip download entirely."""
        from vtsearch.datasets.downloader import video as vid_module

        data_dir = tmp_path / "data"
        video_dir = data_dir / "video"
        hmdb_dir = video_dir / "hmdb51" / "brush_hair"
        hmdb_dir.mkdir(parents=True)
        (hmdb_dir / "brush_hair_video_0.avi").write_bytes(b"RIFF" + b"\x00" * 20 + b"AVI ")

        download_called = []

        with (
            patch.object(vid_module._core, "DATA_DIR", data_dir),
            patch.object(vid_module._core, "VIDEO_DIR", video_dir),
            patch.object(
                vid_module._core,
                "download_file_with_progress",
                lambda *a, **kw: download_called.append(True),
            ),
        ):
            result = vid_module.download_hmdb51(on_progress=lambda *a: None)

        assert not download_called, "download should be skipped when cache exists"
        assert result.exists()

    def test_temp_files_cleaned_up(self, tmp_path):
        """Temp archive and staging dir should be removed after extraction."""
        from vtsearch.datasets.downloader import video as vid_module

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        video_dir = data_dir / "video"

        def fake_extract_rar(rar_path, extract_to):
            extract_to.mkdir(parents=True, exist_ok=True)
            if rar_path.name.startswith(".dl_"):
                # Treat as outer
                for cat in ("run",):
                    (extract_to / f"{cat}.rar").write_bytes(b"Rar!")
            elif rar_path.suffix == ".rar" and not rar_path.name.startswith(".dl_"):
                # Treat as outer if in staging, inner otherwise
                if "hmdb51_org" in rar_path.name:
                    for cat in ("run",):
                        (extract_to / f"{cat}.rar").write_bytes(b"Rar!")
                else:
                    cat = rar_path.stem
                    (extract_to / f"{cat}_v1.avi").write_bytes(b"RIFF" + b"\x00" * 20 + b"AVI ")

        with (
            patch.object(vid_module._core, "DATA_DIR", data_dir),
            patch.object(vid_module._core, "VIDEO_DIR", video_dir),
            patch.object(
                vid_module._core,
                "download_file_with_progress",
                lambda url, dest, size, cb: dest.write_bytes(b"Rar!"),
            ),
            patch.object(vid_module, "_extract_rar", fake_extract_rar),
        ):
            vid_module.download_hmdb51(on_progress=lambda *a: None)

        # No temp download files should remain in data_dir.
        leftover_dl = [p for p in data_dir.iterdir() if p.name.startswith(".dl_")]
        assert not leftover_dl, f"Temp archive files should be cleaned up: {leftover_dl}"
        leftover_ex = [p for p in data_dir.iterdir() if p.name.startswith(".extract_")]
        assert not leftover_ex, f"Temp staging dirs should be cleaned up: {leftover_ex}"


class TestExtractRar:
    """Unit tests for the _extract_rar helper."""

    def test_raises_when_unrar_missing(self, tmp_path):
        """_extract_rar raises RuntimeError with install instructions when unrar is absent."""
        import pytest

        from vtsearch.datasets.downloader.video import _extract_rar

        rar_file = tmp_path / "test.rar"
        rar_file.write_bytes(b"Rar!")

        with patch("subprocess.run", side_effect=FileNotFoundError("unrar")):
            with pytest.raises(RuntimeError, match="unrar"):
                _extract_rar(rar_file, tmp_path / "out")


# ---------------------------------------------------------------------------
# UCF-101 Full
# ---------------------------------------------------------------------------


class TestDownloadUcf101Full:
    """Verify download_ucf101_full downloads, extracts, and moves to VIDEO_DIR."""

    def _make_ucf101_zip(self, zip_path: Path) -> None:
        """Create a minimal UCF-101.zip with category subdirectories."""
        with zipfile.ZipFile(zip_path, "w") as zf:
            for cat in ("Archery", "Basketball", "Diving"):
                for i in range(3):
                    fname = f"UCF-101/{cat}/v_{cat}_g{i:02d}_c01.avi"
                    zf.writestr(fname, b"RIFF" + b"\x00" * 20 + b"AVI ")

    def test_extracts_into_category_dirs(self, tmp_path):
        """download_ucf101_full should create per-category dirs with .avi files."""
        from vtsearch.datasets.downloader import video as vid_module

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        video_dir = data_dir / "video"
        zip_path = tmp_path / "UCF-101.zip"
        self._make_ucf101_zip(zip_path)

        with (
            patch.object(vid_module._core, "DATA_DIR", data_dir),
            patch.object(vid_module._core, "VIDEO_DIR", video_dir),
            patch(
                "vtsearch.datasets.downloader.core.download_file_with_progress",
                lambda url, dest, size, cb: shutil.copy(str(zip_path), str(dest)),
            ),
        ):
            result = vid_module.download_ucf101_full(on_progress=lambda *a: None)

        assert result.exists()
        assert result.name == "ucf101_full"
        assert (result / "Archery").is_dir()
        assert (result / "Basketball").is_dir()
        assert (result / "Diving").is_dir()
        assert len(list((result / "Archery").glob("*.avi"))) == 3

    def test_staging_dir_cleaned_up(self, tmp_path):
        """The UCF-101 staging directory should be removed after moving."""
        from vtsearch.datasets.downloader import video as vid_module

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        video_dir = data_dir / "video"
        zip_path = tmp_path / "UCF-101.zip"
        self._make_ucf101_zip(zip_path)

        with (
            patch.object(vid_module._core, "DATA_DIR", data_dir),
            patch.object(vid_module._core, "VIDEO_DIR", video_dir),
            patch(
                "vtsearch.datasets.downloader.core.download_file_with_progress",
                lambda url, dest, size, cb: shutil.copy(str(zip_path), str(dest)),
            ),
        ):
            vid_module.download_ucf101_full(on_progress=lambda *a: None)

        staging = data_dir / "UCF-101"
        assert not staging.exists(), "Staging directory should be removed after moving"

    def test_skips_if_already_present(self, tmp_path):
        """If ucf101_full/ already has videos, skip download entirely."""
        from vtsearch.datasets.downloader import video as vid_module

        data_dir = tmp_path / "data"
        video_dir = data_dir / "video"
        ucf_dir = video_dir / "ucf101_full" / "Archery"
        ucf_dir.mkdir(parents=True)
        (ucf_dir / "v_Archery_g01_c01.avi").write_bytes(b"RIFF" + b"\x00" * 20 + b"AVI ")

        download_called = []

        with (
            patch.object(vid_module._core, "DATA_DIR", data_dir),
            patch.object(vid_module._core, "VIDEO_DIR", video_dir),
            patch(
                "vtsearch.datasets.downloader.core.download_file_with_progress",
                lambda *a, **kw: download_called.append(True),
            ),
        ):
            result = vid_module.download_ucf101_full(on_progress=lambda *a: None)

        assert not download_called, "download should be skipped when cache exists"
        assert result.exists()


# ---------------------------------------------------------------------------
# KTH Actions
# ---------------------------------------------------------------------------


class TestDownloadKth:
    """Verify download_kth downloads and extracts per-action zips."""

    def _make_kth_zip(self, zip_path: Path, action: str) -> None:
        """Create a minimal KTH action zip with .avi files."""
        with zipfile.ZipFile(zip_path, "w") as zf:
            for person in range(1, 4):
                for scenario in range(1, 3):
                    fname = f"person{person:02d}_{action}_d{scenario}_uncomp.avi"
                    zf.writestr(fname, b"RIFF" + b"\x00" * 20 + b"AVI ")

    def test_extracts_into_category_dirs(self, tmp_path):
        """download_kth should create per-action dirs with .avi files."""
        from vtsearch.datasets.downloader import video as vid_module

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        video_dir = data_dir / "video"

        # Pre-create zip fixtures for each action.
        action_zips = {}
        for action in ("walking", "jogging", "running", "boxing", "handwaving", "handclapping"):
            zp = tmp_path / f"{action}.zip"
            self._make_kth_zip(zp, action)
            action_zips[action] = zp

        def fake_download(url, dest, size, cb):
            # Extract the action name from the URL.
            action = Path(url).stem
            shutil.copy(str(action_zips[action]), str(dest))

        with (
            patch.object(vid_module._core, "DATA_DIR", data_dir),
            patch.object(vid_module._core, "VIDEO_DIR", video_dir),
            patch.object(vid_module._core, "download_file_with_progress", fake_download),
        ):
            result = vid_module.download_kth(on_progress=lambda *a: None)

        assert result.exists()
        assert result.name == "kth"
        for action in ("walking", "jogging", "running", "boxing", "handwaving", "handclapping"):
            cat_dir = result / action
            assert cat_dir.is_dir(), f"Missing category dir: {action}"
            avi_files = list(cat_dir.glob("*.avi"))
            # 3 persons × 2 scenarios = 6 files per action
            assert len(avi_files) == 6, f"Expected 6 .avi files for {action}, got {len(avi_files)}"

    def test_skips_if_already_present(self, tmp_path):
        """If kth/ already has all action videos, skip download entirely."""
        from vtsearch.datasets.downloader import video as vid_module

        data_dir = tmp_path / "data"
        video_dir = data_dir / "video"

        # Populate ALL 6 actions so the early-exit check passes.
        for action in ("walking", "jogging", "running", "boxing", "handwaving", "handclapping"):
            cat_dir = video_dir / "kth" / action
            cat_dir.mkdir(parents=True)
            (cat_dir / f"person01_{action}_d1_uncomp.avi").write_bytes(
                b"RIFF" + b"\x00" * 20 + b"AVI "
            )

        download_called = []

        with (
            patch.object(vid_module._core, "DATA_DIR", data_dir),
            patch.object(vid_module._core, "VIDEO_DIR", video_dir),
            patch.object(
                vid_module._core,
                "download_file_with_progress",
                lambda *a, **kw: download_called.append(True),
            ),
        ):
            result = vid_module.download_kth(on_progress=lambda *a: None)

        assert not download_called, "download should be skipped when cache exists"
        assert result.exists()

    def test_skips_already_extracted_actions(self, tmp_path):
        """Actions already extracted should not be re-downloaded."""
        from vtsearch.datasets.downloader import video as vid_module

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        video_dir = data_dir / "video"

        # Pre-populate one action
        walking_dir = video_dir / "kth" / "walking"
        walking_dir.mkdir(parents=True)
        (walking_dir / "person01_walking_d1_uncomp.avi").write_bytes(
            b"RIFF" + b"\x00" * 20 + b"AVI "
        )

        downloaded_urls = []

        def fake_download(url, dest, size, cb):
            downloaded_urls.append(url)
            action = Path(url).stem
            zp = tmp_path / f"{action}.zip"
            self._make_kth_zip(zp, action)
            shutil.copy(str(zp), str(dest))

        with (
            patch.object(vid_module._core, "DATA_DIR", data_dir),
            patch.object(vid_module._core, "VIDEO_DIR", video_dir),
            patch.object(vid_module._core, "download_file_with_progress", fake_download),
        ):
            vid_module.download_kth(on_progress=lambda *a: None)

        # walking should NOT have been downloaded (already present)
        walking_urls = [u for u in downloaded_urls if "walking" in u]
        assert not walking_urls, "walking action should be skipped"
        # Other actions SHOULD have been downloaded
        assert len(downloaded_urls) == 5  # all except walking

    def test_temp_zips_cleaned_up(self, tmp_path):
        """Temp zip files should be removed after extraction."""
        from vtsearch.datasets.downloader import video as vid_module

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        video_dir = data_dir / "video"

        def fake_download(url, dest, size, cb):
            action = Path(url).stem
            zp = tmp_path / f"{action}.zip"
            self._make_kth_zip(zp, action)
            shutil.copy(str(zp), str(dest))

        with (
            patch.object(vid_module._core, "DATA_DIR", data_dir),
            patch.object(vid_module._core, "VIDEO_DIR", video_dir),
            patch.object(vid_module._core, "download_file_with_progress", fake_download),
        ):
            vid_module.download_kth(on_progress=lambda *a: None)

        leftover = [p for p in data_dir.iterdir() if p.name.startswith(".dl_")]
        assert not leftover, f"Temp zip files should be cleaned up: {leftover}"


# ---------------------------------------------------------------------------
# Demo dataset registration
# ---------------------------------------------------------------------------


class TestVideoDemoDatasetRegistration:
    """Verify new video demo datasets appear in the registry."""

    def test_hmdb51_variants_registered(self):
        """HMDB51 S/M/L/A variants should appear in demo_datasets."""
        from vtsearch.media.video.media_type import VideoMediaType

        mt = VideoMediaType()
        ids = [d.id for d in mt.demo_datasets]
        for variant in ("hmdb51_s", "hmdb51_m", "hmdb51_l", "hmdb51_a"):
            assert variant in ids, f"{variant} not found in demo_datasets"

    def test_ucf101_full_variants_registered(self):
        """UCF-101 Full S/M/L/A variants should appear in demo_datasets."""
        from vtsearch.media.video.media_type import VideoMediaType

        mt = VideoMediaType()
        ids = [d.id for d in mt.demo_datasets]
        for variant in ("ucf101_full_s", "ucf101_full_m", "ucf101_full_l", "ucf101_full_a"):
            assert variant in ids, f"{variant} not found in demo_datasets"

    def test_kth_variants_registered(self):
        """KTH Actions S/M/L/A variants should appear in demo_datasets."""
        from vtsearch.media.video.media_type import VideoMediaType

        mt = VideoMediaType()
        ids = [d.id for d in mt.demo_datasets]
        for variant in ("kth_s", "kth_m", "kth_l", "kth_a"):
            assert variant in ids, f"{variant} not found in demo_datasets"

    def test_hmdb51_has_51_categories(self):
        """HMDB51 demo datasets should reference all 51 action categories."""
        from vtsearch.media.video.media_type import VideoMediaType

        mt = VideoMediaType()
        hmdb = next(d for d in mt.demo_datasets if d.id == "hmdb51_a")
        assert len(hmdb.categories) == 51

    def test_ucf101_full_has_101_categories(self):
        """UCF-101 Full demo datasets should reference all 101 action categories."""
        from vtsearch.media.video.media_type import VideoMediaType

        mt = VideoMediaType()
        ucf = next(d for d in mt.demo_datasets if d.id == "ucf101_full_a")
        assert len(ucf.categories) == 101

    def test_kth_has_6_categories(self):
        """KTH Actions demo datasets should reference all 6 action categories."""
        from vtsearch.media.video.media_type import VideoMediaType

        mt = VideoMediaType()
        kth = next(d for d in mt.demo_datasets if d.id == "kth_a")
        assert len(kth.categories) == 6

    def test_source_dirs_registered(self):
        """New video sources should appear in the demo importer source dirs."""
        from vtsearch.datasets.importers import demo as demo_module

        # Force lazy init
        demo_module._source_directory("hmdb51")

        assert "hmdb51" in demo_module._SOURCE_DIRS
        assert "ucf101_full" in demo_module._SOURCE_DIRS
        assert "kth" in demo_module._SOURCE_DIRS


# ---------------------------------------------------------------------------
# load_demo_source — video (hmdb51, ucf101_full, kth)
# ---------------------------------------------------------------------------


class TestLoadDemoSourceHmdb51:
    """VideoMediaType.load_demo_source with source='hmdb51'."""

    def _make_mock_embedder(self):
        mock_emb = MagicMock()
        mock_emb.name = "xclip"
        mock_emb.media_type_id = "video"
        mock_emb._model = True
        mock_emb.embed_media = MagicMock(return_value=np.zeros(512))
        return mock_emb

    def test_hmdb51_source_populates_clips(self, tmp_path):
        """load_demo_source with source='hmdb51' fills the clips dict."""
        from vtsearch.datasets import downloader as dl_module
        from vtsearch.datasets import loader as loader_module
        from vtsearch.media.video.media_type import VideoMediaType

        fake_metadata = {
            "brush_hair/brush_hair_video_0.avi": {
                "category": "brush_hair",
                "path": tmp_path / "brush_hair_video_0.avi",
            },
            "cartwheel/cartwheel_video_0.avi": {
                "category": "cartwheel",
                "path": tmp_path / "cartwheel_video_0.avi",
            },
        }
        for meta in fake_metadata.values():
            meta["path"].write_bytes(b"RIFF" + b"\x00" * 40)

        mt = VideoMediaType()
        mt.load_media_data = MagicMock(return_value={"media_bytes": b"", "duration": 2.5})
        mock_emb = self._make_mock_embedder()
        clips: dict = {}

        with (
            patch.object(dl_module, "download_hmdb51", return_value=tmp_path),
            patch.object(loader_module, "load_video_metadata_from_folders", return_value=fake_metadata),
        ):
            mt.load_demo_source(
                source="hmdb51",
                categories=["brush_hair", "cartwheel"],
                slice_start=0,
                slice_end=10,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=mock_emb,
            )

        assert len(clips) == 2
        categories_seen = {c["category"] for c in clips.values()}
        assert categories_seen == {"brush_hair", "cartwheel"}


class TestLoadDemoSourceUcf101Full:
    """VideoMediaType.load_demo_source with source='ucf101_full'."""

    def _make_mock_embedder(self):
        mock_emb = MagicMock()
        mock_emb.name = "xclip"
        mock_emb.media_type_id = "video"
        mock_emb._model = True
        mock_emb.embed_media = MagicMock(return_value=np.zeros(512))
        return mock_emb

    def test_ucf101_full_source_populates_clips(self, tmp_path):
        """load_demo_source with source='ucf101_full' fills the clips dict."""
        from vtsearch.datasets import downloader as dl_module
        from vtsearch.datasets import loader as loader_module
        from vtsearch.media.video.media_type import VideoMediaType

        fake_metadata = {
            "Archery/v_Archery_g01_c01.avi": {
                "category": "Archery",
                "path": tmp_path / "v_Archery_g01_c01.avi",
            },
            "Diving/v_Diving_g01_c01.avi": {
                "category": "Diving",
                "path": tmp_path / "v_Diving_g01_c01.avi",
            },
        }
        for meta in fake_metadata.values():
            meta["path"].write_bytes(b"RIFF" + b"\x00" * 40)

        mt = VideoMediaType()
        mt.load_media_data = MagicMock(return_value={"media_bytes": b"", "duration": 7.0})
        mock_emb = self._make_mock_embedder()
        clips: dict = {}

        with (
            patch.object(dl_module, "download_ucf101_full", return_value=tmp_path),
            patch.object(loader_module, "load_video_metadata_from_folders", return_value=fake_metadata),
        ):
            mt.load_demo_source(
                source="ucf101_full",
                categories=["Archery", "Diving"],
                slice_start=0,
                slice_end=10,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=mock_emb,
            )

        assert len(clips) == 2
        categories_seen = {c["category"] for c in clips.values()}
        assert categories_seen == {"Archery", "Diving"}


class TestLoadDemoSourceKth:
    """VideoMediaType.load_demo_source with source='kth'."""

    def _make_mock_embedder(self):
        mock_emb = MagicMock()
        mock_emb.name = "xclip"
        mock_emb.media_type_id = "video"
        mock_emb._model = True
        mock_emb.embed_media = MagicMock(return_value=np.zeros(512))
        return mock_emb

    def test_kth_source_populates_clips(self, tmp_path):
        """load_demo_source with source='kth' fills the clips dict."""
        from vtsearch.datasets import downloader as dl_module
        from vtsearch.datasets import loader as loader_module
        from vtsearch.media.video.media_type import VideoMediaType

        fake_metadata = {
            "walking/person01_walking_d1_uncomp.avi": {
                "category": "walking",
                "path": tmp_path / "person01_walking_d1_uncomp.avi",
            },
            "boxing/person01_boxing_d1_uncomp.avi": {
                "category": "boxing",
                "path": tmp_path / "person01_boxing_d1_uncomp.avi",
            },
        }
        for meta in fake_metadata.values():
            meta["path"].write_bytes(b"RIFF" + b"\x00" * 40)

        mt = VideoMediaType()
        mt.load_media_data = MagicMock(return_value={"media_bytes": b"", "duration": 12.0})
        mock_emb = self._make_mock_embedder()
        clips: dict = {}

        with (
            patch.object(dl_module, "download_kth", return_value=tmp_path),
            patch.object(loader_module, "load_video_metadata_from_folders", return_value=fake_metadata),
        ):
            mt.load_demo_source(
                source="kth",
                categories=["walking", "boxing"],
                slice_start=0,
                slice_end=10,
                clips=clips,
                on_progress=lambda *a: None,
                embedder=mock_emb,
            )

        assert len(clips) == 2
        categories_seen = {c["category"] for c in clips.values()}
        assert categories_seen == {"walking", "boxing"}

    def test_unsupported_source_raises(self):
        """Non-existent sources still raise ValueError."""
        from vtsearch.media.video.media_type import VideoMediaType

        mt = VideoMediaType()
        import pytest

        with pytest.raises(ValueError, match="Unsupported video source"):
            mt.load_demo_source(
                source="unknown_source",
                categories=[],
                slice_start=0,
                slice_end=10,
                clips={},
                on_progress=lambda *a: None,
            )
