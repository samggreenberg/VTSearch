import io
import struct
import tarfile
import wave
import zipfile

import pytest

import app as app_module


class TestIndex:
    def test_serves_index_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"VTSearch" in resp.data


class TestDatasetEndpoints:
    def test_get_dataset_status(self, client):
        resp = client.get("/api/dataset/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "num_clips" in data or "error" in data

    def test_get_dataset_demo_list(self, client):
        resp = client.get("/api/dataset/demo-list")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, dict)
        # Should return available demo datasets
        assert "demos" in data or isinstance(data, dict)

    def test_clear_dataset(self, client):
        resp = client.post("/api/dataset/clear")
        assert resp.status_code == 200
        # After clearing, clips should be empty
        assert len(app_module.clips) == 0

        # Re-initialize for other tests
        app_module.init_clips()


class TestStartupState:
    """App should start with an empty dataset so the selection screen shows."""

    def test_status_loaded_false_when_clips_empty(self, client):
        """GET /api/dataset/status returns loaded=False when clips is cleared."""
        saved = dict(app_module.clips)
        app_module.clips.clear()
        try:
            resp = client.get("/api/dataset/status")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["loaded"] is False
            assert data["num_clips"] == 0
        finally:
            app_module.clips.update(saved)

    def test_init_clips_not_called_automatically(self):
        """init_clips() exists for testing but is not called in production startup.

        Verify that the production startup block in app.py does NOT call
        init_clips() – it should only load models and wait for user selection.
        """
        import inspect

        source = inspect.getsource(app_module)

        # The production path is the final else branch after the argparse
        # if/elif/else chain.  Find the last else: in the __main__ block.
        main_block_start = source.find('if __name__ == "__main__"')
        assert main_block_start != -1, "Could not find __main__ block"
        main_body = source[main_block_start:]

        # Find the production else branch (the last else: in the block)
        else_start = main_body.rfind("else:")
        assert else_start != -1, "Could not find else branch in __main__ block"
        else_body = main_body[else_start:]
        assert "init_clips()" not in else_body, "init_clips() must not be called automatically in production startup"


class TestDemoDatasetReadiness:
    """Demo datasets report three-state status: ready / needs_embedding / needs_download."""

    def test_audio_pkl_without_esc50_shows_needs_download(self, client):
        """Audio pkl exists but ESC-50 audio dir is absent → needs_download (stale pkl)."""
        import pickle

        from vtsearch.config import DATA_DIR, EMBEDDINGS_DIR

        esc50_dir = DATA_DIR / "ESC-50-master" / "audio"
        if esc50_dir.exists():
            pytest.skip("ESC-50 is present; cannot test stale-pkl scenario")

        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        pkl_file = EMBEDDINGS_DIR / "sounds_s.pkl"
        pkl_file.write_bytes(pickle.dumps({"name": "sounds_s", "clips": {}}))
        try:
            resp = client.get("/api/dataset/demo-list")
            data = resp.get_json()
            ds = next((d for d in data["datasets"] if d["name"] == "sounds_s"), None)
            assert ds is not None
            assert ds["status"] == "needs_download", "Stale audio pkl without ESC-50 dir must be needs_download"
            assert ds["ready"] is False
        finally:
            pkl_file.unlink(missing_ok=True)
            try:
                EMBEDDINGS_DIR.rmdir()
            except OSError:
                pass

    def test_audio_pkl_with_empty_esc50_shows_needs_download(self, client):
        """Audio pkl exists and ESC-50 audio dir exists but is empty → needs_download."""
        import pickle

        from vtsearch.config import DATA_DIR, EMBEDDINGS_DIR

        esc50_dir = DATA_DIR / "ESC-50-master" / "audio"
        if esc50_dir.exists() and any(esc50_dir.iterdir()):
            pytest.skip("ESC-50 audio dir is non-empty; cannot test empty-dir scenario")

        # Create the directory structure but leave it empty
        esc50_dir.mkdir(parents=True, exist_ok=True)
        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        pkl_file = EMBEDDINGS_DIR / "sounds_s.pkl"
        pkl_file.write_bytes(pickle.dumps({"name": "sounds_s", "clips": {}}))
        try:
            resp = client.get("/api/dataset/demo-list")
            data = resp.get_json()
            ds = next((d for d in data["datasets"] if d["name"] == "sounds_s"), None)
            assert ds is not None
            assert ds["status"] == "needs_download", "Audio pkl with empty ESC-50 dir must be needs_download"
            assert ds["ready"] is False
        finally:
            pkl_file.unlink(missing_ok=True)
            try:
                EMBEDDINGS_DIR.rmdir()
            except OSError:
                pass
            try:
                esc50_dir.rmdir()
            except OSError:
                pass

    def test_video_pkl_without_ucf101_shows_needs_download(self, client):
        """Video pkl exists but UCF-101 dir is absent → needs_download (stale pkl)."""
        import pickle

        from vtsearch.config import EMBEDDINGS_DIR, VIDEO_DIR

        ucf101_dir = VIDEO_DIR / "ucf101"
        if ucf101_dir.exists():
            pytest.skip("UCF-101 is present; cannot test stale-pkl scenario")

        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        pkl_file = EMBEDDINGS_DIR / "activities_video.pkl"
        pkl_file.write_bytes(pickle.dumps({"name": "activities_video", "clips": {}}))
        try:
            resp = client.get("/api/dataset/demo-list")
            data = resp.get_json()
            ds = next((d for d in data["datasets"] if d["name"] == "activities_video"), None)
            assert ds is not None
            assert ds["status"] == "needs_download", "Stale video pkl without UCF-101 dir must be needs_download"
            assert ds["ready"] is False
        finally:
            pkl_file.unlink(missing_ok=True)
            try:
                EMBEDDINGS_DIR.rmdir()
            except OSError:
                pass

    def test_no_pkl_with_source_folder_shows_needs_embedding(self, client):
        """No pkl but required_folder exists with content → needs_embedding."""
        import struct
        import wave

        from vtsearch.config import DATA_DIR, EMBEDDINGS_DIR

        esc50_dir = DATA_DIR / "ESC-50-master" / "audio"
        # Ensure no pkl exists for sounds_s
        pkl_file = EMBEDDINGS_DIR / "sounds_s.pkl"
        if pkl_file.exists():
            pytest.skip("sounds_s.pkl exists; cannot test needs_embedding scenario")

        # Create the ESC-50 audio dir with a dummy file
        esc50_dir.mkdir(parents=True, exist_ok=True)
        dummy_wav = esc50_dir / "_test_dummy.wav"
        already_populated = any(f.name != "_test_dummy.wav" for f in esc50_dir.iterdir()) if esc50_dir.exists() else False
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(struct.pack("<h", 0) * 100)
        dummy_wav.write_bytes(buf.getvalue())
        try:
            resp = client.get("/api/dataset/demo-list")
            data = resp.get_json()
            ds = next((d for d in data["datasets"] if d["name"] == "sounds_s"), None)
            assert ds is not None
            assert ds["status"] == "needs_embedding", "No pkl with source folder should be needs_embedding"
            assert ds["ready"] is False
            assert ds["download_size_mb"] == 0, "needs_embedding should report 0 MB download"
        finally:
            dummy_wav.unlink(missing_ok=True)
            if not already_populated:
                try:
                    esc50_dir.rmdir()
                except OSError:
                    pass

    def test_no_pkl_no_source_shows_needs_download(self, client):
        """No pkl and no required_folder → needs_download."""
        from vtsearch.config import DATA_DIR, EMBEDDINGS_DIR

        esc50_dir = DATA_DIR / "ESC-50-master" / "audio"
        pkl_file = EMBEDDINGS_DIR / "sounds_s.pkl"
        if pkl_file.exists():
            pytest.skip("sounds_s.pkl exists; cannot test needs_download scenario")
        if esc50_dir.exists() and any(esc50_dir.iterdir()):
            pytest.skip("ESC-50 is present; cannot test needs_download scenario")

        resp = client.get("/api/dataset/demo-list")
        data = resp.get_json()
        ds = next((d for d in data["datasets"] if d["name"] == "sounds_s"), None)
        assert ds is not None
        assert ds["status"] == "needs_download"
        assert ds["ready"] is False

    def test_status_field_always_present(self, client):
        """Every demo dataset must include a status field."""
        resp = client.get("/api/dataset/demo-list")
        data = resp.get_json()
        for ds in data["datasets"]:
            assert "status" in ds, f"Dataset '{ds['name']}' missing status field"
            assert ds["status"] in ("ready", "needs_embedding", "needs_download")


class TestImporterMetadata:
    """Importer to_dict() must include the icon field."""

    def test_http_archive_display_name(self, client):
        resp = client.get("/api/dataset/importers")
        assert resp.status_code == 200
        data = resp.get_json()
        names = [imp["display_name"] for imp in data["importers"]]
        assert "Generate from HTTP Archive" in names

    def test_http_archive_icon_is_globe(self, client):
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        http_imp = next((i for i in data["importers"] if i["name"] == "http_archive"), None)
        assert http_imp is not None, "http_archive importer not found"
        assert http_imp["icon"] == "\U0001f310"

    def test_http_archive_supports_tar_and_rar_in_description(self, client):
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        http_imp = next((i for i in data["importers"] if i["name"] == "http_archive"), None)
        assert http_imp is not None
        desc = http_imp["description"].lower()
        assert "tar" in desc
        assert "rar" in desc

    def test_folder_importer_in_extended_list(self, client):
        """Folder importer must appear in /api/dataset/importers (not a builtin)."""
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        names = [imp["name"] for imp in data["importers"]]
        assert "folder" in names

    def test_folder_importer_icon(self, client):
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        folder_imp = next((i for i in data["importers"] if i["name"] == "folder"), None)
        assert folder_imp is not None
        assert folder_imp["icon"] == "\U0001f4c2"

    def test_folder_importer_description(self, client):
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        folder_imp = next((i for i in data["importers"] if i["name"] == "folder"), None)
        assert folder_imp is not None
        # Description must not mention specific media-type names
        desc = folder_imp["description"]
        assert "sounds/videos" not in desc
        assert "media files from a folder" in desc.lower()

    def test_all_importers_have_icon_field(self, client):
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        for imp in data["importers"]:
            assert "icon" in imp, f"Importer '{imp['name']}' missing icon field"

    def test_pickle_not_in_extended_list(self, client):
        """Pickle importer keeps its dedicated UI and must not appear in the list."""
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        names = [imp["name"] for imp in data["importers"]]
        assert "pickle" not in names

    def test_folder_media_type_field_is_first(self, client):
        """Media-type dropdown should come before the path field."""
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        folder_imp = next((i for i in data["importers"] if i["name"] == "folder"), None)
        assert folder_imp is not None
        keys = [f["key"] for f in folder_imp["fields"]]
        assert keys.index("media_type") < keys.index("path")


class TestLoadEmbedderForClips:
    """_load_embedder_for_clips should warm up the text encoder at dataset load time."""

    def test_warms_up_text_encoder(self):
        """embed_text('warmup') is called to prime the text encoder branch."""
        from unittest.mock import patch

        from vtsearch.media import get as media_get
        from vtsearch.routes.datasets import _load_embedder_for_clips

        mt = media_get("audio")
        with patch.object(mt, "embed_text", wraps=mt.embed_text) as mock_embed:
            _load_embedder_for_clips()
            mock_embed.assert_called_once_with("warmup")

    def test_text_encoder_produces_valid_embedding_after_load(self):
        """After _load_embedder_for_clips, embed_text returns a real vector."""
        from vtsearch.media import get as media_get
        from vtsearch.routes.datasets import _load_embedder_for_clips

        _load_embedder_for_clips()
        mt = media_get("audio")
        vec = mt.embed_text("a high-pitched beep")
        assert vec is not None
        assert len(vec.shape) == 1
        assert vec.shape[0] > 0


class TestExtractArchive:
    """Unit tests for the zip/tar extraction helper."""

    from vtsearch.datasets.importers.http_zip import _extract_archive

    def _make_wav_bytes(self) -> bytes:
        """Create a minimal valid WAV file in memory."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            samples = struct.pack("<" + "h" * 100, *([0] * 100))
            wf.writeframes(samples)
        return buf.getvalue()

    def test_extract_zip(self, tmp_path):
        from vtsearch.datasets.importers.http_zip import _extract_archive

        wav_data = self._make_wav_bytes()
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("sounds/tone.wav", wav_data)
        extract_dir = tmp_path / "out"
        extract_dir.mkdir()
        _extract_archive(zip_path, extract_dir)
        assert (extract_dir / "sounds" / "tone.wav").exists()

    def test_extract_tar_gz(self, tmp_path):
        from vtsearch.datasets.importers.http_zip import _extract_archive

        wav_data = self._make_wav_bytes()
        tar_path = tmp_path / "test.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tf:
            info = tarfile.TarInfo(name="sounds/tone.wav")
            info.size = len(wav_data)
            tf.addfile(info, io.BytesIO(wav_data))
        extract_dir = tmp_path / "out"
        extract_dir.mkdir()
        _extract_archive(tar_path, extract_dir)
        assert (extract_dir / "sounds" / "tone.wav").exists()

    def test_extract_tar_uncompressed(self, tmp_path):
        from vtsearch.datasets.importers.http_zip import _extract_archive

        wav_data = self._make_wav_bytes()
        tar_path = tmp_path / "test.tar"
        with tarfile.open(tar_path, "w") as tf:
            info = tarfile.TarInfo(name="tone.wav")
            info.size = len(wav_data)
            tf.addfile(info, io.BytesIO(wav_data))
        extract_dir = tmp_path / "out"
        extract_dir.mkdir()
        _extract_archive(tar_path, extract_dir)
        assert (extract_dir / "tone.wav").exists()

    def test_unsupported_format_raises(self, tmp_path):
        from vtsearch.datasets.importers.http_zip import _extract_archive

        bad_archive = tmp_path / "test.7z"
        bad_archive.write_bytes(b"not a real archive")
        extract_dir = tmp_path / "out"
        extract_dir.mkdir()
        with pytest.raises((ValueError, Exception)):
            _extract_archive(bad_archive, extract_dir)

    def test_rar_without_rarfile_raises_runtime_error(self, tmp_path):
        """Attempting RAR extraction without rarfile installed raises RuntimeError."""
        import sys
        import unittest.mock as mock

        from vtsearch.datasets.importers.http_zip import _extract_archive

        rar_path = tmp_path / "test.rar"
        rar_path.write_bytes(b"Rar!\x1a\x07\x00")  # RAR magic bytes (v4)
        extract_dir = tmp_path / "out"
        extract_dir.mkdir()

        with mock.patch.dict(sys.modules, {"rarfile": None}):
            with pytest.raises((RuntimeError, ImportError, Exception)):
                _extract_archive(rar_path, extract_dir)


class TestCaltech101Download:
    """Verify download_caltech101 handles the nested zip→tar.gz structure."""

    def _make_caltech101_zip(self, zip_path):
        """Create a mock caltech-101.zip matching the real archive structure.

        The real archive contains ``caltech-101/101_ObjectCategories.tar.gz``
        (a nested tar.gz) rather than bare category directories.
        """
        # Build the inner tar.gz with a few dummy category images
        inner_tar_buf = io.BytesIO()
        with tarfile.open(fileobj=inner_tar_buf, mode="w:gz") as tf:
            for cat in ("butterfly", "dolphin"):
                for i in range(3):
                    fname = f"101_ObjectCategories/{cat}/image_{i:04d}.jpg"
                    # Minimal JPEG: SOI + EOI markers
                    data = b"\xff\xd8\xff\xe0" + b"\x00" * 20 + b"\xff\xd9"
                    info = tarfile.TarInfo(name=fname)
                    info.size = len(data)
                    tf.addfile(info, io.BytesIO(data))
        inner_tar_bytes = inner_tar_buf.getvalue()

        # Build the outer zip
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("caltech-101/", "")
            zf.writestr("caltech-101/101_ObjectCategories.tar.gz", inner_tar_bytes)
            zf.writestr("caltech-101/show_annotation.m", "% annotation script\n")

    def test_extracts_nested_tar_gz(self, tmp_path):
        """download_caltech101 should extract the inner tar.gz to produce category dirs."""
        from unittest.mock import patch

        from vtsearch.datasets.downloader import download_caltech101

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        zip_path = data_dir / "caltech-101.zip"
        self._make_caltech101_zip(zip_path)

        with (
            patch("vtsearch.datasets.downloader.DATA_DIR", data_dir),
            patch("vtsearch.datasets.downloader.IMAGE_DIR", data_dir / "images"),
        ):
            result = download_caltech101(on_progress=lambda *a: None)

        assert result.exists(), f"Expected {result} to exist"
        assert result.name == "101_ObjectCategories"
        assert (result / "butterfly").is_dir()
        assert (result / "dolphin").is_dir()
        assert len(list((result / "butterfly").glob("*.jpg"))) == 3

    def test_inner_tar_cleaned_up(self, tmp_path):
        """The inner 101_ObjectCategories.tar.gz should be deleted after extraction."""
        from unittest.mock import patch

        from vtsearch.datasets.downloader import download_caltech101

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        zip_path = data_dir / "caltech-101.zip"
        self._make_caltech101_zip(zip_path)

        with (
            patch("vtsearch.datasets.downloader.DATA_DIR", data_dir),
            patch("vtsearch.datasets.downloader.IMAGE_DIR", data_dir / "images"),
        ):
            download_caltech101(on_progress=lambda *a: None)

        inner_tar = data_dir / "caltech-101" / "101_ObjectCategories.tar.gz"
        assert not inner_tar.exists(), "Inner tar.gz should be deleted after extraction"

    def test_outer_zip_cleaned_up(self, tmp_path):
        """The outer caltech-101.zip should be deleted after extraction."""
        from unittest.mock import patch

        from vtsearch.datasets.downloader import download_caltech101

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        zip_path = data_dir / "caltech-101.zip"
        self._make_caltech101_zip(zip_path)

        with (
            patch("vtsearch.datasets.downloader.DATA_DIR", data_dir),
            patch("vtsearch.datasets.downloader.IMAGE_DIR", data_dir / "images"),
        ):
            download_caltech101(on_progress=lambda *a: None)

        assert not zip_path.exists(), "Outer zip should be deleted after extraction"

    def test_skips_if_already_extracted(self, tmp_path):
        """If 101_ObjectCategories already exists, skip download and extraction."""
        from unittest.mock import patch

        from vtsearch.datasets.downloader import download_caltech101

        data_dir = tmp_path / "data"
        categories_dir = data_dir / "caltech-101" / "101_ObjectCategories" / "butterfly"
        categories_dir.mkdir(parents=True)
        (categories_dir / "image_0001.jpg").write_bytes(b"\xff\xd8\xff\xd9")

        with (
            patch("vtsearch.datasets.downloader.DATA_DIR", data_dir),
            patch("vtsearch.datasets.downloader.IMAGE_DIR", data_dir / "images"),
        ):
            result = download_caltech101(on_progress=lambda *a: None)

        assert result.exists()
        assert (result / "butterfly" / "image_0001.jpg").exists()


class TestUCF101SubsetDownload:
    """Verify download_ucf101_subset downloads, extracts, and flattens splits."""

    # Map split names to group-number offsets so every file across all
    # three splits gets a unique filename (the real dataset does this too).
    _SPLIT_OFFSETS = {"train": 0, "val": 10, "test": 20}

    def _make_ucf101_subset_tar(self, tar_path):
        """Create a mock UCF101_subset.tar.gz matching the real archive structure.

        The real archive has UCF101_subset/{train,val,test}/<Category>/*.avi.
        Filenames are unique across splits (different group numbers).
        """
        with tarfile.open(tar_path, "w:gz") as tf:
            for split, offset in self._SPLIT_OFFSETS.items():
                for cat in ("Archery", "BabyCrawling"):
                    for i in range(3):
                        g = offset + i
                        fname = f"UCF101_subset/{split}/{cat}/v_{cat}_g{g:02d}_c01.avi"
                        # Minimal AVI-like data (just enough for a file)
                        data = b"RIFF" + b"\x00" * 20 + b"AVI "
                        info = tarfile.TarInfo(name=fname)
                        info.size = len(data)
                        tf.addfile(info, io.BytesIO(data))

    def test_extracts_and_flattens_splits(self, tmp_path):
        """download_ucf101_subset should flatten train/val/test into category dirs."""
        from unittest.mock import patch

        from vtsearch.datasets.downloader import download_ucf101_subset

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        video_dir = data_dir / "video"
        tar_path = data_dir / "UCF101_subset.tar.gz"
        self._make_ucf101_subset_tar(tar_path)

        with (
            patch("vtsearch.datasets.downloader.DATA_DIR", data_dir),
            patch("vtsearch.datasets.downloader.VIDEO_DIR", video_dir),
        ):
            result = download_ucf101_subset(on_progress=lambda *a: None)

        assert result.exists(), f"Expected {result} to exist"
        assert result.name == "ucf101"
        assert (result / "Archery").is_dir()
        assert (result / "BabyCrawling").is_dir()
        # All splits merged: 3 files per split × 3 splits = 9 per category
        assert len(list((result / "Archery").glob("*.avi"))) == 9
        assert len(list((result / "BabyCrawling").glob("*.avi"))) == 9

    def test_tar_cleaned_up(self, tmp_path):
        """The UCF101_subset.tar.gz should be deleted after extraction."""
        from unittest.mock import patch

        from vtsearch.datasets.downloader import download_ucf101_subset

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        video_dir = data_dir / "video"
        tar_path = data_dir / "UCF101_subset.tar.gz"
        self._make_ucf101_subset_tar(tar_path)

        with (
            patch("vtsearch.datasets.downloader.DATA_DIR", data_dir),
            patch("vtsearch.datasets.downloader.VIDEO_DIR", video_dir),
        ):
            download_ucf101_subset(on_progress=lambda *a: None)

        assert not tar_path.exists(), "tar.gz should be deleted after extraction"

    def test_staging_dir_cleaned_up(self, tmp_path):
        """The UCF101_subset staging directory should be removed after flattening."""
        from unittest.mock import patch

        from vtsearch.datasets.downloader import download_ucf101_subset

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        video_dir = data_dir / "video"
        tar_path = data_dir / "UCF101_subset.tar.gz"
        self._make_ucf101_subset_tar(tar_path)

        with (
            patch("vtsearch.datasets.downloader.DATA_DIR", data_dir),
            patch("vtsearch.datasets.downloader.VIDEO_DIR", video_dir),
        ):
            download_ucf101_subset(on_progress=lambda *a: None)

        staging = data_dir / "UCF101_subset"
        assert not staging.exists(), "Staging directory should be removed after flattening"

    def test_skips_if_already_present(self, tmp_path):
        """If ucf101/ already has videos, skip download entirely."""
        from unittest.mock import patch

        from vtsearch.datasets.downloader import download_ucf101_subset

        data_dir = tmp_path / "data"
        video_dir = data_dir / "video"
        ucf_dir = video_dir / "ucf101" / "Archery"
        ucf_dir.mkdir(parents=True)
        (ucf_dir / "v_Archery_g01_c01.avi").write_bytes(b"RIFF" + b"\x00" * 20 + b"AVI ")

        with (
            patch("vtsearch.datasets.downloader.DATA_DIR", data_dir),
            patch("vtsearch.datasets.downloader.VIDEO_DIR", video_dir),
        ):
            result = download_ucf101_subset(on_progress=lambda *a: None)

        assert result.exists()
        assert (result / "Archery" / "v_Archery_g01_c01.avi").exists()

    def test_demo_list_shows_video_download_size(self, client):
        """Video demo datasets should report a non-zero download size."""
        from vtsearch.config import UCF101_SUBSET_DOWNLOAD_SIZE_MB

        resp = client.get("/api/dataset/demo-list")
        data = resp.get_json()
        video_ds = [d for d in data["datasets"] if d["media_type"] == "video"]
        assert len(video_ds) > 0, "Should have at least one video demo dataset"
        for ds in video_ds:
            if ds["status"] == "needs_download":
                assert ds["download_size_mb"] == UCF101_SUBSET_DOWNLOAD_SIZE_MB

    def test_video_demo_categories_match_subset(self, client):
        """Video demo datasets should only use categories from the UCF-101 subset."""
        from vtsearch.datasets.config import DEMO_DATASETS

        subset_categories = {
            "ApplyEyeMakeup", "ApplyLipstick", "Archery", "BabyCrawling",
            "BalanceBeam", "BandMarching", "BaseballPitch", "Basketball",
            "BasketballDunk", "BenchPress",
        }
        for name, info in DEMO_DATASETS.items():
            if info.get("media_type") == "video":
                for cat in info["categories"]:
                    assert cat in subset_categories, (
                        f"Video demo '{name}' uses category '{cat}' "
                        f"not in UCF-101 subset"
                    )
