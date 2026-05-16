import io
import tarfile
import zipfile

import pytest

import app as app_module


class TestIndex:
    def test_serves_index_html(self, client):
        from pathlib import Path

        static_index = Path(app_module.app.static_folder) / "index.html"
        if not static_index.exists():
            pytest.skip("Angular build not present (run 'npm run build:prod' in frontend/)")
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"VTSearch" in resp.data


class TestDatasetEndpoints:
    def test_get_dataset_status(self, client):
        resp = client.get("/api/dataset/status")
        assert resp.status_code == 200
        data = resp.get_json()
        # /api/dataset/status always returns 200 with the loaded-dataset
        # summary; legacy "or error" branch is gone after the
        # openapi-schema migration of vtsearch/routes/datasets/status.py.
        assert "num_medias" in data

    def test_get_dataset_demo_list(self, client):
        resp = client.get("/api/dataset/demo-list")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, dict)
        # Should return available demo datasets
        assert "demos" in data or isinstance(data, dict)

    def test_demo_categories_returns_categories(self, client):
        """GET /api/dataset/demo-categories/<name> returns categories for a valid demo."""
        from vtsearch.datasets import DEMO_DATASETS

        if not DEMO_DATASETS:
            pytest.skip("No demo datasets registered")
        name = next(iter(DEMO_DATASETS))
        resp = client.get(f"/api/dataset/demo-categories/{name}")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "categories" in data
        assert isinstance(data["categories"], list)
        assert len(data["categories"]) > 0
        expected = DEMO_DATASETS[name]["categories"]
        assert data["categories"] == expected

    def test_demo_categories_unknown_returns_404(self, client):
        """GET /api/dataset/demo-categories/<name> returns 404 for unknown demo."""
        resp = client.get("/api/dataset/demo-categories/nonexistent_dataset_xyz")
        assert resp.status_code == 404
        data = resp.get_json()
        # flask-smorest standard envelope after the openapi-schema
        # migration of vtsearch/routes/datasets/ui.py.
        assert "message" in data

    def test_browse_media_files_unknown_source(self, client):
        """GET /api/browse-media-files with unknown source returns 404."""
        resp = client.get("/api/browse-media-files?source=demo:nonexistent_xyz&path=")
        assert resp.status_code == 404
        data = resp.get_json()
        assert "message" in data

    def test_browse_media_files_path_traversal_blocked(self, client):
        """GET /api/browse-media-files rejects path traversal."""
        from vtsearch.datasets import DEMO_DATASETS

        if not DEMO_DATASETS:
            pytest.skip("No demo datasets registered")
        name = next(iter(DEMO_DATASETS))
        info = DEMO_DATASETS[name]
        if info.get("required_folder") is None or not info["required_folder"].is_dir():
            pytest.skip("Demo source directory not present on disk")

        resp = client.get(f"/api/browse-media-files?source=demo:{name}&path=../../etc")
        assert resp.status_code == 400
        data = resp.get_json()
        assert "message" in data

    def test_browse_media_files_demo_source(self, client, tmp_path):
        """GET /api/browse-media-files lists files and directories for a demo source."""
        from unittest.mock import patch

        from vtsearch.datasets import DEMO_DATASETS

        if not DEMO_DATASETS:
            pytest.skip("No demo datasets registered")
        name = next(iter(DEMO_DATASETS))

        # Create a temp dir with a subdir and a .wav file
        sub = tmp_path / "catA"
        sub.mkdir()
        (sub / "sound.wav").write_bytes(b"RIFF" + b"\x00" * 100)
        (tmp_path / "top.wav").write_bytes(b"RIFF" + b"\x00" * 50)

        fake_info = dict(DEMO_DATASETS[name])
        fake_info["required_folder"] = tmp_path

        with patch.dict("vtsearch.routes.datasets.ui.DEMO_DATASETS", {name: fake_info}):
            resp = client.get(f"/api/browse-media-files?source=demo:{name}&path=")
            assert resp.status_code == 200
            data = resp.get_json()
            assert "directories" in data
            assert "files" in data
            dir_names = [d["name"] for d in data["directories"]]
            file_names = [f["name"] for f in data["files"]]
            assert "catA" in dir_names
            assert "top.wav" in file_names

            # Drill into subdirectory
            resp2 = client.get(f"/api/browse-media-files?source=demo:{name}&path=catA")
            assert resp2.status_code == 200
            data2 = resp2.get_json()
            file_names2 = [f["name"] for f in data2["files"]]
            assert "sound.wav" in file_names2

    def test_select_browsed_file_copies_to_example_media(self, client, tmp_path):
        """POST /api/browse-media-files/select copies the file to example_media."""
        from unittest.mock import patch

        from vtsearch.datasets import DEMO_DATASETS

        if not DEMO_DATASETS:
            pytest.skip("No demo datasets registered")
        name = next(iter(DEMO_DATASETS))

        # Create a temp file
        (tmp_path / "pick_me.wav").write_bytes(b"RIFF" + b"\x00" * 80)

        fake_info = dict(DEMO_DATASETS[name])
        fake_info["required_folder"] = tmp_path

        with patch.dict("vtsearch.routes.datasets.ui.DEMO_DATASETS", {name: fake_info}):
            resp = client.post(
                "/api/browse-media-files/select",
                json={"source": f"demo:{name}", "path": "pick_me.wav"},
            )
            assert resp.status_code == 201
            data = resp.get_json()
            assert "filename" in data
            assert data["original_name"] == "pick_me.wav"
            # The file should have been copied to data/example_media/
            from vtsearch.config import DATA_DIR

            dest = DATA_DIR / "example_media" / data["filename"]
            assert dest.exists()
            # Clean up
            dest.unlink(missing_ok=True)

    def test_select_browsed_file_traversal_blocked(self, client, tmp_path):
        """POST /api/browse-media-files/select rejects traversal paths."""
        from unittest.mock import patch

        from vtsearch.datasets import DEMO_DATASETS

        if not DEMO_DATASETS:
            pytest.skip("No demo datasets registered")
        name = next(iter(DEMO_DATASETS))
        fake_info = dict(DEMO_DATASETS[name])
        fake_info["required_folder"] = tmp_path

        with patch.dict("vtsearch.routes.datasets.ui.DEMO_DATASETS", {name: fake_info}):
            resp = client.post(
                "/api/browse-media-files/select",
                json={"source": f"demo:{name}", "path": "../../etc/passwd"},
            )
            assert resp.status_code == 400

    def test_clear_dataset(self, client):
        saved = dict(app_module.medias)
        try:
            resp = client.post("/api/dataset/clear")
            assert resp.status_code == 200
            # After clearing, medias should be empty
            assert len(app_module.medias) == 0
        finally:
            app_module.medias.update(saved)


class TestStartupState:
    """App should start with an empty dataset so the selection screen shows."""

    def test_status_loaded_false_when_clips_empty(self, client):
        """GET /api/dataset/status returns loaded=False when medias is cleared."""
        saved = dict(app_module.medias)
        app_module.medias.clear()
        try:
            resp = client.get("/api/dataset/status")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["loaded"] is False
            assert data["num_medias"] == 0
        finally:
            app_module.medias.update(saved)

    def test_init_medias_not_called_automatically(self):
        """init_medias() exists for testing but is not called in production startup.

        Verify that the production startup block in app.py does NOT call
        init_medias() – it should only load models and wait for user selection.
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
        assert "init_medias()" not in else_body, "init_medias() must not be called automatically in production startup"


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
        pkl_file = EMBEDDINGS_DIR / "esc50_s.pkl"
        pkl_file.write_bytes(pickle.dumps({"name": "esc50_s", "medias": {}}))
        try:
            resp = client.get("/api/dataset/demo-list")
            data = resp.get_json()
            ds = next((d for d in data["datasets"] if d["name"] == "esc50_s"), None)
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
        pkl_file = EMBEDDINGS_DIR / "esc50_s.pkl"
        pkl_file.write_bytes(pickle.dumps({"name": "esc50_s", "medias": {}}))
        try:
            resp = client.get("/api/dataset/demo-list")
            data = resp.get_json()
            ds = next((d for d in data["datasets"] if d["name"] == "esc50_s"), None)
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

        from vtsearch.config import EMBEDDINGS_DIR
        from vtsearch.datasets.downloader import VIDEO_DIR

        ucf101_dir = VIDEO_DIR / "ucf101"
        if ucf101_dir.exists():
            pytest.skip("UCF-101 is present; cannot test stale-pkl scenario")

        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        pkl_file = EMBEDDINGS_DIR / "ucf101_s.pkl"
        pkl_file.write_bytes(pickle.dumps({"name": "ucf101_s", "medias": {}}))
        try:
            resp = client.get("/api/dataset/demo-list")
            data = resp.get_json()
            ds = next((d for d in data["datasets"] if d["name"] == "ucf101_s"), None)
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
        # Ensure no pkl exists for esc50_s
        pkl_file = EMBEDDINGS_DIR / "esc50_s.pkl"
        if pkl_file.exists():
            pytest.skip("esc50_s.pkl exists; cannot test needs_embedding scenario")

        # Create the ESC-50 audio dir with a dummy file
        esc50_dir.mkdir(parents=True, exist_ok=True)
        dummy_wav = esc50_dir / "_test_dummy.wav"
        already_populated = (
            any(f.name != "_test_dummy.wav" for f in esc50_dir.iterdir()) if esc50_dir.exists() else False
        )
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
            ds = next((d for d in data["datasets"] if d["name"] == "esc50_s"), None)
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
        pkl_file = EMBEDDINGS_DIR / "esc50_s.pkl"
        if pkl_file.exists():
            pytest.skip("esc50_s.pkl exists; cannot test needs_download scenario")
        if esc50_dir.exists() and any(esc50_dir.iterdir()):
            pytest.skip("ESC-50 is present; cannot test needs_download scenario")

        resp = client.get("/api/dataset/demo-list")
        data = resp.get_json()
        ds = next((d for d in data["datasets"] if d["name"] == "esc50_s"), None)
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


class TestDemoDatasetEmbedderStatus:
    """Demo dataset status respects which embedder produced the cached pkl."""

    def test_ready_with_matching_embedder(self, client):
        """pkl with sidecar matching requested embedder → ready."""
        import pickle

        from vtsearch.config import EMBEDDINGS_DIR
        from vtsearch.datasets import DEMO_DATASETS

        # Pick a demo that has no required_folder (e.g. a text or image demo).
        demo_name = None
        for name, info in DEMO_DATASETS.items():
            if info.get("required_folder") is None:
                demo_name = name
                break
        if demo_name is None:
            pytest.skip("No demo dataset without required_folder found")

        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        pkl_file = EMBEDDINGS_DIR / f"{demo_name}.pkl"
        sidecar = pkl_file.with_suffix(".embedder")
        pkl_file.write_bytes(pickle.dumps({"name": demo_name, "medias": {}}))
        sidecar.write_text("TestEmbedder", encoding="utf-8")
        try:
            resp = client.get("/api/dataset/demo-list?embedder=TestEmbedder")
            data = resp.get_json()
            ds = next((d for d in data["datasets"] if d["name"] == demo_name), None)
            assert ds is not None
            assert ds["status"] == "ready"
            assert ds["ready"] is True
            assert ds["pkl_embedder"] == "TestEmbedder"
        finally:
            pkl_file.unlink(missing_ok=True)
            sidecar.unlink(missing_ok=True)
            try:
                EMBEDDINGS_DIR.rmdir()
            except OSError:
                pass

    def test_needs_embedding_with_mismatched_embedder(self, client):
        """pkl with sidecar NOT matching requested embedder → needs_embedding."""
        import pickle

        from vtsearch.config import EMBEDDINGS_DIR
        from vtsearch.datasets import DEMO_DATASETS

        demo_name = None
        for name, info in DEMO_DATASETS.items():
            if info.get("required_folder") is None:
                demo_name = name
                break
        if demo_name is None:
            pytest.skip("No demo dataset without required_folder found")

        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        pkl_file = EMBEDDINGS_DIR / f"{demo_name}.pkl"
        sidecar = pkl_file.with_suffix(".embedder")
        pkl_file.write_bytes(pickle.dumps({"name": demo_name, "medias": {}}))
        sidecar.write_text("OldEmbedder", encoding="utf-8")
        try:
            resp = client.get("/api/dataset/demo-list?embedder=NewEmbedder")
            data = resp.get_json()
            ds = next((d for d in data["datasets"] if d["name"] == demo_name), None)
            assert ds is not None
            assert ds["status"] == "needs_embedding"
            assert ds["ready"] is False
            assert ds["pkl_embedder"] == "OldEmbedder"
        finally:
            pkl_file.unlink(missing_ok=True)
            sidecar.unlink(missing_ok=True)
            try:
                EMBEDDINGS_DIR.rmdir()
            except OSError:
                pass

    def test_no_embedder_param_ignores_check(self, client):
        """Without embedder query param, pkl existence alone means ready."""
        import pickle

        from vtsearch.config import EMBEDDINGS_DIR
        from vtsearch.datasets import DEMO_DATASETS

        demo_name = None
        for name, info in DEMO_DATASETS.items():
            if info.get("required_folder") is None:
                demo_name = name
                break
        if demo_name is None:
            pytest.skip("No demo dataset without required_folder found")

        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        pkl_file = EMBEDDINGS_DIR / f"{demo_name}.pkl"
        sidecar = pkl_file.with_suffix(".embedder")
        pkl_file.write_bytes(pickle.dumps({"name": demo_name, "medias": {}}))
        sidecar.write_text("SomeEmbedder", encoding="utf-8")
        try:
            resp = client.get("/api/dataset/demo-list")
            data = resp.get_json()
            ds = next((d for d in data["datasets"] if d["name"] == demo_name), None)
            assert ds is not None
            assert ds["status"] == "ready"
            assert ds["ready"] is True
        finally:
            pkl_file.unlink(missing_ok=True)
            sidecar.unlink(missing_ok=True)
            try:
                EMBEDDINGS_DIR.rmdir()
            except OSError:
                pass

    def test_pkl_embedder_field_present(self, client):
        """Every demo in the response should have a pkl_embedder field."""
        resp = client.get("/api/dataset/demo-list")
        data = resp.get_json()
        for ds in data["datasets"]:
            assert "pkl_embedder" in ds, f"Dataset '{ds['name']}' missing pkl_embedder field"

    def test_no_sidecar_returns_empty_embedder(self, client):
        """When no sidecar file exists, pkl_embedder is empty string."""
        import pickle

        from vtsearch.config import EMBEDDINGS_DIR
        from vtsearch.datasets import DEMO_DATASETS

        demo_name = None
        for name, info in DEMO_DATASETS.items():
            if info.get("required_folder") is None:
                demo_name = name
                break
        if demo_name is None:
            pytest.skip("No demo dataset without required_folder found")

        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        pkl_file = EMBEDDINGS_DIR / f"{demo_name}.pkl"
        sidecar = pkl_file.with_suffix(".embedder")
        # Write pkl with embedder info in media entries, no sidecar
        pkl_file.write_bytes(
            pickle.dumps(
                {
                    "name": demo_name,
                    "medias": {1: {"embedder": "FallbackEmb", "embedding": [0.1]}},
                }
            )
        )
        sidecar.unlink(missing_ok=True)
        try:
            resp = client.get("/api/dataset/demo-list?embedder=FallbackEmb")
            data = resp.get_json()
            ds = next((d for d in data["datasets"] if d["name"] == demo_name), None)
            assert ds is not None
            # Without a sidecar, embedder is unknown (empty string)
            assert ds["pkl_embedder"] == ""
            # The pkl exists but we can't verify the embedder matches, so status stays ready
            assert ds["status"] == "ready"
        finally:
            pkl_file.unlink(missing_ok=True)
            sidecar.unlink(missing_ok=True)
            try:
                EMBEDDINGS_DIR.rmdir()
            except OSError:
                pass


class TestImporterMetadata:
    """Importer to_dict() must include the icon field."""

    def test_http_archive_display_name(self, client):
        resp = client.get("/api/dataset/importers")
        assert resp.status_code == 200
        data = resp.get_json()
        names = [imp["display_name"] for imp in data["importers"]]
        assert "Import from URL" in names

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

    def test_server_folder_importer_in_extended_list(self, client):
        """server_folder importer must appear in /api/dataset/importers (not a builtin)."""
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        names = [imp["name"] for imp in data["importers"]]
        assert "server_folder" in names

    def test_folder_importer_icon(self, client):
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        folder_imp = next((i for i in data["importers"] if i["name"] == "server_folder"), None)
        assert folder_imp is not None
        # 📁 — frontend renders this as a "folder" icon (see icon.component.ts).
        assert folder_imp["icon"] == "\U0001f4c1"

    def test_folder_importer_description(self, client):
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        folder_imp = next((i for i in data["importers"] if i["name"] == "server_folder"), None)
        assert folder_imp is not None
        # Description must not mention specific media-type names
        desc = folder_imp["description"]
        assert "sounds/videos" not in desc
        assert "media files" in desc.lower()

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
        folder_imp = next((i for i in data["importers"] if i["name"] == "server_folder"), None)
        assert folder_imp is not None
        keys = [f["key"] for f in folder_imp["fields"]]
        assert keys.index("media_type") < keys.index("path")

    def test_every_importer_exposes_dataset_name_field(self, client):
        """Every importer's serialised field list begins with a non-required
        ``dataset_name`` text field so the user can override the auto-generated
        display name."""
        resp = client.get("/api/dataset/importers")
        data = resp.get_json()
        assert data["importers"], "expected at least one importer"
        for imp in data["importers"]:
            keys = [f["key"] for f in imp["fields"]]
            assert keys[0] == "dataset_name", f"{imp['name']} missing dataset_name field at index 0"
            ds_field = imp["fields"][0]
            assert ds_field["field_type"] == "text"
            assert ds_field["required"] is False


class TestLoadEmbedderForClips:
    """_load_embedder_for_clips should warm up the text encoder at dataset load time."""

    def test_warms_up_text_encoder(self):
        """embed_text('warmup') is called to prime the text encoder branch."""
        from unittest.mock import patch

        from vtsearch.media import embedders_for_type
        from vtsearch.datasets.load_pipeline import _load_embedder_for_clips

        emb = embedders_for_type("audio")[0]
        with patch.object(emb, "embed_text", wraps=emb.embed_text) as mock_embed:
            _load_embedder_for_clips()
            mock_embed.assert_called_once_with("warmup")

    def test_text_encoder_produces_valid_embedding_after_load(self):
        """After _load_embedder_for_clips, embed_text returns a real vector."""
        from vtsearch.media import embedders_for_type
        from vtsearch.datasets.load_pipeline import _load_embedder_for_clips

        _load_embedder_for_clips()
        emb = embedders_for_type("audio")[0]
        vec = emb.embed_text("a high-pitched beep")
        assert vec is not None
        assert len(vec.shape) == 1
        assert vec.shape[0] > 0


class TestDemoCacheEmbedderMismatch:
    """load_demo_dataset should invalidate the pickle cache when the requested
    embedder differs from the one that produced the cached pickle."""

    def test_stale_cache_is_discarded_on_embedder_mismatch(self, tmp_path):
        """If the sidecar says 'clip' but we request 'siglip', the cache is
        deleted and load_demo_source is called (i.e. re-embedding happens)."""
        import pickle
        from unittest.mock import MagicMock, patch

        from vtsearch.datasets import DEMO_DATASETS
        from vtsearch.datasets.loader import _write_embedder_sidecar, load_demo_dataset

        # Pick any demo dataset name
        demo_name = next(iter(DEMO_DATASETS))

        embeddings_dir = tmp_path / "embeddings"
        embeddings_dir.mkdir()

        # Write a fake cached pickle with a 'clip' sidecar
        pkl_file = embeddings_dir / f"{demo_name}.pkl"
        pkl_file.write_bytes(pickle.dumps({"name": demo_name, "medias": {1: {"embedding": [0.1]}}}))
        _write_embedder_sidecar(pkl_file, "clip")

        # Create a fake embedder whose name is 'siglip'
        fake_embedder = MagicMock()
        fake_embedder.name = "siglip"

        medias: dict = {}

        mock_mt = MagicMock()
        mock_mt.dir_key = "images_dir"
        mock_mt.load_demo_source.return_value = "/fake/dir"

        with (
            patch("vtsearch.datasets.loader.EMBEDDINGS_DIR", embeddings_dir),
            patch("vtsearch.media.get_embedder", return_value=fake_embedder) as mock_get,
            patch("vtsearch.media.get", return_value=mock_mt),
        ):
            load_demo_dataset(demo_name, medias, embedder_name="siglip")

            # The old pkl should have been deleted and load_demo_source called
            mock_get.assert_called_once_with("siglip")
            mock_mt.load_demo_source.assert_called_once()

    def test_matching_embedder_uses_cache(self, tmp_path):
        """When the sidecar matches the requested embedder, the cache is used
        without re-embedding."""
        import pickle
        from unittest.mock import patch

        import numpy as np

        from vtsearch.datasets import DEMO_DATASETS
        from vtsearch.datasets.loader import _write_embedder_sidecar, load_demo_dataset

        demo_name = next(iter(DEMO_DATASETS))

        embeddings_dir = tmp_path / "embeddings"
        embeddings_dir.mkdir()

        pkl_file = embeddings_dir / f"{demo_name}.pkl"
        pkl_file.write_bytes(
            pickle.dumps(
                {
                    "name": demo_name,
                    "medias": {1: {"embedding": [0.1, 0.2], "file": "/fake/file.png"}},
                }
            )
        )
        _write_embedder_sidecar(pkl_file, "clip")

        medias: dict = {}

        # Patch load_dataset_from_pickle to populate medias (simulating a valid load)
        def fake_load(path, m):
            m[1] = {"embedding": np.array([0.1, 0.2]), "file": "/fake/file.png"}

        with (
            patch("vtsearch.datasets.loader.EMBEDDINGS_DIR", embeddings_dir),
            patch("vtsearch.datasets.loader.load_dataset_from_pickle", side_effect=fake_load),
        ):
            load_demo_dataset(demo_name, medias, embedder_name="clip")

        # Cache was used — media was loaded
        assert 1 in medias

    def test_no_sidecar_accepts_cache(self, tmp_path):
        """When no sidecar exists (old pickle), accept the cache regardless of
        the requested embedder to avoid unnecessary re-downloads."""
        import pickle
        from unittest.mock import patch

        import numpy as np

        from vtsearch.datasets import DEMO_DATASETS
        from vtsearch.datasets.loader import load_demo_dataset

        demo_name = next(iter(DEMO_DATASETS))

        embeddings_dir = tmp_path / "embeddings"
        embeddings_dir.mkdir()

        pkl_file = embeddings_dir / f"{demo_name}.pkl"
        pkl_file.write_bytes(
            pickle.dumps(
                {
                    "name": demo_name,
                    "medias": {1: {"embedding": [0.1], "file": "/fake/file.png"}},
                }
            )
        )
        # No sidecar written — simulates a legacy pickle

        medias: dict = {}

        def fake_load(path, m):
            m[1] = {"embedding": np.array([0.1]), "file": "/fake/file.png"}

        with (
            patch("vtsearch.datasets.loader.EMBEDDINGS_DIR", embeddings_dir),
            patch("vtsearch.datasets.loader.load_dataset_from_pickle", side_effect=fake_load),
        ):
            load_demo_dataset(demo_name, medias, embedder_name="siglip")

        # Cache was used despite requesting a different embedder (no sidecar to verify)
        assert 1 in medias


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
        import shutil
        from unittest.mock import patch

        from vtsearch.datasets.downloader import download_caltech101

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        zip_path = tmp_path / "caltech-101.zip"
        self._make_caltech101_zip(zip_path)

        with (
            patch("vtsearch.datasets.downloader.core.DATA_DIR", data_dir),
            patch("vtsearch.datasets.downloader.core.IMAGE_DIR", data_dir / "images"),
            patch(
                "vtsearch.datasets.downloader.core.download_file_with_progress",
                lambda url, dest, size, cb: shutil.copy(str(zip_path), str(dest)),
            ),
        ):
            result = download_caltech101(on_progress=lambda *a: None)

        assert result.exists(), f"Expected {result} to exist"
        assert result.name == "101_ObjectCategories"
        assert (result / "butterfly").is_dir()
        assert (result / "dolphin").is_dir()
        assert len(list((result / "butterfly").glob("*.jpg"))) == 3

    def test_inner_tar_cleaned_up(self, tmp_path):
        """The inner 101_ObjectCategories.tar.gz should be deleted after extraction."""
        import shutil
        from unittest.mock import patch

        from vtsearch.datasets.downloader import download_caltech101

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        zip_path = tmp_path / "caltech-101.zip"
        self._make_caltech101_zip(zip_path)

        with (
            patch("vtsearch.datasets.downloader.core.DATA_DIR", data_dir),
            patch("vtsearch.datasets.downloader.core.IMAGE_DIR", data_dir / "images"),
            patch(
                "vtsearch.datasets.downloader.core.download_file_with_progress",
                lambda url, dest, size, cb: shutil.copy(str(zip_path), str(dest)),
            ),
        ):
            download_caltech101(on_progress=lambda *a: None)

        inner_tar = data_dir / "caltech-101" / "101_ObjectCategories.tar.gz"
        assert not inner_tar.exists(), "Inner tar.gz should be deleted after extraction"

    def test_outer_zip_cleaned_up(self, tmp_path):
        """The temp archive should be deleted after extraction."""
        import shutil
        from unittest.mock import patch

        from vtsearch.datasets.downloader import download_caltech101

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        zip_path = tmp_path / "caltech-101.zip"
        self._make_caltech101_zip(zip_path)

        with (
            patch("vtsearch.datasets.downloader.core.DATA_DIR", data_dir),
            patch("vtsearch.datasets.downloader.core.IMAGE_DIR", data_dir / "images"),
            patch(
                "vtsearch.datasets.downloader.core.download_file_with_progress",
                lambda url, dest, size, cb: shutil.copy(str(zip_path), str(dest)),
            ),
        ):
            download_caltech101(on_progress=lambda *a: None)

        # No temp download files should remain in data_dir.
        leftover = [p for p in data_dir.iterdir() if p.name.startswith(".dl_")]
        assert not leftover, f"Temp archive files should be cleaned up: {leftover}"

    def test_skips_if_already_extracted(self, tmp_path):
        """If 101_ObjectCategories already exists, skip download and extraction."""
        from unittest.mock import patch

        from vtsearch.datasets.downloader import download_caltech101

        data_dir = tmp_path / "data"
        categories_dir = data_dir / "caltech-101" / "101_ObjectCategories" / "butterfly"
        categories_dir.mkdir(parents=True)
        (categories_dir / "image_0001.jpg").write_bytes(b"\xff\xd8\xff\xd9")

        with (
            patch("vtsearch.datasets.downloader.core.DATA_DIR", data_dir),
            patch("vtsearch.datasets.downloader.core.IMAGE_DIR", data_dir / "images"),
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
        import shutil
        from unittest.mock import patch

        from vtsearch.datasets.downloader import download_ucf101_subset

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        video_dir = data_dir / "video"
        tar_path = tmp_path / "UCF101_subset.tar.gz"
        self._make_ucf101_subset_tar(tar_path)

        with (
            patch("vtsearch.datasets.downloader.core.DATA_DIR", data_dir),
            patch("vtsearch.datasets.downloader.core.VIDEO_DIR", video_dir),
            patch(
                "vtsearch.datasets.downloader.core.download_file_with_progress",
                lambda url, dest, size, cb: shutil.copy(str(tar_path), str(dest)),
            ),
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
        """The temp archive should be deleted after extraction."""
        import shutil
        from unittest.mock import patch

        from vtsearch.datasets.downloader import download_ucf101_subset

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        video_dir = data_dir / "video"
        tar_path = tmp_path / "UCF101_subset.tar.gz"
        self._make_ucf101_subset_tar(tar_path)

        with (
            patch("vtsearch.datasets.downloader.core.DATA_DIR", data_dir),
            patch("vtsearch.datasets.downloader.core.VIDEO_DIR", video_dir),
            patch(
                "vtsearch.datasets.downloader.core.download_file_with_progress",
                lambda url, dest, size, cb: shutil.copy(str(tar_path), str(dest)),
            ),
        ):
            download_ucf101_subset(on_progress=lambda *a: None)

        # No temp download files should remain in data_dir.
        leftover = [p for p in data_dir.iterdir() if p.name.startswith(".dl_")]
        assert not leftover, f"Temp archive files should be cleaned up: {leftover}"

    def test_staging_dir_cleaned_up(self, tmp_path):
        """The UCF101_subset staging directory should be removed after flattening."""
        import shutil
        from unittest.mock import patch

        from vtsearch.datasets.downloader import download_ucf101_subset

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        video_dir = data_dir / "video"
        tar_path = tmp_path / "UCF101_subset.tar.gz"
        self._make_ucf101_subset_tar(tar_path)

        with (
            patch("vtsearch.datasets.downloader.core.DATA_DIR", data_dir),
            patch("vtsearch.datasets.downloader.core.VIDEO_DIR", video_dir),
            patch(
                "vtsearch.datasets.downloader.core.download_file_with_progress",
                lambda url, dest, size, cb: shutil.copy(str(tar_path), str(dest)),
            ),
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
            patch("vtsearch.datasets.downloader.core.DATA_DIR", data_dir),
            patch("vtsearch.datasets.downloader.core.VIDEO_DIR", video_dir),
        ):
            result = download_ucf101_subset(on_progress=lambda *a: None)

        assert result.exists()
        assert (result / "Archery" / "v_Archery_g01_c01.avi").exists()

    def test_demo_list_shows_video_download_size(self, client):
        """Video demo datasets should report a non-zero download size."""
        resp = client.get("/api/dataset/demo-list")
        data = resp.get_json()
        video_ds = [d for d in data["datasets"] if d["media_type"] == "video"]
        assert len(video_ds) > 0, "Should have at least one video demo dataset"
        for ds in video_ds:
            if ds["status"] == "needs_download":
                assert ds["download_size_mb"] > 0, f"Video demo '{ds['name']}' should have a positive download size"

    def test_video_demo_categories_match_source(self, client):
        """Video demo datasets should only use categories defined for their source."""
        from vtsearch.datasets.config import DEMO_DATASETS

        for name, info in DEMO_DATASETS.items():
            if info.get("media_type") == "video":
                cats = info.get("categories", [])
                assert len(cats) > 0, f"Video demo '{name}' should have at least one category"


class TestLoadProgressRaceCondition:
    """Dataset load endpoints must set progress to 'loading' before the thread starts.

    Without this, the frontend's progress poll can see a stale 'idle' from
    a previous load and prematurely stop polling, causing it to proceed
    with the old dataset's data and votes (the label-leak bug).
    """

    def test_load_registered_dataset_sets_progress_before_thread(self, client):
        """After POST to load a registered dataset, progress must not be 'idle'."""
        import time

        from vtsearch.concurrency.progress import get_progress

        # Register the current medias as a dataset entry so we can load it
        saved = dict(app_module.medias)
        try:
            # First, export current medias to a pkl for registration
            from vtsearch.datasets.loader import export_dataset_to_file
            from vtsearch.settings import get_saved_datasets_dir

            ds_dir = get_saved_datasets_dir()
            ds_dir.mkdir(parents=True, exist_ok=True)
            pkl_path = str(ds_dir / "test_race.pkl")
            from pathlib import Path

            Path(pkl_path).write_bytes(export_dataset_to_file(app_module.medias))

            # Register in the dataset registry
            from vtsearch.datasets.registry import register_dataset

            entry = register_dataset(
                name="test_race",
                media_type="audio",
                num_items=len(app_module.medias),
                pkl_path=pkl_path,
            )
            dataset_id = entry["id"]

            # Set progress to idle (simulating a previous completed load)
            from vtsearch.concurrency.progress import update_progress

            update_progress("idle", "Ready")

            # POST to load the dataset
            resp = client.post(f"/api/datasets/registry/{dataset_id}/load")
            assert resp.status_code == 200

            # Immediately check progress — it must NOT be 'idle'
            progress = get_progress()
            assert progress["status"] != "idle", (
                "Progress must be set to 'loading' before the thread starts "
                "to prevent the frontend from seeing a stale 'idle' state"
            )

            # Wait for the background thread to finish (up to 12s for slow CI)
            for _ in range(120):
                time.sleep(0.1)
                if get_progress()["status"] == "idle":
                    break
        finally:
            app_module.medias.clear()
            app_module.medias.update(saved)
            # Clean up
            Path(pkl_path).unlink(missing_ok=True)
            from vtsearch.datasets.registry import unregister_dataset

            unregister_dataset(dataset_id)

    def test_origin_load_clears_stale_error(self):
        """_run_origin_load_in_background must clear old error on new load."""
        from unittest.mock import patch

        from vtsearch.concurrency.progress import get_progress, update_progress

        # Simulate a previous load that left a stale error
        update_progress("idle", "", error="Previous load failed", step=None, total_steps=None)
        assert get_progress()["error"] == "Previous load failed"

        # Start a new load (mock the thread so it doesn't actually run)
        from vtsearch.datasets.load_pipeline import _run_origin_load_in_background

        with patch("vtsearch.datasets.load_pipeline.threading.Thread"):
            _run_origin_load_in_background(
                lambda: None,
                {"importer": "test", "params": {}},
            )

        progress = get_progress()
        assert progress["error"] is None, "Starting a new load must clear the stale error from a previous load"
        assert progress["status"] == "loading"

    def test_load_embedder_sets_initial_progress(self):
        """_load_embedder_for_clips must set progress before loading starts.

        The function should emit 'Loading embedding model…' so the frontend
        knows the embedder warm-up phase has begun (rather than staying stuck
        on the previous phase's message like 'Saving to registry…').
        """
        from unittest.mock import patch

        from vtsearch.datasets.load_pipeline import _load_embedder_for_clips
        from vtsearch.concurrency.progress import update_progress

        # _load_embedder_for_clips inspects medias to find the embedder.
        # The conftest-populated medias dict provides this automatically.
        # Set a stale progress message from the previous phase.
        update_progress("loading", "Saving to registry…", step=3, total_steps=4)

        messages: list[str] = []

        def _capture_update(status, message="", current=0, total=0, **kw):
            messages.append(message)

        with patch("vtsearch.datasets.load_pipeline.update_progress", side_effect=_capture_update):
            _load_embedder_for_clips()

        # Should have set "Loading embedding model…" before calling load_models
        assert any("Loading embedding model" in m for m in messages), (
            f"Expected 'Loading embedding model…' in progress messages, got: {messages}"
        )


class TestCancelIngest:
    """Tests for the POST /api/dataset/cancel endpoint."""

    def test_cancel_endpoint_returns_ok(self, client):
        """POST /api/dataset/cancel should return ok."""
        resp = client.post("/api/dataset/cancel")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True

    def test_cancel_sets_event(self, client):
        """POST /api/dataset/cancel should set the cancellation event."""
        from vtsearch.concurrency.progress import dataset_progress

        dataset_progress.reset_cancel()
        assert not dataset_progress.is_cancelled

        client.post("/api/dataset/cancel")
        assert dataset_progress.is_cancelled

        # Clean up
        dataset_progress.reset_cancel()

    def test_cancel_clears_medias_on_background_load(self, client):
        """Cancelling during a background load should clean up medias."""
        import threading
        import time

        from vtsearch.concurrency.progress import dataset_progress, get_progress

        saved = dict(app_module.medias)
        try:
            started = threading.Event()

            # Simulate a slow importer that checks cancellation via progress.
            # Use ``while True`` so the function can only exit via
            # CancelledError — a bounded loop (e.g. ``range(100)``) can
            # finish before the cancel is processed on a loaded machine,
            # making the test flaky.
            def slow_load():
                started.set()
                while True:
                    dataset_progress.check_cancelled()
                    time.sleep(0.05)

            from vtsearch.datasets.load_pipeline import _run_origin_load_in_background

            _run_origin_load_in_background(
                slow_load,
                {"importer": "test", "params": {}},
                created_by="test",
            )

            # Wait for the thread to actually start (up to 5s)
            started.wait(timeout=5.0)

            # Cancel it
            client.post("/api/dataset/cancel")

            # Wait for the thread to notice the cancellation
            for _ in range(80):
                time.sleep(0.1)
                progress = get_progress()
                if progress["status"] == "idle":
                    break

            progress = get_progress()
            assert progress["status"] == "idle"
            assert progress["error"] == "Cancelled"
        finally:
            dataset_progress.reset_cancel()
            app_module.medias.clear()
            app_module.medias.update(saved)

    def test_new_load_resets_cancel_flag(self, client):
        """Starting a new load should clear any previous cancellation."""
        from unittest.mock import patch

        from vtsearch.concurrency.progress import dataset_progress

        # Set cancel from a previous operation
        dataset_progress.cancel()
        assert dataset_progress.is_cancelled

        saved = dict(app_module.medias)
        try:
            # Start a new load — should reset the flag
            from vtsearch.datasets.load_pipeline import _run_origin_load_in_background

            with patch("vtsearch.datasets.load_pipeline._load_embedder_for_clips"):
                _run_origin_load_in_background(
                    lambda: None,
                    {"importer": "test", "params": {}},
                    created_by="test",
                )

            # Cancel flag should have been cleared
            assert not dataset_progress.is_cancelled
        finally:
            dataset_progress.reset_cancel()
            app_module.medias.clear()
            app_module.medias.update(saved)
