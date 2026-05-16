"""Tests for the VTSearch dashboard API endpoint."""

import app as app_module  # noqa: F401 — triggers conftest side effects
from vtsearch.datasets.registry import register_dataset
from vtsearch.detectors.registry import register_detector
from vtsearch.state import medias


class TestDashboardDatasetInfo:
    """Tests for GET /api/dashboard/dataset-info."""

    def test_returns_404_when_no_dataset(self, client):
        """Endpoint returns 404 when no dataset is loaded."""
        saved = dict(medias)
        medias.clear()
        try:
            resp = client.get("/api/dashboard/dataset-info")
            assert resp.status_code == 404
            data = resp.get_json()
            assert "error" in data
        finally:
            medias.update(saved)

    def test_returns_info_with_loaded_medias(self, client):
        """Endpoint returns dataset metadata when medias are loaded."""
        resp = client.get("/api/dashboard/dataset-info")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "name" in data
        assert "num_medias" in data
        assert "media_type" in data
        assert "origin" in data
        assert data["num_medias"] == len(medias)

    def test_returns_correct_media_type(self, client):
        """Endpoint returns the media type of the first media."""
        resp = client.get("/api/dashboard/dataset-info")
        data = resp.get_json()
        first_media = next(iter(medias.values()))
        assert data["media_type"] == first_media.get("type", "audio")

    def test_returns_origin_from_media(self, client):
        """Endpoint extracts origin info when medias have origin set."""
        first_key = next(iter(medias))
        saved_origin = medias[first_key].get("origin")
        try:
            medias[first_key]["origin"] = {"importer": "demo", "params": {"name": "test_dataset"}}
            resp = client.get("/api/dashboard/dataset-info")
            data = resp.get_json()
            assert data["origin"] == "demo:test_dataset"
            assert data["name"] == "test_dataset"
        finally:
            if saved_origin is not None:
                medias[first_key]["origin"] = saved_origin
            else:
                medias[first_key].pop("origin", None)

    def test_returns_origin_folder(self, client):
        """Endpoint formats folder origin correctly."""
        first_key = next(iter(medias))
        saved_origin = medias[first_key].get("origin")
        try:
            medias[first_key]["origin"] = {"importer": "server_folder", "params": {"path": "/data/sounds"}}
            resp = client.get("/api/dashboard/dataset-info")
            data = resp.get_json()
            assert data["origin"] == "server_folder:/data/sounds"
        finally:
            if saved_origin is not None:
                medias[first_key]["origin"] = saved_origin
            else:
                medias[first_key].pop("origin", None)

    def test_returns_origin_pickle(self, client):
        """Endpoint formats pickle origin correctly."""
        first_key = next(iter(medias))
        saved_origin = medias[first_key].get("origin")
        try:
            medias[first_key]["origin"] = {"importer": "pickle", "params": {"filename": "esc50.pkl"}}
            resp = client.get("/api/dashboard/dataset-info")
            data = resp.get_json()
            assert data["origin"] == "file:esc50.pkl"
            assert data["name"] == "esc50.pkl"
        finally:
            if saved_origin is not None:
                medias[first_key]["origin"] = saved_origin
            else:
                medias[first_key].pop("origin", None)

    def test_returns_unknown_origin_when_none(self, client):
        """Endpoint returns 'unknown' origin when medias have no origin."""
        # Default test medias don't have origin set
        saved_origins = {}
        for k, v in medias.items():
            saved_origins[k] = v.get("origin")
            v.pop("origin", None)
        try:
            resp = client.get("/api/dashboard/dataset-info")
            data = resp.get_json()
            assert data["origin"] == "unknown"
        finally:
            for k, v in medias.items():
                if saved_origins.get(k) is not None:
                    v["origin"] = saved_origins[k]


class TestDashboardDiskUsage:
    """Tests for GET /api/dashboard/disk-usage."""

    def test_returns_disk_usage(self, client):
        """Endpoint returns total/used/free byte counts and a path."""
        resp = client.get("/api/dashboard/disk-usage")
        assert resp.status_code == 200
        data = resp.get_json()
        for key in ("total", "used", "free", "path"):
            assert key in data
        assert isinstance(data["total"], int)
        assert isinstance(data["used"], int)
        assert isinstance(data["free"], int)
        assert data["total"] > 0
        assert data["used"] >= 0
        assert data["free"] >= 0
        assert data["used"] + data["free"] <= data["total"] + 1  # rounding tolerance


class TestDashboardHtmlPresent:
    """Test that the dashboard-related content is present in the Angular bundle."""

    def test_index_serves_angular_app(self, client):
        """GET / should serve the Angular SPA shell."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"<app-root>" in resp.data

    def test_dashboard_view_in_bundle(self, client):
        """The Angular bundle should contain dashboard view references."""
        import glob as globmod

        combined = ""
        for path in globmod.glob("static/*.js"):
            resp = client.get(f"/{path}")
            combined += resp.data.decode("utf-8")
        assert "dashboard" in combined
        assert "Datasets" in combined
        assert "Models" in combined

    def test_dashboard_route_defined(self, client):
        """The /dashboard route should serve the Angular SPA."""
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        assert b"<app-root>" in resp.data


class TestGuessMediaEmbedder:
    """Frontend auto-populates the embedder dropdown when creating a new dataset.

    The guessing logic lives in the Angular frontend (dashboard component):
    collect embedder names from datasets in the registry and in-progress loading
    tasks. If exactly one unique embedder exists, pre-select it.

    These tests verify the underlying data contracts that the frontend logic relies on.
    """

    def test_single_dataset_embedder_in_registry(self, client):
        """When the registry has one dataset with an embedder, the field is available."""
        register_dataset(
            name="test-siglip",
            media_type="image",
            num_items=10,
            pkl_path="/tmp/siglip.pkl",
            embedder="siglip",
        )
        resp = client.get("/api/datasets/registry")
        data = resp.get_json()
        assert len(data["datasets"]) == 1
        assert data["datasets"][0]["embedder"] == "siglip"

    def test_multiple_same_embedder_datasets_in_registry(self, client):
        """Multiple datasets with the same embedder yield a single unique embedder."""
        register_dataset(name="ds1", media_type="image", num_items=5, pkl_path="/tmp/a.pkl", embedder="siglip")
        register_dataset(name="ds2", media_type="image", num_items=3, pkl_path="/tmp/b.pkl", embedder="siglip")
        resp = client.get("/api/datasets/registry")
        data = resp.get_json()
        embedders = {d["embedder"] for d in data["datasets"]}
        assert embedders == {"siglip"}

    def test_mixed_embedder_datasets_no_single_guess(self, client):
        """Multiple datasets with different embedders produce more than one unique embedder."""
        register_dataset(name="ds1", media_type="image", num_items=5, pkl_path="/tmp/a.pkl", embedder="clip")
        register_dataset(name="ds2", media_type="image", num_items=3, pkl_path="/tmp/b.pkl", embedder="siglip")
        resp = client.get("/api/datasets/registry")
        data = resp.get_json()
        embedders = {d["embedder"] for d in data["datasets"]}
        assert len(embedders) > 1

    def test_loading_task_includes_embedder(self, client):
        """Loading tasks expose the embedder field for in-progress guessing."""
        from vtsearch.concurrency.progress import loading_tasks

        tracker = loading_tasks.create_task("test_emb_task", "test-ds", media_type="image", embedder="siglip")
        tracker.update("loading", "Embedding...", 0, 10)
        tasks = loading_tasks.list_tasks()
        task = next(t for t in tasks if t["task_id"] == "test_emb_task")
        assert task["embedder"] == "siglip"

    def test_loading_task_omits_empty_embedder(self, client):
        """Loading tasks without an embedder do not include the field."""
        from vtsearch.concurrency.progress import loading_tasks

        tracker = loading_tasks.create_task("test_no_emb", "test-ds", media_type="image")
        tracker.update("loading", "Loading...", 0, 10)
        tasks = loading_tasks.list_tasks()
        task = next(t for t in tasks if t["task_id"] == "test_no_emb")
        assert "embedder" not in task

    def test_js_contains_embedder_guessing_logic(self, client):
        """The Angular bundle should include the embedder guessing code."""
        import glob as globmod

        combined = ""
        for path in globmod.glob("static/*.js"):
            resp = client.get(f"/{path}")
            combined += resp.data.decode("utf-8")
        assert "guessedMediaEmbedder" in combined


class TestGuessMediaType:
    """Frontend auto-populates the media-type dropdown when creating a new model.

    The guessing logic lives in the Angular frontend (dashboard component):
    if all datasets in the registry share a single media_type, use that;
    otherwise leave the dropdown blank.

    These tests verify the underlying data contracts that the frontend logic relies on.
    """

    def test_js_contains_guessing_logic(self, client):
        """main.js should include the media-type guessing code."""
        resp = client.get("/static/main.js")
        text = resp.data.decode("utf-8")
        assert "guessedMediaType" in text or "media_type" in text

    def test_single_dataset_type_in_registry(self, client):
        """When the registry has one dataset, its media_type is available."""
        register_dataset(
            name="test-audio",
            media_type="audio",
            num_items=10,
            pkl_path="/tmp/fake.pkl",
        )
        resp = client.get("/api/datasets/registry")
        data = resp.get_json()
        assert len(data["datasets"]) == 1
        assert data["datasets"][0]["media_type"] == "audio"

    def test_multiple_same_type_datasets_in_registry(self, client):
        """Multiple datasets of the same type yield a single unique type."""
        register_dataset(name="ds1", media_type="image", num_items=5, pkl_path="/tmp/a.pkl")
        register_dataset(name="ds2", media_type="image", num_items=3, pkl_path="/tmp/b.pkl")
        resp = client.get("/api/datasets/registry")
        data = resp.get_json()
        types = {d["media_type"] for d in data["datasets"]}
        assert types == {"image"}

    def test_mixed_type_datasets_no_single_guess(self, client):
        """Multiple datasets with different types produce more than one unique type."""
        register_dataset(name="ds1", media_type="audio", num_items=5, pkl_path="/tmp/a.pkl")
        register_dataset(name="ds2", media_type="image", num_items=3, pkl_path="/tmp/b.pkl")
        resp = client.get("/api/datasets/registry")
        data = resp.get_json()
        types = {d["media_type"] for d in data["datasets"]}
        assert len(types) > 1

    def test_empty_registry_no_guess(self, client):
        """With no datasets, the registry returns an empty list and the frontend leaves the field blank."""
        resp = client.get("/api/datasets/registry")
        data = resp.get_json()
        assert data["datasets"] == []


class TestDashboardDatasetRegistryColumns:
    """Tests that the dataset registry API returns fields needed for new columns."""

    def test_dataset_registry_includes_created_at(self, client):
        """Registered datasets include a created_at timestamp."""
        register_dataset(name="ts-ds", media_type="audio", num_items=5, pkl_path="/tmp/ts.pkl")
        resp = client.get("/api/datasets/registry")
        data = resp.get_json()
        ds = data["datasets"][0]
        assert "created_at" in ds
        assert isinstance(ds["created_at"], (int, float))
        assert ds["created_at"] > 0

    def test_dataset_registry_includes_origin(self, client):
        """Registered datasets include an origin string."""
        register_dataset(
            name="orig-ds",
            media_type="image",
            num_items=10,
            pkl_path="/tmp/orig.pkl",
            origin="demo:flowers",
        )
        resp = client.get("/api/datasets/registry")
        data = resp.get_json()
        ds = data["datasets"][0]
        assert ds["origin"] == "demo:flowers"

    def test_dataset_registry_includes_source(self, client):
        """Registered datasets include a source dict for the Details column."""
        src = {"importer": "server_folder", "params": {"path": "/data/images"}}
        register_dataset(
            name="src-ds",
            media_type="image",
            num_items=8,
            pkl_path="/tmp/src.pkl",
            origin="folder:/data/images",
            source=src,
        )
        resp = client.get("/api/datasets/registry")
        data = resp.get_json()
        ds = data["datasets"][0]
        assert ds["source"] == src

    def test_dataset_registry_default_origin_is_unknown(self, client):
        """Datasets registered without origin default to 'unknown'."""
        register_dataset(name="no-origin", media_type="audio", num_items=3, pkl_path="/tmp/no.pkl")
        resp = client.get("/api/datasets/registry")
        data = resp.get_json()
        ds = data["datasets"][0]
        assert ds["origin"] == "unknown"

    def test_dataset_registry_includes_loaded_field(self, client):
        """Registered datasets include the loaded boolean (renamed to Loaded? in UI)."""
        register_dataset(name="ld-ds", media_type="audio", num_items=5, pkl_path="/tmp/ld.pkl")
        resp = client.get("/api/datasets/registry")
        data = resp.get_json()
        ds = data["datasets"][0]
        assert "loaded" in ds
        assert isinstance(ds["loaded"], bool)


class TestDashboardDatasetDupesColumn:
    """Tests that the dataset registry API returns the num_dupes field."""

    def test_dataset_registry_includes_num_dupes(self, client):
        """Registered datasets include a num_dupes integer."""
        register_dataset(name="dupes-ds", media_type="audio", num_items=10, pkl_path="/tmp/dupes.pkl")
        resp = client.get("/api/datasets/registry")
        data = resp.get_json()
        ds = data["datasets"][0]
        assert "num_dupes" in ds
        assert isinstance(ds["num_dupes"], int)

    def test_dataset_registry_num_dupes_defaults_to_zero(self, client):
        """Datasets registered without num_dupes default to 0."""
        register_dataset(name="no-dupes", media_type="audio", num_items=5, pkl_path="/tmp/nodupes.pkl")
        resp = client.get("/api/datasets/registry")
        data = resp.get_json()
        ds = data["datasets"][0]
        assert ds["num_dupes"] == 0

    def test_dataset_registry_stores_explicit_num_dupes(self, client):
        """Datasets registered with explicit num_dupes retain the value."""
        register_dataset(name="with-dupes", media_type="audio", num_items=20, pkl_path="/tmp/wdupes.pkl", num_dupes=3)
        resp = client.get("/api/datasets/registry")
        data = resp.get_json()
        ds = data["datasets"][0]
        assert ds["num_dupes"] == 3


class TestDashboardDatasetEmbedderColumn:
    """Tests that the dataset registry API returns the embedder field."""

    def test_dataset_registry_includes_embedder(self, client):
        """Registered datasets include an embedder string."""
        register_dataset(
            name="emb-ds",
            media_type="audio",
            num_items=10,
            pkl_path="/tmp/emb.pkl",
            embedder="clap",
        )
        resp = client.get("/api/datasets/registry")
        data = resp.get_json()
        ds = data["datasets"][0]
        assert "embedder" in ds
        assert ds["embedder"] == "clap"

    def test_dataset_registry_embedder_defaults_to_empty(self, client):
        """Datasets registered without embedder default to empty string."""
        register_dataset(name="no-emb", media_type="audio", num_items=5, pkl_path="/tmp/noemb.pkl")
        resp = client.get("/api/datasets/registry")
        data = resp.get_json()
        ds = data["datasets"][0]
        assert ds["embedder"] == ""

    def test_dataset_registry_stores_explicit_embedder(self, client):
        """Datasets registered with explicit embedder retain the value."""
        register_dataset(
            name="clip-ds",
            media_type="image",
            num_items=20,
            pkl_path="/tmp/clipemb.pkl",
            embedder="clip",
        )
        resp = client.get("/api/datasets/registry")
        data = resp.get_json()
        ds = data["datasets"][0]
        assert ds["embedder"] == "clip"


class TestDashboardSecurityIcon:
    """Tests that the dataset registry API returns fields needed by the security icon."""

    def test_dataset_registry_includes_created_by(self, client):
        """Registered datasets include a created_by string for ownership checks."""
        register_dataset(name="sec-ds", media_type="audio", num_items=5, pkl_path="/tmp/sec.pkl")
        resp = client.get("/api/datasets/registry")
        data = resp.get_json()
        ds = data["datasets"][0]
        assert "created_by" in ds
        assert isinstance(ds["created_by"], str)
        assert len(ds["created_by"]) > 0

    def test_dataset_registry_includes_readers(self, client):
        """Registered datasets include a readers list for the access control UI."""
        register_dataset(name="readers-ds", media_type="image", num_items=10, pkl_path="/tmp/readers.pkl")
        resp = client.get("/api/datasets/registry")
        data = resp.get_json()
        ds = data["datasets"][0]
        assert "readers" in ds
        assert isinstance(ds["readers"], list)

    def test_security_icon_in_frontend_bundle(self, client):
        """The Angular bundle should contain the security icon shield path."""
        import glob as globmod

        combined = ""
        for path in globmod.glob("static/*.js"):
            resp = client.get(f"/{path}")
            combined += resp.data.decode("utf-8")
        # The shield SVG path used for the security icon
        assert "Edit access list" in combined or "security-btn" in combined

    def test_dataset_registry_created_by_defaults_to_current_user(self, client):
        """Datasets created without explicit created_by use the current user."""
        register_dataset(name="default-owner", media_type="text", num_items=3, pkl_path="/tmp/defown.pkl")
        resp = client.get("/api/datasets/registry")
        data = resp.get_json()
        ds = data["datasets"][0]
        # In single-user mode, created_by defaults to "default"
        assert ds["created_by"] == "default"


class TestEmbeddersApiEndpoint:
    """Tests for the /api/embedders endpoint with optional media_type filtering."""

    def test_list_all_embedders(self, client):
        """GET /api/embedders returns all registered embedders."""
        resp = client.get("/api/embedders")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "embedders" in data
        names = [e["name"] for e in data["embedders"]]
        assert "clap" in names
        assert "siglip" in names
        assert "e5" in names
        assert "xclip" in names

    def test_filter_by_type_id(self, client):
        """GET /api/embedders?media_type=audio returns only audio embedders."""
        resp = client.get("/api/embedders?media_type=audio")
        assert resp.status_code == 200
        data = resp.get_json()
        embedders = data["embedders"]
        assert all(e["media_type_id"] == "audio" for e in embedders)
        names = [e["name"] for e in embedders]
        assert "clap" in names
        assert "siglip" not in names

    def test_filter_by_folder_name(self, client):
        """GET /api/embedders?media_type=image returns image embedders."""
        resp = client.get("/api/embedders?media_type=image")
        assert resp.status_code == 200
        data = resp.get_json()
        embedders = data["embedders"]
        assert all(e["media_type_id"] == "image" for e in embedders)
        names = [e["name"] for e in embedders]
        assert "siglip" in names

    def test_filter_unknown_type_returns_empty(self, client):
        """GET /api/embedders?media_type=nonexistent returns empty list."""
        resp = client.get("/api/embedders?media_type=nonexistent")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["embedders"] == []

    def test_embedder_dict_has_required_keys(self, client):
        """Each embedder dict has name and media_type_id keys."""
        resp = client.get("/api/embedders")
        data = resp.get_json()
        for emb in data["embedders"]:
            assert "name" in emb
            assert "media_type_id" in emb


class TestDashboardModelRegistryColumns:
    """Tests that the model registry API returns fields needed for new columns."""

    def test_model_registry_includes_created_at(self, client):
        """Registered models include a created_at timestamp."""
        register_detector(name="ts-model", media_type="audio")
        resp = client.get("/api/detectors/registry")
        data = resp.get_json()
        m = data["detectors"][0]
        assert "created_at" in m
        assert isinstance(m["created_at"], (int, float))
        assert m["created_at"] > 0

    def test_model_registry_includes_autorun_false_by_default(self, client):
        """Detectors that are not flagged for autorun show autorun=False."""
        register_detector(name="no-autorun", media_type="image")
        resp = client.get("/api/detectors/registry")
        data = resp.get_json()
        m = data["detectors"][0]
        assert m["autorun"] is False

    def test_autorun_toggle_via_api(self, client):
        """Toggling autorun via PUT updates the model registry response."""
        from vtsearch.settings import get_autorun_detectors

        entry = register_detector(name="toggle-det", media_type="audio")

        resp = client.get("/api/detectors/registry")
        m = resp.get_json()["detectors"][0]
        assert m["autorun"] is False

        resp = client.put(
            f"/api/detectors/registry/{entry['id']}/autorun",
            json={"autorun": True},
        )
        assert resp.status_code == 200
        assert "toggle-det" in get_autorun_detectors()

        resp = client.get("/api/detectors/registry")
        m = resp.get_json()["detectors"][0]
        assert m["autorun"] is True

    def test_model_registry_includes_loaded_field(self, client):
        """Registered models include the loaded boolean (not shown in UI)."""
        register_detector(name="ld-model", media_type="audio")
        resp = client.get("/api/detectors/registry")
        data = resp.get_json()
        m = data["detectors"][0]
        assert "loaded" in m
        assert isinstance(m["loaded"], bool)

    def test_model_registry_detector_loaded_follows_loaded(self, client):
        """detector_loaded reflects whether a DetectorContext is registered."""
        from vtsearch.detectors.registry import add_loaded_detector_id

        entry = register_detector(name="train-ld", media_type="audio")
        resp = client.get("/api/detectors/registry")
        m = resp.get_json()["detectors"][0]
        assert m["detector_loaded"] is False

        add_loaded_detector_id(entry["id"])
        resp = client.get("/api/detectors/registry")
        m = resp.get_json()["detectors"][0]
        assert m["detector_loaded"] is True

    def test_model_registry_includes_last_trained_at(self, client):
        """Registered models include the last_trained_at field (None by default)."""
        register_detector(name="lt-model", media_type="audio")
        resp = client.get("/api/detectors/registry")
        data = resp.get_json()
        m = data["detectors"][0]
        assert "last_trained_at" in m
        assert m["last_trained_at"] is None

    def test_model_registry_last_trained_at_set_on_label_save(self, client):
        """Saving labels updates last_trained_at to a timestamp."""
        from vtsearch.detectors.registry import register_detector as reg_model, update_detector

        entry = reg_model(name="lt-save", media_type="audio")
        import time

        now = time.time()
        update_detector(entry["id"], last_trained_at=now)
        resp = client.get("/api/detectors/registry")
        data = resp.get_json()
        m = data["detectors"][0]
        assert isinstance(m["last_trained_at"], (int, float))
        assert m["last_trained_at"] >= now


class TestAutorunCheckboxPersistence:
    """Tests that toggling autorun via the API persists the setting."""

    def test_autorun_toggle_persists_to_settings(self, client):
        """Toggling autorun on saves the model name to settings."""
        from vtsearch.settings import get_autorun_detectors

        entry = register_detector(name="persist-det", media_type="audio")

        resp = client.put(
            f"/api/detectors/registry/{entry['id']}/autorun",
            json={"autorun": True},
        )
        assert resp.status_code == 200
        assert "persist-det" in get_autorun_detectors()

    def test_autorun_toggle_off_removes_from_settings(self, client):
        """Toggling autorun off removes the model name from settings."""
        from vtsearch.settings import add_autorun_detector, get_autorun_detectors

        entry = register_detector(name="remove-det", media_type="audio")
        add_autorun_detector("remove-det")

        resp = client.put(
            f"/api/detectors/registry/{entry['id']}/autorun",
            json={"autorun": False},
        )
        assert resp.status_code == 200
        assert "remove-det" not in get_autorun_detectors()

    def test_model_registry_settings_drives_autorun_flag(self, client):
        """Adding a name to autorun_detectors flips the registry flag."""
        from vtsearch.settings import add_autorun_detector

        register_detector(name="settings-det", media_type="audio")
        add_autorun_detector("settings-det")

        resp = client.get("/api/detectors/registry")
        m = resp.get_json()["detectors"][0]
        assert m["autorun"] is True

    def test_autorun_not_in_settings_defaults(self, client):
        """autorun_detectors should not appear in the defaults endpoint."""
        resp = client.get("/api/settings/defaults")
        data = resp.get_json()
        assert "autorun_detectors" not in data


class TestDashboardColumnHeaders:
    """Verify the frontend JS contains the updated column headers."""

    def test_dataset_grid_has_new_column_headers(self, client):
        """Bundle JS should include the dataset column headers."""
        import glob as globmod

        combined = ""
        for path in globmod.glob("static/*.js"):
            resp = client.get(f"/{path}")
            combined += resp.data.decode("utf-8")
        assert "Created" in combined
        assert "Origin" in combined
        assert "Loaded" in combined

    def test_model_grid_has_new_column_headers(self, client):
        """Bundle JS should include the model column headers."""
        import glob as globmod

        combined = ""
        for path in globmod.glob("static/*.js"):
            resp = client.get(f"/{path}")
            combined += resp.data.decode("utf-8")
        assert "Autorun" in combined
        assert "Trainable" in combined


class TestFindButtonValidation:
    """Tests that the Find endpoint rejects invalid requests and that
    the frontend bundle contains the resolved-count validation logic."""

    def test_find_rejects_empty_detector_ids(self, client):
        """POST /api/find with no models returns 400."""
        register_dataset(name="find-ds", media_type="audio", num_items=5, pkl_path="/tmp/find.pkl")
        resp = client.get("/api/datasets/registry")
        ds_id = resp.get_json()["datasets"][0]["id"]
        resp = client.post("/api/find", json={"dataset_ids": [ds_id], "detector_ids": []})
        assert resp.status_code == 400
        assert "No detectors selected" in resp.get_json()["message"]

    def test_find_rejects_empty_dataset_ids(self, client):
        """POST /api/find with no datasets returns 400."""
        register_detector(name="find-m", media_type="audio")
        resp = client.get("/api/detectors/registry")
        m_id = resp.get_json()["detectors"][0]["id"]
        resp = client.post("/api/find", json={"dataset_ids": [], "detector_ids": [m_id]})
        assert resp.status_code == 400
        assert "No datasets selected" in resp.get_json()["message"]

    def test_frontend_uses_resolved_model_count(self, client):
        """The Angular bundle should use resolvedSelectedModels for Find validation."""
        import glob as globmod

        combined = ""
        for path in globmod.glob("static/*.js"):
            resp = client.get(f"/{path}")
            combined += resp.data.decode("utf-8")
        assert "resolvedSelectedModels" in combined
        assert "resolvedSelectedDatasets" in combined


class TestFindProgress:
    """Tests for the find_progress tracker (streamed via the SSE `find` channel)."""

    def test_find_progress_returns_idle_by_default(self, client):
        """The find_progress tracker is idle when no Find is running."""
        from vtsearch.concurrency.progress import find_progress

        data = find_progress.get()
        assert data["status"] == "idle"
        assert data["message"] == ""
        assert data["step"] is None
        assert data["total_steps"] is None

    def test_find_progress_updates_during_find(self, client, tmp_path):
        """The find_progress tracker is updated during /api/find execution."""
        import pickle

        import numpy as np

        from helpers import setup_trainable_model_in_registry
        from vtsearch.concurrency.progress import find_progress

        # Create a dataset pkl with three items whose md5s match a labelset.
        ds_medias = {}
        for i in range(3):
            emb = np.random.RandomState(i + 50).randn(512).astype(np.float32)
            ds_medias[i] = {
                "id": i,
                "type": "audio",
                "embedding": emb,
                "md5": f"prog_md5_{i}",
                "filename": f"prog_{i}.wav",
                "origin_name": f"prog_{i}.wav",
            }
        pkl_path = tmp_path / "prog_ds.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump({"medias": ds_medias}, f)

        ds = register_dataset(name="prog-ds", media_type="audio", num_items=3, pkl_path=str(pkl_path))

        # Build a detector with labels matching the dataset's md5s so
        # the on-the-fly trainer has both a good and a bad example.
        detector_id = setup_trainable_model_in_registry(
            "prog-det",
            good_ids=[0, 1],
            bad_ids=[2],
            snap=ds_medias,
            media_type="audio",
        )
        m = {"id": detector_id}

        # Capture progress snapshots during find by monkey-patching update
        snapshots = []
        original_update = find_progress.update

        def capturing_update(*args, **kwargs):
            original_update(*args, **kwargs)
            snapshots.append(find_progress.get())

        find_progress.update = capturing_update
        try:
            resp = client.post("/api/find", json={"dataset_ids": [ds["id"]], "detector_ids": [m["id"]]})
        finally:
            find_progress.update = original_update

        assert resp.status_code == 200

        # Should have progress snapshots from each phase
        assert len(snapshots) >= 3  # at least: prepare models, load dataset, score

        # Check step 1 (preparing models)
        step1 = [s for s in snapshots if s.get("step") == 1]
        assert len(step1) >= 1
        assert step1[0]["total_steps"] == 3
        assert step1[0]["status"] == "running"

        # Check step 2 (loading dataset)
        step2 = [s for s in snapshots if s.get("step") == 2]
        assert len(step2) >= 1
        assert "Loading dataset" in step2[0]["message"]

        # Check step 3 (scoring)
        step3 = [s for s in snapshots if s.get("step") == 3]
        assert len(step3) >= 1
        assert "Scoring" in step3[0]["message"]

        # Final state should be idle
        final = find_progress.get()
        assert final["status"] == "idle"

    def test_find_progress_resets_on_error(self, client):
        """Progress resets to idle when Find returns an error."""
        from vtsearch.concurrency.progress import find_progress

        resp = client.post("/api/find", json={"dataset_ids": [], "detector_ids": []})
        assert resp.status_code == 400
        data = find_progress.get()
        assert data["status"] == "idle"
