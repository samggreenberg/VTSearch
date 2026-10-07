"""Background AutoFind: the dataset ⋯ **Run AutoFind** action and the import hook (#4252).

Covers :mod:`vtsearch.autofind` end to end through its two routes:

- ``POST /api/datasets/registry/<dataset_id>/autofind`` starts a run of the
  caller's AutoFind detectors (or, given ``detector_ids``, of the detectors
  those ids name) on a loaded dataset, as a ``loading-tasks`` task keyed to
  that dataset;
- ``GET /api/autofind/runs/<run_id>`` serves the kept results to the user who
  started the run.

plus the ``post_load`` hook a web import installs (:func:`import_post_load`),
which is what makes AutoFind detectors actually run on an imported dataset.
"""

from __future__ import annotations

import threading

import pytest

import vtsearch.autofind as autofind_mod
from tests import load_detector_and_wait, wait_for_loading_task
from tests.helpers import setup_trainable_model_in_registry
from vtscore.concurrency.progress import find_progress, loading_tasks
from vtscore.state.core import DatasetContext, get_detector_context, register_context
from vtsearch import settings
from vtsearch.autofind import TASK_PREFIX, get_autofind_run, import_post_load
from vtsearch.state import snapshot_medias

GOOD = [1, 2, 3]
BAD = [18, 19, 20]


def _registered_copy(name: str, ids: set[int] | None = None) -> tuple[dict, DatasetContext]:
    """Register + load a dataset holding (a subset of) the test medias.

    It is deliberately *not* made the thread's active dataset: the route must
    score the dataset it was asked about, not the request's active one.
    """
    from vtscore.datasets.registry import add_loaded_id, register_dataset

    snap = snapshot_medias()
    keep = {mid: m for mid, m in snap.items() if ids is None or mid in ids}
    first = next(iter(keep.values()))
    entry = register_dataset(
        name=name,
        media_type=first.get("media_type", "audio"),
        num_items=len(keep),
        pkl_path=f"/tmp/autofind_{name.replace(' ', '_').lower()}.pkl",
        embedder=first.get("embedder", ""),
    )
    ctx = DatasetContext(entry["id"])
    for mid, media in keep.items():
        ctx.medias[mid] = dict(media)
    ctx.dataset_display_name = name
    register_context(ctx)
    add_loaded_id(entry["id"])
    return entry, ctx


def _autofind_detector(name: str = "ar-det") -> str:
    """A trainable detector on the caller's AutoFind list; returns its registry id."""
    detector_id = setup_trainable_model_in_registry(name, good_ids=GOOD, bad_ids=BAD, snap=snapshot_medias())
    settings.set_autofind_detectors([name])
    return detector_id


def _start(client, dataset_id: str) -> str:
    resp = client.post(f"/api/datasets/registry/{dataset_id}/autofind")
    assert resp.status_code == 200, resp.get_json()
    task_id = resp.get_json()["task_id"]
    assert task_id.startswith(TASK_PREFIX)
    return task_id


def _autofind_tasks() -> list[dict]:
    return [t for t in loading_tasks.list_tasks() if t["task_id"].startswith(TASK_PREFIX)]


class TestRunAutoFindRoute:
    def test_unknown_dataset_is_404(self, client):
        resp = client.post("/api/datasets/registry/does-not-exist/autofind")
        assert resp.status_code == 404

    def test_unloaded_dataset_is_409(self, client):
        from vtscore.datasets.registry import register_dataset

        entry = register_dataset(name="Not loaded", media_type="audio", num_items=1, pkl_path="/tmp/ar_nl.pkl")
        _autofind_detector()
        resp = client.post(f"/api/datasets/registry/{entry['id']}/autofind")
        assert resp.status_code == 409
        assert "Load the dataset" in resp.get_json()["message"]

    def test_no_autofind_detectors_is_400_and_starts_nothing(self, client):
        entry, _ctx = _registered_copy("No detectors")
        resp = client.post(f"/api/datasets/registry/{entry['id']}/autofind")
        assert resp.status_code == 400
        assert "no AutoFind detectors" in resp.get_json()["message"]
        assert _autofind_tasks() == []

    def test_runs_in_the_background_and_keeps_the_results(self, client):
        entry, ctx = _registered_copy("AutoFind target")
        _autofind_detector("ar-bg")

        task_id = _start(client, entry["id"])
        final = wait_for_loading_task(task_id)

        assert final["error"] is None, final
        block = final["autofind"]
        assert block["run_id"] == task_id
        assert block["trigger"] == "manual"
        assert block["owner"] == "default"
        assert block["dataset_id"] == entry["id"]
        assert block["dataset_name"] == "AutoFind target"
        assert block["detectors_run"] == 1

        resp = client.get(f"/api/autofind/runs/{task_id}")
        assert resp.status_code == 200, resp.get_json()
        data = resp.get_json()
        assert data["dataset_id"] == entry["id"]
        assert data["trigger"] == "manual"
        result = data["results"]["ar-bg"]
        scored = {h["id"] for h in result["hits"]} | {h["id"] for h in result["negative_hits"]}
        assert scored == set(ctx.medias), "every media of the dataset gets a verdict"
        assert block["total_hits"] == result["total_hits"]

    def test_task_row_is_keyed_to_the_dataset(self, client, monkeypatch):
        """Keyed by ``dataset_id``, the task renders inline on the dataset's row."""
        entry, _ctx = _registered_copy("Inline row")
        _autofind_detector()
        started = threading.Event()
        proceed = threading.Event()

        def fake_resolve(*_a, **_k):
            started.set()
            proceed.wait(timeout=10)
            return None, 0.5, None

        monkeypatch.setattr(autofind_mod, "resolve_or_train_detector", fake_resolve)
        task_id = _start(client, entry["id"])
        try:
            assert started.wait(timeout=10)
            (row,) = _autofind_tasks()
            assert row["task_id"] == task_id
            assert row["dataset_id"] == entry["id"]
            assert row["status"] != "idle"
        finally:
            proceed.set()
        wait_for_loading_task(task_id)

    def test_scores_the_named_dataset_not_the_requests_active_one(self, client):
        subset = {1, 2, 3, 4, 5, 18, 19, 20}
        entry, _ctx = _registered_copy("Subset", ids=subset)
        _autofind_detector("ar-subset")

        task_id = _start(client, entry["id"])
        assert wait_for_loading_task(task_id)["error"] is None

        run = get_autofind_run(task_id, "default")
        assert run is not None
        result = run["results"]["ar-subset"]
        scored = {h["id"] for h in result["hits"]} | {h["id"] for h in result["negative_hits"]}
        assert scored == subset

    def test_another_user_cannot_read_the_results(self, client):
        entry, _ctx = _registered_copy("Private")
        _autofind_detector()
        task_id = _start(client, entry["id"])
        wait_for_loading_task(task_id)

        assert get_autofind_run(task_id, "default") is not None
        assert get_autofind_run(task_id, "someone-else") is None
        assert client.get("/api/autofind/runs/_autofind_nope").status_code == 404

    def test_leaves_the_find_tracker_alone(self, client):
        """A cold train reports on the run's own task, not the shared Find bar."""
        entry, _ctx = _registered_copy("Quiet find")
        _autofind_detector()
        seen: list[str] = []

        def record(snap: dict) -> None:
            seen.append(snap["status"])

        find_progress.subscribe(record)
        try:
            task_id = _start(client, entry["id"])
            assert wait_for_loading_task(task_id)["error"] is None
        finally:
            find_progress.unsubscribe(record)

        assert "running" not in seen, "a background AutoFind painted the Find progress bar"

    def test_leaves_a_loaded_detectors_live_context_alone(self, client):
        entry, _ctx = _registered_copy("Live context")
        detector_id = _autofind_detector("ar-live")
        load_detector_and_wait(client, detector_id)
        det_ctx = get_detector_context(detector_id)
        assert det_ctx is not None
        model_before = det_ctx.model

        task_id = _start(client, entry["id"])
        assert wait_for_loading_task(task_id)["error"] is None

        assert det_ctx.model is model_before, "the run trained into the detector the user has loaded"

    def test_cancel_stops_the_run_and_keeps_nothing(self, client, monkeypatch):
        entry, _ctx = _registered_copy("Cancelled")
        _autofind_detector()
        started = threading.Event()
        proceed = threading.Event()

        def fake_resolve(*_a, **_k):
            started.set()
            proceed.wait(timeout=10)
            return None, 0.5, None

        monkeypatch.setattr(autofind_mod, "resolve_or_train_detector", fake_resolve)
        task_id = _start(client, entry["id"])
        assert started.wait(timeout=10)
        assert loading_tasks.cancel_task(task_id)
        proceed.set()

        assert wait_for_loading_task(task_id)["error"] == "Cancelled"
        assert get_autofind_run(task_id, "default") is None

    def test_sends_the_results_to_the_auto_find_exporter(self, client, tmp_path):
        out = tmp_path / "autofind.json"
        settings.set_autofind_exporter("server_json_file")
        settings.set_autofind_exporter_field_values({"server_json_file": {"filepath": str(out)}})
        entry, _ctx = _registered_copy("Exported")
        _autofind_detector()

        task_id = _start(client, entry["id"])
        final = wait_for_loading_task(task_id)

        assert final["autofind"]["auto_export"]["success"] is True, final["autofind"]
        assert out.exists()


class TestRunPickedDetectors:
    """``detector_ids``: the Dashboard's big Find button runs the ticked detectors (#4529)."""

    @staticmethod
    def _post(client, dataset_id: str, detector_ids):
        return client.post(f"/api/datasets/registry/{dataset_id}/autofind", json={"detector_ids": detector_ids})

    def test_runs_only_the_picked_detectors_drafts_included(self, client):
        entry, _ctx = _registered_copy("Picked")
        _autofind_detector("ar-listed")
        draft_id = setup_trainable_model_in_registry("ar-draft", good_ids=GOOD, bad_ids=BAD, snap=snapshot_medias())

        resp = self._post(client, entry["id"], [draft_id])
        assert resp.status_code == 200, resp.get_json()
        task_id = resp.get_json()["task_id"]
        final = wait_for_loading_task(task_id)

        assert final["error"] is None, final
        # A Find, not an AutoFind: the row and its notice say so (#4615).
        assert final["autofind"]["trigger"] == "find"
        assert final["message"].startswith("Find found ")
        run = get_autofind_run(task_id, "default")
        assert run is not None
        assert run["trigger"] == "find"
        assert set(run["results"]) == {"ar-draft"}, "the AutoFind list must not ride along"
        assert settings.get_autofind_detectors() == ["ar-listed"], "a picked draft is not moved to AutoFind"

    def test_runs_with_no_autofind_list_at_all(self, client):
        entry, _ctx = _registered_copy("No list")
        first = setup_trainable_model_in_registry("pick-1", good_ids=GOOD, bad_ids=BAD, snap=snapshot_medias())
        second = setup_trainable_model_in_registry("pick-2", good_ids=GOOD, bad_ids=BAD, snap=snapshot_medias())

        resp = self._post(client, entry["id"], [first, second, first])
        assert resp.status_code == 200, resp.get_json()
        task_id = resp.get_json()["task_id"]
        final = wait_for_loading_task(task_id)

        assert final["error"] is None, final
        assert final["autofind"]["detectors_run"] == 2, "a repeated id runs once"
        run = get_autofind_run(task_id, "default")
        assert run is not None
        assert set(run["results"]) == {"pick-1", "pick-2"}

    def test_empty_list_is_400_and_starts_nothing(self, client):
        entry, _ctx = _registered_copy("Empty pick")
        _autofind_detector()
        resp = self._post(client, entry["id"], [])
        assert resp.status_code == 400
        assert "Select at least one detector" in resp.get_json()["message"]
        assert _autofind_tasks() == []

    def test_unknown_id_is_404(self, client):
        entry, _ctx = _registered_copy("Unknown pick")
        resp = self._post(client, entry["id"], ["no-such-detector"])
        assert resp.status_code == 404
        assert _autofind_tasks() == []

    def test_another_users_private_detector_is_404(self, client):
        from vtscore.detectors.registry import register_detector

        entry, _ctx = _registered_copy("Private pick")
        theirs = register_detector(name="theirs", media_type="audio", num_training=6, created_by="someone-else")
        resp = self._post(client, entry["id"], [theirs["id"]])
        assert resp.status_code == 404
        assert _autofind_tasks() == []

    def test_picked_detectors_of_another_media_type_is_400(self, client):
        entry, _ctx = _registered_copy("Audio only")
        image_id = setup_trainable_model_in_registry(
            "pick-image", good_ids=GOOD, bad_ids=BAD, snap=snapshot_medias(), media_type="image"
        )
        resp = self._post(client, entry["id"], [image_id])
        assert resp.status_code == 400
        assert "None of the selected detectors are for audio datasets" in resp.get_json()["message"]

    @pytest.mark.parametrize("bad", [[1, 2], "det-id", {"id": "x"}])
    def test_non_list_of_strings_is_422(self, client, bad):
        entry, _ctx = _registered_copy("Malformed pick")
        assert self._post(client, entry["id"], bad).status_code == 422

    def test_plan_refuses_a_name_and_ids_together(self):
        with pytest.raises(ValueError, match="not both"):
            autofind_mod.plan_autofind(snapshot_medias(), detector_name="x", detector_ids=["y"])

    def test_explicit_null_runs_the_autofind_list(self, client):
        entry, _ctx = _registered_copy("Null pick")
        _autofind_detector("ar-null")
        resp = self._post(client, entry["id"], None)
        assert resp.status_code == 200, resp.get_json()
        task_id = resp.get_json()["task_id"]
        final = wait_for_loading_task(task_id)
        assert final["error"] is None
        assert final["autofind"]["trigger"] == "manual", "no picked detectors: the ⋯ menu's AutoFind"
        run = get_autofind_run(task_id, "default")
        assert run is not None
        assert set(run["results"]) == {"ar-null"}


class TestAutoFindAfterImport:
    def test_hook_starts_an_import_triggered_run(self, client):
        _entry, ctx = _registered_copy("Imported")
        _autofind_detector()

        hook = import_post_load("true")
        assert hook is not None
        hook(ctx)

        (row,) = _autofind_tasks()
        final = wait_for_loading_task(row["task_id"])
        assert final["error"] is None
        assert final["autofind"]["trigger"] == "import"
        assert final["autofind"]["dataset_id"] == ctx.dataset_id

    def test_hook_is_silent_for_a_user_with_no_autofind_detectors(self, client):
        _entry, ctx = _registered_copy("Nothing to run")

        hook = import_post_load("true")
        assert hook is not None
        hook(ctx)  # must not raise: most imports have nothing to run

        assert _autofind_tasks() == []

    def test_hook_reports_why_nothing_ran_when_no_detector_applies(self, client):
        """The user asked for AutoFind and got none: their browser is told why."""
        _entry, ctx = _registered_copy("Wrong media type")
        setup_trainable_model_in_registry(
            "ar-image-only", good_ids=GOOD, bad_ids=BAD, snap=snapshot_medias(), media_type="image"
        )
        settings.set_autofind_detectors(["ar-image-only"])

        hook = import_post_load("true")
        assert hook is not None
        hook(ctx)

        (row,) = _autofind_tasks()
        final = wait_for_loading_task(row["task_id"])
        assert final["status"] == "idle"
        assert final["error"] is None, "a skip is not a failure; it must not raise an error toast"
        assert final["autofind"]["trigger"] == "import"
        assert "audio datasets" in final["autofind"]["skipped"]
        assert get_autofind_run(row["task_id"], "default") is None


class TestImportPostLoad:
    def test_defaults_to_running(self):
        assert settings.get_autofind_on_import() is True
        assert import_post_load(None) is not None

    @pytest.mark.parametrize(("raw", "expected"), [("false", False), ("true", True), (False, False), ("0", False)])
    def test_a_sent_flag_decides_and_is_remembered(self, raw, expected):
        settings.set_autofind_on_import(not expected)
        hook = import_post_load(raw)
        assert (hook is not None) is expected
        assert settings.get_autofind_on_import() is expected

    @pytest.mark.parametrize("raw", [None, ""])
    def test_an_absent_flag_follows_the_setting(self, raw):
        settings.set_autofind_on_import(False)
        assert import_post_load(raw) is None
        assert settings.get_autofind_on_import() is False
        settings.set_autofind_on_import(True)
        assert import_post_load(raw) is not None


class TestImportRoutesPassTheChoice:
    """Each Add Dataset route resolves ``autofind`` into the load's ``post_load``."""

    def test_generic_importer_route(self, client, monkeypatch):
        import vtsearch.routes.datasets.staging as staging_mod

        captured: dict = {}

        def fake_run(importer, field_values, *, post_load=None, on_finished=None):
            captured.update(field_values=dict(field_values), post_load=post_load)
            return "task"

        monkeypatch.setattr(staging_mod, "_run_importer_in_background", fake_run)
        resp = client.post("/api/dataset/import/synthetic", json={"autofind": "false"})
        assert resp.status_code == 200, resp.get_json()
        assert captured["post_load"] is None
        assert "autofind" not in captured["field_values"], "the flag must not reach the importer"
        assert settings.get_autofind_on_import() is False

        resp = client.post("/api/dataset/import/synthetic", json={"autofind": "true"})
        assert resp.status_code == 200, resp.get_json()
        assert captured["post_load"] is not None
        assert settings.get_autofind_on_import() is True

    def test_demo_route(self, client, monkeypatch):
        import vtsearch.routes.datasets.load as load_mod
        from vtscore.datasets import DEMO_DATASETS

        captured: dict = {}

        def fake_run(importer, field_values, *, post_load=None, on_finished=None):
            captured.update(post_load=post_load)
            return "task"

        monkeypatch.setattr(load_mod, "_run_importer_in_background", fake_run)
        resp = client.post("/api/dataset/load-demo", json={"name": next(iter(DEMO_DATASETS)), "autofind": "false"})
        assert resp.status_code == 200, resp.get_json()
        assert captured["post_load"] is None
        assert settings.get_autofind_on_import() is False

    def test_local_folder_route(self, client, monkeypatch):
        import io

        import vtsearch.routes.datasets.load as load_mod

        captured: dict = {}

        def fake_run(load_fn, origin, **kwargs):
            captured.update(kwargs)
            return "task"

        monkeypatch.setattr(load_mod, "_run_origin_load_in_background", fake_run)
        resp = client.post(
            "/api/dataset/import-local-folder",
            data={"media_type": "audio", "autofind": "false", "files": (io.BytesIO(b"x"), "f/a.wav")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200, resp.get_json()
        assert captured["post_load"] is None
        assert settings.get_autofind_on_import() is False


class TestKeptRuns:
    """Kept results are bounded by run count and by total hits, oldest first."""

    @staticmethod
    def _record(n_hits: int) -> dict:
        hits = [{"id": i} for i in range(n_hits)]
        return {"owner": "default", "results": {"d": {"hits": hits, "negative_hits": []}}}

    def test_oldest_runs_go_past_the_run_cap(self, monkeypatch):
        monkeypatch.setattr(autofind_mod, "MAX_KEPT_RUNS", 2)
        for run_id in ("a", "b", "c"):
            autofind_mod._keep_run(run_id, self._record(1))
        assert get_autofind_run("a", "default") is None
        assert get_autofind_run("b", "default") is not None
        assert get_autofind_run("c", "default") is not None

    def test_oldest_runs_go_past_the_hit_budget_but_the_newest_stays(self, monkeypatch):
        monkeypatch.setattr(autofind_mod, "MAX_KEPT_HITS", 10)
        autofind_mod._keep_run("small", self._record(4))
        autofind_mod._keep_run("medium", self._record(5))
        assert get_autofind_run("small", "default") is not None
        autofind_mod._keep_run("huge", self._record(50))
        assert get_autofind_run("small", "default") is None
        assert get_autofind_run("medium", "default") is None
        assert get_autofind_run("huge", "default") is not None
