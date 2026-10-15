"""Labelset-source auto-sync (#4913): bindings, paced pushes, merging pulls, routes.

A detector linked to a labelset source - by a binding on its registry entry,
or by inheriting the deployment's ``default_labelset_source`` - has its whole
labelset pushed there whenever its file is written, paced so a burst of votes
costs one push per window and two pushes of one detector never overlap.  A
pull merges what the source holds back into the file, which is how a detector
whose server lost its data comes back.
"""

from __future__ import annotations

import json
import threading

import pytest

from vtscore.detectors.registry import get_detector, register_detector, update_detector
from vtscore.detectors.store import _detector_path, _read_detector, _write_detector
from vtscore.labels import sync as label_sync
from vtscore.labels.binding import (
    OPT_OUT,
    labelset_source_status,
    resolve_labelset_source,
    unfixed_fields,
)
from vtscore.labels.sources import get_labelset_source
from vtscore.labels.sync import (
    flush_pending_label_syncs,
    reconcile_labelset_source,
    reset_label_sync_for_tests,
    sync_from_labelset_source,
)
from vtsearch import settings


def _binding(path) -> dict:
    return {"source_name": "server_json_file", "field_values": {"filepath": str(path)}}


def _entry(detector_id: str) -> dict:
    """The registry entry for *detector_id*, which the test made and must exist."""
    entry = get_detector(detector_id)
    assert entry is not None
    return entry


def _detector_file(name: str) -> dict:
    """*name*'s detector JSON, which the test wrote and must exist."""
    data = _read_detector(_detector_path(name))
    assert data is not None
    return data


def _make_detector(name: str, labels: list[dict], *, binding: dict | None = None, created_by: str = "default") -> dict:
    """Write *name*'s detector file, register it, and link *binding* (if any)."""
    _write_detector(_detector_path(name), {"name": name, "media_type": "audio", "labelset": {"labels": labels}})
    entry = register_detector(name=name, media_type="audio", num_training=len(labels), created_by=created_by)
    if binding is not None:
        update_detector(entry["id"], labelset_source=binding)
    return _entry(entry["id"])


def _rewrite_labels(name: str, labels: list[dict]) -> None:
    """Rewrite a detector's labels the way a vote does: one file write."""
    data = _detector_file(name)
    data["labelset"] = {"labels": labels}
    _write_detector(_detector_path(name), data)


def _remote_labels(path) -> list[dict]:
    return json.loads(path.read_text())["labels"]


def _local_labels(name: str) -> list[dict]:
    return _detector_file(name)["labelset"]["labels"]


def _md5s(labels: list[dict]) -> set[str]:
    return {el["md5"] for el in labels}


@pytest.fixture(autouse=True)
def _clean_sync_state():
    reset_label_sync_for_tests()
    yield
    reset_label_sync_for_tests()


@pytest.fixture
def counting_save(monkeypatch):
    """Count (and record the payloads of) every save the JSON source makes."""
    src = get_labelset_source("server_json_file")
    original = src.save
    calls: list[dict] = []

    def save(labelset, field_values):
        calls.append(labelset.to_dict())
        return original(labelset, field_values)

    monkeypatch.setattr(src, "save", save)
    return calls


GOOD_A = {"md5": "a" * 32, "label": "good"}
BAD_B = {"md5": "b" * 32, "label": "bad"}
GOOD_C = {"md5": "c" * 32, "label": "good"}


# ---------------------------------------------------------------------------
# Binding resolution
# ---------------------------------------------------------------------------


class TestBindingResolution:
    def test_no_binding_and_no_default_syncs_nowhere(self):
        entry = _make_detector("unbound", [GOOD_A])
        assert resolve_labelset_source(entry) is None
        assert labelset_source_status(entry) is None

    def test_explicit_binding(self, tmp_path):
        entry = _make_detector("explicit", [GOOD_A], binding=_binding(tmp_path / "x.json"))
        resolved = resolve_labelset_source(entry)
        assert resolved is not None
        assert resolved.source_name == "server_json_file"
        assert resolved.field_values == {"filepath": str(tmp_path / "x.json")}
        assert not resolved.inherited
        assert resolved.ready

    def test_detector_inherits_the_deployment_default(self, tmp_path):
        settings.set_default_labelset_source(_binding(tmp_path / "{detector_name}.json"))
        entry = _make_detector("inheritor", [GOOD_A])
        resolved = resolve_labelset_source(entry)
        assert resolved is not None and resolved.inherited
        assert resolved.field_values == {"filepath": str(tmp_path / "{detector_name}.json")}
        assert resolved.fixed_keys == frozenset({"filepath"})

    def test_explicit_binding_beats_the_default(self, tmp_path):
        settings.set_default_labelset_source({"source_name": "server_json_file", "field_values": {}})
        entry = _make_detector("own", [GOOD_A], binding=_binding(tmp_path / "own.json"))
        resolved = resolve_labelset_source(entry)
        assert resolved is not None and not resolved.inherited
        assert resolved.field_values["filepath"] == str(tmp_path / "own.json")

    def test_opt_out_beats_the_default(self, tmp_path):
        settings.set_default_labelset_source(_binding(tmp_path / "d.json"))
        entry = _make_detector("opted_out", [GOOD_A], binding={"source_name": OPT_OUT})
        assert resolve_labelset_source(entry) is None

    def test_fixed_values_overlay_the_detectors_own(self, tmp_path):
        fixed = str(tmp_path / "fixed.json")
        settings.set_default_labelset_source(_binding(fixed))
        entry = _make_detector("overlay", [GOOD_A], binding=_binding(tmp_path / "mine.json"))
        resolved = resolve_labelset_source(entry)
        assert resolved is not None
        # The admin fixed the path; a detector's own value cannot move it.
        assert resolved.field_values["filepath"] == fixed

    def test_missing_required_field_is_reported(self):
        settings.set_default_labelset_source({"source_name": "server_json_file", "field_values": {}})
        entry = _make_detector("missing", [GOOD_A])
        resolved = resolve_labelset_source(entry)
        assert resolved is not None
        assert resolved.missing_fields == ("filepath",)
        assert not resolved.ready
        status = labelset_source_status(entry)
        assert status == {
            "source_name": "server_json_file",
            "display_name": "Server JSON File",
            "inherited": True,
            "missing_fields": ["filepath"],
        }

    def test_unknown_source_syncs_nowhere(self):
        entry = _make_detector("ghost", [GOOD_A], binding={"source_name": "no_such_source", "field_values": {}})
        assert resolve_labelset_source(entry) is None

    def test_context_binding_still_honoured(self, tmp_path):
        """A source linked on a loaded context (the library-level way) still resolves."""
        from vtscore.state.core import DetectorContext, register_detector_context, unregister_detector_context

        entry = _make_detector("ctx_bound", [GOOD_A])
        ctx = DetectorContext(entry["id"], name="ctx_bound")
        ctx.labelset_source = _binding(tmp_path / "ctx.json")
        register_detector_context(ctx)
        try:
            resolved = resolve_labelset_source(entry)
            assert resolved is not None
            assert resolved.field_values["filepath"] == str(tmp_path / "ctx.json")
        finally:
            unregister_detector_context(entry["id"])

    def test_unfixed_fields_drop_what_the_default_fixes(self, tmp_path):
        source = get_labelset_source("server_json_file")
        assert [f.key for f in unfixed_fields(source, None)] == ["filepath"]
        assert [f.key for f in unfixed_fields(source, {"source_name": "server_json_file"})] == ["filepath"]
        assert unfixed_fields(source, _binding(tmp_path / "f.json")) == []


# ---------------------------------------------------------------------------
# Push
# ---------------------------------------------------------------------------


class TestPush:
    def test_a_detector_write_pushes_the_whole_labelset(self, tmp_path):
        remote = tmp_path / "remote.json"
        _make_detector("pushed", [GOOD_A], binding=_binding(remote))
        # Labels from datasets that are not loaded travel too: the payload is
        # the detector file, not the active dataset's votes.
        _rewrite_labels("pushed", [GOOD_A, BAD_B, GOOD_C])
        flush_pending_label_syncs()

        assert _md5s(_remote_labels(remote)) == {GOOD_A["md5"], BAD_B["md5"], GOOD_C["md5"]}
        meta = json.loads(remote.read_text())["detector_meta"]
        assert meta["media_type"] == "audio"
        assert "threshold" not in meta

    def test_the_default_source_resolves_detector_name_templates(self, tmp_path):
        settings.set_default_labelset_source(_binding(tmp_path / "{detector_name}.labels.json"))
        _make_detector("Templated", [GOOD_A])
        _rewrite_labels("Templated", [GOOD_A, BAD_B])
        flush_pending_label_syncs()
        assert _md5s(_remote_labels(tmp_path / "Templated.labels.json")) == {GOOD_A["md5"], BAD_B["md5"]}

    def test_unbound_detector_schedules_nothing(self, counting_save):
        _make_detector("quiet", [GOOD_A])
        _rewrite_labels("quiet", [GOOD_A, BAD_B])
        assert "quiet" not in label_sync._slots
        flush_pending_label_syncs()
        assert counting_save == []

    def test_detector_missing_a_field_value_is_not_pushed(self, counting_save):
        settings.set_default_labelset_source({"source_name": "server_json_file", "field_values": {}})
        _make_detector("incomplete", [GOOD_A])
        _rewrite_labels("incomplete", [GOOD_A, BAD_B])
        flush_pending_label_syncs()
        assert counting_save == []

    def test_live_threshold_travels_only_from_a_trained_head(self, tmp_path):
        from vtscore.state.core import DetectorContext, register_detector_context, unregister_detector_context

        remote = tmp_path / "meta.json"
        entry = _make_detector("thresholded", [GOOD_A], binding=_binding(remote))
        ctx = DetectorContext(entry["id"], name="thresholded", media_type="audio")
        register_detector_context(ctx)
        try:
            ctx.model = None
            ctx.threshold = 0.99
            _rewrite_labels("thresholded", [GOOD_A, BAD_B])
            flush_pending_label_syncs()
            assert "threshold" not in json.loads(remote.read_text())["detector_meta"]

            ctx.model = object()  # stands in for a trained MLP
            ctx.threshold = 0.31
            _rewrite_labels("thresholded", [GOOD_A, BAD_B, GOOD_C])
            flush_pending_label_syncs()
            assert json.loads(remote.read_text())["detector_meta"]["threshold"] == pytest.approx(0.31)
        finally:
            unregister_detector_context(entry["id"])

    def test_the_first_push_merges_what_the_source_holds(self, tmp_path):
        """A write before any load (the create itself) must not overwrite a backup."""
        remote = tmp_path / "backup.json"
        remote.write_text(json.dumps({"labels": [GOOD_A, BAD_B]}))
        _make_detector("fresh", [], binding=_binding(remote))
        _rewrite_labels("fresh", [GOOD_C])
        flush_pending_label_syncs()
        everything = {GOOD_A["md5"], BAD_B["md5"], GOOD_C["md5"]}
        assert _md5s(_remote_labels(remote)) == everything
        assert _md5s(_local_labels("fresh")) == everything

    def test_a_source_that_cannot_be_read_is_never_overwritten(self, tmp_path, counting_save):
        remote = tmp_path / "corrupt.json"
        remote.write_text("[not a label object]")
        _make_detector("careful", [GOOD_A], binding=_binding(remote))
        _rewrite_labels("careful", [GOOD_A, BAD_B])
        flush_pending_label_syncs()
        assert counting_save == []
        assert remote.read_text() == "[not a label object]"

    def test_an_unchanged_payload_is_not_pushed_again(self, tmp_path, counting_save):
        _make_detector("same", [GOOD_A], binding=_binding(tmp_path / "same.json"))
        _rewrite_labels("same", [GOOD_A, BAD_B])
        flush_pending_label_syncs()
        _rewrite_labels("same", [GOOD_A, BAD_B])
        flush_pending_label_syncs()
        assert len(counting_save) == 1

    def test_a_failed_push_is_retried_on_the_next_change(self, tmp_path, monkeypatch):
        remote = tmp_path / "flaky.json"
        _make_detector("flaky", [GOOD_A], binding=_binding(remote))
        src = get_labelset_source("server_json_file")
        original = src.save
        monkeypatch.setattr(src, "save", lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk gone")))
        _rewrite_labels("flaky", [GOOD_A, BAD_B])
        flush_pending_label_syncs()
        assert not remote.exists()

        monkeypatch.setattr(src, "save", original)
        _rewrite_labels("flaky", [GOOD_A, BAD_B, GOOD_C])
        flush_pending_label_syncs()
        assert len(_remote_labels(remote)) == 3

    def test_a_hook_failure_never_fails_the_write(self, tmp_path, monkeypatch):
        _make_detector("sturdy", [GOOD_A], binding=_binding(tmp_path / "s.json"))

        def boom(*_args, **_kwargs):
            raise RuntimeError("scheduling failed")

        monkeypatch.setattr(label_sync, "schedule_labelset_push", boom)
        _rewrite_labels("sturdy", [GOOD_A, BAD_B])
        assert len(_local_labels("sturdy")) == 2


# ---------------------------------------------------------------------------
# Pacing
# ---------------------------------------------------------------------------


class TestPacing:
    def test_a_burst_of_writes_is_one_push(self, tmp_path, counting_save):
        _make_detector("burst", [GOOD_A], binding=_binding(tmp_path / "burst.json"))
        for i in range(20):
            _rewrite_labels("burst", [GOOD_A, {"md5": f"{i:032d}", "label": "bad"}])
        flush_pending_label_syncs()
        assert len(counting_save) == 1
        # ...carrying the newest labels, read when the push ran.
        assert _md5s(counting_save[0]["labels"]) == {GOOD_A["md5"], f"{19:032d}"}

    def test_the_push_never_runs_on_the_writers_thread(self, tmp_path, monkeypatch):
        """Asserted structurally (no wall clock): the save is never on the caller's stack."""
        _make_detector("offthread", [GOOD_A], binding=_binding(tmp_path / "o.json"))
        src = get_labelset_source("server_json_file")
        original = src.save
        caller = threading.get_ident()
        in_write = False
        saves_on_caller: list[int] = []

        def save(labelset, field_values):
            if in_write and threading.get_ident() == caller:
                saves_on_caller.append(1)
            return original(labelset, field_values)

        monkeypatch.setattr(src, "save", save)
        in_write = True
        try:
            _rewrite_labels("offthread", [GOOD_A, BAD_B])
        finally:
            in_write = False
        assert saves_on_caller == []
        flush_pending_label_syncs()
        assert (tmp_path / "o.json").exists()

    def test_the_interval_holds_the_next_push_back(self, tmp_path, counting_save):
        settings.set_labelset_source_push_interval_seconds(3600)
        _make_detector("paced", [GOOD_A], binding=_binding(tmp_path / "p.json"))
        _rewrite_labels("paced", [GOOD_A, BAD_B])
        flush_pending_label_syncs()
        assert len(counting_save) == 1

        _rewrite_labels("paced", [GOOD_A, BAD_B, GOOD_C])
        slot = label_sync._slots["paced"]
        assert slot.timer is not None
        assert slot.timer.interval > 3000  # armed for the end of the window...
        assert len(counting_save) == 1  # ...not pushed yet
        flush_pending_label_syncs()  # a flush (shutdown) ignores the pace
        assert len(counting_save) == 2

    def test_a_change_during_a_push_rides_the_next_one(self, tmp_path, monkeypatch):
        settings.set_labelset_source_push_interval_seconds(0)
        _make_detector("overlap", [GOOD_A], binding=_binding(tmp_path / "ov.json"))
        src = get_labelset_source("server_json_file")
        original = src.save
        entered = threading.Event()
        release = threading.Event()
        second_done = threading.Event()
        payloads: list[dict] = []
        active = 0
        max_active = 0
        lock = threading.Lock()

        def save(labelset, field_values):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            try:
                payloads.append(labelset.to_dict())
                if len(payloads) == 1:
                    entered.set()
                    assert release.wait(timeout=10)
                return original(labelset, field_values)
            finally:
                with lock:
                    active -= 1
                if len(payloads) == 2:
                    second_done.set()

        monkeypatch.setattr(src, "save", save)
        _rewrite_labels("overlap", [GOOD_A, BAD_B])
        assert entered.wait(timeout=10), "the first push never started"
        _rewrite_labels("overlap", [GOOD_A, BAD_B, GOOD_C])  # lands mid-push
        release.set()
        assert second_done.wait(timeout=10), "the change made mid-push was never pushed"
        assert max_active == 1
        assert _md5s(payloads[1]["labels"]) == {GOOD_A["md5"], BAD_B["md5"], GOOD_C["md5"]}

    def test_reset_drops_a_pending_push_without_running_it(self, tmp_path):
        _make_detector("dropped", [GOOD_A], binding=_binding(tmp_path / "d.json"))
        _rewrite_labels("dropped", [GOOD_A, BAD_B])
        token = label_sync._slots["dropped"].token
        reset_label_sync_for_tests()
        # A timer that fired just as the reset landed runs as a no-op.
        label_sync._fire("dropped", token)
        assert not (tmp_path / "d.json").exists()

    def test_two_detectors_keep_separate_slots(self, tmp_path):
        _make_detector("left", [GOOD_A], binding=_binding(tmp_path / "left.json"))
        _make_detector("right", [BAD_B], binding=_binding(tmp_path / "right.json"))
        _rewrite_labels("left", [GOOD_A, GOOD_C])
        _rewrite_labels("right", [BAD_B])
        flush_pending_label_syncs()
        assert _md5s(_remote_labels(tmp_path / "left.json")) == {GOOD_A["md5"], GOOD_C["md5"]}
        assert _md5s(_remote_labels(tmp_path / "right.json")) == {BAD_B["md5"]}


# ---------------------------------------------------------------------------
# Pull
# ---------------------------------------------------------------------------


class TestPull:
    def test_reconcile_restores_labels_only_the_source_has(self, tmp_path):
        remote = tmp_path / "r.json"
        remote.write_text(json.dumps({"labels": [GOOD_A, BAD_B]}))
        entry = _make_detector("restored", [GOOD_A], binding=_binding(remote))

        assert reconcile_labelset_source("restored") == 1
        assert _md5s(_local_labels("restored")) == {GOOD_A["md5"], BAD_B["md5"]}
        assert _entry(entry["id"])["num_training"] == 2

    def test_reconcile_keeps_the_files_label_and_pushes_it_back(self, tmp_path):
        remote = tmp_path / "r.json"
        remote.write_text(json.dumps({"labels": [{"md5": GOOD_A["md5"], "label": "bad"}]}))
        _make_detector("local_wins", [GOOD_A, GOOD_C], binding=_binding(remote))

        reconcile_labelset_source("local_wins")
        assert {el["md5"]: el["label"] for el in _local_labels("local_wins")}[GOOD_A["md5"]] == "good"
        flush_pending_label_syncs()
        # The source was behind; the file's labels catch it up.
        assert {el["md5"]: el["label"] for el in _remote_labels(remote)} == {
            GOOD_A["md5"]: "good",
            GOOD_C["md5"]: "good",
        }

    def test_manual_sync_takes_the_sources_label(self, tmp_path):
        remote = tmp_path / "r.json"
        remote.write_text(json.dumps({"labels": [{"md5": GOOD_A["md5"], "label": "bad"}, GOOD_C]}))
        entry = _make_detector("source_wins", [GOOD_A, BAD_B], binding=_binding(remote))

        labels = sync_from_labelset_source(entry["id"])
        assert labels is not None and len(labels) == 2
        local = {el["md5"]: el["label"] for el in _local_labels("source_wins")}
        # The source's label won the disagreement; nothing was removed.
        assert local == {GOOD_A["md5"]: "bad", BAD_B["md5"]: "bad", GOOD_C["md5"]: "good"}

    def test_pull_folds_in_input_spec_but_never_the_threshold(self, tmp_path):
        remote = tmp_path / "r.json"
        remote.write_text(
            json.dumps(
                {
                    "labels": [],
                    "detector_meta": {
                        "media_type": "audio",
                        "input_spec": {"clipper": "sound_tiling", "clipper_params": {"duration": "2.0"}},
                        "threshold": 0.5,
                    },
                }
            )
        )
        _make_detector("meta_in", [GOOD_A], binding=_binding(remote))
        reconcile_labelset_source("meta_in")
        data = _detector_file("meta_in")
        assert data["input_spec"] == {"clipper": "sound_tiling", "clipper_params": {"duration": "2.0"}}
        assert "threshold" not in data

    def test_an_unreadable_source_never_fails_the_caller(self, tmp_path):
        remote = tmp_path / "broken.json"
        remote.write_text("[not a label object]")
        _make_detector("unreadable", [GOOD_A], binding=_binding(remote))
        assert reconcile_labelset_source("unreadable") == 0
        assert _local_labels("unreadable") == [GOOD_A]

    def test_sync_without_a_source_returns_none(self):
        entry = _make_detector("sourceless", [GOOD_A])
        assert sync_from_labelset_source(entry["id"]) is None

    def test_manual_sync_rederives_a_loaded_detectors_votes(self, tmp_path):
        from vtscore.state.core import DetectorContext, register_detector_context, unregister_detector_context
        from vtsearch.state import medias

        first_id = next((mid for mid, m in medias.items() if m.get("md5")), None)
        if first_id is None:
            pytest.skip("No test media with an md5")
        md5 = medias[first_id]["md5"]
        remote = tmp_path / "r.json"
        remote.write_text(json.dumps({"labels": [{"md5": md5, "label": "good"}]}))
        entry = _make_detector("loaded_pull", [], binding=_binding(remote))
        ctx = DetectorContext(entry["id"], name="loaded_pull", media_type="audio")
        register_detector_context(ctx)
        try:
            sync_from_labelset_source(entry["id"])
            assert first_id in ctx.good_votes
        finally:
            unregister_detector_context(entry["id"])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


class TestDefaultSourceRoute:
    def test_null_without_a_default(self, client):
        resp = client.get("/api/labelset-sources/default")
        assert resp.status_code == 200
        assert resp.get_json() is None

    def test_asks_only_for_the_fields_the_admin_left_open(self, client, tmp_path):
        settings.set_default_labelset_source({"source_name": "server_json_file", "field_values": {}})
        settings.set_labelset_source_push_interval_seconds(120)
        body = client.get("/api/labelset-sources/default").get_json()
        assert body["name"] == "server_json_file"
        assert [f["key"] for f in body["fields"]] == ["filepath"]
        assert body["fixed_keys"] == []
        assert body["push_interval_seconds"] == 120

        settings.set_default_labelset_source(_binding(tmp_path / "{detector_name}.json"))
        body = client.get("/api/labelset-sources/default").get_json()
        assert body["fields"] == []
        assert body["fixed_keys"] == ["filepath"]
        # The fixed value itself never leaves the server.
        assert str(tmp_path) not in json.dumps(body)


class TestDetectorSourceRoutes:
    def test_get_unknown_detector_is_404(self, client):
        assert client.get("/api/detectors/nonexistent/labelset-source").status_code == 404

    def test_get_unbound_detector_is_null(self, client):
        entry = _make_detector("route_unbound", [GOOD_A])
        resp = client.get(f"/api/detectors/{entry['id']}/labelset-source")
        assert resp.status_code == 200
        assert resp.get_json() is None

    def test_get_reports_an_inherited_default_without_its_values(self, client, tmp_path):
        settings.set_default_labelset_source(_binding(tmp_path / "{detector_name}.json"))
        entry = _make_detector("route_inherit", [GOOD_A])
        body = client.get(f"/api/detectors/{entry['id']}/labelset-source").get_json()
        assert body["source_name"] == "server_json_file"
        assert body["inherited"] is True
        assert body["field_values"] == {}
        assert body["fixed_keys"] == ["filepath"]
        assert body["missing_fields"] == []

    def test_put_persists_the_binding_on_the_registry(self, client, tmp_path):
        entry = _make_detector("route_put", [GOOD_A])
        resp = client.put(
            f"/api/detectors/{entry['id']}/labelset-source",
            json={"source_name": "server_json_file", "field_values": {"filepath": str(tmp_path / "p.json")}},
        )
        assert resp.status_code == 200, resp.get_json()
        # No loaded context needed: the binding lives on the registry entry.
        stored = _entry(entry["id"])["labelset_source"]
        assert stored == _binding(tmp_path / "p.json")
        body = client.get(f"/api/detectors/{entry['id']}/labelset-source").get_json()
        assert body["field_values"] == {"filepath": str(tmp_path / "p.json")}
        assert body["inherited"] is False
        flush_pending_label_syncs()
        assert _remote_labels(tmp_path / "p.json") == [GOOD_A]

    def test_put_also_accepts_the_detector_name(self, client, tmp_path):
        _make_detector("By Name", [GOOD_A])
        resp = client.put(
            "/api/detectors/By Name/labelset-source",
            json={"source_name": "server_json_file", "field_values": {"filepath": str(tmp_path / "n.json")}},
        )
        assert resp.status_code == 200

    def test_put_restores_labels_the_source_already_holds(self, client, tmp_path):
        remote = tmp_path / "backup.json"
        remote.write_text(json.dumps({"labels": [GOOD_A, BAD_B]}))
        entry = _make_detector("route_restore", [GOOD_A])
        resp = client.put(
            f"/api/detectors/{entry['id']}/labelset-source",
            json={"source_name": "server_json_file", "field_values": {"filepath": str(remote)}},
        )
        assert "Restored 1 label(s)" in resp.get_json()["message"]
        assert _md5s(_local_labels("route_restore")) == {GOOD_A["md5"], BAD_B["md5"]}

    def test_put_refuses_a_missing_required_value(self, client):
        entry = _make_detector("route_missing", [GOOD_A])
        resp = client.put(
            f"/api/detectors/{entry['id']}/labelset-source",
            json={"source_name": "server_json_file", "field_values": {}},
        )
        assert resp.status_code == 400
        assert _entry(entry["id"]).get("labelset_source") is None

    def test_put_never_stores_what_the_default_fixes(self, client, tmp_path):
        settings.set_default_labelset_source(_binding(tmp_path / "{detector_name}.json"))
        entry = _make_detector("route_fixed", [GOOD_A])
        resp = client.put(
            f"/api/detectors/{entry['id']}/labelset-source",
            json={"source_name": "server_json_file", "field_values": {"filepath": "/etc/elsewhere.json"}},
        )
        assert resp.status_code == 200, resp.get_json()
        assert _entry(entry["id"])["labelset_source"] == {"source_name": "server_json_file", "field_values": {}}

    def test_put_none_opts_out_and_empty_clears(self, client, tmp_path):
        settings.set_default_labelset_source(_binding(tmp_path / "{detector_name}.json"))
        entry = _make_detector("route_optout", [GOOD_A])
        assert (
            client.put(f"/api/detectors/{entry['id']}/labelset-source", json={"source_name": "none"}).status_code == 200
        )
        assert client.get(f"/api/detectors/{entry['id']}/labelset-source").get_json() is None
        assert client.put(f"/api/detectors/{entry['id']}/labelset-source", json={}).status_code == 200
        assert client.get(f"/api/detectors/{entry['id']}/labelset-source").get_json()["inherited"] is True

    def test_put_unknown_source_is_404(self, client):
        entry = _make_detector("route_unknown", [GOOD_A])
        resp = client.put(
            f"/api/detectors/{entry['id']}/labelset-source",
            json={"source_name": "nonexistent", "field_values": {}},
        )
        assert resp.status_code == 404

    def test_only_the_creator_may_change_the_binding(self, client, tmp_path):
        entry = _make_detector("route_theirs", [GOOD_A], created_by="alice")
        resp = client.put(
            f"/api/detectors/{entry['id']}/labelset-source",
            json={"source_name": "server_json_file", "field_values": {"filepath": str(tmp_path / "t.json")}},
        )
        assert resp.status_code == 403

    def test_sync_route_merges_and_counts(self, client, tmp_path):
        remote = tmp_path / "r.json"
        remote.write_text(json.dumps({"labels": [GOOD_A, BAD_B]}))
        entry = _make_detector("route_sync", [GOOD_A], binding=_binding(remote))
        body = client.post(f"/api/detectors/{entry['id']}/labelset-source/sync").get_json()
        assert body["ok"] is True
        assert body["labels"] == 2
        assert body["changed"] == 1

    def test_registry_listing_reports_each_detectors_status(self, client):
        settings.set_default_labelset_source({"source_name": "server_json_file", "field_values": {}})
        entry = _make_detector("listed", [GOOD_A])
        rows = client.get("/api/detectors/registry").get_json()["detectors"]
        row = next(r for r in rows if r["id"] == entry["id"])
        assert row["labelset_source_status"] == {
            "source_name": "server_json_file",
            "display_name": "Server JSON File",
            "inherited": True,
            "missing_fields": ["filepath"],
        }
        # The raw binding (with its field values) is never listed.
        assert "labelset_source" not in row


class TestRecoveryOnLoad:
    def test_recreating_a_lost_detector_restores_its_labels_on_load(self, client, tmp_path):
        """The server lost the detector; its source kept the labels.  Re-create, load, done."""
        from tests import load_detector_and_wait
        from vtsearch.state import medias

        voted = [mid for mid, m in medias.items() if m.get("md5")][:2]
        if len(voted) < 2:
            pytest.skip("Need two test medias with an md5")
        settings.set_default_labelset_source(_binding(tmp_path / "{detector_name}.labels.json"))
        backup = [{"md5": medias[voted[0]]["md5"], "label": "good"}, {"md5": medias[voted[1]]["md5"], "label": "bad"}]
        (tmp_path / "Barks.labels.json").write_text(json.dumps({"labels": backup}))

        created = client.post(
            "/api/detectors/registry", json={"name": "Barks", "media_type": "audio", "text_query": "dog"}
        )
        assert created.status_code == 201
        detector_id = created.get_json()["detector"]["id"]
        load_detector_and_wait(client, detector_id)

        assert _md5s(_local_labels("Barks")) >= {el["md5"] for el in backup}
        from vtscore.state.core import get_detector_context

        ctx = get_detector_context(detector_id)
        assert ctx is not None
        assert voted[0] in ctx.good_votes
        assert voted[1] in ctx.bad_votes
