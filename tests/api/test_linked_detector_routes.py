"""A received (linked) detector is read-only at every route that changes a labelset or an identity (#5575).

Each mutating route answers 409 ``error_code: "detector_read_only"`` before
any side effect - the file is byte-identical and the in-memory votes are as
they were - while the same route works on a forked copy.  Find votes, which
never reach the labelset, stay open; so do the routes that change only
local, derived state (AutoFind, readers), and combine, which only ever
creates a new detector.

``test_the_guarded_routes_are_exactly_these`` pins the inventory, so a new
labelset-changing route has to be added here on purpose.
"""

from __future__ import annotations

import io
import json
from typing import Any

import pytest

import app as app_module
from tests import load_detector_and_wait
from tests.helpers import make_wav_bytes
from vtscore.datasets.labelset import LabelSet
from vtscore.detectors.labelset_elements import stable_element_id
from vtscore.detectors.registry import get_detector, update_detector
from vtscore.detectors.store import DetectorReadOnlyError, _detector_path, _read_detector, flush_detector_writes
from vtscore.state.core import get_detector_context
from vtsearch.errors import DETECTOR_READ_ONLY

OWNER = "alice"


def _make(client, name: str) -> str:
    """Create, load and label a detector: media 1 good, media 2 bad."""
    res = client.post("/api/detectors/registry", json={"name": name, "media_type": "audio", "text_query": "t"})
    assert res.status_code in (200, 201), res.get_data(as_text=True)
    detector_id = res.get_json()["detector"]["id"]
    load_detector_and_wait(client, detector_id)
    assert client.post("/api/medias/1/vote", json={"target": "good"}).status_code == 200
    assert client.post("/api/medias/2/vote", json={"target": "bad"}).status_code == 200
    return detector_id


@pytest.fixture
def detectors(client):
    """``{"forked": (id, name), "linked": (id, name)}``; the linked detector is loaded and active."""
    forked = _make(client, "My dogs")
    update_detector(forked, sync_mode="forked", forked_from={"detector_id": "x", "name": "Dog barks", "owner": OWNER})
    linked = _make(client, "Shared dogs")
    update_detector(
        linked,
        sync_mode="linked",
        labelset_source={
            "source_name": "shared_folder",
            "field_values": {},
            "remote_ref": "ref-1",
            "remote_owner": OWNER,
        },
    )
    return {"forked": (forked, "My dogs"), "linked": (linked, "Shared dogs")}


@pytest.fixture
def labels_file(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"labels": [{"md5": _md5(3), "label": "good"}]}), encoding="utf-8")
    return str(path)


def _md5(media_id: int) -> str:
    from vtsearch.state import get_media

    media = get_media(media_id)
    assert media is not None
    return media["md5"]


def _element(name: str) -> str:
    data = _read_detector(_detector_path(name))
    assert data is not None
    return stable_element_id(LabelSet.from_dict(data["labelset"]).elements[0])


#: ``(case id, method, path template, request kwargs, endpoint)``.  The path
#: template fills ``{id}``, ``{name}`` and ``{element}``; kwargs may name
#: ``{labels_file}``.  ``endpoint`` is the rule the guard sits on.
CASES = [
    ("vote", "post", "/api/medias/3/vote", {"json": {"target": "good"}}, ("POST", "/api/medias/<int:media_id>/vote")),
    ("vote-bulk", "post", "/api/medias/vote-bulk", {"json": {"target": "good", "ids": [3, 4]}}, None),
    ("add-to-pile", "post", "/api/medias/add-to-pile", {"pile": True}, None),
    ("precision-check-votes", "post", "/api/precision-check/votes", {"json": {"votes": []}}, None),
    ("votes-clear", "post", "/api/votes/clear", {}, None),
    ("labels-import", "post", "/api/labels/import", {"json": {"labels": "{md5_3_good}"}}, None),
    ("fill-from-sort", "post", "/api/labels/fill-from-sort", {"json": {"sort_results": [], "threshold": 0.5}}, None),
    (
        "label-importer-import",
        "post",
        "/api/label-importers/import/server_json_file",
        {"json": {"filepath": "{labels_file}"}},
        ("POST", "/api/label-importers/import/<importer_name>"),
    ),
    (
        "ingest-missing",
        "post",
        "/api/label-importers/ingest-missing",
        {"json": {"entries": [{"md5": "0" * 32, "label": "good"}]}},
        None,
    ),
    ("corrections-to-detector", "post", "/api/find/corrections-to-detector", {}, None),
    ("save-labels", "post", "/api/detectors/{name}/labels", {}, ("POST", "/api/detectors/<name>/labels")),
    (
        "import-labels",
        "post",
        "/api/detectors/{name}/import-labels/server_json_file",
        {"json": {"filepath": "{labels_file}"}},
        ("POST", "/api/detectors/<name>/import-labels/<importer_name>"),
    ),
    (
        "label-vote",
        "post",
        "/api/detectors/{name}/labels/{element}/vote",
        {"json": {"target": "remove"}},
        ("POST", "/api/detectors/<name>/labels/<element_id>/vote"),
    ),
    (
        "rename",
        "put",
        "/api/detectors/{name}/rename",
        {"json": {"new_name": "{name} renamed"}},
        ("PUT", "/api/detectors/<name>/rename"),
    ),
    (
        "registry-rename",
        "put",
        "/api/detectors/registry/{id}/rename",
        {"json": {"name": "{name} renamed"}},
        ("PUT", "/api/detectors/registry/<detector_id>/rename"),
    ),
    (
        "examples",
        "put",
        "/api/detectors/{name}/examples",
        {"json": {"examples": [{"type": "text", "value": "a dog"}]}},
        ("PUT", "/api/detectors/<name>/examples"),
    ),
    (
        "bind-labelset-source",
        "put",
        "/api/detectors/{id}/labelset-source",
        {"json": {}},
        ("PUT", "/api/detectors/<detector_ref>/labelset-source"),
    ),
    (
        "labelset-source-sync",
        "post",
        "/api/detectors/{id}/labelset-source/sync",
        {},
        ("POST", "/api/detectors/<detector_ref>/labelset-source/sync"),
    ),
]


#: Routes whose request needs more than a detector to succeed: on the forked
#: copy they answer their own precondition, not the read-only refusal.
PRECONDITIONS = {
    "precision-check-votes": (409, "No spot check is running"),
    "corrections-to-detector": (400, "No Tune run"),
}


def _endpoint(case) -> tuple[str, str]:
    _cid, method, path, _kwargs, endpoint = case
    return endpoint or (method.upper(), path)


def _fill(value: Any, **subs: str) -> Any:
    if isinstance(value, str):
        if value == "{md5_3_good}":
            return [{"md5": _md5(3), "label": "good"}]
        return value.format(**subs)
    if isinstance(value, dict):
        return {k: _fill(v, **subs) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill(v, **subs) for v in value]
    return value


def _call(client, case, detector_id: str, name: str, labels_file: str):
    _cid, method, path, kwargs, _endpoint = case
    subs = {"id": detector_id, "name": name, "labels_file": labels_file}
    if "{element}" in path:
        subs["element"] = _element(name)
    kwargs: dict[str, Any] = dict(kwargs)
    if kwargs.pop("pile", False):
        kwargs = {
            "data": {"file": (io.BytesIO(make_wav_bytes()), "pile.wav"), "label": "good"},
            "content_type": "multipart/form-data",
        }
    else:
        kwargs = _fill(kwargs, **subs)
    return getattr(client, method)(path.format(**subs), **kwargs)


def _votes(detector_id: str) -> tuple[dict, dict]:
    ctx = get_detector_context(detector_id)
    assert ctx is not None
    return dict(ctx.good_votes), dict(ctx.bad_votes)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_a_linked_detector_refuses_before_any_side_effect(client, detectors, labels_file, case):
    detector_id, name = detectors["linked"]
    flush_detector_writes()
    before = _detector_path(name).read_bytes()
    votes = _votes(detector_id)

    resp = _call(client, case, detector_id, name, labels_file)

    assert resp.status_code == 409, resp.get_data(as_text=True)
    body = resp.get_json()
    assert body["error_code"] == DETECTOR_READ_ONLY
    assert body["detector_id"] == detector_id
    assert body["remote_owner"] == OWNER
    assert "Fork" in body["message"]
    flush_detector_writes()
    assert _detector_path(name).read_bytes() == before
    assert _votes(detector_id) == votes
    entry = get_detector(detector_id)
    assert entry is not None and entry["name"] == name


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_the_same_route_works_on_a_forked_copy(client, detectors, labels_file, case):
    detector_id, name = detectors["forked"]
    load_detector_and_wait(client, detector_id)  # switching away from the linked detector must not fail

    resp = _call(client, case, detector_id, name, labels_file)

    assert (resp.get_json() or {}).get("error_code") != DETECTOR_READ_ONLY
    expected = PRECONDITIONS.get(case[0])
    if expected is None:
        assert 200 <= resp.status_code < 300, resp.get_data(as_text=True)
    else:
        status, message = expected
        assert resp.status_code == status, resp.get_data(as_text=True)
        assert message in resp.get_json()["message"]


def test_the_guarded_routes_are_exactly_these():
    guarded = set()
    for rule in app_module.app.url_map.iter_rules():
        view = app_module.app.view_functions[rule.endpoint]
        while view is not None:
            if getattr(view, "refuses_linked_detector", None):
                guarded.update((m, rule.rule) for m in (rule.methods or set()) - {"HEAD", "OPTIONS"})
                break
            view = getattr(view, "__wrapped__", None)
    assert guarded == {_endpoint(c) for c in CASES}


class TestWhatStaysOpen:
    def test_find_votes_are_not_refused(self, client, detectors):
        detector_id, name = detectors["linked"]
        flush_detector_writes()
        before = _detector_path(name).read_bytes()
        ctx = get_detector_context(detector_id)
        assert ctx is not None
        ctx.find_mode = True
        try:
            assert client.post("/api/medias/3/vote", json={"target": "good"}).status_code == 200
        finally:
            ctx.find_mode = False
        flush_detector_writes()
        assert _detector_path(name).read_bytes() == before

    def test_autofind_and_readers(self, client, detectors):
        detector_id, _name = detectors["linked"]
        assert client.put(f"/api/detectors/registry/{detector_id}/autofind", json={"autofind": True}).status_code == 200
        assert client.put(f"/api/detectors/registry/{detector_id}/readers", json={"readers": ["*"]}).status_code == 200

    def test_a_linked_detector_may_be_a_combine_source(self, client, detectors):
        _linked_id, linked = detectors["linked"]
        _forked_id, forked = detectors["forked"]
        resp = client.post(
            "/api/detectors/combine", json={"names": [linked, forked], "new_name": "Both", "conflict_policy": "drop"}
        )
        assert resp.status_code == 201, resp.get_data(as_text=True)

    def test_unloading_does_not_rewrite_it(self, client, detectors):
        detector_id, name = detectors["linked"]
        flush_detector_writes()
        before = _detector_path(name).read_bytes()
        assert client.post(f"/api/detectors/registry/{detector_id}/unload").status_code == 200
        assert _detector_path(name).read_bytes() == before

    def test_deleting_the_local_copy(self, client, detectors):
        detector_id, name = detectors["linked"]
        assert client.delete(f"/api/detectors/registry/{detector_id}").status_code == 200
        assert get_detector(detector_id) is None


class TestListing:
    def test_sharing_state_without_field_values_and_a_note(self, client, detectors):
        linked_id, _ = detectors["linked"]
        forked_id, _ = detectors["forked"]
        rows = {d["id"]: d for d in client.get("/api/detectors/registry").get_json()["detectors"]}

        linked = rows[linked_id]
        assert linked["sync_mode"] == "linked"
        assert linked["sharing"]["remote_ref"] == "ref-1"
        assert linked["sharing"]["remote_owner"] == OWNER
        assert "field_values" not in linked["sharing"]
        assert "labelset_source" not in linked
        assert "sharing is off" in linked["sharing_note"]
        assert linked["forked_from"] is None

        forked = rows[forked_id]
        assert forked["sync_mode"] == "forked"
        assert forked["sharing"] is None and forked["sharing_note"] is None
        assert forked["forked_from"]["owner"] == OWNER

    def test_a_detector_from_before_sharing_lists_as_local(self, client):
        detector_id = _make(client, "Plain")
        row = next(d for d in client.get("/api/detectors/registry").get_json()["detectors"] if d["id"] == detector_id)
        assert row["sync_mode"] == "local"
        assert row["sharing"] is None


def test_a_store_refusal_is_a_409_with_the_code(client):
    from vtsearch.errors import _handle_detector_read_only

    with app_module.app.test_request_context("/api/anything"):
        resp, status = _handle_detector_read_only(DetectorReadOnlyError("det-1", "Shared dogs", OWNER))
    assert status == 409
    body = resp.get_json()
    assert body["error_code"] == DETECTOR_READ_ONLY
    assert body["remote_owner"] == OWNER
    assert body["detector_id"] == "det-1"
