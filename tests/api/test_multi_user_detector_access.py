"""Tests for multi-user detector access: readers list and access control.

Detectors are user-shared just like datasets (see
``test_multi_user_dataset_access.py``). Verifies that:

- Detectors have a ``readers`` field controlling who can see/load them
- Only the creator and listed readers (or ``"*"`` for public) can see one
- Only the creator can modify readers, rename, or delete a detector
- ``PUT /api/detectors/registry/<id>/readers`` works correctly
- ``GET /api/detectors/registry`` filters by access
- Load, delete, rename, and autorun-toggle enforce access/ownership
"""

from __future__ import annotations

from vtsearch.auth import (
    get_login_provider,
    set_login_provider,
    TrivialLoginProvider,
)
from vtscore.detectors.registry import (
    can_user_access_detector,
    is_detector_owner,
    list_detectors_for_user,
    register_detector,
    set_detector_readers,
)


def _make_detector(name="det", created_by="alice", readers=None, **kw):
    """Register a detector and return the entry."""
    return register_detector(
        name=name,
        media_type="audio",
        created_by=created_by,
        readers=readers,
        **kw,
    )


# ---------------------------------------------------------------------------
# Registry-level access control
# ---------------------------------------------------------------------------


class TestReadersField:
    def test_default_readers_is_empty_list(self):
        assert _make_detector("d1")["readers"] == []

    def test_readers_stored_on_register(self):
        assert _make_detector("d2", readers=["bob", "carol"])["readers"] == ["bob", "carol"]

    def test_wildcard_readers(self):
        assert _make_detector("d3", readers=["*"])["readers"] == ["*"]


class TestCanUserAccess:
    def test_creator_always_has_access(self):
        e = _make_detector("acc1", created_by="alice")
        assert can_user_access_detector(e["id"], "alice") is True

    def test_non_reader_denied(self):
        e = _make_detector("acc2", created_by="alice")
        assert can_user_access_detector(e["id"], "bob") is False

    def test_listed_reader_allowed(self):
        e = _make_detector("acc3", created_by="alice", readers=["bob"])
        assert can_user_access_detector(e["id"], "bob") is True

    def test_wildcard_grants_all(self):
        e = _make_detector("acc4", created_by="alice", readers=["*"])
        assert can_user_access_detector(e["id"], "anyone") is True

    def test_nonexistent_detector(self):
        assert can_user_access_detector("nope", "alice") is False


class TestIsOwner:
    def test_creator_is_owner(self):
        e = _make_detector("own1", created_by="alice")
        assert is_detector_owner(e["id"], "alice") is True

    def test_reader_is_not_owner(self):
        e = _make_detector("own2", created_by="alice", readers=["bob"])
        assert is_detector_owner(e["id"], "bob") is False


class TestListDetectorsForUser:
    def test_creator_sees_own_private_detector(self):
        e = _make_detector("l1", created_by="alice")
        assert e["id"] in [r["id"] for r in list_detectors_for_user("alice")]

    def test_other_user_cannot_see_private_detector(self):
        e = _make_detector("l2", created_by="alice")
        assert e["id"] not in [r["id"] for r in list_detectors_for_user("bob")]

    def test_reader_sees_shared_detector(self):
        e = _make_detector("l3", created_by="alice", readers=["bob"])
        assert e["id"] in [r["id"] for r in list_detectors_for_user("bob")]

    def test_wildcard_visible_to_everyone(self):
        e = _make_detector("l4", created_by="alice", readers=["*"])
        assert e["id"] in [r["id"] for r in list_detectors_for_user("zed")]


class TestSetReaders:
    def test_creator_can_set_readers(self):
        e = _make_detector("sr1", created_by="alice")
        ok, err = set_detector_readers(e["id"], ["bob"], "alice")
        assert ok is True and err == ""
        assert can_user_access_detector(e["id"], "bob") is True

    def test_non_creator_cannot_set_readers(self):
        e = _make_detector("sr2", created_by="alice")
        ok, err = set_detector_readers(e["id"], ["bob"], "bob")
        assert ok is False and "creator" in err

    def test_set_readers_nonexistent(self):
        ok, err = set_detector_readers("nope", ["bob"], "alice")
        assert ok is False and "not found" in err.lower()

    def test_set_readers_to_empty_revokes(self):
        e = _make_detector("sr3", created_by="alice", readers=["bob"])
        assert can_user_access_detector(e["id"], "bob") is True
        set_detector_readers(e["id"], [], "alice")
        assert can_user_access_detector(e["id"], "bob") is False


# ---------------------------------------------------------------------------
# API endpoint tests (default user = "default")
# ---------------------------------------------------------------------------


class TestReadersEndpoint:
    def test_set_readers_via_api(self, client):
        e = _make_detector("api1", created_by="default")
        resp = client.put(f"/api/detectors/registry/{e['id']}/readers", json={"readers": ["bob", "carol"]})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True and data["readers"] == ["bob", "carol"]

    def test_set_readers_non_string_items(self, client):
        e = _make_detector("api2", created_by="default")
        resp = client.put(f"/api/detectors/registry/{e['id']}/readers", json={"readers": [123]})
        assert resp.status_code == 422

    def test_set_readers_non_creator_forbidden(self, client):
        e = _make_detector("api3", created_by="other_user")
        resp = client.put(f"/api/detectors/registry/{e['id']}/readers", json={"readers": ["bob"]})
        assert resp.status_code == 403

    def test_set_readers_nonexistent(self, client):
        resp = client.put("/api/detectors/registry/nope/readers", json={"readers": ["bob"]})
        assert resp.status_code == 404


class TestRegistryListingFiltered:
    def test_default_user_sees_own(self, client):
        e = _make_detector("f1", created_by="default")
        ids = [d["id"] for d in client.get("/api/detectors/registry").get_json()["detectors"]]
        assert e["id"] in ids

    def test_default_user_cannot_see_others_private(self, client):
        e = _make_detector("f2", created_by="other_user")
        ids = [d["id"] for d in client.get("/api/detectors/registry").get_json()["detectors"]]
        assert e["id"] not in ids

    def test_default_user_sees_shared(self, client):
        e = _make_detector("f3", created_by="other_user", readers=["default"])
        ids = [d["id"] for d in client.get("/api/detectors/registry").get_json()["detectors"]]
        assert e["id"] in ids

    def test_response_has_readers_and_is_owner(self, client):
        e = _make_detector("f4", created_by="default", readers=["bob"])
        match = [d for d in client.get("/api/detectors/registry").get_json()["detectors"] if d["id"] == e["id"]]
        assert len(match) == 1
        assert match[0]["readers"] == ["bob"]
        assert match[0]["is_owner"] is True


class TestAccessEnforcement:
    def test_load_private_detector_denied(self, client):
        e = _make_detector("load1", created_by="other_user")
        resp = client.post("/api/detectors/registry/load", json={"detector_id": e["id"]})
        assert resp.status_code == 403

    def test_delete_others_detector_denied(self, client):
        e = _make_detector("del1", created_by="other_user")
        assert client.delete(f"/api/detectors/registry/{e['id']}").status_code == 403

    def test_rename_others_detector_denied(self, client):
        e = _make_detector("ren1", created_by="other_user")
        resp = client.put(f"/api/detectors/registry/{e['id']}/rename", json={"name": "new"})
        assert resp.status_code == 403

    def test_autorun_toggle_on_inaccessible_denied(self, client):
        e = _make_detector("ar1", created_by="other_user")
        resp = client.put(f"/api/detectors/registry/{e['id']}/autorun", json={"autorun": True})
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Multi-user flow with TrivialLoginProvider
# ---------------------------------------------------------------------------


class TestMultiUserDetectorFlow:
    def test_share_and_access_flow(self, client):
        original = get_login_provider()
        try:
            set_login_provider(TrivialLoginProvider())

            client.post("/api/auth/login", json={"username": "alice"})
            e = _make_detector("flow1", created_by="alice")
            ids = [d["id"] for d in client.get("/api/detectors/registry").get_json()["detectors"]]
            assert e["id"] in ids

            client.post("/api/auth/login", json={"username": "bob"})
            ids = [d["id"] for d in client.get("/api/detectors/registry").get_json()["detectors"]]
            assert e["id"] not in ids

            client.post("/api/auth/login", json={"username": "alice"})
            assert (
                client.put(f"/api/detectors/registry/{e['id']}/readers", json={"readers": ["bob"]}).status_code == 200
            )

            client.post("/api/auth/login", json={"username": "bob"})
            ids = [d["id"] for d in client.get("/api/detectors/registry").get_json()["detectors"]]
            assert e["id"] in ids
            # Bob (a reader) cannot delete or re-share.
            assert client.delete(f"/api/detectors/registry/{e['id']}").status_code == 403
            assert client.put(f"/api/detectors/registry/{e['id']}/readers", json={"readers": ["x"]}).status_code == 403
        finally:
            set_login_provider(original)

    def test_per_user_autorun_is_isolated(self, client):
        """Flagging a detector for autorun affects only the calling user."""
        original = get_login_provider()
        try:
            set_login_provider(TrivialLoginProvider())

            # Alice creates a public detector and flags it for her autorun.
            client.post("/api/auth/login", json={"username": "alice"})
            e = _make_detector("flow2", created_by="alice", readers=["*"])
            assert client.put(f"/api/detectors/registry/{e['id']}/autorun", json={"autorun": True}).status_code == 200
            alice_autorun = client.get("/api/settings").get_json()["autorun_detectors"]
            assert e["name"] in alice_autorun

            # Bob can see it (public) but his autorun list is untouched.
            client.post("/api/auth/login", json={"username": "bob"})
            bob_autorun = client.get("/api/settings").get_json()["autorun_detectors"]
            assert e["name"] not in bob_autorun
        finally:
            set_login_provider(original)
