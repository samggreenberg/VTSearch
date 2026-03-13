"""Tests for multi-user dataset access: readers list and access control.

Verifies that:
- Datasets have a ``readers`` field controlling who can see/load them
- Only the creator and listed readers (or ``"*"`` for public) can see a dataset
- Only the creator can modify readers, rename, or delete a dataset
- The ``PUT /api/datasets/registry/<id>/readers`` endpoint works correctly
- The ``GET /api/datasets/registry`` endpoint filters by access
- Load, delete, and rename enforce access/ownership checks
"""

from __future__ import annotations

import pytest

from vtsearch.auth import (
    DefaultLoginProvider,
    LoginProvider,
    get_login_provider,
    set_login_provider,
    TrivialLoginProvider,
)
from vtsearch.datasets.registry import (
    can_user_access,
    is_owner,
    list_datasets_for_user,
    register_dataset,
    set_readers,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(name="ds", created_by="alice", readers=None, **kw):
    """Register a dataset and return the entry."""
    return register_dataset(
        name=name,
        media_type="audio",
        num_items=10,
        pkl_path=f"/tmp/{name}.pkl",
        created_by=created_by,
        readers=readers,
        **kw,
    )


# ---------------------------------------------------------------------------
# Registry-level access control
# ---------------------------------------------------------------------------


class TestReadersField:
    """Test that the readers field is stored and returned."""

    def test_default_readers_is_empty_list(self):
        entry = _make_dataset("ds1")
        assert entry["readers"] == []

    def test_readers_stored_on_register(self):
        entry = _make_dataset("ds2", readers=["bob", "carol"])
        assert entry["readers"] == ["bob", "carol"]

    def test_wildcard_readers(self):
        entry = _make_dataset("ds3", readers=["*"])
        assert entry["readers"] == ["*"]


class TestCanUserAccess:
    """Test the can_user_access() helper."""

    def test_creator_always_has_access(self):
        entry = _make_dataset("acc1", created_by="alice")
        assert can_user_access(entry["id"], "alice") is True

    def test_non_reader_denied(self):
        entry = _make_dataset("acc2", created_by="alice")
        assert can_user_access(entry["id"], "bob") is False

    def test_listed_reader_allowed(self):
        entry = _make_dataset("acc3", created_by="alice", readers=["bob"])
        assert can_user_access(entry["id"], "bob") is True

    def test_unlisted_user_denied(self):
        entry = _make_dataset("acc4", created_by="alice", readers=["bob"])
        assert can_user_access(entry["id"], "carol") is False

    def test_wildcard_grants_all(self):
        entry = _make_dataset("acc5", created_by="alice", readers=["*"])
        assert can_user_access(entry["id"], "anyone") is True

    def test_nonexistent_dataset(self):
        assert can_user_access("nonexistent_id", "alice") is False


class TestIsOwner:
    """Test the is_owner() helper."""

    def test_creator_is_owner(self):
        entry = _make_dataset("own1", created_by="alice")
        assert is_owner(entry["id"], "alice") is True

    def test_reader_is_not_owner(self):
        entry = _make_dataset("own2", created_by="alice", readers=["bob"])
        assert is_owner(entry["id"], "bob") is False

    def test_nonexistent_dataset(self):
        assert is_owner("nonexistent_id", "alice") is False


class TestListDatasetsForUser:
    """Test filtered dataset listing."""

    def test_creator_sees_own_private_dataset(self):
        entry = _make_dataset("list1", created_by="alice")
        results = list_datasets_for_user("alice")
        ids = [r["id"] for r in results]
        assert entry["id"] in ids

    def test_other_user_cannot_see_private_dataset(self):
        entry = _make_dataset("list2", created_by="alice")
        results = list_datasets_for_user("bob")
        ids = [r["id"] for r in results]
        assert entry["id"] not in ids

    def test_reader_sees_shared_dataset(self):
        entry = _make_dataset("list3", created_by="alice", readers=["bob"])
        results = list_datasets_for_user("bob")
        ids = [r["id"] for r in results]
        assert entry["id"] in ids

    def test_wildcard_visible_to_everyone(self):
        entry = _make_dataset("list4", created_by="alice", readers=["*"])
        results = list_datasets_for_user("zed")
        ids = [r["id"] for r in results]
        assert entry["id"] in ids


class TestSetReaders:
    """Test the set_readers() function."""

    def test_creator_can_set_readers(self):
        entry = _make_dataset("sr1", created_by="alice")
        ok, err = set_readers(entry["id"], ["bob", "carol"], "alice")
        assert ok is True
        assert err == ""
        # Verify bob now has access
        assert can_user_access(entry["id"], "bob") is True

    def test_non_creator_cannot_set_readers(self):
        entry = _make_dataset("sr2", created_by="alice")
        ok, err = set_readers(entry["id"], ["bob"], "bob")
        assert ok is False
        assert "creator" in err

    def test_set_readers_nonexistent_dataset(self):
        ok, err = set_readers("nonexistent_id", ["bob"], "alice")
        assert ok is False
        assert "not found" in err.lower()

    def test_set_readers_to_empty_revokes_access(self):
        entry = _make_dataset("sr3", created_by="alice", readers=["bob"])
        assert can_user_access(entry["id"], "bob") is True
        ok, _ = set_readers(entry["id"], [], "alice")
        assert ok is True
        assert can_user_access(entry["id"], "bob") is False


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


class TestReadersEndpoint:
    """Test PUT /api/datasets/registry/<id>/readers."""

    def test_set_readers_via_api(self, client):
        entry = _make_dataset("api1", created_by="default")
        resp = client.put(
            f"/api/datasets/registry/{entry['id']}/readers",
            json={"readers": ["bob", "carol"]},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["readers"] == ["bob", "carol"]

    def test_set_readers_invalid_body(self, client):
        entry = _make_dataset("api2", created_by="default")
        resp = client.put(
            f"/api/datasets/registry/{entry['id']}/readers",
            json={"readers": "not_a_list"},
        )
        assert resp.status_code == 400

    def test_set_readers_non_string_items(self, client):
        entry = _make_dataset("api3", created_by="default")
        resp = client.put(
            f"/api/datasets/registry/{entry['id']}/readers",
            json={"readers": [123]},
        )
        assert resp.status_code == 400

    def test_set_readers_non_creator_forbidden(self, client):
        entry = _make_dataset("api4", created_by="other_user")
        resp = client.put(
            f"/api/datasets/registry/{entry['id']}/readers",
            json={"readers": ["bob"]},
        )
        assert resp.status_code == 403

    def test_set_readers_nonexistent_dataset(self, client):
        resp = client.put(
            "/api/datasets/registry/nonexistent/readers",
            json={"readers": ["bob"]},
        )
        assert resp.status_code == 404


class TestRegistryListingFiltered:
    """Test that GET /api/datasets/registry filters by user access."""

    def test_default_user_sees_own_datasets(self, client):
        entry = _make_dataset("filt1", created_by="default")
        resp = client.get("/api/datasets/registry")
        assert resp.status_code == 200
        ids = [d["id"] for d in resp.get_json()["datasets"]]
        assert entry["id"] in ids

    def test_default_user_cannot_see_others_private_datasets(self, client):
        entry = _make_dataset("filt2", created_by="other_user")
        resp = client.get("/api/datasets/registry")
        ids = [d["id"] for d in resp.get_json()["datasets"]]
        assert entry["id"] not in ids

    def test_default_user_sees_shared_datasets(self, client):
        entry = _make_dataset("filt3", created_by="other_user", readers=["default"])
        resp = client.get("/api/datasets/registry")
        ids = [d["id"] for d in resp.get_json()["datasets"]]
        assert entry["id"] in ids

    def test_public_dataset_visible_to_all(self, client):
        entry = _make_dataset("filt4", created_by="other_user", readers=["*"])
        resp = client.get("/api/datasets/registry")
        ids = [d["id"] for d in resp.get_json()["datasets"]]
        assert entry["id"] in ids

    def test_readers_field_in_response(self, client):
        entry = _make_dataset("filt5", created_by="default", readers=["bob"])
        resp = client.get("/api/datasets/registry")
        datasets = resp.get_json()["datasets"]
        match = [d for d in datasets if d["id"] == entry["id"]]
        assert len(match) == 1
        assert match[0]["readers"] == ["bob"]


class TestAccessEnforcementOnLoad:
    """Test that load checks access."""

    def test_load_private_dataset_denied(self, client):
        """A user cannot load a dataset they don't have access to."""
        entry = _make_dataset("load1", created_by="other_user")
        resp = client.post(f"/api/datasets/registry/{entry['id']}/load")
        assert resp.status_code == 403


class TestOwnershipEnforcementOnDelete:
    """Test that delete checks ownership."""

    def test_delete_others_dataset_denied(self, client):
        entry = _make_dataset("del1", created_by="other_user")
        resp = client.delete(f"/api/datasets/registry/{entry['id']}")
        assert resp.status_code == 403

    def test_reader_cannot_delete(self, client):
        entry = _make_dataset("del2", created_by="other_user", readers=["default"])
        resp = client.delete(f"/api/datasets/registry/{entry['id']}")
        assert resp.status_code == 403


class TestOwnershipEnforcementOnRename:
    """Test that rename checks ownership."""

    def test_rename_others_dataset_denied(self, client):
        entry = _make_dataset("ren1", created_by="other_user")
        resp = client.put(
            f"/api/datasets/registry/{entry['id']}/rename",
            json={"name": "new_name"},
        )
        assert resp.status_code == 403

    def test_reader_cannot_rename(self, client):
        entry = _make_dataset("ren2", created_by="other_user", readers=["default"])
        resp = client.put(
            f"/api/datasets/registry/{entry['id']}/rename",
            json={"name": "new_name"},
        )
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Multi-user flow with TrivialLoginProvider
# ---------------------------------------------------------------------------


class TestMultiUserDatasetFlow:
    """End-to-end test: alice creates, shares with bob, bob can see it."""

    def test_share_and_access_flow(self, client):
        original = get_login_provider()
        try:
            set_login_provider(TrivialLoginProvider())

            # Alice logs in and creates a dataset
            client.post("/api/auth/login", json={"username": "alice"})
            entry = _make_dataset("flow1", created_by="alice")

            # Alice's registry shows the dataset
            resp = client.get("/api/datasets/registry")
            ids = [d["id"] for d in resp.get_json()["datasets"]]
            assert entry["id"] in ids

            # Bob logs in — can't see alice's private dataset
            client.post("/api/auth/login", json={"username": "bob"})
            resp = client.get("/api/datasets/registry")
            ids = [d["id"] for d in resp.get_json()["datasets"]]
            assert entry["id"] not in ids

            # Alice shares with bob
            client.post("/api/auth/login", json={"username": "alice"})
            resp = client.put(
                f"/api/datasets/registry/{entry['id']}/readers",
                json={"readers": ["bob"]},
            )
            assert resp.status_code == 200

            # Bob can now see the dataset
            client.post("/api/auth/login", json={"username": "bob"})
            resp = client.get("/api/datasets/registry")
            ids = [d["id"] for d in resp.get_json()["datasets"]]
            assert entry["id"] in ids

            # But bob can't delete it
            resp = client.delete(f"/api/datasets/registry/{entry['id']}")
            assert resp.status_code == 403

            # And bob can't change readers
            resp = client.put(
                f"/api/datasets/registry/{entry['id']}/readers",
                json={"readers": ["bob", "carol"]},
            )
            assert resp.status_code == 403

        finally:
            set_login_provider(original)

    def test_public_dataset_flow(self, client):
        original = get_login_provider()
        try:
            set_login_provider(TrivialLoginProvider())

            # Alice creates a public dataset
            client.post("/api/auth/login", json={"username": "alice"})
            entry = _make_dataset("flow2", created_by="alice", readers=["*"])

            # Random user can see it
            client.post("/api/auth/login", json={"username": "zed"})
            resp = client.get("/api/datasets/registry")
            ids = [d["id"] for d in resp.get_json()["datasets"]]
            assert entry["id"] in ids

        finally:
            set_login_provider(original)

    def test_revoke_access_flow(self, client):
        original = get_login_provider()
        try:
            set_login_provider(TrivialLoginProvider())

            # Alice creates and shares with bob
            client.post("/api/auth/login", json={"username": "alice"})
            entry = _make_dataset("flow3", created_by="alice", readers=["bob"])

            # Bob can see it
            client.post("/api/auth/login", json={"username": "bob"})
            resp = client.get("/api/datasets/registry")
            ids = [d["id"] for d in resp.get_json()["datasets"]]
            assert entry["id"] in ids

            # Alice revokes
            client.post("/api/auth/login", json={"username": "alice"})
            resp = client.put(
                f"/api/datasets/registry/{entry['id']}/readers",
                json={"readers": []},
            )
            assert resp.status_code == 200

            # Bob can no longer see it
            client.post("/api/auth/login", json={"username": "bob"})
            resp = client.get("/api/datasets/registry")
            ids = [d["id"] for d in resp.get_json()["datasets"]]
            assert entry["id"] not in ids

        finally:
            set_login_provider(original)
