"""The detector-sharing routes (#5577), end to end over the ``shared_folder`` source.

Alice shares a detector of hers with Bob; Bob lists what is shared with him,
receives it as a read-only (linked) detector, refreshes it, and forks it.
``simulate_foreign_modification`` plays a recipient who writes to the
container anyway, which a refresh's integrity check catches.

Two users log in through ``TrivialLoginProvider``, as
``test_multi_user_detector_access.py`` does.  ``shared_folder`` stands in for
the external service both would reach: its ``folder`` is a server path, which a
multi-user provider would confine to each user's own data directory, so the
``users`` fixture lifts that confinement for these tests (the source's
docstring says to run it single-user for that reason).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vtsearch.auth import TrivialLoginProvider, get_login_provider, set_login_provider
from vtscore.config import register_setting_getter
from vtscore.datasets.labelset import LabeledElement, LabelSet
from vtscore.detectors.registry import get_detector, register_detector
from vtscore.detectors.store import _detector_path, _read_detector, _write_detector, flush_detector_writes
from vtscore.labels.binding import sharing_binding
from vtscore.labels.sources.base import LabelsetSource
from vtscore.labels.sources.shared_folder import SharedFolderLabelsetSource, simulate_foreign_modification
from vtsearch.errors import (
    ALREADY_RECEIVED,
    DETECTOR_NOT_LINKED,
    DETECTOR_NOT_SHARED,
    DETECTOR_READ_ONLY,
    REMOTE_NOT_FOUND,
    REMOTE_PERMISSION_DENIED,
    SHARING_DISABLED,
    SHARING_UNAVAILABLE,
    SHARING_UNSUPPORTED,
)

ALICE, BOB, CAROL = "alice", "bob", "carol"

MD5_A, MD5_B, MD5_C = "a" * 32, "b" * 32, "c" * 32


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def users(monkeypatch):
    """Log-in-able users, with server paths unconfined so two of them can reach one shared folder."""
    import vtscore.security.path_validation as path_validation

    monkeypatch.setattr(path_validation, "get_file_access_base_dir", lambda: None)
    original = get_login_provider()
    set_login_provider(TrivialLoginProvider())
    try:
        yield
    finally:
        set_login_provider(original)


@pytest.fixture
def folder(tmp_path, users) -> str:
    """Sharing on, through a ``shared_folder`` in a scratch directory."""
    shared = str(tmp_path / "shared")
    register_setting_getter(
        "detector_sharing_source", lambda: {"source_name": "shared_folder", "field_values": {"folder": shared}}
    )
    return shared


@pytest.fixture
def sharing_off():
    register_setting_getter("detector_sharing_source", lambda: None)


def _login(client, user: str) -> None:
    assert client.post("/api/auth/login", json={"username": user}).status_code == 200


def _labelset(**labels: str) -> dict:
    md5s = {"a": MD5_A, "b": MD5_B, "c": MD5_C}
    return LabelSet([LabeledElement(md5=md5s[k], label=v) for k, v in labels.items()]).to_dict()


def _detector(name: str = "Dog barks", created_by: str = ALICE, **entry_fields) -> str:
    """A detector with a labels file (a good, b bad, c good); its id.

    The file is written before the entry exists, so a ``linked`` entry can be
    made this way too (the store refuses a write to a linked detector).
    """
    _write_detector(
        _detector_path(name),
        {"name": name, "media_type": "image", "labelset": _labelset(a="good", b="bad", c="good")},
    )
    flush_detector_writes()
    entry = register_detector(name=name, media_type="image", created_by=created_by, num_training=3, **entry_fields)
    return entry["id"]


def _entry(detector_id: str) -> dict:
    entry = get_detector(detector_id)
    assert entry is not None, f"no entry {detector_id!r}"
    return entry


def _file_labels(detector_id: str) -> dict[str, str]:
    flush_detector_writes()
    data = _read_detector(_detector_path(_entry(detector_id)["name"]))
    assert data is not None
    return {el.md5: el.label for el in LabelSet.from_dict(data["labelset"]).elements if el.md5}


def _share(client, detector_id: str, **body) -> dict:
    resp = client.post(f"/api/detectors/registry/{detector_id}/share", json=body)
    assert resp.status_code == 200, resp.get_data(as_text=True)
    return resp.get_json()


def _remote_ref(detector_id: str) -> str:
    binding = sharing_binding(get_detector(detector_id))
    assert binding is not None, f"{detector_id} shares nothing"
    return binding["remote_ref"]


def _container(folder: str, remote_ref: str) -> dict:
    return json.loads((Path(folder) / f"{remote_ref}.json").read_text(encoding="utf-8"))


def _received(client) -> list[dict]:
    resp = client.get("/api/detectors/sharing/received")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    return resp.get_json()["received"]


def _receive(client, remote_ref: str, **body) -> dict:
    resp = client.post("/api/detectors/sharing/receive", json={"remote_ref": remote_ref, **body})
    assert resp.status_code == 201, resp.get_data(as_text=True)
    return resp.get_json()


@pytest.fixture
def shared(client, folder) -> dict:
    """Alice's detector, shared with Bob, and Bob's received copy; Bob is logged in."""
    _login(client, ALICE)
    owned = _detector()
    _share(client, owned, add=[BOB])
    ref = _remote_ref(owned)
    _login(client, BOB)
    received = _receive(client, ref)["detector"]["id"]
    return {"owned": owned, "received": received, "ref": ref, "folder": folder}


def _error(resp, status: int, code: str | None = None) -> dict:
    assert resp.status_code == status, resp.get_data(as_text=True)
    body = resp.get_json()
    if code is not None:
        assert body["error_code"] == code, body
    return body


# ---------------------------------------------------------------------------
# Sharing off
# ---------------------------------------------------------------------------


#: Every route but the capability one and fork: ``(method, path, json)``.
OFF_ROUTES = [
    ("get", "/api/detectors/sharing/received", None),
    ("post", "/api/detectors/sharing/receive", {"remote_ref": "anything"}),
    ("post", "/api/detectors/sharing/search-recipients", {"query": "b"}),
    ("get", "/api/detectors/registry/{id}/sharing", None),
    ("post", "/api/detectors/registry/{id}/share", {"add": [BOB]}),
    ("delete", "/api/detectors/registry/{id}/share", None),
    ("post", "/api/detectors/registry/{id}/refresh", {}),
]


class TestSharingOff:
    def test_the_capability_route_says_disabled(self, client, sharing_off):
        resp = client.get("/api/detectors/sharing")
        assert resp.status_code == 200
        assert resp.get_json() == {
            "enabled": False,
            "name": None,
            "display_name": None,
            "icon": None,
            "supports_recipient_search": False,
            "missing_fields": [],
        }

    @pytest.mark.parametrize(("method", "path", "body"), OFF_ROUTES, ids=[f"{m} {p}" for m, p, _ in OFF_ROUTES])
    def test_every_other_route_is_404_sharing_disabled(self, client, sharing_off, method, path, body):
        detector_id = _detector(created_by="default")
        kwargs = {} if body is None else {"json": body}
        resp = getattr(client, method)(path.format(id=detector_id), **kwargs)
        _error(resp, 404, SHARING_DISABLED)

    def test_fork_still_works_and_is_local_only(self, client, sharing_off):
        detector_id = _detector(created_by="default")
        resp = client.post(f"/api/detectors/registry/{detector_id}/fork", json={})
        assert resp.status_code == 201, resp.get_data(as_text=True)
        body = resp.get_json()
        assert body["detector"]["sync_mode"] == "forked"
        assert body["detector"]["sharing"] is None
        assert "off" in body["remote_sync"]
        assert _file_labels(body["detector"]["id"]) == _file_labels(detector_id)

    def test_the_rest_of_the_detector_routes_behave_as_before(self, client, sharing_off):
        detector_id = _detector(created_by="default")
        rows = {d["id"]: d for d in client.get("/api/detectors/registry").get_json()["detectors"]}
        assert rows[detector_id]["sync_mode"] == "local"
        assert rows[detector_id]["sharing"] is None
        resp = client.put(f"/api/detectors/registry/{detector_id}/rename", json={"name": "Cat purrs"})
        assert resp.status_code == 200, resp.get_data(as_text=True)


# ---------------------------------------------------------------------------
# Capability
# ---------------------------------------------------------------------------


def test_the_capability_route_names_the_active_source(client, folder):
    body = client.get("/api/detectors/sharing").get_json()
    assert body["enabled"] is True
    assert body["name"] == "shared_folder"
    assert body["display_name"] == "Shared Folder"
    assert body["icon"]
    assert body["supports_recipient_search"] is True
    assert body["missing_fields"] == []


# ---------------------------------------------------------------------------
# Share
# ---------------------------------------------------------------------------


class TestShare:
    def test_the_first_call_creates_the_container_and_binds(self, client, folder):
        _login(client, ALICE)
        detector_id = _detector()

        body = _share(client, detector_id)

        assert body["created"] is True
        assert "Shared 'Dog barks' through Shared Folder" in body["message"]
        assert body["is_owner"] is True and body["owner"] == ALICE
        assert body["recipients"] == [] and body["public"] is False
        ref = _remote_ref(detector_id)
        assert body["sharing"]["remote_ref"] == ref
        assert body["sharing"]["source_name"] == "shared_folder"
        container = _container(folder, ref)
        assert container["owner"] == ALICE
        assert sorted(r["element"]["md5"] for r in container["records"]) == [MD5_A, MD5_B, MD5_C]
        assert _entry(detector_id)["labelset_source"]["field_values"] == {}  # the folder stays in the setting

    def test_a_plain_binding_is_replaced_and_the_message_says_so(self, client, folder, tmp_path):
        _login(client, ALICE)
        detector_id = _detector(
            labelset_source={"source_name": "server_json_file", "field_values": {"filepath": str(tmp_path / "a.json")}}
        )

        body = _share(client, detector_id)

        assert "no longer auto-exported to Server JSON File" in body["message"]
        binding = sharing_binding(get_detector(detector_id))
        assert binding is not None and binding["source_name"] == "shared_folder"

    def test_add_remove_and_public_round_trip_through_the_sharing_view(self, client, folder):
        _login(client, ALICE)
        detector_id = _detector()
        assert _share(client, detector_id, add=[BOB, CAROL])["recipients"] == [BOB, CAROL]

        body = _share(client, detector_id, remove=[CAROL], public=True)

        assert body["created"] is False
        view = client.get(f"/api/detectors/registry/{detector_id}/sharing").get_json()
        assert view["recipients"] == [BOB]
        assert view["public"] is True
        assert view["sharing_note"] is None
        _share(client, detector_id, public=False)
        assert client.get(f"/api/detectors/registry/{detector_id}/sharing").get_json()["public"] is False

    def test_a_non_creator_gets_403(self, client, folder):
        _login(client, ALICE)
        detector_id = _detector()
        _login(client, BOB)
        _error(client.post(f"/api/detectors/registry/{detector_id}/share", json={}), 403)
        assert sharing_binding(get_detector(detector_id)) is None

    def test_a_received_detector_gets_409_read_only(self, client, shared):
        resp = client.post(f"/api/detectors/registry/{shared['received']}/share", json={"add": [CAROL]})
        body = _error(resp, 409, DETECTOR_READ_ONLY)
        assert body["remote_owner"] == ALICE

    def test_owner_saves_reach_the_container(self, client, shared):
        from vtscore.labels.sync import flush_pending_label_syncs

        data = _read_detector(_detector_path("Dog barks"))
        assert data is not None
        data["labelset"] = _labelset(a="good", b="good")
        _write_detector(_detector_path("Dog barks"), data)
        flush_detector_writes()
        flush_pending_label_syncs()
        records = {
            r["element"]["md5"]: r["element"]["label"] for r in _container(shared["folder"], shared["ref"])["records"]
        }
        assert records == {MD5_A: "good", MD5_B: "good"}


# ---------------------------------------------------------------------------
# Received and receive
# ---------------------------------------------------------------------------


class TestReceive:
    def test_the_listing_receive_and_a_second_receive(self, client, folder):
        _login(client, ALICE)
        owned = _detector()
        _share(client, owned, add=[BOB])
        ref = _remote_ref(owned)
        _login(client, BOB)

        (listed,) = _received(client)
        assert listed["remote_ref"] == ref
        assert listed["name"] == "Dog barks" and listed["owner"] == ALICE
        assert listed["shared_via"] == "direct" and listed["record_count"] == 3
        assert listed["detector_id"] is None

        body = _receive(client, ref)
        detector = body["detector"]
        assert body["applied"] is True and body["integrity"]["status"] == "clean"
        assert detector["sync_mode"] == "linked"
        assert detector["created_by"] == BOB and detector["is_owner"] is True
        assert detector["sharing"]["remote_owner"] == ALICE
        assert detector["name"] == "Dog barks (2)"  # Alice's file holds the name
        assert _file_labels(detector["id"]) == {MD5_A: "good", MD5_B: "bad", MD5_C: "good"}

        assert _received(client)[0]["detector_id"] == detector["id"]
        again = _error(client.post("/api/detectors/sharing/receive", json={"remote_ref": ref}), 409, ALREADY_RECEIVED)
        assert again["detector_id"] == detector["id"]

    def test_a_container_that_does_not_exist_is_404(self, client, folder):
        _login(client, BOB)
        _error(client.post("/api/detectors/sharing/receive", json={"remote_ref": "nope"}), 404, REMOTE_NOT_FOUND)

    def test_a_container_not_shared_with_the_caller_is_403(self, client, folder):
        _login(client, ALICE)
        owned = _detector()
        _share(client, owned)
        _login(client, CAROL)
        resp = client.post("/api/detectors/sharing/receive", json={"remote_ref": _remote_ref(owned)})
        _error(resp, 403, REMOTE_PERMISSION_DENIED)

    def test_abort_with_foreign_records_creates_nothing(self, client, folder):
        _login(client, ALICE)
        owned = _detector()
        _share(client, owned, add=[BOB, CAROL])
        ref = _remote_ref(owned)
        simulate_foreign_modification(folder, ref, actor=CAROL, relabel=[MD5_A])
        _login(client, BOB)

        resp = client.post("/api/detectors/sharing/receive", json={"remote_ref": ref})

        assert resp.status_code == 200, resp.get_data(as_text=True)
        body = resp.get_json()
        assert body["applied"] is False and body["detector"] is None
        assert body["integrity"]["status"] == "foreign_modifications"
        assert body["name"] == "Dog barks" and body["owner"] == ALICE
        assert _received(client)[0]["detector_id"] is None
        assert _receive(client, ref, foreign="skip")["detector"]["sync_mode"] == "linked"


# ---------------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------------


class TestRefresh:
    def test_abort_with_foreign_records_reports_and_applies_nothing(self, client, shared):
        simulate_foreign_modification(shared["folder"], shared["ref"], actor=CAROL, relabel=[MD5_A])
        before = _file_labels(shared["received"])

        resp = client.post(f"/api/detectors/registry/{shared['received']}/refresh", json={})

        assert resp.status_code == 200, resp.get_data(as_text=True)
        body = resp.get_json()
        assert body["applied"] is False
        assert body["integrity"]["status"] == "foreign_modifications"
        assert "not made by its owner" in body["message"]
        assert _file_labels(shared["received"]) == before
        assert body["detector"]["sharing"]["integrity"]["status"] == "foreign_modifications"

    def test_skip_takes_only_the_owners_records_and_import_takes_them_all(self, client, shared):
        simulate_foreign_modification(shared["folder"], shared["ref"], actor=CAROL, relabel=[MD5_A])
        refresh = f"/api/detectors/registry/{shared['received']}/refresh"

        skipped = client.post(refresh, json={"foreign": "skip"}).get_json()
        assert skipped["applied"] is True and skipped["removed"] == 1
        assert _file_labels(shared["received"]) == {MD5_B: "bad", MD5_C: "good"}

        imported = client.post(refresh, json={"foreign": "import"}).get_json()
        assert imported["applied"] is True and imported["added"] == 1
        assert _file_labels(shared["received"]) == {MD5_A: "bad", MD5_B: "bad", MD5_C: "good"}

    def test_an_up_to_date_detector_says_so(self, client, shared):
        body = client.post(f"/api/detectors/registry/{shared['received']}/refresh", json={}).get_json()
        assert body["applied"] is True
        assert (body["added"], body["changed"], body["removed"]) == (0, 0, 0)
        assert "up to date" in body["message"]

    def test_a_detector_that_is_not_received_is_409(self, client, shared):
        _login(client, ALICE)
        resp = client.post(f"/api/detectors/registry/{shared['owned']}/refresh", json={})
        _error(resp, 409, DETECTOR_NOT_LINKED)

    def test_a_policy_that_is_not_one_is_422(self, client, shared):
        resp = client.post(f"/api/detectors/registry/{shared['received']}/refresh", json={"foreign": "merge"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Fork
# ---------------------------------------------------------------------------


class TestFork:
    def test_forking_a_received_detector_gives_an_editable_shared_copy(self, client, shared):
        resp = client.post(f"/api/detectors/registry/{shared['received']}/fork", json={"name": "Bob's barks"})

        assert resp.status_code == 201, resp.get_data(as_text=True)
        body = resp.get_json()
        fork = body["detector"]
        assert fork["name"] == "Bob's barks"
        assert fork["sync_mode"] == "forked" and fork["created_by"] == BOB
        assert fork["forked_from"]["owner"] == ALICE
        assert fork["forked_from"]["remote_ref"] == shared["ref"]
        assert body["remote_sync"] is None
        assert fork["sharing"]["remote_ref"] not in (None, shared["ref"])  # a container of Bob's own
        assert _file_labels(fork["id"]) == _file_labels(shared["received"])
        rename = client.put(f"/api/detectors/registry/{fork['id']}/rename", json={"name": "Barks, mine"})
        assert rename.status_code == 200, rename.get_data(as_text=True)

    def test_without_a_name_it_is_named_after_the_original(self, client, shared):
        body = client.post(f"/api/detectors/registry/{shared['received']}/fork", json={"name": "  "}).get_json()
        assert body["detector"]["name"] == "Dog barks (2) (fork)"

    def test_a_failed_container_create_leaves_a_local_fork_with_the_note(self, client, shared, monkeypatch):
        def fail(self, labelset, meta, field_values):
            raise OSError("disk full")

        monkeypatch.setattr(SharedFolderLabelsetSource, "_do_share_create", fail)
        body = client.post(f"/api/detectors/registry/{shared['received']}/fork", json={}).get_json()
        assert body["detector"]["sync_mode"] == "forked"
        assert body["detector"]["sharing"] is None
        assert "disk full" in body["remote_sync"]

    def test_a_taken_name_is_409(self, client, shared):
        resp = client.post(f"/api/detectors/registry/{shared['received']}/fork", json={"name": "Dog barks"})
        _error(resp, 409)

    def test_a_detector_the_caller_cannot_see_is_403(self, client, shared):
        _login(client, CAROL)
        _error(client.post(f"/api/detectors/registry/{shared['owned']}/fork", json={}), 403)

    def test_an_unknown_detector_is_404(self, client, folder):
        _error(client.post("/api/detectors/registry/nope/fork", json={}), 404)


# ---------------------------------------------------------------------------
# Recipient search
# ---------------------------------------------------------------------------


class _NoSearchSource(SharedFolderLabelsetSource):
    """``shared_folder`` without the optional recipient-search hook."""

    name = "shared_folder_no_search"
    display_name = "Shared Folder (no search)"
    _do_share_search_recipients = LabelsetSource._do_share_search_recipients


class TestSearchRecipients:
    def test_suggestions_come_from_the_source(self, client, shared):
        _login(client, ALICE)
        resp = client.post("/api/detectors/sharing/search-recipients", json={"query": "bo"})
        assert resp.status_code == 200, resp.get_data(as_text=True)
        assert resp.get_json()["suggestions"] == [{"id": BOB, "display_name": BOB, "kind": "user"}]

    def test_a_source_without_the_hook_is_501(self, client, users, tmp_path, monkeypatch):
        import vtscore.labels.sources as sources

        no_search = _NoSearchSource()
        real = sources.get_labelset_source
        monkeypatch.setattr(sources, "get_labelset_source", lambda n: no_search if n == no_search.name else real(n))
        shared_dir = str(tmp_path / "shared")
        register_setting_getter(
            "detector_sharing_source", lambda: {"source_name": no_search.name, "field_values": {"folder": shared_dir}}
        )

        assert client.get("/api/detectors/sharing").get_json()["supports_recipient_search"] is False
        resp = client.post("/api/detectors/sharing/search-recipients", json={"query": "bo"})
        _error(resp, 501, SHARING_UNSUPPORTED)


# ---------------------------------------------------------------------------
# The sharing view, and a binding whose source is not the active one
# ---------------------------------------------------------------------------


class TestSharingView:
    def test_a_received_detector_shows_its_container(self, client, shared):
        view = client.get(f"/api/detectors/registry/{shared['received']}/sharing").get_json()
        assert view["sync_mode"] == "linked"
        assert view["is_owner"] is False
        assert view["owner"] == ALICE and view["recipients"] == [BOB]
        assert view["sharing"]["remote_ref"] == shared["ref"]

    def test_a_detector_that_shares_nothing_is_404(self, client, folder):
        _login(client, ALICE)
        detector_id = _detector()
        resp = client.get(f"/api/detectors/registry/{detector_id}/sharing")
        _error(resp, 404, DETECTOR_NOT_SHARED)

    def test_a_detector_the_caller_cannot_see_is_403(self, client, shared):
        _login(client, CAROL)
        _error(client.get(f"/api/detectors/registry/{shared['owned']}/sharing"), 403)

    def test_another_sources_container_is_shown_with_the_note_and_not_read(self, client, folder):
        _login(client, BOB)
        detector_id = _detector(
            created_by=BOB,
            sync_mode="linked",
            labelset_source={
                "source_name": "gone_source",
                "field_values": {},
                "remote_ref": "r1",
                "remote_owner": ALICE,
            },
        )

        view = client.get(f"/api/detectors/registry/{detector_id}/sharing").get_json()
        assert "not installed" in view["sharing_note"]
        assert view["recipients"] is None and view["public"] is None and view["owner"] is None

        resp = client.post(f"/api/detectors/registry/{detector_id}/refresh", json={})
        body = _error(resp, 409, SHARING_UNAVAILABLE)
        assert "not installed" in body["message"]


# ---------------------------------------------------------------------------
# Unshare
# ---------------------------------------------------------------------------


class TestUnshare:
    def test_the_container_is_archived_and_the_entry_unbound(self, client, shared):
        _login(client, ALICE)

        resp = client.delete(f"/api/detectors/registry/{shared['owned']}/share")

        assert resp.status_code == 200, resp.get_data(as_text=True)
        body = resp.get_json()
        assert body["detector"]["sharing"] is None and body["detector"]["sync_mode"] == "local"
        assert sharing_binding(get_detector(shared["owned"])) is None
        assert _container(shared["folder"], shared["ref"])["archived"] is True
        assert _file_labels(shared["owned"]) == {MD5_A: "good", MD5_B: "bad", MD5_C: "good"}
        _login(client, BOB)
        assert _received(client) == []

    def test_a_detector_that_shares_nothing_is_404(self, client, folder):
        _login(client, ALICE)
        detector_id = _detector()
        _error(client.delete(f"/api/detectors/registry/{detector_id}/share"), 404, DETECTOR_NOT_SHARED)

    def test_a_non_creator_gets_403(self, client, shared):
        _error(client.delete(f"/api/detectors/registry/{shared['owned']}/share"), 403)
        assert sharing_binding(get_detector(shared["owned"])) is not None
