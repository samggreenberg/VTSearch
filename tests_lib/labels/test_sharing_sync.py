"""Role-aware label sync (#5576): owners push to their container, received detectors never sync.

The sync engine keeps one slot and one pace per detector; what a push does
depends on the detector's role (``vtscore.labels.sync.sync_role``).  Alice owns
a detector shared through a ``shared_folder``: her votes rewrite the container
(a rewrite, not a merge - a recipient's foreign record is gone afterwards), the
outcome lands on her binding, and nothing is ever pulled back into her file.
Bob's received copy is never pushed or pulled; it changes only by refresh.

Every armed push is held for ``flush_pending_label_syncs`` (the debounce is an
hour), so no test depends on a timer firing.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Any

import pytest

from vtscore import config
from vtscore.config import register_setting_getter
from vtscore.datasets.labelset import LabeledElement, LabelSet
from vtscore.detectors.registry import get_detector, register_detector, update_detector
from vtscore.detectors.store import _detector_path, _read_detector, _write_detector, allow_linked_write
from vtscore.labels import sync as label_sync
from vtscore.labels.binding import sharing_binding
from vtscore.labels.sharing import (
    INTEGRITY_CLEAN,
    SharingError,
    build_manifest,
    compute_integrity,
    fork_detector,
    receive_container,
    refresh_linked_detector,
    share_payload,
)
from vtscore.labels.sources import get_labelset_source
from vtscore.labels.sources.shared_folder import SharedFolderLabelsetSource, simulate_foreign_modification
from vtscore.labels.sync import (
    SYNC_ROLE_LINKED,
    SYNC_ROLE_NONE,
    SYNC_ROLE_OWNER,
    SYNC_ROLE_PLAIN,
    SYNC_ROLE_UNAVAILABLE,
    flush_pending_label_syncs,
    reconcile_labelset_source,
    sync_from_labelset_source,
    sync_role,
)
from vtscore.state.core import get_active_context
from vtscore.state.current_user import get_current_user, thread_user

OWNER, BOB, CAROL = "alice", "bob", "carol"
NAME = "Dog barks"


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def folder(tmp_path, monkeypatch):
    """Sharing on, through a ``shared_folder``; a scratch detectors dir; every push held for a flush."""
    scratch = dataclasses.replace(config.CoreConfig.from_settings(), detectors_dir=tmp_path / "detectors")
    config.register_core_config_builder(lambda path=None: scratch)
    shared = str(tmp_path / "shared")
    register_setting_getter(
        "detector_sharing_source", lambda: {"source_name": "shared_folder", "field_values": {"folder": shared}}
    )
    # Hold every armed push until the test flushes: with the real 200ms
    # debounce a loaded machine can fire one between two writes.
    monkeypatch.setattr(label_sync, "_DEBOUNCE_DELAY", 3600.0)
    return shared


@pytest.fixture
def pushes(monkeypatch) -> list[dict[str, Any]]:
    """Every ``share_save`` the shared folder runs: the caller it ran as, the ref and the labels it was given."""
    calls: list[dict[str, Any]] = []
    original = SharedFolderLabelsetSource._do_share_save

    def record(self, remote_ref, labelset, meta, field_values):
        calls.append(
            {
                "user": get_current_user(),
                "remote_ref": remote_ref,
                "labels": {el.md5: el.label for el in labelset.elements},
                "manifest": meta.manifest,
            }
        )
        return original(self, remote_ref, labelset, meta, field_values)

    monkeypatch.setattr(SharedFolderLabelsetSource, "_do_share_save", record)
    return calls


def _sharing_off() -> None:
    register_setting_getter("detector_sharing_source", lambda: None)


MD5: dict[str, str] = {}


@pytest.fixture(autouse=True)
def _md5():
    """md5s of medias in the active test dataset, so a loaded context could resolve them."""
    out = [m["md5"] for m in get_active_context().medias.values() if m.get("md5")]
    assert len(out) >= 4, "the test dataset has too few medias"
    MD5.update(a=out[0], b=out[1], c=out[2], d=out[3])


def _labels(**labels: str) -> dict:
    return LabelSet([LabeledElement(md5=MD5[k], label=v) for k, v in labels.items()]).to_dict()


def _detector(name: str = NAME, **labels: str) -> dict:
    return {
        "name": name,
        "media_type": "image",
        "input_spec": {"clipper": "whole_image", "params": {}},
        "threshold": 0.42,
        "labelset": _labels(**(labels or {"a": "good", "b": "bad", "c": "good"})),
    }


def _source() -> SharedFolderLabelsetSource:
    source = get_labelset_source("shared_folder")
    assert isinstance(source, SharedFolderLabelsetSource)
    return source


def _share(folder: str, name: str = NAME) -> dict:
    """Alice's detector *name*, shared with Bob: its registry entry, bound to a fresh container."""
    data = _detector(name)
    _write_detector(_detector_path(name), data)
    entry = register_detector(name=name, media_type="image", num_training=3, created_by=OWNER)
    labelset, meta = share_payload(data)
    with thread_user(OWNER):
        ref = _source().share_create(labelset, meta, {"folder": folder})
        _source().share_add_recipients(ref, [BOB], {"folder": folder})
    update_detector(
        entry["id"],
        labelset_source={
            "source_name": "shared_folder",
            "field_values": {},
            "remote_ref": ref,
            "last_synced": 1.0,
            "sync_error": None,
        },
    )
    return _entry(entry["id"])


def _entry(detector_id: str) -> dict:
    entry = get_detector(detector_id)
    assert entry is not None, f"no entry {detector_id!r}"
    return entry


def _binding(detector_id: str) -> dict:
    binding = sharing_binding(_entry(detector_id))
    assert binding is not None, f"entry {detector_id!r} shares nothing"
    return binding


def _read(name: str) -> dict:
    data = _read_detector(_detector_path(name))
    assert data is not None, f"no file for {name!r}"
    return data


def _vote(name: str, **labels: str) -> None:
    """Rewrite *name*'s labels the way a vote does: one file write."""
    _write_detector(_detector_path(name), {**_read(name), "labelset": _labels(**labels)})


def _container(folder: str, ref: str):
    with thread_user(OWNER):
        return _source().share_refresh(ref, {"folder": folder})


def _container_labels(folder: str, ref: str) -> dict[str, str]:
    return {r.element.md5: r.element.label for r in _container(folder, ref).records}


def _file_labels(name: str) -> dict[str, str]:
    return {el.md5: el.label for el in LabelSet.from_dict(_read(name)["labelset"]).elements}


def _receive(ref: str) -> dict:
    with thread_user(BOB):
        return receive_container(ref)


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------


class TestSyncRole:
    def test_each_entry_shape_has_its_role(self, folder, tmp_path):
        assert sync_role(None) == SYNC_ROLE_NONE
        unbound = register_detector(name="unbound", media_type="image")
        assert sync_role(unbound) == SYNC_ROLE_NONE

        plain = register_detector(name="plain", media_type="image")
        update_detector(
            plain["id"],
            labelset_source={"source_name": "server_json_file", "field_values": {"filepath": str(tmp_path / "p.json")}},
        )
        assert sync_role(_entry(plain["id"])) == SYNC_ROLE_PLAIN

        owned = _share(folder)
        assert sync_role(owned) == SYNC_ROLE_OWNER
        assert sync_role(_receive(_binding(owned["id"])["remote_ref"])) == SYNC_ROLE_LINKED

        _sharing_off()
        assert sync_role(_entry(owned["id"])) == SYNC_ROLE_UNAVAILABLE

    def test_a_fork_with_a_container_is_its_owner(self, folder):
        received = _receive(_binding(_share(folder)["id"])["remote_ref"])
        with thread_user(BOB):
            fork = fork_detector(received["id"])
        assert sync_role(_entry(fork["id"])) == SYNC_ROLE_OWNER

    def test_a_linked_entry_is_linked_even_when_its_source_is_gone(self, folder):
        received = _receive(_binding(_share(folder)["id"])["remote_ref"])
        _sharing_off()
        assert sync_role(_entry(received["id"])) == SYNC_ROLE_LINKED


# ---------------------------------------------------------------------------
# The owner's push
# ---------------------------------------------------------------------------


class TestOwnerPush:
    def test_a_vote_reaches_the_container_with_its_manifest(self, folder, pushes):
        entry = _share(folder)
        ref = _binding(entry["id"])["remote_ref"]
        _vote(NAME, a="good", b="good", d="bad")
        flush_pending_label_syncs()

        assert len(pushes) == 1
        assert _container_labels(folder, ref) == {MD5["a"]: "good", MD5["b"]: "good", MD5["d"]: "bad"}
        container = _container(folder, ref)
        exported, _meta = share_payload(_read(NAME))
        assert container.manifest == build_manifest(exported)
        assert container.manifest_modified_by == OWNER
        assert compute_integrity(container) == {"status": INTEGRITY_CLEAN, "count": 0}
        assert "threshold" not in container.detector_meta

    def test_a_recipients_foreign_record_is_gone_after_the_push(self, folder, pushes):
        entry = _share(folder)
        ref = _binding(entry["id"])["remote_ref"]
        simulate_foreign_modification(
            folder, ref, actor=BOB, relabel=[MD5["a"]], add=[LabeledElement(md5=MD5["d"], label="good")]
        )
        assert _container_labels(folder, ref)[MD5["d"]] == "good"

        _vote(NAME, a="good", b="bad", c="bad")
        flush_pending_label_syncs()

        # A rewrite, not a merge: exactly the owner's labels, the foreign
        # addition dropped and the foreign relabel overwritten...
        assert _container_labels(folder, ref) == {MD5["a"]: "good", MD5["b"]: "bad", MD5["c"]: "bad"}
        assert compute_integrity(_container(folder, ref))["status"] == INTEGRITY_CLEAN
        # ...and nothing came back into the owner's file.
        assert _file_labels(NAME) == {MD5["a"]: "good", MD5["b"]: "bad", MD5["c"]: "bad"}

    def test_an_unchanged_payload_is_not_pushed_again(self, folder, pushes):
        _share(folder)
        _vote(NAME, a="good", b="bad")
        flush_pending_label_syncs()
        # A write that changes only a derived number carries the same labels.
        _write_detector(_detector_path(NAME), {**_read(NAME), "threshold": 0.9})
        flush_pending_label_syncs()
        assert len(pushes) == 1

    def test_two_writes_inside_the_pace_window_are_one_push(self, folder, pushes):
        _share(folder)
        _vote(NAME, a="bad")
        _vote(NAME, a="bad", b="good")
        assert pushes == []  # both folded into the one armed push
        flush_pending_label_syncs()
        assert len(pushes) == 1
        assert pushes[0]["labels"] == {MD5["a"]: "bad", MD5["b"]: "good"}  # the newest, read when it ran

    def test_the_interval_holds_the_next_push_back(self, folder, pushes, monkeypatch):
        # The real debounce, so it is the interval alone that holds the push.
        monkeypatch.setattr(label_sync, "_DEBOUNCE_DELAY", 0.2)
        register_setting_getter("labelset_source_push_interval", lambda: 3600.0)
        _share(folder)
        _vote(NAME, a="bad")
        flush_pending_label_syncs()
        assert len(pushes) == 1
        _vote(NAME, a="bad", b="good")
        slot = label_sync._slots[NAME]
        assert slot.timer is not None and slot.timer.interval > 3000  # one slot, one pace, whatever the role
        assert len(pushes) == 1
        flush_pending_label_syncs()
        assert len(pushes) == 2

    def test_the_push_runs_as_the_creator_not_the_voter(self, folder, pushes):
        entry = _share(folder)
        ref = _binding(entry["id"])["remote_ref"]
        # Carol votes on Alice's detector (a reader of it on this server), and
        # the flush even runs the push on Carol's thread.
        with thread_user(CAROL):
            _vote(NAME, a="bad")
            flush_pending_label_syncs()
        assert [p["user"] for p in pushes] == [OWNER]
        container = _container(folder, ref)
        assert container.manifest_modified_by == OWNER
        assert {r.last_modified_by for r in container.records} == {OWNER}

    def test_success_stamps_last_synced_and_clears_sync_error(self, folder, pushes):
        entry = _share(folder)
        update_detector(entry["id"], labelset_source={**_binding(entry["id"]), "sync_error": "old trouble"})
        before = time.time()
        _vote(NAME, a="bad")
        flush_pending_label_syncs()
        binding = _binding(entry["id"])
        assert binding["sync_error"] is None
        assert binding["last_synced"] >= before

    def test_a_failed_push_is_recorded_and_the_next_change_clears_it(self, folder, pushes, monkeypatch):
        entry = _share(folder)
        ref = _binding(entry["id"])["remote_ref"]
        container_before = _container_labels(folder, ref)
        original = SharedFolderLabelsetSource._do_share_save

        def down(self, remote_ref, labelset, meta, field_values):
            raise SharingError("the service is down")

        monkeypatch.setattr(SharedFolderLabelsetSource, "_do_share_save", down)
        _vote(NAME, a="bad")
        written = _detector_path(NAME).read_bytes()
        flush_pending_label_syncs()

        assert _detector_path(NAME).read_bytes() == written  # the file is the record, untouched
        assert _container_labels(folder, ref) == container_before
        binding = _binding(entry["id"])
        assert binding["sync_error"] == "the service is down"
        assert binding["last_synced"] == 1.0  # moved only by a success

        monkeypatch.setattr(SharedFolderLabelsetSource, "_do_share_save", original)
        _vote(NAME, a="bad", b="good")
        flush_pending_label_syncs()
        binding = _binding(entry["id"])
        assert binding["sync_error"] is None
        assert binding["last_synced"] > 1.0
        assert _container_labels(folder, ref) == {MD5["a"]: "bad", MD5["b"]: "good"}

    def test_a_failed_push_of_the_same_labels_is_retried(self, folder, pushes, monkeypatch):
        """The dedup remembers only what landed, so a write that changes nothing retries a failure."""
        entry = _share(folder)
        original = SharedFolderLabelsetSource._do_share_save
        monkeypatch.setattr(
            SharedFolderLabelsetSource,
            "_do_share_save",
            lambda *_a: (_ for _ in ()).throw(SharingError("the service is down")),
        )
        _vote(NAME, a="bad")
        flush_pending_label_syncs()
        monkeypatch.setattr(SharedFolderLabelsetSource, "_do_share_save", original)
        _write_detector(_detector_path(NAME), {**_read(NAME), "threshold": 0.9})
        flush_pending_label_syncs()
        assert _binding(entry["id"])["sync_error"] is None

    def test_the_outcome_merges_into_the_binding_as_it_is_now(self, folder, pushes, monkeypatch):
        """A field written to the binding while the push ran survives the push's bookkeeping."""
        entry = _share(folder)
        original = SharedFolderLabelsetSource._do_share_save

        def meanwhile(self, remote_ref, labelset, meta, field_values):
            update_detector(entry["id"], labelset_source={**_binding(entry["id"]), "remote_owner": OWNER})
            return original(self, remote_ref, labelset, meta, field_values)

        monkeypatch.setattr(SharedFolderLabelsetSource, "_do_share_save", meanwhile)
        _vote(NAME, a="bad")
        flush_pending_label_syncs()
        binding = _binding(entry["id"])
        assert binding["remote_owner"] == OWNER
        assert binding["sync_error"] is None

    def test_an_unshare_during_the_push_is_not_undone(self, folder, monkeypatch):
        entry = _share(folder)
        original = SharedFolderLabelsetSource._do_share_save

        def unshare(self, remote_ref, labelset, meta, field_values):
            update_detector(entry["id"], labelset_source=None)
            return original(self, remote_ref, labelset, meta, field_values)

        monkeypatch.setattr(SharedFolderLabelsetSource, "_do_share_save", unshare)
        _vote(NAME, a="bad")
        flush_pending_label_syncs()
        assert _entry(entry["id"]).get("labelset_source") is None

    def test_a_fork_pushes_to_its_own_container(self, folder, pushes):
        original = _share(folder)
        received = _receive(_binding(original["id"])["remote_ref"])
        with thread_user(BOB):
            fork = fork_detector(received["id"])
        flush_pending_label_syncs()
        pushes.clear()

        _vote(fork["name"], a="bad")
        flush_pending_label_syncs()

        fork_ref = _binding(fork["id"])["remote_ref"]
        assert [(p["user"], p["remote_ref"]) for p in pushes] == [(BOB, fork_ref)]
        with thread_user(BOB):
            mirrored = _source().share_refresh(fork_ref, {"folder": folder})
        assert {r.element.md5: r.element.label for r in mirrored.records} == {MD5["a"]: "bad"}
        # The original's container is untouched.
        assert _container_labels(folder, _binding(original["id"])["remote_ref"]) == {
            MD5["a"]: "good",
            MD5["b"]: "bad",
            MD5["c"]: "good",
        }


# ---------------------------------------------------------------------------
# Nothing is pulled into an owner or a received detector
# ---------------------------------------------------------------------------


class TestNoPull:
    def test_reconcile_on_load_leaves_an_owner_alone(self, folder, pushes):
        entry = _share(folder)
        ref = _binding(entry["id"])["remote_ref"]
        simulate_foreign_modification(folder, ref, actor=BOB, add=[LabeledElement(md5=MD5["d"], label="good")])
        before = _detector_path(NAME).read_bytes()

        assert reconcile_labelset_source(NAME, detector_id=entry["id"]) == 0

        assert _detector_path(NAME).read_bytes() == before
        assert NAME not in label_sync._slots  # and no push is scheduled on load either
        flush_pending_label_syncs()
        assert pushes == []

    def test_reconcile_on_load_leaves_a_received_detector_alone(self, folder, pushes):
        original = _share(folder)
        received = _receive(_binding(original["id"])["remote_ref"])
        _vote(NAME, a="bad", d="good")
        flush_pending_label_syncs()  # the owner's container moves on
        mirror = _detector_path(received["name"]).read_bytes()

        assert reconcile_labelset_source(received["name"], detector_id=received["id"]) == 0

        assert _detector_path(received["name"]).read_bytes() == mirror  # only a refresh updates it
        assert received["name"] not in label_sync._slots

    def test_a_manual_sync_returns_early_for_an_owner_and_a_received_detector(self, folder):
        owned = _share(folder)
        received = _receive(_binding(owned["id"])["remote_ref"])
        before = {n: _detector_path(n).read_bytes() for n in (owned["name"], received["name"])}

        assert sync_from_labelset_source(owned["id"]) is None
        assert sync_from_labelset_source(received["id"]) is None

        assert {n: _detector_path(n).read_bytes() for n in before} == before


# ---------------------------------------------------------------------------
# A received detector never pushes
# ---------------------------------------------------------------------------


class TestLinkedNeverPushes:
    def test_receive_and_refresh_schedule_nothing(self, folder, pushes):
        owned = _share(folder)
        ref = _binding(owned["id"])["remote_ref"]
        received = _receive(ref)
        assert received["name"] not in label_sync._slots

        _vote(NAME, a="bad")
        flush_pending_label_syncs()
        pushes.clear()
        with thread_user(BOB):
            assert refresh_linked_detector(received["id"]).applied
        assert received["name"] not in label_sync._slots
        flush_pending_label_syncs()
        assert pushes == []

    def test_a_mirror_write_and_a_derived_write_schedule_nothing(self, folder, pushes):
        received = _receive(_binding(_share(folder)["id"])["remote_ref"])
        path = _detector_path(received["name"])
        with allow_linked_write():
            _write_detector(path, {**_read(received["name"]), "labelset": _labels(a="bad")})
        _write_detector(path, {**_read(received["name"]), "threshold": 0.7})  # a derived number: allowed
        assert received["name"] not in label_sync._slots
        flush_pending_label_syncs()
        assert pushes == []

    def test_a_receive_pushes_nothing_even_to_a_deployment_default(self, folder, tmp_path, monkeypatch):
        """The receive writes the file before its entry exists, when the default would claim it."""
        plain = type(get_labelset_source("server_json_file"))
        original = plain._do_save
        saved: list[str] = []

        def save(self, labelset, field_values):
            saved.append(field_values["filepath"])
            return original(self, labelset, field_values)

        monkeypatch.setattr(plain, "_do_save", save)
        register_setting_getter(
            "default_labelset_source",
            lambda: {
                "source_name": "server_json_file",
                "field_values": {"filepath": str(tmp_path / "{detector_id}.json")},
            },
        )
        received = _receive(_binding(_share(folder)["id"])["remote_ref"])
        assert received["name"] not in label_sync._slots
        flush_pending_label_syncs()
        assert saved == []


# ---------------------------------------------------------------------------
# A binding whose source is not the active one
# ---------------------------------------------------------------------------


class TestUnavailableSource:
    def test_it_pushes_nothing_and_logs_once(self, folder, pushes, caplog):
        entry = _share(folder)
        _sharing_off()
        with caplog.at_level(logging.WARNING, logger="vtscore.labels.sync"):
            _vote(NAME, a="bad")
            _vote(NAME, a="bad", b="good")
            assert reconcile_labelset_source(NAME, detector_id=entry["id"]) == 0
            flush_pending_label_syncs()
        assert pushes == []
        notes = [r for r in caplog.records if r.name == "vtscore.labels.sync" and NAME in r.getMessage()]
        assert len(notes) == 1
        assert "sharing is off" in notes[0].getMessage()

    def test_a_source_that_is_not_installed_is_never_swapped_for_the_active_one(self, folder, pushes):
        entry = _share(folder)
        update_detector(entry["id"], labelset_source={**_binding(entry["id"]), "source_name": "acme_drive"})
        assert sync_role(_entry(entry["id"])) == SYNC_ROLE_UNAVAILABLE
        _vote(NAME, a="bad")
        flush_pending_label_syncs()
        assert pushes == []
        assert _binding(entry["id"])["sync_error"] is None  # nothing was attempted, so nothing failed
