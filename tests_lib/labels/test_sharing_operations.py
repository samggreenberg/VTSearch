"""Receive, refresh and fork, and the integrity check (#5575), against ``shared_folder``.

Alice owns a container made from a detector of hers; Bob receives it as a
read-only *linked* detector, refreshes it as Alice saves, and forks it into an
editable copy of his own.  ``simulate_foreign_modification`` plays a recipient
who writes to the container anyway, which the integrity check catches.
"""

from __future__ import annotations

import dataclasses

import pytest

from vtscore import config
from vtscore.config import register_setting_getter
from vtscore.datasets.labelset import LabeledElement, LabelSet
from vtscore.detectors.registry import (
    SYNC_MODE_FORKED,
    SYNC_MODE_LINKED,
    get_detector,
    list_detectors,
    register_detector,
    sync_mode_of,
    update_detector,
)
from vtscore.detectors.store import DetectorReadOnlyError, _detector_path, _read_detector, _write_detector
from vtscore.labels.binding import (
    binding_matches_active_source,
    labelset_source_status,
    sharing_binding,
    sharing_note,
)
from vtscore.labels.sharing import (
    FOREIGN_IMPORT,
    FOREIGN_SKIP,
    INTEGRITY_CLEAN,
    INTEGRITY_FOREIGN,
    ForeignModifications,
    SharingUnavailable,
    compute_integrity,
    fork_detector,
    receive_container,
    refresh_linked_detector,
    share_payload,
)
from vtscore.labels.sources import get_labelset_source
from vtscore.labels.sources.shared_folder import simulate_foreign_modification
from vtscore.state.core import DetectorContext, get_active_context, register_detector_context
from vtscore.state.current_user import thread_user

OWNER, BOB, CAROL = "alice", "bob", "carol"


@pytest.fixture
def folder(tmp_path):
    """Sharing on, through a ``shared_folder``; a scratch detectors dir."""
    scratch = dataclasses.replace(config.CoreConfig.from_settings(), detectors_dir=tmp_path / "detectors")
    config.register_core_config_builder(lambda path=None: scratch)
    shared = str(tmp_path / "shared")
    register_setting_getter(
        "detector_sharing_source", lambda: {"source_name": "shared_folder", "field_values": {"folder": shared}}
    )
    return shared


def _sharing_off() -> None:
    register_setting_getter("detector_sharing_source", lambda: None)


def _md5s(n: int) -> list[str]:
    """md5s of *n* medias in the active test dataset, so a loaded context can resolve them."""
    out = [m["md5"] for m in get_active_context().medias.values() if m.get("md5")]
    assert len(out) >= n, "the test dataset has too few medias"
    return out[:n]


MD5 = {}


@pytest.fixture(autouse=True)
def _md5():
    a, b, c, d = _md5s(4)
    MD5.update(a=a, b=b, c=c, d=d)


def _labels(**labels: str) -> dict:
    return LabelSet([LabeledElement(md5=MD5[k], label=v) for k, v in labels.items()]).to_dict()


def _owner_detector(name: str = "Dog barks", **labels: str) -> dict:
    return {
        "name": name,
        "media_type": "image",
        "input_spec": {"clipper": "whole_image", "params": {}},
        "threshold": 0.42,
        "labelset": _labels(**(labels or {"a": "good", "b": "bad", "c": "good"})),
    }


def _publish(folder: str, data: dict | None = None, recipients: tuple[str, ...] = (BOB, CAROL)) -> str:
    """Alice shares a detector with *recipients*: the container's ``remote_ref``."""
    labelset, meta = share_payload(data or _owner_detector())
    source = get_labelset_source("shared_folder")
    with thread_user(OWNER):
        ref = source.share_create(labelset, meta, {"folder": folder})
        source.share_add_recipients(ref, list(recipients), {"folder": folder})
    return ref


def _save(folder: str, ref: str, data: dict) -> None:
    labelset, meta = share_payload(data)
    with thread_user(OWNER):
        get_labelset_source("shared_folder").share_save(ref, labelset, meta, {"folder": folder})


def _receive(ref: str, who: str = BOB, **kwargs) -> dict:
    with thread_user(who):
        return receive_container(ref, **kwargs)


def _refresh(detector_id: str, **kwargs):
    with thread_user(BOB):
        return refresh_linked_detector(detector_id, **kwargs)


def _read(name: str) -> dict:
    data = _read_detector(_detector_path(name))
    assert data is not None, f"no file for {name!r}"
    return data


def _entry(detector_id: str) -> dict:
    entry = get_detector(detector_id)
    assert entry is not None, f"no entry {detector_id!r}"
    return entry


def _binding(detector_id: str) -> dict:
    binding = sharing_binding(_entry(detector_id))
    assert binding is not None, f"entry {detector_id!r} shares nothing"
    return binding


def _file_labels(entry: dict) -> dict[str, str]:
    data = _read(entry["name"])
    return {el.md5: el.label for el in LabelSet.from_dict(data["labelset"]).elements}


def _container(folder: str, ref: str):
    with thread_user(OWNER):
        return get_labelset_source("shared_folder").share_refresh(ref, {"folder": folder})


# ---------------------------------------------------------------------------
# compute_integrity
# ---------------------------------------------------------------------------


class TestIntegrity:
    def test_a_container_only_its_owner_wrote_is_clean(self, folder):
        ref = _publish(folder)
        assert compute_integrity(_container(folder, ref)) == {"status": INTEGRITY_CLEAN, "count": 0}

    def test_a_foreign_relabel_counts_once(self, folder):
        ref = _publish(folder)
        simulate_foreign_modification(folder, ref, actor=BOB, relabel=[MD5["a"]])
        assert compute_integrity(_container(folder, ref)) == {"status": INTEGRITY_FOREIGN, "count": 1}

    def test_a_foreign_addition_is_a_foreign_record_and_a_manifest_that_miscounts(self, folder):
        ref = _publish(folder)
        simulate_foreign_modification(folder, ref, actor=BOB, add=[LabeledElement(md5=MD5["d"], label="bad")])
        assert compute_integrity(_container(folder, ref)) == {"status": INTEGRITY_FOREIGN, "count": 2}

    def test_a_deletion_is_caught_by_the_manifest(self, folder):
        ref = _publish(folder)
        simulate_foreign_modification(folder, ref, actor=BOB, delete=[MD5["b"]])
        container = _container(folder, ref)
        assert {r.last_modified_by for r in container.records} == {OWNER}
        assert compute_integrity(container) == {"status": INTEGRITY_FOREIGN, "count": 1}

    def test_a_manifest_rewritten_by_a_recipient_is_foreign(self, folder):
        ref = _publish(folder)
        simulate_foreign_modification(folder, ref, actor=BOB, delete=[MD5["b"]], touch_manifest=True)
        assert compute_integrity(_container(folder, ref)) == {"status": INTEGRITY_FOREIGN, "count": 1}


# ---------------------------------------------------------------------------
# receive
# ---------------------------------------------------------------------------


class TestReceive:
    def test_a_linked_entry_with_the_right_owner_mode_and_binding(self, folder):
        ref = _publish(folder)
        entry = _receive(ref)

        assert entry["created_by"] == BOB
        assert sync_mode_of(entry) == SYNC_MODE_LINKED
        assert entry["name"] == "Dog barks"
        assert entry["media_type"] == "image"
        assert entry["num_training"] == 3
        binding = sharing_binding(get_detector(entry["id"]))
        assert binding is not None
        assert binding["source_name"] == "shared_folder"
        assert binding["remote_ref"] == ref
        assert binding["remote_owner"] == OWNER
        assert binding["integrity"] == {"status": INTEGRITY_CLEAN, "count": 0}
        assert binding["last_synced"] > 0
        assert binding["field_values"] == {}  # the fixed folder is laid in at call time, never stored
        assert _file_labels(entry) == {MD5["a"]: "good", MD5["b"]: "bad", MD5["c"]: "good"}
        data = _read(entry["name"])
        assert data["input_spec"] == {"clipper": "whole_image", "params": {}}
        assert "threshold" not in data

    def test_the_received_detector_is_read_only(self, folder):
        entry = _receive(_publish(folder))
        data = _read(entry["name"])
        with pytest.raises(DetectorReadOnlyError, match=OWNER):
            _write_detector(_detector_path(entry["name"]), {**data, "labelset": _labels(a="bad")})

    def test_the_same_container_twice_is_refused(self, folder):
        ref = _publish(folder)
        _receive(ref)
        with pytest.raises(ValueError, match="already have"):
            _receive(ref)

    def test_another_user_may_receive_it_too(self, folder):
        ref = _publish(folder)
        _receive(ref)
        carols = _receive(ref, who=CAROL)
        assert carols["created_by"] == CAROL
        assert carols["name"] == "Dog barks (2)"

    def test_a_name_collision_is_suffixed(self, folder):
        _write_detector(_detector_path("Dog barks"), {"name": "Dog barks", "media_type": "image", "labelset": {}})
        register_detector(name="Dog barks", media_type="image", created_by=BOB)
        entry = _receive(_publish(folder))
        assert entry["name"] == "Dog barks (2)"
        assert _read("Dog barks")["labelset"] == {}

    def test_abort_creates_nothing_when_foreign_records_exist(self, folder):
        ref = _publish(folder)
        simulate_foreign_modification(folder, ref, actor=CAROL, relabel=[MD5["a"]])
        before = len(list_detectors())
        with pytest.raises(ForeignModifications) as err:
            _receive(ref)
        assert err.value.integrity == {"status": INTEGRITY_FOREIGN, "count": 1}
        assert len(list_detectors()) == before
        assert not _detector_path("Dog barks").exists()

    def test_skip_takes_only_the_owners_records(self, folder):
        ref = _publish(folder)
        simulate_foreign_modification(
            folder, ref, actor=CAROL, relabel=[MD5["a"]], add=[LabeledElement(md5=MD5["d"], label="good")]
        )
        entry = _receive(ref, foreign=FOREIGN_SKIP)
        assert _file_labels(entry) == {MD5["b"]: "bad", MD5["c"]: "good"}
        assert _binding(entry["id"])["integrity"]["status"] == INTEGRITY_FOREIGN

    def test_import_takes_every_record(self, folder):
        ref = _publish(folder)
        simulate_foreign_modification(folder, ref, actor=CAROL, relabel=[MD5["a"]])
        entry = _receive(ref, foreign=FOREIGN_IMPORT)
        assert _file_labels(entry) == {MD5["a"]: "bad", MD5["b"]: "bad", MD5["c"]: "good"}

    def test_a_container_not_shared_with_the_caller_is_refused(self, folder):
        from vtscore.labels.sharing import RemotePermissionDenied

        ref = _publish(folder, recipients=(CAROL,))
        with pytest.raises(RemotePermissionDenied):
            _receive(ref)
        assert not _detector_path("Dog barks").exists()

    def test_sharing_off_is_unavailable(self, folder):
        ref = _publish(folder)
        _sharing_off()
        with pytest.raises(SharingUnavailable):
            _receive(ref)

    def test_an_unknown_policy_is_refused(self, folder):
        with pytest.raises(ValueError, match="foreign"):
            _receive(_publish(folder), foreign="merge")


# ---------------------------------------------------------------------------
# refresh
# ---------------------------------------------------------------------------


class TestRefresh:
    def test_an_owners_save_replaces_the_local_copy(self, folder):
        ref = _publish(folder)
        entry = _receive(ref)
        synced = _binding(entry["id"])["last_synced"]
        _save(folder, ref, _owner_detector(a="good", b="good", d="bad"))

        result = _refresh(entry["id"])

        assert result.applied
        assert (result.added, result.changed, result.removed) == (1, 1, 1)
        assert result.integrity == {"status": INTEGRITY_CLEAN, "count": 0}
        assert _file_labels(entry) == {MD5["a"]: "good", MD5["b"]: "good", MD5["d"]: "bad"}
        assert _binding(entry["id"])["last_synced"] >= synced
        assert _entry(entry["id"])["num_training"] == 3

    def test_abort_leaves_the_file_untouched_and_reports_the_count(self, folder):
        ref = _publish(folder)
        entry = _receive(ref)
        before = _detector_path(entry["name"]).read_bytes()
        simulate_foreign_modification(folder, ref, actor=CAROL, relabel=[MD5["a"], MD5["b"]])

        result = _refresh(entry["id"])

        assert not result.applied
        assert result.integrity == {"status": INTEGRITY_FOREIGN, "count": 2}
        assert _detector_path(entry["name"]).read_bytes() == before
        assert _binding(entry["id"])["integrity"]["count"] == 2

    def test_skip_keeps_only_the_owners_records(self, folder):
        ref = _publish(folder)
        entry = _receive(ref)
        simulate_foreign_modification(folder, ref, actor=CAROL, relabel=[MD5["a"]])
        result = _refresh(entry["id"], foreign=FOREIGN_SKIP)
        assert result.applied and result.removed == 1
        assert _file_labels(entry) == {MD5["b"]: "bad", MD5["c"]: "good"}

    def test_import_keeps_all(self, folder):
        ref = _publish(folder)
        entry = _receive(ref)
        simulate_foreign_modification(folder, ref, actor=CAROL, relabel=[MD5["a"]])
        result = _refresh(entry["id"], foreign=FOREIGN_IMPORT)
        assert result.applied and result.changed == 1
        assert _file_labels(entry)[MD5["a"]] == "bad"

    def test_a_loaded_context_sees_the_new_labels(self, folder):
        ref = _publish(folder)
        entry = _receive(ref)
        ctx = DetectorContext(entry["id"], name=entry["name"])
        register_detector_context(ctx)
        ids = {m["md5"]: cid for cid, m in get_active_context().medias.items() if m.get("md5")}

        _save(folder, ref, _owner_detector(a="good", d="good"))
        _refresh(entry["id"])

        assert ids[MD5["d"]] in ctx.good_votes
        assert ids[MD5["c"]] not in ctx.good_votes

    def test_only_a_received_detector_refreshes(self, folder):
        local = register_detector(name="mine", media_type="image")
        with pytest.raises(ValueError, match="not a received"):
            _refresh(local["id"])
        with pytest.raises(LookupError):
            _refresh("no-such-id")


# ---------------------------------------------------------------------------
# The source-mismatch rule
# ---------------------------------------------------------------------------


class TestMismatchedSource:
    @pytest.fixture
    def received(self, folder):
        return _receive(_publish(folder))

    def test_matches_while_the_source_is_active(self, received):
        assert binding_matches_active_source(get_detector(received["id"]))
        assert sharing_note(get_detector(received["id"])) is None

    def test_sharing_off_leaves_a_note_and_refuses_refresh(self, received):
        _sharing_off()
        entry = get_detector(received["id"])
        assert not binding_matches_active_source(entry)
        note = sharing_note(entry)
        assert note is not None and "sharing is off" in note
        with pytest.raises(SharingUnavailable, match="sharing is off"):
            _refresh(received["id"])

    def test_a_source_that_is_not_installed_leaves_a_note(self, received):
        binding = {**_binding(received["id"]), "source_name": "acme_drive"}
        update_detector(received["id"], labelset_source=binding)
        note = sharing_note(get_detector(received["id"]))
        assert note is not None and "not installed" in note
        with pytest.raises(SharingUnavailable, match="not installed"):
            _refresh(received["id"])

    def test_a_different_active_source_leaves_a_note(self, received):
        register_setting_getter("detector_sharing_source", lambda: {"source_name": "other_share", "field_values": {}})
        # ``other_share`` is not registered, so sharing is off - the note says that.
        assert "sharing is off" in (sharing_note(get_detector(received["id"])) or "")

    def test_it_stays_read_only_and_can_still_be_forked(self, received):
        _sharing_off()
        data = _read(received["name"])
        with pytest.raises(DetectorReadOnlyError):
            _write_detector(_detector_path(received["name"]), {**data, "labelset": _labels(a="bad")})
        with thread_user(BOB):
            fork = fork_detector(received["id"])
        assert sync_mode_of(fork) == SYNC_MODE_FORKED


class TestSyncStatus:
    """What the listing says a sharing binding syncs to (#5579), read as the sync engine reads it.

    A sharing binding stores no field values: the ``detector_sharing_source``
    setting's are laid in at push time, so the status must count them, or every
    shared and received row warns that its auto-export is not set up.
    """

    @pytest.fixture
    def received(self, folder):
        return _receive(_publish(folder))

    def test_the_setting_fills_what_the_source_needs(self, received):
        assert labelset_source_status(get_detector(received["id"])) == {
            "source_name": "shared_folder",
            "display_name": get_labelset_source("shared_folder").display_name,
            "inherited": False,
            "missing_fields": [],
        }

    def test_an_owners_shared_detector_lacks_nothing_either(self, folder):
        entry = register_detector(name="Owned", media_type="image", created_by=OWNER)
        binding = {"source_name": "shared_folder", "field_values": {}, "remote_ref": _publish(folder)}
        update_detector(entry["id"], labelset_source=binding)
        assert labelset_source_status(get_detector(entry["id"]))["missing_fields"] == []

    def test_the_deployment_default_has_no_say(self, received):
        register_setting_getter(
            "default_labelset_source",
            lambda: {"source_name": "server_json_file", "field_values": {}},
        )
        status = labelset_source_status(get_detector(received["id"]))
        assert status["source_name"] == "shared_folder"
        assert status["missing_fields"] == []

    def test_a_setting_that_leaves_a_required_field_empty_says_so(self, received):
        register_setting_getter("detector_sharing_source", lambda: {"source_name": "shared_folder", "field_values": {}})
        assert labelset_source_status(get_detector(received["id"]))["missing_fields"] == ["folder"]

    def test_a_binding_whose_source_is_not_in_use_lacks_nothing(self, received):
        _sharing_off()
        status = labelset_source_status(get_detector(received["id"]))
        assert status["source_name"] == "shared_folder"
        assert status["missing_fields"] == []


# ---------------------------------------------------------------------------
# fork
# ---------------------------------------------------------------------------


class TestFork:
    @pytest.fixture
    def received(self, folder):
        return _receive(_publish(folder))

    def _fork(self, detector_id: str, **kwargs) -> dict:
        with thread_user(BOB):
            return fork_detector(detector_id, **kwargs)

    def test_a_new_editable_detector_owned_by_the_caller(self, received, folder):
        original = _detector_path(received["name"]).read_bytes()
        fork = self._fork(received["id"])

        assert fork["id"] != received["id"]
        assert fork["name"] == "Dog barks (fork)"
        assert fork["created_by"] == BOB
        assert sync_mode_of(fork) == SYNC_MODE_FORKED
        assert fork["forked_from"] == {
            "detector_id": received["id"],
            "name": received["name"],
            "owner": OWNER,
            "source_name": "shared_folder",
            "remote_ref": _binding(received["id"])["remote_ref"],
        }
        assert _file_labels(fork) == _file_labels(received)
        assert _detector_path(received["name"]).read_bytes() == original

        data = _read(fork["name"])
        _write_detector(_detector_path(fork["name"]), {**data, "labelset": _labels(a="bad")})
        assert _file_labels(fork) == {MD5["a"]: "bad"}

    def test_it_gets_its_own_container_when_sharing_is_on(self, received, folder):
        fork = self._fork(received["id"])
        assert fork["share_error"] is None
        binding = sharing_binding(get_detector(fork["id"]))
        assert binding is not None
        assert binding["remote_ref"] != _binding(received["id"])["remote_ref"]
        with thread_user(BOB):
            mine = get_labelset_source("shared_folder").share_list_mine({"folder": folder})
        assert [s.remote_ref for s in mine] == [binding["remote_ref"]]

    def test_sharing_off_forks_locally(self, received):
        _sharing_off()
        fork = self._fork(received["id"])
        assert fork["share_error"] is None
        assert sharing_binding(get_detector(fork["id"])) is None

    def test_a_failed_create_leaves_a_working_local_fork(self, received, monkeypatch):
        from vtscore.labels.sharing import SharingError
        from vtscore.labels.sources.shared_folder import SharedFolderLabelsetSource

        def fail(self, labelset, meta, field_values):
            raise SharingError("the service is down")

        monkeypatch.setattr(SharedFolderLabelsetSource, "_do_share_create", fail)
        fork = self._fork(received["id"])
        assert fork["share_error"] == "the service is down"
        assert sharing_binding(get_detector(fork["id"])) is None
        assert sync_mode_of(get_detector(fork["id"])) == SYNC_MODE_FORKED

    def test_a_local_detector_forks_with_provenance(self, folder):
        _write_detector(_detector_path("mine"), _owner_detector(name="mine"))
        local = register_detector(name="mine", media_type="image", created_by=BOB)
        fork = self._fork(local["id"], name="mine, again")
        assert fork["name"] == "mine, again"
        assert fork["forked_from"] == {"detector_id": local["id"], "name": "mine", "owner": BOB}

    def test_a_taken_or_blank_name_is_refused(self, received):
        with pytest.raises(ValueError, match="already exists"):
            self._fork(received["id"], name=received["name"])
        with pytest.raises(ValueError):
            self._fork(received["id"], name="  ")

    def test_a_second_default_fork_is_suffixed(self, received):
        self._fork(received["id"])
        assert self._fork(received["id"])["name"] == "Dog barks (fork) (2)"
