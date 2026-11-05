"""The read-only guarantee for received detectors, at the store (#5575).

Every detector write passes through ``_write_detector`` or
``queue_detector_write``; for a ``linked`` (received) detector both refuse one
that changes its name or labelset, except inside ``allow_linked_write()``,
while a write that leaves both as they are (a new threshold, a test verdict)
goes through.  The vote-persist path never writes a linked detector at all.
"""

from __future__ import annotations

import dataclasses

import pytest

from vtscore import config
from vtscore.datasets.labelset import LabeledElement, LabelSet
from vtscore.detectors import store
from vtscore.detectors.label_sync import _asserts_new_labels
from vtscore.detectors.registry import (
    SYNC_MODE_FORKED,
    SYNC_MODE_LINKED,
    SYNC_MODE_LOCAL,
    free_detector_name,
    get_detector,
    is_linked,
    register_detector,
    sync_mode_of,
)
from vtscore.detectors.store import (
    DetectorReadOnlyError,
    _detector_path,
    _read_detector,
    _write_detector,
    allow_linked_write,
    pending_detector_writes,
    queue_detector_write,
)

MD5_A, MD5_B, MD5_C = "a" * 32, "b" * 32, "c" * 32


@pytest.fixture(autouse=True)
def detectors_dir(tmp_path):
    """A scratch detectors dir (``reset_contexts`` restores the builder)."""
    scratch = dataclasses.replace(config.CoreConfig.from_settings(), detectors_dir=tmp_path / "detectors")
    config.register_core_config_builder(lambda path=None: scratch)
    return tmp_path / "detectors"


def _labelset(*pairs: tuple[str, str]) -> dict:
    return LabelSet([LabeledElement(md5=md5, label=label) for md5, label in pairs]).to_dict()


def _detector(name: str, *, mode: str | None = SYNC_MODE_LINKED) -> tuple[dict, dict]:
    """``(registry entry, file data)`` for a detector holding two labels."""
    data = {
        "name": name,
        "media_type": "image",
        "threshold": 0.4,
        "labelset": _labelset((MD5_A, "good"), (MD5_B, "bad")),
    }
    with allow_linked_write():
        _write_detector(_detector_path(name), data)
    binding = {"source_name": "shared_folder", "field_values": {}, "remote_ref": "ref-1", "remote_owner": "alice"}
    entry = register_detector(
        name=name,
        media_type="image",
        num_training=2,
        sync_mode=mode,
        labelset_source=binding if mode == SYNC_MODE_LINKED else None,
    )
    return entry, data


def _read(name: str) -> dict:
    data = _read_detector(_detector_path(name))
    assert data is not None, f"no file for {name!r}"
    return data


def _entry(detector_id: str) -> dict:
    entry = get_detector(detector_id)
    assert entry is not None, f"no entry {detector_id!r}"
    return entry


def _bytes(name: str) -> bytes:
    return _detector_path(name).read_bytes()


class TestSyncMode:
    def test_an_entry_without_a_mode_is_local(self):
        entry = register_detector(name="plain", media_type="image")
        assert "sync_mode" not in entry
        assert sync_mode_of(entry) == SYNC_MODE_LOCAL
        assert not is_linked(entry)

    def test_modes_are_stored(self):
        linked, _ = _detector("received")
        forked = register_detector(
            name="mine", media_type="image", sync_mode=SYNC_MODE_FORKED, forked_from={"name": "x"}
        )
        assert _entry(linked["id"])["sync_mode"] == SYNC_MODE_LINKED
        assert _entry(linked["id"])["labelset_source"]["remote_ref"] == "ref-1"
        assert _entry(forked["id"])["forked_from"] == {"name": "x"}
        assert is_linked(linked) and not is_linked(forked)

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ValueError, match="sync_mode"):
            register_detector(name="odd", media_type="image", sync_mode="mirrored")

    def test_a_free_name_is_suffixed_never_reused(self):
        _detector("Dog barks")
        assert free_detector_name("Dog barks") == "Dog barks (2)"
        assert free_detector_name("dog BARKS") == "dog BARKS (2)"  # same file slug
        assert free_detector_name("Cat purrs") == "Cat purrs"


class TestDirectWrites:
    def test_a_changed_labelset_is_refused_and_the_file_kept(self):
        _detector("received")
        before = _bytes("received")
        changed = {**_read("received"), "labelset": _labelset((MD5_A, "bad"), (MD5_B, "bad"))}

        with pytest.raises(DetectorReadOnlyError) as err:
            _write_detector(_detector_path("received"), changed)

        assert _bytes("received") == before
        assert err.value.owner == "alice"
        assert "alice" in str(err.value) and "Fork" in str(err.value)

    def test_a_new_name_in_the_file_is_refused(self):
        _detector("received")
        data = {**_read("received"), "name": "renamed"}
        with pytest.raises(DetectorReadOnlyError):
            _write_detector(_detector_path("received"), data)

    def test_a_slug_twin_of_the_path_is_guarded_too(self):
        _detector("Received")
        data = {"name": "received", "media_type": "image", "labelset": _labelset((MD5_C, "good"))}
        assert _detector_path("received") == _detector_path("Received")
        with pytest.raises(DetectorReadOnlyError):
            _write_detector(_detector_path("received"), data)

    def test_an_unchanged_labelset_with_a_new_threshold_goes_through(self):
        _, data = _detector("received")
        _write_detector(_detector_path("received"), {**data, "threshold": 0.7})
        assert _read("received")["threshold"] == 0.7

    def test_a_reordered_labelset_is_the_same_labelset(self):
        _, data = _detector("received")
        reordered = {**data, "labelset": _labelset((MD5_B, "bad"), (MD5_A, "good"))}
        _write_detector(_detector_path("received"), reordered)

    def test_allow_linked_write_admits_the_refresh(self):
        _, data = _detector("received")
        with allow_linked_write():
            _write_detector(_detector_path("received"), {**data, "labelset": _labelset((MD5_C, "good"))})
        assert _read("received")["labelset"] == _labelset((MD5_C, "good"))

    def test_allow_linked_write_is_scoped_to_the_block(self):
        _, data = _detector("received")
        with allow_linked_write(), allow_linked_write():
            pass
        with pytest.raises(DetectorReadOnlyError):
            _write_detector(_detector_path("received"), {**data, "labelset": _labelset((MD5_C, "good"))})

    @pytest.mark.parametrize("mode", [None, SYNC_MODE_FORKED])
    def test_a_local_or_forked_detector_is_not_guarded(self, mode):
        _, data = _detector("mine", mode=mode)
        _write_detector(_detector_path("mine"), {**data, "labelset": _labelset((MD5_C, "good"))})
        assert _read("mine")["labelset"] == _labelset((MD5_C, "good"))


class TestQueuedWrites:
    def test_the_queue_refuses_at_enqueue(self, monkeypatch):
        _, data = _detector("received")
        before = _bytes("received")
        monkeypatch.setattr(store, "_writer_autostart", False)
        store.set_detector_write_mode("async")
        try:
            with pytest.raises(DetectorReadOnlyError):
                queue_detector_write(_detector_path("received"), {**data, "labelset": _labelset((MD5_C, "good"))})
            assert pending_detector_writes() == []
        finally:
            store.set_detector_write_mode(None)
        assert _bytes("received") == before

    def test_the_queue_takes_an_unchanged_labelset(self, monkeypatch):
        _, data = _detector("received")
        store.set_detector_write_mode("sync")
        try:
            queue_detector_write(_detector_path("received"), {**data, "threshold": 0.9})
        finally:
            store.set_detector_write_mode(None)
        assert _read("received")["threshold"] == 0.9


class TestVotesNeverRewriteAMirror:
    """``_asserts_new_labels``: when the vote-persist path must refuse a linked detector."""

    def _ls(self, *elements: LabeledElement) -> LabelSet:
        return LabelSet(list(elements))

    def test_the_same_labels_under_another_origin_are_not_new(self):
        held = self._ls(LabeledElement(md5=MD5_A, label="good", origin={"importer": "folder", "params": {"p": "/a"}}))
        here = self._ls(LabeledElement(md5=MD5_A, label="good", origin={"importer": "folder", "params": {"p": "/b"}}))
        assert not _asserts_new_labels(held, here)

    def test_a_relabel_is_new(self):
        assert _asserts_new_labels(
            self._ls(LabeledElement(md5=MD5_A, label="good")), self._ls(LabeledElement(md5=MD5_A, label="bad"))
        )

    def test_an_addition_is_new(self):
        held = self._ls(LabeledElement(md5=MD5_A, label="good"))
        assert _asserts_new_labels(held, self._ls(*held.elements, LabeledElement(md5=MD5_B, label="bad")))

    def test_a_drawn_box_is_part_of_the_label(self):
        held = self._ls(LabeledElement(md5=MD5_A, label="good"))
        boxed = self._ls(LabeledElement(md5=MD5_A, label="good", region_box=(0.1, 0.1, 0.5, 0.5)))
        assert _asserts_new_labels(held, boxed)

    def test_a_dropped_label_is_not_reported(self):
        held = self._ls(LabeledElement(md5=MD5_A, label="good"), LabeledElement(md5=MD5_B, label="bad"))
        assert not _asserts_new_labels(held, self._ls(LabeledElement(md5=MD5_A, label="good")))
