"""``update_sharing_binding`` merges into the binding as it is on disk, only while it names the container (#5576)."""

from __future__ import annotations

from vtscore.detectors.registry import (
    get_detector,
    register_detector,
    rename_detector,
    update_detector,
    update_sharing_binding,
)

BINDING = {"source_name": "shared_folder", "field_values": {}, "remote_ref": "ref-1", "sync_error": None}


def _shared(name: str = "Dog barks") -> str:
    entry = register_detector(name=name, media_type="image", created_by="alice")
    update_detector(entry["id"], labelset_source=dict(BINDING))
    return entry["id"]


def _stored(detector_id: str) -> dict:
    entry = get_detector(detector_id)
    assert entry is not None
    return entry


def test_merges_into_the_current_binding():
    detector_id = _shared()
    update_detector(detector_id, labelset_source={**BINDING, "integrity": {"status": "clean", "count": 0}})

    merged = update_sharing_binding(detector_id, "ref-1", sync_error="down", last_synced=5.0)

    assert merged == {**BINDING, "integrity": {"status": "clean", "count": 0}, "sync_error": "down", "last_synced": 5.0}
    assert _stored(detector_id)["labelset_source"] == merged


def test_a_rename_made_meanwhile_is_kept():
    detector_id = _shared()
    rename_detector(detector_id, "Dog barks, renamed")
    update_sharing_binding(detector_id, "ref-1", sync_error=None, last_synced=5.0)
    assert _stored(detector_id)["name"] == "Dog barks, renamed"


def test_a_binding_that_no_longer_names_the_container_is_left_alone():
    detector_id = _shared()
    update_detector(detector_id, labelset_source={**BINDING, "remote_ref": "ref-2"})
    assert update_sharing_binding(detector_id, "ref-1", sync_error="down") is None
    assert _stored(detector_id)["labelset_source"]["remote_ref"] == "ref-2"
    assert _stored(detector_id)["labelset_source"]["sync_error"] is None

    update_detector(detector_id, labelset_source=None)
    assert update_sharing_binding(detector_id, "ref-2", sync_error="down") is None
    assert _stored(detector_id).get("labelset_source") is None


def test_an_unknown_detector_is_none():
    assert update_sharing_binding("no-such-id", "ref-1", sync_error="down") is None
