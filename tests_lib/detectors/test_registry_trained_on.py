"""``record_detector_training`` keeps a detector's ``trained_on`` list (#5142).

The list names the datasets the detector holds labels from, so the Dashboard
can tell the dataset a detector was trained on from one it has never seen.
"""

from __future__ import annotations

from vtscore.detectors import registry


def _trained_on(detector_id: str) -> list[str] | None:
    entry = registry.get_detector(detector_id)
    assert entry is not None
    return entry.get("trained_on")


class TestRecordDetectorTraining:
    def test_stamps_count_time_and_dataset(self):
        a = registry.register_detector(name="A", media_type="image")

        assert registry.record_detector_training(a["id"], 4, dataset_id="ds-x") is True

        entry = registry.get_detector(a["id"])
        assert entry is not None
        assert entry["num_training"] == 4
        assert entry["last_trained_at"] is not None
        assert entry["trained_on"] == ["ds-x"]

    def test_each_dataset_is_listed_once(self):
        a = registry.register_detector(name="A", media_type="image")

        registry.record_detector_training(a["id"], 1, dataset_id="ds-x")
        registry.record_detector_training(a["id"], 2, dataset_id="ds-y")
        registry.record_detector_training(a["id"], 3, dataset_id="ds-x")

        assert sorted(_trained_on(a["id"]) or []) == ["ds-x", "ds-y"]

    def test_last_label_gone_drops_the_dataset(self):
        a = registry.register_detector(name="A", media_type="image")
        registry.record_detector_training(a["id"], 1, dataset_id="ds-x")
        registry.record_detector_training(a["id"], 2, dataset_id="ds-y")

        registry.record_detector_training(a["id"], 1, dataset_id="ds-x", labeled_in_dataset=False)

        assert _trained_on(a["id"]) == ["ds-y"]

    def test_no_dataset_leaves_the_list_alone(self):
        # A label import names no dataset: it changes the count, not where from.
        a = registry.register_detector(name="A", media_type="image")
        registry.record_detector_training(a["id"], 1, dataset_id="ds-x")

        registry.record_detector_training(a["id"], 5)

        entry = registry.get_detector(a["id"])
        assert entry is not None
        assert entry["num_training"] == 5
        assert entry["trained_on"] == ["ds-x"]

    def test_unknown_detector_reports_not_found(self):
        assert registry.record_detector_training("missing", 1, dataset_id="ds-x") is False
