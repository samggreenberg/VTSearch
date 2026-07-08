"""Regression (audit #2): atomic load reservation for the detector registry.

Mirrors the dataset-registry reservation. Two concurrent
``POST /api/detectors/registry/load`` both passed the ``is_detector_loaded``
check (the flag is only set at the end of the loader) and spawned twin loaders
that shared a task id and raced to register/tear-down the same context.
``begin_detector_load`` closes that window.
"""

from __future__ import annotations

from vtscore.detectors import registry


class TestDetectorLoadReservation:
    def test_first_caller_reserves_others_see_in_progress(self):
        registry.reset_for_tests()
        det_id = "a" * 32

        assert registry.begin_detector_load(det_id) == "reserved"
        assert registry.begin_detector_load(det_id) == "in_progress"
        assert registry.begin_detector_load(det_id) == "in_progress"
        assert registry.is_detector_loaded(det_id) is False

    def test_reports_loaded_after_completion(self):
        registry.reset_for_tests()
        det_id = "b" * 32

        assert registry.begin_detector_load(det_id) == "reserved"
        registry.add_loaded_detector_id(det_id)
        registry.end_detector_load(det_id)
        assert registry.begin_detector_load(det_id) == "loaded"

    def test_failed_load_releases_reservation_for_retry(self):
        registry.reset_for_tests()
        det_id = "c" * 32

        assert registry.begin_detector_load(det_id) == "reserved"
        registry.end_detector_load(det_id)
        assert registry.is_detector_loaded(det_id) is False
        assert registry.begin_detector_load(det_id) == "reserved"

    def test_distinct_ids_are_independent(self):
        registry.reset_for_tests()
        first, second = "d" * 32, "e" * 32

        assert registry.begin_detector_load(first) == "reserved"
        assert registry.begin_detector_load(second) == "reserved"
        assert registry.begin_detector_load(first) == "in_progress"
        assert registry.begin_detector_load(second) == "in_progress"
