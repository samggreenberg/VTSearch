"""Regression (audit #2): atomic load reservation for the dataset registry.

The ``.../load`` handler used an unguarded check-then-act — it read
``is_loaded()`` (the flag is only set at the *end* of the loader) and spawned a
background load. Two concurrent requests both saw ``False`` and started twin
loaders that shared a task id and raced to register/tear-down the same context.
``begin_load`` closes that window: the decision and the reservation happen under
a single lock.
"""

from __future__ import annotations

from vtscore.datasets import registry


class TestDatasetLoadReservation:
    def test_first_caller_reserves_others_see_in_progress(self):
        registry.reset_for_tests()
        ds_id = "a" * 32

        # The first caller wins the race and must run the load.
        assert registry.begin_load(ds_id) == "reserved"
        # Concurrent callers see the load already in flight and attach to it
        # instead of spawning a twin loader.
        assert registry.begin_load(ds_id) == "in_progress"
        assert registry.begin_load(ds_id) == "in_progress"
        # The loaded flag stays False until the loader completes.
        assert registry.is_loaded(ds_id) is False

    def test_reports_loaded_after_completion(self):
        registry.reset_for_tests()
        ds_id = "b" * 32

        assert registry.begin_load(ds_id) == "reserved"
        # Loader publishes the context, flips the loaded flag, releases the slot.
        registry.add_loaded_id(ds_id)
        registry.end_load(ds_id)
        # Any later caller is told it's already loaded (no reload, no twin).
        assert registry.begin_load(ds_id) == "loaded"

    def test_failed_load_releases_reservation_for_retry(self):
        registry.reset_for_tests()
        ds_id = "c" * 32

        assert registry.begin_load(ds_id) == "reserved"
        # Load failed: the flag is never set; the worker's ``finally`` releases
        # the reservation. A retry can reserve again rather than being wedged.
        registry.end_load(ds_id)
        assert registry.is_loaded(ds_id) is False
        assert registry.begin_load(ds_id) == "reserved"

    def test_distinct_ids_are_independent(self):
        registry.reset_for_tests()
        first, second = "d" * 32, "e" * 32

        assert registry.begin_load(first) == "reserved"
        # A different id is not blocked by an in-flight load of another.
        assert registry.begin_load(second) == "reserved"
        assert registry.begin_load(first) == "in_progress"
        assert registry.begin_load(second) == "in_progress"
