"""``vtscore.concurrency.progress`` keeps its public surface after the split (#5429).

The module became a facade over ``progress_tracker``, ``loading_tracker`` and
``progress_cancel``.  Every name a caller imports from it must still resolve,
and as the *same object* its home module holds: a second ``loading_tasks``
would import and run fine while loads reported into one instance and
``tests_shared/state_reset.py`` cleared the other.
"""

from __future__ import annotations

import types

import pytest

from vtscore.concurrency import loading_tracker, progress, progress_cancel, progress_tracker

HOMES = (progress_tracker, loading_tracker, progress_cancel)


@pytest.mark.parametrize("name", progress.__all__)
def test_a_public_name_is_the_object_every_module_holds(name):
    obj = getattr(progress, name)
    for module in HOMES:
        if name in vars(module):
            assert vars(module)[name] is obj, f"{module.__name__}.{name} is a second copy"


@pytest.mark.parametrize("module", HOMES, ids=lambda m: m.__name__)
def test_every_public_name_of_a_split_module_is_re_exported(module):
    defined_here = {
        name
        for name, value in vars(module).items()
        if not name.startswith("_")
        and not isinstance(value, types.ModuleType)
        and getattr(value, "__module__", module.__name__) == module.__name__
    }
    assert defined_here - set(progress.__all__) == set()


def test_cancellation_reads_the_loading_tasks_the_facade_exports():
    tracker = progress.loading_tasks.create_task("split-check", "Loading")
    tracker.update("loading", "working", 1, 2)

    report = progress.cancel_dataset_task("split-check", grace_seconds=0)

    # No worker was ever registered, so the row is a phantom: found, and parked.
    assert report is not None
    assert report["unresponsive"] == ["split-check"]
    assert tracker.get()["status"] == "idle"
