"""Regression tests for re-entrant discovery in :class:`PluginRegistry`.

``_discover()`` imports the modules it scans, and an imported module is free
to call ``get()`` / ``list()`` on the very registry that is discovering it --
on the discovering thread, while the discovery lock is already held.  The
``_discovering`` guard in ``_ensure_discovered`` exists to hand that call a
partial registry, but it is only reachable if the lock is re-entrant; with a
plain ``threading.Lock`` the re-entrant acquire blocks forever and the process
hangs at import time with no error.

Every test here runs the exercise on a worker thread with a join timeout, so a
regression fails the suite instead of wedging it.
"""

from __future__ import annotations

import threading

from vtscore.plugins import PluginRegistry

#: Generous enough to absorb a loaded machine, short enough that a real
#: deadlock does not stall the suite for long.
JOIN_TIMEOUT = 30.0


def _run_with_deadlock_watchdog(fn):
    """Run *fn* on a worker thread; fail if it does not finish in time.

    Returns ``fn``'s value.  A deadlocked worker leaves the (daemon) thread
    parked on the lock forever, so this must never be used for work with
    side effects the rest of the suite depends on.
    """
    box: dict[str, object] = {}

    def target():
        try:
            box["value"] = fn()
        except BaseException as exc:  # pragma: no cover - surfaced below
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=JOIN_TIMEOUT)
    assert not thread.is_alive(), (
        f"re-entrant registry access did not return within {JOIN_TIMEOUT}s; "
        "the discovery lock is deadlocked (is it a plain Lock instead of an RLock?)"
    )
    if "error" in box:
        raise box["error"]  # type: ignore[misc]
    return box["value"]


class _FakePlugin:
    def __init__(self, name: str) -> None:
        self.name = name


def _deferred_registry() -> PluginRegistry:
    """A registry that has not discovered yet, so we can drive discovery."""
    return PluginRegistry(
        package="vtscore.exporters",
        sentinel="EXPORTER",
        label="exporter",
        eager=False,
    )


class TestReentrantDiscovery:
    def test_get_during_discovery_returns_partial_registry(self):
        """A scanned module calling ``get()`` at import time must not hang."""
        reg = _deferred_registry()
        seen: dict[str, object] = {}

        def fake_discover():
            # Stands in for the first scanned module registering its plugin.
            reg._items["early"] = _FakePlugin("early")  # type: ignore[assignment]
            # Stands in for the *next* scanned module calling back into the
            # registry at import time, on this same thread.
            seen["early"] = reg.get("early")
            seen["late"] = reg.get("late")
            reg._items["late"] = _FakePlugin("late")  # type: ignore[assignment]

        reg._discover = fake_discover  # type: ignore[method-assign]
        _run_with_deadlock_watchdog(reg.list)

        # The re-entrant call saw the partial registry: what was registered
        # before it, and nothing that came after.
        assert seen["early"] is not None
        assert seen["late"] is None
        # Discovery still completed and the registry is whole afterwards.
        assert reg._discovered is True
        assert {p.name for p in reg.list()} == {"early", "late"}

    def test_list_during_discovery_returns_partial_registry(self):
        reg = _deferred_registry()
        snapshots: list[list[str]] = []

        def fake_discover():
            reg._items["first"] = _FakePlugin("first")  # type: ignore[assignment]
            snapshots.append([p.name for p in reg.list()])
            reg._items["second"] = _FakePlugin("second")  # type: ignore[assignment]
            snapshots.append([p.name for p in reg.list()])

        reg._discover = fake_discover  # type: ignore[method-assign]
        _run_with_deadlock_watchdog(reg.list)

        assert snapshots == [["first"], ["first", "second"]]
        assert [p.name for p in reg.list()] == ["first", "second"]

    def test_reentrant_return_does_not_mark_discovered(self):
        """The early return must not claim discovery finished.

        If the re-entrant path set ``_discovered``, the outer discovery's own
        results would be published under a flag that was already true, and a
        *failed* discovery would leave the registry permanently marked done.
        """
        reg = _deferred_registry()
        observed: list[bool] = []

        def fake_discover():
            reg.list()
            observed.append(reg._discovered)

        reg._discover = fake_discover  # type: ignore[method-assign]
        _run_with_deadlock_watchdog(reg.list)

        assert observed == [False]
        assert reg._discovered is True

    def test_discovering_flag_cleared_when_discovery_raises(self):
        """A failed discovery must leave the registry retryable, not wedged."""
        reg = _deferred_registry()

        def exploding_discover():
            reg.get("anything")  # re-entrant call still must not deadlock
            raise RuntimeError("boom")

        reg._discover = exploding_discover  # type: ignore[method-assign]

        def attempt():
            try:
                reg.list()
            except RuntimeError as exc:
                return str(exc)
            return None

        assert _run_with_deadlock_watchdog(attempt) == "boom"
        assert reg._discovering is False
        assert reg._discovered is False

        # A later call re-runs discovery rather than serving an empty registry.
        reg._discover = lambda: reg._items.update(  # type: ignore[method-assign]
            {"recovered": _FakePlugin("recovered")}  # type: ignore[dict-item]
        )
        assert [p.name for p in reg.list()] == ["recovered"]
