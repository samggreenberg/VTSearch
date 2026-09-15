"""The background detector-file writer (issue #3853).

Every vote rewrites the whole detector JSON; on the GRID that file lives on
a busy NFS export whose ``fsync`` tail runs to seconds, and the write sat
inline in ``POST /api/medias/<id>/vote``.  :func:`queue_detector_write`
moves it to a background thread.  These tests pin the contract that keeps
the file "true enough" while it lags: in-process readers see queued text,
direct writers drain the queue first, deletes discard it, a failure is
surfaced on the next call, and the vote route itself returns while the
filesystem is still stalled.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from vtscore.detectors import store
from vtscore.detectors.store import (
    DetectorWriteError,
    _read_detector,
    _write_detector,
    discard_pending_detector_write,
    flush_detector_writes,
    pending_detector_writes,
    queue_detector_write,
    reset_detector_write_queue_for_tests,
    set_detector_write_mode,
)


@pytest.fixture
def async_writes():
    """Run the test in ``async`` mode; the suite-wide default is ``sync``."""
    set_detector_write_mode("async")
    try:
        yield
    finally:
        reset_detector_write_queue_for_tests()
        set_detector_write_mode("sync")


class _GatedWriter:
    """Stand-in for ``_atomic_write_text`` that blocks until released."""

    def __init__(self, real, *, fail_first: bool = False) -> None:
        self.real = real
        self.gate = threading.Event()
        self.calls: list[str] = []
        self.fail_first = fail_first

    def __call__(self, path, text):
        self.gate.wait(timeout=10)
        self.calls.append(text)
        if self.fail_first:
            self.fail_first = False
            raise OSError("simulated ENOSPC")
        self.real(path, text)


def _path(name="det"):
    return store.get_detectors_dir() / f"{name}.json"


def test_sync_mode_writes_inline():
    set_detector_write_mode("sync")
    path = _path()
    ran: list[str] = []
    queue_detector_write(path, {"n": 1}, after=lambda: ran.append("after"))
    assert json.loads(path.read_text()) == {"n": 1}
    assert ran == ["after"]
    assert pending_detector_writes() == []


def test_async_queue_returns_before_the_write_and_coalesces(monkeypatch, async_writes):
    path = _path()
    gate = _GatedWriter(store._atomic_write_text)
    monkeypatch.setattr(store, "_atomic_write_text", gate)
    order: list[str] = []

    t0 = time.perf_counter()
    queue_detector_write(path, {"n": 1}, after=lambda: order.append("after-1"))
    queue_detector_write(path, {"n": 2}, after=lambda: order.append("after-2"))
    assert time.perf_counter() - t0 < 1.0, "queueing must not wait for the filesystem"
    assert pending_detector_writes() == [path]
    # Readers see what the file *will* contain, not the (absent) file.
    assert _read_detector(path) == {"n": 2}
    assert not path.exists()

    gate.gate.set()
    flush_detector_writes(path)
    assert json.loads(path.read_text()) == {"n": 2}
    assert pending_detector_writes() == []
    # The second queue call replaced the first before the writer got to it,
    # so at most two physical writes happened and the newest text won.
    assert json.loads(gate.calls[-1]) == {"n": 2}
    assert order[-1] == "after-2"


def test_read_detector_returns_a_private_copy_of_queued_text(monkeypatch, async_writes):
    path = _path()
    gate = _GatedWriter(store._atomic_write_text)
    monkeypatch.setattr(store, "_atomic_write_text", gate)
    queue_detector_write(path, {"labelset": {"labels": [1]}})
    seen = _read_detector(path)
    assert seen is not None
    seen["labelset"]["labels"].append(2)
    assert _read_detector(path) == {"labelset": {"labels": [1]}}
    gate.gate.set()
    flush_detector_writes()


def test_direct_write_drains_the_queue_first(monkeypatch, async_writes):
    """A read-modify-write through ``_write_detector`` can never be overtaken by older queued text."""
    path = _path()
    gate = _GatedWriter(store._atomic_write_text)
    monkeypatch.setattr(store, "_atomic_write_text", gate)
    order: list[str] = []
    queue_detector_write(path, {"n": 1}, after=lambda: order.append("queued-landed"))
    gate.gate.set()
    data = _read_detector(path)
    assert data is not None and data == {"n": 1}
    data["n"] = 2
    _write_detector(path, data)
    order.append("direct-landed")
    assert json.loads(path.read_text()) == {"n": 2}
    assert order == ["queued-landed", "direct-landed"]
    assert pending_detector_writes() == []


def test_failed_background_write_is_raised_from_the_next_queue_call(monkeypatch, async_writes):
    path = _path()
    gate = _GatedWriter(store._atomic_write_text, fail_first=True)
    monkeypatch.setattr(store, "_atomic_write_text", gate)
    gate.gate.set()
    queue_detector_write(path, {"n": 1})
    flush_detector_writes(path)
    assert not path.exists()
    assert pending_detector_writes() == []
    with pytest.raises(DetectorWriteError, match="previous write"):
        queue_detector_write(path, {"n": 2})
    # The newer text was still queued, so the next successful pass repairs the file.
    flush_detector_writes(path)
    assert json.loads(path.read_text()) == {"n": 2}
    # Reported once.
    queue_detector_write(path, {"n": 3})
    flush_detector_writes(path)


def test_discard_pending_drops_the_write(monkeypatch, async_writes):
    # Park the queue (no writer thread) so the discard races nothing.
    monkeypatch.setattr(store, "_writer_autostart", False)
    path = _path()
    queue_detector_write(path, {"n": 1})
    assert pending_detector_writes() == [path]
    discard_pending_detector_write(path)
    flush_detector_writes()
    assert not path.exists()
    assert pending_detector_writes() == []


def test_parked_queue_lands_on_flush_in_order(monkeypatch, async_writes):
    monkeypatch.setattr(store, "_writer_autostart", False)
    a, b = _path("a"), _path("b")
    queue_detector_write(a, {"n": 1})
    queue_detector_write(b, {"n": 1})
    queue_detector_write(a, {"n": 2})
    assert sorted(pending_detector_writes()) == sorted([a, b])
    flush_detector_writes(a)
    assert json.loads(a.read_text()) == {"n": 2}
    assert pending_detector_writes() == [b]
    flush_detector_writes()
    assert json.loads(b.read_text()) == {"n": 1}
    assert pending_detector_writes() == []


class TestVoteRoute:
    """The real chain: vote -> ``sync_labels_to_loaded_detector`` -> queue -> disk."""

    @pytest.fixture(autouse=True)
    def clean_detectors_dir(self):
        import shutil

        from vtsearch.settings import get_detectors_dir

        def _wipe():
            d = get_detectors_dir()
            if d.is_dir():
                shutil.rmtree(d)

        _wipe()
        yield
        _wipe()

    def test_vote_returns_while_the_filesystem_is_stalled(self, client, monkeypatch, async_writes):
        """The whole point: a blocked ``fsync`` no longer holds the vote POST."""
        from tests import load_detector_and_wait
        from vtscore.detectors.store import _detector_path
        from vtsearch.state import good_votes, medias

        if not medias:
            pytest.skip("No medias loaded")
        media_id = next(iter(medias))
        name = "StallQueue"
        res = client.post("/api/detectors/registry", json={"name": name, "media_type": "audio", "text_query": "t"})
        assert res.status_code in (200, 201), res.get_data(as_text=True)
        load_detector_and_wait(client, res.get_json()["detector"]["id"])
        path = _detector_path(name)
        flush_detector_writes(path)

        # From here the filesystem is "stalled": every physical write blocks.
        gate = _GatedWriter(store._atomic_write_text)
        monkeypatch.setattr(store, "_atomic_write_text", gate)
        try:
            t0 = time.perf_counter()
            resp = client.post(f"/api/medias/{media_id}/vote", json={"target": "good"})
            elapsed = time.perf_counter() - t0
            assert resp.status_code == 200, resp.get_data(as_text=True)
            assert media_id in good_votes
            assert elapsed < 2.0, f"vote waited {elapsed:.1f}s on a stalled write"
            assert pending_detector_writes() == [path]
            # Every in-process reader already sees the label ...
            queued = _read_detector(path)
            assert queued is not None
            assert [el["label"] for el in queued["labelset"]["labels"]] == ["good"]
            resp = client.get(f"/api/detectors/{name}")
            assert resp.status_code == 200
            assert [el["label"] for el in resp.get_json()["labelset"]["labels"]] == ["good"]
            # ... while the file itself still says nothing was voted.
            assert json.loads(path.read_text())["labelset"]["labels"] == []
        finally:
            gate.gate.set()
        # The file catches up once the filesystem answers.
        flush_detector_writes(path)
        assert [el["label"] for el in json.loads(path.read_text())["labelset"]["labels"]] == ["good"]
        assert pending_detector_writes() == []
