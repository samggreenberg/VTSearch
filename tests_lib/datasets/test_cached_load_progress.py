"""Library-tier tests: loading a dataset out of a ``.pkl`` reports progress.

Reading the cached container is the *whole* job for the two import paths
covered here — a demo the picker shows as "Ready", and a user-uploaded
``.pkl`` — and on a large dataset it is the slowest thing in the load.  Both
call sites used to invoke ``load_dataset_from_pickle`` without forwarding a
progress callback, so the dashboard row parked on a single static message
with no counter and no bar movement until the entire read finished.  These
tests pin the callback wiring so the row keeps ticking.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vtscore.datasets import loader_demo as _loader_demo
from vtscore.datasets.config import DEMO_DATASETS
from vtscore.datasets.loader import export_dataset_to_file, load_demo_dataset

#: A real image-typed demo id, so ``_try_load_cached`` takes the cached branch
#: for a media type whose pickled medias round-trip without external files.
DEMO_ID = "caltech101_s"


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _image_medias(n: int) -> dict[int, dict]:
    """Build *n* image medias with inline bytes and deterministic embeddings."""
    rng = np.random.default_rng(7)
    medias: dict[int, dict] = {}
    for i in range(n):
        src = _png_bytes((i % 255, (3 * i) % 255, (7 * i) % 255))
        medias[i] = {
            "id": i,
            "media_type": "image",
            "duration": 0,
            "file_size": len(src),
            "md5": f"md5-{i}",
            "embedder": "siglip",
            "embeddings": {"siglip": rng.standard_normal(32).astype(np.float32)},
            "filename": f"{i}.png",
            "category": "test",
            "media_bytes": src,
            "width": 8,
            "height": 8,
        }
    return medias


def _write_cached_demo(tmp_path: Path, monkeypatch, n: int = 120) -> dict[int, dict]:
    """Plant a cached demo container in a redirected ``EMBEDDINGS_DIR``."""
    medias = _image_medias(n)
    container = export_dataset_to_file(medias, embedder="siglip", media_type="image")
    monkeypatch.setattr(_loader_demo, "EMBEDDINGS_DIR", tmp_path)
    (tmp_path / f"{DEMO_ID}.pkl").write_bytes(container)
    return medias


class _Recorder:
    """Collect ``(status, message, current, total)`` progress calls."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, int, int]] = []

    def __call__(self, status: str, message: str = "", current: int = 0, total: int = 0, **_: object) -> None:
        self.events.append((status, message, current, total))

    @property
    def counted(self) -> list[tuple[str, str, int, int]]:
        """Events that carry a real denominator, i.e. move a determinate bar."""
        return [e for e in self.events if e[3] > 0]

    @property
    def fractions(self) -> list[float]:
        """The bar fraction each counted event puts on screen."""
        return [current / total for _, _, current, total in self.counted]

    def phase(self, prefix: str) -> list[float]:
        """Fractions reported by counted events whose message starts *prefix*."""
        return [c / t for _, message, c, t in self.counted if message.startswith(prefix)]


def assert_one_monotone_bar(progress: _Recorder) -> None:
    """Pin the invariant that makes the read reportable without stalling the loop.

    The load's two sub-phases measure different things (bytes consumed, then
    items built), so they must still share one denominator and one
    non-decreasing fraction.  Were they each given their native scale, the
    handover would rewind the fraction to zero — and because every consumer
    clamps monotonically (``ProgressTracker._compute_overall`` pins
    ``overall``, ``AdaptiveLoadPacer.update`` pins ``_frac``), that rewind
    surfaces as the bar *freezing* for the whole item loop rather than
    visibly retreating.
    """
    counted = progress.counted
    assert counted, "load must report progress with a real denominator"
    assert len({total for _, _, _, total in counted}) == 1, "both sub-phases must report against one shared denominator"
    fractions = progress.fractions
    assert all(b >= a for a, b in zip(fractions, fractions[1:])), f"bar fraction must never retreat, got {fractions}"


def test_cached_demo_load_reports_per_item_progress(tmp_path: Path, monkeypatch):
    """A "Ready" demo ticks per-item progress, not one static message."""
    assert DEMO_ID in DEMO_DATASETS, "test fixture must name a real demo id"
    medias = _write_cached_demo(tmp_path, monkeypatch)

    loaded: dict = {}
    progress = _Recorder()
    load_demo_dataset(DEMO_ID, loaded, on_progress=progress)

    assert loaded.keys() == medias.keys()
    assert_one_monotone_bar(progress)
    # More than the single opening tick, so the bar actually advances mid-read.
    assert max(current for _, _, current, _ in progress.counted) > 0


def test_cached_demo_load_still_finishes_idle(tmp_path: Path, monkeypatch):
    """Forwarding the callback must not disturb the terminal ``idle`` event."""
    _write_cached_demo(tmp_path, monkeypatch, n=20)

    loaded: dict = {}
    progress = _Recorder()
    load_demo_dataset(DEMO_ID, loaded, on_progress=progress)

    assert progress.events[-1][0] == "idle"


def test_pickle_importer_reports_per_item_progress(tmp_path: Path):
    """The uploaded-``.pkl`` importer reports through the per-task callback."""
    from vtscore.concurrency.progress import clear_thread_progress, set_thread_progress
    from vtscore.datasets import get_importer
    from vtscore.plugins.uploads import BytesIOUploadedFile

    medias = _image_medias(120)
    container = export_dataset_to_file(medias, embedder="siglip", media_type="image")

    importer = get_importer("pickle")
    assert importer is not None

    progress = _Recorder()
    loaded: dict = {}
    set_thread_progress(progress)
    try:
        importer.run({"file": BytesIOUploadedFile(container, "ds.pkl")}, loaded)
    finally:
        clear_thread_progress()

    assert loaded.keys() == medias.keys()
    assert_one_monotone_bar(progress)


@pytest.mark.parametrize("n", [2, 51, 120])
def test_progress_stays_one_monotone_bar_at_every_size(tmp_path: Path, monkeypatch, n: int):
    """The bar stays single-denominator and monotone at every dataset size.

    ``load_dataset_from_pickle`` derives its tick interval from the item
    count, so a size that lands on an interval boundary must not silently
    drop every tick.
    """
    _write_cached_demo(tmp_path, monkeypatch, n=n)

    loaded: dict = {}
    progress = _Recorder()
    load_demo_dataset(DEMO_ID, loaded, on_progress=progress)

    assert len(loaded) == n
    assert_one_monotone_bar(progress)


def test_container_read_advances_the_bar(tmp_path: Path, monkeypatch):
    """The container read reports real movement instead of a dark window.

    Reading the container — streaming ``medias.pkl`` and deserialising it —
    used to report ``total == 0`` for its whole duration, which renders as an
    indeterminate bar that never advances.  On the large demos that is tens of
    seconds of dead bar right after the user clicks Import.
    """
    _write_cached_demo(tmp_path, monkeypatch, n=120)

    loaded: dict = {}
    progress = _Recorder()
    load_demo_dataset(DEMO_ID, loaded, on_progress=progress)

    read_fractions = progress.phase("Reading")
    assert read_fractions, "the container read must report counted progress"
    assert max(read_fractions) > 0, "the read must advance the bar, not park it at zero"


def test_read_and_item_phases_partition_the_bar(tmp_path: Path, monkeypatch):
    """The read hands the bar to the item loop; neither replays the other's span.

    The read owns ``[0, _READ_SHARE]`` and the item loop ``[_READ_SHARE, 1]``,
    so the two together sweep the bar exactly once.
    """
    from vtscore.datasets.loader_pickle import _READ_SHARE

    _write_cached_demo(tmp_path, monkeypatch, n=120)

    loaded: dict = {}
    progress = _Recorder()
    load_demo_dataset(DEMO_ID, loaded, on_progress=progress)

    read_fractions = progress.phase("Reading")
    item_fractions = progress.phase("Processing")
    assert max(read_fractions) <= _READ_SHARE + 1e-9
    assert min(item_fractions) >= _READ_SHARE - 1e-9
    # The item loop finishes the bar rather than stopping short of it.
    assert max(item_fractions) == pytest.approx(1.0, abs=1e-3)
