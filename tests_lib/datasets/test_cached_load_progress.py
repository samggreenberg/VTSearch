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

from vtscore.datasets import loader as _loader
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
    monkeypatch.setattr(_loader, "EMBEDDINGS_DIR", tmp_path)
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


def test_cached_demo_load_reports_per_item_progress(tmp_path: Path, monkeypatch):
    """A "Ready" demo ticks per-item progress, not one static message."""
    assert DEMO_ID in DEMO_DATASETS, "test fixture must name a real demo id"
    medias = _write_cached_demo(tmp_path, monkeypatch)

    loaded: dict = {}
    progress = _Recorder()
    load_demo_dataset(DEMO_ID, loaded, on_progress=progress)

    assert loaded.keys() == medias.keys()
    counted = progress.counted
    assert counted, "cached demo load must report progress with a real total"
    assert all(total == len(medias) for _, _, _, total in counted)
    # More than the single opening tick, so the bar actually advances mid-read.
    assert max(current for _, _, current, _ in counted) > 0


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
    counted = progress.counted
    assert counted, "pickle import must report progress with a real total"
    assert all(total == len(medias) for _, _, _, total in counted)


@pytest.mark.parametrize("n", [2, 51, 120])
def test_progress_totals_match_item_count(tmp_path: Path, monkeypatch, n: int):
    """The reported denominator is the item count at every dataset size.

    ``load_dataset_from_pickle`` derives its tick interval from the item
    count, so a size that lands on an interval boundary must not silently
    drop every tick.
    """
    _write_cached_demo(tmp_path, monkeypatch, n=n)

    loaded: dict = {}
    progress = _Recorder()
    load_demo_dataset(DEMO_ID, loaded, on_progress=progress)

    assert len(loaded) == n
    assert progress.counted, f"no counted progress for n={n}"
    assert all(total == n for _, _, _, total in progress.counted)
