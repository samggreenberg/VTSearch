"""Progress-event cadence for the folder loaders.

Regression guard: the loaders must cap the *update count* at ~50 for the whole
walk, not cap the *interval* at 50 files.  The old ``min(50, total // 50)``
form inverted above 2500 files — a 300k-file import emitted ~6,000 progress
events, each a full task-list re-serialisation pushed to every open SSE stream.
"""

from pathlib import Path
from typing import Any

import pytest

from vtscore.datasets.loader import (
    load_dataset_from_folder,
    load_dataset_from_folder_chunked,
)

# Well above the 2500-file point where the old formula inverted: with the
# correct formula this folder yields ~50 events, with the old one ~120.
_FILE_COUNT = 6000


@pytest.fixture(scope="module")
def big_text_folder(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("big_text_folder")
    for i in range(_FILE_COUNT):
        (folder / f"f{i}.txt").write_text("hello")
    return folder


def _count_progress_calls(fn) -> int:
    calls: list[tuple] = []
    fn(lambda *args: calls.append(args))
    return len(calls)


class TestFolderProgressCadence:
    def test_monolithic_loader_emits_about_fifty_updates(self, big_text_folder):
        medias: dict[int, dict[str, Any]] = {}
        count = _count_progress_calls(
            lambda cb: load_dataset_from_folder(big_text_folder, "text", medias, thin=True, on_progress=cb)
        )
        assert len(medias) == _FILE_COUNT
        # ~50 per-file ticks plus the scan/finish bookends — never total // 50.
        assert 40 <= count <= 60, f"expected ~50 progress events for {_FILE_COUNT} files, got {count}"

    def test_chunked_loader_emits_about_fifty_updates_when_total_known(self, big_text_folder):
        # An override map forces the eager (known-total) branch of the chunked loader.
        def run(cb):
            for _chunk in load_dataset_from_folder_chunked(
                big_text_folder,
                "text",
                500,
                custom_metadata_map={"f0.txt": {"note": "x"}},
                on_progress=cb,
                thin=True,
            ):
                pass

        count = _count_progress_calls(run)
        assert 40 <= count <= 60, f"expected ~50 progress events for {_FILE_COUNT} files, got {count}"

    def test_chunked_loader_uses_fixed_interval_when_total_unknown(self, big_text_folder):
        # No override map: files stream lazily, so the total is unknown and the
        # loader falls back to a fixed 200-file interval.
        def run(cb):
            for _chunk in load_dataset_from_folder_chunked(
                big_text_folder, "text", 500, on_progress=cb, thin=True
            ):
                pass

        count = _count_progress_calls(run)
        assert 20 <= count <= 40, f"expected ~30 progress events for {_FILE_COUNT} files, got {count}"
