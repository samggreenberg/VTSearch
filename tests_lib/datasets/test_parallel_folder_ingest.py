"""Parallel per-file build in the folder loaders.

The full-mode folder loaders decode/encode each file's media on a small thread
pool (PIL/audio decode releases the GIL), but must produce results that are
byte-for-byte identical to the old serial path: same media IDs, same order,
same md5/filename/origin_name.  These tests pin that determinism, plus the
cancellation and worker-exception contracts the pool must preserve.

Library tier: no Flask / app modules.  A mock media type keeps the tests
hermetic — md5 is computed from the file bytes independently of
``load_media_data``, so ordering and hashing are exercised for real.
"""

from __future__ import annotations

import unittest.mock as mock
from contextlib import ExitStack
from pathlib import Path

import pytest

import vtscore.datasets.loader_folder as lf
from vtscore.concurrency.progress import CancelledError
from vtscore.datasets.loader import load_dataset_from_folder, load_dataset_from_folder_chunked
from tests_lib.helpers import make_wav_bytes as _make_wav_bytes

# Comfortably above _PARALLEL_MIN_FILES (64) so the pool actually engages.
_N_FILES = 130


def _make_mock_audio_type():
    mt = mock.MagicMock()
    mt.type_id = "audio"
    mt.file_extensions = ["*.wav"]
    mt.load_media_data.return_value = {"duration": 1.0}
    return mt


def _patch_registry(mt) -> ExitStack:
    stack = ExitStack()
    stack.enter_context(mock.patch("vtscore.media.get_by_folder_name", return_value=mt))
    return stack


def _make_files(folder: Path, n: int) -> None:
    """Write *n* distinct WAV files across two subdirs (exercises rel_path order)."""
    (folder / "a").mkdir()
    (folder / "b").mkdir()
    for i in range(n):
        sub = "a" if i % 2 == 0 else "b"
        # Distinct frequency per file → distinct bytes → distinct md5.
        (folder / sub / f"clip_{i:04d}.wav").write_bytes(_make_wav_bytes(frequency=100.0 + i))


def _fingerprint(medias: dict[int, dict]) -> tuple[list[int], dict[int, tuple]]:
    """Reduce a medias dict to the order-and-identity fields we assert on."""
    order = list(medias.keys())
    ident = {mid: (m["id"], m["md5"], m["filename"], m["origin_name"], m["file_size"]) for mid, m in medias.items()}
    return order, ident


class TestParallelMatchesSerial:
    def test_monolithic_parallel_equals_serial(self, tmp_path, monkeypatch):
        _make_files(tmp_path, _N_FILES)
        mt = _make_mock_audio_type()

        parallel: dict[int, dict] = {}
        with _patch_registry(mt):
            load_dataset_from_folder(tmp_path, "audio", parallel, on_progress=lambda *a: None)

        # Force the serial path by lifting the pool threshold above the file count.
        monkeypatch.setattr(lf, "_PARALLEL_MIN_FILES", 10**9)
        serial: dict[int, dict] = {}
        with _patch_registry(mt):
            load_dataset_from_folder(tmp_path, "audio", serial, on_progress=lambda *a: None)

        assert len(parallel) == _N_FILES
        assert _fingerprint(parallel) == _fingerprint(serial)
        # IDs are the contiguous 1..N the serial loader assigned.
        assert list(parallel.keys()) == list(range(1, _N_FILES + 1))

    def test_chunked_parallel_equals_serial(self, tmp_path, monkeypatch):
        _make_files(tmp_path, _N_FILES)
        mt = _make_mock_audio_type()

        with _patch_registry(mt):
            par_chunks = list(load_dataset_from_folder_chunked(tmp_path, "audio", chunk_size=70))

        monkeypatch.setattr(lf, "_PARALLEL_MIN_FILES", 10**9)
        with _patch_registry(mt):
            ser_chunks = list(load_dataset_from_folder_chunked(tmp_path, "audio", chunk_size=70))

        assert [len(c) for c in par_chunks] == [len(c) for c in ser_chunks] == [70, 60]
        for pc, sc in zip(par_chunks, ser_chunks):
            assert _fingerprint(pc) == _fingerprint(sc)
            # Each chunk restarts IDs at 1.
            assert list(pc.keys()) == list(range(1, len(pc) + 1))


class TestCancellation:
    def test_cancel_from_progress_stops_the_pool(self, tmp_path):
        _make_files(tmp_path, _N_FILES)
        mt = _make_mock_audio_type()

        calls = {"n": 0}

        def cancelling_progress(status, msg="", current=0, total=0):
            calls["n"] += 1
            if calls["n"] >= 2:
                raise CancelledError("user cancelled")

        target: dict[int, dict] = {}
        with _patch_registry(mt):
            with pytest.raises(CancelledError):
                load_dataset_from_folder(tmp_path, "audio", target, on_progress=cancelling_progress)


class TestWorkerException:
    def test_worker_error_surfaces_first_failing_file(self, tmp_path):
        _make_files(tmp_path, _N_FILES)
        mt = _make_mock_audio_type()

        # Exactly one file fails to decode.  The collection loop consumes in
        # submission order, so the failure surfaces at that file's position with
        # its path in the message, even though later files may have already been
        # decoded on other workers.
        def boom(file_path, media_bytes=None):
            if Path(file_path).name == "clip_0008.wav":
                raise ValueError(f"decode failed for {Path(file_path).name}")
            return {"duration": 1.0}

        mt.load_media_data.side_effect = boom

        target: dict[int, dict] = {}
        with _patch_registry(mt):
            with pytest.raises(ValueError, match="decode failed for clip_0008.wav"):
                load_dataset_from_folder(tmp_path, "audio", target, on_progress=lambda *a: None)
