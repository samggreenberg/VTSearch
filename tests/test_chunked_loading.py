"""Tests for chunked (piecewise) dataset loading.

Verifies that datasets can be loaded in chunks via the new
``load_dataset_from_folder_chunked``, ``load_dataset_from_pickle_chunked``
functions, the ``DatasetImporter.run_chunked`` / ``run_chunked_cli``
interface, and the CLI ``_merge_detector_results`` helper.
"""

import hashlib
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from vtsearch.datasets.loader import (
    load_dataset_from_folder,
    load_dataset_from_folder_chunked,
    load_dataset_from_pickle_chunked,
)


def _make_wav_bytes(frequency: float = 440.0, duration: float = 0.1) -> bytes:
    """Generate a minimal WAV file for testing."""
    from vtsearch.audio import generate_wav

    return generate_wav(frequency, duration)


def _make_wav_file(tmp_dir: Path, name: str, frequency: float = 440.0) -> Path:
    """Write a WAV file and return its path."""
    p = tmp_dir / name
    p.write_bytes(_make_wav_bytes(frequency))
    return p


def _make_pickle_with_base_freq(tmp_path: Path, num_clips: int, base_freq: float = 440.0) -> Path:
    """Create a test pickle with distinct WAV bytes per clip (using base_freq)."""
    clips_data: dict[int, dict[str, Any]] = {}
    for i in range(1, num_clips + 1):
        wav_bytes = _make_wav_bytes(frequency=base_freq + i * 10)
        clip: dict[str, Any] = {
            "id": i,
            "type": "audio",
            "duration": 0.1,
            "file_size": len(wav_bytes),
            "md5": hashlib.md5(wav_bytes).hexdigest(),
            "embedding": np.random.randn(512).tolist(),
            "filename": f"clip_{i}.wav",
            "category": f"cat_{i % 3}",
            "clip_bytes": wav_bytes,
        }
        clips_data[i] = clip

    pkl_path = tmp_path / "test_chunked.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"clips": clips_data}, f)
    return pkl_path


def _make_pickle(tmp_path: Path, num_clips: int, inline_bytes: bool = True) -> Path:
    """Create a test pickle with *num_clips* audio clips."""
    clips_data: dict[int, dict[str, Any]] = {}
    for i in range(1, num_clips + 1):
        wav_bytes = _make_wav_bytes(frequency=440.0 + i)
        clip: dict[str, Any] = {
            "id": i,
            "type": "audio",
            "duration": 0.1,
            "file_size": len(wav_bytes),
            "md5": hashlib.md5(wav_bytes).hexdigest(),
            "embedding": np.random.randn(512).tolist(),
            "filename": f"clip_{i}.wav",
            "category": f"cat_{i % 3}",
        }
        if inline_bytes:
            clip["clip_bytes"] = wav_bytes
        clips_data[i] = clip

    pkl_path = tmp_path / "test_chunked.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"clips": clips_data}, f)
    return pkl_path


# ======================================================================
# load_dataset_from_folder_chunked
# ======================================================================


class TestFolderChunked:
    """Test load_dataset_from_folder_chunked."""

    def test_single_chunk_when_fewer_than_chunk_size(self, tmp_path):
        """When total files < chunk_size, yields exactly one chunk."""
        _make_wav_file(tmp_path, "a.wav")
        _make_wav_file(tmp_path, "b.wav")
        chunks = list(load_dataset_from_folder_chunked(tmp_path, "sounds", chunk_size=10, thin=True))
        assert len(chunks) == 1
        assert len(chunks[0]) == 2

    def test_multiple_chunks(self, tmp_path):
        """Files are split across multiple chunks of the correct size."""
        for i in range(5):
            _make_wav_file(tmp_path, f"file_{i}.wav", frequency=440.0 + i * 10)
        chunks = list(load_dataset_from_folder_chunked(tmp_path, "sounds", chunk_size=2, thin=True))
        assert len(chunks) == 3  # 2, 2, 1
        assert len(chunks[0]) == 2
        assert len(chunks[1]) == 2
        assert len(chunks[2]) == 1

    def test_chunk_ids_start_at_one(self, tmp_path):
        """Each chunk's clip IDs start at 1 (not continuing from prior chunk)."""
        for i in range(4):
            _make_wav_file(tmp_path, f"file_{i}.wav", frequency=440.0 + i * 10)
        chunks = list(load_dataset_from_folder_chunked(tmp_path, "sounds", chunk_size=2, thin=True))
        for chunk in chunks:
            assert 1 in chunk

    def test_thin_mode_no_bytes(self, tmp_path):
        """Thin mode: clip_bytes is None, media_path is set."""
        _make_wav_file(tmp_path, "test.wav")
        chunks = list(load_dataset_from_folder_chunked(tmp_path, "sounds", chunk_size=10, thin=True))
        clip = chunks[0][1]
        assert clip["clip_bytes"] is None
        assert clip["media_path"] is not None
        assert Path(clip["media_path"]).exists()

    def test_full_mode_has_bytes(self, tmp_path):
        """Full mode: clip_bytes is populated."""
        _make_wav_file(tmp_path, "test.wav")
        chunks = list(load_dataset_from_folder_chunked(tmp_path, "sounds", chunk_size=10, thin=False))
        clip = chunks[0][1]
        assert clip["clip_bytes"] is not None

    def test_embeddings_present(self, tmp_path):
        """Each clip in a chunk has an embedding array."""
        _make_wav_file(tmp_path, "test.wav")
        chunks = list(load_dataset_from_folder_chunked(tmp_path, "sounds", chunk_size=10, thin=True))
        clip = chunks[0][1]
        assert isinstance(clip["embedding"], np.ndarray)
        assert len(clip["embedding"]) > 0

    def test_all_files_covered(self, tmp_path):
        """The total number of clips across all chunks equals total files."""
        for i in range(7):
            _make_wav_file(tmp_path, f"f_{i}.wav", frequency=440.0 + i * 10)
        chunks = list(load_dataset_from_folder_chunked(tmp_path, "sounds", chunk_size=3, thin=True))
        total_clips = sum(len(c) for c in chunks)
        assert total_clips == 7

    def test_matches_monolithic_load(self, tmp_path):
        """Chunked loading produces the same filenames as monolithic loading."""
        for i in range(5):
            _make_wav_file(tmp_path, f"f_{i}.wav", frequency=440.0 + i * 10)

        # Monolithic
        mono_clips: dict[int, dict[str, Any]] = {}
        load_dataset_from_folder(tmp_path, "sounds", mono_clips, thin=True)
        mono_filenames = {c["filename"] for c in mono_clips.values()}

        # Chunked
        chunked_filenames: set[str] = set()
        for chunk in load_dataset_from_folder_chunked(tmp_path, "sounds", chunk_size=2, thin=True):
            for clip in chunk.values():
                chunked_filenames.add(clip["filename"])

        assert mono_filenames == chunked_filenames

    def test_invalid_media_type_raises(self, tmp_path):
        _make_wav_file(tmp_path, "test.wav")
        try:
            list(load_dataset_from_folder_chunked(tmp_path, "bogus", chunk_size=10, thin=True))
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "Invalid media type" in str(e)

    def test_empty_folder_raises(self, tmp_path):
        try:
            list(load_dataset_from_folder_chunked(tmp_path, "sounds", chunk_size=10, thin=True))
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "No sounds files found" in str(e)


# ======================================================================
# load_dataset_from_pickle_chunked
# ======================================================================


class TestPickleChunked:
    """Test load_dataset_from_pickle_chunked."""

    def test_single_chunk(self, tmp_path):
        pkl_path = _make_pickle(tmp_path, 3)
        chunks = list(load_dataset_from_pickle_chunked(pkl_path, chunk_size=10, thin=True))
        assert len(chunks) == 1
        assert len(chunks[0]) == 3

    def test_multiple_chunks(self, tmp_path):
        pkl_path = _make_pickle(tmp_path, 5)
        chunks = list(load_dataset_from_pickle_chunked(pkl_path, chunk_size=2, thin=True))
        assert len(chunks) == 3  # 2, 2, 1
        assert len(chunks[0]) == 2
        assert len(chunks[1]) == 2
        assert len(chunks[2]) == 1

    def test_chunk_ids_start_at_one(self, tmp_path):
        pkl_path = _make_pickle(tmp_path, 4)
        chunks = list(load_dataset_from_pickle_chunked(pkl_path, chunk_size=2, thin=True))
        for chunk in chunks:
            assert 1 in chunk

    def test_thin_mode_drops_bytes(self, tmp_path):
        pkl_path = _make_pickle(tmp_path, 2, inline_bytes=True)
        chunks = list(load_dataset_from_pickle_chunked(pkl_path, chunk_size=10, thin=True))
        for clip in chunks[0].values():
            assert clip["clip_bytes"] is None

    def test_full_mode_keeps_bytes(self, tmp_path):
        pkl_path = _make_pickle(tmp_path, 2, inline_bytes=True)
        chunks = list(load_dataset_from_pickle_chunked(pkl_path, chunk_size=10, thin=False))
        for clip in chunks[0].values():
            assert clip["clip_bytes"] is not None

    def test_embeddings_are_numpy(self, tmp_path):
        pkl_path = _make_pickle(tmp_path, 2)
        chunks = list(load_dataset_from_pickle_chunked(pkl_path, chunk_size=10, thin=True))
        for clip in chunks[0].values():
            assert isinstance(clip["embedding"], np.ndarray)

    def test_all_clips_covered(self, tmp_path):
        pkl_path = _make_pickle(tmp_path, 7)
        chunks = list(load_dataset_from_pickle_chunked(pkl_path, chunk_size=3, thin=True))
        total = sum(len(c) for c in chunks)
        assert total == 7

    def test_metadata_preserved(self, tmp_path):
        pkl_path = _make_pickle(tmp_path, 1)
        chunks = list(load_dataset_from_pickle_chunked(pkl_path, chunk_size=10, thin=True))
        clip = chunks[0][1]
        assert clip["type"] == "audio"
        assert clip["filename"] == "clip_1.wav"
        assert clip["category"] == "cat_1"


# ======================================================================
# DatasetImporter.run_chunked / run_chunked_cli (base class default)
# ======================================================================


class TestBaseImporterChunkedDefault:
    """Test that the default run_chunked/run_chunked_cli on the base class
    delegates to run/run_cli and yields one chunk."""

    def test_default_run_chunked_yields_one_chunk(self, tmp_path):
        from vtsearch.datasets.importers.base import DatasetImporter, ImporterField

        class DummyImporter(DatasetImporter):
            name = "dummy"
            display_name = "Dummy"
            description = "Test"
            icon = ""
            fields: list[ImporterField] = []

            def run(self, field_values, clips, thin=False):
                clips[1] = {"id": 1, "type": "audio", "embedding": np.zeros(4)}
                clips[2] = {"id": 2, "type": "audio", "embedding": np.ones(4)}

        imp = DummyImporter()
        assert imp.supports_chunked is False

        chunks = list(imp.run_chunked({}, chunk_size=1, thin=True))
        assert len(chunks) == 1
        assert len(chunks[0]) == 2


# ======================================================================
# Folder importer run_chunked
# ======================================================================


class TestFolderImporterChunked:
    def test_supports_chunked(self):
        from vtsearch.datasets.importers.folder import FolderDatasetImporter

        assert FolderDatasetImporter().supports_chunked is True

    def test_run_chunked(self, tmp_path):
        for i in range(4):
            _make_wav_file(tmp_path, f"s_{i}.wav", frequency=440.0 + i * 10)
        from vtsearch.datasets.importers.folder import FolderDatasetImporter

        imp = FolderDatasetImporter()
        chunks = list(imp.run_chunked({"path": str(tmp_path), "media_type": "sounds"}, chunk_size=2, thin=True))
        assert len(chunks) == 2
        for chunk in chunks:
            assert len(chunk) == 2

    def test_run_chunked_cli(self, tmp_path):
        _make_wav_file(tmp_path, "test.wav")
        from vtsearch.datasets.importers.folder import FolderDatasetImporter

        imp = FolderDatasetImporter()
        chunks = list(imp.run_chunked_cli({"path": str(tmp_path), "media_type": "sounds"}, chunk_size=10, thin=True))
        assert len(chunks) == 1
        assert len(chunks[0]) == 1

    def test_run_chunked_cli_missing_folder(self, tmp_path):
        from vtsearch.datasets.importers.folder import FolderDatasetImporter

        imp = FolderDatasetImporter()
        try:
            list(imp.run_chunked_cli({"path": "/nonexistent/path", "media_type": "sounds"}, chunk_size=10))
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass


# ======================================================================
# Pickle importer run_chunked_cli
# ======================================================================


class TestPickleImporterChunked:
    def test_supports_chunked(self):
        from vtsearch.datasets.importers.pickle import PickleDatasetImporter

        assert PickleDatasetImporter().supports_chunked is True

    def test_run_chunked_cli(self, tmp_path):
        pkl_path = _make_pickle(tmp_path, 4)
        from vtsearch.datasets.importers.pickle import PickleDatasetImporter

        imp = PickleDatasetImporter()
        chunks = list(imp.run_chunked_cli({"file": str(pkl_path)}, chunk_size=2, thin=True))
        assert len(chunks) == 2
        total = sum(len(c) for c in chunks)
        assert total == 4

    def test_run_chunked_cli_missing_file(self, tmp_path):
        from vtsearch.datasets.importers.pickle import PickleDatasetImporter

        imp = PickleDatasetImporter()
        try:
            list(imp.run_chunked_cli({"file": "/nonexistent.pkl"}, chunk_size=10))
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError:
            pass


# ======================================================================
# CombineDatasets importer run_chunked
# ======================================================================


class TestCombineDatasetsImporterChunked:
    def test_supports_chunked(self):
        from vtsearch.datasets.importers.combine_datasets import CombineDatasetsImporter

        assert CombineDatasetsImporter().supports_chunked is True

    def test_yields_one_chunk_per_source(self, tmp_path):
        (tmp_path / "d1").mkdir(exist_ok=True)
        (tmp_path / "d2").mkdir(exist_ok=True)
        # Use different base frequencies so the WAV bytes (and MD5s) differ
        # between pickles, avoiding cross-source dedup.
        pkl1 = _make_pickle_with_base_freq(tmp_path / "d1", 3, base_freq=440.0)
        pkl2 = _make_pickle_with_base_freq(tmp_path / "d2", 2, base_freq=880.0)

        from vtsearch.datasets.importers.combine_datasets import CombineDatasetsImporter

        imp = CombineDatasetsImporter()
        chunks = list(imp.run_chunked({"datasets": f"{pkl1},{pkl2}"}, chunk_size=100))
        assert len(chunks) == 2
        assert len(chunks[0]) == 3
        assert len(chunks[1]) == 2


# ======================================================================
# _merge_detector_results
# ======================================================================


class TestMergeDetectorResults:
    def test_merge_new_detector(self):
        from vtsearch.cli import _merge_detector_results

        acc: dict[str, dict[str, Any]] = {}
        new = {
            "det_a": {
                "detector_name": "det_a",
                "threshold": 0.5,
                "total_hits": 2,
                "hits": [
                    {"filename": "f1.wav", "score": 0.9},
                    {"filename": "f2.wav", "score": 0.6},
                ],
            }
        }
        _merge_detector_results(acc, new)
        assert acc["det_a"]["total_hits"] == 2
        assert len(acc["det_a"]["hits"]) == 2

    def test_merge_extends_existing(self):
        from vtsearch.cli import _merge_detector_results

        acc = {
            "det_a": {
                "detector_name": "det_a",
                "threshold": 0.5,
                "total_hits": 1,
                "hits": [{"filename": "f1.wav", "score": 0.9}],
            }
        }
        new = {
            "det_a": {
                "detector_name": "det_a",
                "threshold": 0.5,
                "total_hits": 1,
                "hits": [{"filename": "f2.wav", "score": 0.7}],
            }
        }
        _merge_detector_results(acc, new)
        assert acc["det_a"]["total_hits"] == 2
        assert len(acc["det_a"]["hits"]) == 2
        # Verify sorted by score descending
        assert acc["det_a"]["hits"][0]["score"] >= acc["det_a"]["hits"][1]["score"]

    def test_merge_sorts_descending(self):
        from vtsearch.cli import _merge_detector_results

        acc = {
            "det_a": {
                "detector_name": "det_a",
                "threshold": 0.5,
                "total_hits": 1,
                "hits": [{"filename": "low.wav", "score": 0.5}],
            }
        }
        new = {
            "det_a": {
                "detector_name": "det_a",
                "threshold": 0.5,
                "total_hits": 1,
                "hits": [{"filename": "high.wav", "score": 0.99}],
            }
        }
        _merge_detector_results(acc, new)
        assert acc["det_a"]["hits"][0]["filename"] == "high.wav"
        assert acc["det_a"]["hits"][1]["filename"] == "low.wav"

    def test_merge_multiple_detectors(self):
        from vtsearch.cli import _merge_detector_results

        acc: dict[str, dict[str, Any]] = {}
        new = {
            "det_a": {"detector_name": "det_a", "threshold": 0.5, "total_hits": 1, "hits": [{"score": 0.9}]},
            "det_b": {"detector_name": "det_b", "threshold": 0.3, "total_hits": 1, "hits": [{"score": 0.8}]},
        }
        _merge_detector_results(acc, new)
        assert "det_a" in acc
        assert "det_b" in acc
