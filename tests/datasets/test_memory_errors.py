"""Tests for graceful MemoryError handling during dataset loading."""

import pickle
from unittest import mock

import numpy as np
import pytest

import app as app_module
from vtscore.datasets.config import DEMO_DATASETS
from vtsearch.state import medias
from vtscore.concurrency.progress import (
    get_progress,
    update_progress,
)
from vtsearch.state import clear_medias


class TestClearClipsGarbageCollection:
    """clear_medias() should call gc.collect() to free old dataset memory."""

    def test_clear_medias_calls_gc_collect(self):
        with mock.patch("vtscore.state.gc.collect") as mock_gc:
            clear_medias()
            mock_gc.assert_called_once()

    def test_clear_medias_empties_dict(self):
        medias[999] = {"id": 999, "media_type": "audio", "embedding": np.zeros(10)}
        clear_medias()
        assert len(medias) == 0
        # Restore test medias
        app_module.init_medias()


class TestPickleMemoryError:
    """load_dataset_from_pickle should handle MemoryError gracefully."""

    def test_pickle_load_oom_raises_with_message(self, tmp_path):
        """If pickle.load itself OOMs, a clear MemoryError is raised."""
        from vtscore.datasets.loader import load_dataset_from_pickle

        pkl = tmp_path / "big.pkl"
        pkl.write_bytes(pickle.dumps({"medias": {}}))

        target: dict = {}
        with mock.patch("vtscore.datasets.loader_pickle.safe_pickle_load", side_effect=MemoryError):
            with pytest.raises(MemoryError, match="too large for available RAM"):
                load_dataset_from_pickle(pkl, target)

        assert len(target) == 0

    def test_pickle_clip_loop_oom_clears_clips(self, tmp_path):
        """If MemoryError occurs during media processing, medias are cleared."""
        from vtscore.datasets.loader import load_dataset_from_pickle

        # Create a pickle with several medias
        medias_data = {}
        for i in range(1, 6):
            medias_data[i] = {
                "media_type": "audio",
                "embedding": [0.0] * 10,
                "media_bytes": b"\x00" * 100,
                "filename": f"clip_{i}.wav",
                "md5": f"md5_{i}",
            }
        pkl = tmp_path / "medium.pkl"
        pkl.write_bytes(pickle.dumps({"medias": medias_data}))

        target: dict = {}

        # Make np.array raise MemoryError on the 3rd call
        call_count = 0
        original_np_array = np.array

        def oom_on_third_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                raise MemoryError("simulated OOM")
            return original_np_array(*args, **kwargs)

        with mock.patch("vtscore.datasets.loader_pickle.np.array", side_effect=oom_on_third_call):
            with pytest.raises(MemoryError, match="Out of memory after loading"):
                load_dataset_from_pickle(pkl, target)

        # Clips should have been cleared on error
        assert len(target) == 0

    def test_pickle_data_freed_after_successful_load(self, tmp_path):
        """After a successful pickle load, the raw data is released via gc."""
        from vtscore.datasets.loader import load_dataset_from_pickle

        medias_data = {
            1: {
                "media_type": "text",
                "embedding": [0.0] * 10,
                "media_string": "hello",
                "filename": "t.txt",
                "md5": "abc",
            }
        }
        pkl = tmp_path / "small.pkl"
        pkl.write_bytes(pickle.dumps({"medias": medias_data}))

        target: dict = {}
        with mock.patch("vtscore.datasets.loader_pickle.gc.collect") as mock_gc:
            load_dataset_from_pickle(pkl, target)
            # gc.collect() should be called after building medias
            assert mock_gc.call_count >= 1

        assert len(target) == 1


class TestFolderMemoryError:
    """load_dataset_from_folder should handle MemoryError gracefully."""

    def test_folder_oom_clears_clips_and_raises(self, tmp_path):
        from vtscore.datasets.loader import load_dataset_from_folder

        # Create some dummy files
        for i in range(3):
            (tmp_path / f"clip_{i}.wav").write_bytes(b"\x00" * 100)

        target: dict = {}
        progress_calls = []

        def mock_progress(status, msg="", current=0, total=0, error=None):
            progress_calls.append((status, msg))

        mock_mt = mock.MagicMock()
        mock_mt.file_extensions = ["*.wav"]
        mock_mt.type_id = "audio"

        mock_emb = mock.MagicMock()
        mock_emb.name = "clap"
        mock_emb.media_type_id = "audio"
        mock_emb._model = True

        call_count = 0

        def embed_then_oom(path):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise MemoryError("simulated OOM")
            return np.zeros(10)

        mock_emb.embed_media.side_effect = embed_then_oom
        # Route the bulk entrypoint through embed_media so the per-file OOM
        # simulator still fires under the loader's bulk dispatch.
        mock_emb.embed_media_bulk.side_effect = lambda medias: [mock_emb.embed_media(m) for m in medias]
        mock_mt.load_media_data.return_value = {"media_bytes": b"\x00", "duration": 1}

        with (
            mock.patch("vtscore.media.get_by_folder_name", return_value=mock_mt),
            mock.patch("vtscore.media.embedders_for_type", return_value=[mock_emb]),
        ):
            with pytest.raises(MemoryError, match="Out of memory after loading"):
                load_dataset_from_folder(
                    tmp_path,
                    "audio",
                    target,
                    on_progress=mock_progress,
                )

        assert len(target) == 0


class TestCombineMemoryError:
    """CombineDatasetsImporter.run should handle MemoryError gracefully."""

    def test_combine_oom_clears_and_raises(self, tmp_path):
        from vtscore.datasets.importers.combine_datasets import CombineDatasetsImporter

        # Create two small pickles
        for name in ("a.pkl", "b.pkl"):
            medias_data = {
                1: {
                    "media_type": "audio",
                    "embedding": [0.0] * 10,
                    "media_bytes": b"\x00" * 100,
                    "filename": f"{name}_clip.wav",
                    "md5": f"md5_{name}",
                }
            }
            (tmp_path / name).write_bytes(pickle.dumps({"medias": medias_data}))

        importer = CombineDatasetsImporter()
        target: dict = {}

        # Make the second _load_clips_from_pickle call OOM
        call_count = 0

        def oom_second_load(path, thin=False):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise MemoryError("simulated OOM")
            from vtscore.datasets.loader import load_dataset_from_pickle

            temp: dict = {}
            load_dataset_from_pickle(path, temp, thin=thin)
            return temp

        with mock.patch(
            "vtscore.datasets.importers.combine_datasets._load_clips_from_pickle",
            side_effect=oom_second_load,
        ):
            with pytest.raises(MemoryError, match="Out of memory while combining"):
                importer.run(
                    {"datasets": [str(tmp_path / "a.pkl"), str(tmp_path / "b.pkl")]},
                    target,
                )

        assert len(target) == 0


class TestBackgroundImportMemoryError:
    """The background import thread should report MemoryError to the user."""

    def test_importer_background_oom_reports_error(self, client):
        """When an importer OOMs, the progress endpoint shows a user-friendly error."""
        from vtscore.datasets.load_pipeline import _run_importer_in_background

        mock_importer = mock.MagicMock()
        mock_importer.supports_chunked = False
        mock_importer.run.side_effect = MemoryError("simulated")

        # Reset progress
        update_progress("idle", "")

        # Patch threading.Thread to run synchronously so we don't race
        with mock.patch("vtscore.datasets.load_pipeline.threading") as mock_threading:
            captured_target = {}

            def fake_thread(target, daemon=True):
                captured_target["fn"] = target
                thread = mock.MagicMock()
                thread.start = lambda: target()
                return thread

            mock_threading.Thread.side_effect = fake_thread
            _run_importer_in_background(mock_importer, {})

        progress = get_progress()
        assert progress["error"] is not None
        assert "Out of memory" in progress["error"]
        assert progress["status"] == "idle"

        # Reinitialise medias for subsequent tests
        app_module.init_medias()

    def test_demo_load_oom_reports_error(self, client):
        """When loading a demo dataset OOMs, the progress shows the error."""
        update_progress("idle", "")

        def sync_thread(target, daemon=True):
            thread = mock.MagicMock()
            thread.start = lambda: target()
            return thread

        with (
            mock.patch(
                "vtscore.datasets.importers.demo.load_demo_dataset",
                side_effect=MemoryError("simulated"),
            ),
            mock.patch(
                "vtscore.datasets.load_pipeline.threading.Thread",
                side_effect=sync_thread,
            ),
        ):
            resp = client.post(
                "/api/dataset/load-demo",
                json={"name": list(DEMO_DATASETS.keys())[0]},
            )
            assert resp.status_code == 200

            progress = get_progress()
            assert progress["error"] is not None
            assert "Out of memory" in progress["error"]

        # Reinitialise medias for subsequent tests
        app_module.init_medias()
