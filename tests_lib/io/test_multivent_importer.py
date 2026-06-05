"""Tests for the multivent/microvent WebDataset importer.

Covers:
- Importer registration and metadata
- FileNotFoundError on missing data/embedding directories
- Full synthetic import with mock tar shards and mocked ffmpeg
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from helpers import make_raw_wav_bytes


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_audio_tar(path: Path, chunks: list[str], audio_bytes: bytes) -> None:
    """Write a tar file containing one <chunk>.m4a per chunk_id."""
    with tarfile.open(path, "w") as t:
        for chunk_id in chunks:
            info = tarfile.TarInfo(name=f"{chunk_id}.m4a")
            info.size = len(audio_bytes)
            t.addfile(info, io.BytesIO(audio_bytes))


def _make_embedding_tar(
    path: Path, chunks: list[str], n_windows: int, emb_dim: int, tag: str
) -> None:
    """Write a tar file containing one NPZ per chunk_id."""
    with tarfile.open(path, "w") as t:
        for chunk_id in chunks:
            rng = np.random.default_rng(hash(chunk_id) % (2**31))
            keyframe_ids = np.array([f"t{i * 10:06d}" for i in range(n_windows)])
            embeddings = rng.standard_normal((n_windows, emb_dim)).astype(np.float32)

            buf = io.BytesIO()
            np.savez(buf, keyframe_ids=keyframe_ids, embeddings=embeddings)
            buf_bytes = buf.getvalue()

            info = tarfile.TarInfo(name=f"{chunk_id}.{tag}.npz")
            info.size = len(buf_bytes)
            t.addfile(info, io.BytesIO(buf_bytes))


def _make_audio_catalog(path: Path, chunks: list[str], shard_idx: int = 0) -> None:
    rows = [
        {
            "chunk_id": c,
            "video_id": c.rsplit("_", 1)[0],
            "chunk_index": 0,
            "chunk_count": 1,
            "shard_index": shard_idx,
            "has_audio": True,
            "acodec": "aac",
            "asample_rate_hz": 44100,
            "achannels": 2,
            "duration_sec": 30.0,
            "size_bytes": 100,
        }
        for c in chunks
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _fake_ffmpeg_extract(
    m4a_bytes: bytes, t_start: float, duration: float, out_path: Path
) -> bool:
    """Write the first 100 bytes as a .wav (readable by soundfile) to out_path."""
    out_path.write_bytes(make_raw_wav_bytes())
    return True


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


class TestMultiventImporterRegistration:
    def test_registered_in_importer_registry(self):
        from vtscore.datasets.importers import get_importer, list_importers

        names = [i.name for i in list_importers()]
        assert "multivent" in names
        imp = get_importer("multivent")
        assert imp.name == "multivent"
        assert "multivent" in imp.display_name.lower() or "microvent" in imp.display_name.lower()
        assert imp.fields  # has at least one form field


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------


class TestMultiventImporterErrors:
    def test_missing_audio_dir_raises(self, tmp_path):
        from vtscore.datasets.importers.multivent import IMPORTER

        data_dir = tmp_path / "dataset"
        data_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="audio"):
            IMPORTER.run({"data_dir": str(data_dir)}, {})

    def test_missing_embedding_dir_raises(self, tmp_path):
        from vtscore.datasets.importers.multivent import IMPORTER

        data_dir = tmp_path / "dataset"
        (data_dir / "audio").mkdir(parents=True)
        _make_audio_catalog(data_dir / "audio" / "catalog.csv", ["chunk_0000"])
        # no embeddings/ dir
        with pytest.raises(FileNotFoundError, match="[Ee]mbedding"):
            IMPORTER.run({"data_dir": str(data_dir)}, {})


# ---------------------------------------------------------------------------
# synthetic import
# ---------------------------------------------------------------------------


class TestMultiventImporterSyntheticImport:
    @pytest.fixture()
    def dataset_dir(self, tmp_path):
        """Build a minimal synthetic dataset with 2 chunks × 2 windows × 512-dim."""
        data_dir = tmp_path / "microvent"
        cache_dir = tmp_path / "clips"

        chunks = ["test_vid_0000", "test_vid_0001"]
        audio_bytes = make_raw_wav_bytes()
        tag = "audemb_largerclapgeneral"
        n_windows = 2
        emb_dim = 512

        # audio shard + catalog
        audio_dir = data_dir / "audio"
        audio_dir.mkdir(parents=True)
        _make_audio_tar(audio_dir / "shard_000000.tar", chunks, audio_bytes)
        _make_audio_catalog(audio_dir / "catalog.csv", chunks, shard_idx=0)

        # embedding shard (no catalog needed)
        emb_dir = data_dir / "embeddings" / tag
        emb_dir.mkdir(parents=True)
        _make_embedding_tar(emb_dir / "shard_000000.tar", chunks, n_windows, emb_dim, tag)

        return data_dir, cache_dir, chunks, n_windows, emb_dim

    def test_import_produces_correct_number_of_media(self, dataset_dir):
        from vtscore.datasets.importers.multivent import IMPORTER

        data_dir, cache_dir, chunks, n_windows, _ = dataset_dir

        medias = {}
        with patch(
            "vtscore.datasets.importers.multivent._ffmpeg_extract_clip",
            side_effect=_fake_ffmpeg_extract,
        ):
            IMPORTER.run(
                {"data_dir": str(data_dir), "cache_dir": str(cache_dir)},
                medias,
            )

        assert len(medias) == len(chunks) * n_windows

    def test_embeddings_are_injected(self, dataset_dir):
        from vtscore.datasets.importers.multivent import IMPORTER

        data_dir, cache_dir, chunks, n_windows, emb_dim = dataset_dir

        medias = {}
        with patch(
            "vtscore.datasets.importers.multivent._ffmpeg_extract_clip",
            side_effect=_fake_ffmpeg_extract,
        ):
            IMPORTER.run(
                {"data_dir": str(data_dir), "cache_dir": str(cache_dir)},
                medias,
            )

        for media in medias.values():
            assert media["embedding"] is not None, "All windows must have pre-computed embeddings"
            assert media["embedding"].shape == (emb_dim,)
            assert media["embedder"] == "clap_general"

    def test_custom_metadata_has_chunk_id_and_t_offset(self, dataset_dir):
        from vtscore.datasets.importers.multivent import IMPORTER

        data_dir, cache_dir, chunks, n_windows, _ = dataset_dir

        medias = {}
        with patch(
            "vtscore.datasets.importers.multivent._ffmpeg_extract_clip",
            side_effect=_fake_ffmpeg_extract,
        ):
            IMPORTER.run(
                {"data_dir": str(data_dir), "cache_dir": str(cache_dir)},
                medias,
            )

        for media in medias.values():
            cm = media.get("custom_metadata") or {}
            assert "chunk_id" in cm
            assert "t_offset" in cm
            assert "t_start_sec" in cm
            assert cm["chunk_id"] in chunks
            t_label = cm["t_offset"]
            assert t_label.startswith("t")
            assert cm["t_start_sec"] == int(t_label[1:])

    def test_clips_are_cached_on_disk(self, dataset_dir):
        from vtscore.datasets.importers.multivent import IMPORTER

        data_dir, cache_dir, chunks, n_windows, _ = dataset_dir

        medias = {}
        with patch(
            "vtscore.datasets.importers.multivent._ffmpeg_extract_clip",
            side_effect=_fake_ffmpeg_extract,
        ):
            IMPORTER.run(
                {"data_dir": str(data_dir), "cache_dir": str(cache_dir)},
                medias,
            )

        cached = list(cache_dir.glob("*.m4a"))
        assert len(cached) == len(chunks) * n_windows, "One clip file per window"

    def test_cached_clips_reused_on_second_import(self, dataset_dir):
        from vtscore.datasets.importers.multivent import IMPORTER

        data_dir, cache_dir, chunks, n_windows, _ = dataset_dir

        call_count = {"n": 0}

        def counting_ffmpeg(m4a_bytes, t_start, duration, out_path):
            call_count["n"] += 1
            return _fake_ffmpeg_extract(m4a_bytes, t_start, duration, out_path)

        with patch(
            "vtscore.datasets.importers.multivent._ffmpeg_extract_clip",
            side_effect=counting_ffmpeg,
        ):
            IMPORTER.run(
                {"data_dir": str(data_dir), "cache_dir": str(cache_dir)},
                {},
            )
            first_count = call_count["n"]
            IMPORTER.run(
                {"data_dir": str(data_dir), "cache_dir": str(cache_dir)},
                {},
            )
            second_count = call_count["n"] - first_count

        assert first_count == len(chunks) * n_windows
        assert second_count == 0, "No ffmpeg calls on second import (clips already cached)"

    def test_max_shards_limits_import(self, dataset_dir):
        from vtscore.datasets.importers.multivent import IMPORTER

        data_dir, cache_dir, chunks, n_windows, _ = dataset_dir

        medias = {}
        with patch(
            "vtscore.datasets.importers.multivent._ffmpeg_extract_clip",
            side_effect=_fake_ffmpeg_extract,
        ):
            # Only 1 shard exists, max_shards=0 should import nothing
            IMPORTER.run(
                {"data_dir": str(data_dir), "cache_dir": str(cache_dir), "max_shards": "0"},
                medias,
            )

        assert len(medias) == 0

    def test_has_audio_false_skips_chunk(self, tmp_path):
        from vtscore.datasets.importers.multivent import IMPORTER

        data_dir = tmp_path / "dataset"
        cache_dir = tmp_path / "clips"
        tag = "audemb_largerclapgeneral"
        chunks = ["chunk_0000", "chunk_0001"]

        audio_dir = data_dir / "audio"
        audio_dir.mkdir(parents=True)
        _make_audio_tar(audio_dir / "shard_000000.tar", chunks, make_raw_wav_bytes())

        # Mark chunk_0000 as has_audio=False
        rows = [
            {"chunk_id": "chunk_0000", "video_id": "chunk", "chunk_index": 0,
             "chunk_count": 2, "shard_index": 0, "has_audio": False, "acodec": "aac",
             "asample_rate_hz": 44100, "achannels": 2, "duration_sec": 30, "size_bytes": 100},
            {"chunk_id": "chunk_0001", "video_id": "chunk", "chunk_index": 1,
             "chunk_count": 2, "shard_index": 0, "has_audio": True, "acodec": "aac",
             "asample_rate_hz": 44100, "achannels": 2, "duration_sec": 30, "size_bytes": 100},
        ]
        pd.DataFrame(rows).to_csv(audio_dir / "catalog.csv", index=False)

        emb_dir = data_dir / "embeddings" / tag
        emb_dir.mkdir(parents=True)
        _make_embedding_tar(emb_dir / "shard_000000.tar", chunks, 2, 512, tag)

        medias = {}
        with patch(
            "vtscore.datasets.importers.multivent._ffmpeg_extract_clip",
            side_effect=_fake_ffmpeg_extract,
        ):
            IMPORTER.run({"data_dir": str(data_dir), "cache_dir": str(cache_dir)}, medias)

        # Only chunk_0001's 2 windows should appear (chunk_0000 has no audio)
        assert len(medias) == 2
