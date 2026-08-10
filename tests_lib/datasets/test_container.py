"""Tests for the ZIP-based dataset container format."""

from __future__ import annotations

import os
import pickle
import zipfile

import numpy as np
import pytest

from vtscore.datasets.container import (
    append_projection,
    read_container,
    read_meta,
    read_projection,
    write_container,
)
from vtscore.projection.pyramid import build_pyramid
from vtscore.projection.umap_projection import Projection


def _medias_pkl_bytes(n: int = 5) -> bytes:
    rng = np.random.default_rng(42)
    medias = {
        i: {
            "id": i,
            "media_type": "audio",
            "embedding": rng.standard_normal(8).tolist(),
            "duration": 1.0,
            "file_size": 100,
            "md5": f"md5_{i}",
            "filename": f"clip_{i}.wav",
        }
        for i in range(n)
    }
    return pickle.dumps({"medias": medias})


def _meta() -> dict:
    return {
        "format_version": 1,
        "embedder": "test_embedder",
        "clipper": "test_clipper",
        "media_type": "audio",
        "name": "Test Dataset",
        "created_at": 1700000000.0,
        "expires_at": None,
    }


class TestContainerFormat:
    def test_write_creates_zip(self, tmp_path):
        path = tmp_path / "dataset.pkl"
        write_container(path, _medias_pkl_bytes(), _meta())
        assert path.exists()
        with zipfile.ZipFile(str(path), "r") as zf:
            assert "medias.pkl" in zf.namelist()
            assert "meta.json" in zf.namelist()

    def test_read_container_round_trip(self, tmp_path):
        path = tmp_path / "dataset.pkl"
        write_container(path, _medias_pkl_bytes(), _meta())
        data, meta = read_container(path)
        assert "medias" in data
        assert len(data["medias"]) == 5
        assert meta["embedder"] == "test_embedder"
        assert meta["clipper"] == "test_clipper"
        assert meta["name"] == "Test Dataset"

    def test_read_reports_byte_progress(self, tmp_path):
        """``on_progress`` tracks the read against the entry's byte size.

        The counter wraps the stream the *unpickler* pulls from, so it spans
        deserialisation too — not just the transfer.  That is the whole point:
        on a text-shaped container the raw transfer is a small minority of the
        phase, so counting transfer alone would leave most of it dark.
        """
        path = tmp_path / "dataset.pkl"
        write_container(path, _medias_pkl_bytes(200), _meta())
        with zipfile.ZipFile(str(path), "r") as zf:
            entry_size = zf.getinfo("medias.pkl").file_size

        seen: list[tuple[int, int]] = []
        data, _meta_out = read_container(path, on_progress=lambda c, t: seen.append((c, t)))

        assert len(data["medias"]) == 200, "streaming must not disturb the payload"
        assert seen, "a read with a callback must report progress"
        assert all(total == entry_size for _, total in seen)
        assert [c for c, _ in seen] == sorted(c for c, _ in seen), "counter must not retreat"
        assert all(current <= entry_size for current, _ in seen)

    def test_read_publishes_denominator_before_streaming(self, tmp_path):
        """The first tick carries the total, even when the read fits in one gulp.

        Callers scale their whole phase against this denominator, so a
        container small enough to finish inside a single tick must still
        announce it rather than leaving the caller with no scale.
        """
        path = tmp_path / "dataset.pkl"
        write_container(path, _medias_pkl_bytes(1), _meta())

        seen: list[tuple[int, int]] = []
        read_container(path, on_progress=lambda c, t: seen.append((c, t)))

        assert seen[0][0] == 0
        assert seen[0][1] > 0

    def test_read_without_progress_still_round_trips(self, tmp_path):
        """The no-callback path stays the plain streamed read."""
        path = tmp_path / "dataset.pkl"
        write_container(path, _medias_pkl_bytes(20), _meta())
        data, meta = read_container(path)
        assert len(data["medias"]) == 20
        assert meta["embedder"] == "test_embedder"

    def test_streamed_read_still_rejects_forbidden_classes(self, tmp_path):
        """Streaming must not weaken the restricted unpickler's allowlist."""
        import pickle as _pickle

        path = tmp_path / "evil.pkl"
        payload = b"c__builtin__\neval\n(S'1+1'\ntR."
        with zipfile.ZipFile(str(path), "w") as zf:
            zf.writestr("medias.pkl", payload)
            zf.writestr("meta.json", "{}")

        with pytest.raises(_pickle.UnpicklingError):
            read_container(path, on_progress=lambda c, t: None)

    def test_payload_stored_uncompressed(self, tmp_path):
        """medias.pkl must be stored (ZIP_STORED), not DEFLATE-compressed.

        The payload is float32 embeddings + already-compressed media_bytes,
        both incompressible: DEFLATE burned seconds scanning every byte for
        zero gain, the bulk of the post-coverage "Saving to registry…"
        stall.  Pin STORED so the latency fix can't silently regress.
        """
        path = tmp_path / "dataset.pkl"
        write_container(path, _medias_pkl_bytes(), _meta())
        with zipfile.ZipFile(str(path), "r") as zf:
            for info in zf.infolist():
                assert info.compress_type == zipfile.ZIP_STORED, (
                    f"{info.filename} is compressed ({info.compress_type}); expected ZIP_STORED"
                )

    def test_reads_legacy_deflate_container(self, tmp_path):
        """Pre-existing DEFLATE-compressed containers must still load.

        Switching writes to STORED must not orphan datasets written by an
        older build; zipfile decompresses any method on read.
        """
        path = tmp_path / "legacy.pkl"
        with zipfile.ZipFile(str(path), "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("medias.pkl", _medias_pkl_bytes())
            import json

            zf.writestr("meta.json", json.dumps(_meta()))
        data, meta = read_container(path)
        assert len(data["medias"]) == 5
        assert meta["embedder"] == "test_embedder"

    def test_read_meta_only(self, tmp_path):
        path = tmp_path / "dataset.pkl"
        write_container(path, _medias_pkl_bytes(), _meta())
        meta = read_meta(path)
        assert meta["embedder"] == "test_embedder"
        assert meta["format_version"] == 1

    def test_extra_pickle_keys(self, tmp_path):
        path = tmp_path / "dataset.pkl"
        write_container(
            path,
            _medias_pkl_bytes(),
            _meta(),
            extra_pickle_keys={"audio_dir": "/data/audio"},
        )
        data, meta = read_container(path)
        assert data["audio_dir"] == "/data/audio"

    def test_expires_at_in_meta(self, tmp_path):
        path = tmp_path / "dataset.pkl"
        meta = _meta()
        meta["expires_at"] = 1800000000.0
        write_container(path, _medias_pkl_bytes(), meta)
        loaded_meta = read_meta(path)
        assert loaded_meta["expires_at"] == 1800000000.0


class TestContainerProjection:
    def _make_projection(self, n: int = 20) -> tuple:
        rng = np.random.default_rng(42)
        coords = rng.standard_normal((n, 2)).astype(np.float32)
        proj = Projection("container-pid", list(range(n)), coords, "test")
        pyr = build_pyramid(proj, n_levels=2)
        return proj, pyr

    def test_append_and_read_projection(self, tmp_path):
        path = tmp_path / "dataset.pkl"
        write_container(path, _medias_pkl_bytes(20), _meta())

        proj, pyr = self._make_projection()
        append_projection(path, proj, pyr)

        loaded = read_projection(path)
        assert loaded is not None
        proj2, pyr2 = loaded
        assert proj2.projection_id == "container-pid"
        np.testing.assert_array_equal(proj2.coords, proj.coords)
        assert pyr2.point_count == pyr.point_count

    def test_read_projection_when_absent(self, tmp_path):
        path = tmp_path / "dataset.pkl"
        write_container(path, _medias_pkl_bytes(), _meta())
        assert read_projection(path) is None

    def test_append_overwrites_existing_projection(self, tmp_path):
        path = tmp_path / "dataset.pkl"
        write_container(path, _medias_pkl_bytes(20), _meta())

        proj1, pyr1 = self._make_projection()
        append_projection(path, proj1, pyr1)

        rng = np.random.default_rng(99)
        coords2 = rng.standard_normal((20, 2)).astype(np.float32)
        proj2 = Projection("second-pid", list(range(20)), coords2, "test")
        pyr2 = build_pyramid(proj2, n_levels=2)
        append_projection(path, proj2, pyr2)

        loaded = read_projection(path)
        assert loaded is not None
        assert loaded[0].projection_id == "second-pid"


def _forbid_replace_under_open_handle(monkeypatch, path) -> None:
    """Make ``os.replace`` on *path* fail while a write handle is open on it.

    That is exactly Windows' behaviour, and the reason the append helpers must
    close their probe before rewriting the container.  On POSIX the same
    overlap is silently tolerated, so without this shim the regression is
    invisible here.
    """
    open_writers: list = []
    real_zipfile = zipfile.ZipFile
    real_replace = os.replace

    class TrackingZipFile(real_zipfile):  # type: ignore[misc,valid-type]
        def __init__(self, file, mode="r", *args, **kwargs):
            super().__init__(file, mode, *args, **kwargs)
            self._tracked = mode in ("a", "w") and str(file) == str(path)
            if self._tracked:
                open_writers.append(self)

        def close(self):
            if getattr(self, "_tracked", False):
                self._tracked = False
                open_writers.remove(self)
            super().close()

    def guarded_replace(src, dst):
        if open_writers and str(dst) == str(path):
            raise PermissionError(f"cannot replace {dst}: a write handle is still open on it")
        return real_replace(src, dst)

    monkeypatch.setattr(zipfile, "ZipFile", TrackingZipFile)
    monkeypatch.setattr(os, "replace", guarded_replace)


class TestContainerAppendHandleSafety:
    """Replacing an existing entry must not rewrite the container under an open handle."""

    def test_append_projection_replaces_without_open_handle(self, tmp_path, monkeypatch):
        path = tmp_path / "dataset.pkl"
        write_container(path, _medias_pkl_bytes(20), _meta())

        rng = np.random.default_rng(7)
        coords = rng.standard_normal((20, 2)).astype(np.float32)
        proj = Projection("first-pid", list(range(20)), coords, "test")
        append_projection(path, proj, build_pyramid(proj, n_levels=2))

        _forbid_replace_under_open_handle(monkeypatch, path)

        proj2 = Projection("second-pid", list(range(20)), coords, "test")
        append_projection(path, proj2, build_pyramid(proj2, n_levels=2))

        loaded = read_projection(path)
        assert loaded is not None
        assert loaded[0].projection_id == "second-pid"

    def test_append_region_labels_replaces_without_open_handle(self, tmp_path, monkeypatch):
        from vtscore.datasets.container import append_region_labels, read_region_labels
        from vtscore.projection.labels import RegionLabel, make_label_set

        path = tmp_path / "dataset.pkl"
        write_container(path, _medias_pkl_bytes(), _meta())

        def labels(text: str):
            return make_label_set("pid", [RegionLabel(level=0.0, x=0.0, y=0.0, text=text)])

        append_region_labels(path, labels("old"), "sig-old")

        _forbid_replace_under_open_handle(monkeypatch, path)

        append_region_labels(path, labels("new"), "sig-new")

        result = read_region_labels(path)
        assert result is not None
        label_set, signature = result
        assert [lab.text for lab in label_set.labels] == ["new"]
        assert signature == "sig-new"


class TestContainerMetaReaders:
    def test_read_embedder_from_container(self, tmp_path):
        from vtscore.datasets.loader_pickle import read_pkl_embedder

        path = tmp_path / "dataset.pkl"
        write_container(path, _medias_pkl_bytes(), _meta())
        assert read_pkl_embedder(path) == "test_embedder"

    def test_read_clipper_from_container(self, tmp_path):
        from vtscore.datasets.loader_pickle import read_pkl_clipper

        path = tmp_path / "dataset.pkl"
        write_container(path, _medias_pkl_bytes(), _meta())
        assert read_pkl_clipper(path) == "test_clipper"

    def test_meta_readers_return_none_for_unreadable_pkl(self, tmp_path):
        """A corrupt / legacy non-zip pkl must degrade to None, not raise.

        The demo catalog reads metadata from every cached file; one unreadable
        file used to raise ``BadZipFile`` and 500 the whole listing.
        """
        from vtscore.datasets.loader_pickle import read_pkl_clipper, read_pkl_embedder

        path = tmp_path / "legacy.pkl"
        # A raw (non-zip) pickle: old on-disk format, no zip magic.
        path.write_bytes(b"\x80\x05not-a-zip-container")
        assert read_pkl_embedder(path) is None
        assert read_pkl_clipper(path) is None
