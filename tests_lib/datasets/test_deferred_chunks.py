"""Deferred chunks: a chunked import walked without being built (#5314).

``run_chunked_cli_deferred`` lets a resumed CLI run pass the chunks it already
wrote without reading their files.  These pin the library half: the deferred
stream is the built stream (the same chunks, in the same order, at the same
boundaries, PDF, converter and archive chunks included), no file is read until
a chunk is built, and a chunk's key says what it holds from ``stat`` alone.
The CLI half is in ``tests/cli/test_cli_resume.py``.
"""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests_lib.helpers import make_png_bytes, make_wav_file
from vtscore.datasets.chunks import DeferredChunk
from vtscore.datasets.loader import iter_folder_chunks_deferred, load_dataset_from_folder_chunked


def _wav_folder(folder: Path, n: int) -> Path:
    folder.mkdir(exist_ok=True)
    for i in range(n):
        make_wav_file(folder, f"clip_{i}.wav", frequency=440.0 + i * 10)
    return folder


def _names(chunks: list[dict[int, dict[str, Any]]]) -> list[list[str]]:
    return [[media.get("origin_name") or media.get("filename") for media in chunk.values()] for chunk in chunks]


@pytest.fixture
def built_files(monkeypatch) -> list[str]:
    """The relative path of every file the folder loader builds, in order."""
    import vtscore.datasets.loader_folder as loader_folder

    built: list[str] = []
    real = loader_folder._build_per_file_media

    def spy(**kwargs: Any) -> dict[str, Any]:
        built.append(kwargs["rel_path"])
        return real(**kwargs)

    monkeypatch.setattr(loader_folder, "_build_per_file_media", spy)
    return built


class TestDeferredChunk:
    def test_built_wraps_a_chunk_with_no_key(self):
        chunk = {1: {"id": 1}, 2: {"id": 2}}
        deferred = DeferredChunk.built(chunk)

        assert (deferred.count, deferred.key) == (2, None)
        assert deferred.build() is chunk


class TestFolderWalk:
    def test_yields_the_chunks_the_loader_builds(self, tmp_path):
        folder = _wav_folder(tmp_path / "clips", 5)

        deferred = list(iter_folder_chunks_deferred(folder, "audio", 2, thin=True))
        loaded = list(load_dataset_from_folder_chunked(folder, "audio", 2, thin=True))

        assert [d.count for d in deferred] == [len(c) for c in loaded] == [2, 2, 1]
        assert _names([d.build() for d in deferred]) == _names(loaded)

    def test_reads_no_file_until_a_chunk_is_built(self, tmp_path, built_files):
        folder = _wav_folder(tmp_path / "clips", 5)

        deferred = list(iter_folder_chunks_deferred(folder, "audio", 2, thin=True))
        keys = [d.key() for d in deferred if d.key is not None]
        assert built_files == []

        second = deferred[1].build()

        assert built_files == [row[0] for row in keys[1]] == [m["origin_name"] for m in second.values()]

    def test_the_key_is_each_files_path_size_and_mtime(self, tmp_path):
        folder = _wav_folder(tmp_path / "clips", 3)

        (deferred,) = iter_folder_chunks_deferred(folder, "audio", 10)
        assert deferred.key is not None

        rows = deferred.key()
        assert sorted(row[0] for row in rows) == ["clip_0.wav", "clip_1.wav", "clip_2.wav"]
        for rel_path, size, mtime_ns in rows:
            st = (folder / rel_path).stat()
            assert (size, mtime_ns) == (st.st_size, st.st_mtime_ns)

    def test_the_key_moves_when_a_file_is_rewritten_or_touched(self, tmp_path):
        folder = _wav_folder(tmp_path / "clips", 2)

        def key() -> Any:
            (deferred,) = iter_folder_chunks_deferred(folder, "audio", 10)
            assert deferred.key is not None
            return deferred.key()

        before = key()
        assert key() == before

        (folder / "clip_0.wav").write_bytes((folder / "clip_0.wav").read_bytes() + b"\x00")
        rewritten = key()
        assert rewritten != before

        st = (folder / "clip_1.wav").stat()
        os.utime(folder / "clip_1.wav", ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        assert key() != rewritten

    def test_without_mtimes_the_key_is_path_and_size(self, tmp_path):
        folder = _wav_folder(tmp_path / "clips", 2)

        (deferred,) = iter_folder_chunks_deferred(folder, "audio", 10, stamp_mtime=False)
        assert deferred.key is not None

        assert sorted(deferred.key()) == sorted(
            [name, (folder / name).stat().st_size] for name in ("clip_0.wav", "clip_1.wav")
        )

    def test_an_empty_folder_is_refused_after_the_walk(self, tmp_path):
        (tmp_path / "empty").mkdir()

        with pytest.raises(ValueError, match="No audio files found"):
            list(iter_folder_chunks_deferred(tmp_path / "empty", "audio", 2))


class TestImporterHook:
    def test_the_default_wraps_run_chunked_cli_built(self):
        from vtscore.datasets.importers.base import DatasetImporter, PluginField

        class DummyImporter(DatasetImporter):
            name = "dummy"
            display_name = "Dummy"
            description = "Test"
            icon = ""
            fields: list[PluginField] = []

            def run(self, field_values, medias, thin=False):
                medias[1] = {"id": 1, "media_type": "audio", "embeddings": {"clap": np.zeros(4)}}
                medias[2] = {"id": 2, "media_type": "audio", "embeddings": {"clap": np.ones(4)}}

        (deferred,) = DummyImporter().run_chunked_cli_deferred({}, chunk_size=1)

        assert (deferred.count, deferred.key) == (2, None)
        assert sorted(deferred.build()) == [1, 2]


def _write_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def _image_folder(folder: Path, n: int) -> Path:
    folder.mkdir(exist_ok=True)
    for i in range(n):
        (folder / f"img_{i}.png").write_bytes(make_png_bytes(color=(10 * i, 64, 200)))
    return folder


@pytest.fixture
def archive_cache(tmp_path, monkeypatch) -> Path:
    from vtscore.datasets import archive as archive_mod

    data_dir = tmp_path / "data"
    monkeypatch.setattr(archive_mod, "DATA_DIR", data_dir)
    return data_dir


class TestServerFolderDeferred:
    """``server_folder``'s deferred chunks are its built ones, position for position."""

    def _both(self, field_values: dict[str, Any], chunk_size: int) -> tuple[list[Any], list[Any]]:
        from vtscore.datasets.importers.server_folder import IMPORTER

        built = list(IMPORTER.run_chunked_cli(dict(field_values), chunk_size, thin=True))
        deferred = list(IMPORTER.run_chunked_cli_deferred(dict(field_values), chunk_size, thin=True))
        return built, deferred

    def test_pdf_and_converter_chunks_keep_their_places(self, tmp_path, monkeypatch):
        import fitz

        folder = _image_folder(tmp_path / "imgs", 3)
        doc = fitz.open()
        for _ in range(2):
            doc.new_page(width=100, height=100)
        doc.save(str(folder / "doc.pdf"))
        doc.close()

        def fake_converters(folder, output_type, specs, medias, **kwargs):
            if specs:
                medias[1] = {"id": 1, "media_type": "image", "filename": "frame.png", "origin_name": "frame.png"}

        monkeypatch.setattr("vtscore.datasets.importers.server_folder._run_converter_specs", fake_converters)
        field_values = {
            "path": str(folder),
            "media_type": "image",
            "source_specs": [
                {"source_type": "image", "converter": None, "params": {}},
                {"source_type": "video", "converter": "video2image", "params": {}},
            ],
        }

        built, deferred = self._both(field_values, 2)

        # Two folder chunks, the PDF's pages, the converter's row.
        assert [d.count for d in deferred] == [len(c) for c in built] == [2, 1, 2, 1]
        assert [d.key is not None for d in deferred] == [True, True, False, False]
        assert _names([d.build() for d in deferred]) == _names(built)

    def test_dug_archives_keep_their_places(self, tmp_path, archive_cache):
        folder = _wav_folder(tmp_path / "clips", 3)
        make_wav_file(tmp_path, "inner.wav")
        inner = (tmp_path / "inner.wav").read_bytes()
        _write_zip(folder / "more.zip", {"a/one.wav": inner, "a/two.wav": inner + b"\x00", "three.wav": inner})
        field_values = {"path": str(folder), "media_type": "audio", "dig_archives": True}

        built, deferred = self._both(field_values, 2)

        assert [d.count for d in deferred] == [len(c) for c in built] == [2, 1, 2, 1]
        assert all(d.key is not None for d in deferred)
        assert _names([d.build() for d in deferred]) == _names(built)

    def test_an_archive_key_survives_reextraction_but_not_a_new_archive(self, tmp_path, archive_cache):
        from vtscore.datasets.importers.server_folder import IMPORTER

        make_wav_file(tmp_path, "inner.wav")
        inner = (tmp_path / "inner.wav").read_bytes()
        archive = _write_zip(tmp_path / "clips.zip", {"one.wav": inner, "two.wav": inner + b"\x00"})
        field_values = {"path": str(archive), "media_type": "audio"}

        def keys() -> list[Any]:
            return [d.key() for d in IMPORTER.run_chunked_cli_deferred(dict(field_values), 10) if d.key is not None]

        first = keys()
        assert first[0]["archive"][0] == "clips.zip"
        assert sorted(first[0]["files"]) == [["one.wav", len(inner)], ["two.wav", len(inner) + 1]]

        # A re-extraction stamps the members with new mtimes; the key holds.
        shutil.rmtree(archive_cache)
        assert keys() == first

        _write_zip(archive, {"one.wav": inner, "two.wav": inner + b"\x00\x00"})
        assert keys() != first

    def test_a_missing_path_is_refused(self):
        from vtscore.datasets.importers.server_folder import IMPORTER

        with pytest.raises(FileNotFoundError):
            list(IMPORTER.run_chunked_cli_deferred({"path": "/nonexistent/path", "media_type": "audio"}, 10))
