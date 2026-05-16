"""Tests for ``.npz`` support in the ``server_files`` and ``local_files`` importers.

The ``server_files`` importer accepts a ``.npz`` archive (in addition to
plain text) that supplies both the media-file paths and their
pre-computed embedding vectors.  Listed files are still symlinked into a
staging dir and run through the regular folder loader, but the loader
skips re-embedding for files whose names appear in the supplied
``content_vectors`` map.

The ``/api/dataset/import-local-folder`` endpoint additionally accepts
an optional ``vectors_file`` multipart form field (a ``.npz`` archive
keyed by uploaded-file name) so users can attach pre-computed vectors
to a browser-side multi-file upload.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from vtsearch.datasets.importers._npz_vectors import read_npz_filenames_and_vectors
from vtsearch.datasets.importers.server_files import (
    ServerFilesDatasetImporter,
    _read_npz_paths_file,
    _read_paths_and_vectors,
    _read_paths_file,
)


# ---------------------------------------------------------------------------
# Low-level NPZ reader helper
# ---------------------------------------------------------------------------


class TestReadNpzFilenamesAndVectors:
    def test_reads_standard_layout(self, tmp_path):
        npz = tmp_path / "vecs.npz"
        filenames = np.array(["a.wav", "b.wav", "c.wav"])
        vectors = np.arange(12, dtype=np.float32).reshape(3, 4)
        np.savez(npz, filenames=filenames, vectors=vectors)

        mapping = read_npz_filenames_and_vectors(npz)
        assert list(mapping.keys()) == ["a.wav", "b.wav", "c.wav"]
        np.testing.assert_array_equal(mapping["a.wav"], vectors[0])
        np.testing.assert_array_equal(mapping["b.wav"], vectors[1])
        np.testing.assert_array_equal(mapping["c.wav"], vectors[2])

    def test_reads_per_key_layout_fallback(self, tmp_path):
        npz = tmp_path / "vecs.npz"
        v1 = np.array([1, 2, 3], dtype=np.float32)
        v2 = np.array([4, 5, 6], dtype=np.float32)
        np.savez(npz, **{"x.wav": v1, "y.wav": v2})

        mapping = read_npz_filenames_and_vectors(npz)
        assert set(mapping) == {"x.wav", "y.wav"}
        np.testing.assert_array_equal(mapping["x.wav"], v1)
        np.testing.assert_array_equal(mapping["y.wav"], v2)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_npz_filenames_and_vectors(tmp_path / "missing.npz")

    def test_mismatched_lengths_raises(self, tmp_path):
        npz = tmp_path / "bad.npz"
        np.savez(
            npz,
            filenames=np.array(["a.wav", "b.wav"]),
            vectors=np.zeros((3, 4), dtype=np.float32),
        )
        with pytest.raises(ValueError, match="mismatched"):
            read_npz_filenames_and_vectors(npz)

    def test_2d_filenames_array_rejected(self, tmp_path):
        npz = tmp_path / "bad.npz"
        np.savez(
            npz,
            filenames=np.array([["a", "b"], ["c", "d"]]),
            vectors=np.zeros((2, 4), dtype=np.float32),
        )
        with pytest.raises(ValueError, match="1-D"):
            read_npz_filenames_and_vectors(npz)

    def test_blank_names_are_skipped(self, tmp_path):
        npz = tmp_path / "vecs.npz"
        filenames = np.array(["a.wav", "", "  ", "b.wav"])
        vectors = np.arange(16, dtype=np.float32).reshape(4, 4)
        np.savez(npz, filenames=filenames, vectors=vectors)
        mapping = read_npz_filenames_and_vectors(npz)
        assert set(mapping) == {"a.wav", "b.wav"}


# ---------------------------------------------------------------------------
# server_files paths-file plumbing
# ---------------------------------------------------------------------------


class TestServerFilesFieldsAdvertiseNpz:
    """The frontend file-browser uses the ``accept`` attribute to filter
    the picker.  Make sure both txt/list and npz are advertised."""

    def test_paths_file_accept_includes_npz(self):
        imp = ServerFilesDatasetImporter()
        paths_field = next(f for f in imp.fields if f.key == "paths_file")
        accept_exts = {e.strip() for e in (paths_field.accept or "").split(",")}
        assert ".txt" in accept_exts
        assert ".list" in accept_exts
        assert ".npz" in accept_exts

    def test_paths_file_description_mentions_npz(self):
        imp = ServerFilesDatasetImporter()
        paths_field = next(f for f in imp.fields if f.key == "paths_file")
        assert ".npz" in (paths_field.description or "").lower()


class TestServerFilesNpzPathsFile:
    def test_read_paths_file_dispatches_on_suffix(self, tmp_path):
        # txt path still works
        txt = tmp_path / "list.txt"
        txt.write_text("/a.wav\n/b.wav\n")
        assert _read_paths_file(txt) == [Path("/a.wav"), Path("/b.wav")]

        # npz with absolute paths
        npz = tmp_path / "list.npz"
        np.savez(
            npz,
            filenames=np.array(["/x.wav", "/y.wav"]),
            vectors=np.zeros((2, 4), dtype=np.float32),
        )
        assert _read_paths_file(npz) == [Path("/x.wav"), Path("/y.wav")]

    def test_read_npz_paths_file_returns_paths_and_vectors(self, tmp_path):
        media_a = tmp_path / "a.wav"
        media_b = tmp_path / "b.wav"
        media_a.write_bytes(b"A")
        media_b.write_bytes(b"B")
        npz = tmp_path / "list.npz"
        vectors = np.arange(8, dtype=np.float32).reshape(2, 4)
        np.savez(npz, filenames=np.array([str(media_a), str(media_b)]), vectors=vectors)

        paths, path_to_vector = _read_npz_paths_file(npz)
        assert paths == [Path(str(media_a)), Path(str(media_b))]
        assert set(path_to_vector) == {str(media_a), str(media_b)}
        np.testing.assert_array_equal(path_to_vector[str(media_a)], vectors[0])

    def test_relative_npz_paths_resolved_against_npz_dir(self, tmp_path):
        media = tmp_path / "data" / "x.wav"
        media.parent.mkdir(parents=True)
        media.write_bytes(b"x")
        npz = tmp_path / "list.npz"
        np.savez(
            npz,
            filenames=np.array(["data/x.wav"]),
            vectors=np.zeros((1, 4), dtype=np.float32),
        )

        paths, vecs = _read_npz_paths_file(npz)
        assert paths == [media.resolve()]
        # Vector is keyed by the resolved absolute path so the staging
        # rekey step can look it up.
        assert str(media.resolve()) in vecs

    def test_read_paths_and_vectors_txt_returns_empty_vectors(self, tmp_path):
        txt = tmp_path / "list.txt"
        txt.write_text("/a.wav\n")
        paths, vecs = _read_paths_and_vectors(txt)
        assert paths == [Path("/a.wav")]
        assert vecs == {}


# ---------------------------------------------------------------------------
# server_files end-to-end: NPZ vectors skip re-embedding
# ---------------------------------------------------------------------------


class TestServerFilesNpzRunsEndToEnd:
    def test_npz_vectors_are_used_instead_of_embedder(self, tmp_path):
        """When the npz supplies a vector for a file, the resulting media's
        embedding is exactly that vector (the embedder is bypassed)."""
        from helpers import make_raw_wav_bytes

        src_a = tmp_path / "src_a.wav"
        src_b = tmp_path / "src_b.wav"
        src_a.write_bytes(make_raw_wav_bytes())
        src_b.write_bytes(make_raw_wav_bytes() + b"\x00\x00")

        # Distinctive vectors so we can verify they make it through
        # unmodified instead of being replaced by the embedder output.
        rng = np.random.default_rng(42)
        vec_a = rng.standard_normal(512).astype(np.float32) * 100.0
        vec_b = rng.standard_normal(512).astype(np.float32) * 100.0

        npz = tmp_path / "list.npz"
        np.savez(
            npz,
            filenames=np.array([str(src_a), str(src_b)]),
            vectors=np.stack([vec_a, vec_b]),
        )

        imp = ServerFilesDatasetImporter()
        medias: dict = {}
        imp.run({"paths_file": str(npz), "media_type": "audio"}, medias)

        assert len(medias) == 2
        # Index medias by their original source path so we can pair each
        # with its expected vector.
        by_source = {m["origin_name"]: m for m in medias.values()}
        np.testing.assert_array_equal(by_source[str(src_a)]["embedding"], vec_a)
        np.testing.assert_array_equal(by_source[str(src_b)]["embedding"], vec_b)
        # Origin still points at the npz so subsequent reloads work.
        for media in medias.values():
            assert media["origin"]["importer"] == "server_files"
            assert media["origin"]["params"]["paths_file"] == str(npz)

    def test_npz_per_key_layout_is_accepted(self, tmp_path):
        from helpers import make_raw_wav_bytes

        src = tmp_path / "only.wav"
        src.write_bytes(make_raw_wav_bytes())

        # Per-key layout: each filename is a top-level key in the npz.
        vec = np.full(256, 7.0, dtype=np.float32)
        npz = tmp_path / "list.npz"
        np.savez(npz, **{str(src): vec})

        imp = ServerFilesDatasetImporter()
        medias: dict = {}
        imp.run({"paths_file": str(npz), "media_type": "audio"}, medias)

        assert len(medias) == 1
        media = next(iter(medias.values()))
        np.testing.assert_array_equal(media["embedding"], vec)


# ---------------------------------------------------------------------------
# /api/dataset/import-local-folder: vectors_file npz alongside media files
# ---------------------------------------------------------------------------


class TestLocalUploadVectorsFile:
    """The /api/dataset/import-local-folder endpoint parses the optional
    ``vectors_file`` npz and hands the resulting ``{filename: vector}``
    map to the server_folder importer via its ``content_vectors``
    attribute.  The background task runner is stubbed so the test can
    invoke the importer's ``load_fn`` synchronously and inspect the
    importer's state at the moment it would have been called."""

    @staticmethod
    def _stub_run_and_observe(observed: dict):
        """Build a ``_run_origin_load_in_background`` stub that sniffs the
        server_folder importer's ``content_vectors`` at the moment the
        load function would call into the importer.  Works for both the
        chunked and non-chunked dispatch paths."""
        from vtsearch.datasets.importers import get_importer

        def _fake_run(load_fn, origin, **kwargs):
            importer = get_importer("server_folder")
            assert importer is not None

            def _sniff(field_values, medias=None, thin=False):
                observed["content_vectors"] = dict(importer.content_vectors or {})
                # Return an empty generator for the chunked path.
                return iter(())

            with (
                patch.object(importer, "run", side_effect=_sniff),
                patch.object(importer, "run_chunked", side_effect=_sniff),
            ):
                load_fn({})
            return "task-fake-npz"

        return _fake_run

    def test_upload_with_vectors_file_populates_importer_content_vectors(self, client, tmp_path, monkeypatch):
        from vtsearch.datasets.importers import get_importer

        monkeypatch.setattr(
            "vtsearch.routes.datasets.load.LOCAL_UPLOADS_DIR",
            tmp_path / "uploads",
        )

        vec = np.full(384, 3.5, dtype=np.float32)
        npz_bytes_io = io.BytesIO()
        np.savez(
            npz_bytes_io,
            filenames=np.array(["clip.wav"]),
            vectors=np.stack([vec]),
        )
        npz_bytes_io.seek(0)

        observed: dict = {}
        with patch(
            "vtsearch.routes.datasets.load._run_origin_load_in_background",
            side_effect=self._stub_run_and_observe(observed),
        ):
            resp = client.post(
                "/api/dataset/import-local-folder",
                data={
                    "media_type": "audio",
                    "files": (io.BytesIO(b"AAA"), "clip.wav"),
                    "vectors_file": (npz_bytes_io, "vectors.npz"),
                },
                content_type="multipart/form-data",
            )

        assert resp.status_code == 200, resp.get_data(as_text=True)
        # The npz was parsed and forwarded to the importer.
        assert "clip.wav" in observed["content_vectors"]
        np.testing.assert_array_equal(observed["content_vectors"]["clip.wav"], vec)

        # After the load returns, content_vectors is restored so the
        # next upload doesn't inherit stale vectors.
        assert get_importer("server_folder").content_vectors == {}

    def test_upload_without_vectors_file_leaves_importer_unchanged(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "vtsearch.routes.datasets.load.LOCAL_UPLOADS_DIR",
            tmp_path / "uploads",
        )

        observed: dict = {}
        with patch(
            "vtsearch.routes.datasets.load._run_origin_load_in_background",
            side_effect=self._stub_run_and_observe(observed),
        ):
            resp = client.post(
                "/api/dataset/import-local-folder",
                data={
                    "media_type": "audio",
                    "files": (io.BytesIO(b"AAA"), "clip.wav"),
                },
                content_type="multipart/form-data",
            )

        assert resp.status_code == 200, resp.get_data(as_text=True)
        # With no npz attached, the importer's content_vectors stays
        # empty for the run.
        assert observed["content_vectors"] == {}

    def test_upload_rejects_invalid_vectors_file(self, client, tmp_path, monkeypatch):
        """A bogus ``vectors_file`` is rejected up front with a 400."""
        monkeypatch.setattr(
            "vtsearch.routes.datasets.load.LOCAL_UPLOADS_DIR",
            tmp_path / "uploads",
        )
        bogus = io.BytesIO(b"not really a npz archive")
        resp = client.post(
            "/api/dataset/import-local-folder",
            data={
                "media_type": "audio",
                "files": (io.BytesIO(b"RIFF...."), "clip.wav"),
                "vectors_file": (bogus, "broken.npz"),
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        body = resp.get_json()
        # flask-smorest error envelope: ``message`` (not ``error``).
        assert "vectors_file" in body["message"].lower()
