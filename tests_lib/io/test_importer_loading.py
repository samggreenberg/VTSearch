"""Importer loading tests.

Tests for load_dataset_from_folder: content_vectors, content_md5s,
relative paths, archive extraction directory isolation, and the
resolve_file contract.

The loader does **not** call any embedder — items leave with
``embedding=None`` unless a pre-computed vector is supplied via
``content_vectors`` or ``custom_metadata_map``.  The framework
``embed_missing`` stage fills the rest in.
"""

from __future__ import annotations

import zipfile

import pytest

from helpers import make_raw_wav_bytes as _make_wav_bytes


class TestLoadDatasetContentVectors:
    """Verify that load_dataset_from_folder uses pre-computed content vectors."""

    def _write_wav(self, path):
        """Write a minimal WAV file to *path*."""
        path.write_bytes(_make_wav_bytes())

    def _make_fake_media_type(self, embed_return):
        """Return a mock media-type object and mock embedder for testing.

        ``embed_return`` is the value returned by ``embed_media()``.
        """
        import unittest.mock as mock

        mt = mock.MagicMock()
        mt.type_id = "audio"
        mt.file_extensions = ["*.wav"]
        mt.embed_media.return_value = embed_return
        mt.load_media_data.return_value = {"duration": 1.0}
        # Also set up a mock embedder since load_dataset_from_folder uses the embedder registry
        mt._mock_embedder = mock.MagicMock()
        mt._mock_embedder.name = "clap"
        mt._mock_embedder.media_type_id = "audio"
        mt._mock_embedder._model = True
        mt._mock_embedder.embed_media.return_value = embed_return
        # Route the bulk entrypoint through the per-file mock so tests that
        # configured embed_media keep working under the loader's bulk dispatch.
        mt._mock_embedder.embed_media_bulk.side_effect = lambda medias: [
            mt._mock_embedder.embed_media(m) for m in medias
        ]
        return mt

    def _patch_media_registry(self, mt):
        """Context manager that patches both get_by_folder_name and embedders_for_type."""
        import unittest.mock as mock
        from contextlib import ExitStack

        stack = ExitStack()
        stack.enter_context(mock.patch("vtscore.media.get_by_folder_name", return_value=mt))
        stack.enter_context(mock.patch("vtscore.media.embedders_for_type", return_value=[mt._mock_embedder]))
        return stack

    def test_uses_content_vector_when_provided(self, tmp_path):
        """A file whose name is in content_vectors should use that vector."""
        import numpy as np

        from vtscore.datasets.loader import load_dataset_from_folder

        wav = tmp_path / "a.wav"
        self._write_wav(wav)

        pre_vector = np.array([10.0, 20.0, 30.0])
        mt = self._make_fake_media_type(embed_return=np.zeros(3))

        medias: dict = {}

        def _noop(*a):
            pass

        with self._patch_media_registry(mt):
            load_dataset_from_folder(
                tmp_path, "audio", medias, content_vectors={"a.wav": pre_vector}, on_progress=_noop
            )

        assert len(medias) == 1
        np.testing.assert_array_equal(medias[1]["embedding"], pre_vector)
        mt._mock_embedder.embed_media.assert_not_called()

    def test_leaves_embedding_none_when_not_in_content_vectors(self, tmp_path):
        """The loader no longer embeds; non-override files leave with embedding=None."""
        from vtscore.datasets.loader import load_dataset_from_folder

        wav = tmp_path / "b.wav"
        self._write_wav(wav)

        mt = self._make_fake_media_type(embed_return=None)

        medias: dict = {}

        with self._patch_media_registry(mt):
            load_dataset_from_folder(tmp_path, "audio", medias, content_vectors={}, on_progress=lambda *a: None)

        assert len(medias) == 1
        assert medias[1]["embedding"] is None
        assert medias[1]["embedder"] == ""
        mt._mock_embedder.embed_media.assert_not_called()
        mt._mock_embedder.embed_media_bulk.assert_not_called()

    def test_mixed_content_vectors_and_embedding(self, tmp_path):
        """Only files in content_vectors get a pre-computed vector; others get None."""
        import numpy as np

        from vtscore.datasets.loader import load_dataset_from_folder

        wav_a = tmp_path / "a.wav"
        wav_b = tmp_path / "b.wav"
        self._write_wav(wav_a)
        self._write_wav(wav_b)

        pre_vector = np.array([10.0, 20.0])
        mt = self._make_fake_media_type(embed_return=None)

        medias: dict = {}

        with self._patch_media_registry(mt):
            load_dataset_from_folder(
                tmp_path, "audio", medias, content_vectors={"a.wav": pre_vector}, on_progress=lambda *a: None
            )

        assert len(medias) == 2
        embeddings = {c["filename"]: c["embedding"] for c in medias.values()}
        np.testing.assert_array_equal(embeddings["a.wav"], pre_vector)
        assert embeddings["b.wav"] is None
        mt._mock_embedder.embed_media_bulk.assert_not_called()

    def test_no_content_vectors_param_leaves_all_none(self, tmp_path):
        """When content_vectors is None (default), all files get embedding=None."""
        from vtscore.datasets.loader import load_dataset_from_folder

        wav = tmp_path / "c.wav"
        self._write_wav(wav)

        mt = self._make_fake_media_type(embed_return=None)

        medias: dict = {}

        with self._patch_media_registry(mt):
            load_dataset_from_folder(tmp_path, "audio", medias, on_progress=lambda *a: None)

        assert len(medias) == 1
        assert medias[1]["embedding"] is None
        assert medias[1]["embedder"] == ""
        mt._mock_embedder.embed_media.assert_not_called()
        mt._mock_embedder.embed_media_bulk.assert_not_called()

    def test_content_vector_file_has_empty_embedder_id(self, tmp_path):
        """Files whose vectors came from content_vectors should not be stamped
        with any embedder name — the external vector may be a different
        dimension.  Files without an override also get ``embedder=""`` (the
        framework embed stage stamps the live embedder later).
        """
        import numpy as np

        from vtscore.datasets.loader import load_dataset_from_folder

        wav_pre = tmp_path / "pre.wav"
        wav_model = tmp_path / "model.wav"
        self._write_wav(wav_pre)
        self._write_wav(wav_model)

        external_vector = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        mt = self._make_fake_media_type(embed_return=None)

        medias: dict = {}

        with self._patch_media_registry(mt):
            load_dataset_from_folder(
                tmp_path,
                "audio",
                medias,
                content_vectors={"pre.wav": external_vector},
                on_progress=lambda *a: None,
            )

        by_name = {m["filename"]: m for m in medias.values()}
        assert by_name["pre.wav"]["embedder"] == ""
        assert by_name["model.wav"]["embedder"] == ""
        assert by_name["model.wav"]["embedding"] is None

    def test_content_vector_file_carries_vector_through(self, tmp_path):
        """A file with a content vector is always included with its supplied vector."""
        import unittest.mock as mock

        import numpy as np

        from vtscore.datasets.loader import load_dataset_from_folder

        wav = tmp_path / "d.wav"
        self._write_wav(wav)

        pre_vector = np.array([7.0, 8.0])
        mt = self._make_fake_media_type(embed_return=None)

        medias: dict = {}

        with mock.patch("vtscore.media.get_by_folder_name", return_value=mt):
            load_dataset_from_folder(
                tmp_path,
                "audio",
                medias,
                content_vectors={"d.wav": pre_vector},
                on_progress=lambda *a: None,
            )

        assert len(medias) == 1
        np.testing.assert_array_equal(medias[1]["embedding"], pre_vector)


# ---------------------------------------------------------------------------
# load_dataset_from_folder – content_md5s support
# ---------------------------------------------------------------------------


class TestLoadDatasetContentMD5s:
    """Verify that load_dataset_from_folder uses pre-computed MD5 hashes."""

    def _write_wav(self, path):
        """Write a minimal WAV file to *path*."""
        path.write_bytes(_make_wav_bytes())

    def _make_fake_media_type(self, embed_return):
        import unittest.mock as mock

        mt = mock.MagicMock()
        mt.type_id = "audio"
        mt.file_extensions = ["*.wav"]
        mt.embed_media.return_value = embed_return
        mt.load_media_data.return_value = {"duration": 1.0}
        return mt

    def test_uses_content_md5_when_provided(self, tmp_path):
        """A file whose name is in content_md5s should use that hash."""
        import unittest.mock as mock

        import numpy as np

        from vtscore.datasets.loader import load_dataset_from_folder

        wav = tmp_path / "a.wav"
        self._write_wav(wav)

        pre_md5 = "0" * 32
        mt = self._make_fake_media_type(embed_return=np.zeros(3))

        medias: dict = {}

        def _noop(*a):
            pass

        with mock.patch("vtscore.media.get_by_folder_name", return_value=mt):
            load_dataset_from_folder(tmp_path, "audio", medias, content_md5s={"a.wav": pre_md5}, on_progress=_noop)

        assert len(medias) == 1
        assert medias[1]["md5"] == pre_md5

    def test_computes_md5_when_not_in_content_md5s(self, tmp_path):
        """A file NOT in content_md5s falls back to computing the hash."""
        import hashlib
        import unittest.mock as mock

        import numpy as np

        from vtscore.datasets.loader import load_dataset_from_folder

        wav = tmp_path / "b.wav"
        self._write_wav(wav)
        expected_md5 = hashlib.md5(wav.read_bytes()).hexdigest()

        mt = self._make_fake_media_type(embed_return=np.zeros(3))

        medias: dict = {}

        def _noop(*a):
            pass

        with mock.patch("vtscore.media.get_by_folder_name", return_value=mt):
            load_dataset_from_folder(tmp_path, "audio", medias, content_md5s={}, on_progress=_noop)

        assert len(medias) == 1
        assert medias[1]["md5"] == expected_md5

    def test_mixed_content_md5s_and_computed(self, tmp_path):
        """Only files in content_md5s skip MD5 computation; others are hashed."""
        import hashlib
        import unittest.mock as mock

        import numpy as np

        from vtscore.datasets.loader import load_dataset_from_folder

        wav_a = tmp_path / "a.wav"
        wav_b = tmp_path / "b.wav"
        self._write_wav(wav_a)
        self._write_wav(wav_b)

        pre_md5 = "f" * 32
        computed_md5 = hashlib.md5(wav_b.read_bytes()).hexdigest()
        mt = self._make_fake_media_type(embed_return=np.zeros(3))

        medias: dict = {}

        def _noop(*a):
            pass

        with mock.patch("vtscore.media.get_by_folder_name", return_value=mt):
            load_dataset_from_folder(tmp_path, "audio", medias, content_md5s={"a.wav": pre_md5}, on_progress=_noop)

        assert len(medias) == 2
        md5s = {c["filename"]: c["md5"] for c in medias.values()}
        assert md5s["a.wav"] == pre_md5
        assert md5s["b.wav"] == computed_md5

    def test_no_content_md5s_param_computes_all(self, tmp_path):
        """When content_md5s is None (default), all files are hashed."""
        import hashlib
        import unittest.mock as mock

        import numpy as np

        from vtscore.datasets.loader import load_dataset_from_folder

        wav = tmp_path / "c.wav"
        self._write_wav(wav)
        expected_md5 = hashlib.md5(wav.read_bytes()).hexdigest()

        mt = self._make_fake_media_type(embed_return=np.zeros(3))

        medias: dict = {}

        def _noop(*a):
            pass

        with mock.patch("vtscore.media.get_by_folder_name", return_value=mt):
            load_dataset_from_folder(tmp_path, "audio", medias, on_progress=_noop)

        assert len(medias) == 1
        assert medias[1]["md5"] == expected_md5

    def test_content_md5s_in_thin_mode(self, tmp_path):
        """content_md5s should work in thin mode too."""
        import unittest.mock as mock

        import numpy as np

        from vtscore.datasets.loader import load_dataset_from_folder

        wav = tmp_path / "a.wav"
        self._write_wav(wav)

        pre_md5 = "1" * 32
        mt = self._make_fake_media_type(embed_return=np.zeros(3))

        medias: dict = {}

        def _noop(*a):
            pass

        with mock.patch("vtscore.media.get_by_folder_name", return_value=mt):
            load_dataset_from_folder(
                tmp_path, "audio", medias, content_md5s={"a.wav": pre_md5}, on_progress=_noop, thin=True
            )

        assert len(medias) == 1
        assert medias[1]["md5"] == pre_md5


# ---------------------------------------------------------------------------
# load_dataset_from_folder – relative path preservation
# ---------------------------------------------------------------------------


class TestLoadDatasetRelativePaths:
    """Verify that load_dataset_from_folder preserves relative paths."""

    def _write_wav(self, path):
        path.write_bytes(_make_wav_bytes())

    def _make_fake_media_type(self, embed_return):
        import unittest.mock as mock

        mt = mock.MagicMock()
        mt.type_id = "audio"
        mt.file_extensions = ["*.wav"]
        mt.embed_media.return_value = embed_return
        mt.load_media_data.return_value = {"duration": 1.0}
        return mt

    def test_flat_files_use_basename(self, tmp_path):
        """Files directly in the root folder should have basename-only filenames."""
        import unittest.mock as mock

        import numpy as np

        from vtscore.datasets.loader import load_dataset_from_folder

        self._write_wav(tmp_path / "a.wav")
        mt = self._make_fake_media_type(embed_return=np.zeros(3))
        medias: dict = {}

        with mock.patch("vtscore.media.get_by_folder_name", return_value=mt):
            load_dataset_from_folder(tmp_path, "audio", medias, on_progress=lambda *a: None)

        assert medias[1]["filename"] == "a.wav"
        assert medias[1]["origin_name"] == "a.wav"

    def test_subdir_files_use_relative_path(self, tmp_path):
        """Files in subdirectories should keep the relative path."""
        import unittest.mock as mock

        import numpy as np

        from vtscore.datasets.loader import load_dataset_from_folder

        sub = tmp_path / "cat_a"
        sub.mkdir()
        self._write_wav(sub / "clip.wav")
        mt = self._make_fake_media_type(embed_return=np.zeros(3))
        medias: dict = {}

        with mock.patch("vtscore.media.get_by_folder_name", return_value=mt):
            load_dataset_from_folder(tmp_path, "audio", medias, on_progress=lambda *a: None)

        assert medias[1]["filename"] == "cat_a/clip.wav"
        assert medias[1]["origin_name"] == "cat_a/clip.wav"

    def test_same_name_different_dirs_are_distinct(self, tmp_path):
        """Identically named files in different subdirs get distinct filenames."""
        import unittest.mock as mock

        import numpy as np

        from vtscore.datasets.loader import load_dataset_from_folder

        (tmp_path / "dir_a").mkdir()
        (tmp_path / "dir_b").mkdir()
        self._write_wav(tmp_path / "dir_a" / "clip.wav")
        self._write_wav(tmp_path / "dir_b" / "clip.wav")
        mt = self._make_fake_media_type(embed_return=np.zeros(3))
        medias: dict = {}

        with mock.patch("vtscore.media.get_by_folder_name", return_value=mt):
            load_dataset_from_folder(tmp_path, "audio", medias, on_progress=lambda *a: None)

        filenames = {m["filename"] for m in medias.values()}
        assert "dir_a/clip.wav" in filenames
        assert "dir_b/clip.wav" in filenames
        assert len(filenames) == 2

    def test_content_vectors_fallback_to_basename(self, tmp_path):
        """content_vectors keyed by basename should still work for flat files."""
        import unittest.mock as mock

        import numpy as np

        from vtscore.datasets.loader import load_dataset_from_folder

        self._write_wav(tmp_path / "x.wav")
        pre = np.array([99.0, 99.0, 99.0])
        mt = self._make_fake_media_type(embed_return=np.zeros(3))
        medias: dict = {}

        with mock.patch("vtscore.media.get_by_folder_name", return_value=mt):
            load_dataset_from_folder(
                tmp_path, "audio", medias, content_vectors={"x.wav": pre}, on_progress=lambda *a: None
            )

        np.testing.assert_array_equal(medias[1]["embedding"], pre)

    def test_chunked_preserves_relative_paths(self, tmp_path):
        """Chunked loader should also preserve relative paths."""
        import unittest.mock as mock

        import numpy as np

        from vtscore.datasets.loader import load_dataset_from_folder_chunked

        sub = tmp_path / "sub"
        sub.mkdir()
        self._write_wav(sub / "chunk.wav")
        mt = self._make_fake_media_type(embed_return=np.zeros(3))

        with mock.patch("vtscore.media.get_by_folder_name", return_value=mt):
            chunks = list(load_dataset_from_folder_chunked(tmp_path, "audio", 10, on_progress=lambda *a: None))

        media = chunks[0][1]
        assert media["filename"] == "sub/chunk.wav"
        assert media["origin_name"] == "sub/chunk.wav"


# ---------------------------------------------------------------------------
# HTTP Archive importer – unique extract directories
# ---------------------------------------------------------------------------


class TestHttpArchiveExtractDirIsolation:
    """Concurrent HTTP archive imports must use separate extract directories."""

    def test_run_uses_unique_extract_dir(self, tmp_path):
        """Each run() call should create a uniquely-named extract directory."""
        import unittest.mock as mock

        from vtscore.datasets.importers.http_archive import HttpArchiveDatasetImporter

        imp = HttpArchiveDatasetImporter()
        dirs_used = []

        def capture_load(extract_dir, *args, **kwargs):
            dirs_used.append(str(extract_dir))

        with (
            mock.patch(
                "vtscore.datasets.importers.http_archive.validate_url",
            ),
            mock.patch(
                "vtscore.datasets.importers.http_archive.download_file_with_progress",
                side_effect=lambda url, path, **kw: _write_zip_to(path, tmp_path),
            ),
            mock.patch(
                "vtscore.datasets.importers.http_archive.load_dataset_from_folder",
                side_effect=capture_load,
            ),
            mock.patch(
                "vtscore.datasets.importers.http_archive.DATA_DIR",
                tmp_path,
            ),
        ):
            imp.run({"url": "http://example.com/a.zip", "media_type": "audio"}, {})
            imp.run({"url": "http://example.com/a.zip", "media_type": "audio"}, {})

        assert len(dirs_used) == 2
        # The two extract dirs must be different
        assert dirs_used[0] != dirs_used[1]

    def test_old_shared_dir_name_not_used(self, tmp_path):
        """The old fixed name 'http_archive_extract' must no longer appear."""
        import unittest.mock as mock

        from vtscore.datasets.importers.http_archive import HttpArchiveDatasetImporter

        imp = HttpArchiveDatasetImporter()

        with (
            mock.patch(
                "vtscore.datasets.importers.http_archive.validate_url",
            ),
            mock.patch(
                "vtscore.datasets.importers.http_archive.download_file_with_progress",
                side_effect=lambda url, path, **kw: _write_zip_to(path, tmp_path),
            ),
            mock.patch(
                "vtscore.datasets.importers.http_archive.load_dataset_from_folder",
            ),
            mock.patch(
                "vtscore.datasets.importers.http_archive.DATA_DIR",
                tmp_path,
            ),
        ):
            imp.run({"url": "http://example.com/a.zip", "media_type": "audio"}, {})

        # The old shared directory should not exist
        assert not (tmp_path / "http_archive_extract").exists()

    def test_extract_dir_cleaned_up_after_run(self, tmp_path):
        """The unique extract directory should be removed after run() completes."""
        import unittest.mock as mock

        from vtscore.datasets.importers.http_archive import HttpArchiveDatasetImporter

        imp = HttpArchiveDatasetImporter()

        with (
            mock.patch(
                "vtscore.datasets.importers.http_archive.validate_url",
            ),
            mock.patch(
                "vtscore.datasets.importers.http_archive.download_file_with_progress",
                side_effect=lambda url, path, **kw: _write_zip_to(path, tmp_path),
            ),
            mock.patch(
                "vtscore.datasets.importers.http_archive.load_dataset_from_folder",
            ),
            mock.patch(
                "vtscore.datasets.importers.http_archive.DATA_DIR",
                tmp_path,
            ),
        ):
            imp.run({"url": "http://example.com/a.zip", "media_type": "audio"}, {})

        # No http_archive_extract_* dirs should remain
        remaining = list(tmp_path.glob("http_archive_extract_*"))
        assert remaining == []

    def test_extract_dir_cleaned_up_on_error(self, tmp_path):
        """The extract directory should still be cleaned up if loading fails."""
        import unittest.mock as mock

        from vtscore.datasets.importers.http_archive import HttpArchiveDatasetImporter

        imp = HttpArchiveDatasetImporter()

        with (
            mock.patch(
                "vtscore.datasets.importers.http_archive.validate_url",
            ),
            mock.patch(
                "vtscore.datasets.importers.http_archive.download_file_with_progress",
                side_effect=lambda url, path, **kw: _write_zip_to(path, tmp_path),
            ),
            mock.patch(
                "vtscore.datasets.importers.http_archive.load_dataset_from_folder",
                side_effect=RuntimeError("boom"),
            ),
            mock.patch(
                "vtscore.datasets.importers.http_archive.DATA_DIR",
                tmp_path,
            ),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                imp.run({"url": "http://example.com/a.zip", "media_type": "audio"}, {})

        remaining = list(tmp_path.glob("http_archive_extract_*"))
        assert remaining == []

    def test_chunked_extract_dir_cleaned_up(self, tmp_path):
        """run_chunked() should clean up its extract directory after iteration."""
        import unittest.mock as mock

        from vtscore.datasets.importers.http_archive import HttpArchiveDatasetImporter

        imp = HttpArchiveDatasetImporter()

        with (
            mock.patch(
                "vtscore.datasets.importers.http_archive.validate_url",
            ),
            mock.patch(
                "vtscore.datasets.importers.http_archive.download_file_with_progress",
                side_effect=lambda url, path, **kw: _write_zip_to(path, tmp_path),
            ),
            mock.patch(
                "vtscore.datasets.importers.http_archive._extract_archive",
            ),
            mock.patch(
                "vtscore.datasets.loader.load_dataset_from_folder_chunked",
                return_value=iter([{1: {"test": True}}]),
            ),
            mock.patch(
                "vtscore.datasets.importers.http_archive.DATA_DIR",
                tmp_path,
            ),
        ):
            chunks = list(imp.run_chunked({"url": "http://example.com/a.zip", "media_type": "audio"}, 10))

        assert len(chunks) == 1
        remaining = list(tmp_path.glob("http_archive_extract_*"))
        assert remaining == []


def _write_zip_to(archive_path, tmp_path):
    """Helper: write a minimal .zip so _extract_archive succeeds."""
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("dummy.wav", _make_wav_bytes())


# ---------------------------------------------------------------------------
# Registry-level contract: importers that store files must override resolve_file
# ---------------------------------------------------------------------------


class TestImporterResolveFileContract:
    """Verify that importers which produce disk-backed media override resolve_file.

    This prevents a repeat of the demo-importer bug where a missing
    resolve_file() caused cross-dataset detector labels to silently fail
    to resolve, producing "N/A" verdicts with no error.

    Importers that legitimately cannot resolve files (e.g. ``pickle`` for
    browser uploads, ``combine_datasets`` which delegates to sub-origins)
    are excluded.
    """

    # Importers whose media origins always delegate resolution elsewhere:
    # - pickle: browser-uploaded files with no guaranteed server path
    # - combine_datasets: each element retains its source dataset's origin
    # - local_folder / local_files: UI-only placeholders; uploads are
    #   streamed to a temp directory and re-imported through the regular
    #   `server_folder` importer, so resolution for those medias goes
    #   through `server_folder.resolve_file`.
    _DELEGATE_IMPORTERS = {
        "pickle",
        "combine_datasets",
        "local_folder",
        "local_files",
    }

    def test_all_disk_importers_override_resolve_file(self):
        """Every registered importer that stores files must override resolve_file."""
        from vtscore.datasets.importers import list_importers
        from vtscore.datasets.importers.base import DatasetImporter

        default_method = DatasetImporter.resolve_file

        missing = []
        for imp in list_importers():
            if imp.name in self._DELEGATE_IMPORTERS:
                continue
            # Check if the importer's resolve_file is the unoverridden default
            if type(imp).resolve_file is default_method:
                missing.append(imp.name)

        assert missing == [], (
            f"Importers {missing} do not override resolve_file(). "
            "Cross-dataset label resolution will silently fail for media "
            "loaded by these importers. See DatasetImporter.resolve_file docstring."
        )


# ---------------------------------------------------------------------------
# Recursive vs. top-level folder scanning
# ---------------------------------------------------------------------------


class TestLoadDatasetRecursive:
    """``load_dataset_from_folder`` honours the ``recursive`` flag."""

    def _write_wav(self, path):
        path.write_bytes(_make_wav_bytes())

    def _make_fake_media_type(self):
        import unittest.mock as mock

        import numpy as np

        mt = mock.MagicMock()
        mt.type_id = "audio"
        mt.file_extensions = ["*.wav"]
        mt.load_media_data.return_value = {"duration": 1.0}
        embedder = mock.MagicMock()
        embedder.name = "clap"
        embedder.media_type_id = "audio"
        embedder._model = True
        embedder.embed_media.return_value = np.zeros(3)
        embedder.embed_media_bulk.side_effect = lambda medias: [np.zeros(3) for _ in medias]
        mt._mock_embedder = embedder
        return mt

    def _patch_media_registry(self, mt):
        import unittest.mock as mock
        from contextlib import ExitStack

        stack = ExitStack()
        stack.enter_context(mock.patch("vtscore.media.get_by_folder_name", return_value=mt))
        stack.enter_context(mock.patch("vtscore.media.embedders_for_type", return_value=[mt._mock_embedder]))
        return stack

    def _seed_layout(self, tmp_path):
        """Create ``tmp_path/top.wav`` and ``tmp_path/sub/nested.wav``."""
        self._write_wav(tmp_path / "top.wav")
        sub = tmp_path / "sub"
        sub.mkdir()
        self._write_wav(sub / "nested.wav")

    def test_recursive_default_includes_subfolders(self, tmp_path):
        from vtscore.datasets.loader import load_dataset_from_folder

        self._seed_layout(tmp_path)
        medias: dict = {}
        with self._patch_media_registry(self._make_fake_media_type()):
            load_dataset_from_folder(tmp_path, "audio", medias, on_progress=lambda *a: None)

        names = sorted(m["filename"] for m in medias.values())
        assert names == ["sub/nested.wav", "top.wav"]

    def test_recursive_false_only_top_level(self, tmp_path):
        from vtscore.datasets.loader import load_dataset_from_folder

        self._seed_layout(tmp_path)
        medias: dict = {}
        with self._patch_media_registry(self._make_fake_media_type()):
            load_dataset_from_folder(tmp_path, "audio", medias, on_progress=lambda *a: None, recursive=False)

        names = sorted(m["filename"] for m in medias.values())
        assert names == ["top.wav"]

    def test_chunked_recursive_false_only_top_level(self, tmp_path):
        from vtscore.datasets.loader import load_dataset_from_folder_chunked

        self._seed_layout(tmp_path)
        with self._patch_media_registry(self._make_fake_media_type()):
            chunks = list(
                load_dataset_from_folder_chunked(
                    tmp_path,
                    "audio",
                    chunk_size=10,
                    on_progress=lambda *a: None,
                    recursive=False,
                )
            )

        names = sorted(m["filename"] for chunk in chunks for m in chunk.values())
        assert names == ["top.wav"]


class TestServerFolderImporterRecursive:
    """The ``server_folder`` importer exposes a ``recursive`` checkbox field."""

    def test_field_present_default_true(self):
        from vtscore.datasets.importers.server_folder import IMPORTER

        recursive = next((f for f in IMPORTER.fields if f.key == "recursive"), None)
        assert recursive is not None
        assert recursive.field_type == "checkbox"
        assert str(recursive.default).lower() == "true"

    def test_build_origin_records_recursive(self):
        from vtscore.datasets.importers.server_folder import IMPORTER

        origin = IMPORTER.build_origin({"path": "/tmp/x", "media_type": "audio", "recursive": False})
        assert origin["params"]["recursive"] == "false"


# ---------------------------------------------------------------------------
# Symlinked importer discovery
# ---------------------------------------------------------------------------
