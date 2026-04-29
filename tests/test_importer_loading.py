"""Importer loading tests.

Tests for load_dataset_from_folder: content_vectors, skip_embedding,
content_md5s, relative paths, archive extraction directory isolation,
and the resolve_file contract.
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
        stack.enter_context(mock.patch("vtsearch.media.get_by_folder_name", return_value=mt))
        stack.enter_context(mock.patch("vtsearch.media.embedders_for_type", return_value=[mt._mock_embedder]))
        return stack

    def test_uses_content_vector_when_provided(self, tmp_path):
        """A file whose name is in content_vectors should use that vector."""
        import numpy as np

        from vtsearch.datasets.loader import load_dataset_from_folder

        wav = tmp_path / "a.wav"
        self._write_wav(wav)

        pre_vector = np.array([10.0, 20.0, 30.0])
        mt = self._make_fake_media_type(embed_return=np.zeros(3))

        medias: dict = {}

        def _noop(*a):
            None

        with self._patch_media_registry(mt):
            load_dataset_from_folder(
                tmp_path, "audio", medias, content_vectors={"a.wav": pre_vector}, on_progress=_noop
            )

        assert len(medias) == 1
        np.testing.assert_array_equal(medias[1]["embedding"], pre_vector)
        mt._mock_embedder.embed_media.assert_not_called()

    def test_embeds_normally_when_not_in_content_vectors(self, tmp_path):
        """A file NOT in content_vectors falls back to embed_media()."""
        import numpy as np

        from vtsearch.datasets.loader import load_dataset_from_folder

        wav = tmp_path / "b.wav"
        self._write_wav(wav)

        model_vector = np.array([1.0, 2.0, 3.0])
        mt = self._make_fake_media_type(embed_return=model_vector)

        medias: dict = {}

        def _noop(*a):
            None

        with self._patch_media_registry(mt):
            load_dataset_from_folder(tmp_path, "audio", medias, content_vectors={}, on_progress=_noop)

        assert len(medias) == 1
        np.testing.assert_array_equal(medias[1]["embedding"], model_vector)
        mt._mock_embedder.embed_media.assert_called_once()

    def test_mixed_content_vectors_and_embedding(self, tmp_path):
        """Only files in content_vectors skip embed_media; others are embedded."""
        import numpy as np

        from vtsearch.datasets.loader import load_dataset_from_folder

        wav_a = tmp_path / "a.wav"
        wav_b = tmp_path / "b.wav"
        self._write_wav(wav_a)
        self._write_wav(wav_b)

        pre_vector = np.array([10.0, 20.0])
        model_vector = np.array([1.0, 2.0])
        mt = self._make_fake_media_type(embed_return=model_vector)

        medias: dict = {}

        def _noop(*a):
            None

        with self._patch_media_registry(mt):
            load_dataset_from_folder(
                tmp_path, "audio", medias, content_vectors={"a.wav": pre_vector}, on_progress=_noop
            )

        assert len(medias) == 2
        # One media should have the pre-computed vector, the other the model vector
        embeddings = {c["filename"]: c["embedding"] for c in medias.values()}
        np.testing.assert_array_equal(embeddings["a.wav"], pre_vector)
        np.testing.assert_array_equal(embeddings["b.wav"], model_vector)

    def test_no_content_vectors_param_embeds_all(self, tmp_path):
        """When content_vectors is None (default), all files are embedded."""
        import numpy as np

        from vtsearch.datasets.loader import load_dataset_from_folder

        wav = tmp_path / "c.wav"
        self._write_wav(wav)

        model_vector = np.array([5.0, 6.0])
        mt = self._make_fake_media_type(embed_return=model_vector)

        medias: dict = {}

        def _noop(*a):
            None

        with self._patch_media_registry(mt):
            load_dataset_from_folder(tmp_path, "audio", medias, on_progress=_noop)

        assert len(medias) == 1
        np.testing.assert_array_equal(medias[1]["embedding"], model_vector)
        mt._mock_embedder.embed_media.assert_called_once()

    def test_content_vector_file_skips_none_embed_check(self, tmp_path):
        """A file with a content vector is included even if embed_media would return None."""
        import unittest.mock as mock

        import numpy as np

        from vtsearch.datasets.loader import load_dataset_from_folder

        wav = tmp_path / "d.wav"
        self._write_wav(wav)

        pre_vector = np.array([7.0, 8.0])
        # embed_media returns None, which would normally skip the file
        mt = self._make_fake_media_type(embed_return=None)

        medias: dict = {}

        def _noop(*a):
            None

        with mock.patch("vtsearch.media.get_by_folder_name", return_value=mt):
            load_dataset_from_folder(
                tmp_path, "audio", medias, content_vectors={"d.wav": pre_vector}, on_progress=_noop
            )

        assert len(medias) == 1
        np.testing.assert_array_equal(medias[1]["embedding"], pre_vector)


# ---------------------------------------------------------------------------
# load_dataset_from_folder – skip_embedding support
# ---------------------------------------------------------------------------


class TestLoadDatasetSkipEmbedding:
    """Verify that load_dataset_from_folder respects skip_embedding=True."""

    def _write_wav(self, path):
        """Write a minimal WAV file to *path*."""
        path.write_bytes(_make_wav_bytes())

    def test_skip_embedding_with_content_vectors(self, tmp_path):
        """Pre-computed vectors are used; no embedder is resolved."""
        import unittest.mock as mock

        import numpy as np

        from vtsearch.datasets.loader import load_dataset_from_folder

        wav = tmp_path / "a.wav"
        self._write_wav(wav)

        pre_vector = np.array([1.0, 2.0, 3.0])
        medias: dict = {}

        mt = mock.MagicMock()
        mt.type_id = "audio"
        mt.file_extensions = ["*.wav"]
        mt.load_media_data.return_value = {"duration": 1.0}

        with (
            mock.patch("vtsearch.media.get_by_folder_name", return_value=mt),
            mock.patch("vtsearch.media.embedders_for_type") as mock_emb_for_type,
            mock.patch("vtsearch.media.get_embedder") as mock_get_emb,
        ):
            load_dataset_from_folder(
                tmp_path,
                "audio",
                medias,
                content_vectors={"a.wav": pre_vector},
                on_progress=lambda *a: None,
                skip_embedding=True,
            )

        assert len(medias) == 1
        np.testing.assert_array_equal(medias[1]["embedding"], pre_vector)
        # Embedder registry should never be consulted.
        mock_emb_for_type.assert_not_called()
        mock_get_emb.assert_not_called()

    def test_skip_embedding_without_vectors_sets_none(self, tmp_path):
        """Files without pre-computed vectors get embedding=None."""
        import unittest.mock as mock

        from vtsearch.datasets.loader import load_dataset_from_folder

        wav = tmp_path / "a.wav"
        self._write_wav(wav)

        medias: dict = {}

        mt = mock.MagicMock()
        mt.type_id = "audio"
        mt.file_extensions = ["*.wav"]
        mt.load_media_data.return_value = {"duration": 1.0}

        with (
            mock.patch("vtsearch.media.get_by_folder_name", return_value=mt),
            mock.patch("vtsearch.media.embedders_for_type") as mock_emb_for_type,
        ):
            load_dataset_from_folder(
                tmp_path,
                "audio",
                medias,
                on_progress=lambda *a: None,
                skip_embedding=True,
            )

        assert len(medias) == 1
        assert medias[1]["embedding"] is None
        assert medias[1]["embedder"] == ""
        mock_emb_for_type.assert_not_called()

    def test_skip_embedding_progress_says_loading(self, tmp_path):
        """Progress messages should say 'Loading' not 'Embedding'."""
        import unittest.mock as mock

        from vtsearch.datasets.loader import load_dataset_from_folder

        wav = tmp_path / "a.wav"
        self._write_wav(wav)

        mt = mock.MagicMock()
        mt.type_id = "audio"
        mt.file_extensions = ["*.wav"]
        mt.load_media_data.return_value = {"duration": 1.0}

        progress_calls: list = []

        def track_progress(*args):
            progress_calls.append(args)

        medias: dict = {}
        with (
            mock.patch("vtsearch.media.get_by_folder_name", return_value=mt),
            mock.patch("vtsearch.media.embedders_for_type"),
        ):
            load_dataset_from_folder(
                tmp_path,
                "audio",
                medias,
                on_progress=track_progress,
                skip_embedding=True,
            )

        # The per-file progress calls should use "loading" phase, not "embedding".
        per_file = [c for c in progress_calls if len(c) >= 4 and c[2] > 0]
        assert len(per_file) >= 1
        assert per_file[0][0] == "loading"
        assert "Loading" in per_file[0][1]

    def test_skip_embedding_chunked(self, tmp_path):
        """skip_embedding works with the chunked variant too."""
        import unittest.mock as mock

        import numpy as np

        from vtsearch.datasets.loader import load_dataset_from_folder_chunked

        for name in ("a.wav", "b.wav"):
            self._write_wav(tmp_path / name)

        vectors = {
            "a.wav": np.array([1.0, 2.0]),
            "b.wav": np.array([3.0, 4.0]),
        }

        mt = mock.MagicMock()
        mt.type_id = "audio"
        mt.file_extensions = ["*.wav"]
        mt.load_media_data.return_value = {"duration": 1.0}

        with (
            mock.patch("vtsearch.media.get_by_folder_name", return_value=mt),
            mock.patch("vtsearch.media.embedders_for_type") as mock_emb_for_type,
        ):
            chunks = list(
                load_dataset_from_folder_chunked(
                    tmp_path,
                    "audio",
                    chunk_size=10,
                    content_vectors=vectors,
                    on_progress=lambda *a: None,
                    skip_embedding=True,
                )
            )

        all_medias = {}
        for chunk in chunks:
            all_medias.update(chunk)
        assert len(all_medias) == 2
        mock_emb_for_type.assert_not_called()


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

        from vtsearch.datasets.loader import load_dataset_from_folder

        wav = tmp_path / "a.wav"
        self._write_wav(wav)

        pre_md5 = "0" * 32
        mt = self._make_fake_media_type(embed_return=np.zeros(3))

        medias: dict = {}

        def _noop(*a):
            None

        with mock.patch("vtsearch.media.get_by_folder_name", return_value=mt):
            load_dataset_from_folder(tmp_path, "audio", medias, content_md5s={"a.wav": pre_md5}, on_progress=_noop)

        assert len(medias) == 1
        assert medias[1]["md5"] == pre_md5

    def test_computes_md5_when_not_in_content_md5s(self, tmp_path):
        """A file NOT in content_md5s falls back to computing the hash."""
        import hashlib
        import unittest.mock as mock

        import numpy as np

        from vtsearch.datasets.loader import load_dataset_from_folder

        wav = tmp_path / "b.wav"
        self._write_wav(wav)
        expected_md5 = hashlib.md5(wav.read_bytes()).hexdigest()

        mt = self._make_fake_media_type(embed_return=np.zeros(3))

        medias: dict = {}

        def _noop(*a):
            None

        with mock.patch("vtsearch.media.get_by_folder_name", return_value=mt):
            load_dataset_from_folder(tmp_path, "audio", medias, content_md5s={}, on_progress=_noop)

        assert len(medias) == 1
        assert medias[1]["md5"] == expected_md5

    def test_mixed_content_md5s_and_computed(self, tmp_path):
        """Only files in content_md5s skip MD5 computation; others are hashed."""
        import hashlib
        import unittest.mock as mock

        import numpy as np

        from vtsearch.datasets.loader import load_dataset_from_folder

        wav_a = tmp_path / "a.wav"
        wav_b = tmp_path / "b.wav"
        self._write_wav(wav_a)
        self._write_wav(wav_b)

        pre_md5 = "f" * 32
        computed_md5 = hashlib.md5(wav_b.read_bytes()).hexdigest()
        mt = self._make_fake_media_type(embed_return=np.zeros(3))

        medias: dict = {}

        def _noop(*a):
            None

        with mock.patch("vtsearch.media.get_by_folder_name", return_value=mt):
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

        from vtsearch.datasets.loader import load_dataset_from_folder

        wav = tmp_path / "c.wav"
        self._write_wav(wav)
        expected_md5 = hashlib.md5(wav.read_bytes()).hexdigest()

        mt = self._make_fake_media_type(embed_return=np.zeros(3))

        medias: dict = {}

        def _noop(*a):
            None

        with mock.patch("vtsearch.media.get_by_folder_name", return_value=mt):
            load_dataset_from_folder(tmp_path, "audio", medias, on_progress=_noop)

        assert len(medias) == 1
        assert medias[1]["md5"] == expected_md5

    def test_content_md5s_in_thin_mode(self, tmp_path):
        """content_md5s should work in thin mode too."""
        import unittest.mock as mock

        import numpy as np

        from vtsearch.datasets.loader import load_dataset_from_folder

        wav = tmp_path / "a.wav"
        self._write_wav(wav)

        pre_md5 = "1" * 32
        mt = self._make_fake_media_type(embed_return=np.zeros(3))

        medias: dict = {}

        def _noop(*a):
            None

        with mock.patch("vtsearch.media.get_by_folder_name", return_value=mt):
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

        from vtsearch.datasets.loader import load_dataset_from_folder

        self._write_wav(tmp_path / "a.wav")
        mt = self._make_fake_media_type(embed_return=np.zeros(3))
        medias: dict = {}

        with mock.patch("vtsearch.media.get_by_folder_name", return_value=mt):
            load_dataset_from_folder(tmp_path, "audio", medias, on_progress=lambda *a: None)

        assert medias[1]["filename"] == "a.wav"
        assert medias[1]["origin_name"] == "a.wav"

    def test_subdir_files_use_relative_path(self, tmp_path):
        """Files in subdirectories should keep the relative path."""
        import unittest.mock as mock

        import numpy as np

        from vtsearch.datasets.loader import load_dataset_from_folder

        sub = tmp_path / "cat_a"
        sub.mkdir()
        self._write_wav(sub / "clip.wav")
        mt = self._make_fake_media_type(embed_return=np.zeros(3))
        medias: dict = {}

        with mock.patch("vtsearch.media.get_by_folder_name", return_value=mt):
            load_dataset_from_folder(tmp_path, "audio", medias, on_progress=lambda *a: None)

        assert medias[1]["filename"] == "cat_a/clip.wav"
        assert medias[1]["origin_name"] == "cat_a/clip.wav"

    def test_same_name_different_dirs_are_distinct(self, tmp_path):
        """Identically named files in different subdirs get distinct filenames."""
        import unittest.mock as mock

        import numpy as np

        from vtsearch.datasets.loader import load_dataset_from_folder

        (tmp_path / "dir_a").mkdir()
        (tmp_path / "dir_b").mkdir()
        self._write_wav(tmp_path / "dir_a" / "clip.wav")
        self._write_wav(tmp_path / "dir_b" / "clip.wav")
        mt = self._make_fake_media_type(embed_return=np.zeros(3))
        medias: dict = {}

        with mock.patch("vtsearch.media.get_by_folder_name", return_value=mt):
            load_dataset_from_folder(tmp_path, "audio", medias, on_progress=lambda *a: None)

        filenames = {m["filename"] for m in medias.values()}
        assert "dir_a/clip.wav" in filenames
        assert "dir_b/clip.wav" in filenames
        assert len(filenames) == 2

    def test_content_vectors_fallback_to_basename(self, tmp_path):
        """content_vectors keyed by basename should still work for flat files."""
        import unittest.mock as mock

        import numpy as np

        from vtsearch.datasets.loader import load_dataset_from_folder

        self._write_wav(tmp_path / "x.wav")
        pre = np.array([99.0, 99.0, 99.0])
        mt = self._make_fake_media_type(embed_return=np.zeros(3))
        medias: dict = {}

        with mock.patch("vtsearch.media.get_by_folder_name", return_value=mt):
            load_dataset_from_folder(
                tmp_path, "audio", medias, content_vectors={"x.wav": pre}, on_progress=lambda *a: None
            )

        np.testing.assert_array_equal(medias[1]["embedding"], pre)

    def test_chunked_preserves_relative_paths(self, tmp_path):
        """Chunked loader should also preserve relative paths."""
        import unittest.mock as mock

        import numpy as np

        from vtsearch.datasets.loader import load_dataset_from_folder_chunked

        sub = tmp_path / "sub"
        sub.mkdir()
        self._write_wav(sub / "chunk.wav")
        mt = self._make_fake_media_type(embed_return=np.zeros(3))

        with mock.patch("vtsearch.media.get_by_folder_name", return_value=mt):
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

        from vtsearch.datasets.importers.http_archive import HttpArchiveDatasetImporter

        imp = HttpArchiveDatasetImporter()
        dirs_used = []

        def capture_load(extract_dir, *args, **kwargs):
            dirs_used.append(str(extract_dir))

        with (
            mock.patch(
                "vtsearch.datasets.importers.http_archive.validate_url",
            ),
            mock.patch(
                "vtsearch.datasets.importers.http_archive.download_file_with_progress",
                side_effect=lambda url, path, **kw: _write_zip_to(path, tmp_path),
            ),
            mock.patch(
                "vtsearch.datasets.importers.http_archive.load_dataset_from_folder",
                side_effect=capture_load,
            ),
            mock.patch(
                "vtsearch.datasets.importers.http_archive.DATA_DIR",
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

        from vtsearch.datasets.importers.http_archive import HttpArchiveDatasetImporter

        imp = HttpArchiveDatasetImporter()

        with (
            mock.patch(
                "vtsearch.datasets.importers.http_archive.validate_url",
            ),
            mock.patch(
                "vtsearch.datasets.importers.http_archive.download_file_with_progress",
                side_effect=lambda url, path, **kw: _write_zip_to(path, tmp_path),
            ),
            mock.patch(
                "vtsearch.datasets.importers.http_archive.load_dataset_from_folder",
            ),
            mock.patch(
                "vtsearch.datasets.importers.http_archive.DATA_DIR",
                tmp_path,
            ),
        ):
            imp.run({"url": "http://example.com/a.zip", "media_type": "audio"}, {})

        # The old shared directory should not exist
        assert not (tmp_path / "http_archive_extract").exists()

    def test_extract_dir_cleaned_up_after_run(self, tmp_path):
        """The unique extract directory should be removed after run() completes."""
        import unittest.mock as mock

        from vtsearch.datasets.importers.http_archive import HttpArchiveDatasetImporter

        imp = HttpArchiveDatasetImporter()

        with (
            mock.patch(
                "vtsearch.datasets.importers.http_archive.validate_url",
            ),
            mock.patch(
                "vtsearch.datasets.importers.http_archive.download_file_with_progress",
                side_effect=lambda url, path, **kw: _write_zip_to(path, tmp_path),
            ),
            mock.patch(
                "vtsearch.datasets.importers.http_archive.load_dataset_from_folder",
            ),
            mock.patch(
                "vtsearch.datasets.importers.http_archive.DATA_DIR",
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

        from vtsearch.datasets.importers.http_archive import HttpArchiveDatasetImporter

        imp = HttpArchiveDatasetImporter()

        with (
            mock.patch(
                "vtsearch.datasets.importers.http_archive.validate_url",
            ),
            mock.patch(
                "vtsearch.datasets.importers.http_archive.download_file_with_progress",
                side_effect=lambda url, path, **kw: _write_zip_to(path, tmp_path),
            ),
            mock.patch(
                "vtsearch.datasets.importers.http_archive.load_dataset_from_folder",
                side_effect=RuntimeError("boom"),
            ),
            mock.patch(
                "vtsearch.datasets.importers.http_archive.DATA_DIR",
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

        from vtsearch.datasets.importers.http_archive import HttpArchiveDatasetImporter

        imp = HttpArchiveDatasetImporter()

        with (
            mock.patch(
                "vtsearch.datasets.importers.http_archive.validate_url",
            ),
            mock.patch(
                "vtsearch.datasets.importers.http_archive.download_file_with_progress",
                side_effect=lambda url, path, **kw: _write_zip_to(path, tmp_path),
            ),
            mock.patch(
                "vtsearch.datasets.importers.http_archive._extract_archive",
            ),
            mock.patch(
                "vtsearch.datasets.loader.load_dataset_from_folder_chunked",
                return_value=iter([{1: {"test": True}}]),
            ),
            mock.patch(
                "vtsearch.datasets.importers.http_archive.DATA_DIR",
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
    # - local_folder: a UI-only placeholder; uploads are streamed to a temp
    #   directory and re-imported through the regular `folder` importer,
    #   so resolution for those medias goes through `folder.resolve_file`.
    _DELEGATE_IMPORTERS = {"pickle", "combine_datasets", "local_folder"}

    def test_all_disk_importers_override_resolve_file(self):
        """Every registered importer that stores files must override resolve_file."""
        from vtsearch.datasets.importers import list_importers
        from vtsearch.datasets.importers.base import DatasetImporter

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
# Symlinked importer discovery
# ---------------------------------------------------------------------------
