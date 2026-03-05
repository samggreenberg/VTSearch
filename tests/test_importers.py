"""Tests for the importer subsystem that don't require ML/torch dependencies.

These tests verify:
- HTTP Archive importer metadata (name, icon, description)
- Folder importer metadata (icon, description, field ordering)
- _extract_archive helper (zip and tar)
- DatasetImporter base class icon field and content_vectors attribute
- DatasetImporter base class content_md5s attribute
- Folder importer is not in _BUILTIN_IMPORTER_NAMES
- load_dataset_from_folder content_vectors support
- load_dataset_from_folder content_md5s support
- load_dataset_from_folder preserves relative paths for files in subdirs
"""

from __future__ import annotations

import io
import struct
import tarfile
import wave
import zipfile

import pytest


def _make_wav_bytes() -> bytes:
    """Create a minimal valid WAV file in memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        samples = struct.pack("<" + "h" * 100, *([0] * 100))
        wf.writeframes(samples)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# DatasetImporter base class – icon field and content_vectors
# ---------------------------------------------------------------------------


class TestImporterBaseContentVectors:
    def test_base_class_instance_has_content_vectors(self):
        from vtsearch.datasets.importers.base import DatasetImporter

        class MinimalImporter(DatasetImporter):
            name = "minimal"
            display_name = "Minimal"
            description = "Minimal importer."
            fields = []

            def run(self, field_values, medias):
                pass

        imp = MinimalImporter()
        assert hasattr(imp, "content_vectors")
        assert imp.content_vectors == {}

    def test_content_vectors_defaults_to_empty_dict(self):
        from vtsearch.datasets.importers.base import DatasetImporter

        class Imp(DatasetImporter):
            name = "t"
            display_name = "T"
            description = "T"
            fields = []

            def run(self, field_values, medias):
                pass

        assert Imp().content_vectors == {}

    def test_content_vectors_are_independent_across_instances(self):
        from vtsearch.datasets.importers.base import DatasetImporter

        class Imp(DatasetImporter):
            name = "t"
            display_name = "T"
            description = "T"
            fields = []

            def run(self, field_values, medias):
                pass

        a = Imp()
        b = Imp()
        a.content_vectors["file.wav"] = [1, 2, 3]
        assert b.content_vectors == {}

    def test_subclass_can_populate_content_vectors_during_run(self):
        import numpy as np

        from vtsearch.datasets.importers.base import DatasetImporter

        class VectorImporter(DatasetImporter):
            name = "vec"
            display_name = "Vector"
            description = "Provides vectors."
            fields = []

            def run(self, field_values, medias):
                self.content_vectors["test.wav"] = np.array([1.0, 2.0, 3.0])

        imp = VectorImporter()
        imp.run({}, {})
        assert "test.wav" in imp.content_vectors
        assert list(imp.content_vectors["test.wav"]) == [1.0, 2.0, 3.0]


class TestImporterBaseContentMD5s:
    def test_base_class_instance_has_content_md5s(self):
        from vtsearch.datasets.importers.base import DatasetImporter

        class MinimalImporter(DatasetImporter):
            name = "minimal"
            display_name = "Minimal"
            description = "Minimal importer."
            fields = []

            def run(self, field_values, medias):
                pass

        imp = MinimalImporter()
        assert hasattr(imp, "content_md5s")
        assert imp.content_md5s == {}

    def test_content_md5s_defaults_to_empty_dict(self):
        from vtsearch.datasets.importers.base import DatasetImporter

        class Imp(DatasetImporter):
            name = "t"
            display_name = "T"
            description = "T"
            fields = []

            def run(self, field_values, medias):
                pass

        assert Imp().content_md5s == {}

    def test_content_md5s_are_independent_across_instances(self):
        from vtsearch.datasets.importers.base import DatasetImporter

        class Imp(DatasetImporter):
            name = "t"
            display_name = "T"
            description = "T"
            fields = []

            def run(self, field_values, medias):
                pass

        a = Imp()
        b = Imp()
        a.content_md5s["file.wav"] = "abc123"
        assert b.content_md5s == {}

    def test_subclass_can_populate_content_md5s_during_run(self):
        from vtsearch.datasets.importers.base import DatasetImporter

        class MD5Importer(DatasetImporter):
            name = "md5"
            display_name = "MD5"
            description = "Provides MD5s."
            fields = []

            def run(self, field_values, medias):
                self.content_md5s["test.wav"] = "d41d8cd98f00b204e9800998ecf8427e"

        imp = MD5Importer()
        imp.run({}, {})
        assert "test.wav" in imp.content_md5s
        assert imp.content_md5s["test.wav"] == "d41d8cd98f00b204e9800998ecf8427e"


class TestImporterBaseIcon:
    def test_base_class_has_icon_attribute(self):
        from vtsearch.datasets.importers.base import DatasetImporter

        assert hasattr(DatasetImporter, "icon")
        assert DatasetImporter.icon == "🔌"

    def test_to_dict_includes_icon(self):
        from vtsearch.datasets.importers.base import DatasetImporter

        class DummyImporter(DatasetImporter):
            name = "dummy"
            display_name = "Dummy"
            description = "A dummy importer."
            icon = "🧪"
            fields = []

            def run(self, field_values, medias):
                pass

        d = DummyImporter().to_dict()
        assert "icon" in d
        assert d["icon"] == "🧪"

    def test_to_dict_uses_default_icon_when_not_overridden(self):
        from vtsearch.datasets.importers.base import DatasetImporter

        class MinimalImporter(DatasetImporter):
            name = "minimal"
            display_name = "Minimal"
            description = "No icon override."
            fields = []

            def run(self, field_values, medias):
                pass

        d = MinimalImporter().to_dict()
        assert d["icon"] == "🔌"


# ---------------------------------------------------------------------------
# HTTP Archive importer metadata
# ---------------------------------------------------------------------------


class TestHttpArchiveImporterMetadata:
    def _get_importer(self):
        from vtsearch.datasets.importers.http_zip import IMPORTER

        return IMPORTER

    def test_name_is_http_archive(self):
        assert self._get_importer().name == "http_archive"

    def test_display_name(self):
        assert self._get_importer().display_name == "Generate from HTTP Archive"

    def test_icon_is_globe(self):
        assert self._get_importer().icon == "🌐"

    def test_description_mentions_zip_tar_rar(self):
        desc = self._get_importer().description.lower()
        assert "zip" in desc
        assert "tar" in desc
        assert "rar" in desc

    def test_to_dict_includes_icon(self):
        d = self._get_importer().to_dict()
        assert d["icon"] == "🌐"
        assert d["name"] == "http_archive"
        assert d["display_name"] == "Generate from HTTP Archive"

    def test_fields_include_url_and_media_type(self):
        fields = {f.key: f for f in self._get_importer().fields}
        assert "url" in fields
        assert "media_type" in fields

    def test_url_field_type(self):
        fields = {f.key: f for f in self._get_importer().fields}
        assert fields["url"].field_type == "url"

    def test_media_type_options(self):
        fields = {f.key: f for f in self._get_importer().fields}
        opts = fields["media_type"].options
        assert "sounds" in opts
        assert "videos" in opts
        assert "images" in opts
        assert "paragraphs" in opts


# ---------------------------------------------------------------------------
# Folder importer metadata
# ---------------------------------------------------------------------------


class TestFolderImporterMetadata:
    def _get_importer(self):
        from vtsearch.datasets.importers.folder import IMPORTER

        return IMPORTER

    def test_name_is_folder(self):
        assert self._get_importer().name == "folder"

    def test_icon_is_folder_emoji(self):
        assert self._get_importer().icon == "📂"

    def test_description_says_media_files_from_a_folder(self):
        desc = self._get_importer().description.lower()
        assert "media files from a folder" in desc

    def test_description_does_not_list_specific_media_types(self):
        desc = self._get_importer().description
        assert "sounds/videos" not in desc
        assert "(sounds" not in desc

    def test_media_type_field_before_path_field(self):
        keys = [f.key for f in self._get_importer().fields]
        assert keys.index("media_type") < keys.index("path")

    def test_path_field_type_is_folder(self):
        fields = {f.key: f for f in self._get_importer().fields}
        assert fields["path"].field_type == "folder"

    def test_to_dict_includes_icon(self):
        d = self._get_importer().to_dict()
        assert d["icon"] == "📂"


# ---------------------------------------------------------------------------
# Folder importer not in builtin names
# ---------------------------------------------------------------------------


class TestBuiltinImporterNames:
    def test_folder_not_in_builtin_names(self):
        from vtsearch.routes.datasets import _BUILTIN_IMPORTER_NAMES

        assert "folder" not in _BUILTIN_IMPORTER_NAMES

    def test_pickle_still_in_builtin_names(self):
        from vtsearch.routes.datasets import _BUILTIN_IMPORTER_NAMES

        assert "pickle" in _BUILTIN_IMPORTER_NAMES


# ---------------------------------------------------------------------------
# _extract_archive helper
# ---------------------------------------------------------------------------


class TestExtractArchive:
    def test_extract_zip(self, tmp_path):
        from vtsearch.datasets.importers.http_zip import _extract_archive

        wav_data = _make_wav_bytes()
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("sounds/tone.wav", wav_data)
        extract_dir = tmp_path / "out"
        extract_dir.mkdir()
        _extract_archive(zip_path, extract_dir)
        assert (extract_dir / "sounds" / "tone.wav").exists()

    def test_extract_tar_uncompressed(self, tmp_path):
        from vtsearch.datasets.importers.http_zip import _extract_archive

        wav_data = _make_wav_bytes()
        tar_path = tmp_path / "test.tar"
        with tarfile.open(tar_path, "w") as tf:
            info = tarfile.TarInfo(name="tone.wav")
            info.size = len(wav_data)
            tf.addfile(info, io.BytesIO(wav_data))
        extract_dir = tmp_path / "out"
        extract_dir.mkdir()
        _extract_archive(tar_path, extract_dir)
        assert (extract_dir / "tone.wav").exists()

    def test_extract_tar_gz(self, tmp_path):
        from vtsearch.datasets.importers.http_zip import _extract_archive

        wav_data = _make_wav_bytes()
        tar_path = tmp_path / "test.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tf:
            info = tarfile.TarInfo(name="sounds/tone.wav")
            info.size = len(wav_data)
            tf.addfile(info, io.BytesIO(wav_data))
        extract_dir = tmp_path / "out"
        extract_dir.mkdir()
        _extract_archive(tar_path, extract_dir)
        assert (extract_dir / "sounds" / "tone.wav").exists()

    def test_extract_tar_bz2(self, tmp_path):
        from vtsearch.datasets.importers.http_zip import _extract_archive

        wav_data = _make_wav_bytes()
        tar_path = tmp_path / "test.tar.bz2"
        with tarfile.open(tar_path, "w:bz2") as tf:
            info = tarfile.TarInfo(name="tone.wav")
            info.size = len(wav_data)
            tf.addfile(info, io.BytesIO(wav_data))
        extract_dir = tmp_path / "out"
        extract_dir.mkdir()
        _extract_archive(tar_path, extract_dir)
        assert (extract_dir / "tone.wav").exists()

    def test_unsupported_extension_raises_value_error(self, tmp_path):
        from vtsearch.datasets.importers.http_zip import _extract_archive

        # A file that is not a zip or tar and doesn't end in .rar
        bad_archive = tmp_path / "test.7z"
        bad_archive.write_bytes(b"not a valid archive format at all")
        extract_dir = tmp_path / "out"
        extract_dir.mkdir()
        with pytest.raises((ValueError, Exception)):
            _extract_archive(bad_archive, extract_dir)

    def test_rar_without_rarfile_raises_runtime_error(self, tmp_path):
        """Attempting RAR extraction without rarfile installed should fail gracefully."""
        import sys
        import unittest.mock as mock

        from vtsearch.datasets.importers.http_zip import _extract_archive

        rar_path = tmp_path / "test.rar"
        # Write RAR v4 magic bytes so it's identified as .rar by extension
        rar_path.write_bytes(b"Rar!\x1a\x07\x00" + b"\x00" * 20)
        extract_dir = tmp_path / "out"
        extract_dir.mkdir()

        with mock.patch.dict(sys.modules, {"rarfile": None}):
            with pytest.raises((RuntimeError, ImportError, Exception)):
                _extract_archive(rar_path, extract_dir)

    def test_zip_preserves_multiple_files(self, tmp_path):
        from vtsearch.datasets.importers.http_zip import _extract_archive

        zip_path = tmp_path / "multi.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for i in range(3):
                zf.writestr(f"file{i}.wav", _make_wav_bytes())
        extract_dir = tmp_path / "out"
        extract_dir.mkdir()
        _extract_archive(zip_path, extract_dir)
        assert len(list(extract_dir.glob("*.wav"))) == 3


# ---------------------------------------------------------------------------
# load_dataset_from_folder – content_vectors support
# ---------------------------------------------------------------------------


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
                tmp_path, "sounds", medias, content_vectors={"a.wav": pre_vector}, on_progress=_noop
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
            load_dataset_from_folder(tmp_path, "sounds", medias, content_vectors={}, on_progress=_noop)

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
                tmp_path, "sounds", medias, content_vectors={"a.wav": pre_vector}, on_progress=_noop
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
            load_dataset_from_folder(tmp_path, "sounds", medias, on_progress=_noop)

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
                tmp_path, "sounds", medias, content_vectors={"d.wav": pre_vector}, on_progress=_noop
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

        from vtsearch.datasets.loader import load_dataset_from_folder

        wav = tmp_path / "a.wav"
        self._write_wav(wav)

        pre_md5 = "0" * 32
        mt = self._make_fake_media_type(embed_return=np.zeros(3))

        medias: dict = {}

        def _noop(*a):
            None

        with mock.patch("vtsearch.media.get_by_folder_name", return_value=mt):
            load_dataset_from_folder(tmp_path, "sounds", medias, content_md5s={"a.wav": pre_md5}, on_progress=_noop)

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
            load_dataset_from_folder(tmp_path, "sounds", medias, content_md5s={}, on_progress=_noop)

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
            load_dataset_from_folder(tmp_path, "sounds", medias, content_md5s={"a.wav": pre_md5}, on_progress=_noop)

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
            load_dataset_from_folder(tmp_path, "sounds", medias, on_progress=_noop)

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
                tmp_path, "sounds", medias, content_md5s={"a.wav": pre_md5}, on_progress=_noop, thin=True
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
            load_dataset_from_folder(tmp_path, "sounds", medias, on_progress=lambda *a: None)

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
            load_dataset_from_folder(tmp_path, "sounds", medias, on_progress=lambda *a: None)

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
            load_dataset_from_folder(tmp_path, "sounds", medias, on_progress=lambda *a: None)

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
                tmp_path, "sounds", medias, content_vectors={"x.wav": pre}, on_progress=lambda *a: None
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
            chunks = list(
                load_dataset_from_folder_chunked(tmp_path, "sounds", 10, on_progress=lambda *a: None)
            )

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

        from vtsearch.datasets.importers.http_zip import HttpArchiveDatasetImporter

        imp = HttpArchiveDatasetImporter()
        dirs_used = []

        real_load = None

        def capture_load(extract_dir, *args, **kwargs):
            dirs_used.append(str(extract_dir))

        with (
            mock.patch(
                "vtsearch.datasets.importers.http_zip.validate_url",
            ),
            mock.patch(
                "vtsearch.datasets.importers.http_zip.download_file_with_progress",
                side_effect=lambda url, path, **kw: _write_zip_to(path, tmp_path),
            ),
            mock.patch(
                "vtsearch.datasets.importers.http_zip.load_dataset_from_folder",
                side_effect=capture_load,
            ),
            mock.patch(
                "vtsearch.datasets.importers.http_zip.DATA_DIR", tmp_path,
            ),
        ):
            imp.run({"url": "http://example.com/a.zip", "media_type": "sounds"}, {})
            imp.run({"url": "http://example.com/a.zip", "media_type": "sounds"}, {})

        assert len(dirs_used) == 2
        # The two extract dirs must be different
        assert dirs_used[0] != dirs_used[1]

    def test_old_shared_dir_name_not_used(self, tmp_path):
        """The old fixed name 'http_archive_extract' must no longer appear."""
        import unittest.mock as mock

        from vtsearch.datasets.importers.http_zip import HttpArchiveDatasetImporter

        imp = HttpArchiveDatasetImporter()

        with (
            mock.patch(
                "vtsearch.datasets.importers.http_zip.validate_url",
            ),
            mock.patch(
                "vtsearch.datasets.importers.http_zip.download_file_with_progress",
                side_effect=lambda url, path, **kw: _write_zip_to(path, tmp_path),
            ),
            mock.patch(
                "vtsearch.datasets.importers.http_zip.load_dataset_from_folder",
            ),
            mock.patch(
                "vtsearch.datasets.importers.http_zip.DATA_DIR", tmp_path,
            ),
        ):
            imp.run({"url": "http://example.com/a.zip", "media_type": "sounds"}, {})

        # The old shared directory should not exist
        assert not (tmp_path / "http_archive_extract").exists()

    def test_extract_dir_cleaned_up_after_run(self, tmp_path):
        """The unique extract directory should be removed after run() completes."""
        import unittest.mock as mock

        from vtsearch.datasets.importers.http_zip import HttpArchiveDatasetImporter

        imp = HttpArchiveDatasetImporter()

        with (
            mock.patch(
                "vtsearch.datasets.importers.http_zip.validate_url",
            ),
            mock.patch(
                "vtsearch.datasets.importers.http_zip.download_file_with_progress",
                side_effect=lambda url, path, **kw: _write_zip_to(path, tmp_path),
            ),
            mock.patch(
                "vtsearch.datasets.importers.http_zip.load_dataset_from_folder",
            ),
            mock.patch(
                "vtsearch.datasets.importers.http_zip.DATA_DIR", tmp_path,
            ),
        ):
            imp.run({"url": "http://example.com/a.zip", "media_type": "sounds"}, {})

        # No http_archive_extract_* dirs should remain
        remaining = list(tmp_path.glob("http_archive_extract_*"))
        assert remaining == []

    def test_extract_dir_cleaned_up_on_error(self, tmp_path):
        """The extract directory should still be cleaned up if loading fails."""
        import unittest.mock as mock

        from vtsearch.datasets.importers.http_zip import HttpArchiveDatasetImporter

        imp = HttpArchiveDatasetImporter()

        with (
            mock.patch(
                "vtsearch.datasets.importers.http_zip.validate_url",
            ),
            mock.patch(
                "vtsearch.datasets.importers.http_zip.download_file_with_progress",
                side_effect=lambda url, path, **kw: _write_zip_to(path, tmp_path),
            ),
            mock.patch(
                "vtsearch.datasets.importers.http_zip.load_dataset_from_folder",
                side_effect=RuntimeError("boom"),
            ),
            mock.patch(
                "vtsearch.datasets.importers.http_zip.DATA_DIR", tmp_path,
            ),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                imp.run({"url": "http://example.com/a.zip", "media_type": "sounds"}, {})

        remaining = list(tmp_path.glob("http_archive_extract_*"))
        assert remaining == []

    def test_chunked_extract_dir_cleaned_up(self, tmp_path):
        """run_chunked() should clean up its extract directory after iteration."""
        import unittest.mock as mock

        from vtsearch.datasets.importers.http_zip import HttpArchiveDatasetImporter

        imp = HttpArchiveDatasetImporter()

        with (
            mock.patch(
                "vtsearch.datasets.importers.http_zip.validate_url",
            ),
            mock.patch(
                "vtsearch.datasets.importers.http_zip.download_file_with_progress",
                side_effect=lambda url, path, **kw: _write_zip_to(path, tmp_path),
            ),
            mock.patch(
                "vtsearch.datasets.importers.http_zip._extract_archive",
            ),
            mock.patch(
                "vtsearch.datasets.loader.load_dataset_from_folder_chunked",
                return_value=iter([{1: {"test": True}}]),
            ),
            mock.patch(
                "vtsearch.datasets.importers.http_zip.DATA_DIR", tmp_path,
            ),
        ):
            chunks = list(imp.run_chunked({"url": "http://example.com/a.zip", "media_type": "sounds"}, 10))

        assert len(chunks) == 1
        remaining = list(tmp_path.glob("http_archive_extract_*"))
        assert remaining == []


def _write_zip_to(archive_path, tmp_path):
    """Helper: write a minimal .zip so _extract_archive succeeds."""
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("dummy.wav", _make_wav_bytes())
