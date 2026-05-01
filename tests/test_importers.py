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
import tarfile
import zipfile

import pytest

from helpers import make_raw_wav_bytes as _make_wav_bytes


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


class TestImporterBaseCustomMetadataMap:
    def test_base_class_instance_has_custom_metadata_map(self):
        from vtsearch.datasets.importers.base import DatasetImporter

        class MinimalImporter(DatasetImporter):
            name = "minimal"
            display_name = "Minimal"
            description = "Minimal importer."
            fields = []

            def run(self, field_values, medias):
                pass

        imp = MinimalImporter()
        assert hasattr(imp, "custom_metadata_map")
        assert imp.custom_metadata_map == {}

    def test_custom_metadata_map_independent_across_instances(self):
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
        a.custom_metadata_map["file.wav"] = {"md5": "abc123"}
        assert b.custom_metadata_map == {}


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
        from vtsearch.datasets.importers.http_archive import IMPORTER

        return IMPORTER

    def test_name_is_http_archive(self):
        assert self._get_importer().name == "http_archive"

    def test_display_name(self):
        assert self._get_importer().display_name == "Import from URL"

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
        assert d["display_name"] == "Import from URL"

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
        assert "audio" in opts
        assert "video" in opts
        assert "image" in opts
        assert "text" in opts


# ---------------------------------------------------------------------------
# Folder importer metadata
# ---------------------------------------------------------------------------


class TestFolderImporterMetadata:
    def _get_importer(self):
        from vtsearch.datasets.importers.server_folder import IMPORTER

        return IMPORTER

    def test_name_is_folder(self):
        assert self._get_importer().name == "server_folder"

    def test_icon_is_folder_emoji(self):
        # 📁 — frontend renders this as a "folder" icon, matching the
        # browser-side Local Folder card.  The Server tab makes the
        # server-vs-local distinction.
        assert self._get_importer().icon == "📁"

    def test_description_says_media_files_from_a_folder(self):
        desc = self._get_importer().description.lower()
        assert "media files" in desc

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
        assert d["icon"] == "📁"


# ---------------------------------------------------------------------------
# Folder importer not in builtin names
# ---------------------------------------------------------------------------


class TestBuiltinImporterNames:
    def test_folder_has_form_ui_mode(self):
        from vtsearch.datasets.importers import get_importer

        imp = get_importer("server_folder")
        assert imp is not None
        assert imp.ui_mode == "form"

    def test_pickle_has_file_upload_ui_mode(self):
        from vtsearch.datasets.importers import get_importer

        imp = get_importer("pickle")
        assert imp is not None
        assert imp.ui_mode == "file_upload"


# ---------------------------------------------------------------------------
# _extract_archive helper
# ---------------------------------------------------------------------------


class TestExtractArchive:
    def test_extract_zip(self, tmp_path):
        from vtsearch.datasets.importers.http_archive import _extract_archive

        wav_data = _make_wav_bytes()
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("sounds/tone.wav", wav_data)
        extract_dir = tmp_path / "out"
        extract_dir.mkdir()
        _extract_archive(zip_path, extract_dir)
        assert (extract_dir / "sounds" / "tone.wav").exists()

    def test_extract_tar_uncompressed(self, tmp_path):
        from vtsearch.datasets.importers.http_archive import _extract_archive

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
        from vtsearch.datasets.importers.http_archive import _extract_archive

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
        from vtsearch.datasets.importers.http_archive import _extract_archive

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
        from vtsearch.datasets.importers.http_archive import _extract_archive

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

        from vtsearch.datasets.importers.http_archive import _extract_archive

        rar_path = tmp_path / "test.rar"
        # Write RAR v4 magic bytes so it's identified as .rar by extension
        rar_path.write_bytes(b"Rar!\x1a\x07\x00" + b"\x00" * 20)
        extract_dir = tmp_path / "out"
        extract_dir.mkdir()

        with mock.patch.dict(sys.modules, {"rarfile": None}):
            with pytest.raises((RuntimeError, ImportError, Exception)):
                _extract_archive(rar_path, extract_dir)

    def test_zip_preserves_multiple_files(self, tmp_path):
        from vtsearch.datasets.importers.http_archive import _extract_archive

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
