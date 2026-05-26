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
        from vtscore.datasets.importers.base import DatasetImporter

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
        from vtscore.datasets.importers.base import DatasetImporter

        class Imp(DatasetImporter):
            name = "t"
            display_name = "T"
            description = "T"
            fields = []

            def run(self, field_values, medias):
                pass

        assert Imp().content_vectors == {}

    def test_content_vectors_are_independent_across_instances(self):
        from vtscore.datasets.importers.base import DatasetImporter

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

        from vtscore.datasets.importers.base import DatasetImporter

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
        from vtscore.datasets.importers.base import DatasetImporter

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
        from vtscore.datasets.importers.base import DatasetImporter

        class Imp(DatasetImporter):
            name = "t"
            display_name = "T"
            description = "T"
            fields = []

            def run(self, field_values, medias):
                pass

        assert Imp().content_md5s == {}

    def test_content_md5s_are_independent_across_instances(self):
        from vtscore.datasets.importers.base import DatasetImporter

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
        from vtscore.datasets.importers.base import DatasetImporter

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
        from vtscore.datasets.importers.base import DatasetImporter

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
        from vtscore.datasets.importers.base import DatasetImporter

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
        from vtscore.datasets.importers.base import DatasetImporter

        assert hasattr(DatasetImporter, "icon")
        assert DatasetImporter.icon == "🔌"

    def test_to_dict_includes_icon(self):
        from vtscore.datasets.importers.base import DatasetImporter

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
        from vtscore.datasets.importers.base import DatasetImporter

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
        from vtscore.datasets.importers.http_archive import IMPORTER

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
        from vtscore.datasets.importers.server_folder import IMPORTER

        return IMPORTER

    def test_name_is_folder(self):
        assert self._get_importer().name == "server_folder"

    def test_icon_is_folder_emoji(self):
        # 📁 - frontend renders this as a "folder" icon, matching the
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
# DatasetImporter base class - user-typed dataset name
# ---------------------------------------------------------------------------


class TestImporterDatasetName:
    """The base DatasetImporter exposes a user-typeable ``dataset_name``
    field that overrides the per-importer default name."""

    def _make_importer(self):
        from vtscore.datasets.importers.base import DatasetImporter

        class Imp(DatasetImporter):
            name = "named"
            display_name = "Named"
            description = "Has a default name."
            fields = []

            def default_display_name(self, field_values):
                return field_values.get("flavour") or self.display_name

            def run(self, field_values, medias):
                pass

        return Imp()

    def test_to_dict_appends_dataset_name_field(self):
        from vtscore.datasets.importers.base import (
            DATASET_NAME_FIELD_KEY,
            DatasetImporter,
            ImporterField,
        )

        class Imp(DatasetImporter):
            name = "imp"
            display_name = "Imp"
            description = "."
            fields = [ImporterField("path", "Path", "text")]

            def run(self, field_values, medias):
                pass

        d = Imp().to_dict()
        assert d["fields"][0]["key"] == "path"
        assert d["fields"][-1]["key"] == DATASET_NAME_FIELD_KEY
        assert d["fields"][-1]["required"] is False
        assert d["fields"][-1]["field_type"] == "text"

    def test_class_fields_attribute_unchanged_by_to_dict(self):
        """to_dict() appends dataset_name only on the serialised payload -
        the class-level ``fields`` attribute remains as the developer wrote it."""
        from vtscore.datasets.importers.base import DatasetImporter, ImporterField

        class Imp(DatasetImporter):
            name = "imp2"
            display_name = "Imp"
            description = "."
            fields = [ImporterField("path", "Path", "text")]

            def run(self, field_values, medias):
                pass

        imp = Imp()
        # to_dict shouldn't mutate the class attribute.
        imp.to_dict()
        keys = [f.key for f in Imp.fields]
        assert keys == ["path"]
        assert "dataset_name" not in keys

    def test_resolve_display_name_prefers_user_value(self):
        imp = self._make_importer()
        assert imp.resolve_display_name({"dataset_name": "My Pictures", "flavour": "blue"}) == "My Pictures"

    def test_resolve_display_name_falls_back_to_default(self):
        imp = self._make_importer()
        assert imp.resolve_display_name({"flavour": "blue"}) == "blue"

    def test_resolve_display_name_strips_whitespace(self):
        imp = self._make_importer()
        # An all-whitespace name is treated as empty - the default wins.
        assert imp.resolve_display_name({"dataset_name": "   "}) == imp.display_name

    def test_resolve_display_name_empty_field_values(self):
        imp = self._make_importer()
        # Defensive: still returns *something* for empty input.
        assert imp.resolve_display_name({}) == imp.display_name
        assert imp.resolve_display_name(None) == imp.display_name


class TestImporterDefaultDisplayName:
    """Each importer derives a sensible default name from its field values."""

    def test_demo_uses_entry_label(self):
        from vtscore.datasets.config import DEMO_DATASETS
        from vtscore.datasets.importers.demo import IMPORTER

        first_name = next(iter(DEMO_DATASETS.keys()))
        first_label = DEMO_DATASETS[first_name].get("label", first_name)
        assert IMPORTER.default_display_name({"name": first_name}) == first_label

    def test_demo_user_typed_name_wins(self):
        from vtscore.datasets.config import DEMO_DATASETS
        from vtscore.datasets.importers.demo import IMPORTER

        first_name = next(iter(DEMO_DATASETS.keys()))
        assert IMPORTER.resolve_display_name({"name": first_name, "dataset_name": "Override"}) == "Override"

    def test_synthetic_default_name_matches_size_and_type(self):
        from vtscore.datasets.importers.synthetic import IMPORTER

        out = IMPORTER.default_display_name({"media_type": "audio", "size": "12"})
        assert "audio" in out
        assert "12" in out

    def test_server_folder_uses_leaf_path(self):
        from vtscore.datasets.importers.server_folder import IMPORTER

        assert IMPORTER.default_display_name({"path": "/data/sounds/sirens"}) == "sirens"
        assert IMPORTER.default_display_name({}) == IMPORTER.display_name

    def test_http_archive_strips_archive_extension(self):
        from vtscore.datasets.importers.http_archive import IMPORTER

        assert IMPORTER.default_display_name({"url": "https://example.org/data/genres.tar.gz"}) == "genres"
        assert IMPORTER.default_display_name({"url": "https://example.org/data/photos.zip"}) == "photos"
        # No URL → falls back to display_name
        assert IMPORTER.default_display_name({}) == IMPORTER.display_name

    def test_pickle_default_name_strips_extension(self):
        from vtscore.datasets.importers.pickle import IMPORTER

        assert IMPORTER.default_display_name({"file": "/tmp/genres.pkl"}) == "genres"

    def test_server_files_default_name_uses_paths_file_stem(self):
        from vtscore.datasets.importers.server_files import IMPORTER

        assert IMPORTER.default_display_name({"paths_file": "/tmp/audio_list.txt"}) == "audio_list"

    def test_origin_does_not_include_dataset_name(self):
        """The user-typed display name is UI metadata, not provenance -
        it must NOT leak into origin params (which feed Detector reload)."""
        from vtscore.datasets.importers.server_folder import IMPORTER

        origin = IMPORTER.build_origin({"path": "/data/x", "media_type": "audio", "dataset_name": "My Set"})
        assert "dataset_name" not in origin["params"]


# ---------------------------------------------------------------------------
# Folder importer not in builtin names
# ---------------------------------------------------------------------------


class TestBuiltinImporterNames:
    def test_folder_has_form_ui_mode(self):
        from vtscore.datasets.importers import get_importer

        imp = get_importer("server_folder")
        assert imp is not None
        assert imp.ui_mode == "form"

    def test_pickle_has_file_upload_ui_mode(self):
        from vtscore.datasets.importers import get_importer

        imp = get_importer("pickle")
        assert imp is not None
        assert imp.ui_mode == "file_upload"


# ---------------------------------------------------------------------------
# _extract_archive helper
# ---------------------------------------------------------------------------


class TestExtractArchive:
    def test_extract_zip(self, tmp_path):
        from vtscore.datasets.importers.http_archive import _extract_archive

        wav_data = _make_wav_bytes()
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("sounds/tone.wav", wav_data)
        extract_dir = tmp_path / "out"
        extract_dir.mkdir()
        _extract_archive(zip_path, extract_dir)
        assert (extract_dir / "sounds" / "tone.wav").exists()

    def test_extract_tar_uncompressed(self, tmp_path):
        from vtscore.datasets.importers.http_archive import _extract_archive

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
        from vtscore.datasets.importers.http_archive import _extract_archive

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
        from vtscore.datasets.importers.http_archive import _extract_archive

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
        from vtscore.datasets.importers.http_archive import _extract_archive

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

        from vtscore.datasets.importers.http_archive import _extract_archive

        rar_path = tmp_path / "test.rar"
        # Write RAR v4 magic bytes so it's identified as .rar by extension
        rar_path.write_bytes(b"Rar!\x1a\x07\x00" + b"\x00" * 20)
        extract_dir = tmp_path / "out"
        extract_dir.mkdir()

        with mock.patch.dict(sys.modules, {"rarfile": None}):
            with pytest.raises((RuntimeError, ImportError, Exception)):
                _extract_archive(rar_path, extract_dir)

    def test_zip_preserves_multiple_files(self, tmp_path):
        from vtscore.datasets.importers.http_archive import _extract_archive

        zip_path = tmp_path / "multi.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for i in range(3):
                zf.writestr(f"file{i}.wav", _make_wav_bytes())
        extract_dir = tmp_path / "out"
        extract_dir.mkdir()
        _extract_archive(zip_path, extract_dir)
        assert len(list(extract_dir.glob("*.wav"))) == 3

    def test_zip_traversal_rejected_before_extraction(self, tmp_path):
        """A zip member with ``..`` traversal must be rejected before any file lands on disk."""
        from vtscore.datasets.importers.http_archive import _extract_archive

        zip_path = tmp_path / "evil.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("../escape.wav", _make_wav_bytes())
        extract_dir = tmp_path / "out"
        extract_dir.mkdir()
        with pytest.raises(ValueError, match="traversal"):
            _extract_archive(zip_path, extract_dir)
        # Neither the in-tree nor the escaped target should exist.
        assert not (tmp_path / "escape.wav").exists()
        assert not (extract_dir / "escape.wav").exists()

    def test_zip_absolute_path_rejected(self, tmp_path):
        """A zip member with an absolute path must be rejected."""
        from vtscore.datasets.importers.http_archive import _extract_archive

        zip_path = tmp_path / "abs.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("/etc/escape.wav", _make_wav_bytes())
        extract_dir = tmp_path / "out"
        extract_dir.mkdir()
        with pytest.raises(ValueError, match="traversal"):
            _extract_archive(zip_path, extract_dir)

    def test_zip_prefix_collision_rejected(self, tmp_path):
        """``extract_dir`` prefix collision must NOT pass the traversal check.

        Regression: the previous string-prefix ``startswith`` check would
        accept ``../out_evil/x`` when extracting into ``.../out`` because
        ``str.startswith`` does not respect path separators.
        """
        from vtscore.datasets.importers.http_archive import _extract_archive

        zip_path = tmp_path / "prefix.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("../out_evil/escape.wav", _make_wav_bytes())
        extract_dir = tmp_path / "out"
        extract_dir.mkdir()
        with pytest.raises(ValueError, match="traversal"):
            _extract_archive(zip_path, extract_dir)
        assert not (tmp_path / "out_evil").exists()

    def test_tar_traversal_rejected_before_extraction(self, tmp_path):
        """A tar member with ``..`` traversal must be rejected before extraction."""
        from vtscore.datasets.importers.http_archive import _extract_archive

        tar_path = tmp_path / "evil.tar"
        wav_data = _make_wav_bytes()
        with tarfile.open(tar_path, "w") as tf:
            info = tarfile.TarInfo(name="../escape.wav")
            info.size = len(wav_data)
            tf.addfile(info, io.BytesIO(wav_data))
        extract_dir = tmp_path / "out"
        extract_dir.mkdir()
        with pytest.raises((ValueError, Exception), match="(?i)traversal|outside"):
            _extract_archive(tar_path, extract_dir)
        assert not (tmp_path / "escape.wav").exists()


# ---------------------------------------------------------------------------
# DatasetImporter bulk-record hooks (list_records / fetch_record /
# fetch_records_bulk).
# ---------------------------------------------------------------------------


class TestImporterBulkHooks:
    """Verify the per-record / bulk-record split mirrors the embedder pattern."""

    def _make_importer(self, *, fetch_one=None, fetch_bulk=None, records=None):
        from vtscore.datasets.importers.base import DatasetImporter

        class _BulkTestImporter(DatasetImporter):
            name = "bulk_test"
            display_name = "Bulk Test"
            description = "Test importer for bulk hooks."
            fields = []

            def list_records(self, field_values):
                return list(records or [])

            if fetch_one is not None:

                def fetch_record(self, record, field_values, thin=False):
                    assert fetch_one is not None  # narrowed by enclosing if
                    return fetch_one(record, field_values, thin)

            if fetch_bulk is not None:

                def _fetch_records_bulk_impl(self, recs, field_values, thin=False):
                    assert fetch_bulk is not None
                    return fetch_bulk(recs, field_values, thin)

        return _BulkTestImporter()

    def test_default_run_uses_list_and_fetch_record(self):
        records = [{"name": "a"}, {"name": "b"}, {"name": "c"}]

        def fetch_one(record, _fv, _thin):
            return {"media_type": "audio", "filename": record["name"], "embedding": None}

        imp = self._make_importer(records=records, fetch_one=fetch_one)
        medias: dict = {}
        imp.run({}, medias)

        assert list(medias.keys()) == [1, 2, 3]
        assert [medias[i]["filename"] for i in (1, 2, 3)] == ["a", "b", "c"]
        # ID is assigned by the framework even though fetch_record didn't set one.
        assert medias[1]["id"] == 1

    def test_default_run_fills_origin_from_build_origin(self):
        records = [{"name": "x"}]

        def fetch_one(record, _fv, _thin):
            # fetch_record may omit origin / origin_name - run() fills them in.
            return {"media_type": "audio", "filename": record["name"], "embedding": None}

        imp = self._make_importer(records=records, fetch_one=fetch_one)
        medias: dict = {}
        imp.run({"foo": "bar"}, medias)

        assert medias[1]["origin"] == imp.build_origin({"foo": "bar"})
        assert medias[1]["origin_name"] == "x"

    def test_default_run_skips_none_records(self):
        records = ["a", "skip", "b"]

        def fetch_one(record, _fv, _thin):
            if record == "skip":
                return None
            return {"media_type": "audio", "filename": record, "embedding": None}

        imp = self._make_importer(records=records, fetch_one=fetch_one)
        medias: dict = {}
        imp.run({}, medias)

        assert list(medias.keys()) == [1, 2]
        assert [medias[i]["filename"] for i in (1, 2)] == ["a", "b"]

    def test_default_bulk_impl_loops_fetch_record(self):
        # If a subclass implements fetch_record but NOT _fetch_records_bulk_impl,
        # the default bulk impl loops fetch_record once per record.
        seen: list = []

        def fetch_one(record, _fv, _thin):
            seen.append(record)
            return {"media_type": "audio", "filename": record, "embedding": None}

        imp = self._make_importer(records=["x", "y"], fetch_one=fetch_one)
        out = imp.fetch_records_bulk(["x", "y"], {})

        assert seen == ["x", "y"]
        assert all(m is not None for m in out)
        assert [m["filename"] for m in out if m is not None] == ["x", "y"]

    def test_bulk_override_used_when_implemented(self):
        # When a subclass overrides _fetch_records_bulk_impl, fetch_record must
        # NOT be called - the bulk path takes over completely.
        per_item_calls: list = []

        def fetch_one(record, _fv, _thin):
            per_item_calls.append(record)
            return {"media_type": "audio", "filename": record, "embedding": None}

        def fetch_bulk(records, _fv, _thin):
            # Pretend we did one batched call.
            return [{"media_type": "audio", "filename": f"bulk:{r}", "embedding": None} for r in records]

        imp = self._make_importer(records=["a", "b"], fetch_one=fetch_one, fetch_bulk=fetch_bulk)
        medias: dict = {}
        imp.run({}, medias)

        assert per_item_calls == []  # bulk override replaced the per-item loop
        assert [medias[i]["filename"] for i in (1, 2)] == ["bulk:a", "bulk:b"]

    def test_fetch_records_bulk_empty_returns_empty(self):
        imp = self._make_importer(records=[], fetch_one=lambda *a: None)
        assert imp.fetch_records_bulk([], {}) == []

    def test_run_raises_when_neither_run_nor_hooks_implemented(self):
        from vtscore.datasets.importers.base import DatasetImporter

        class _BareImporter(DatasetImporter):
            name = "bare"
            display_name = "Bare"
            description = ""
            fields = []

        imp = _BareImporter()
        with pytest.raises(NotImplementedError, match="list_records"):
            imp.run({}, {})


class TestReCallerMultiMedia:
    """ReCaller is the worked example of a multi-source-type service importer.

    Verifies ``fetch_source_media(spec, ...)`` filters records by
    ``spec.source_type`` and that the framework's default :meth:`run`
    drives converter calls so the importer doesn't need to.
    """

    def _stub_apis(self, monkeypatch, *, media_types=None):
        import numpy as np

        from vtscore.datasets.importers import recaller as rc

        if media_types is None:
            media_types = ["audio"] * 3
        results = [
            {
                "contentID": f"C{i}",
                "mediaID": f"M{i}",
                "media_url": f"http://pw/{i}",
                "media_type": mt,
                "md5": f"md5_{i}",
            }
            for i, mt in enumerate(media_types)
        ]
        monkeypatch.setattr(rc, "_rc_fetch_results", lambda _q: list(results))

        rng = np.random.default_rng(42)
        embeddings = {f"M{i}": rng.standard_normal(8).astype(np.float32) for i in range(len(results))}
        monkeypatch.setattr(
            rc,
            "_dw_get_embedding",
            lambda mid: {"embedding": embeddings[mid], "embedder": "fake-embedder"},
        )
        monkeypatch.setattr(rc, "_pw_fetch_media", lambda url: f"bytes-for-{url}".encode())
        return results

    def test_fetch_source_media_filters_by_source_type(self, monkeypatch):
        from vtscore.datasets.importers.base import SourceSpec
        from vtscore.datasets.importers.recaller import ReCallerDatasetImporter

        self._stub_apis(monkeypatch, media_types=["audio", "image", "audio"])
        imp = ReCallerDatasetImporter()

        audio_yields = list(
            imp.fetch_source_media(
                SourceSpec(source_type="audio", converter=None, params={}),
                {"query_id": "Q1", "media_type": "audio"},
                thin=True,
            )
        )
        assert [m["filename"] for m in audio_yields] == ["C0", "C2"]
        assert all(m["media_type"] == "audio" for m in audio_yields)

        image_yields = list(
            imp.fetch_source_media(
                SourceSpec(source_type="image", converter=None, params={}),
                {"query_id": "Q1", "media_type": "audio"},
                thin=True,
            )
        )
        assert [m["filename"] for m in image_yields] == ["C1"]
        assert image_yields[0]["media_type"] == "image"

    def test_fetch_source_media_requires_query_id(self, monkeypatch):
        from vtscore.datasets.importers.base import SourceSpec
        from vtscore.datasets.importers.recaller import ReCallerDatasetImporter

        self._stub_apis(monkeypatch)
        imp = ReCallerDatasetImporter()
        with pytest.raises(ValueError, match="query ID"):
            list(
                imp.fetch_source_media(
                    SourceSpec(source_type="audio", converter=None, params={}),
                    {"query_id": "", "media_type": "audio"},
                    thin=True,
                )
            )

    def test_default_run_ingests_direct_spec(self, monkeypatch):
        from vtscore.datasets.importers.recaller import ReCallerDatasetImporter

        self._stub_apis(monkeypatch)
        imp = ReCallerDatasetImporter()
        medias: dict = {}
        imp.run({"query_id": "Q1", "media_type": "audio"}, medias, thin=True)

        assert list(medias.keys()) == [1, 2, 3]
        for i in (1, 2, 3):
            assert medias[i]["origin"]["params"]["contentID"] == f"C{i - 1}"
            assert medias[i]["origin"]["params"]["mediaID"] == f"M{i - 1}"

    def test_default_run_drives_converter_per_spec(self, monkeypatch):
        """Framework calls converter.convert(); importer never does."""
        import json

        from vtscore.datasets.importers.recaller import ReCallerDatasetImporter

        # Mix of audio (direct) and video (must go through video2image).
        self._stub_apis(monkeypatch, media_types=["image", "video", "image"])

        # Stub video2image to return a deterministic 2-frame expansion.
        from vtscore.converters import get_converter

        v2i = get_converter("video2image")
        assert v2i is not None
        observed_params: list[dict] = []

        def fake_convert(media, params):
            observed_params.append(dict(params))
            return [
                {
                    "media_type": "image",
                    "filename": f"{media['filename']}_frame_{k}.png",
                    "media_bytes": None,
                    "media_path": None,
                    "media_url": media.get("media_url"),
                    "embedding": media["embedding"],
                    "embedder": media["embedder"],
                    "md5": f"{media['md5']}_{k}",
                    "duration": 0,
                    "category": "",
                    "file_size": 0,
                }
                for k in range(2)
            ]

        monkeypatch.setattr(v2i, "convert", fake_convert)

        imp = ReCallerDatasetImporter()
        medias: dict = {}
        source_specs = json.dumps(
            [
                {"source_type": "image", "converter": None, "params": {}},
                {"source_type": "video", "converter": "video2image", "params": {"n_clips": "2"}},
            ]
        )
        imp.run(
            {"query_id": "Q1", "media_type": "image", "source_specs": source_specs},
            medias,
            thin=True,
        )

        # Two image records (direct) + one video record × 2 frames = 4 medias.
        assert len(medias) == 4
        types = sorted(m["media_type"] for m in medias.values())
        assert types == ["image", "image", "image", "image"]
        # The framework, not the importer, called video2image.convert with
        # the spec's params.  The framework's ``convert_normalized`` pass
        # (Phase C #9) validates params through the converter's declared
        # schema, so ``n_clips`` arrives at ``convert`` as the coerced
        # ``int`` rather than the raw string the spec carried.
        assert observed_params == [{"n_clips": 2}]


# ---------------------------------------------------------------------------
# load_dataset_from_folder – content_vectors support
# ---------------------------------------------------------------------------
