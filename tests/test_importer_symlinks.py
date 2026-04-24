"""Importer symlink tests.

Tests for symlinked importer discovery, rglob following symlinks,
and symlinked folder import.
"""

from __future__ import annotations


from helpers import make_raw_wav_bytes as _make_wav_bytes


class TestSymlinkedImporterDiscovery:
    """PluginRegistry should discover importers in symlinked directories."""

    def test_symlinked_package_is_discovered(self, tmp_path):
        """A symlink inside the importers directory pointing to a valid
        package should be discovered exactly like a regular directory."""
        import os
        import sys

        from vtsearch.utils.registry import PluginRegistry

        # Create a minimal importer package outside the importers tree.
        ext_pkg = tmp_path / "my_custom_importer"
        ext_pkg.mkdir()
        (ext_pkg / "__init__.py").write_text(
            "from vtsearch.datasets.importers.base import DatasetImporter\n"
            "from vtsearch.utils.registry import PluginField\n"
            "\n"
            "class _Imp(DatasetImporter):\n"
            '    name = "symlink_test_imp"\n'
            '    display_name = "Symlink Test"\n'
            '    description = "Test importer via symlink"\n'
            "    fields = [PluginField(key='path', label='Path', field_type='folder')]\n"
            "    def run(self, field_values, medias): return []\n"
            "\n"
            "IMPORTER = _Imp()\n"
        )

        # Symlink it into the real importers directory.
        import importlib

        parent = importlib.import_module("vtsearch.datasets.importers")
        pkg_dir = os.path.dirname(parent.__file__)
        link = os.path.join(pkg_dir, "symlink_test_pkg")
        os.symlink(str(ext_pkg), link)

        try:
            # Build a fresh registry so discovery runs from scratch.
            reg = PluginRegistry(
                package="vtsearch.datasets.importers",
                sentinel="IMPORTER",
                label="dataset importer",
            )
            names = [p.name for p in reg.list()]
            assert "symlink_test_imp" in names
        finally:
            os.unlink(link)
            # Clean up cached module so other tests are unaffected.
            for key in list(sys.modules):
                if "symlink_test_pkg" in key:
                    del sys.modules[key]

    def test_dotted_symlink_name_is_skipped(self, tmp_path):
        """A symlink whose name contains a dot (e.g. 'foo.symbolic_link')
        should be silently skipped — dots in directory names break importlib
        module path construction."""
        import os
        import sys

        from vtsearch.utils.registry import PluginRegistry

        # Create a valid importer package.
        ext_pkg = tmp_path / "dotted_imp"
        ext_pkg.mkdir()
        (ext_pkg / "__init__.py").write_text(
            "from vtsearch.datasets.importers.base import DatasetImporter\n"
            "from vtsearch.utils.registry import PluginField\n"
            "\n"
            "class _Imp(DatasetImporter):\n"
            '    name = "dotted_symlink_test"\n'
            '    display_name = "Dotted Symlink Test"\n'
            '    description = "Should not be loaded"\n'
            "    fields = [PluginField(key='path', label='Path', field_type='folder')]\n"
            "    def run(self, field_values, medias): return []\n"
            "\n"
            "IMPORTER = _Imp()\n"
        )

        # Symlink with a dotted name into the importers directory.
        import importlib

        parent = importlib.import_module("vtsearch.datasets.importers")
        pkg_dir = os.path.dirname(parent.__file__)
        link = os.path.join(pkg_dir, "dx_uuid.symbolic_link")
        os.symlink(str(ext_pkg), link)

        try:
            reg = PluginRegistry(
                package="vtsearch.datasets.importers",
                sentinel="IMPORTER",
                label="dataset importer",
            )
            names = [p.name for p in reg.list()]
            # The dotted-name symlink must NOT appear.
            assert "dotted_symlink_test" not in names
        finally:
            os.unlink(link)
            for key in list(sys.modules):
                if "dx_uuid" in key:
                    del sys.modules[key]

    def test_symlinked_package_sets_module_attributes(self, tmp_path):
        """A symlinked package should have correct __name__, __package__,
        and __path__ attributes after being loaded via spec_from_file_location."""
        import os
        import sys

        from vtsearch.utils.registry import PluginRegistry

        ext_pkg = tmp_path / "attr_check_importer"
        ext_pkg.mkdir()
        (ext_pkg / "__init__.py").write_text(
            "from vtsearch.datasets.importers.base import DatasetImporter\n"
            "from vtsearch.utils.registry import PluginField\n"
            "\n"
            "class _Imp(DatasetImporter):\n"
            '    name = "attr_check_imp"\n'
            '    display_name = "Attr Check"\n'
            '    description = "Test module attributes"\n'
            "    fields = [PluginField(key='path', label='Path', field_type='folder')]\n"
            "    def run(self, field_values, medias): return []\n"
            "\n"
            "IMPORTER = _Imp()\n"
        )

        import importlib

        parent = importlib.import_module("vtsearch.datasets.importers")
        pkg_dir = os.path.dirname(parent.__file__)
        link = os.path.join(pkg_dir, "attr_check_pkg")
        os.symlink(str(ext_pkg), link)

        try:
            reg = PluginRegistry(
                package="vtsearch.datasets.importers",
                sentinel="IMPORTER",
                label="dataset importer",
            )
            names = [p.name for p in reg.list()]
            assert "attr_check_imp" in names

            mod = sys.modules["vtsearch.datasets.importers.attr_check_pkg"]
            assert mod.__name__ == "vtsearch.datasets.importers.attr_check_pkg"
            # __path__ should be set (it's a package)
            assert hasattr(mod, "__path__")
            assert len(mod.__path__) == 1
        finally:
            os.unlink(link)
            for key in list(sys.modules):
                if "attr_check_pkg" in key:
                    del sys.modules[key]

    def test_symlinked_flat_module_is_discovered(self, tmp_path):
        """A symlink to a .py file should be discovered when
        discover_modules=True."""
        import os
        import sys

        from vtsearch.utils.registry import PluginRegistry

        # Create an external .py module with a sentinel.
        ext_module = tmp_path / "my_flat_source.py"
        ext_module.write_text('class _Src:\n    name = "symlinked_flat_mod"\n\nFAKE_SENTINEL = _Src()\n')

        # We need a real package dir with __init__.py for the registry.
        pkg_dir = tmp_path / "fake_pkg"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")

        # Symlink the .py file into the package.
        link = pkg_dir / "linked_source.py"
        os.symlink(str(ext_module), str(link))

        # Register the fake package so importlib can find it.
        sys.modules["_test_flat_pkg"] = type(sys)("_test_flat_pkg")
        sys.modules["_test_flat_pkg"].__file__ = str(pkg_dir / "__init__.py")
        sys.modules["_test_flat_pkg"].__path__ = [str(pkg_dir)]
        sys.modules["_test_flat_pkg"].__package__ = "_test_flat_pkg"

        try:
            reg = PluginRegistry(
                package="_test_flat_pkg",
                sentinel="FAKE_SENTINEL",
                label="test source",
                discover_modules=True,
            )
            names = [p.name for p in reg.list()]
            assert "symlinked_flat_mod" in names
        finally:
            for key in list(sys.modules):
                if "_test_flat_pkg" in key:
                    del sys.modules[key]


class TestRglobFollowSymlinks:
    """rglob_follow_symlinks should descend into symlinked directories."""

    def test_finds_files_through_symlinked_directory(self, tmp_path):
        from vtsearch.utils.paths import rglob_follow_symlinks

        root = tmp_path / "root"
        root.mkdir()
        (root / "a.wav").write_bytes(b"a")

        external = tmp_path / "external"
        external.mkdir()
        (external / "b.wav").write_bytes(b"b")

        (root / "linked").symlink_to(external)

        results = rglob_follow_symlinks(root, "*.wav")
        names = {p.name for p in results}
        assert "a.wav" in names
        assert "b.wav" in names

    def test_no_matches_returns_empty(self, tmp_path):
        from vtsearch.utils.paths import rglob_follow_symlinks

        root = tmp_path / "empty"
        root.mkdir()
        assert rglob_follow_symlinks(root, "*.wav") == []


class TestSymlinkedFolderImport:
    """load_dataset_from_folder must discover files inside symlinked subdirs."""

    def _write_wav(self, path):
        path.write_bytes(_make_wav_bytes())

    def _make_fake_media_type(self):
        import unittest.mock as mock

        import numpy as np

        mt = mock.MagicMock()
        mt.type_id = "audio"
        mt.file_extensions = ["*.wav"]
        mt.embed_media.return_value = np.zeros(3)
        mt.load_media_data.return_value = {"duration": 1.0}
        mt._mock_embedder = mock.MagicMock()
        mt._mock_embedder.name = "clap"
        mt._mock_embedder.media_type_id = "audio"
        mt._mock_embedder._model = True
        mt._mock_embedder.embed_media.return_value = np.zeros(3)
        return mt

    def _patch_media_registry(self, mt):
        import unittest.mock as mock
        from contextlib import ExitStack

        stack = ExitStack()
        stack.enter_context(mock.patch("vtsearch.media.get_by_folder_name", return_value=mt))
        stack.enter_context(mock.patch("vtsearch.media.embedders_for_type", return_value=[mt._mock_embedder]))
        return stack

    def test_load_dataset_from_folder_follows_symlinks(self, tmp_path):
        from vtsearch.datasets.loader import load_dataset_from_folder

        root = tmp_path / "root"
        root.mkdir()
        self._write_wav(root / "a.wav")

        external = tmp_path / "external"
        external.mkdir()
        self._write_wav(external / "b.wav")

        (root / "linked").symlink_to(external)

        mt = self._make_fake_media_type()
        medias: dict = {}
        with self._patch_media_registry(mt):
            load_dataset_from_folder(root, "audio", medias, on_progress=lambda *a: None)

        filenames = {m["filename"] for m in medias.values()}
        assert "a.wav" in filenames
        assert "linked/b.wav" in filenames

    def test_load_dataset_from_folder_chunked_follows_symlinks(self, tmp_path):
        from vtsearch.datasets.loader import load_dataset_from_folder_chunked

        root = tmp_path / "root"
        root.mkdir()
        self._write_wav(root / "a.wav")

        external = tmp_path / "external"
        external.mkdir()
        self._write_wav(external / "b.wav")

        (root / "linked").symlink_to(external)

        mt = self._make_fake_media_type()
        with self._patch_media_registry(mt):
            chunks = list(
                load_dataset_from_folder_chunked(
                    root,
                    "audio",
                    chunk_size=10,
                    on_progress=lambda *a: None,
                )
            )

        all_medias = {}
        for chunk in chunks:
            all_medias.update(chunk)

        filenames = {m["filename"] for m in all_medias.values()}
        assert "a.wav" in filenames
        assert "linked/b.wav" in filenames
