"""Tests for name-collision handling in :class:`PluginRegistry`.

When two sub-packages (or flat modules, with ``discover_modules=True``)
under the scanned package each declare a sentinel with the same
``.name``, the registry must warn and keep the first discovered plugin
rather than silently overwriting it.
"""

from __future__ import annotations

import os
import sys
import warnings

from vtscore.plugins import PluginRegistry


class TestSubPackageNameCollision:
    """Two sub-packages declaring the same ``.name`` should not silently
    shadow each other."""

    def test_collision_warns_and_keeps_first(self, tmp_path):
        # Create two external importer packages that both claim the same
        # plugin name. They land in the real importers directory via
        # symlinks so PluginRegistry's directory scan picks them up.
        importer_src = (
            "from vtscore.datasets.importers.base import DatasetImporter\n"
            "from vtscore.plugins import PluginField\n"
            "\n"
            "class _Imp(DatasetImporter):\n"
            '    name = "collision_test_imp"\n'
            "    display_name = {display!r}\n"
            "    description = {desc!r}\n"
            "    fields = [PluginField(key='path', label='Path', field_type='folder')]\n"
            "    def run(self, field_values, medias): return []\n"
            "\n"
            "IMPORTER = _Imp()\n"
        )

        pkg_a = tmp_path / "collision_pkg_a"
        pkg_a.mkdir()
        (pkg_a / "__init__.py").write_text(importer_src.format(display="First", desc="First registration"))

        pkg_b = tmp_path / "collision_pkg_b"
        pkg_b.mkdir()
        (pkg_b / "__init__.py").write_text(importer_src.format(display="Second", desc="Should be skipped"))

        import importlib

        parent = importlib.import_module("vtscore.datasets.importers")
        assert parent.__file__ is not None
        pkg_dir = os.path.dirname(parent.__file__)
        # Names are deliberately ordered so the alphabetical scan visits
        # zz_collision_a first, then zz_collision_b second.
        link_a = os.path.join(pkg_dir, "zz_collision_a")
        link_b = os.path.join(pkg_dir, "zz_collision_b")
        os.symlink(str(pkg_a), link_a)
        os.symlink(str(pkg_b), link_b)

        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                reg = PluginRegistry(
                    package="vtscore.datasets.importers",
                    sentinel="IMPORTER",
                    label="dataset importer",
                )
                plugin = reg.get("collision_test_imp")

            # The first plugin discovered wins.
            assert plugin is not None
            assert plugin.display_name == "First"

            # The second registration emits a collision warning that
            # names the offending module.
            collision_warnings = [str(w.message) for w in caught if "already registered" in str(w.message)]
            assert collision_warnings, "expected a collision warning"
            assert any("zz_collision_b" in msg for msg in collision_warnings)
        finally:
            os.unlink(link_a)
            os.unlink(link_b)
            for key in list(sys.modules):
                if "zz_collision_" in key:
                    del sys.modules[key]


class TestFlatModuleNameCollision:
    """``discover_modules=True`` registries must also catch collisions
    between flat ``.py`` modules."""

    def test_collision_warns_and_keeps_first(self, tmp_path):
        # Build a fresh package on disk so we can populate it with two
        # flat modules whose sentinels share a name.
        pkg_root = tmp_path / "_collision_flat_pkg"
        pkg_root.mkdir()
        (pkg_root / "__init__.py").write_text("")
        (pkg_root / "alpha.py").write_text('class _S:\n    name = "shared_name"\n    label = "alpha"\n\nFAKE = _S()\n')
        (pkg_root / "beta.py").write_text('class _S:\n    name = "shared_name"\n    label = "beta"\n\nFAKE = _S()\n')

        sys.path.insert(0, str(tmp_path))
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                reg = PluginRegistry(
                    package="_collision_flat_pkg",
                    sentinel="FAKE",
                    label="test plugin",
                    discover_modules=True,
                )
                plugin = reg.get("shared_name")

            # Alphabetical scan visits alpha.py first, so it wins.
            assert plugin is not None
            assert plugin.label == "alpha"

            collision_warnings = [str(w.message) for w in caught if "already registered" in str(w.message)]
            assert collision_warnings, "expected a collision warning"
            assert any("beta" in msg for msg in collision_warnings)
        finally:
            sys.path.remove(str(tmp_path))
            for key in list(sys.modules):
                if key == "_collision_flat_pkg" or key.startswith("_collision_flat_pkg."):
                    del sys.modules[key]
