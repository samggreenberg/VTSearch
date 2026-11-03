"""`tests_lib/` mirrors `vtscore/`, and every folder has a group (#5403).

A library test lives in the folder of the `vtscore` package it covers
(`tests_lib/eval/` for `vtscore/eval/`, the tree's root for `vtscore`'s
top-level modules), so "where is the test for X" has one answer.  Folders and
`./run-tests.sh` groups therefore no longer coincide, and
`tests_shared.pytest_plumbing.LIB_GROUPS` maps one onto the other.

These tests keep that map honest from both ends: no folder of tests without a
group, no group for a folder that is gone, no group name pytest does not know,
and no folder that names a package `vtscore` does not have.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests_shared.pytest_plumbing import LIB_GROUPS, NON_GROUP_DIRS, add_group_markers

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LIB = _REPO_ROOT / "tests_lib"

#: Folders whose subject is not a `vtscore` package: the repo's own tooling,
#: the CUDA-only tests, and the end-to-end workflows.
_NOT_PACKAGES = frozenset({"meta", "gpu", "integration"})


def _test_folders() -> set[str]:
    """Every folder under ``tests_lib/`` holding a test file, as a path under it."""
    folders = set()
    for path in _LIB.rglob("test_*.py"):
        parts = path.parent.relative_to(_LIB).parts
        if parts and (parts[0] in NON_GROUP_DIRS or parts[0].startswith("_")):
            continue
        folders.add("/".join(parts))
    return folders


def _registered_markers() -> set[str]:
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return {entry.split(":", 1)[0].strip() for entry in pyproject["tool"]["pytest"]["ini_options"]["markers"]}


class TestGroupMap:
    def test_every_test_folder_has_a_group(self):
        missing = sorted(_test_folders() - set(LIB_GROUPS))
        assert not missing, (
            f"tests_lib folders with no group in LIB_GROUPS: {missing}. Move the tests into "
            "the folder of the vtscore package they cover, or map the folder in "
            "tests_shared/pytest_plumbing.py."
        )

    def test_every_group_entry_has_tests(self):
        stale = sorted(set(LIB_GROUPS) - _test_folders())
        assert not stale, f"LIB_GROUPS maps folders that hold no tests: {stale}"

    def test_every_group_is_a_registered_marker(self):
        unknown = sorted(set(LIB_GROUPS.values()) - _registered_markers())
        assert not unknown, f"LIB_GROUPS names groups pyproject.toml does not register: {unknown}"

    def test_every_test_folder_is_a_package(self):
        # Without an __init__.py, two folders' same-named test files collide
        # under pytest's prepend import mode.
        missing = sorted(f for f in LIB_GROUPS if f and not (_LIB / f / "__init__.py").exists())
        assert not missing, f"tests_lib folders without an __init__.py: {missing}"


class TestMirrorsTheSource:
    def test_each_folder_names_a_source_package(self):
        def source(folder: str) -> Path:
            if folder == "imagekit":
                return _REPO_ROOT / "imagekit"
            return _REPO_ROOT / "vtscore" / folder

        orphans = sorted(f for f in LIB_GROUPS if f not in _NOT_PACKAGES and not any(source(f).glob("*.py")))
        assert not orphans, f"tests_lib folders that mirror no source package: {orphans}"


def _item(rootpath: Path, relpath: str):
    marks = []
    item = SimpleNamespace(
        path=rootpath / relpath,
        config=SimpleNamespace(rootpath=rootpath),
        add_marker=lambda mark: marks.append(mark.name),
    )
    return item, marks


class TestAddGroupMarkers:
    def test_a_mapped_folder_gets_its_group(self, tmp_path):
        item, marks = _item(tmp_path, "tests_lib/eval/test_x.py")
        add_group_markers([item], "tests_lib", groups={"eval": "detectors"})
        assert marks == ["detectors"]

    def test_the_tree_root_is_a_folder_like_any_other(self, tmp_path):
        item, marks = _item(tmp_path, "tests_lib/test_x.py")
        add_group_markers([item], "tests_lib", groups={"": "core"})
        assert marks == ["core"]

    def test_an_unmapped_folder_stops_the_run(self, tmp_path):
        item, _ = _item(tmp_path, "tests_lib/brand_new/test_x.py")
        with pytest.raises(pytest.UsageError, match="tests_lib/brand_new/"):
            add_group_markers([item], "tests_lib", groups={"eval": "detectors"})

    def test_fixtures_get_no_group(self, tmp_path):
        item, marks = _item(tmp_path, "tests_lib/fixtures/test_x.py")
        add_group_markers([item], "tests_lib", groups={})
        assert marks == []

    def test_the_other_trees_items_are_left_alone(self, tmp_path):
        # Both conftests' hooks see every item; each marks only its own tree.
        app, app_marks = _item(tmp_path, "tests/core/test_x.py")
        lib, lib_marks = _item(tmp_path, "tests_lib/eval/test_x.py")
        add_group_markers([app, lib], "tests_lib", groups={"eval": "detectors"})
        add_group_markers([app, lib], "tests")
        assert app_marks == ["core"]
        assert lib_marks == ["detectors"]
