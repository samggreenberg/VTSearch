"""The ``vtscore.config`` package keeps the surface its single-module form had.

Three invariants, all of which a seventh submodule could break silently:

* every public name the module exposed is still reachable as ``vtscore.config.X``;
* ``_RELOAD_ORDER`` names every submodule, so :func:`vtscore.config._reload_all`
  cannot quietly stop re-reading one file's environment variables;
* the order is a real dependency order, so reloading in it never leaves a
  submodule bound to a stale copy of one it reads from.

The first is what makes the split non-breaking for out-of-tree importers; the
other two are what keep the env-var tests honest.
"""

from __future__ import annotations

import ast
from pathlib import Path

import vtscore.config as config

PACKAGE_DIR = Path(config.__file__).parent


def _submodule_files() -> set[str]:
    return {p.stem for p in PACKAGE_DIR.glob("*.py") if p.stem != "__init__"}


def test_reload_order_names_every_submodule():
    assert set(config._RELOAD_ORDER) == _submodule_files()


def test_reload_order_has_no_duplicates():
    assert len(config._RELOAD_ORDER) == len(set(config._RELOAD_ORDER))


def test_reload_order_is_a_dependency_order():
    """A submodule may only import from ones already reloaded before it."""
    position = {name: i for i, name in enumerate(config._RELOAD_ORDER)}
    for name, index in position.items():
        tree = ast.parse((PACKAGE_DIR / f"{name}.py").read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if node.module == "vtscore.config":
                # ``from vtscore.config import <submodule>`` - the alias form
                # used to keep one patch target; still a dependency.
                deps = [a.name for a in node.names if a.name in position]
            elif node.module.startswith("vtscore.config."):
                deps = [node.module.removeprefix("vtscore.config.")]
            else:
                continue
            for dep in deps:
                assert position[dep] < index, f"{name} imports {dep}, which reloads after it"


def test_every_public_name_resolves_on_the_package():
    """``__all__`` is the pre-split public surface; each entry must be reachable."""
    for name in config.__all__:
        assert hasattr(config, name), name


def test_public_submodule_names_are_all_re_exported():
    """Nothing public may live only on a submodule: ``vtscore.config.X`` is the
    documented import path, and an out-of-tree caller uses it."""
    exported = set(config.__all__)
    for name in config._RELOAD_ORDER:
        tree = ast.parse((PACKAGE_DIR / f"{name}.py").read_text())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef | ast.ClassDef):
                bound = [node.name]
            elif isinstance(node, ast.Assign):
                bound = [t.id for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                bound = [node.target.id]
            else:
                continue
            for symbol in bound:
                if not symbol.startswith("_"):
                    assert symbol in exported, f"{name}.{symbol} is public but not re-exported"


def test_private_names_are_not_re_exported():
    """A copy of a private name on the package would be a stub target that
    silently does nothing - the submodule global is what its readers resolve."""
    for name in ("_cuda_can_run", "_cuda_runnable", "_core_config_builder", "_transformers_major"):
        assert not hasattr(config, name), name
