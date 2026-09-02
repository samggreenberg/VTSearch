"""Structural gate over the duplicated per-tier test support files.

`tests/` (app tier) and `tests_lib/` (library tier) each carry their own copy
of `helpers.py` and `fixtures/medias.py` so that either tree can be collected
on its own — `./run-tests.sh vtscore-clean` runs `tests_lib/` alone, under an
import hook that bans Flask.

That self-containment used to be fictional: `pythonpath = ["tests",
"tests_lib"]` put both directories on `sys.path`, pytest inserts those entries
in reverse order, and so a bare `from helpers import ...` resolved to
`tests/helpers.py` for tests in *both* trees.  `tests_lib/helpers.py` was never
imported, and the two copies drifted without anything noticing.  The imports
are now package-qualified (`from tests.helpers import ...` /
`from tests_lib.helpers import ...`), which fixes the resolution; these tests
keep the copies honest afterwards.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

# (relative path within each tier) — files duplicated between the two trees.
_DUPLICATED = ["helpers.py", "fixtures/medias.py"]


class TestTierCopiesAgree:
    """The duplicated support files must stay byte-identical across tiers."""

    @pytest.mark.parametrize("relpath", _DUPLICATED)
    def test_copies_are_identical(self, relpath):
        app_tier = _REPO_ROOT / "tests" / relpath
        lib_tier = _REPO_ROOT / "tests_lib" / relpath
        assert app_tier.exists(), f"missing app-tier copy: {app_tier}"
        assert lib_tier.exists(), f"missing library-tier copy: {lib_tier}"
        assert app_tier.read_text() == lib_tier.read_text(), (
            f"tests/{relpath} and tests_lib/{relpath} have drifted.  They are "
            "intentional duplicates so each tier is self-contained; apply the "
            "edit to both copies.  If the change genuinely belongs to one tier "
            "only, give it a distinct name rather than letting the shared file "
            "diverge."
        )


class TestNoBareHelperImports:
    """Test modules must import their own tier's helpers, not a bare `helpers`.

    A bare ``import helpers`` only resolves when a tier directory is on
    ``sys.path``, which is exactly the ambiguity this gate exists to prevent.
    """

    @pytest.mark.parametrize("tier", ["tests", "tests_lib"])
    def test_helpers_imports_are_tier_qualified(self, tier):
        offenders = []
        for path in sorted((_REPO_ROOT / tier).rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level == 0:
                    module = node.module or ""
                    if module == "helpers" or module.startswith("helpers."):
                        offenders.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "helpers" or alias.name.startswith("helpers."):
                            offenders.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}")
        assert not offenders, (
            "bare `helpers` imports found (they resolve to whichever tier "
            f"happens to be on sys.path first); use `from {tier}.helpers import "
            "...` instead:\n  " + "\n  ".join(offenders)
        )


# Names that used to be copy-pasted into both conftests and drifted (#3424):
# the library tier's fake audio embedder fell back to a ``PYTHONHASHSEED``-salted
# ``hash()``, and its reset fixture never dropped the detector-file mtime cache.
# They now live in ``tests_shared`` and are imported by both.
_SHARED_CONFTEST_NAMES = (
    "fake_embed_audio",
    "fake_embed_text",
    "allow_test_tmp_paths",
    "reset_shared_state",
    "install_startup_contexts",
    "pin_training_budget",
    "freeze_startup_heap",
    "add_group_markers",
    "print_summary_and_exit",
)


class TestSharedConftestIsTheSingleSource:
    """Neither conftest may re-define what ``tests_shared`` owns.

    Unlike ``helpers.py`` — duplicated on purpose, and gated byte-identical
    above — the conftests' shared machinery is genuinely single-sourced.  A
    local re-definition would shadow the import and silently re-open the drift.
    """

    @pytest.mark.parametrize("tier", ["tests", "tests_lib"])
    def test_conftest_defines_no_shared_name(self, tier):
        path = _REPO_ROOT / tier / "conftest.py"
        tree = ast.parse(path.read_text(), filename=str(path))
        offenders = sorted(
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            # The copies spelled these with a leading underscore; catch both.
            and node.name.lstrip("_") in _SHARED_CONFTEST_NAMES
        )
        assert not offenders, (
            f"{tier}/conftest.py defines {offenders}, which tests_shared owns.  "
            "Import from tests_shared instead — a local copy is how the two "
            "conftests drifted in the first place (issue #3424)."
        )

    @pytest.mark.parametrize("tier", ["tests", "tests_lib"])
    def test_conftest_imports_the_shared_package(self, tier):
        path = _REPO_ROOT / tier / "conftest.py"
        tree = ast.parse(path.read_text(), filename=str(path))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("tests_shared")
        }
        assert imported, f"{tier}/conftest.py no longer imports tests_shared"


class TestSharedPackageIsLibrarySafe:
    """``tests_shared`` is imported by ``tests_lib/`` under the Flask blocker.

    ``./run-tests.sh vtscore-clean`` bans ``flask`` / ``werkzeug`` /
    ``flask_smorest`` from the library-tier session, and ``tests_lib/`` is meant
    to stay clear of the app tier generally.  A ``vtsearch`` import added here
    would break that for both — mutable app-tier objects (the ``medias`` map)
    are passed in by the caller instead.
    """

    _BANNED_PREFIXES = ("flask", "werkzeug", "flask_smorest", "vtsearch")

    def test_no_app_tier_imports(self):
        offenders = []
        for path in sorted((_REPO_ROOT / "tests_shared").rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [node.module or ""]
                elif isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                else:
                    continue
                for name in names:
                    root = name.split(".")[0]
                    if root in self._BANNED_PREFIXES:
                        offenders.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno} imports {name}")
        assert not offenders, (
            "tests_shared/ must stay importable by the library tier under the "
            "Flask blocker; pass app-tier objects in as arguments instead:\n  " + "\n  ".join(offenders)
        )
