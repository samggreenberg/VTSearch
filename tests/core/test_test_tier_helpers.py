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
