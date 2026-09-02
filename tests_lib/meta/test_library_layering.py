"""The library tier must not import the app tier.

``vtscore`` is meant to be usable without Flask - and without
:mod:`vtsearch` at all.  Every dependency runs one way: ``vtsearch``
imports ``vtscore``, never the reverse (see
``vtscore/docs/architecture.md``).  Lazy function-level imports are the
easy way to break that silently: the module still imports cleanly, and
the inverted dependency only bites at call time, in whatever deployment
lacks Flask.  Issue #2931 was exactly that shape -
``JobManager.start()`` raised ``ImportError`` under a Flask-free
environment because it reached into ``vtsearch.auth``.

This test scans the AST of every ``vtscore`` module for imports of
``vtsearch``, at any nesting depth, so a new one has to be argued for
rather than merged unnoticed.

The same scan runs over ``tests_lib/`` itself.  ``tests_lib/__init__.py``
promises the library-tier suite never reaches into an app-tier module, and
nothing used to check it: ``conftest.py`` and ``fixtures/medias.py`` both
imported ``vtsearch.state`` (issue #3421).  That is not a cosmetic slip -
``vtsearch.state``'s ``medias`` is a *proxy*, so the library tier was
depending on app-tier laziness to get the per-test context reset right, and
under ``./run-tests.sh vtscore-clean`` the promise the gate advertises was
simply false.  Note the asymmetry with the Flask blocker: that hook bans
``flask``/``werkzeug``/``flask_smorest`` at import time, but ``vtsearch``
itself has to stay importable (the library lives inside the same
distribution), so only a static scan can see these.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VTSCORE = _REPO_ROOT / "vtscore"
_TESTS_LIB = _REPO_ROOT / "tests_lib"

#: Imports of ``vtsearch`` that are allowed to remain, with the reason.
#: Both are *optional* - wrapped in ``try``/``except`` so the library
#: keeps working when the app package is absent, which is what makes them
#: tolerable.  Anything else fails this test.
_ALLOWED_VTSCORE: dict[str, str] = {
    "exporters/portable_detector/__init__.py": (
        "guarded `import vtsearch` used only to stamp the producing app's version "
        "into a bundle; falls back to a literal when the app package is absent"
    ),
    "embedding/loader.py": "optional transformers logging bridge, wrapped in try/except Exception",
}


#: Imports of ``vtsearch`` under ``tests_lib/`` that are allowed to remain,
#: with the reason.  Empty on purpose: unlike the library's two entries below,
#: no test *needs* the app tier, and the one that looked like it did (a patch of
#: ``vtsearch.logging_config.install_transformers_logging_bridge``) was really a
#: missing seam in ``vtscore.embedding.loader``, now filled by
#: ``_install_transformers_logging_bridge``.  Reach for that fix first; an entry
#: here means the library has no seam and nobody added one.
_ALLOWED_TESTS_LIB: dict[str, str] = {}


def _vtsearch_imports(path: Path) -> list[tuple[int, str]]:
    """Return ``(lineno, module)`` for every ``vtsearch`` import in *path*."""
    found: list[tuple[int, str]] = []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "vtsearch":
                    found.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            # ``node.level > 0`` is a relative import, which can never
            # escape the vtscore package.
            if node.level == 0 and node.module and node.module.split(".")[0] == "vtsearch":
                found.append((node.lineno, node.module))
    return found


def _patch_targets(path: Path) -> list[tuple[int, str]]:
    """Return ``(lineno, target)`` for every ``patch("dotted.name")`` in *path*.

    Only the first positional argument of a call spelled ``patch`` /
    ``mock.patch`` / ``patch.object`` counts.  Matching every string literal
    that merely contains "vtsearch" would flag this file's own error messages
    and any docstring that names the package.
    """
    found: list[tuple[int, str]] = []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name not in {"patch", "object", "dict"}:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            found.append((node.lineno, first.value))
    return found


class TestVtscoreDoesNotImportVtsearch:
    def test_no_unlisted_vtsearch_imports(self):
        offenders: list[str] = []
        for path in sorted(_VTSCORE.rglob("*.py")):
            rel = path.relative_to(_VTSCORE).as_posix()
            if rel in _ALLOWED_VTSCORE:
                continue
            for lineno, module in _vtsearch_imports(path):
                offenders.append(f"vtscore/{rel}:{lineno} imports {module}")

        assert not offenders, (
            "vtscore must not import vtsearch (the dependency runs the other way).\n"
            + "\n".join(offenders)
            + "\n\nRoute app-tier behaviour through a registered hook instead - see "
            "vtscore/state/current_user.py and vtscore/achievements_hooks.py for the pattern."
        )

    def test_allowlist_has_no_stale_entries(self):
        """Every allowlisted file must still contain a ``vtsearch`` import.

        Otherwise the allowlist quietly grows into a list of exemptions
        nobody has re-earned.
        """
        stale = [rel for rel in _ALLOWED_VTSCORE if not _vtsearch_imports(_VTSCORE / rel)]
        assert not stale, f"Allowlisted files no longer import vtsearch; drop them from _ALLOWED_VTSCORE: {stale}"


class TestJobManagerHasNoAppTierImport:
    def test_async_jobs_resolves_the_user_through_the_library_seam(self):
        """Regression guard for the issue's headline symptom.

        ``JobManager.start()`` used to call ``vtsearch.auth.get_current_user``,
        whose ``from flask import g`` propagated ``ImportError`` when Flask
        was unavailable.
        """
        source = (_VTSCORE / "concurrency" / "async_jobs.py").read_text(encoding="utf-8")
        assert "vtsearch" not in source
        assert "from vtscore.state.current_user import" in source


class TestTestsLibDoesNotImportVtsearch:
    """``tests_lib/`` must honour the tier its ``__init__`` advertises."""

    def test_no_unlisted_vtsearch_imports(self):
        offenders: list[str] = []
        for path in sorted(_TESTS_LIB.rglob("*.py")):
            rel = path.relative_to(_TESTS_LIB).as_posix()
            if rel in _ALLOWED_TESTS_LIB:
                continue
            for lineno, module in _vtsearch_imports(path):
                offenders.append(f"tests_lib/{rel}:{lineno} imports {module}")

        assert not offenders, (
            "tests_lib/ is the library-tier suite and must not import vtsearch "
            "(see tests_lib/__init__.py).\n"
            + "\n".join(offenders)
            + "\n\nUse the vtscore equivalent - e.g. get_active_context().medias rather than "
            "the vtsearch.state proxy - or add a library-tier seam to patch, the way "
            "vtscore.embedding.loader._install_transformers_logging_bridge exists so the "
            "preload tests need not patch vtsearch.logging_config. If the test genuinely "
            "belongs to the app tier, move it to tests/."
        )

    def test_patch_targets_are_library_tier_too(self):
        """A ``mock.patch("vtsearch...")`` target is an import the AST cannot see.

        ``unittest.mock.patch`` imports the module named in its target string,
        so it is every bit as much an app-tier dependency as an ``import``
        statement - and it is exactly the shape the scan above misses.  Both
        halves have to hold or the gate only covers the easy one.  This is not
        hypothetical: three ``patch("vtsearch.logging_config...")`` calls in
        ``cli/test_preload_progress.py`` were the tier's only surviving app
        dependency once the two ``vtsearch.state`` imports were fixed.
        """
        offenders: list[str] = []
        for path in sorted(_TESTS_LIB.rglob("*.py")):
            rel = path.relative_to(_TESTS_LIB).as_posix()
            if rel in _ALLOWED_TESTS_LIB:
                continue
            for lineno, target in _patch_targets(path):
                if target.split(".")[0] == "vtsearch":
                    offenders.append(f"tests_lib/{rel}:{lineno} patches {target!r}")

        assert not offenders, (
            "tests_lib/ must not name a vtsearch module as a patch target - mock.patch "
            "imports it, so this is an app-tier dependency the import scan cannot see.\n"
            + "\n".join(offenders)
            + "\n\nPatch a library-tier seam instead; add one to vtscore if none exists "
            "(see vtscore.embedding.loader._install_transformers_logging_bridge)."
        )

    def test_allowlist_has_no_stale_entries(self):
        stale = [rel for rel in _ALLOWED_TESTS_LIB if not _vtsearch_imports(_TESTS_LIB / rel)]
        assert not stale, f"Allowlisted files no longer import vtsearch; drop them from _ALLOWED_TESTS_LIB: {stale}"
