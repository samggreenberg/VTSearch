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
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VTSCORE = _REPO_ROOT / "vtscore"

#: Imports of ``vtsearch`` that are allowed to remain, with the reason.
#: The first two are *optional* - wrapped in ``try``/``except`` so the
#: library keeps working when the app package is absent, which is what
#: makes them tolerable.  The third is a genuine remaining violation with
#: its own issue.  Anything else fails this test.
_ALLOWED: dict[str, str] = {
    "exporters/portable_detector/__init__.py": (
        "guarded `import vtsearch` used only to stamp the producing app's version "
        "into a bundle; falls back to a literal when the app package is absent"
    ),
    "embedding/loader.py": "optional transformers logging bridge, wrapped in try/except Exception",
    "security/path_validation.py": (
        "UNGUARDED, tracked in issue #3042: the whole LoginProvider abstraction "
        "still lives in vtsearch.auth, so moving it is its own change, not a hook"
    ),
}


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


class TestVtscoreDoesNotImportVtsearch:
    def test_no_unlisted_vtsearch_imports(self):
        offenders: list[str] = []
        for path in sorted(_VTSCORE.rglob("*.py")):
            rel = path.relative_to(_VTSCORE).as_posix()
            if rel in _ALLOWED:
                continue
            for lineno, module in _vtsearch_imports(path):
                offenders.append(f"vtscore/{rel}:{lineno} imports {module}")

        assert not offenders, (
            "vtscore must not import vtsearch (the dependency runs the other way).\n"
            + "\n".join(offenders)
            + "\n\nRoute app-tier behaviour through a registered hook instead - see "
            "vtscore/user.py and vtscore/achievements_hooks.py for the pattern."
        )

    def test_allowlist_has_no_stale_entries(self):
        """Every allowlisted file must still contain a ``vtsearch`` import.

        Otherwise the allowlist quietly grows into a list of exemptions
        nobody has re-earned.
        """
        stale = [rel for rel in _ALLOWED if not _vtsearch_imports(_VTSCORE / rel)]
        assert not stale, f"Allowlisted files no longer import vtsearch; drop them from _ALLOWED: {stale}"


class TestJobManagerHasNoAppTierImport:
    def test_async_jobs_resolves_the_user_through_the_library_seam(self):
        """Regression guard for the issue's headline symptom.

        ``JobManager.start()`` used to call ``vtsearch.auth.get_current_user``,
        whose ``from flask import g`` propagated ``ImportError`` when Flask
        was unavailable.
        """
        source = (_VTSCORE / "concurrency" / "async_jobs.py").read_text(encoding="utf-8")
        assert "vtsearch" not in source
        assert "from vtscore.user import" in source
