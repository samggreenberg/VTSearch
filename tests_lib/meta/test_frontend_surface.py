"""The frontend-only fast path's premise: which tests can see the frontend.

``run-tests.sh`` narrows itself when a branch changes nothing outside
``frontend/``: pyright, pip-audit and the vulture whitelist check are skipped,
and pytest is cut to ``tests_shared.frontend_surface.FRONTEND_TEST_SURFACE``.
That is sound only while the surface really is every test that can observe the
frontend, and the failure mode if it stops being true is silent: a test added
to ``tests/sorting/`` that serves the SPA shell would simply stop running on
frontend changes, and the run would still print ``RUN PASSED``.

So the premise is checked rather than trusted.  A test reaches the frontend in
one of three shapes, and each gets a rule:

``PATH``
    a code string naming a file under ``frontend/`` (or the bare segment
    ``"frontend"`` / ``"static"``, the way a ``Path`` is assembled), or a bundle
    artefact (``index.html``, ``main.js``, ``styles.css``, ``static/``).

``FIXTURE``
    the ``angular_bundle`` fixture, by parameter name or through
    ``usefixtures``.  It exists to build the bundle for a test that reads it.

``ROUTE``
    a Flask test-client call on a route outside ``/api/``.  The SPA catch-all
    in ``vtsearch/routes/main.py`` answers every such route with the bundle,
    except the health probes in ``NON_BUNDLE_ROUTES``, which are their own
    handlers.

Docstrings are excluded from the string scan, as in
``tests_lib/meta/test_markdown_surface.py``: prose that *mentions*
``frontend/src/app/utils/plugin-fields.ts`` is not a read, and flagging it
would train people to route around this gate.
"""

from __future__ import annotations

import ast
import functools
import subprocess
from pathlib import Path

import pytest

from tests_shared.frontend_surface import (
    BUNDLE_LITERALS,
    FRONTEND_PATH_PREFIXES,
    FRONTEND_TEST_SURFACE,
    NON_BUNDLE_ROUTES,
)

# The sweep parses every test file once; one worker keeps the parse cache warm
# for every test in this module.
pytestmark = pytest.mark.xdist_group("frontend-surface")

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOTS = ("tests", "tests_lib")

#: Methods of the Flask test client (and werkzeug's ``Client``) whose first
#: positional argument is a route.
CLIENT_METHODS = frozenset({"get", "post", "put", "patch", "delete", "head", "options", "open"})
PATH_SEGMENTS = frozenset({"frontend", "static"})
FIXTURE_NAME = "angular_bundle"


@functools.cache
def _tracked_test_files() -> tuple[str, ...]:
    out = subprocess.run(  # noqa: S603  # fixed argv, no shell
        ["git", "ls-files", "-z", "--", *TEST_ROOTS],  # noqa: S607  # git resolves on PATH, as everywhere else here
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    )
    return tuple(p for p in out.stdout.decode("utf-8").split("\0") if p.endswith(".py"))


def _code_strings(tree: ast.AST) -> list[str]:
    """Every string constant except docstrings."""
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                docstrings.add(id(first.value))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings
    ]


def _is_api_route(route: str) -> bool:
    return route == "/api" or route.startswith("/api/")


def frontend_references(text: str) -> dict[str, set[str]]:
    """What a test module reaches in the frontend, by rule; empty when nothing.

    Returns ``{"PATH": {...}, "FIXTURE": {...}, "ROUTE": {...}}`` with only the
    non-empty rules present.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:  # pragma: no cover - the repo does not carry unparseable Python
        return {}
    found: dict[str, set[str]] = {"PATH": set(), "FIXTURE": set(), "ROUTE": set()}
    for literal in _code_strings(tree):
        if literal.startswith(FRONTEND_PATH_PREFIXES) or literal in PATH_SEGMENTS:
            found["PATH"].add(literal)
        elif any(marker in literal for marker in BUNDLE_LITERALS):
            if FIXTURE_NAME in literal:
                found["FIXTURE"].add(literal)
            else:
                found["PATH"].add(literal)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            for arg in (*node.args.args, *node.args.kwonlyargs):
                if arg.arg == FIXTURE_NAME:
                    found["FIXTURE"].add(arg.arg)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in CLIENT_METHODS
            and node.args
        ):
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                route = first.value
                if route.startswith("/") and not _is_api_route(route) and route not in NON_BUNDLE_ROUTES:
                    found["ROUTE"].add(route)
    return {rule: hits for rule, hits in found.items() if hits}


@functools.cache
def _references_in(rel: str) -> dict[str, set[str]]:
    return frontend_references((REPO_ROOT / rel).read_text(encoding="utf-8", errors="ignore"))


def _in_surface(rel: str) -> bool:
    return any(rel == entry or rel.startswith(entry.rstrip("/") + "/") for entry in FRONTEND_TEST_SURFACE)


class TestSurfaceIsWellFormed:
    def test_every_surface_entry_exists(self):
        for entry in FRONTEND_TEST_SURFACE:
            assert (REPO_ROOT / entry).exists(), (
                f"FRONTEND_TEST_SURFACE names {entry}, which no longer exists. "
                f"run-tests.sh hands this list straight to pytest, so a stale "
                f"entry fails every frontend-only run."
            )

    def test_every_surface_entry_is_a_test_path(self):
        for entry in FRONTEND_TEST_SURFACE:
            assert entry.startswith(TEST_ROOTS), f"{entry} is not under a test tree"

    def test_run_tests_reads_the_surface(self):
        """The runner must take its selection from here, not from a copy."""
        text = (REPO_ROOT / "run-tests.sh").read_text(encoding="utf-8")
        assert "tests_shared.frontend_surface" in text and "FRONTEND_TEST_SURFACE" in text, (
            "run-tests.sh no longer reads FRONTEND_TEST_SURFACE from "
            "tests_shared/frontend_surface.py. A second copy of the list is how "
            "this gate stops describing what actually runs."
        )

    def test_non_bundle_routes_are_real_handlers(self):
        """A route listed as non-bundle must be registered on its own, not served by the catch-all."""
        routes_py = "\n".join(p.read_text(encoding="utf-8") for p in (REPO_ROOT / "vtsearch" / "routes").glob("*.py"))
        for route in NON_BUNDLE_ROUTES:
            assert f'"{route}"' in routes_py, (
                f"NON_BUNDLE_ROUTES lists {route}, but no handler under vtsearch/routes/ "
                f"registers it, so the SPA catch-all would answer it with the bundle."
            )


class TestNoFrontendReaderOutsideTheSurface:
    """The rule that makes the fast path's pytest selection true."""

    def test_no_frontend_reference_outside_the_surface(self):
        offenders = {}
        for rel in _tracked_test_files():
            if _in_surface(rel):
                continue
            refs = _references_in(rel)
            if refs:
                offenders[rel] = refs
        assert not offenders, (
            "These tests can observe the frontend but sit outside FRONTEND_TEST_SURFACE, "
            "so a frontend-only run would not run them:\n"
            + "\n".join(
                f"  {rel}: " + "; ".join(f"{rule} {sorted(hits)}" for rule, hits in refs.items())
                for rel, refs in offenders.items()
            )
            + "\n\nEither add the test's path to FRONTEND_TEST_SURFACE in "
            "tests_shared/frontend_surface.py, or stop reaching the frontend from it."
        )

    def test_the_surface_is_not_padded(self):
        """Every file entry in the surface really reaches the frontend (the meta folder is exempt)."""
        idle = [entry for entry in FRONTEND_TEST_SURFACE if entry.endswith(".py") and not _references_in(entry)]
        assert not idle, (
            f"These FRONTEND_TEST_SURFACE entries no longer reach the frontend: {idle}. "
            f"Drop them so a frontend-only run stays as narrow as its premise allows."
        )


class TestTheRulesActuallyFire:
    """A gate nobody has seen fail is a gate nobody knows is wired up."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ('P = "frontend/src/app/x.ts"', {"PATH": {"frontend/src/app/x.ts"}}),
            ('P = ROOT / "frontend" / "src"', {"PATH": {"frontend"}}),
            ('resp = client.get("/static/main.js")', {"PATH": {"/static/main.js"}, "ROUTE": {"/static/main.js"}}),
            ('idx = Path(static) / "index.html"', {"PATH": {"index.html"}}),
            ("def test_x(client, angular_bundle):\n    pass", {"FIXTURE": {"angular_bundle"}}),
            ('@pytest.mark.usefixtures("angular_bundle")\ndef test_x():\n    pass', {"FIXTURE": {"angular_bundle"}}),
            ('resp = client.get("/")', {"ROUTE": {"/"}}),
            ('resp = c.post("/login", data={})', {"ROUTE": {"/login"}}),
            # API routes and the health probes do not read the bundle.
            ('resp = client.get("/api/version")', {}),
            ('resp = client.get("/api")', {}),
            ('resp = client.get("/healthz")', {}),
            # Prose is not a read.
            ('"""Mirrors frontend/src/app/utils/plugin-fields.ts."""', {}),
            # A route built at runtime is not a literal this scan can see, and
            # is not claimed; the surface is curated for exactly that reason.
            ("resp = client.get(route)", {}),
        ],
    )
    def test_rules(self, text, expected):
        assert frontend_references(text) == expected
