"""The frontend-only fast path: which Python tests can observe the frontend.

``run-tests.sh`` narrows itself when a branch changes nothing outside
``frontend/``.  Almost none of the whole-repo gates can see such a change:
pyright excludes that tree (``pyrightconfig.json``), pip-audit reads the venv,
and the vulture whitelist check reads Python.  What *can* see it runs in full:
every cheap gate in stage 1 (codespell and check-docs read TypeScript and
SCSS), the production build, ``npm audit`` and the Vitest suite.  Of the Python
tests, the only ones a TypeScript or SCSS edit can reach are the ones that read
a file under ``frontend/``, read the built bundle in ``static/`` (``index.html``,
``main.js``, ``styles.css``, the chunks), or ask the Flask test client for a
route the SPA catch-all in ``vtsearch/routes/main.py`` answers with that bundle.
Everything else tests Python the change cannot touch.

``FRONTEND_TEST_SURFACE`` names those tests, as paths pytest accepts, and
``tests_lib/meta/test_frontend_surface.py`` checks the premise rather than
trusting it: a test outside the surface that names a frontend path, a bundle
artefact, the ``angular_bundle`` fixture, or a non-API route fails that gate,
so the narrowing cannot rot silently the way a hand-kept list would.

Both constants are read by the runner (via ``python -c``) and by that gate, so
the two cannot disagree.  The shape mirrors ``tests_shared/markdown_surface.py``.
"""

from __future__ import annotations

# The pytest selection a frontend-only run uses, as paths rather than markers.
#
#   tests_lib/meta                   the repo's own tooling tests: SCSS text
#                                    scans, the frontend-gate and npm-audit-gate
#                                    self-tests, package.json checks.  Every
#                                    deliberate reader of the frontend tree.
#   tests/core/test_frontend.py      serves the built bundle and reads main.js.
#   tests/api/test_dashboard.py      serves the bundle for the dashboard shell.
#   tests/api/test_auth_enforcement.py
#   tests/api/test_error_envelope.py
#   tests/datasets/test_datasets.py  each asks the client for a route the SPA
#                                    catch-all answers with index.html.
#   tests_lib/detectors/test_label_quota.py
#                                    reads autopilot-state.service.ts to hold
#                                    the TypeScript mirror to the Python rule.
#   tests_lib/media/test_media_type_icons.py
#                                    reads the icon SVG tables to hold every
#                                    media type's served icon to a real glyph.
FRONTEND_TEST_SURFACE: tuple[str, ...] = (
    "tests_lib/meta",
    "tests/core/test_frontend.py",
    "tests/api/test_dashboard.py",
    "tests/api/test_auth_enforcement.py",
    "tests/api/test_error_envelope.py",
    "tests/datasets/test_datasets.py",
    "tests_lib/detectors/test_label_quota.py",
    "tests_lib/media/test_media_type_icons.py",
)

#: Paths under which a change is a frontend change.  ``frontend/openapi.json``
#: is deliberately inside it: a route change regenerates that snapshot, and the
#: generated TypeScript client built from it is what the production build
#: compiles, so such a branch is a mixed one and runs everything.
FRONTEND_PATH_PREFIXES: tuple[str, ...] = ("frontend/",)

#: Routes outside ``/api/`` that are their own Flask handlers rather than the
#: SPA catch-all, so hitting one does not read the bundle.  A test may use
#: these without joining the surface.
NON_BUNDLE_ROUTES: tuple[str, ...] = ("/healthz", "/readyz")

#: Substrings that, in a code string literal, name the built bundle or the
#: fixture that builds it.
BUNDLE_LITERALS: tuple[str, ...] = (
    "index.html",
    "main.js",
    "styles.css",
    "static/",
    "angular_bundle",
)
