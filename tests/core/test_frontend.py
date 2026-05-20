"""Frontend serving tests.

Covers:
- SPA entry point (GET /)
- Static file serving (Angular build output: index.html, main.js, styles.css)
- Favicon variants (smile, frown, surprised) and unknown variants
- Logo serving (SVG)
- Content types and cache behavior
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

# Tests below hit the Angular SPA shell or its bundle artefacts
# (main.js / polyfills.js / styles.css / index.html), which only exist
# after `npm run build:prod` has populated `static/`.  `./run-tests.sh`
# builds the bundle as part of the core / full-suite path, so the
# normal flow is already covered.  For plain `pytest` invocations the
# session fixture below builds the bundle on demand (~16s, once per
# session) so the tests run for real instead of being silently skipped.
# Only if the build itself isn't possible (no npm, no node_modules, or
# `npm run build:prod` fails) do we fall back to `pytest.skip` with a
# clear reason.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_STATIC_DIR = _REPO_ROOT / "static"
_FRONTEND_DIR = _REPO_ROOT / "frontend"


def _bundle_built() -> bool:
    return (_STATIC_DIR / "index.html").exists() and (_STATIC_DIR / "main.js").exists()


@pytest.fixture(scope="session", autouse=True)
def _ensure_angular_bundle() -> None:
    """Build the Angular bundle on demand if it isn't already on disk.

    Runs once per session.  No-op when the bundle is already present
    (the normal `./run-tests.sh` flow builds it before pytest starts).
    """
    if _bundle_built():
        return
    npm = shutil.which("npm")
    if npm is None or not (_FRONTEND_DIR / "node_modules").exists():
        pytest.skip(
            "Angular bundle not built and cannot build it here "
            f"(npm={'found' if npm else 'missing'}, "
            f"node_modules={'present' if (_FRONTEND_DIR / 'node_modules').exists() else 'missing'}). "
            "Run: cd frontend && npm install && npm run build:prod",
            allow_module_level=True,
        )
    try:
        subprocess.run(  # noqa: S603 — npm resolved via shutil.which, args constant
            [npm, "run", "build:prod"],
            cwd=_FRONTEND_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        pytest.skip(
            f"Angular build failed: {exc.stderr.strip() or exc.stdout.strip() or exc}",
            allow_module_level=True,
        )
    if not _bundle_built():
        pytest.skip(
            "Angular build completed but static/main.js or static/index.html still missing",
            allow_module_level=True,
        )


class TestIndexRoute:
    """GET / should serve the Angular SPA entry point."""

    def test_index_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_index_returns_html(self, client):
        resp = client.get("/")
        assert "text/html" in resp.content_type

    def test_index_contains_main_js_reference(self, client):
        resp = client.get("/")
        assert b"main.js" in resp.data

    def test_index_contains_polyfills_reference(self, client):
        resp = client.get("/")
        assert b"polyfills.js" in resp.data

    def test_index_contains_doctype(self, client):
        resp = client.get("/")
        assert resp.data.strip().startswith(b"<!DOCTYPE html>") or resp.data.strip().startswith(b"<!doctype html>")

    def test_index_contains_app_root(self, client):
        resp = client.get("/")
        assert b"<app-root>" in resp.data


class TestAngularRoutes:
    """Angular client-side routes should return the SPA index."""

    def test_dashboard_route_returns_200(self, client):
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        assert b"<app-root>" in resp.data

    def test_label_route_returns_200(self, client):
        resp = client.get("/label")
        assert resp.status_code == 200
        assert b"<app-root>" in resp.data


class TestStaticFiles:
    """Static assets should be accessible under /static/."""

    def test_main_js_accessible(self, client):
        resp = client.get("/static/main.js")
        assert resp.status_code == 200
        assert "javascript" in resp.content_type

    def test_polyfills_js_accessible(self, client):
        resp = client.get("/static/polyfills.js")
        assert resp.status_code == 200
        assert "javascript" in resp.content_type

    def test_styles_css_accessible(self, client):
        resp = client.get("/static/styles.css")
        assert resp.status_code == 200
        assert "css" in resp.content_type

    def test_main_js_is_nonempty(self, client):
        resp = client.get("/static/main.js")
        assert len(resp.data) > 1000

    def test_styles_css_is_nonempty(self, client):
        resp = client.get("/static/styles.css")
        assert len(resp.data) > 1000

    def test_nonexistent_static_returns_404(self, client):
        resp = client.get("/static/does_not_exist.xyz")
        assert resp.status_code == 404


class TestRootStaticFiles:
    """Static assets should also be accessible at root paths (no /static/ prefix).

    The Angular build uses <base href="/"> so the browser requests main.js,
    polyfills.js, and styles.css at the root.
    """

    def test_main_js_at_root(self, client):
        resp = client.get("/main.js")
        assert resp.status_code == 200
        assert "javascript" in resp.content_type

    def test_polyfills_js_at_root(self, client):
        resp = client.get("/polyfills.js")
        assert resp.status_code == 200
        assert "javascript" in resp.content_type

    def test_styles_css_at_root(self, client):
        resp = client.get("/styles.css")
        assert resp.status_code == 200
        assert "css" in resp.content_type

    def test_logo_png_at_root(self, client):
        resp = client.get("/logo.png")
        assert resp.status_code == 200

    def test_unknown_path_returns_spa(self, client):
        """Unknown paths should return the Angular SPA for client-side routing."""
        resp = client.get("/some/unknown/route")
        assert resp.status_code == 200
        assert b"<app-root>" in resp.data


class TestFavicon:
    """Favicon routes should serve valid .ico files."""

    def test_default_favicon(self, client):
        resp = client.get("/favicon.ico")
        assert resp.status_code in (200, 204)

    def test_favicon_smile(self, client):
        resp = client.get("/favicon-smile.ico")
        assert resp.status_code in (200, 204)

    def test_favicon_frown(self, client):
        resp = client.get("/favicon-frown.ico")
        assert resp.status_code in (200, 204)

    def test_favicon_surprised(self, client):
        resp = client.get("/favicon-surprised.ico")
        assert resp.status_code in (200, 204)

    def test_favicon_unknown_variant_returns_404(self, client):
        resp = client.get("/favicon-angry.ico")
        assert resp.status_code == 404

    def test_favicon_empty_variant_returns_spa_fallback(self, client):
        resp = client.get("/favicon-.ico")
        # Empty variant doesn't match the favicon-<variant>.ico route,
        # so the catch-all serves the SPA (standard SPA fallback behavior).
        assert resp.status_code == 200
        assert b"<app-root>" in resp.data

    def test_favicon_content_type(self, client):
        resp = client.get("/favicon.ico")
        if resp.status_code == 200:
            assert "icon" in resp.content_type or "octet" in resp.content_type


class TestLogo:
    """Logo route should serve SVG."""

    def test_logo_svg(self, client):
        resp = client.get("/logo.svg")
        assert resp.status_code in (200, 204)

    def test_logo_svg_content_type(self, client):
        resp = client.get("/logo.svg")
        if resp.status_code == 200:
            assert "svg" in resp.content_type


class TestFrontendContentIntegrity:
    """Verify the Angular SPA content contains expected structure."""

    def test_index_has_angular_app_root(self, client):
        resp = client.get("/")
        assert b"<app-root>" in resp.data

    def test_index_has_base_href(self, client):
        resp = client.get("/")
        assert b'base href="/"' in resp.data

    def test_index_has_title(self, client):
        resp = client.get("/")
        assert b"VTSearch" in resp.data

    def test_index_loads_main_js_as_module(self, client):
        resp = client.get("/")
        assert b'type="module"' in resp.data

    def test_main_js_contains_angular_code(self, client):
        resp = client.get("/static/main.js")
        text = resp.data.decode("utf-8")
        # Angular bundles contain component class names and framework references
        assert "Component" in text

    def test_bundle_contains_api_references(self, client):
        """API references may live in main.js or lazy-loaded chunks."""
        import glob as globmod

        js_files = globmod.glob("static/*.js")
        combined = ""
        for path in js_files:
            resp = client.get(f"/{path}")
            combined += resp.data.decode("utf-8")
        assert "/api/" in combined

    def test_styles_has_theme_variables(self, client):
        resp = client.get("/static/styles.css")
        text = resp.data.decode("utf-8")
        assert "--bg-body" in text or "--accent" in text

    def test_styles_has_layout_classes(self, client):
        resp = client.get("/static/styles.css")
        text = resp.data.decode("utf-8")
        # Angular global styles contain layout and panel classes
        assert "panel" in text or "grid" in text or "--bg-body" in text


class TestVersionEndpoint:
    """GET /api/version should return the app version string."""

    def test_returns_200(self, client):
        resp = client.get("/api/version")
        assert resp.status_code == 200

    def test_returns_version_field(self, client):
        from vtsearch import __version__

        resp = client.get("/api/version")
        data = resp.get_json()
        assert data == {"version": __version__}

    def test_version_is_iso_utc_timestamp(self, client):
        from datetime import datetime

        resp = client.get("/api/version")
        version = resp.get_json()["version"]
        # Must be parseable as an ISO 8601 timestamp ending in Z (UTC).
        assert version.endswith("Z")
        datetime.fromisoformat(version.replace("Z", "+00:00"))


class TestVersionResolution:
    """`vtsearch.__init__` resolves __version__ from git, then a baked file, then a fallback."""

    def test_git_resolver_returns_iso_utc_for_real_repo(self):
        from vtsearch import _version_from_git

        version = _version_from_git()
        assert version is not None, "tests run from a git checkout — git resolution must succeed"
        assert version.endswith("Z")

    def test_file_resolver_reads_baked_version(self, tmp_path, monkeypatch):
        import vtsearch as pkg

        baked = tmp_path / "_version.txt"
        baked.write_text("2030-01-02T03:04:05Z\n", encoding="utf-8")
        monkeypatch.setattr(pkg, "__file__", str(tmp_path / "__init__.py"))
        assert pkg._version_from_file() == "2030-01-02T03:04:05Z"

    def test_file_resolver_returns_none_when_missing(self, tmp_path, monkeypatch):
        import vtsearch as pkg

        monkeypatch.setattr(pkg, "__file__", str(tmp_path / "__init__.py"))
        assert pkg._version_from_file() is None

    def test_fallback_when_git_and_file_unavailable(self, monkeypatch):
        import vtsearch as pkg

        monkeypatch.setattr(pkg, "_version_from_git", lambda: None)
        monkeypatch.setattr(pkg, "_version_from_file", lambda: None)
        assert pkg._resolve_version() == "0.0.0-unknown"
