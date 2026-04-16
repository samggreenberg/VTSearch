"""Frontend serving tests.

Covers:
- SPA entry point (GET /)
- Static file serving (Angular build output: index.html, main.js, styles.css)
- Favicon variants (smile, frown, surprised) and unknown variants
- Logo serving (SVG)
- Content types and cache behavior
"""

from __future__ import annotations


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
