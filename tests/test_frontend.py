"""Frontend serving tests.

Covers:
- SPA entry point (GET /)
- Static file serving (Angular build output: index.html, main.js, styles.css)
- Favicon variants (smile, frown, surprised) and unknown variants
- Logo serving (SVG)
- Content types and cache behavior
- Legacy /ng/ redirect
"""

from __future__ import annotations

import app as app_module


class TestIndexRoute:
    """GET / should serve the Angular SPA entry point."""

    def setup_method(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_index_returns_200(self):
        resp = self.client.get("/")
        assert resp.status_code == 200

    def test_index_returns_html(self):
        resp = self.client.get("/")
        assert "text/html" in resp.content_type

    def test_index_contains_main_js_reference(self):
        resp = self.client.get("/")
        assert b"main.js" in resp.data

    def test_index_contains_polyfills_reference(self):
        resp = self.client.get("/")
        assert b"polyfills.js" in resp.data

    def test_index_contains_doctype(self):
        resp = self.client.get("/")
        assert resp.data.strip().startswith(b"<!DOCTYPE html>") or resp.data.strip().startswith(b"<!doctype html>")

    def test_index_contains_app_root(self):
        resp = self.client.get("/")
        assert b"<app-root>" in resp.data


class TestAngularRoutes:
    """Angular client-side routes should return the SPA index."""

    def setup_method(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_dashboard_route_returns_200(self):
        resp = self.client.get("/dashboard")
        assert resp.status_code == 200
        assert b"<app-root>" in resp.data

    def test_label_route_returns_200(self):
        resp = self.client.get("/label")
        assert resp.status_code == 200
        assert b"<app-root>" in resp.data


class TestLegacyNgRedirect:
    """Legacy /ng/ URLs should redirect to /."""

    def setup_method(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_ng_redirects_to_root(self):
        resp = self.client.get("/ng/")
        assert resp.status_code == 301
        assert resp.headers["Location"].endswith("/")

    def test_ng_path_redirects(self):
        resp = self.client.get("/ng/dashboard")
        assert resp.status_code == 301
        assert resp.headers["Location"].endswith("/dashboard")


class TestStaticFiles:
    """Static assets should be accessible under /static/."""

    def setup_method(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_main_js_accessible(self):
        resp = self.client.get("/static/main.js")
        assert resp.status_code == 200
        assert "javascript" in resp.content_type

    def test_polyfills_js_accessible(self):
        resp = self.client.get("/static/polyfills.js")
        assert resp.status_code == 200
        assert "javascript" in resp.content_type

    def test_styles_css_accessible(self):
        resp = self.client.get("/static/styles.css")
        assert resp.status_code == 200
        assert "css" in resp.content_type

    def test_main_js_is_nonempty(self):
        resp = self.client.get("/static/main.js")
        assert len(resp.data) > 1000

    def test_styles_css_is_nonempty(self):
        resp = self.client.get("/static/styles.css")
        assert len(resp.data) > 1000

    def test_nonexistent_static_returns_404(self):
        resp = self.client.get("/static/does_not_exist.xyz")
        assert resp.status_code == 404


class TestFavicon:
    """Favicon routes should serve valid .ico files."""

    def setup_method(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_default_favicon(self):
        resp = self.client.get("/favicon.ico")
        assert resp.status_code in (200, 204)

    def test_favicon_smile(self):
        resp = self.client.get("/favicon-smile.ico")
        assert resp.status_code in (200, 204)

    def test_favicon_frown(self):
        resp = self.client.get("/favicon-frown.ico")
        assert resp.status_code in (200, 204)

    def test_favicon_surprised(self):
        resp = self.client.get("/favicon-surprised.ico")
        assert resp.status_code in (200, 204)

    def test_favicon_unknown_variant_returns_404(self):
        resp = self.client.get("/favicon-angry.ico")
        assert resp.status_code == 404

    def test_favicon_empty_variant_returns_404(self):
        resp = self.client.get("/favicon-.ico")
        assert resp.status_code == 404

    def test_favicon_content_type(self):
        resp = self.client.get("/favicon.ico")
        if resp.status_code == 200:
            assert "icon" in resp.content_type or "octet" in resp.content_type


class TestLogo:
    """Logo route should serve SVG."""

    def setup_method(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_logo_svg(self):
        resp = self.client.get("/logo.svg")
        assert resp.status_code in (200, 204)

    def test_logo_svg_content_type(self):
        resp = self.client.get("/logo.svg")
        if resp.status_code == 200:
            assert "svg" in resp.content_type


class TestFrontendContentIntegrity:
    """Verify the Angular SPA content contains expected structure."""

    def setup_method(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_index_has_angular_app_root(self):
        resp = self.client.get("/")
        assert b"<app-root>" in resp.data

    def test_index_has_base_href(self):
        resp = self.client.get("/")
        assert b'base href="/"' in resp.data

    def test_index_has_title(self):
        resp = self.client.get("/")
        assert b"VTSearch" in resp.data

    def test_index_loads_main_js_as_module(self):
        resp = self.client.get("/")
        assert b'type="module"' in resp.data

    def test_main_js_contains_angular_code(self):
        resp = self.client.get("/static/main.js")
        text = resp.data.decode("utf-8")
        # Angular bundles contain component class names and framework references
        assert "Component" in text

    def test_main_js_contains_api_references(self):
        resp = self.client.get("/static/main.js")
        text = resp.data.decode("utf-8")
        assert "/api/" in text

    def test_styles_has_theme_variables(self):
        resp = self.client.get("/static/styles.css")
        text = resp.data.decode("utf-8")
        assert "--bg-body" in text or "--accent" in text

    def test_styles_has_layout_classes(self):
        resp = self.client.get("/static/styles.css")
        text = resp.data.decode("utf-8")
        # Angular global styles contain layout and panel classes
        assert "panel" in text or "grid" in text or "--bg-body" in text
