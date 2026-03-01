"""Frontend serving tests.

Covers:
- SPA entry point (GET /)
- Static file serving (index.html, app.js, styles.css)
- Favicon variants (smile, frown, surprised) and unknown variants
- Logo serving (SVG)
- Content types and cache behavior
"""

from __future__ import annotations

import app as app_module


class TestIndexRoute:
    """GET / should serve the SPA entry point."""

    def setup_method(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_index_returns_200(self):
        resp = self.client.get("/")
        assert resp.status_code == 200

    def test_index_returns_html(self):
        resp = self.client.get("/")
        assert "text/html" in resp.content_type

    def test_index_contains_app_js_reference(self):
        resp = self.client.get("/")
        assert b"app.js" in resp.data

    def test_index_contains_doctype(self):
        resp = self.client.get("/")
        assert resp.data.strip().startswith(b"<!DOCTYPE html>") or resp.data.strip().startswith(b"<!doctype html>")


class TestStaticFiles:
    """Static assets should be accessible under /static/."""

    def setup_method(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_app_js_accessible(self):
        resp = self.client.get("/static/app.js")
        assert resp.status_code == 200
        assert "javascript" in resp.content_type

    def test_styles_css_accessible(self):
        resp = self.client.get("/static/styles.css")
        assert resp.status_code == 200
        assert "css" in resp.content_type

    def test_app_js_is_nonempty(self):
        resp = self.client.get("/static/app.js")
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
    """Verify the SPA content contains expected structure."""

    def setup_method(self):
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def test_index_has_media_list_element(self):
        resp = self.client.get("/")
        assert b"media-list" in resp.data

    def test_index_has_center_panel(self):
        resp = self.client.get("/")
        assert b"center" in resp.data

    def test_index_has_vote_buttons(self):
        resp = self.client.get("/")
        assert b"vote-good" in resp.data
        assert b"vote-bad" in resp.data

    def test_index_has_sort_controls(self):
        resp = self.client.get("/")
        assert b"sort-mode" in resp.data or b"text-sort" in resp.data

    def test_index_has_settings_modal(self):
        resp = self.client.get("/")
        assert b"settings-modal" in resp.data

    def test_index_has_dashboard_view(self):
        resp = self.client.get("/")
        assert b"dashboard-view" in resp.data

    def test_app_js_contains_fetch_calls(self):
        resp = self.client.get("/static/app.js")
        text = resp.data.decode("utf-8")
        assert "fetch(" in text

    def test_app_js_contains_api_sort(self):
        resp = self.client.get("/static/app.js")
        text = resp.data.decode("utf-8")
        assert "/api/sort" in text

    def test_app_js_contains_api_votes(self):
        resp = self.client.get("/static/app.js")
        text = resp.data.decode("utf-8")
        assert "/api/votes" in text

    def test_app_js_contains_api_medias(self):
        resp = self.client.get("/static/app.js")
        text = resp.data.decode("utf-8")
        assert "/api/medias" in text

    def test_styles_has_theme_variables(self):
        resp = self.client.get("/static/styles.css")
        text = resp.data.decode("utf-8")
        assert "--bg-body" in text or "--accent" in text

    def test_styles_has_media_item_class(self):
        resp = self.client.get("/static/styles.css")
        text = resp.data.decode("utf-8")
        assert ".media-item" in text

    def test_styles_has_dark_and_light_themes(self):
        resp = self.client.get("/static/styles.css")
        text = resp.data.decode("utf-8")
        assert "data-theme" in text
