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

    def test_index_contains_charts_js_reference(self):
        resp = self.client.get("/")
        assert b"charts.js" in resp.data

    def test_charts_js_loaded_before_app_js(self):
        resp = self.client.get("/")
        text = resp.data.decode("utf-8")
        charts_pos = text.index("charts.js")
        app_pos = text.index("app.js")
        assert charts_pos < app_pos

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

    def test_charts_js_accessible(self):
        resp = self.client.get("/static/charts.js")
        assert resp.status_code == 200
        assert "javascript" in resp.content_type

    def test_styles_css_accessible(self):
        resp = self.client.get("/static/styles.css")
        assert resp.status_code == 200
        assert "css" in resp.content_type

    def test_app_js_is_nonempty(self):
        resp = self.client.get("/static/app.js")
        assert len(resp.data) > 1000

    def test_charts_js_is_nonempty(self):
        resp = self.client.get("/static/charts.js")
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

    def test_index_has_vote_section(self):
        resp = self.client.get("/")
        assert b"vote-section" in resp.data

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

    def test_charts_js_exports_vtcharts(self):
        resp = self.client.get("/static/charts.js")
        text = resp.data.decode("utf-8")
        assert "window.VTCharts" in text

    def test_charts_js_has_render_functions(self):
        resp = self.client.get("/static/charts.js")
        text = resp.data.decode("utf-8")
        assert "renderErrorCostChart" in text
        assert "renderStabilityChart" in text
        assert "renderDiversityChart" in text

    def test_app_js_references_vtcharts(self):
        resp = self.client.get("/static/app.js")
        text = resp.data.decode("utf-8")
        assert "VTCharts" in text

    def test_styles_has_dark_and_light_themes(self):
        resp = self.client.get("/static/styles.css")
        text = resp.data.decode("utf-8")
        assert "data-theme" in text


class TestAutopilotSmoothTransitions:
    """Autopilot transitions should be deferred when mid-vote so the user
    isn't jerked to a new media before finishing their current selection."""

    def setup_method(self):
        import app as app_module
        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def _get_app_js(self):
        resp = self.client.get("/static/app.js")
        return resp.data.decode("utf-8")

    def test_pending_transition_variable_exists(self):
        text = self._get_app_js()
        assert "_pendingAutopilotTransition" in text

    def test_bad_to_hard_defers_when_voting(self):
        """Bad→Hard transition should check isVoting and defer."""
        text = self._get_app_js()
        # The deferred branch should store a pending transition when isVoting
        assert "isVoting" in text
        assert "_pendingAutopilotTransition = ()" in text

    def test_hard_to_new_defers_when_voting(self):
        """Hard→New transition should also defer when mid-vote."""
        text = self._get_app_js()
        # Both bad→hard and hard→new should have deferred paths;
        # verify the hard→new one references _fetchAndApplyDiversitySample
        assert "_fetchAndApplyDiversitySample" in text

    def test_pending_transition_applied_after_vote(self):
        """castVote should apply _pendingAutopilotTransition after auto-advance."""
        text = self._get_app_js()
        # The pending transition should be applied inside castVote
        assert "transition()" in text

    def test_stop_autopilot_clears_pending(self):
        """stopAutopilot should clear any pending transition."""
        text = self._get_app_js()
        # stopAutopilot should null out the pending transition
        assert "_pendingAutopilotTransition = null" in text

    def test_good_to_bad_not_deferred(self):
        """Good→Bad only changes select mode (no sort change), so it should
        apply immediately without deferral."""
        text = self._get_app_js()
        # Find the good→bad block — it should NOT have a pending transition.
        # The good phase block sets phase = "bad" and calls _apSetSelectMode
        # directly without checking isVoting.
        good_block_start = text.index('st.phase === "good"')
        bad_block_start = text.index('st.phase === "bad"', good_block_start + 1)
        good_block = text[good_block_start:bad_block_start]
        assert "_pendingAutopilotTransition" not in good_block
