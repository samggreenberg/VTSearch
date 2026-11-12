"""Frontend serving tests.

Covers:
- SPA entry point (GET /)
- Static file serving (Angular build output: index.html, main.js, styles.css)
- Favicon variants (smile, frown, surprised) and unknown variants
- Logo serving (SVG)
- Content types and cache behavior
"""

from __future__ import annotations

import pathlib
import re

import pytest

# Tests below hit the Angular SPA shell or its bundle artefacts
# (main.js / styles.css / index.html; the app is zoneless, so there is no
# polyfills.js bundle), which only exist after `npm run build:prod` has
# populated `static/`.  `./run-tests.sh` builds the bundle as part of the
# core / full-suite path, so the normal flow is already covered.  For plain
# `pytest` invocations the shared session fixture `angular_bundle` (in
# tests/conftest.py, guarded by a cross-worker file lock for `-n auto`)
# builds the bundle on demand, or skips with a clear reason when the build
# isn't possible (no npm, no node_modules, or `npm run build:prod` fails).
pytestmark = pytest.mark.usefixtures("angular_bundle")


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

    The Angular build uses <base href="/"> so the browser requests main.js
    and styles.css at the root. (The app is zoneless, so there is no
    polyfills.js bundle.)
    """

    def test_main_js_at_root(self, client):
        resp = client.get("/main.js")
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

    def test_favicon_smile_is_the_plain_favicon(self, client):
        # The plain favicon already smiles, so there is no smile variant (#4680).
        resp = client.get("/favicon-smile.ico")
        assert resp.status_code == 404

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


class TestEagerBundleComposition:
    """Guards on what is allowed into the INITIAL (eager) bundle.

    The eager path is framework-dominated and deliberately carries headroom
    for Angular's own patch releases; the rationale lives next to
    `maximumWarning` in frontend/angular.json.  The budget alone cannot police
    this: #3811 reclaimed ~38kB, so a re-added FormsModule would still fit
    under the number while silently spending the margin that exists so a
    framework bump does not turn the gate red for everyone (#3795).  The same
    goes for @angular/cdk/a11y, which #5297 took off the eager path, for
    the toast stack, which #5505 put behind `@defer`, and for the detector
    sharing stack, which #5657 moved behind a dynamic import.
    """

    @staticmethod
    def _eager_bundle_text() -> str:
        """`main.js` plus every chunk it imports STATICALLY, concatenated.

        `from"./chunk-X.js"` (or a bare `import"./chunk-X.js"`) is a static
        import and keeps that chunk on the eager path.  `import("./chunk-Y.js")`
        is the `@defer` / lazy-route boundary and is deliberately NOT followed:
        following it would walk the whole app and make every assertion here
        vacuous.  With the chunk optimizer on (#5657) the eager path is
        `main.js` alone, but the walk still covers a build without it: esbuild
        names chunks `chunk-ABC123.js`, rolldown `chunk-aB_c-12.js`.
        """
        static_dir = pathlib.Path("static")
        static_import = re.compile(r'(?:from|import)\s*["`]\./(chunk-[\w-]+\.js)["`]')
        seen: set[str] = set()
        queue = ["main.js"]
        parts: list[str] = []
        while queue:
            name = queue.pop()
            if name in seen:
                continue
            path = static_dir / name
            if not path.exists():
                continue
            seen.add(name)
            text = path.read_text(encoding="utf-8")
            parts.append(text)
            queue.extend(static_import.findall(text))
        return "".join(parts)

    def test_eager_bundle_resolves(self):
        """Fail loudly if the walk above stops matching the build output."""
        text = self._eager_bundle_text()
        assert len(text) > 100_000, (
            "Could not resolve the eager bundle from static/main.js. The build "
            "output's import syntax probably changed, which would make the "
            "FormsModule guard below silently vacuous -- fix the regex."
        )

    def test_forms_module_is_not_eager(self):
        # `ngNoForm` comes from NgForm's selector (`form:not([ngNoForm])`) and
        # appears nowhere outside @angular/forms, so it is a precise marker for
        # the package landing in the initial bundle.
        assert "ngNoForm" not in self._eager_bundle_text(), (
            "@angular/forms (~37kB) is back in the EAGER bundle.\n"
            "Something on the eager path imports FormsModule -- most likely a new\n"
            "`ngModel` in vt-login or vt-dialog-host, the two eager components that\n"
            "used to pull it (#3811). A single uncontrolled input does not need it:\n"
            "use `[value]` + `(input)` instead. If the import is genuinely required,\n"
            "re-audit the budget in frontend/angular.json rather than deleting this."
        )

    def test_cdk_focus_trap_is_not_eager(self):
        # `cdk-focus-trap-anchor` is the class CDK's FocusTrap gives its Tab
        # sentinels and appears nowhere outside @angular/cdk/a11y.  `vt-modal`
        # is a native <dialog> (#5297), so nothing on the eager path needs it.
        assert "cdk-focus-trap-anchor" not in self._eager_bundle_text(), (
            "@angular/cdk/a11y's focus trap (~9kB) is back in the EAGER bundle.\n"
            "Something on the eager path imports CdkTrapFocus / FocusTrap -- most\n"
            "likely `vt-modal`, which reaches the initial bundle through the\n"
            "always-rendered `vt-dialog-host`. vt-modal opens a <dialog> with\n"
            "showModal(), which handles focus natively (#5297); a lazy component may\n"
            "still use the CDK. If the eager import is genuinely required, re-audit\n"
            "the budget in frontend/angular.json rather than deleting this."
        )

    def test_toast_container_is_not_eager(self):
        # The Dismiss All button's label is written only in vt-toast-container's
        # template, which compiles into the component's own chunk.  The
        # `vt-toast-container` tag name is no marker: AppComponent's template
        # still names it inside the `@defer` block, on the eager path.
        assert "Dismiss all notifications" not in self._eager_bundle_text(), (
            "vt-toast-container (~12kB) is back in the EAGER bundle.\n"
            "AppComponent mounts it inside `@defer (on immediate)` (#5505), and Angular\n"
            "only splits a component off when nothing outside a @defer block uses it:\n"
            "most likely the `@defer` wrapper was dropped, or something eager now\n"
            "references ToastContainerComponent directly (a viewChild, a second\n"
            "un-deferred <vt-toast-container>). Toasts need no eager renderer --\n"
            "ToastService holds the stack, so one raised before the chunk lands shows\n"
            "on arrival. If the eager import is genuinely required, re-audit the\n"
            "budget in frontend/angular.json rather than deleting this."
        )

    def test_detector_sharing_is_not_eager(self):
        # The sharing endpoints' paths are written only in their generated
        # functions (`generated/api-client/fn/detectors-sharing/`), which only
        # DetectorsSharingApiService imports.
        assert "/api/detectors/sharing" not in self._eager_bundle_text(), (
            "The detector sharing API is in the EAGER bundle (~6kB with the flows\n"
            "and capability services it comes with).\n"
            "Something on the eager path imports it -- most likely the error\n"
            "interceptor, whose read-only toast offers a Fork: it imports\n"
            "DetectorSharingFlowsService at the click, with import() (#5657), and a\n"
            "static import would put the whole sharing stack back on the eager path.\n"
            "If the eager import is genuinely required, re-audit the budget in\n"
            "frontend/angular.json rather than deleting this."
        )

    def test_card_menu_is_not_eager(self):
        # The detector card's Browse label is written only in
        # `card-context-menu-items.ts`, which the lazy Dashboard imports.
        assert "Browse this detector's positives" not in self._eager_bundle_text(), (
            "The dashboard card menu (card-context-menu-items.ts) is in the EAGER bundle.\n"
            "Something on the eager path imports from it -- most likely a route guard,\n"
            "which app.routes.ts loads eagerly. The People guard once took\n"
            "PEOPLE_MEDIA_TYPE from it and put the bundle over budget; a constant a\n"
            "guard needs belongs in a module of its own (utils/people.ts).\n"
            "If the eager import is genuinely required, re-audit the budget in\n"
            "frontend/angular.json rather than deleting this."
        )


class TestChunkOptimizer:
    """`build:prod` runs Angular's chunk optimizer (#5657).

    Without it the eager path is ~32 files: `main.js`, ten `modulepreload`ed
    chunks, then 21 more that an HTTP/1.1 browser fetches in four waves, which
    held Browse's first paint back ~0.5 s at 100 ms of latency.  The optimizer
    merges them into one `main.js`.  It is switched on by an environment
    variable only, so nothing fails if the variable is dropped from the script
    or Angular stops reading it: the build just goes quietly slower.  These
    read the configuration rather than `static/`, which can predate it.
    """

    FRONTEND = pathlib.Path("frontend")

    def test_build_prod_turns_the_optimizer_on(self):
        import json

        scripts = json.loads((self.FRONTEND / "package.json").read_text())["scripts"]
        assert "NG_BUILD_OPTIMIZE_CHUNKS=1" in scripts["build:prod"].split(), (
            "frontend/package.json's build:prod no longer sets NG_BUILD_OPTIMIZE_CHUNKS=1.\n"
            "Without it every page load fetches the eager bundle as ~32 files in waves\n"
            "(#5657). The Docker images run the same script, so it is the one place to set it."
        )

    def test_angular_still_reads_the_variable(self):
        options = self.FRONTEND / "node_modules/@angular/build/src/utils/environment-options.js"
        if not options.exists():
            pytest.skip("frontend/node_modules not installed")
        assert "process.env['NG_BUILD_OPTIMIZE_CHUNKS']" in options.read_text(), (
            "@angular/build no longer reads NG_BUILD_OPTIMIZE_CHUNKS, so build:prod's setting\n"
            "does nothing and the eager bundle is back to ~32 files (#5657). Find what\n"
            "replaced it (likely an `optimization` option in angular.json) and switch to that."
        )


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
