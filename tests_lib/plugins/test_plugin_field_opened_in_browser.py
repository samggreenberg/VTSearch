"""``PluginField.opened_in_browser`` swaps the SSRF guard for the scheme guard.

A ``url`` field is SSRF-guarded by default, because the framework cannot know
whether the plugin will fetch it.  A field whose URL only the user's browser
ever opens (an ``open_url`` exporter's template) declares that, and is then
held to :func:`~vtscore.security.url_validation.validate_browser_url` instead:
``localhost`` and LAN hosts pass, dangerous schemes still do not (issue #4078).
"""

from __future__ import annotations

import pytest

from vtscore.plugins import PluginBase, PluginField
from vtscore.plugins.normalize import normalize_field_values


class _FakePlugin(PluginBase):
    """Bare-minimum stand-in for a real plugin instance."""

    def __init__(self, fields: list[PluginField]) -> None:
        self.fields = fields


def _url_plugin(*, opened_in_browser: bool, field_type: str = "url") -> _FakePlugin:
    return _FakePlugin(
        [PluginField(key="url", label="URL", field_type=field_type, opened_in_browser=opened_in_browser)]  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8000/review?ids={ids}",
        "http://127.0.0.1:9000/viewer",
        "http://[::1]:9000/viewer",
        "http://192.168.1.20/viewer",
        "https://example.com/review",
    ],
)
def test_browser_opened_field_accepts_local_and_public_hosts(url):
    values = {"url": url}
    normalize_field_values(_url_plugin(opened_in_browser=True), values)
    assert values["url"] == url


@pytest.mark.parametrize("url", ["javascript:alert(1)", "data:text/html,x", "file:///etc/passwd"])
def test_browser_opened_field_still_rejects_dangerous_schemes(url):
    with pytest.raises(ValueError, match="http or https"):
        normalize_field_values(_url_plugin(opened_in_browser=True), {"url": url})


def test_browser_opened_field_names_a_missing_scheme():
    with pytest.raises(ValueError, match="Did you mean 'http://localhost:8000/x'"):
        normalize_field_values(_url_plugin(opened_in_browser=True), {"url": "localhost:8000/x"})


def test_undeclared_url_field_keeps_the_ssrf_guard():
    with pytest.raises(ValueError, match="private/internal"):
        normalize_field_values(_url_plugin(opened_in_browser=False), {"url": "http://127.0.0.1:9000/viewer"})


def test_hidden_default_is_held_to_the_browser_guard():
    plugin = _FakePlugin(
        [
            PluginField(
                key="url_template",
                label="URL Template",
                field_type="url",
                default="http://localhost:8000/review?ids={ids}",
                hidden=True,
                opened_in_browser=True,
            )
        ]
    )
    values = {"url_template": ""}
    normalize_field_values(plugin, values)
    assert values["url_template"] == "http://localhost:8000/review?ids={ids}"


def test_flag_is_ignored_on_non_url_fields():
    # A ``text`` field is not URL-validated at all; the flag must not start
    # validating it.
    values = {"url": "not a url"}
    normalize_field_values(_url_plugin(opened_in_browser=True, field_type="text"), values)
    assert values["url"] == "not a url"


def test_flag_defaults_to_false():
    assert PluginField(key="u", label="U", field_type="url").opened_in_browser is False
