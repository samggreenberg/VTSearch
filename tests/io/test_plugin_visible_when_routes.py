"""``visible_when`` on the HTTP upload path (issue #5343).

``tests_lib/core/test_plugin_field_visible_when.py`` covers the evaluator and
every library-tier check.  What lives here is the one app-tier piece:
:func:`vtsearch.routes._plugins.validate_plugin_args` reads file uploads off
the request itself, outside the marshmallow schema, so a required file field
a condition switches off must neither 422 as missing nor hand the plugin a
file the user could no longer see.
"""

from __future__ import annotations

import io

import pytest
from werkzeug.exceptions import HTTPException

from vtscore.plugins import PluginBase, PluginField
from vtsearch.routes._plugins import validate_plugin_args


class _FakePlugin(PluginBase):
    name = "fake_visible_when"
    display_name = "Fake"

    fields = [
        PluginField(key="attach", label="Attach a file", field_type="checkbox", default="false"),
        PluginField(key="upload", label="File", field_type="file", visible_when={"attach": True}),
    ]


def _validate(client, data: dict) -> dict:
    with client.application.test_request_context("/", method="POST", data=data, content_type="multipart/form-data"):
        return validate_plugin_args(_FakePlugin())


class TestFileFieldVisibleWhen:
    def test_switched_off_required_file_may_be_omitted(self, client):
        assert _validate(client, {"attach": "false"})["upload"] is None

    def test_switched_off_file_upload_is_discarded(self, client):
        values = _validate(client, {"attach": "false", "upload": (io.BytesIO(b"x"), "a.txt")})
        assert values["upload"] is None

    def test_switched_on_required_file_is_still_required(self, client):
        with pytest.raises(HTTPException) as exc:
            _validate(client, {"attach": "true"})
        assert exc.value.code == 422

    def test_switched_on_file_arrives(self, client):
        values = _validate(client, {"attach": "true", "upload": (io.BytesIO(b"x"), "a.txt")})
        assert values["upload"].filename == "a.txt"
