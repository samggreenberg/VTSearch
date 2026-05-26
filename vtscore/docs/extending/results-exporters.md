# Writing a `LabelsetExporter`

A results exporter sends autodetect output or a label export to a
destination - a file on the server, a webhook, an email, a queue, an
S3 object, anything reachable from the Python process. The library
auto-discovers exporters under `vtscore.exporters` (sentinel
`EXPORTER`) and walks the `vtscore.exporters` entry-point group so a
third-party distribution can `pip install` one in. Subclass
[`LabelsetExporter`](../../exporters/base.py)
([`vtscore/exporters/base.py:59`](../../exporters/base.py)), declare
your `fields`, and implement `export(results, field_values) -> dict`.

The base class is named `LabelsetExporter` for historical reasons; in
practice it handles both autodetect-results payloads and label
exports. The `export()` method should detect which is which (label
payloads carry a top-level `"labels"` key - see
[`vtscore/exporters/server_json_file/__init__.py:75`](../../exporters/server_json_file/__init__.py)
for the pattern).

**App-side counterpart:** [`docs/EXTENDING-plugins.md § Adding a
Results Exporter`](../../../docs/EXTENDING-plugins.md#adding-a-results-exporter)
covers the UI / route wiring. This guide focuses on the library API
and third-party packaging.

## Contents

- [The contract](#the-contract)
- [Two payload shapes](#two-payload-shapes)
- [Template-variable interpolation](#template-variable-interpolation)
- [Server-path and URL validation](#server-path-and-url-validation)
- [Entry-point registration](#entry-point-registration)
- [Worked example](#worked-example)
- [Testing pattern](#testing-pattern)

## The contract

`LabelsetExporter` is a `PluginBase` subclass. Required overrides:

| Member | Type | Purpose |
|--------|------|---------|
| `name: str` | class attr | Snake-case identifier - registry key, CLI subcommand, API path segment |
| `display_name: str` | class attr | Human-readable label |
| `description: str` | class attr | One-sentence subtitle |
| `fields: list[PluginField]` | class attr | User-configurable inputs |
| `export(results, field_values)` | method | Perform the export; return a dict with at least `"message"` |

Optional overrides:

| Member | Default | Purpose |
|--------|---------|---------|
| `icon: str` | `"📤"` | Emoji rendered in the UI |
| `export_cli(results, field_values)` | delegates to `export()` | Override only when CLI-supplied values need different handling (rare) |

Expose `EXPORTER = YourExporter()` at module level so the registry
picks it up. The sentinel must be an already-instantiated object, not
the class.

The return value's `"message"` is shown to the user as confirmation;
extra keys (`"filepath"`, `"status_code"`, …) are passed through to
the API response.

## Two payload shapes

Exporters receive one of two dict shapes and should detect which:

**Autodetect results** (from `/api/auto-detect` or CLI autodetect):

```python
{
    "media_type": "audio",
    "detectors_run": 2,
    "results": {
        "<detector_name>": {
            "detector_name": "...",
            "threshold": 0.5,
            "total_hits": 15,
            "hits": [{"id": 1, "score": 0.82, "media": {...}}, ...],
        },
        ...
    },
}
```

**Label exports** (from the label-export modal with `enrich=true`):

```python
{
    "labels": [
        {"md5": "...", "label": "good", "filename": "...", "custom_metadata": {...}, ...},
        ...
    ],
    "selected_columns": ["md5", "label", "Catalogue ID"],
}
```

The standard idiom:

```python
def export(self, results: dict, field_values: dict) -> dict:
    if "labels" in results:
        return self._export_labels(results, field_values)
    return self._export_autodetect(results, field_values)
```

The built-in [`server_json_file`](../../exporters/server_json_file/__init__.py),
[`server_csv_file`](../../exporters/server_csv_file/), and
[`webhook`](../../exporters/webhook/__init__.py) exporters all follow
this pattern.

## Template-variable interpolation

`server_path` fields support template variables resolved at export
time:

- `{YYYYMMDD-HHMMSS}` - current UTC timestamp
- `{detector_name}` / `{detector_id}` - active detector identity
- `{username}` - current user (single-user installs: `"default"`)

The interpolation helper lives at
[`vtscore/exporters/_template.py`](../../exporters/_template.py); use
`resolve_export_filepath(filepath_str)` to apply every supported
substitution to a user-supplied path string. Don't roll your own
regex - `resolve_export_filepath` also sanitises substituted values
via `sanitize_template_value` so a malicious `{detector_name}`
containing `../` can't escape the configured `SERVER_ROOTS`.

## Server-path and URL validation

Any exporter that accepts a file path **must** validate it through
`vtscore.security.validate_server_filepath`
([`vtscore/security/path_validation.py:35`](../../security/path_validation.py)).
This refuses paths that resolve outside the configured
`SERVER_ROOTS` (defaulting to the process CWD, overridable via
`VTSEARCH_SERVER_ROOTS`):

```python
from vtscore.security.path_validation import validate_server_filepath

path = validate_server_filepath(field_values["filepath"])
# Raises ValueError if outside SERVER_ROOTS - let it propagate.
```

Any exporter that makes an outbound HTTP request **must** validate
the URL through `vtscore.security.validate_url`
([`vtscore/security/url_validation.py:30`](../../security/url_validation.py)).
It rejects non-HTTP(S) schemes and resolves the hostname to refuse
private / loopback / link-local IPs - the standard SSRF guard:

```python
from vtscore.security.url_validation import validate_url

url = validate_url(field_values["url"])
requests.post(url, json=results, timeout=30, allow_redirects=False)
```

Both helpers are import-clean of Flask, so library-only exporter tests
can exercise them directly.

## Entry-point registration

In-tree:

```
vtscore/exporters/<your_exporter>/__init__.py
```

Out-of-tree:

```toml
[project]
name = "vtsearch-myexporter"
version = "0.1.0"
dependencies = ["vtsearch"]

[project.entry-points."vtscore.exporters"]
my_exporter = "my_pkg.exporter:EXPORTER"
```

The value must resolve to an already-instantiated `LabelsetExporter`.
After `pip install`, the exporter appears in `list_exporters()`, the
`/api/exporters` endpoint, the CLI's `--exporter` flag, and `python
app.py --list-plugins`.

## Worked example

A minimal third-party exporter that POSTs labels (and only labels - it
ignores autodetect-results payloads) to a webhook with HMAC signing.

```python
# my_pkg/exporter.py
from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

import requests

from vtscore.exporters.base import ExporterField, LabelsetExporter
from vtscore.security.url_validation import validate_url


class SignedWebhookExporter(LabelsetExporter):
    """POST labels to a webhook with an HMAC-SHA256 signature header."""

    name = "signed_webhook"
    display_name = "Webhook (HMAC-signed)"
    description = "POST labels JSON to a URL with an HMAC-SHA256 signature."
    icon = "\U0001f510"  # closed lock with key
    fields = [
        ExporterField(
            key="url",
            label="Webhook URL",
            field_type="url",
            description="The URL to POST labels to.",
            required=True,
        ),
        ExporterField(
            key="hmac_secret",
            label="HMAC secret",
            field_type="password",
            description="Shared secret used to sign the payload (X-Signature header).",
            required=True,
        ),
    ]

    def export(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        # Only handle label exports; refuse autodetect-results payloads.
        if "labels" not in results:
            raise ValueError("signed_webhook exporter only handles label exports.")

        url = validate_url(field_values["url"].strip())
        secret = field_values["hmac_secret"].encode("utf-8")
        if not secret:
            raise ValueError("HMAC secret is required.")

        body = json.dumps(results, sort_keys=True).encode("utf-8")
        signature = hmac.new(secret, body, hashlib.sha256).hexdigest()

        resp = requests.post(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Signature": f"sha256={signature}",
            },
            timeout=30,
            allow_redirects=False,
        )
        resp.raise_for_status()

        n = len(results["labels"])
        return {
            "message": f"Posted {n} label(s) to {url} (HTTP {resp.status_code}).",
            "status_code": resp.status_code,
            "url": url,
        }


EXPORTER = SignedWebhookExporter()
```

And the `pyproject.toml`:

```toml
[project.entry-points."vtscore.exporters"]
signed_webhook = "my_pkg.exporter:EXPORTER"
```

After `pip install`, the exporter appears in the export modal and the
CLI: `python app.py --autodetect --dataset … --settings … --exporter
signed_webhook --url https://… --hmac-secret …`.

## Testing pattern

Library-tier exporter tests belong in `tests_lib/io/`. The autouse
fixtures (see [`tests_lib/conftest.py`](../../../tests_lib/conftest.py))
reset every registry and stub embedders; `_allow_test_tmp_paths`
widens path validation so `validate_server_filepath` accepts the test
tmp directory.

```python
# tests_lib/io/test_signed_webhook_exporter.py
import hashlib
import hmac
import json
from unittest.mock import patch

from vtscore.exporters import get_exporter, list_exporters


class TestSignedWebhookRegistration:
    def test_is_discoverable(self):
        names = [e.name for e in list_exporters()]
        assert "signed_webhook" in names

    def test_fields(self):
        exp = get_exporter("signed_webhook")
        keys = [f.key for f in exp.fields]
        assert "url" in keys and "hmac_secret" in keys


class TestSignedWebhookExport:
    def test_signs_label_payload(self):
        exp = get_exporter("signed_webhook")
        results = {"labels": [{"md5": "abc", "label": "good"}]}

        captured: dict = {}
        class _Resp:
            status_code = 200
            def raise_for_status(self): pass

        def _fake_post(url, data, headers, timeout, allow_redirects):
            captured["url"] = url
            captured["data"] = data
            captured["headers"] = headers
            return _Resp()

        with patch("my_pkg.exporter.requests.post", _fake_post), \
             patch("my_pkg.exporter.validate_url", lambda u: u):
            result = exp.export(results, {
                "url": "https://example.com/hook",
                "hmac_secret": "s3cret",
            })

        assert result["status_code"] == 200
        signature = captured["headers"]["X-Signature"]
        expected = "sha256=" + hmac.new(
            b"s3cret",
            json.dumps(results, sort_keys=True).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        assert signature == expected

    def test_rejects_autodetect_payload(self):
        exp = get_exporter("signed_webhook")
        import pytest
        with pytest.raises(ValueError):
            exp.export({"media_type": "audio", "results": {}}, {})
```

See [`tests_lib/io/test_importers.py`](../../../tests_lib/io/test_importers.py)
for the general I/O-plugin test pattern and the existing
[`vtscore/exporters/webhook/__init__.py`](../../exporters/webhook/__init__.py)
+ its app-tier tests for a real working webhook exporter.
