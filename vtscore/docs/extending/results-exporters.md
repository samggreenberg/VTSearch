# Writing a `LabelsetExporter`

A results exporter sends autodetect output or a label export to a
destination - a file on the server, a webhook, an email, a queue, an
S3 object, anything reachable from the Python process. The library
auto-discovers exporters under `vtscore.exporters` (sentinel
`EXPORTER`) and walks the `vtscore.exporters` entry-point group so a
third-party distribution can `pip install` one in. Subclass
[`LabelsetExporter`](../../exporters/base.py)
([`vtscore/exporters/base.py`](../../exporters/base.py)), declare
your `fields`, and implement `export(results, field_values) -> dict`.

The base class is named `LabelsetExporter` for historical reasons; in
practice it handles both autodetect-results payloads and label
exports. The `export()` method should detect which is which (label
payloads carry a top-level `"labels"` key - see
[`vtscore/exporters/server_json_file/__init__.py`](../../exporters/server_json_file/__init__.py)
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
- [Opening a URL in the browser](#opening-a-url-in-the-browser)
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
| `opens_url: bool` | `False` | Set `True` when `export()` always returns an `"open_url"` - see [Opening a URL in the browser](#opening-a-url-in-the-browser) |
| `export_cli(results, field_values)` | delegates to `export()` | Override only when CLI-supplied values need different handling (rare) |

Expose `EXPORTER = YourExporter()` at module level so the registry
picks it up. The sentinel must be an already-instantiated object, not
the class.

The return value's `"message"` is shown to the user as confirmation;
extra keys (`"filepath"`, `"status_code"`, …) are passed through to
the API response. Two of those keys mean something to the frontend:
`"display_results"` (rendered in the Auto-Detect Results modal) and
`"open_url"` (opened in a new browser tab - see below).

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

Declare the variables a field accepts in its `template_vars` tuple; the
framework substitutes them before `export()` is called:

```python
PluginField(
    key="filepath",
    label="Save to (server path)",
    field_type="server_path",
    default=f"{DATA_DIR}/results_{{YYYYMMDD-HHMMSS}}.json",
    template_vars=("YYYYMMDD-HHMMSS", "YYYYMMDD", "YYYY", "MM", "DD", "detector_name", "username"),
)
```

| Variable | Resolves to |
|----------|-------------|
| `{YYYYMMDD-HHMMSS}` | Current UTC timestamp - unique per run, so consecutive exports don't overwrite each other |
| `{YYYYMMDD}` / `{YYYY}` / `{MM}` / `{DD}` | Current UTC date parts, for date-stamped paths from scheduled runs, e.g. `results_{YYYY}.{MM}.{DD}.csv` |
| `{detector_name}` / `{detector_id}` | Active detector identity |
| `{username}` | Current user (single-user installs: `"default"`) |

**Don't substitute by hand, and don't omit `template_vars`.** The
framework pass sanitises every resolved value with
`sanitize_template_value`, so a malicious `{detector_name}` containing
`../` can't escape the per-user data directory in multi-user mode; a
hand-rolled `str.replace` chain gets that wrong. And a field that
doesn't declare `template_vars` gets *no* substitution at all - the
user's `{YYYYMMDD}` ends up literally in the filename. Declaring a name
outside the supported set raises `ValueError` on the first request, so
typos fail loudly.

## Server-path and URL validation

**A declared field is already validated.** Because a field typed
`url` is passed through `vtscore.security.validate_url` and a field
typed `server_path` (or `folder`) through `confine_server_filepath()`
before your `export()` runs — see [Framework-side
normalization](README.md#framework-side-normalization) — the correct
exporter body just uses the value:

```python
def export(self, results: dict, field_values: dict) -> dict:
    path = Path(field_values["filepath"])   # already stripped, substituted, confined
    requests.post(field_values["url"], json=results, timeout=30, allow_redirects=False)
```

Use `field_values[key]`, never a copy of the raw request value you
stashed elsewhere: for path fields the pass writes back the *approved,
canonicalised* path, and re-deriving it yourself resolves against the
process CWD instead of the user's data dir.

**Anything you construct is still yours to validate.** A URL you build
by joining a configured base with a path segment, or a path you join
from a field plus a generated filename, never passed through a declared
field, so run it through `validate_url`
([`vtscore/security/url_validation.py`](../../security/url_validation.py))
or `validate_server_filepath(..., base_dir=get_file_access_base_dir())`
([`vtscore/security/path_validation.py`](../../security/path_validation.py))
yourself:

```python
from vtscore.security.url_validation import validate_url

url = validate_url(urljoin(field_values["base_url"], f"/labelsets/{labelset_id}"))
```

Both helpers are import-clean of Flask, so library-only exporter tests
can exercise them directly. Calling them redundantly on an
already-validated field value is harmless (they're idempotent), just
unnecessary.

## Opening a URL in the browser

Some destinations can't be *delivered* to - a site with a viewer but no
ingest API. You can still hand the user off to one: format the labelset
into that site's URL and return it as `"open_url"`, and the Export modal
opens it in a new tab.

```python
from urllib.parse import quote

from vtscore.security.url_validation import validate_browser_url


class ReviewSiteExporter(LabelsetExporter):
    name = "review_site"
    display_name = "Review Site"
    description = "Open the labelset in the review site."
    opens_url = True          # lets the button read "Open Labelset in Review Site"
    fields = []

    def export(self, results: dict, field_values: dict) -> dict:
        ids = ",".join(e["md5"] for e in results.get("labels", []))
        url = validate_browser_url(f"https://review.example.com/?ids={quote(ids, safe='')}")
        return {"message": "Opening the labelset in Review Site.", "open_url": url}
```

Four things this shape gets right, and that yours should too:

- **`validate_browser_url`, not `validate_url`.** The *browser* makes
  this request, so the SSRF guard is the wrong tool - it would reject a
  legitimate `http://localhost:9000/viewer`. The browser guard is a
  scheme allowlist that stops `javascript:` / `data:` / `file:`. The
  route re-runs it on whatever you return and fails the export if it
  doesn't pass, so a mistake here is a broken export, never an injected
  URL.
- **Percent-encode anything you splice in.** An id list joined with `,`
  and dropped raw into a query string breaks on the first identifier
  containing `&` or `#`.
- **Cap the length.** URLs stop working somewhere around 2000
  characters. Truncate deliberately and say so in `"message"` (the
  built-in `open_url` exporter reports "first 100 of 5,000 item(s)")
  rather than emitting a URL that silently opens the wrong selection.
- **Assume the URL is public.** It reaches the destination site's logs
  and the user's browser history. Identifiers, not content.

Set `opens_url = True` only if you *always* return a URL; the flag is
what the UI reads to label the button before running anything. An
exporter that returns one only sometimes (a webhook whose remote hands
back a permalink) leaves the flag `False` - the URL still opens.

**You don't need an `export_cli` override for this.** The CLI has no
browser, so it surfaces the `open_url` your `export()` returned rather
than dropping it: printed under the confirmation message in text mode,
and carried as an `open_url` field on the `export_complete` event under
`--progress-format json`, where a wrapping script can open it. Don't
`print()` the URL yourself - that duplicates the line and puts prose in
the middle of the NDJSON stream.

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

from vtscore.exporters.base import PluginField, LabelsetExporter


class SignedWebhookExporter(LabelsetExporter):
    """POST labels to a webhook with an HMAC-SHA256 signature header."""

    name = "signed_webhook"
    display_name = "Webhook (HMAC-signed)"
    description = "POST labels JSON to a URL with an HMAC-SHA256 signature."
    icon = "\U0001f510"  # closed lock with key
    fields = [
        PluginField(
            key="url",
            label="Webhook URL",
            field_type="url",
            description="The URL to POST labels to.",
            required=True,
        ),
        PluginField(
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

        # Both fields are declared and required, so by this point they are
        # stripped, non-empty, and (for the "url" field type) SSRF-checked.
        url = field_values["url"]
        secret = field_values["hmac_secret"].encode("utf-8")

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

        # export() takes already-normalized field_values, so a unit test
        # can hand it a plain dict without going through a route.
        with patch("my_pkg.exporter.requests.post", _fake_post):
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
