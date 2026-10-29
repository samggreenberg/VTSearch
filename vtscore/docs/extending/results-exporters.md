# Writing a `ResultsExporter`

A results exporter sends autodetect output or a label export to a
destination - a file on the server, a webhook, an email, a queue, an
S3 object, anything reachable from the Python process. The library
auto-discovers exporters under `vtscore.exporters` (sentinel
`EXPORTER`) and walks the `vtscore.exporters` entry-point group so a
third-party distribution can `pip install` one in. Subclass
[`ResultsExporter`](../../exporters/base.py)
([`vtscore/exporters/base.py`](../../exporters/base.py)), declare
your `fields`, and implement a method per payload kind you support.

`LabelsetExporter` is a permanent module-level alias of
`ResultsExporter`, so imports and subclasses written against that name
keep working untouched.

**App-side counterpart:** [`docs/EXTENDING-plugins.md § Adding a
Results Exporter`](../../../docs/EXTENDING-plugins.md#adding-a-results-exporter)
covers the UI / route wiring. This guide focuses on the library API
and third-party packaging.

## Contents

- [The contract](#the-contract)
- [Payload kinds](#payload-kinds)
- [Template-variable interpolation](#template-variable-interpolation)
- [Server-path and URL validation](#server-path-and-url-validation)
- [Opening a URL in the browser](#opening-a-url-in-the-browser)
- [Streaming from the CLI](#streaming-from-the-cli)
- [Entry-point registration](#entry-point-registration)
- [Worked example](#worked-example)
- [Testing pattern](#testing-pattern)

## The contract

`ResultsExporter` is a `PluginBase` subclass. Required overrides:

| Member | Type | Purpose |
|--------|------|---------|
| `name: str` | class attr | Snake-case identifier - registry key, CLI subcommand, API path segment |
| `display_name: str` | class attr | Human-readable label |
| `description: str` | class attr | One-sentence subtitle |
| `fields: list[PluginField]` | class attr | User-configurable inputs |
| a payload method | method | At least one of `export_find_results()` / `export_labelset()` / `export_cli_detectors()`; return a dict with at least `"message"` |

Optional overrides:

| Member | Default | Purpose |
|--------|---------|---------|
| `icon: str` | `"📤"` | Emoji rendered in the UI |
| `opens_url: bool` | `False` | Set `True` when your export method always returns an `"open_url"` - see [Opening a URL in the browser](#opening-a-url-in-the-browser) |
| `export(results, field_values)` | raises | Legacy single-method form. The named methods delegate here when unoverridden, which keeps pre-existing plugins working; prefer the named methods in new code |
| `export_cli(results, field_values)` | delegates to `export_find_results()` | Override only when CLI-supplied values need different handling (rare) |
| `supports_streaming` / `export_cli_streaming(header, records, field_values)` | `False` / raises | Write a scored run hit by hit for `--stream-results`; see [Streaming from the CLI](#streaming-from-the-cli) |
| `export_labelset_streaming(header, elements, field_values)` | raises | Write the run's hits as label entries for `--stream-results --as-labelset`; `supports_labelset_streaming` is derived from overriding it |
| `export_cli_streaming_resumable(header, chunks, field_values, resume)` / `resumable_stream_path(field_values)` | raises / `""` | Write a scored run a chunk at a time so an interrupted `--stream-results --resume` run can continue; `supports_resumable_streaming` is derived from overriding it. See [Resumable streams](#resumable-streams) |

Expose `EXPORTER = YourExporter()` at module level so the registry
picks it up. The sentinel must be an already-instantiated object, not
the class.

The return value's `"message"` is shown to the user as confirmation;
extra keys (`"filepath"`, `"status_code"`, …) are passed through to
the API response. Two of those keys mean something to the frontend:
`"display_results"` (rendered in the Auto-Detect Results modal) and
`"open_url"` (opened in a new browser tab - see below).

## Payload kinds

An exporter is a *destination*; what gets sent there is a separate
axis. There are three payload kinds, each with its own method:

| Kind | Method | What it is |
|------|--------|------------|
| `find_results` | `export_find_results()` | a scored run - hit lists, scores, thresholds |
| `labelset` | `export_labelset()` | a detector's labels - origins and vote provenance |
| `detector_bundles` | `export_cli_detectors()` | the trained classifiers themselves |

A labelset reaches `export_labelset()` already stripped of the numbers this
server's embedder and head measured (`labelset_for_export`,
`vtscore/datasets/labelset.py`): each element's `vt:provenance` keeps how the
item was drawn but not its `rank_at_vote` / `score_at_vote` (#5131). An
exported labelset is references to labels, so it can rebuild the detector
under another embedder; your exporter has nothing to drop.

`supported_payloads` is **derived** from which of these you overrode
(never declared, so there is nothing to forget), and it is what each
picker filters on: an exporter is only offered for the kinds it claims,
and the route answers 400 for anything else rather than letting it
deliver an empty export.

**Autodetect results** (from `/api/auto-detect` or CLI autodetect):

```python
{
    "media_type": "audio",
    "detectors_run": 2,
    "results": {
        "<detector_name>": {
            "detector_name": "...",
            "threshold": 0.5,
            "balance": {...},  # may be absent or None
            "total_hits": 15,  # len(hits)
            "hits": [
                {"id": 1, "filename": "...", "category": "...", "score": 0.82, "md5": "..."},  # + more keys
                ...,
            ],
            "negative_hits": [...],
        },
        # ... one entry per detector run
    },
    "missing_detectors": [...],
}
```

Each hit is built by `vtscore.datasets.labelset.build_media_hit`; the
`ResultsExporter.export_find_results` docstring has the full key list
(origin, `custom_metadata`, clip bounds) and what `balance` holds.

**Label exports** (from the label-export modal with `enrich=true`):

```python
{
    "labels": [
        {"md5": "...", "label": "good", "filename": "...", "custom_metadata": {...}},  # + more keys
        ...,
    ],
    "selected_columns": ["md5", "label", "Catalogue ID"],
}
```

**Export `hits`; treat `negative_hits` as informational.** `hits` is
the set the export is for, and whoever built the payload has already
chosen which side of the threshold goes into it: the Find Results
dialog's Good / Bad / Both moves the side it lists into `hits` (and
empties `negative_hits`), while a run with nobody to ask (CLI
`--autodetect`, a background AutoFind) sends the above-threshold side.
When `hits` holds both sides, every hit carries a `"label"` of
`"good"` or `"bad"`, the same key a streamed run's records carry; if
you write a row per hit, keep it. Do not export `negative_hits` too:
that overrides the choice, sending below-threshold items from a CLI
run that never asked for them.

The standard idiom is one method per kind, sharing whatever delivery
code they have in common:

```python
def _deliver(self, payload: dict, field_values: dict) -> dict:
    ...  # the connection / write / POST, identical either way

def export_find_results(self, results: dict, field_values: dict) -> dict:
    return self._deliver(results, field_values)

def export_labelset(self, labelset: dict, field_values: dict) -> dict:
    return self._deliver(labelset, field_values)
```

The built-in [`server_json_file`](../../exporters/server_json_file/__init__.py),
[`server_csv_file`](../../exporters/server_csv_file/), and
[`webhook`](../../exporters/webhook/__init__.py) exporters all follow
this pattern. Implement only one when only one makes sense -
[`portable_detector`](../../exporters/portable_detector/__init__.py)
consumes trained detectors rather than either results shape, so it
declares neither and stands alone on `detector_bundles`. An exporter
that files items by the label a human gave them, and has nothing to say
about a scored run, would likewise declare `export_labelset()` alone.

### Legacy single-method exporters

An exporter written against the single-method contract implements one
`export()` and tells the two dict shapes apart itself with
`if "labels" in results`. That still works and is not going away: the default
`export_find_results()` and `export_labelset()` both delegate to
`export()`, so an out-of-tree plugin written against the old contract
needs no changes. Such an exporter is credited with *both* kinds - there
is no way to know which it handles - and logs a line at import time
pointing here. Migrating is a mechanical split of the `if` into two
methods, and buys accurate picker filtering.

## Template-variable interpolation

Declare the variables a field accepts in its `template_vars` tuple; the
framework substitutes them before your export method is called:

```python
PluginField(
    key="filepath",
    label="Save to (server path)",
    field_type="server_path",
    default=f"{DATA_DIR}/results_{{YYYYMMDD-HHMMSS}}.json",
    template_vars=("YYYYMMDD-HHMMSS", "YYYYMMDD", "YYYY", "MM", "DD", "detector_name", "username"),
)
```

The supported names, the sanitising of each resolved value, and why a
field must declare them rather than substitute by hand (an undeclared
`{YYYYMMDD}` reaches your body as a literal) are in
[Template variables](README.md#template-variables). For an exporter, the
point of `{YYYYMMDD-HHMMSS}` in a default path is that it is unique per
run, so consecutive exports don't overwrite each other.

**Declaring is also what makes the GUI show the value.** Run-now plugin
forms resolve a field's *declared* variables client-side while building
the form, so a field carrying `default="{detector_name}"` opens showing
the detector's real name instead of the raw placeholder - visible and
editable before the user hits Export. Resolving the active detector
inside your export method instead gets you an empty box in the UI and
a value that appears out of nowhere on submit.

## Server-path and URL validation

**A declared field is already validated.** Because a field typed
`url` is passed through `vtscore.security.url_validation.validate_url` and a field
typed `server_path` (or `folder`) through `confine_server_filepath()`
before your export method runs — see [Framework-side
normalization](README.md#framework-side-normalization) — the correct
exporter body just uses the value:

```python
from vtscore.security.url_validation import guarded_session


def export_find_results(self, results: dict, field_values: dict) -> dict:
    path = Path(field_values["filepath"])   # already stripped, substituted, confined
    with guarded_session() as session:
        session.post(field_values["url"], json=results, timeout=30, allow_redirects=False)
```

Send a request to a user-supplied URL on `guarded_session()`, as the
built-in `webhook` exporter does, rather than through bare `requests`:
`validate_url` vets the addresses the hostname resolved to when it ran,
and the session re-checks the address each connection actually reaches,
which closes the DNS-rebinding gap between the two lookups.

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


class ReviewSiteExporter(ResultsExporter):
    name = "review_site"
    display_name = "Review Site"
    description = "Open the labelset in the review site."
    opens_url = True          # lets the button read "Open Labelset in Review Site"
    fields = []

    def export_labelset(self, labelset: dict, field_values: dict) -> dict:
        ids = ",".join(e["md5"] for e in labelset.get("labels", []))
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
  built-in `open_url` exporter reports "first 100 of 5000 item(s)")
  rather than emitting a URL that silently opens the wrong selection.
- **Assume the URL is public.** It reaches the destination site's logs
  and the user's browser history. Identifiers, not content.

Set `opens_url = True` only if you *always* return a URL. The flag does
two jobs: it labels the button before anything runs, and it tells the
Export modal to claim the new tab *inside the click handler* and
navigate it when your export method returns. That ordering is the whole
game - a `window.open()` fired from the response instead is exactly what
a popup blocker exists to stop, and it fails silently. An exporter that
returns a URL only sometimes (a webhook whose remote hands back a
permalink) leaves the flag `False`; the URL still opens on a best-effort
basis, with the success toast's **Open** button as the fallback.

**You don't need an `export_cli` override for this.** The CLI has no
browser, so it surfaces the `open_url` your export method returned rather
than dropping it: printed under the confirmation message in text mode,
and carried as an `open_url` field on the `export_complete` event under
`--progress-format json`, where a wrapping script can open it. Don't
`print()` the URL yourself - that duplicates the line and puts prose in
the middle of the NDJSON stream.

## Streaming from the CLI

`python app.py --autodetect --tempimport --chunk-size N --stream-results` scores
a source larger than RAM chunk by chunk, so it never has a whole payload to hand
over. It calls one of two streaming methods instead, each taking a header and a
lazy iterator:

| Run | Method | The iterator yields |
|-----|--------|---------------------|
| `--stream-results` | `export_cli_streaming(header, records, field_values)` | `(detector_name, hit)` tuples: each hit as in `export_find_results`, plus a `"label"` |
| `--stream-results --as-labelset` | `export_labelset_streaming(header, elements, field_values)` | label entries: `LabeledElement` dicts, each with a `"detector"` key |

The records arrive in chunk order, each chunk's sorted. With `--max-results N`
(#5279) the header carries `"max_results": N` and they arrive only once the
last chunk is scored: each detector's best N, detector by detector, sorted by
score. Either way an exporter that writes what it is given in the order it is
given needs nothing more.

Write each record as it arrives; never collect them. A file exporter writes
through `vtscore.utils.io.atomic_write_stream`, and a delivery exporter (a POST, an
email) groups the stream with `iter_stream_batches(items, batch_size)` from
`vtscore.exporters.base`, after `resolve_stream_batch_size` has coerced its
`batch_size` field.

The labelset stream is the one to implement if what you deliver should read
back in as labels. Its header carries `media_type`, the `detectors` names and
`keep_negatives`, and none of the head's numbers. Each entry has been through
`label_entry_for_export`, with no score and no media id, so an export can
rebuild the detector under another embedder (#5131). Every detector's entries
arrive in the one stream, an item two detectors score once per detector, so
keep the `detector` key if your format groups by it. The built-in file
exporters' output re-imports through the label importer of the same name:
`server_json_file` writes `{"labels": [...]}` with the metadata beside it, and
`server_csv_file` writes the label CSV's columns after a leading `detector`.

Neither method is a payload kind. The GUI never streams, so
`supported_payloads` and the pickers ignore both; `supports_streaming` is
declared, and `supports_labelset_streaming` is derived from your override.

### Resumable streams

A `--stream-results --resume` run can be killed and run again, continuing
after the last chunk it wrote. It calls
`export_cli_streaming_resumable(header, chunks, field_values, resume)` in place
of `export_cli_streaming`: the same header and hits, but `chunks` yields one
list of `(detector_name, hit)` tuples per chunk (possibly empty), and `resume`
is a `vtscore.exporters.base.StreamResume`. The CLI keeps a checkpoint beside
the file `resumable_stream_path(field_values)` names and never asks the
exporter to read it; the exporter's side is three rules:

- `resume.position is None`: write the destination afresh, header first, then
  call `resume.commit(position)`.
- Otherwise do not write the header. Roll the destination back to
  `resume.position`, which is what this method last committed, and append.
- After each chunk is durable, call `resume.commit(position)`. Never commit a
  chunk only partly written.

For a text file, `vtscore.utils.io.resumable_write_stream(path, position)` opens the
file (truncating a fresh one, or cutting a resumed one back to `position`) and
`vtscore.utils.io.commit_write_stream(f)` flushes, `fsync`s and returns the byte
offset to commit. The built-in `server_json_file` and `server_csv_file` are
both under thirty lines on top of them. Write in place, not through
`atomic_write_stream`: an atomic stream leaves nothing behind to resume.

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

The value must resolve to an already-instantiated `ResultsExporter`.
After `pip install`, the exporter appears in `list_exporters()`, the
`/api/exporters` endpoint, the CLI's `--exporter` flag, and `python
app.py --list-plugins`.

## Worked example

A minimal third-party exporter that POSTs labels (and only labels - it
refuses a scored run) to a webhook with HMAC signing.

```python
# my_pkg/exporter.py
from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

from vtscore.exporters.base import PluginField, ResultsExporter
from vtscore.security.url_validation import guarded_session


class SignedWebhookExporter(ResultsExporter):
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

    # Declaring only this method is what makes the exporter labelset-only:
    # `supported_payloads` is derived from the overrides, so it never appears
    # in a find-results picker and the route refuses a scored run with a 400.
    # No hand-written shape check needed.
    def export_labelset(self, labelset: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        # Both fields are declared and required, so by this point they are
        # stripped, non-empty, and (for the "url" field type) SSRF-checked.
        url = field_values["url"]
        secret = field_values["hmac_secret"].encode("utf-8")

        body = json.dumps(labelset, sort_keys=True).encode("utf-8")
        signature = hmac.new(secret, body, hashlib.sha256).hexdigest()

        with guarded_session() as session:
            resp = session.post(
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

        n = len(labelset["labels"])
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

After `pip install`, the exporter appears in the Export modal's labelset
picker. It has no CLI use: `--autodetect --exporter` hands an exporter a
scored run, which this one refuses with `UnsupportedPayloadError`.

## Testing pattern

Library-tier exporter tests belong in `tests_lib/io/`. The autouse
fixtures (see [`tests_lib/conftest.py`](../../../tests_lib/conftest.py))
reset library-tier global state and stub embedders; `_allow_test_tmp_paths`
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

        class _Session:
            def __enter__(self): return self
            def __exit__(self, *exc): return False
            def post(self, url, data, headers, timeout, allow_redirects):
                captured.update(url=url, data=data, headers=headers)
                return _Resp()

        # Call the payload method directly: it takes already-normalized
        # field_values, so a unit test can hand it a plain dict without
        # going through a route. (The legacy export() is not implemented
        # on this class and would raise NotImplementedError.)
        with patch("my_pkg.exporter.guarded_session", _Session):
            result = exp.export_labelset(results, {
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
        from vtscore.exporters.base import UnsupportedPayloadError

        exp = get_exporter("signed_webhook")
        assert exp.supported_payloads == frozenset({"labelset"})
        import pytest
        with pytest.raises(UnsupportedPayloadError):  # a ValueError subclass
            exp.export_find_results({"media_type": "audio", "results": {}}, {})
```

See [`tests_lib/io/test_importers.py`](../../../tests_lib/io/test_importers.py)
for the general I/O-plugin test pattern and the existing
[`vtscore/exporters/webhook/__init__.py`](../../exporters/webhook/__init__.py)
+ its app-tier tests for a real working webhook exporter.
