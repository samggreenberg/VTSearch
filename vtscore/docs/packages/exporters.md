# `vtscore.exporters`

Result/labelset exporters: plugins that take a labelset or an
autodetect-results dict and deliver it somewhere - a file on disk, an
HTTP webhook, an email, an external labelling service. Every exporter is one
subclass of `LabelsetExporter` plus a module-level `EXPORTER` sentinel,
discovered by the standard `vtscore.plugins` registry. External code
adds more by either dropping a module under `vtscore/exporters/<name>/`
or declaring an entry point in the `vtscore.exporters` group.

Label *importers* (the reverse direction - pulling labels in from an
external source) are not here; they live in
[`vtscore.labels.importers`](../../labels/importers/). Labelset
*sources* (bidirectional sync) live in
[`vtscore.labels.sources`](../../labels/sources/) and are built on the
[`vtscore.sync.SyncSource`](sync.md) ABC.

## Contents

| Module | Concern |
|--------|---------|
| `vtscore/exporters/base.py` | The `LabelsetExporter` ABC and its siblings |
| `vtscore/exporters/__init__.py` | The auto-discovering registry and its accessors |
| `vtscore/exporters/server_json_file/` | Write results to a `.json` file on the server |
| `vtscore/exporters/server_csv_file/` | Write results to a `.csv` file on the server |
| `vtscore/exporters/webhook/` | POST results to an arbitrary URL |
| `vtscore/exporters/email_smtp/` | Email results directly via MX lookup |
| `vtscore/exporters/gui/` | Show results in the browser, or print them on the CLI |
| `vtscore/exporters/open_url/` | Format the labelset into a URL for the browser to open |
| `vtscore/exporters/portable_detector/` | Write standalone ONNX scoring bundles headlessly |

- [Registry and accessors](#registry-and-accessors)
- [`LabelsetExporter` ABC](#labelsetexporter-abc)
- [The export contract](#the-export-contract)
- [`PluginField`](#pluginfield)
- [Built-in exporters](#built-in-exporters)
- [Template variables in path fields](#template-variables-in-path-fields)
- [Writing a custom exporter](#writing-a-custom-exporter)

## Registry and accessors

`vtscore/exporters/__init__.py` is a one-line registry built with
`make_plugin_registry`:

```python
from vtscore.plugins import make_plugin_registry

get_exporter, list_exporters = make_plugin_registry(
    package=__name__,
    sentinel="EXPORTER",
    label="labelset exporter",
    entry_point_group="vtscore.exporters",
)
```

Discovery is **eager** - by the time `from vtscore.exporters import
list_exporters` returns, every sub-package under
`vtscore/exporters/` exposing an `EXPORTER` attribute is registered,
and the `vtscore.exporters` entry-point group has been scanned. Failed
imports emit a `UserWarning` and are skipped; broken third-party entry
points warn and are skipped; built-ins win on name clash. See
[`plugins.md`](plugins.md) for the underlying mechanics.

```python
from vtscore.exporters import get_exporter, list_exporters

exporter = get_exporter("server_json_file")
print([e.name for e in list_exporters()])
# ['email_smtp', 'gui', 'open_url', 'portable_detector', 'server_csv_file', 'server_json_file', 'webhook']
```

`get_exporter(name)` returns `None` when the name is unknown - it does
*not* raise `KeyError`. Callers that want a hard failure should check
the result.

## `LabelsetExporter` ABC

`vtscore/exporters/base.py` - abstract base class. Subclasses set
the standard `PluginBase` class attributes (`name`, `display_name`,
`description`, `icon`, `fields`, optionally `ui_mode` and
`hidden_from_picker`) and implement `export()`. The default `icon` is
`"\U0001f4e4"` (outbox tray).

```python
from vtscore.exporters.base import LabelsetExporter, PluginField


class MyExporter(LabelsetExporter):
    name = "my_exporter"
    display_name = "My Exporter"
    description = "..."
    fields = [PluginField("path", "Path", "server_path")]

    def export(self, results: dict, field_values: dict) -> dict:
        ...
        return {"message": "..."}
```

`LabelsetExporter` inherits `PluginBase`, so CLI flags are auto-derived
from `fields` via `add_cli_arguments()`, JSON metadata comes from
`to_dict()`, and required-field validation comes from
`validate_cli_field_values()`. See [`plugins.md`](plugins.md#pluginbase)
for the inherited surface.

## The export contract

```python
exporter.export(results: dict, field_values: dict) -> dict
```

`results` is one of two shapes - the exporter detects which:

- **Autodetect results** (from `/api/auto-detect` or the CLI
  `--autodetect` flow):

  ```python
  {
      "media_type": "audio",
      "detectors_run": 2,
      "results": {
          "detector_name": {
              "detector_name": "...",
              "threshold": 0.5,
              "total_hits": 15,
              "hits": [{...}, ...],
          },
          ...
      },
  }
  ```

- **Labels** (from the `/api/labels/export` enrich flow):

  ```python
  {
      "labels": [
          {"label": "good", "md5": "...", "filename": "...", "origin": {...}, ...},
          ...
      ],
      "selected_columns": ["label", "filename", "origin", ...],   # optional
  }
  ```

The convention is `if "labels" in results: ...` to discriminate. The
built-in JSON/CSV/webhook/email exporters all handle both shapes;
custom exporters that only target one of them should raise
`ValueError` on the other.

### Return value

`export()` returns a `dict` that **must** include a `"message"` key - a
short human-readable confirmation string. The exporter may also include
arbitrary extra keys (`"filepath"` for file-based exporters,
`"status_code"` and `"url"` for the webhook, an external package id for a
service exporter, …). The route handler renders the message back to the user; the extra
keys are passed through unchanged.

Two extra keys are *interpreted* by the frontend rather than merely
passed through:

| Key | Effect |
|-----|--------|
| `"display_results"` | The results dict is rendered in the Auto-Detect Results modal. Used by `gui`. |
| `"open_url"` | An `http(s)` URL the browser opens in a new tab. |

### `open_url`: handing the user off to another site

An exporter can end by sending the user somewhere instead of (or as well
as) delivering the labelset. That is how a third-party site with **no
ingest API** receives a selection: you can't POST to it, but you can link
into it, because its viewer takes identifiers in the query string. Format
the labelset into that site's own URL and return it as `"open_url"`; the
Export modal opens it in a new tab.

It also fits a delivery exporter whose remote hands back a permalink to
what was just uploaded - return the permalink and the user lands on it.

Two things to know when returning one:

- **Set `opens_url = True`** on the exporter class if it *always* returns
  a URL. The flag rides `GET /api/exporters` and is what lets the button
  read "Open Labelset in `<name>`" *before* the export runs. An
  `open_url` from an exporter that leaves the flag `False` still opens;
  it just can't be advertised up front.
- **The route re-validates the URL** with
  `vtscore.security.url_validation.validate_browser_url` and fails the
  export (500) if it doesn't pass, so no plugin can push a `javascript:`
  URL into the browser. That guard is a *scheme allowlist*, deliberately
  not the `validate_url` SSRF guard: the browser makes this request, not
  the server, so `http://localhost:9000/viewer` is a legitimate target
  and resolving the host would buy nothing. See
  [`security`](security.md#browser-url-validation).

Whatever you encode into the URL is visible to the destination site and
lands in the user's browser history, so keep it to identifiers.

### CLI variant

`LabelsetExporter.export_cli(results, field_values)` defaults to
delegating to `export()`. Override it when CLI invocation needs a
different behaviour - the GUI exporter does this so it can print to
stdout instead of asking the (nonexistent) frontend to render.

## `PluginField`

`vtscore/exporters/base.py` re-exports `PluginField` so exporters can
import it alongside `LabelsetExporter`:

```python
from vtscore.exporters.base import LabelsetExporter, PluginField
```

Field semantics - `field_type` literals, `dynamic_options`,
`depends_on`, number-field type inference - are documented in detail in
[`plugins.md#pluginfield`](plugins.md#pluginfield).

A `dynamic_options` select is served by
`POST /api/exporters/field-options/<name>`, which calls the exporter's
`get_field_options(field_key, current_values)`. Both surfaces that render
an exporter's fields use it - the app's Export modal and its Auto-Find
results-exporter settings - so an exporter whose destinations are only
knowable at runtime fills its dropdown in either place.

## Built-in exporters

| Name | Target | Notes |
|------|--------|-------|
| `server_json_file` | Writes a JSON file to the server filesystem | Atomic write via tmp + rename; supports `{YYYYMMDD-HHMMSS}` / `{YYYYMMDD}` / `{YYYY}` / `{MM}` / `{DD}` / `{detector_name}` / `{username}` template variables in the path; default path under `DATA_DIR` |
| `server_csv_file` | Writes a CSV file to the server filesystem | Atomic write; auto-detects which optional clip columns (`clip_start`, `clip_end`, `clip_box`) are present; cells beginning with `=`/`+`/`-`/`@`/`\t`/`\r` are quote-prefixed to defeat formula injection |
| `webhook` | `POST`s the results dict as JSON to a URL | Optional `Authorization` header (`password` field), 30s timeout, redirects disabled, URL validated by `vtscore.security.validate_url` (SSRF guard) |
| `open_url` | Formats the labelset into a URL and returns it as `open_url` for the browser to open in a new tab | `opens_url = True`. No network call server-side; substitutes `{ids}` / `{count}` into a user-supplied template, URL-encoding the joined identifiers. Truncates to `max_items` (reported in the message) and refuses a URL over ~2000 characters |
| `email_smtp` | Sends an email via direct MX delivery | Resolves the recipient domain's MX record (`dnspython`), connects on port 25, sends a multipart plain+HTML summary. Requires a sender domain you control |
| `gui` | Displays results in the browser (GUI) or prints to stdout (CLI) | `hidden_from_picker = True`. The default exporter for the web UI's autodetect modal; in CLI mode `export_cli()` prints origin + name of each Good hit |
| `portable_detector` | Writes one standalone ONNX scoring bundle per trained detector | `hidden_from_picker = True`, CLI-only. See below - it is the one exporter that consumes detectors rather than results |

`hidden_from_picker = True` keeps an exporter out of the generic
picker UI. The `gui` exporter is special-cased by the frontend; the
`portable_detector` exporter is hidden because the GUI has its own
dedicated portable-export modal.

### `portable_detector` is shaped differently

Every other exporter consumes the **scored results**. This one consumes
the **trained classifiers**, so it sets `needs_trained_detectors` and
the pipeline hands it the detectors via `export_cli_detectors` instead
of the usual `export` call. It only works on the `--autodetect` /
`--pipeline` path, which is the only one that produces a trained head.

It is also the sanctioned exception to the "No Persisted Vectors or
MLPs" rule: the bundle persists the trained MLP (as ONNX, alongside a
`manifest.json` and a `README.md`) so a third party can score their own
media without VTSearch. It never writes embeddings or raw media. The
bundle itself is built by `vtscore.detectors.portable_bundle`.

Two per-embedder-type caveats: **structural** (SIFT/VLAD) detectors are
skipped with a note rather than aborting the export, because their
stage-2 RANSAC verification isn't representable as a scoring-only ONNX
graph; **patch** (DINOv2/v3, EUPE) detectors export normally but in a
degraded whole-item-only scoring mode. Use `{detector_name}` in the
path to disambiguate a multi-detector run.

### File-format notes

- **`server_json_file`** (`vtscore/exporters/server_json_file/__init__.py`)
  writes either the full autodetect-results JSON or, when the input is
  a `labels` payload, an object filtered to the user-selected columns.
  Atomic write helper is in `vtscore/exporters/server_json_file/__init__.py`.
- **`server_csv_file`** (`vtscore/exporters/server_csv_file/__init__.py`)
  produces one row per hit (autodetect path) or one row per label
  (labels path). The labels path always re-orders `origin` to the last
  column so the file can be re-imported losslessly.
- **`webhook`** (`vtscore/exporters/webhook/__init__.py`) sends the
  full `results` dict as the JSON body. Returned dict includes
  `status_code` and `url` alongside `message`.
- **`email_smtp`** (`vtscore/exporters/email_smtp/__init__.py`)
  composes both a plain-text and HTML body and sends both as alternative
  MIME parts. Requires the `dnspython` package for MX resolution; the
  import is lazy so installs that don't use email are unaffected.

## Template variables in path fields

A path field declares which placeholders it accepts in its
`PluginField.template_vars` tuple; the framework's
`normalize_field_values` pass (`vtscore/plugins/normalize.py`)
substitutes them - and confines the resolved `server_path` to the
user's data dir - before `export()` runs:

| Placeholder | Substituted with |
|-------------|------------------|
| `{YYYYMMDD-HHMMSS}` | Current UTC timestamp, e.g. `20260516-143022` - included in the default path so consecutive runs do not silently overwrite each other |
| `{YYYYMMDD}` / `{YYYY}` / `{MM}` / `{DD}` | Current UTC date parts, so a scheduled (e.g. daily) Auto-Find can write to a path named after today's date, e.g. `results_{YYYY}.{MM}.{DD}.csv` |
| `{detector_name}` | The active `DetectorContext.name`, sanitised by `vtscore.security.sanitize_template_value` |
| `{username}` | The current request user (from `vtscore.state.current_user.get_current_user`), sanitised the same way; falls back to `"default"` |

Defaults for `server_json_file` and `server_csv_file` already
interpolate from `DATA_DIR`:

```python
_DEFAULT_JSON_PATH = f"{DATA_DIR}/autodetect_results_{{YYYYMMDD-HHMMSS}}.json"
_DEFAULT_CSV_PATH  = f"{DATA_DIR}/autodetect_results_{{YYYYMMDD-HHMMSS}}.csv"
```

This is part of the Phase 4 filesystem-seam work: every path placeholder
resolves against `vtscore.config.DATA_DIR` (which honours
`$VTSEARCH_DATA_DIR`) so plugin defaults are absolute paths rather than
implicit-cwd relative paths. Custom exporters writing path defaults
should follow the same pattern.

The template resolver reads `vtscore.state.current_user.get_current_user`
(app-tier wires the request-scoped resolver via
`register_request_user_resolver`; the library-side default is `"default"`),
so `{username}` interpolation is only useful when a resolver or the
thread-local has been set. Library-tier callers that drive an exporter
directly should either avoid `{username}` in their paths or register a
user resolver before invoking `export()`.

## Writing a custom exporter

Drop a sub-package under `vtscore/exporters/<name>/` (or expose it as
an entry point - see [`plugins.md`](plugins.md#entry-point-integration)
for the third-party path):

```python
# vtscore/exporters/sftp/__init__.py
from __future__ import annotations
from typing import Any

import paramiko

from vtscore.exporters.base import LabelsetExporter, PluginField


class SftpLabelsetExporter(LabelsetExporter):
    name = "sftp"
    display_name = "SFTP Upload"
    description = "Upload the results JSON to a remote SFTP server."
    icon = "\U0001f4e1"
    fields = [
        PluginField("host",     "Hostname",    "text"),
        PluginField("user",     "Username",    "text"),
        PluginField("password", "Password",    "password"),
        PluginField(
            "path", "Remote Path", "text",
            default="/results/autodetect.json",
        ),
    ]

    def export(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        import json

        host = field_values["host"]
        path = field_values["path"]

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(host, username=field_values["user"], password=field_values["password"])
        sftp = ssh.open_sftp()
        with sftp.open(path, "w") as f:
            f.write(json.dumps(results, indent=2))
        sftp.close()
        ssh.close()

        return {"message": f"Uploaded to {host}:{path}"}


EXPORTER = SftpLabelsetExporter()
```

### Checklist

- Subclass `LabelsetExporter`; set `name`, `display_name`,
  `description`, `fields`.
- Implement `export(results, field_values) -> dict` returning at
  minimum `{"message": "..."}`.
- Decide which result shape(s) you support; raise `ValueError` on
  shapes you don't.
- For file destinations, declare the placeholders you accept in the
  field's `template_vars` and let `normalize_field_values` substitute
  them; default the path under `vtscore.config.DATA_DIR`. Write through
  `vtscore.io.atomic_write_bytes` / `atomic_write_text` rather than
  hand-rolling the tmp-file + `os.replace` ritual.
- For URL destinations, run the URL through
  `vtscore.security.validate_url` first (SSRF guard).
- Expose a module-level `EXPORTER = YourExporter()` constant.
- (Third-party only) declare a `vtscore.exporters` entry point in
  your `pyproject.toml`.

## Cross-references

- [`plugins.md`](plugins.md) - registry mechanics, sentinels, entry
  points, `PluginField` reference, schema helpers.
- [`sync.md`](sync.md) - the `SyncSource` ABC behind labelset/settings
  sources, the bidirectional-sync counterparts to exporters.
- Repo-level [`docs/EXTENDING-plugins.md`](../../../docs/EXTENDING-plugins.md)
  has the app-tier perspective and walks through the HTTP routes that
  invoke exporters.
