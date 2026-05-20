# `vtscore.exporters`

Result/labelset exporters: plugins that take a labelset or an
autodetect-results dict and deliver it somewhere — a file on disk, an
HTTP webhook, an email, a Holder package. Every exporter is one
subclass of `LabelsetExporter` plus a module-level `EXPORTER` sentinel,
discovered by the standard `vtscore.plugins` registry. The package
ships six built-ins (`server_json_file`, `server_csv_file`, `webhook`,
`email_smtp`, `gui`, `holder`) and external code adds more by either
dropping a module under `vtscore/exporters/<name>/` or declaring an
entry point in the `vtscore.exporters` group.

Label *importers* (the reverse direction — pulling labels in from an
external source) are not here; they live in
[`vtscore.labels.importers`](../../labels/importers/). Labelset
*sources* (bidirectional sync) live in
[`vtscore.labels.sources`](../../labels/sources/) and are built on the
[`vtscore.sync.SyncSource`](sync.md) ABC.

## Contents

- [Registry and accessors](#registry-and-accessors)
- [`LabelsetExporter` ABC](#labelsetexporter-abc)
- [The export contract](#the-export-contract)
- [`ExporterField`](#exporterfield)
- [Built-in exporters](#built-in-exporters)
- [Template variables in path fields](#template-variables-in-path-fields)
- [Writing a custom exporter](#writing-a-custom-exporter)

## Registry and accessors

`vtscore/exporters/__init__.py:17` is a one-line registry built with
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

Discovery is **eager** — by the time `from vtscore.exporters import
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
# ['email_smtp', 'gui', 'holder', 'server_csv_file', 'server_json_file', 'webhook']
```

`get_exporter(name)` returns `None` when the name is unknown — it does
*not* raise `KeyError`. Callers that want a hard failure should check
the result.

## `LabelsetExporter` ABC

`vtscore/exporters/base.py:59` — abstract base class. Subclasses set
the standard `PluginBase` class attributes (`name`, `display_name`,
`description`, `icon`, `fields`, optionally `ui_mode` and
`hidden_from_picker`) and implement `export()`. The default `icon` is
`"\U0001f4e4"` (outbox tray).

```python
from vtscore.exporters.base import LabelsetExporter, ExporterField


class MyExporter(LabelsetExporter):
    name = "my_exporter"
    display_name = "My Exporter"
    description = "..."
    fields = [ExporterField("path", "Path", "server_path")]

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

`results` is one of two shapes — the exporter detects which:

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

`export()` returns a `dict` that **must** include a `"message"` key — a
short human-readable confirmation string. The exporter may also include
arbitrary extra keys (`"filepath"` for file-based exporters,
`"status_code"` and `"url"` for the webhook, `"holder_id"` for Holder,
…). The route handler renders the message back to the user; the extra
keys are passed through unchanged.

### CLI variant

`LabelsetExporter.export_cli(results, field_values)` defaults to
delegating to `export()`. Override it when CLI invocation needs a
different behaviour — the GUI exporter does this so it can print to
stdout instead of asking the (nonexistent) frontend to render.

## `ExporterField`

A backwards-compatibility alias for `PluginField`:

```python
# vtscore/exporters/base.py:54
ExporterField = PluginField
```

Use whichever name reads better at the call site; the two are
literally identical. Field semantics — `field_type` literals,
`dynamic_options`, `depends_on`, number-field type inference — are
documented in detail in [`plugins.md#pluginfield`](plugins.md#pluginfield).

## Built-in exporters

| Name | Target | Notes |
|------|--------|-------|
| `server_json_file` | Writes a JSON file to the server filesystem | Atomic write via tmp + rename; supports `{YYYYMMDD-HHMMSS}` / `{detector_name}` / `{username}` template variables in the path; default path under `DATA_DIR` |
| `server_csv_file` | Writes a CSV file to the server filesystem | Atomic write; auto-detects which optional clip columns (`clip_start`, `clip_end`, `clip_box`) are present; cells beginning with `=`/`+`/`-`/`@`/`\t`/`\r` are quote-prefixed to defeat formula injection |
| `webhook` | `POST`s the results dict as JSON to a URL | Optional `Authorization` header (`password` field), 30s timeout, redirects disabled, URL validated by `vtscore.security.validate_url` (SSRF guard) |
| `email_smtp` | Sends an email via direct MX delivery | Resolves the recipient domain's MX record (`dnspython`), connects on port 25, sends a multipart plain+HTML summary. Requires a sender domain you control |
| `gui` | Displays results in the browser (GUI) or prints to stdout (CLI) | `hidden_from_picker = True`. The default exporter for the web UI's autodetect modal; in CLI mode `export_cli()` prints origin + name of each Good hit |
| `holder` | Creates a new Holder package with Good/Bad folders | `hidden_from_picker = True` until the Holder API client lands. Only labels carrying a `contentID` (typically from a ReCaller import) are written; everything else is silently skipped |

`hidden_from_picker = True` keeps an exporter out of the generic
picker UI. The `gui` exporter is special-cased by the frontend; the
`holder` exporter is a scaffold with `NotImplementedError` stubs for
its Holder API client (see `vtscore/exporters/holder/__init__.py:46`).

### File-format notes

- **`server_json_file`** (`vtscore/exporters/server_json_file/__init__.py:39`)
  writes either the full autodetect-results JSON or, when the input is
  a `labels` payload, an object filtered to the user-selected columns.
  Atomic write helper is at `vtscore/exporters/server_json_file/__init__.py:25`.
- **`server_csv_file`** (`vtscore/exporters/server_csv_file/__init__.py:57`)
  produces one row per hit (autodetect path) or one row per label
  (labels path). The labels path always re-orders `origin` to the last
  column so the file can be re-imported losslessly.
- **`webhook`** (`vtscore/exporters/webhook/__init__.py:16`) sends the
  full `results` dict as the JSON body. Returned dict includes
  `status_code` and `url` alongside `message`.
- **`email_smtp`** (`vtscore/exporters/email_smtp/__init__.py:84`)
  composes both a plain-text and HTML body and sends both as alternative
  MIME parts. Requires the `dnspython` package for MX resolution; the
  import is lazy so installs that don't use email are unaffected.

## Template variables in path fields

File-based exporters interpolate three placeholders in their path
fields via `resolve_export_filepath`
(`vtscore/exporters/_template.py:19`):

| Placeholder | Substituted with |
|-------------|------------------|
| `{YYYYMMDD-HHMMSS}` | Current UTC timestamp, e.g. `20260516-143022` — included in the default path so consecutive runs do not silently overwrite each other |
| `{detector_name}` | The active `DetectorContext.name`, sanitised by `vtscore.security.sanitize_template_value` |
| `{username}` | The current request user (from `vtsearch.auth.get_current_user`), sanitised the same way; falls back to `"default"` |

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

The template resolver imports `vtsearch.auth.get_current_user`, so
`{username}` interpolation is only meaningful inside the app process.
Library-tier callers that drive an exporter directly should either
avoid `{username}` in their paths or set up an auth provider before
invoking `export()`.

## Writing a custom exporter

Drop a sub-package under `vtscore/exporters/<name>/` (or expose it as
an entry point — see [`plugins.md`](plugins.md#entry-point-integration)
for the third-party path):

```python
# vtscore/exporters/sftp/__init__.py
from __future__ import annotations
from typing import Any

import paramiko

from vtscore.exporters.base import LabelsetExporter, ExporterField


class SftpLabelsetExporter(LabelsetExporter):
    name = "sftp"
    display_name = "SFTP Upload"
    description = "Upload the results JSON to a remote SFTP server."
    icon = "\U0001f4e1"
    fields = [
        ExporterField("host",     "Hostname",    "text"),
        ExporterField("user",     "Username",    "text"),
        ExporterField("password", "Password",    "password"),
        ExporterField(
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
- For file destinations, use
  `vtscore.exporters._template.resolve_export_filepath` for template
  variables and default the path under `vtscore.config.DATA_DIR`.
  Write atomically (tmp file + `os.replace`) — the two server-file
  exporters have helpers worth copying.
- For URL destinations, run the URL through
  `vtscore.security.validate_url` first (SSRF guard).
- Expose a module-level `EXPORTER = YourExporter()` constant.
- (Third-party only) declare a `vtscore.exporters` entry point in
  your `pyproject.toml`.

## Cross-references

- [`plugins.md`](plugins.md) — registry mechanics, sentinels, entry
  points, `PluginField` reference, schema helpers.
- [`sync.md`](sync.md) — the `SyncSource` ABC behind labelset/settings
  sources, the bidirectional-sync counterparts to exporters.
- Repo-level [`docs/EXTENDING-plugins.md`](../../../docs/EXTENDING-plugins.md)
  has the app-tier perspective and walks through the HTTP routes that
  invoke exporters.
