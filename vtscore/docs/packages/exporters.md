# `vtscore.exporters`

Results exporters: plugins that take a scored run, a detector's
labelset, or the trained detectors themselves, and deliver it somewhere -
a file on disk, an HTTP webhook, an email, an external labelling service.
Every exporter is one subclass of `ResultsExporter` plus a module-level
`EXPORTER` sentinel, discovered by the standard `vtscore.plugins`
registry. External code adds more by either dropping a module under
`vtscore/exporters/<name>/` or declaring an entry point in the
`vtscore.exporters` group.

**Writing one is documented in
[`extending/results-exporters.md`](../extending/results-exporters.md).**
That guide owns the authoring contract - payload kinds, template
variables, validation, packaging, a worked example. This page is the map
of what the package already contains: the registry API, the built-in
exporters, and their per-destination quirks. It deliberately does not
restate the contract; a second copy is a second thing to rot.

Label *importers* (the reverse direction - pulling labels in from an
external source) are not here; they live in
[`vtscore.labels.importers`](../../labels/importers/). Labelset
*sources* (bidirectional sync) live in
[`vtscore.labels.sources`](../../labels/sources/) and are built on the
[`vtscore.sync.SyncSource`](sync.md) ABC.

## Contents

| Module | Concern |
|--------|---------|
| `vtscore/exporters/base.py` | The `ResultsExporter` ABC, `PAYLOAD_KINDS`, and their siblings |
| `vtscore/exporters/__init__.py` | The auto-discovering registry and its accessors |
| `vtscore/exporters/server_json_file/` | Write results to a `.json` file on the server |
| `vtscore/exporters/server_csv_file/` | Write results to a `.csv` file on the server |
| `vtscore/exporters/webhook/` | POST results to an arbitrary URL |
| `vtscore/exporters/email_smtp/` | Email results directly via MX lookup |
| `vtscore/exporters/gui/` | Show results in the browser, or print them on the CLI |
| `vtscore/exporters/open_url/` | Format the labelset into a URL for the browser to open |
| `vtscore/exporters/portable_detector/` | Write standalone ONNX scoring bundles headlessly |

- [Registry and accessors](#registry-and-accessors)
- [`ResultsExporter` ABC](#resultsexporter-abc)
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

## `ResultsExporter` ABC

`vtscore/exporters/base.py` - abstract base class. Subclasses set the
standard `PluginBase` class attributes (`name`, `display_name`,
`description`, `icon`, `fields`, optionally `ui_mode` and
`hidden_from_picker`) and implement one method per **payload kind** they
support. The default `icon` is `"\U0001f4e4"` (outbox tray).

```python
from vtscore.exporters.base import PluginField, ResultsExporter


class MyExporter(ResultsExporter):
    name = "my_exporter"
    display_name = "My Exporter"
    description = "..."
    fields = [PluginField("path", "Path", "server_path")]

    def export_find_results(self, results: dict, field_values: dict) -> dict:
        ...
        return {"message": "..."}
```

An exporter is a *destination*; what gets sent there is a separate axis.
`PAYLOAD_KINDS` names the three:

| Kind | Method | What it is |
|------|--------|------------|
| `find_results` | `export_find_results()` | a scored run - hit lists, scores, thresholds |
| `labelset` | `export_labelset()` | a detector's labels - origins and vote provenance |
| `detector_bundles` | `export_cli_detectors()` | the trained classifiers themselves |

`supported_payloads` is **derived** from which of those methods the
subclass overrode - never declared - and is what each picker filters on,
so an exporter is only offered for the kinds it can actually read.
Handing it any other kind raises `UnsupportedPayloadError` (a
`ValueError` subclass, so the route answers 400 rather than 500).

`ResultsExporter` inherits `PluginBase`, so CLI flags are auto-derived
from `fields` via `add_cli_arguments()`, JSON metadata comes from
`to_dict()`, and required-field validation comes from
`validate_cli_field_values()`. See [`plugins.md`](plugins.md#pluginbase)
for the inherited surface.

The full authoring contract - the payload dict shapes, the required
`"message"` return key and the `"display_results"` / `"open_url"` keys
the frontend interprets, template-variable declaration, URL and
server-path validation, entry-point packaging - lives in
[`extending/results-exporters.md`](../extending/results-exporters.md).

### `LabelsetExporter` is a permanent alias

`LabelsetExporter` is a module-level alias for `ResultsExporter`, kept
indefinitely so out-of-tree `from vtscore.exporters.base import
LabelsetExporter` imports and subclasses keep working. The old name
described one of the three payloads the class has always accepted; new
code should use `ResultsExporter`.

The single-method `export(results, field_values)` contract that predates
the payload kinds is likewise still supported: `export_find_results()`
and `export_labelset()` delegate to it when unoverridden, so a plugin
that discriminates the two dict shapes itself (`if "labels" in results`)
needs no changes. Such an exporter is credited with *both* non-detector
kinds - nothing can tell which it handles - and logs an advisory line at
import time pointing at the extending guide. No built-in exporter uses
this path any more; `portable_detector` overrides `export()` only to
explain that hits are the wrong input for it.

### Streaming

The CLI `--stream-results` path (scoring a media source larger than RAM)
asks an exporter to write hits incrementally instead of buffering the
whole run. An exporter opts in with `supports_streaming` and
`export_cli_streaming()`; it is a `find_results` mode only, with no
labelset equivalent. Every built-in that delivers a scored run streams,
except `open_url` - which returns a URL rather than writing anything, so
it has nothing to write incrementally. `portable_detector` is outside
the question entirely: it consumes detectors, not hits.

## `PluginField`

`vtscore/exporters/base.py` re-exports `PluginField` so exporters can
import it alongside `ResultsExporter`:

```python
from vtscore.exporters.base import PluginField, ResultsExporter
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

| Name | Payloads | Target | Notes |
|------|----------|--------|-------|
| `server_json_file` | `find_results`, `labelset` | Writes a JSON file to the server filesystem | Atomic write via tmp + rename; supports the `{YYYYMMDD-HHMMSS}` / `{YYYYMMDD}` / `{YYYY}` / `{MM}` / `{DD}` / `{detector_name}` / `{detector_id}` / `{username}` template variables in the path; default path under `DATA_DIR` |
| `server_csv_file` | `find_results`, `labelset` | Writes a CSV file to the server filesystem | Atomic write; auto-detects which optional clip columns (`clip_start`, `clip_end`, `clip_box`) are present; cells beginning with `=`/`+`/`-`/`@`/`\t`/`\r` are quote-prefixed to defeat formula injection |
| `webhook` | `find_results`, `labelset` | `POST`s the payload dict as JSON to a URL | Optional `Authorization` header (`password` field), 30s timeout, redirects disabled, URL validated by `vtscore.security.validate_url` (SSRF guard) |
| `open_url` | `find_results`, `labelset` | Formats the labelset into a URL and returns it as `open_url` for the browser to open in a new tab | `opens_url = True`. No network call server-side; substitutes `{ids}` / `{count}` into a user-supplied template, URL-encoding the joined identifiers. Truncates to `max_items` (reported in the message) and refuses a URL over ~2000 characters. The one results-carrying built-in that does not stream |
| `email_smtp` | `find_results`, `labelset` | Sends an email via direct MX delivery | Resolves the recipient domain's MX record (`dnspython`), connects on port 25, sends a multipart plain+HTML summary. Requires a sender domain you control |
| `gui` | `find_results`, `labelset` | Displays results in the browser (GUI) or prints to stdout (CLI) | `hidden_from_picker = True`. The default exporter for the web UI's autodetect modal; in CLI mode `export_cli()` prints origin + name of each Good hit |
| `portable_detector` | `detector_bundles` | Writes one standalone ONNX scoring bundle per trained detector | `hidden_from_picker = True`, CLI-only. See below - it is the one exporter that consumes detectors rather than results |

The Payloads column is not a declaration anywhere in the source: each
exporter's `supported_payloads` is derived from the methods it overrides,
so the column is a reading of the code rather than a second copy of it.

`hidden_from_picker = True` keeps an exporter out of the generic
picker UI. The `gui` exporter is special-cased by the frontend; the
`portable_detector` exporter is hidden because the GUI has its own
dedicated portable-export modal.

### `portable_detector` is shaped differently

Every other exporter consumes a detector's **output**. This one consumes
the **trained classifiers**, so it sets `needs_trained_detectors` - which
makes `supported_payloads` report `{"detector_bundles"}` alone - and the
pipeline hands it the detectors via `export_cli_detectors()` instead of
the usual `export_find_results()` / `export_labelset()` call. It only
works on the `--autodetect` / `--pipeline` path, which is the only one
that produces a trained head.

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
  writes either the full `find_results` JSON or, for a `labelset`
  payload, an object filtered to the user-selected columns.
- **`server_csv_file`** (`vtscore/exporters/server_csv_file/__init__.py`)
  produces one row per hit (`find_results`) or one row per label
  (`labelset`). The labelset path always re-orders `origin` to the last
  column so the file can be re-imported losslessly.
- **`webhook`** (`vtscore/exporters/webhook/__init__.py`) sends the
  full payload dict as the JSON body, whichever kind it is. Returned dict includes
  `status_code` and `url` alongside `message`.
- **`email_smtp`** (`vtscore/exporters/email_smtp/__init__.py`)
  composes both a plain-text and HTML body and sends both as alternative
  MIME parts. Requires the `dnspython` package for MX resolution; the
  import is lazy so installs that don't use email are unaffected.

## Template variables in path fields

A path field declares which placeholders it accepts in its
`PluginField.template_vars` tuple; the framework's
`normalize_field_values` pass (`vtscore/plugins/normalize.py`)
substitutes them - and confines the resolved `server_path` to the user's
data dir - before the exporter runs. The supported placeholders, why you
must declare rather than substitute them, and what the declaration buys
you in the GUI are all in
[`extending/results-exporters.md`](../extending/results-exporters.md#template-variable-interpolation).

What is specific to this package: the defaults for `server_json_file`
and `server_csv_file` already interpolate from `DATA_DIR`.

```python
_DEFAULT_JSON_PATH = f"{DATA_DIR}/autodetect_results_{{YYYYMMDD-HHMMSS}}.json"
_DEFAULT_CSV_PATH  = f"{DATA_DIR}/autodetect_results_{{YYYYMMDD-HHMMSS}}.csv"
```

This is part of the Phase 4 filesystem-seam work: every path placeholder
resolves against `vtscore.config.DATA_DIR` (which honours
`$VTSEARCH_DATA_DIR`) so plugin defaults are absolute paths rather than
implicit-cwd relative paths. Custom exporters writing path defaults
should follow the same pattern. The `{YYYYMMDD-HHMMSS}` stamp is in both
defaults so consecutive runs do not silently overwrite each other.

The template resolver reads `vtscore.state.current_user.get_current_user`
(app-tier wires the request-scoped resolver via
`register_request_user_resolver`; the library-side default is `"default"`),
so `{username}` interpolation is only useful when a resolver or the
thread-local has been set. Library-tier callers that drive an exporter
directly should either avoid `{username}` in their paths or register a
user resolver before invoking it.

## Writing a custom exporter

**Start from
[`extending/results-exporters.md`](../extending/results-exporters.md).**
It carries the full walkthrough - a worked third-party exporter, the
payload-method split, template variables, URL and server-path
validation, entry-point packaging, and the testing pattern - and it is
the copy that stays current, because
`scripts/check-extension-docs.py` holds its contract table to the
members `ResultsExporter` actually defines.

Only two things are specific to adding an exporter *to this package*
rather than shipping one from your own distribution:

- Drop a sub-package under `vtscore/exporters/<name>/` exposing a
  module-level `EXPORTER = YourExporter()`. Discovery is by directory,
  so there is no registration list to edit and no entry point to
  declare - the entry-point route in the guide is for out-of-tree
  distributions.
- Write files through `vtscore.io.atomic_write_bytes` /
  `atomic_write_text` - or `atomic_write_stream` when the exporter
  streams its rows - rather than hand-rolling the tmp-file +
  `os.replace` ritual, and default any path field under
  `vtscore.config.DATA_DIR` as the built-ins do (see [Template variables
  in path fields](#template-variables-in-path-fields)). New
  dependencies go in `[project.dependencies]` in the repo's
  `pyproject.toml`; deptry gates that.

## Cross-references

- [`plugins.md`](plugins.md) - registry mechanics, sentinels, entry
  points, `PluginField` reference, schema helpers.
- [`sync.md`](sync.md) - the `SyncSource` ABC behind labelset/settings
  sources, the bidirectional-sync counterparts to exporters.
- Repo-level [`docs/EXTENDING-plugins.md`](../../../docs/EXTENDING-plugins.md)
  has the app-tier perspective and walks through the HTTP routes that
  invoke exporters.
