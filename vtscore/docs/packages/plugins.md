# `vtscore.plugins`

The extensibility framework that every other plugin family in `vtscore`
builds on. It bundles three things: a generic auto-discovering registry
(`PluginRegistry`), a shared mixin every plugin subclasses (`PluginBase`),
and a uniform field-descriptor dataclass (`PluginField`) used to declare
the configurable inputs each plugin exposes. Plus an inventory layer
(`vtscore.plugins.inventory`) that lets tooling enumerate every installed
plugin across every family in one call, and a small marshmallow schema
builder (`vtscore.plugins.schema`) used by HTTP routes to validate
incoming plugin payloads.

If you only use `vtscore` for its dataset / detector primitives you can
ignore this package. If you want to ship a third-party importer,
exporter, converter, media source, or sync source, this is the surface
you implement against.

**Related package docs:** [exporters](exporters.md) ·
[sync](sync.md). The repo-level [`docs/EXTENDING-plugins.md`](../../../docs/EXTENDING-plugins.md)
covers the same machinery from the app side and walks through one
end-to-end example per family.

## Contents

| Module | Concern |
|--------|---------|
| `vtscore/plugins/__init__.py` | `PluginField`, `PluginBase`, `PluginRegistry`, `make_plugin_registry`, sentinel + entry-point discovery |
| `vtscore/plugins/inventory.py` | Enumerate every installed plugin across every family in one call |
| `vtscore/plugins/normalize.py` | Framework-side normalisation of incoming `field_values` |
| `vtscore/plugins/schema.py` | Build marshmallow schemas from `PluginField` declarations |
| `vtscore/plugins/uploads.py` | The normalised upload type shared by library-tier plugin bases |

- [Architecture in one paragraph](#architecture-in-one-paragraph)
- [`PluginField`](#pluginfield)
  - [Field-value normalization (`vtscore.plugins.normalize`)](#field-value-normalization)
- [`PluginBase`](#pluginbase)
- [`PluginRegistry`](#pluginregistry)
- [`make_plugin_registry()` factory](#make_plugin_registry-factory)
- [Sentinel auto-discovery](#sentinel-auto-discovery)
- [Entry-point integration](#entry-point-integration)
- [Inventory (`vtscore.plugins.inventory`)](#inventory)
- [Schema helpers (`vtscore.plugins.schema`)](#schema-helpers)
- [Field-value normalisation](#field-value-normalisation)
- [File uploads](#file-uploads)
- [End-to-end: writing a third-party plugin](#end-to-end-writing-a-third-party-plugin)

---

## Architecture in one paragraph

Every plugin family - dataset importers, results exporters, label
importers, labelset sources, media sources, media converters,
processor importers, settings importers/exporters/sources - is one
`PluginRegistry` instance over a Python package. The registry scans the
package directory at construction time (eager, by default), imports
each sub-package or flat module, and registers any module-level
sentinel attribute (`IMPORTER`, `EXPORTER`, `LABEL_IMPORTER`, …) as a
plugin keyed by `plugin.name`. It then walks the matching
`importlib.metadata` entry-point group so third-party packages can drop
plugins in without forking the repo. Built-ins take precedence on name
clashes; a third-party entry point whose own import raises warns and is
registered as a tombstone (deferring the original error to first use)
without disturbing the rest of the registry. The result is exposed as a
standard `(get, list)` accessor pair every family re-exports.

## `PluginField`

`vtscore/plugins/__init__.py` - a dataclass that describes one
user-configurable input on a plugin. Each plugin declares its inputs as
`fields: list[PluginField]` on the class; the frontend renders a form
from those, the CLI derives flags from those, and the marshmallow
schema builder validates POST bodies against those. The same dataclass
is used by every family; each family's base module re-exports
`PluginField` so plugins can import it alongside their base class.

### Constructor parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `key` | `str` | - | Dict key used in `field_values` payloads and as the CLI flag stem |
| `label` | `str` | - | Display label shown next to the input in the UI |
| `field_type` | `FieldType` | - | One of the literals listed below |
| `description` | `str` | `""` | Helper text under the field (also fed into the placeholder for CLI `--help`) |
| `accept` | `str` | `""` | For `"file"` fields: comma-separated extensions, e.g. `".pkl,.json"` |
| `options` | `list[str]` | `[]` | For `"select"` fields: allowed dropdown values |
| `default` | `str` | `""` | Pre-filled value; checkboxes use `"true"` / `"false"` |
| `required` | `bool` | `True` | Whether the field must be non-empty after `str.strip()` |
| `placeholder` | `str` | `""` | Placeholder text inside the input widget |
| `hint` | `str` | `""` | Format-hint chip rendered below the input (separate from `description`) |
| `dynamic_options` | `bool` | `False` | `"select"` fields whose options come from `plugin.get_field_options()` at runtime |
| `depends_on` | `list[str]` | `[]` | Other field keys that, when changed, trigger a re-fetch of this field's options |
| `allow_free_text` | `bool` | `False` | `"select"` fields: render as a combobox accepting an arbitrary typed value |
| `min` | `str` | `""` | `"number"` fields: minimum value (string form; empty = unbounded) |
| `max` | `str` | `""` | `"number"` fields: maximum value |
| `step` | `str` | `""` | `"number"` fields: step increment; non-integer step → `float` CLI parsing |
| `clears` | `list[str]` | `[]` | Field keys blanked in the UI when this field gets a non-empty value ("supply A *or* B") |
| `include_in_origin` | `bool \| None` | `None` | Copy into the persisted origin dict. `None` = field-type default (`False` for `"file"` / `"password"`, `True` otherwise) |
| `origin_serializer` | `Callable[[Any], str] \| None` | `None` | Custom value → origin-string conversion for list/dict values |
| `template_vars` | `tuple[str, ...]` | `()` | Template placeholders the framework substitutes into this field's value; see [Field-value normalization](#field-value-normalization) |

### `FieldType` values

Defined as `Literal[...]` in `vtscore/plugins/__init__.py`:

| Value | Frontend widget | Notes |
|-------|-----------------|-------|
| `"file"` | OS file picker | Value arrives as a `werkzeug.datastructures.FileStorage` on the web path; skipped by `vtscore.plugins.schema` and populated from `request.files` |
| `"folder"` | Path text input / OS folder picker | Plain string |
| `"url"` | Text input pre-validated as URL | Plain string |
| `"text"` | Generic single-line input | Plain string |
| `"password"` | Masked text input | Plain string |
| `"email"` | Email input | Plain string; loosely validated |
| `"number"` | Numeric input with min/max/step | Coerced to `int` or `float` by `is_integer_number()` (`vtscore/plugins/__init__.py`) |
| `"select"` | Dropdown | `options` must be populated, or `dynamic_options=True` |
| `"server_path"` | Server filesystem path picker | Confined to the user's data dir by `vtscore.security.confine_server_filepath` in the normalization pass, which writes the approved path back |
| `"checkbox"` | Boolean tickbox | `default` is `"true"` / `"false"`; values arrive coerced via `bool(str(v).lower() == "true")` |

### Number-field type inference

`PluginField.is_integer_number()` returns `True` only when `step`,
`default`, `min`, and `max` all lack a decimal point. That decides
whether the auto-generated CLI argument is parsed as `int` or `float`
and whether the marshmallow schema yields `fields.Integer` or
`fields.Float`. If you want a float field, give any one of those values
a decimal - e.g. `step="0.1"` or `default="1.0"`.

### Dynamic-options fields

A `"select"` field marked `dynamic_options=True` has its options
computed at runtime by the plugin's `get_field_options(field_key,
current_values) -> list[str]` method. List dependencies in
`depends_on=[...]` so the frontend re-fetches whenever any depended-on
field changes.

`PluginField.to_dict()` returns the JSON shape served by every plugin
listing endpoint; the keys mirror the dataclass attributes 1-to-1 (with
the exception of `include_in_origin` and `origin_serializer`, which are
server-side only).

### Field-value normalization

`vtscore/plugins/normalize.py` holds the framework's
`normalize_field_values(plugin, field_values)` pass. It mutates
*field_values* in place, returns it, and is idempotent. For each
declared field of a text-like type (`text`, `url`, `email`, `password`,
`folder`, `server_path`, `select`) it:

1. Strips whitespace, and raises `ValueError("<Label> is required.")`
   for a `required` field that is then empty (or was absent).
2. Substitutes each `{name}` listed in the field's `template_vars`,
   running every resolved value through
   `vtscore.security.path_validation.sanitize_template_value`. An
   undeclared or unknown name raises `ValueError`. Supported names:
   `YYYYMMDD-HHMMSS`, `YYYYMMDD`, `YYYY`, `MM`, `DD`, `detector_name`,
   `detector_id`, `username`.
3. Runs the field-type validator: `url` → `validate_url`;
   `server_path` / `folder` → `confine_server_filepath` anchored at
   `get_file_access_base_dir()`, whose **approved path is written back**
   into `field_values` so the plugin body consumes exactly what was
   validated.

File fields and non-string values are skipped.

Step 3's write-back is a security property, not a convenience: under
multi-user confinement the validator resolves a relative path against
the user's data dir while a plugin body would resolve it against the
process CWD, so consuming the raw string can read another user's
directory even though the check passed.

Both ingress points run the pass, so HTTP and CLI behave identically:
`vtsearch/routes/_shared.py`'s `validate_plugin_args()` and
`validate_exporter_field_values()` on the web path,
`PluginBase.validate_cli_field_values()` on the CLI path, and
`SyncSource.load()` / `save()` / `peek_version()` for sync sources (see
[`sync.md`](sync.md)). Plugin bodies are expected to trust the result:
no `.strip()`, no `if not foo: raise`, no manual `validate_url` /
`validate_server_filepath` chains.

**Not covered:** `MediaConverter.params` and `MediaClipper.parameters`
arrive as pass-through payloads rather than plugin form bodies.
Converter params are shape-validated against the `fields` schema by
`MediaConverter.convert_normalized()` — ranges and `select` whitelists,
but no URL or path guard.

## `PluginBase`

`vtscore/plugins/__init__.py` - the mixin every plugin class
inherits from (directly or via a family-specific ABC like
`LabelsetExporter`). It supplies the CLI-flag, JSON-serialisation, and
field-validation glue that's identical across families.

### Class attributes the subclass sets

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | `str` | Yes | Snake-case identifier; the registry key and the URL path segment |
| `display_name` | `str` | Yes | Human-readable label |
| `description` | `str` | Yes | One-sentence subtitle |
| `icon` | `str` | No | Emoji / icon glyph; each family ships a sensible default |
| `fields` | `list[PluginField]` | Yes | Ordered list of user-facing inputs (may be empty) |
| `ui_mode` | `str` | No | `"form"` (default), `"file_upload"`, `"custom"`, or `"none"` |
| `hidden_from_picker` | `bool` | No | When `True`, omitted from generic family pickers; useful for scaffolds and special-cased plugins |

`ui_mode="form"` tells the frontend to render the generic form built
from `fields`. `"file_upload"` skips the form and uses the native file
picker (e.g. the local-folder importer). `"custom"` declares that the
frontend has a dedicated UI block. `"none"` means the plugin has no
user-facing UI at all - the GUI exporter uses this because the
frontend handles its result-display directly.

### Inherited methods

| Method | Description |
|--------|-------------|
| `resolve_display_name(field_values) -> str` | Returns the label for the thing these `field_values` describe: the user-typed `dataset_name` when non-empty, else `default_display_name(field_values)`. Importers override `default_display_name`, not this. Default returns `display_name`. |
| `add_cli_arguments(parser)` | Walks `fields` and adds an `argparse` argument per field. `"checkbox"` fields use `BooleanOptionalAction` (`--<key>` / `--no-<key>`). `"select"` fields get a `choices` constraint. `"number"` fields get `type=int` or `type=float` per `is_integer_number()`. |
| `validate_cli_field_values(field_values)` | Raises `ValueError("Missing required argument: --<flag>")` if any non-checkbox required field is empty. Checkboxes are skipped because argparse always populates them. |
| `to_dict()` | Returns the JSON-serialisable plugin metadata used by listing endpoints: `{name, display_name, description, icon, fields, ui_mode, hidden_from_picker}`. |
| `notify(message, *, level="info", detail=None)` | Shows the user a one-off message (a toast in the app, a printed line under the CLI) with `display_name` as the source, and keeps going. For the recoverable problem that shouldn't fail the run - see [`concurrency.md`](concurrency.md#user-notifications). Never raises. |

Subclasses normally only override `to_dict()` if they need to attach
family-specific metadata (e.g. converters expose `source_type` /
`target_type`).

## `PluginRegistry`

`vtscore/plugins/__init__.py` - generic auto-discovering registry.
One instance per plugin family.

### Constructor

```python
PluginRegistry(
    package: str,
    sentinel: str,
    label: str,
    *,
    discover_modules: bool = False,
    entry_point_group: str | None = None,
    eager: bool = True,
)
```

| Parameter | Description |
|-----------|-------------|
| `package` | Dotted package name to scan, e.g. `"vtscore.exporters"`. Normally just `__name__` when called from the package's `__init__.py`. |
| `sentinel` | Module-level attribute name to look for, e.g. `"EXPORTER"` or `"IMPORTER"`. The found attribute is registered keyed by `attr.name`. |
| `label` | Human-readable noun used in warning messages, e.g. `"labelset exporter"`. |
| `discover_modules` | When `True`, scan flat `.py` files in addition to sub-packages. Used by families where each plugin is a single module (media sources, converters). |
| `entry_point_group` | Optional `importlib.metadata` group name for third-party plugins. See [Entry-point integration](#entry-point-integration). |
| `eager` | `True` (default) runs discovery at construction time. `False` defers until the first `get()` / `list()` call - only needed by tests that want to inspect the pre-discovery state or simulate concurrent first access. |

### Public API

```python
registry.get(name: str) -> T | None     # returns None if not found
registry.list() -> list[T]              # discovery order; alphabetical for built-ins
```

Both methods trigger lazy discovery on first call if `eager=False`. A
re-entrancy guard (`_discovering`) prevents infinite recursion when a
plugin module's import side-effects call `get()` / `list()` on the same
registry; in that window the caller sees a partial registry, which is
filled in once the outer discovery completes.

### Discovery mechanics

Discovery walks `package_dir.iterdir()` in sorted order (so output is
deterministic) and skips dot/underscore prefixes. Sub-packages
(directories with `__init__.py`) are always scanned; flat `.py`
modules are scanned only when `discover_modules=True`. Symlinked
entries are loaded via `importlib.util.spec_from_file_location`
because Python's default `FileFinder` can miss symlinks on some
platforms. Directories whose name contains a `.` are skipped to avoid
nested-module-path misinterpretation. A module that fails to import
emits a `UserWarning` and is skipped; the partial `sys.modules` entry
is cleaned up so a retry can succeed.

`eager=True` is the default because every consumer wants a populated
registry immediately. The mode exists for tests that want to inspect
pre-discovery state.

## `make_plugin_registry()` factory

`vtscore/plugins/__init__.py` - collapses the per-family
boilerplate. Returns the `(get, list)` accessor pair every plugin
family re-exports.

```python
# vtscore/exporters/__init__.py
from vtscore.plugins import make_plugin_registry

get_exporter, list_exporters = make_plugin_registry(
    package=__name__,
    sentinel="EXPORTER",
    label="labelset exporter",
    entry_point_group="vtscore.exporters",
)
```

The factory keeps the `PluginRegistry` instance alive as a closure;
external code only ever sees the two accessors. If you need the
registry object itself (e.g. to call `_discover()` manually in tests),
instantiate `PluginRegistry` directly.

## Sentinel auto-discovery

Every plugin module declares its plugin **at module top level** as a
sentinel constant whose name matches what the registry was constructed
with:

```python
# vtscore/exporters/sftp/__init__.py
class SftpExporter(LabelsetExporter):
    name = "sftp"
    display_name = "SFTP Upload"
    description = "POST results to an SFTP server."
    fields = [...]
    def export(self, results, field_values): ...

EXPORTER = SftpExporter()
```

The convention is one plugin per module. The sentinel attribute is the
single source of truth - the registry doesn't inspect class names or
walk module contents, it just reads `getattr(module, sentinel, None)`.
This means a single module can host multiple classes (a helper, a base
class, the plugin) without confusing the registry, and you can swap
plugin implementations without renaming the constant.

The standard sentinel names per family are listed in the table below;
the column also gives the entry-point group name for the third-party
path.

| Family | Library package | Sentinel | Base class | Entry-point group |
|--------|-----------------|----------|------------|-------------------|
| Dataset importers | `vtscore.datasets.importers` | `IMPORTER` | `DatasetImporter` | `vtscore.importers` |
| Results exporters | `vtscore.exporters` | `EXPORTER` | `LabelsetExporter` | `vtscore.exporters` |
| Label importers | `vtscore.labels.importers` | `LABEL_IMPORTER` | `LabelImporter` | `vtscore.label_importers` |
| Labelset sources | `vtscore.labels.sources` | `LABELSET_SOURCE` | `LabelsetSource` | `vtscore.labelset_sources` |
| Media sources | `vtscore.datasets.sources` | `SOURCE` | `MediaSource` | `vtscore.media_sources` |
| Media types | `vtscore.media` | - (registered via `register_media_type`) | `MediaType` | `vtscore.media_types` |
| Media embedders | `vtscore.media` | - (registered via `register_embedder`) | `MediaEmbedder` | `vtscore.embedders` |
| Media clippers | `vtscore.media` | - (registered via `register_clipper`) | `MediaClipper` | `vtscore.clippers` |
| Media converters | `vtscore.converters` | `CONVERTER` | `MediaConverter` | `vtscore.converters` |

App-tier families keep their own entry-point group prefix, e.g.
`vtsearch.settings_importers`, `vtsearch.settings_exporters`,
`vtsearch.settings_sources` - those plugins live in `vtsearch/` and are
not part of the library tier.

## Entry-point integration

Third-party packages register plugins by declaring an
`importlib.metadata` entry point in the family's group. The value must
resolve to an already-instantiated plugin object (the same object the
in-tree sentinel attribute holds) - typically you point directly at
the sentinel.

```toml
# my_pkg/pyproject.toml
[project.entry-points."vtscore.importers"]
my_importer = "my_pkg.importer:IMPORTER"

[project.entry-points."vtscore.exporters"]
my_exporter = "my_pkg.exporter:EXPORTER"
```

After `pip install` of your package, the plugin appears in
`list_importers()` / `list_exporters()` / etc. without any code change
to `vtscore`. The discovery routine is `_discover_entry_points()` at
`vtscore/plugins/__init__.py`.

### Invariants

- **Built-ins win on name clash.** If an entry point's `plugin.name`
  matches a name already registered by the package scan, the entry
  point is skipped and a `UserWarning` is emitted. This prevents an
  installed third-party package from accidentally shadowing a core
  plugin.
- **Broken entry points warn and defer to first use.** When
  `entry_point.load()` raises — typically the plugin's *own* import
  failing on a missing optional dependency (e.g. `open_clip` absent for
  a `siglip_l` embedder) — the registry emits a `UserWarning` and
  registers an `EntryPointTombstone` under the entry-point name. The
  tombstone resolves via `get(name)` but re-raises the original error the
  moment it's actually invoked, and is kept out of `list()` so
  serialising the family never trips over it. One broken third-party
  plugin cannot block discovery of the rest of the registry, and the
  original error is surfaced (not swallowed) when someone tries to use
  the plugin.
- **No name attribute → rejected.** Entry-point objects that load
  successfully but expose no truthy `.name` attribute warn and skip -
  there's no fallback to the entry-point name on the registry side.

If you need to verify your entry point is registering correctly, the
inventory CLI is the fastest check:

```bash
python -W default -c "from vtscore.exporters import list_exporters; \
  print([e.name for e in list_exporters()])"
```

`-W default` is important - `UserWarning` is what surfaces a
malformed entry point.

## Inventory

`vtscore/plugins/inventory.py` collects every plugin family into one
data structure so tooling (the `python app.py --list-plugins` CLI,
shell-completion scripts, dashboards) can enumerate everything in one
call without knowing the family list in advance.

### Registering a family

A family is a `FamilyProvider(key, label, loader, entry_builder)`
record. `key` is the snake-case identifier used as the dict key and
CLI flag; `label` is the human-readable heading; `loader` is a
zero-arg callable returning the raw plugins; `entry_builder` maps one
raw plugin to a `PluginEntry` (default reads `name` / `display_name` /
`description` off the plugin).

```python
from vtscore.plugins.inventory import FamilyProvider, register_plugin_family
from my_pkg import list_my_plugins

register_plugin_family(FamilyProvider(
    key="my_plugins", label="My plugins", loader=list_my_plugins,
))
```

Library-tier families self-register at module import (see
`_LIBRARY_FAMILIES` in `vtscore/plugins/inventory.py`). App-tier
families (settings importers/exporters/sources) are registered by
`vtsearch/shim/register_app_plugin_families()` at app startup so
`vtscore` stays free of cross-boundary imports.

### `FAMILIES`, `gather_plugins()`, formatters

`FAMILIES` is exposed via a module-level `__getattr__` so importers
see a live tuple snapshot at access time - including app-only families
that the shim installs after `vtscore` imports. `gather_plugins()`
runs each family's loader inside `_safe_list()`, swallowing
`ImportError` / `ModuleNotFoundError` so missing optional deps in one
family can't block the rest. Three formatters (`format_plain`,
`format_names`, `format_json`) render the inventory for humans, shell
completion scripts, and tooling respectively.

`register_family_shortcuts(parser)` adds `--list-<family>` flags to
an `argparse.ArgumentParser`, one per registered family - each
equivalent to `--list-plugins --plugin-family <family>`.

## Schema helpers

`vtscore/plugins/schema.py` builds a marshmallow `Schema` class from a
plugin's declared `fields` at request time, caches it on the plugin
instance, and uses it to validate incoming POST bodies on the
plugin-driven HTTP routes (e.g. `/api/dataset/import/<importer>`).

The mapping is:

| `field_type` | Marshmallow field |
|--------------|-------------------|
| `text`, `url`, `password`, `folder`, `server_path`, `email` | `fields.String` (with non-empty-after-strip validator when required) |
| `number` (integer-looking) | `fields.Integer` |
| `number` (float-looking) | `fields.Float` |
| `select` | `fields.String` + `validate.OneOf(options)` (static options only) |
| `checkbox` | `fields.Function` deserialising `"true"`/`"false"`/bool/int → `bool` |
| `file` | skipped (populated from `request.files` after schema load) |

`make_plugin_arg_schema(plugin)` returns a `Schema` *class*;
`get_plugin_arg_schema(plugin)` returns a cached instance (cached on
the plugin instance, so the schema-build cost is paid once per
process). Unknown keys are dropped (`Meta.unknown = "exclude"`).
This module is used by the Flask routes in `vtsearch/routes/`;
library consumers don't typically interact with it directly.

## Field-value normalisation

Schema validation checks the *shape* of incoming values.
`vtscore/plugins/normalize.py` applies the behaviours that come after,
which used to be every plugin author's job to remember:

```python
def normalize_field_values(plugin: PluginBase, field_values: dict) -> dict: ...
```

1. **Whitespace strip** on every text-like value, so a plugin body can
   trust `field_values[key]` is already trimmed.
2. **Template substitution** for fields declaring
   `PluginField.template_vars` - `YYYYMMDD-HHMMSS`, `YYYYMMDD`, `YYYY`,
   `MM`, `DD`, `detector_name`, `detector_id`, `username`. Each
   resolved value runs through `sanitize_template_value`, so an
   attacker-controlled name cannot escape the directory an
   admin-configured template implies.
3. **Field-type-driven security validation.** `field_type="url"` goes
   through `validate_url`; `field_type="server_path"` goes through
   `confine_server_filepath` anchored at the per-user data dir, and the
   **approved** path is written back into `field_values` so the plugin
   body consumes exactly what was validated.

It is wired into both ingress points - the HTTP path
(`validate_plugin_args`, after marshmallow loads the body) and the CLI
path (`PluginBase.validate_cli_field_values`, after the presence check).
So a plugin body should trust the dict it receives: no
`if not foo: raise ValueError` boilerplate, no manual `validate_url` /
`validate_server_filepath` calls, no bespoke `str.replace("{detector_name}", ...)`
chains. External plugins that still call the validators by hand keep
working - re-validation is idempotent on already-validated values.

It mutates *field_values* in place and returns it, skips file uploads
and non-string values, and raises `ValueError` for a missing required
field, an invalid URL, a path-traversal attempt, or an unknown template
variable. The whole pass is idempotent.

## File uploads

`field_type="file"` inputs reach a plugin body as an `UploadedFile`
(`vtscore/plugins/uploads.py`), a **structural protocol** carrying
`.filename`, `.read()` / `.stream`, and `.save(dst)`. It exists so the
library tier never has to import Werkzeug's `FileStorage`.

Three implementations satisfy it, one per ingress:

| Path | Implementation |
|------|----------------|
| Flask request | Werkzeug's `FileStorage`, passed straight through - it already has the surface |
| CLI (`--file <path>`) | `CliUploadedFile`, backed by a local filesystem path |
| Background-thread upload | `BytesIOUploadedFile`, holding the bytes in memory so the thread can still read them after the request context is torn down |

Write plugin bodies against `.filename` (matching Werkzeug's name) and
they work on all three.

## End-to-end: writing a third-party plugin

A complete example of a third-party labelset exporter shipped as a
separate distribution. The same shape (sentinel constant + entry
point) applies to every family - substitute `IMPORTER` /
`LABEL_IMPORTER` / `CONVERTER` / `LABELSET_SOURCE` / `SOURCE` as
appropriate.

```python
# my_vtscore_plugins/exporter.py
from __future__ import annotations
from typing import Any
import json

from vtscore.exporters.base import LabelsetExporter, PluginField


class StdoutLabelsetExporter(LabelsetExporter):
    name = "stdout"
    display_name = "Stdout (JSON)"
    description = "Print the labelset as JSON to stdout."
    icon = "\U0001f4dd"
    fields = [
        PluginField(
            key="indent",
            label="Indent",
            field_type="number",
            default="2",
            min="0",
            max="8",
            step="1",
        ),
    ]

    def export(self, results: dict[str, Any], field_values: dict[str, Any]) -> dict[str, Any]:
        indent = int(field_values.get("indent", 2))
        print(json.dumps(results, indent=indent))
        return {"message": "Printed to stdout."}


EXPORTER = StdoutLabelsetExporter()
```

```toml
# my_vtscore_plugins/pyproject.toml
[project]
name = "my-vtscore-plugins"
version = "0.1.0"
dependencies = ["vtscore"]

[project.entry-points."vtscore.exporters"]
stdout = "my_vtscore_plugins.exporter:EXPORTER"
```

After `pip install -e .`:

```python
>>> from vtscore.exporters import get_exporter, list_exporters
>>> [e.name for e in list_exporters()]
['email_smtp', 'gui', 'holder', 'server_csv_file', 'server_json_file', 'stdout', 'webhook']
>>> exp = get_exporter("stdout")
>>> exp.export({"detectors_run": 1, "results": {}}, {"indent": "4"})
{
    "detectors_run": 1,
    "results": {}
}
{'message': 'Printed to stdout.'}
```

A CLI driver gets the auto-generated `--indent` flag for free via
`PluginBase.add_cli_arguments()`. The marshmallow schema for the HTTP
route is built the first time the plugin is invoked, then cached.

### Common pitfalls

- **Forgetting the sentinel.** The registry reads `getattr(module,
  sentinel, None)`. `EXPORTER = MyExporter()` works; `exporter =
  MyExporter()` does not.
- **Module-level import errors.** Failures emit a `UserWarning` and
  are skipped - run Python with `-W default` to surface them.
- **Name collision with a built-in.** Third-party entry points lose
  on name clash; pick a unique `name` (e.g. prefix with your package
  name).
- **Mutable `fields` default.** Declare `fields` as a class attribute
  list literal; don't share a list across plugin classes.
- **Plugin-instance state.** Plugin instances are long-lived (one per
  process). Don't stash request-scoped data on `self`; pass it through
  `field_values`.

## Cross-references

- [`vtscore.exporters`](exporters.md) - the labelset-exporter family
  built on this framework.
- [`vtscore.sync`](sync.md) - the bidirectional-sync ABC that
  `LabelsetSource` (in `vtscore.labels.sources`) and `SettingsSource`
  (app-tier) inherit from.
- The repo-level
  [`docs/EXTENDING-plugins.md`](../../../docs/EXTENDING-plugins.md)
  walks through every other family end-to-end.
