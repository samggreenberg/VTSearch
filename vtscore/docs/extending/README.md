# Writing plugins for `vtscore`

`vtscore` ships ten plugin families that share one machinery: a
`PluginRegistry` per family scans its package directory at import time,
registers every sub-package or flat module that exposes a sentinel
attribute (`IMPORTER`, `EXPORTER`, `EMBEDDER`, …), and then walks the
matching `importlib.metadata` entry-point group so third-party
distributions can drop plugins in without forking the repo. Each
plugin subclasses one base ABC (`DatasetImporter`, `LabelsetExporter`,
`MediaEmbedder`, …) that inherits from `PluginBase`, declares its
user-facing inputs as a list of `PluginField`s, and implements one or
two abstract methods.

If you're contributing a plugin to the `vtscore` source tree, drop it
in the family's package; if you're shipping a separate distribution,
declare an `importlib.metadata` entry point in the family's group. Both
discovery paths converge on the same registry, and built-ins win on
name clashes so a stray third-party package can't silently shadow a
core plugin. See [`vtscore/plugins/__init__.py:287`](../../plugins/__init__.py)
for the registry, [`vtscore/plugins/__init__.py:386`](../../plugins/__init__.py)
for the entry-point loader, and the [plugins package
doc](../packages/plugins.md) for the lower-level API surface.

## The families

Library-tier families live inside `vtscore` and need no app code to
run. App-tier families (`vtsearch.*`) wrap user-preferences I/O and are
listed here only so third-party developers know which entry-point
group to pick.

### Library tier (in `vtscore`)

| Family | Entry-point group | Sentinel | Base class | Purpose |
|--------|-------------------|----------|------------|---------|
| Dataset importers | `vtscore.importers` | `IMPORTER` | `DatasetImporter` | Pull media into a dataset from a source (folder, archive, API, etc.) |
| Results exporters | `vtscore.exporters` | `EXPORTER` | `LabelsetExporter` | Send autodetect results or labels somewhere (file, webhook, email, …) |
| Label importers | `vtscore.label_importers` | `LABEL_IMPORTER` | `LabelImporter` | One-shot pull of `(md5, label)` pairs from an external source |
| Labelset sources | `vtscore.labelset_sources` | `LABELSET_SOURCE` | `LabelsetSource` | Bidirectional sync of a detector's labelset with an external store |
| Media converters | `vtscore.converters` | `CONVERTER` | `MediaConverter` | Cross-format access: image → text (OCR), audio → image (spectrogram), … |
| Media sources | `vtscore.media_sources` | `SOURCE` | `MediaSource` (via factory) | Low-level file-resolution for an origin (local folder, HTTP archive, …) |
| Media types | — (in-tree only) | `MEDIA_TYPE` | `MediaType` | A whole new content kind (file extensions, HTTP serving, demos) |
| Media embedders | — (in-tree only) | `EMBEDDER` | `MediaEmbedder` | A new encoder for an existing or new media type |
| Media clippers | — (in-tree only) | `CLIPPERS` (list) | `MediaClipper` | Split one media into many (tiling, sentence-split, scene-split) |

Media types, embedders, and clippers do not currently expose an
entry-point group — they discover via the `vtscore.media`
sub-package scan. To ship one out-of-tree, symlink the
`embedder_<name>.py` (or media-type sub-package) into the appropriate
directory under `vtscore/media/`; both symlinked files and symlinked
directories are loaded via `importlib.util.spec_from_file_location`
([`vtscore/media/__init__.py:260`](../../media/__init__.py)).

### App tier (in `vtsearch`)

These are documented in the repo-level [`EXTENDING-plugins.md`](../../../docs/EXTENDING-plugins.md);
their library counterparts (`vtscore.labels.sources`) handle bidirectional
sync at the model layer.

| Family | Entry-point group | Sentinel | Base class |
|--------|-------------------|----------|------------|
| Settings importers | `vtsearch.settings_importers` | `SETTINGS_IMPORTER` | `SettingsImporter` |
| Settings exporters | `vtsearch.settings_exporters` | `SETTINGS_EXPORTER` | `SettingsExporter` |
| Settings sources | `vtsearch.settings_sources` | `SETTINGS_SOURCE` | `SettingsSource` |

## Per-family guides

- [Dataset importers](dataset-importers.md) — pull media into the system
- [Media types](media-types.md) — add a whole new content kind
- [Embedders](embedders.md) — add a new encoder for a media type
- [Clippers](clippers.md) — split media into sub-clips
- [Converters](converters.md) — transform media between types
- [Results exporters](results-exporters.md) — send results/labels out
- [Label importers](label-importers.md) — one-shot label pull
- [Labelset sources](labelset-sources.md) — bidirectional label sync

## Shared rules for every plugin

These cut across every family and apply to both in-tree and
third-party plugins.

**Declare inputs as `PluginField`s.** Each plugin family re-exports
`PluginField` under a friendlier alias (`ImporterField`,
`ExporterField`, `LabelImporterField`, …) which is a literal `=
PluginField`. The frontend renders a form from `fields`, the CLI
derives `argparse` flags from `fields`, and the marshmallow schema
builder validates POST bodies against `fields`. See [`vtscore/plugins/__init__.py:71`](../../plugins/__init__.py)
for the full dataclass and the [plugins package doc](../packages/plugins.md#pluginfield)
for the field-type matrix.

**Never persist vectors or trained model weights.** Embeddings are
re-derived from origins on demand. MLP weights are retrained from the
linked labelset on each detector load. Plugins must not write either
to disk, to `data/settings.json`, to detector JSON files, or to any
other store. If your plugin appears to need a cache for vectors, put
it on a process-scoped data structure (e.g. a field on
`DetectorContext`), not a file. The single exception is dataset
pickles, which are by design a snapshot of media + their embeddings.

**No hardcoded `data/` paths.** A plugin that needs a default path
should interpolate `vtscore.config.DATA_DIR` (the resolved data
directory honouring `VTSEARCH_DATA_DIR`) into its default field value
rather than hard-coding `"data/foo"`. Models go under
`vtscore.config.MODELS_CACHE_DIR` (honouring `VTSEARCH_MODELS_DIR`).

**No Flask imports, no `vtsearch.settings` imports.** Library-tier
plugins must be importable in a Flask-free environment — the
`tests_lib/` test tier is verified by `./run-tests.sh vtscore-clean`,
which installs a meta-path import hook that refuses `flask`,
`werkzeug`, and `flask_smorest`. Read configuration through
`CoreConfig` ([`vtscore/config.py:209`](../../config.py)); construct
one directly when running outside an app context, or call
`CoreConfig.from_settings()` when an app shim has been registered.

**Validate filesystem paths at the boundary.** Any plugin that accepts
a user-supplied server path must run it through
`vtscore.security.validate_server_filepath` ([`vtscore/security/path_validation.py:35`](../../security/path_validation.py))
so a malicious path can't escape the configured `SERVER_ROOTS`. When
splicing a template variable like `{detector_name}` into a path, also
call `sanitize_template_value` ([`vtscore/security/path_validation.py:82`](../../security/path_validation.py)).

**Validate URLs.** Any plugin that makes an outbound HTTP request must
call `vtscore.security.validate_url` ([`vtscore/security/url_validation.py:30`](../../security/url_validation.py))
before issuing the request. It rejects non-HTTP(S) schemes and resolves
the hostname to refuse private / link-local / loopback IPs (SSRF
guard).

## Entry-point registration in `pyproject.toml`

A third-party distribution registers a plugin by adding one line per
plugin to the family's entry-point group. The value resolves to an
already-instantiated plugin object — the same shape the in-tree
sentinel attribute holds. For example, a third-party dataset importer:

```toml
[project.entry-points."vtscore.importers"]
my_importer = "my_pkg.my_module:IMPORTER"
```

A third-party labelset exporter and labelset source from the same
package:

```toml
[project.entry-points."vtscore.exporters"]
my_exporter = "my_pkg.exporters.thing:EXPORTER"

[project.entry-points."vtscore.labelset_sources"]
my_source = "my_pkg.sources.thing:LABELSET_SOURCE"
```

After `pip install` of your distribution, the plugin appears in
`list_importers()` (or the corresponding `list_*` function),
`gather_plugins()`, and `python app.py --list-plugins` without any
core-repo changes. A failed entry-point load warns and is skipped;
built-in plugins take precedence on name clashes.

## Testing your plugin

The library-tier test suite (`tests_lib/`) exercises plugins through
the library API only — no Flask, no `vtsearch.settings`, no
`vtsearch.routes`. New plugin tests belong in:

- `tests_lib/io/` for importers, exporters, label importers, sources
- `tests_lib/datasets/` for synthesis / load behaviour
- `tests_lib/detectors/` for embedders, clippers, converters
- `tests_lib/core/` for media types, registry behaviour

Drop a file in the matching folder and pytest picks it up via the
folder-as-marker convention. See [`tests_lib/datasets/test_synthetic_importer.py`](../../../tests_lib/datasets/test_synthetic_importer.py)
for a complete importer-registration + run-behaviour test that does
not touch any app code, and [`tests_lib/conftest.py`](../../../tests_lib/conftest.py)
for the autouse fixtures every test inherits (`reset_contexts`,
`_stub_embedding_models`, the test-tmp-path allowance for
`validate_server_filepath`).

## Inventory and listing

`vtscore.plugins.inventory.gather_plugins()` returns every registered
plugin grouped by family — useful for tooling, shell completion, and
the `python app.py --list-plugins` CLI. Library families
self-register at module import; app-only families are injected via
`vtsearch.shim.register_app_plugin_families()` so the inventory module
stays free of cross-tier imports. See [`vtscore/plugins/inventory.py`](../../plugins/inventory.py)
and the [plugins package doc](../packages/plugins.md#inventory) for
the public API.
