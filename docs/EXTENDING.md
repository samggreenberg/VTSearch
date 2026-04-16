# Extending VTSearch

This index points to the three topic-specific extension guides, plus the
cross-cutting sections — authentication, dependencies, and a one-stop
checklist for every extension type. Start here; open the child doc that
matches what you want to build.

## Extension guides

| Guide | What you build |
|-------|----------------|
| [EXTENDING-plugins.md](EXTENDING-plugins.md) | Data importers, results exporters, label importers, processor importers, settings importers/exporters, settings sources, labelset sources — eight auto-discovered plugin families that share a common registry-based architecture. |
| [EXTENDING-media.md](EXTENDING-media.md) | Media types, embedders, clippers, converters, and media sources — anything in `vtsearch/media/` or `vtsearch/converters/`. |
| [EXTENDING-processors.md](EXTENDING-processors.md) | Detectors, localizers, and extractors — the three kinds of `Processor`. |

Each guide explains the interface contract, where files go, how
discovery/registration works, and includes a complete example.

## Cross-cutting reference

- [Authentication Providers](#authentication-providers) — pluggable
  `LoginProvider` ABC
- [Dependency Management](#dependency-management) — per-plugin
  `requirements.txt` files and `install-plugin-deps.sh`
- [Quick Reference: Checklist for Each Extension Type](#quick-reference-checklist-for-each-extension-type) —
  one checklist per extension family

---

## Authentication Providers

VTSearch uses a pluggable `LoginProvider` ABC (`vtsearch/auth/__init__.py`)
so that multi-user deployments can be supported without modifying routes.

### Interface

```python
from vtsearch.auth import LoginProvider

class MyProvider(LoginProvider):
    name = "my_provider"

    def get_user(self, request) -> str:
        """Return username from the request (e.g. header, cookie, cert)."""
        return request.headers.get("X-User", "anonymous")

    def is_authenticated(self, request) -> bool:
        """Return True if the request is authenticated."""
        return "X-User" in request.headers

    def login_required(self) -> bool:
        """Return True to show a login screen in the frontend."""
        return True

    def get_user_data_dir(self, username: str, base_data_dir: Path) -> Path:
        """Return a per-user data directory for isolated storage."""
        return base_data_dir / username
```

### How it works

1. `set_login_provider(provider)` is called once at startup (in `app.py`).
2. The `before_request` middleware calls `provider.get_user(request)` and
   stores the result in `g.user`.
3. Routes call `get_current_user()` to read `g.user`.
4. `GET /api/auth/status` calls `provider.status_dict(request)`.

### Built-in provider

`DefaultLoginProvider` (the default) returns `"default"` for every request,
is always authenticated, and uses the shared `data/` directory.

### Current limitations

In-memory state (votes, medias, labels, settings) is globally shared.
The auth infrastructure supports ownership tracking (`created_by`) and
per-user data directories, but full runtime state isolation is not yet
implemented. Custom providers should be aware that votes and loaded
datasets are shared across all users.

---

## Dependency Management

Runtime dependencies are managed via **per-plugin `requirements.txt`
files**, auto-discovered by `install-plugin-deps.sh`. Each plugin
sub-package (media type, importer, exporter, etc.) can include its own
`requirements.txt` in its directory.

```
vtsearch/media/image/requirements.txt     # Pillow, ultralytics, …
vtsearch/media/audio/requirements.txt     # librosa, soundfile, …
vtsearch/exporters/webhook/requirements.txt
```

Running `bash install-plugin-deps.sh` regenerates
`requirements-plugins.txt` with `-r` references to each plugin's file.
The top-level `requirements.txt` includes this, so `pip install -r
requirements.txt` installs everything.

`pyproject.toml` is kept minimal — only `cpu`/`gpu` (for choosing the
right PyTorch build) and `dev` (for test/lint tools) live there.

### For a new media type, importer, or exporter

Add a `requirements.txt` inside your plugin's directory, then run
`bash install-plugin-deps.sh` to regenerate the dependency tree. Failed
imports of a plugin's sub-package emit a warning rather than crashing, so
missing dependencies degrade gracefully.

---

## Quick Reference: Checklist for Each Extension Type

Each checklist below links to the detailed guide for that extension type.

### New Data Importer Checklist

See [EXTENDING-plugins.md § Adding a Data Importer](EXTENDING-plugins.md#adding-a-data-importer).

- [ ] Create `vtsearch/datasets/importers/<name>/__init__.py`
- [ ] Subclass `DatasetImporter`, set `name`, `display_name`, `description`, `fields`
- [ ] Implement `run(self, field_values, medias, thin=False)` — populate `medias` in-place
- [ ] Expose `IMPORTER = YourImporter()` at module level
- [ ] Add a `requirements.txt` in the plugin directory and run `bash install-plugin-deps.sh`
- [ ] Test: start the app and check `GET /api/dataset/all-importers` includes your importer

### New Results Exporter Checklist

See [EXTENDING-plugins.md § Adding a Results Exporter](EXTENDING-plugins.md#adding-a-results-exporter).

- [ ] Create `vtsearch/exporters/<name>/__init__.py`
- [ ] Subclass `LabelsetExporter`, set `name`, `display_name`, `description`, `fields`
- [ ] Implement `export(self, results, field_values)` — return a dict with a `"message"` key
- [ ] Expose `EXPORTER = YourExporter()` at module level
- [ ] Add a `requirements.txt` in the plugin directory and run `bash install-plugin-deps.sh`
- [ ] Test: start the app and check `GET /api/exporters` includes your exporter

### New Label Importer Checklist

See [EXTENDING-plugins.md § Adding a Label Importer](EXTENDING-plugins.md#adding-a-label-importer).

- [ ] Create `vtsearch/labels/importers/<name>/__init__.py`
- [ ] Subclass `LabelImporter`, set `name`, `display_name`, `description`, `fields`
- [ ] Implement `run(self, field_values)` — return a list of `{"md5": ..., "label": ...}` dicts
- [ ] Expose `LABEL_IMPORTER = YourImporter()` at module level
- [ ] Add a `requirements.txt` in the plugin directory and run `bash install-plugin-deps.sh`
- [ ] Test: start the app and check `GET /api/label-importers` includes your importer

### New Processor Importer Checklist

See [EXTENDING-plugins.md § Adding a Processor Importer](EXTENDING-plugins.md#adding-a-processor-importer).

- [ ] Create `vtsearch/processors/importers/<name>/__init__.py`
- [ ] Subclass `ProcessorImporter`, set `name`, `display_name`, `description`, `fields`
- [ ] Implement `run(self, field_values)` — return a dict with `media_type`, `weights`, `threshold`
- [ ] Expose `PROCESSOR_IMPORTER = YourImporter()` at module level
- [ ] Add a `requirements.txt` in the plugin directory and run `bash install-plugin-deps.sh`
- [ ] Test: start the app and check `GET /api/processor-importers` includes your importer

### New Settings Source Checklist

See [EXTENDING-plugins.md § Adding a Settings Source](EXTENDING-plugins.md#adding-a-settings-source).

- [ ] Create `vtsearch/settings_io/sources/<name>/__init__.py`
- [ ] Subclass `SettingsSource`, set `name`, `display_name`, `description`, `fields`
- [ ] Implement `load(self, field_values)` — return a settings dict
- [ ] Implement `save(self, settings_data, field_values)` — persist settings
- [ ] Expose `SETTINGS_SOURCE = YourSource()` at module level
- [ ] Add a `requirements.txt` in the plugin directory and run `bash install-plugin-deps.sh`
- [ ] Test: start the app and check `GET /api/settings-sources` includes your source

### New Labelset Source Checklist

See [EXTENDING-plugins.md § Adding a Labelset Source](EXTENDING-plugins.md#adding-a-labelset-source).

- [ ] Create `vtsearch/labels/sources/<name>/__init__.py`
- [ ] Subclass `LabelsetSource`, set `name`, `display_name`, `description`, `fields`
- [ ] Implement `load(self, field_values)` — return a list of label dicts
- [ ] Implement `save(self, labelset, field_values)` — persist a `LabelSet`
- [ ] Expose `LABELSET_SOURCE = YourSource()` at module level
- [ ] Add a `requirements.txt` in the plugin directory and run `bash install-plugin-deps.sh`
- [ ] Test: start the app and check `GET /api/labelset-sources` includes your source

### New Settings Importer Checklist

See [EXTENDING-plugins.md § Adding a Settings Importer](EXTENDING-plugins.md#adding-a-settings-importer).

- [ ] Create `vtsearch/settings_io/importers/<name>/__init__.py`
- [ ] Subclass `SettingsImporter`, set `name`, `display_name`, `description`, `fields`
- [ ] Implement `run(self, field_values)` — return a settings dict
- [ ] Expose `SETTINGS_IMPORTER = YourImporter()` at module level
- [ ] Test: start the app and check `GET /api/settings-importers` includes your importer

### New Settings Exporter Checklist

See [EXTENDING-plugins.md § Adding a Settings Exporter](EXTENDING-plugins.md#adding-a-settings-exporter).

- [ ] Create `vtsearch/settings_io/exporters/<name>/__init__.py`
- [ ] Subclass `SettingsExporter`, set `name`, `display_name`, `description`, `fields`
- [ ] Implement `export(self, settings_data, field_values)` — return dict with `"message"` key
- [ ] Expose `SETTINGS_EXPORTER = YourExporter()` at module level
- [ ] Test: start the app and check `GET /api/settings-exporters` includes your exporter

### New Media Source Checklist

See [EXTENDING-media.md § Adding a Media Source](EXTENDING-media.md#adding-a-media-source).

- [ ] Create `vtsearch/datasets/sources/<name>/__init__.py`
- [ ] Create a `MediaSource` subclass with `list_items()`, `fetch_item()`, `resolve_path()`
- [ ] Create a factory class with `name` and `create_from_origin(origin)` method
- [ ] Expose `SOURCE = YourFactory()` at module level
- [ ] Test: create an origin dict for your source and verify `get_source_for_origin()` returns it

### New Media Type Checklist

See [EXTENDING-media.md § Adding a Media Type](EXTENDING-media.md#adding-a-media-type).

- [ ] Create `vtsearch/media/<type>/` directory with `__init__.py`, `media_type.py`
- [ ] Subclass `MediaType` and implement all abstract properties and methods
- [ ] Expose `MEDIA_TYPE`, `EMBEDDERS`, and `CLIPPERS` sentinels in `__init__.py`
- [ ] Add a `requirements.txt` in the plugin directory and run `bash install-plugin-deps.sh`
- [ ] Override `pickle_extra_fields` if you use custom clip keys
- [ ] Test: import a folder of your media type, verify clips appear and are sortable

### New Media Embedder Checklist

See [EXTENDING-media.md § Adding a Media Embedder](EXTENDING-media.md#adding-a-media-embedder).

- [ ] Create `vtsearch/media/<type>/embedder.py` (or `embedder_<variant>.py` for alternatives)
- [ ] Subclass `MediaEmbedder`, implement `name`, `media_type_id`, `_load_models_impl()`, `embed_media()`
- [ ] Optionally implement `embed_text()` for text-query sorting
- [ ] Optionally set `description_wrappers` for enriched text embedding
- [ ] Add to the `EMBEDDERS` list in the media type's `__init__.py`
- [ ] Test: load a dataset and verify embeddings are generated

### New Media Clipper Checklist

See [EXTENDING-media.md § Adding a Media Clipper](EXTENDING-media.md#adding-a-media-clipper).

- [ ] Create or add to `vtsearch/media/<type>/clipper.py`
- [ ] Subclass `MediaClipper`, implement `name`, `media_type`, `clip()`
- [ ] Override `description` with a short tooltip string for the chooser UI
- [ ] If adding `parameters`, include a `description` key in each param dict
- [ ] Add to the `CLIPPERS` list in the media type's `__init__.py`
- [ ] Test: verify `clip()` returns valid media dicts

### New Media Converter Checklist

See [EXTENDING-media.md § Adding a Media Converter](EXTENDING-media.md#adding-a-media-converter).

- [ ] Create `vtsearch/converters/<source>2<target>.py`
- [ ] Subclass `MediaConverter`, implement `source_type`, `target_type`, and `convert()`
- [ ] Expose `CONVERTER = YourConverter()` at module level
- [ ] Test: convert a source-type media and verify output dicts are valid

### New Detector / Localizer / Extractor Checklist

See [EXTENDING-processors.md](EXTENDING-processors.md).

- [ ] Subclass `Detector`, `Localizer`, or `Extractor` from `vtsearch.media.base`
- [ ] Implement `name`, `media_type`, and the type-specific method (`detect`, `localize`, or `extract`)
- [ ] Optionally override `load_model()` for one-time resource loading
- [ ] Register as autorun via `POST /api/autorun-detectors` (or extractors/localizers)

### New Login Provider Checklist

See [Authentication Providers](#authentication-providers) above.

- [ ] Create a new module (e.g. `vtsearch/auth/my_provider.py`)
- [ ] Subclass `LoginProvider` from `vtsearch.auth`
- [ ] Implement `get_user(request)` and `is_authenticated(request)`
- [ ] Override `login_required()` if the frontend should show a login screen
- [ ] Override `get_user_data_dir(username, base_data_dir)` for per-user isolation
- [ ] Call `set_login_provider(MyProvider())` at app startup (in `app.py`)
