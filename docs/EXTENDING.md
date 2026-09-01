# Extending VTSearch

This index points to the three topic-specific extension guides, plus the
cross-cutting sections: authentication, dependencies, and a one-stop
checklist for every extension type. Start here; open the child doc that
matches what you want to build.

## Two doc sets, one contract

Most extension points are documented **twice**, on purpose, for two
different readers:

- **These `docs/EXTENDING-*.md` guides are the in-repo front door.** They
  are self-contained: interface contract, where the file goes, how
  discovery works, app-tier wiring (routes, forms, the pickers the
  frontend renders), and a complete worked example. Read these if you are
  adding a plugin *to this repository*.
- **[`vtscore/docs/extending/`](../vtscore/docs/extending/README.md) is
  the library contract.** It ships inside the semver'd `vtscore` package
  and covers the same ABCs from the library side, plus the things only an
  out-of-tree author needs: `importlib.metadata` entry-point registration,
  packaging, and the Flask-free import rules. Read these if you are
  shipping a plugin as a **separate distribution**.

Where the two overlap, they describe one class, so they can contradict
each other. `scripts/check-extension-docs.py` (a `./run-tests.sh` gate)
holds them to the code: every member either set names in a contract table
must exist, and neither may present a public wrapper as the override point
when the class defines an `_impl` hook behind it. When you change a plugin
ABC, update both sides and let the gate confirm it.

## Extension guides

| Guide | What you build | Library contract |
|-------|----------------|------------------|
| [EXTENDING-plugins.md](EXTENDING-plugins.md) | Data importers, datasource importers, results exporters, label importers, settings importers/exporters/sources, labelset sources: the form-driven auto-discovered plugin families that share a common registry-based architecture (the generated family inventory lives there too). | [dataset-importers](../vtscore/docs/extending/dataset-importers.md), [results-exporters](../vtscore/docs/extending/results-exporters.md), [label-importers](../vtscore/docs/extending/label-importers.md), [labelset-sources](../vtscore/docs/extending/labelset-sources.md) |
| [EXTENDING-media.md](EXTENDING-media.md) | Media types, embedders, clippers, cleaners, converters, and media sources (anything in `vtscore/media/` or `vtscore/converters/`). | [media-types](../vtscore/docs/extending/media-types.md), [embedders](../vtscore/docs/extending/embedders.md), [clippers](../vtscore/docs/extending/clippers.md), [converters](../vtscore/docs/extending/converters.md) |
| [EXTENDING-processors.md](EXTENDING-processors.md) | Detectors, localizers, and extractors: the three kinds of `Processor`. | — (app tier only; processors are not auto-discovered) |

Each guide explains the interface contract, where files go, how
discovery/registration works, and includes a complete example. The
library-tier index — including the families with no app-tier guide of
their own (datasource importers, seed importers, media sources) — is
[`vtscore/docs/extending/README.md`](../vtscore/docs/extending/README.md).

## Cross-cutting reference

- [Authentication Providers](#authentication-providers): pluggable
  `LoginProvider` ABC
- [Dependency Management](#dependency-management): pyproject.toml as
  the single source of truth, with deptry guarding drift
- [Quick Reference: Checklist for Each Extension Type](#quick-reference-checklist-for-each-extension-type):
  one checklist per extension family

---

## Authentication Providers

VTSearch uses a pluggable `LoginProvider` ABC so that multi-user deployments
can be supported without modifying routes. The ABC itself is library-tier
(`vtscore/security/login.py`, because `vtscore`'s path-confinement checks
consult the active provider); `vtsearch.auth` re-exports it alongside the
Flask-backed providers, and is the import path app code should use.

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

**Enforcement is on by default.** The base class's `enforce_auth()` returns
`True`, which makes the server reject any `/api/*` request for which your
`is_authenticated()` returns false with a JSON 401 (only the
`/api/auth/status|login|logout` allowlist is exempt). You get real gating
without writing a decorator or touching routes — but it also means
`is_authenticated()` must return `True` for every request you intend to
serve. Override `enforce_auth()` to return `False` only if anonymous access
is a deliberate mode (see `TrivialLoginProvider`). Optionally override
`www_authenticate()` (e.g. return `"Bearer"`) to set the 401's
`WWW-Authenticate` header.

**Validate any username you didn't construct yourself.** A username returned
by `get_user()` becomes a path component (`data/<username>/`) *and* the
confinement root for server-file path validation, which resolves `..` away
instead of rejecting it. Screen client-supplied values with
`vtsearch.auth.is_safe_username()` and fall back to `"anonymous"`:

```python
from vtsearch.auth import is_safe_username

def get_user(self, request) -> str:
    username = request.headers.get("X-User", "anonymous")
    return username if is_safe_username(username) else "anonymous"
```

This applies regardless of how strong your authentication is — it is a
filesystem-hygiene check, not an authentication one.

### How it works

1. `set_login_provider(provider)` is called once at startup (in `app.py`).
2. The `before_request` middleware calls `provider.get_user(request)` and
   stores the result in `g.user`.
3. Routes call `get_current_user()` to read `g.user`.
4. `GET /api/auth/status` calls `provider.status_dict(request)`.

### Built-in provider

`DefaultLoginProvider` (the default) returns `"default"` for every request,
is always authenticated, and uses the shared `data/` directory.

### Current scope

Per-dataset runtime state (`medias`, diversity tree, display name) is
isolated in `DatasetContext` objects, and per-detector state (votes,
label history, click times, learned scores, inclusion, labelset source)
is isolated in `DetectorContext` objects. The frontend sends
`X-Dataset-Id` / `X-Detector-Id` headers, and a `before_request`
middleware resolves the active contexts per request, so multiple users
can work with different datasets/models simultaneously.

The auth infrastructure supports ownership tracking (`created_by` on
datasets and detectors) and per-user data
directories via `get_user_data_dir(username, base)`. **Settings remain
globally shared** across all users; there is no per-user settings
isolation yet.

---

## Dependency Management

Runtime dependencies are declared in **`pyproject.toml`** under
`[project.dependencies]`: that's the single source of truth, and deptry
(wired into `lint.yml` and the pre-commit hook) verifies that every
imported package is declared there. Dev tools (pytest, ruff,
pre-commit, etc.) live under `[project.optional-dependencies].dev`, and
the two AGPL-3.0 packages (`ultralytics`, `PyMuPDF`) under
`[project.optional-dependencies].agpl` — an extra every default install
path requests, so it is opt-*out*, not opt-in (see
[DEPLOYMENT.md](DEPLOYMENT.md#installing-without-the-agpl-dependencies)).

```
pyproject.toml                       # [project.dependencies] + [project.optional-dependencies] (dev, agpl)
requirements/base.txt                # `--extra-index-url <cpu wheel index>` + `-e .[dev,agpl]`
requirements/gpu.txt                 # `-e .[dev,agpl]` (install.sh / Dockerfile.gpu set --extra-index-url)
requirements/*-no-agpl.txt           # The same two files without the `agpl` extra (VTSEARCH_NO_AGPL=1)
requirements/labbench.txt            # Standalone curated list for Dockerfile.labbench (image+SigLIP only)
requirements/image-embedders*.txt    # Standalone curated lists for Dockerfile.image-embedders[.gpu]
```

The labbench / image-embedders requirements files are deliberately
standalone (they pin a minimal subset for size-constrained Docker
images) and do **not** flow through pyproject.

### For a new media type, importer, or exporter

Add the extra packages to `[project.dependencies]` in `pyproject.toml`
(or to `[project.optional-dependencies].dev` if they're test/lint-only),
then re-run `bash scripts/install.sh` (or any editable install).
Failed imports of a plugin's sub-package emit a warning rather than
crashing, so missing dependencies degrade gracefully; but deptry will
flag any imported package missing from pyproject the next time `./run-tests.sh` runs.

---

## Quick Reference: Checklist for Each Extension Type

Each checklist below links to the detailed guide for that extension type.

### New Data Importer Checklist

See [EXTENDING-plugins.md § Adding a Data Importer](EXTENDING-plugins.md#adding-a-data-importer).

- [ ] Create `vtscore/datasets/importers/<name>/__init__.py`
- [ ] Subclass `DatasetImporter`, set `name`, `display_name`, `description`, `fields`
- [ ] Implement `run(self, field_values, medias, thin=False)`: populate `medias` in-place
- [ ] Expose `IMPORTER = YourImporter()` at module level
- [ ] If your form holds opaque values (ids, query keys), override `default_display_name(field_values)` so the Dataset Name box shows a readable name — see [Naming the imported dataset](EXTENDING-plugins.md#naming-the-imported-dataset)
- [ ] If the plugin needs extra packages, add them to `[project.dependencies]` in `pyproject.toml` and re-run your editable install
- [ ] Test: start the app and check `GET /api/dataset/all-importers` includes your importer

### New Results Exporter Checklist

See [EXTENDING-plugins.md § Adding a Results Exporter](EXTENDING-plugins.md#adding-a-results-exporter).

- [ ] Create `vtscore/exporters/<name>/__init__.py`
- [ ] Subclass `ResultsExporter`, set `name`, `display_name`, `description`, `fields`
- [ ] Implement a payload method per kind you support — `export_find_results(self, results, field_values)` for a scored run, `export_labelset(self, labelset, field_values)` for a detector's labels: return a dict with a `"message"` key
- [ ] To send the user to a web page instead of (or as well as) delivering the labelset, return an `"open_url"` and set `opens_url = True` if you always return one — see [Opening a browser tab](EXTENDING-plugins.md#opening-a-browser-tab-open_url)
- [ ] Expose `EXPORTER = YourExporter()` at module level
- [ ] If the plugin needs extra packages, add them to `[project.dependencies]` in `pyproject.toml` and re-run your editable install
- [ ] Test: start the app and check `GET /api/exporters` includes your exporter

### New Label Importer Checklist

See [EXTENDING-plugins.md § Adding a Label Importer](EXTENDING-plugins.md#adding-a-label-importer).

- [ ] Create `vtscore/labels/importers/<name>/__init__.py`
- [ ] Subclass `LabelImporter`, set `name`, `display_name`, `description`, `fields`
- [ ] Implement `run(self, field_values)`: return a list of `{"md5": ..., "label": ...}` dicts
- [ ] Expose `LABEL_IMPORTER = YourImporter()` at module level
- [ ] If the plugin needs extra packages, add them to `[project.dependencies]` in `pyproject.toml` and re-run your editable install
- [ ] Test: start the app and check `GET /api/label-importers` includes your importer

### New Settings Source Checklist

See [EXTENDING-plugins.md § Adding a Settings Source](EXTENDING-plugins.md#adding-a-settings-source).

- [ ] Create `vtsearch/settings_io/sources/<name>/__init__.py`
- [ ] Subclass `SettingsSource`, set `name`, `display_name`, `description`, `fields`
- [ ] Implement `_do_load(self, field_values)`: return a settings dict (override the underscored hook, not `load`)
- [ ] Implement `_do_save(self, settings_data, field_values)`: persist settings (override the underscored hook, not `save`)
- [ ] Expose `SETTINGS_SOURCE = YourSource()` at module level
- [ ] If the plugin needs extra packages, add them to `[project.dependencies]` in `pyproject.toml` and re-run your editable install
- [ ] Test: start the app and check `GET /api/settings-sources` includes your source

### New Labelset Source Checklist

See [EXTENDING-plugins.md § Adding a Labelset Source](EXTENDING-plugins.md#adding-a-labelset-source).

- [ ] Create `vtscore/labels/sources/<name>/__init__.py`
- [ ] Subclass `LabelsetSource`, set `name`, `display_name`, `description`, `fields`
- [ ] Implement `_do_load(self, field_values)`: return a list of label dicts (override the underscored hook, not `load`)
- [ ] Implement `_do_save(self, labelset, field_values)`: persist a `LabelSet` (override the underscored hook, not `save`)
- [ ] Expose `LABELSET_SOURCE = YourSource()` at module level
- [ ] If the plugin needs extra packages, add them to `[project.dependencies]` in `pyproject.toml` and re-run your editable install
- [ ] Test: start the app and check `GET /api/labelset-sources` includes your source

### New Settings Importer Checklist

See [EXTENDING-plugins.md § Adding a Settings Importer](EXTENDING-plugins.md#adding-a-settings-importer).

- [ ] Create `vtsearch/settings_io/importers/<name>/__init__.py`
- [ ] Subclass `SettingsImporter`, set `name`, `display_name`, `description`, `fields`
- [ ] Implement `run(self, field_values)`: return a settings dict
- [ ] Expose `SETTINGS_IMPORTER = YourImporter()` at module level
- [ ] Test: start the app and check `GET /api/settings-importers` includes your importer

### New Settings Exporter Checklist

See [EXTENDING-plugins.md § Adding a Settings Exporter](EXTENDING-plugins.md#adding-a-settings-exporter).

- [ ] Create `vtsearch/settings_io/exporters/<name>/__init__.py`
- [ ] Subclass `SettingsExporter`, set `name`, `display_name`, `description`, `fields`
- [ ] Implement `export(self, settings_data, field_values)`: return dict with `"message"` key
- [ ] Expose `SETTINGS_EXPORTER = YourExporter()` at module level
- [ ] Test: start the app and check `GET /api/settings-exporters` includes your exporter

### New Media Source Checklist

See [EXTENDING-media.md § Adding a Media Source](EXTENDING-media.md#adding-a-media-source).

- [ ] Create `vtscore/datasets/sources/<name>.py` (media sources are flat `.py` modules, not sub-packages)
- [ ] Create a `MediaSource` subclass with `list_items()`, `fetch_item()`, `resolve_path()`
- [ ] Create a factory class with `name` and `create_from_origin(origin)` method
- [ ] Expose `SOURCE = YourFactory()` at module level
- [ ] Test: create an origin dict for your source and verify `get_source_for_origin()` returns it

### New Media Type Checklist

See [EXTENDING-media.md § Adding a Media Type](EXTENDING-media.md#adding-a-media-type).

- [ ] Create `vtscore/media/<type>/` directory with `__init__.py`, `media_type.py`
- [ ] Subclass `MediaType` and implement all abstract properties and methods
- [ ] Expose `MEDIA_TYPE`, `CLIPPERS`, and (if you ship them) `CLEANERS` sentinels in `__init__.py` (embedders are discovered per-module; see the embedder checklist below)
- [ ] If the plugin needs extra packages, add them to `[project.dependencies]` in `pyproject.toml` and re-run your editable install
- [ ] Override `pickle_extra_fields` if you use custom clip keys
- [ ] Test: import a folder of your media type, verify clips appear and are sortable

### New Media Embedder Checklist

See [EXTENDING-media.md § Adding a Media Embedder](EXTENDING-media.md#adding-a-media-embedder).

- [ ] Create `vtscore/media/<type>/embedder.py` (or `embedder_<variant>.py` for alternatives)
- [ ] Subclass `MediaEmbedder`, implement `name`, `media_type_id`, `_load_models_impl()`, `embed_media()`
- [ ] Optionally implement `embed_text()` for text-query sorting
- [ ] Leave `description_wrappers` empty unless you have *measured* that a prompt ensemble beats the typed query on your checkpoint (see [EXTENDING-media.md](EXTENDING-media.md#adding-a-media-embedder))
- [ ] Expose `EMBEDDER = YourEmbedder()` at module level (auto-discovery picks it up; no `__init__.py` edits needed; symlinked files are supported)
- [ ] Test: load a dataset and verify embeddings are generated

### New Media Clipper Checklist

See [EXTENDING-media.md § Adding a Media Clipper](EXTENDING-media.md#adding-a-media-clipper).

- [ ] Create or add to `vtscore/media/<type>/clipper.py`
- [ ] Subclass `MediaClipper`, implement `name`, `media_type`, `clip()`
- [ ] Override `description` with a short tooltip string for the chooser UI
- [ ] If adding `parameters`, include a `description` key in each param dict
- [ ] Add to the `CLIPPERS` list in the media type's `__init__.py`
- [ ] Test: verify `clip()` returns valid media dicts

### New Media Cleaner Checklist

See [EXTENDING-media.md § Adding a Media Cleaner](EXTENDING-media.md#adding-a-media-cleaner).

- [ ] Create or add to `vtscore/media/<type>/cleaner.py`
- [ ] Subclass `MediaCleaner`, implement `name`, `media_type`, `clean()`
- [ ] Return the media **unchanged** when there is nothing to clean or the payload can't be decoded (never abort a load)
- [ ] Build the output with `dict(media)`; never mutate the input in place (the runner needs a pre-clean payload to snapshot)
- [ ] Update the metadata the rewrite invalidated (`file_size`, `width`/`height`, `duration`, `character_count`)
- [ ] Override `description` with a short tooltip string for the cleanup checkbox
- [ ] Override `default_enabled` to `True` only if leaving the gate off ships known-wrong vectors
- [ ] Add to the `CLEANERS` list in the media type's `__init__.py`
- [ ] Test: verify `clean()` no-ops on a clean item and on undecodable bytes, and that it rewrites the payload otherwise

### New Media Converter Checklist

See [EXTENDING-media.md § Adding a Media Converter](EXTENDING-media.md#adding-a-media-converter).

- [ ] Create `vtscore/converters/<source>2<target>.py`
- [ ] Subclass `MediaConverter`, implement `source_type`, `target_type`, and `convert()`
- [ ] Expose `CONVERTER = YourConverter()` at module level
- [ ] Test: convert a source-type media and verify output dicts are valid

### New Localizer / Extractor Checklist

See [EXTENDING-processors.md](EXTENDING-processors.md).

- [ ] Subclass `Localizer` or `Extractor` from `vtscore.media.processors`
- [ ] Implement `name` (from a constructor argument, **not** hardcoded),
      `media_type`, and the type-specific method (`localize` or `extract`)
- [ ] Optionally override `load_model()` for one-time resource loading
- [ ] Add a `from_config(name, config)` classmethod, and a `to_dict()` that
      reports the matching `extractor_type` / `localizer_type` + `config`
- [ ] **Wire it into the hardcoded factory dict** in
      `vtsearch/routes/processors/crud.py` (`_ensure_extractor_factories` or
      `_ensure_localizer_factories`) — processors are *not* auto-discovered,
      and an unregistered type is a 400 from every endpoint
- [ ] Optionally add it to `_PREGEN_PROCESSORS` in the same file
- [ ] Register as autorun via `POST /api/autorun-extractors` or
      `POST /api/autorun-localizers`, then run it with `POST /api/auto-extract`
      / `POST /api/auto-localize`
- [ ] Test: build it via `_build_extractor` / `_build_localizer` and assert a
      bad config raises (see `tests/detectors/test_processors.py`)

Note that `Detector` (the `Processor` subtype) has no factory dict and no
endpoint; it is not registrable. The ML classifiers users actually train are
detectors in a different sense, covered below.

For ML classifiers, create a detector instead: register it via
`POST /api/detectors/registry`, label items in the right pane, and toggle
its Auto-Find flag with `PUT /api/detectors/registry/<id>/autofind`.

### New Login Provider Checklist

See [Authentication Providers](#authentication-providers) above.

- [ ] Create a new module (e.g. `vtsearch/auth/my_provider.py`)
- [ ] Subclass `LoginProvider` from `vtsearch.auth`
- [ ] Implement `get_user(request)` and `is_authenticated(request)`
- [ ] Decide on `enforce_auth()`: the inherited `True` makes the server 401
      unauthenticated API requests; override to `False` only if anonymous
      access is intended
- [ ] Override `login_required()` if the frontend should show a login screen
- [ ] Override `get_user_data_dir(username, base_data_dir)` for per-user isolation
- [ ] Call `set_login_provider(MyProvider())` at app startup (in `app.py`)
