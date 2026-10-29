# Extending VTSearch

This index points to the three topic-specific extension guides, plus the
cross-cutting sections: authentication, dependencies, and a one-stop
checklist for every extension type. Start here; open the child doc that
matches what you want to build.

## Two doc sets, one contract

Most extension points are documented **twice**, on purpose, for two
different readers:

- **These `docs/EXTENDING-*.md` guides are the in-repo front door.** They
  cover the interface contract, where the file goes, how
  discovery works, app-tier wiring (routes, forms, the pickers the
  frontend renders), and a complete worked example. Read these if you are
  adding a plugin *to this repository*.
- **[`vtscore/docs/extending/`](../vtscore/docs/extending/README.md) is
  the library contract.** It ships inside the semver'd `vtscore` package
  and covers the same ABCs from the library side, plus the things only an
  out-of-tree author needs: `importlib.metadata` entry-point registration,
  packaging, and the Flask-free import rules. Read these if you are
  shipping a plugin as a **separate distribution**.

**Which side owns what.** A mechanism implemented in `vtscore` (field
normalization and template variables, entry-point packaging, the
`MediaType` thumbnail hooks, `FetchedItem`) is stated canonically in the
library guide; the app guide summarises it and links there. App-tier
wiring (routes, the SPA's forms and pickers, settings sync, auth) lives
only here. Each side keeps its own contract table per ABC.

Where the two overlap, they describe one class, so they can contradict
each other. `scripts/gates/check-extension-docs.py` (a `./run-tests.sh` gate)
holds them to the code: every member either set names in a contract table
must exist, and neither may present a public wrapper as the override point
when the class defines an `_impl` hook behind it. When you change a plugin
ABC, update both sides and let the gate confirm it.

## Extension guides

| Guide | What you build | Library contract |
|-------|----------------|------------------|
| [EXTENDING-plugins.md](EXTENDING-plugins.md) | Data importers, datasource importers, seed importers, results exporters, label importers, settings importers/exporters/sources, labelset sources: the form-driven auto-discovered plugin families that share a common registry-based architecture (the generated family inventory and the entry-point group table live there too). | [dataset-importers](../vtscore/docs/extending/dataset-importers.md), [results-exporters](../vtscore/docs/extending/results-exporters.md), [label-importers](../vtscore/docs/extending/label-importers.md), [labelset-sources](../vtscore/docs/extending/labelset-sources.md) |
| [EXTENDING-media.md](EXTENDING-media.md) | Media types, embedders, clippers, cleaners, converters, and media sources (anything in `vtscore/media/` or `vtscore/converters/`). | [media-types](../vtscore/docs/extending/media-types.md), [embedders](../vtscore/docs/extending/embedders.md), [clippers](../vtscore/docs/extending/clippers.md), [converters](../vtscore/docs/extending/converters.md), [media-sources](../vtscore/docs/extending/media-sources.md) |

Each guide explains the interface contract, where files go, how
discovery/registration works, and includes a complete example. The
library-tier index is
[`vtscore/docs/extending/README.md`](../vtscore/docs/extending/README.md).
Datasource importers, seed importers, and media cleaners have no page
there; their library-side contracts live in the package references
([datasource-importers](../vtscore/docs/packages/datasource-importers.md),
[seed-importers](../vtscore/docs/packages/seed-importers.md), and the
cleaner section of [media](../vtscore/docs/packages/media.md)).

## Cross-cutting reference

- [Authentication Providers](#authentication-providers): pluggable
  `LoginProvider` ABC
- [Dataset-Import Hooks](#dataset-import-hooks): a function of yours,
  named by the server admin, that runs when a user's import finishes
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
from pathlib import Path

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

1. `set_login_provider(provider)` is called once at startup. The built-in
   providers are selected by `python app.py --login trivial|api_key`
   (`_run_server` in `vtsearch/cli_main.py`); a custom provider must be
   activated the same way — add a branch there, or call
   `set_login_provider()` from your own startup code before the server
   starts serving.
2. The `before_request` middleware (`vtsearch/hooks.py`) calls
   `provider.get_user(request)` and stores the result in `g.user`; when the
   provider's `enforce_auth()` is true it also 401s unauthenticated `/api/*`
   requests (only `/api/auth/status`, `/api/auth/login` and
   `/api/auth/logout` are excepted).
3. Routes read the user through `get_current_user()`, never `g.user` directly.
4. `GET /api/auth/status` calls `provider.status_dict(request)`.

The full request flow, user resolution outside a request, and how per-user
settings and data directories hang off the username are in
[ARCHITECTURE.md § Authentication and multi-user support](ARCHITECTURE.md#authentication-and-multi-user-support).

### Built-in providers

`DefaultLoginProvider` (active unless `--login` is passed) returns
`"default"` for every request, is always authenticated, and uses the shared
`data/` directory. `vtsearch.auth` also ships `TrivialLoginProvider`
(`--login trivial`: username-only cookie login, for testing multi-user
features) and `ApiKeyLoginProvider` (`--login api_key`: `Authorization:
Bearer` keys hashed in `data/api_keys.json`); both are worked examples of
the interface.

---

## Dataset-Import Hooks

To act when a user's import finishes (email them, post to a chat
channel, record it somewhere), write a plain function and have the
server admin name it at startup. There is nothing to subclass or
register, and the code stays out of this repository.

```python
# mailer.py: anywhere importable on the server (installed, or on PYTHONPATH)
from vtscore.datasets.import_event import FAILED, SUCCEEDED, DatasetImported


def notify(event: DatasetImported) -> None:
    if not event.notify:  # the user turned off import alerts
        return
    address = address_for(event.user)  # VTSearch stores no email addresses
    if event.outcome == SUCCEEDED:
        send(address, f"{event.name} is ready: {event.n_media} {event.media_type} items")
    elif event.outcome == FAILED:
        send(address, f"Importing {event.name} failed: {event.error}")
```

```bash
python app.py --on-dataset-imported mailer:notify
VTSEARCH_ON_DATASET_IMPORTED=mailer:notify     # gunicorn / Docker, which never parse argv
```

The event is a frozen `DatasetImported`:

| Field | Meaning |
|-------|---------|
| `outcome` | `SUCCEEDED` (`"succeeded"`) or `FAILED` (`"failed"`). Compare against both exactly, so the hook keeps behaving if another outcome is ever added. |
| `dataset_id` | The saved dataset's id; empty on failure. |
| `name` | The saved display name on success (the registry may have made it unique); the requested name on failure. |
| `user` | The VTSearch username that started the import. |
| `media_type` | `"image"`, `"audio"`, …; empty if a failure came before the type was known. |
| `n_media` | Items in the saved dataset; `0` on failure. |
| `origin` | The importer and parameters the dataset was built from (a copy). |
| `error` | The message the user was shown; empty on success. |
| `notify` | `False` when the user turned off import alerts; `True` otherwise. |

Fields may be added; none will be removed or change meaning without a
`vtscore/CHANGELOG.md` entry.

**Respect `notify` in a hook that alerts the user.** Each user has an
**Alert me when an import finishes** setting (Settings › Import),
which the Settings modal offers only on a server with hooks configured.
Every hook is still called for every import whatever the user picked,
because VTSearch cannot tell a hook that alerts the user from one that
records the import for the admin. So a mailer checks `event.notify`, as
above, and an audit log ignores it. A hook that never reads it alerts
everyone, so the setting does nothing on that server.

**Which imports fire it.** The ones a user starts and waits for: the Add
Dataset dialog's importers, local folder and file uploads, demo datasets,
and an uploaded dataset file. That is the same set that offers to run
the user's AutoFind detectors. Reloading a saved dataset, combining
datasets, the CLI's `--autodetect`, and a cancelled import do not.

**When and where it runs.** On the import's worker thread, after the
progress bar already shows the import as finished and after AutoFind (if
any) has started, so a hook that waits on a slow mail server holds up
nothing the user sees. Several hooks run in the order the admin named
them. A hook that raises is logged and the next one still runs; the
dataset is saved either way.

**Configuration is the admin's, per server.** The module is imported at
startup: a bad `--on-dataset-imported` stops the server with an error,
and a bad `VTSEARCH_ON_DATASET_IMPORTED` is reported on stdout and
installs none of that variable's hooks. The startup log lists the hooks
in force. There is deliberately no settings-file key for this: a
settings file that could name code to run would let anyone who can
import settings run code on the server. The wiring is
`vtsearch/import_hooks.py`; the pipeline side is the `on_finished`
callback in `vtscore/datasets/load_origin.py`.

---

## Dependency Management

Runtime dependencies are declared in **`pyproject.toml`** under
`[project.dependencies]`: that's the single source of truth, and deptry
(a `./run-tests.sh` gate, and an optional pre-commit hook) verifies that every
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
requirements/labbench.txt            # Standalone curated core for Dockerfile.labbench, every media type
requirements/labbench-<type>.txt     # `-r labbench.txt` + one media type's deps (image, audio)
requirements/image-embedders.txt     # Standalone curated list shared by Dockerfile.image-embedders[.gpu]
```

The labbench / image-embedders requirements files are deliberately
standalone (they pin a minimal subset for size-constrained Docker
images) and do **not** flow through pyproject.

To let `docker/Dockerfile.labbench` build another media type, add a
`requirements/labbench-<type>.txt` (`-r labbench.txt` plus that type's
dependencies) and an entry for the type in `HIDDEN_BY_TYPE` in
`scripts/install/seed_labbench_settings.py`, naming any plugin the slim file leaves
without its dependencies. `tests_lib/meta/test_dockerfile_labbench.py` checks
that the two agree. The all-types build (no `MEDIA_TYPE`) picks the new type up
from its requirements file, with no edit to the Dockerfile.

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
- [ ] Implement one override point: `list_records()` + `fetch_record()`, `fetch_source_media()`, `fetch_all_source_media()`, or `run(self, field_values, medias, thin=False)` (populate `medias` in place); see [Choosing your override point](EXTENDING-plugins.md#choosing-your-override-point)
- [ ] Expose `IMPORTER = YourImporter()` at module level
- [ ] If your form holds opaque values (ids, query keys), override `default_display_name(field_values)` so the Dataset Name box shows a readable name — see [Naming the imported dataset](EXTENDING-plugins.md#naming-the-imported-dataset)
- [ ] If the plugin needs extra packages, add them to `[project.dependencies]` in `pyproject.toml` and re-run your editable install
- [ ] Test: start the app and check `GET /api/dataset/all-importers` includes your importer

### New Datasource / Seed Importer Checklist

See [EXTENDING-plugins.md § Adding a Datasource Importer](EXTENDING-plugins.md#adding-a-datasource-importer) and [§ Adding a Seed Importer](EXTENDING-plugins.md#adding-a-seed-importer).

- [ ] Datasource (one exemplar item): subclass `DataSourceImporter`, set `category` and `fields`, implement `fetch(field_values) -> FetchedMediaItem`, expose `DATASOURCE_IMPORTER`
- [ ] Seed (a batch of unlabeled seeds): subclass `SeedImporter`, set `fields` (and `max_items` if 100 is too few), implement `run(field_values) -> list[SeedMediaItem]`, expose `SEED_IMPORTER`
- [ ] Do outbound fetches through `guarded_session()` / `fetch_validated_url`, keep the source's real file extension, and set `origin` when the item has a durable identity
- [ ] Test: check `GET /api/datasource-importers` / `GET /api/seed-importers` includes it

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
- [ ] If your form holds opaque values (ids, query keys), override `default_detector_name(field_values)` so a new detector's name box shows a readable name — see [Naming the created detector](EXTENDING-plugins.md#naming-the-created-detector)
- [ ] If the plugin needs extra packages, add them to `[project.dependencies]` in `pyproject.toml` and re-run your editable install
- [ ] Test: start the app and check `GET /api/label-importers` includes your importer

### New Settings Source Checklist

See [EXTENDING-plugins.md § Adding a Settings Source](EXTENDING-plugins.md#adding-a-settings-source).
Unlike the one-shot families, a source stays bound to a user and is
consulted on every settings read and write, so read
[§ What the sync engine expects of a source](EXTENDING-plugins.md#what-the-sync-engine-expects-of-a-source)
before you start — it carries the contracts (cheap freshness probe,
dirty-key protection, no lock held) that the base class cannot express.

- [ ] Create `vtsearch/settings_io/sources/<name>/__init__.py`
- [ ] Subclass `SettingsSource`, set `name`, `display_name`, `description`, `fields`
- [ ] Implement `_do_load(self, field_values)`: return a settings dict (override the underscored hook, not `load`)
- [ ] Implement `_do_save(self, settings_data, field_values)`: persist settings (override the underscored hook, not `save`)
- [ ] Implement `_do_peek_version(self, field_values)`: a **cheap** freshness token (mtime, ETag, HEAD) — it runs on the hot path of every settings read
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

- [ ] Create `vtscore/datasets/sources/<name>.py` (every built-in source is a flat `.py` module), or register out-of-tree through the `vtscore.media_sources` entry-point group
- [ ] Create a `MediaSource` subclass with `list_items()`, `fetch_item()`, `resolve_path()`, each returning `FetchedItem` / `MediaItem`
- [ ] Create a factory class with `name` and `create_from_origin(origin)` method; `name` must equal the `origin["importer"]` value the source serves
- [ ] Expose `SOURCE = YourFactory()` at module level
- [ ] Test: create an origin dict for your source and verify `get_source_for_origin()` returns it

### New Media Type Checklist

See [EXTENDING-media.md § Adding a Media Type](EXTENDING-media.md#adding-a-media-type).

- [ ] Create `vtscore/media/<type>/` directory with `__init__.py`, `media_type.py`
- [ ] Subclass `MediaType` and implement all abstract properties and methods
- [ ] Expose `MEDIA_TYPE` and (if you ship them) `CLIPPERS` / `CLEANERS` sentinels in `__init__.py` (embedders are discovered per-module; see the embedder checklist below)
- [ ] If the plugin needs extra packages, add them to `[project.dependencies]` in `pyproject.toml` and re-run your editable install
- [ ] Override `pickle_extra_fields` if you use custom clip keys
- [ ] Test: import a folder of your media type, verify clips appear and are sortable

### New Media Embedder Checklist

See [EXTENDING-media.md § Adding a Media Embedder](EXTENDING-media.md#adding-a-media-embedder).

- [ ] Create `vtscore/media/<type>/embedder.py` (or `embedder_<variant>.py` for alternatives)
- [ ] Subclass `MediaEmbedder`, implement `name`, `media_type_id`, `_load_models_impl()`, `_embed_media_impl()` (the underscored hooks, not `load_models()` / `embed_media()`); keep the loaded model on `self._model`
- [ ] Optionally implement `_embed_text_impl()` for text-query sorting (not `embed_text()`, which L2-normalises); otherwise override `supports_text` to return `False`
- [ ] Declare `embedding_dim` (and `model_id` when there is a downloadable checkpoint)
- [ ] Leave `description_wrappers` empty unless you have *measured* that a prompt ensemble beats the typed query on your checkpoint (see [EXTENDING-media.md](EXTENDING-media.md#adding-a-media-embedder))
- [ ] Expose `EMBEDDER = YourEmbedder()` at module level (auto-discovery picks it up; no `__init__.py` edits needed; an `embedder_<name>/` sub-package and symlinked files/directories are supported too)
- [ ] Run `python scripts/gates/gen-docs-inventories.py` and commit the regenerated embedder tables
- [ ] Test: load a dataset and verify embeddings are generated

### New Media Clipper Checklist

See [EXTENDING-media.md § Adding a Media Clipper](EXTENDING-media.md#adding-a-media-clipper).

- [ ] Create or add to `vtscore/media/<type>/clipper.py`
- [ ] Subclass `MediaClipper`, implement `name`, `media_type`, `clip()`
- [ ] Override `description` with a short tooltip string for the chooser UI
- [ ] If adding `parameters`, include a `description` key in each param dict and implement `with_params()` to return a **new** instance
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
- [ ] Screen any client-supplied username with `is_safe_username()`
- [ ] Activate it with `set_login_provider(MyProvider())` at startup (a new `--login` branch in `_run_server`, `vtsearch/cli_main.py`, or your own startup code)
