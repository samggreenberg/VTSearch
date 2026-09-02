# `vtscore.sync`

A single-class package: the generic `SyncSource[LoadT, SaveT]` ABC that
backs every bidirectional sync plugin in VTSearch. A *sync source* is a
plugin that knows how to both pull data **in** from an external target
(`load`, like an importer) and push data back **out** to it (`save`,
like an exporter), so applications can keep external storage and
in-memory state automatically aligned.

The package is intentionally minimal - it owns only the contract.
Concrete subclasses live wherever the data domain does:

- [`vtscore.labels.sources.LabelsetSource`](../../labels/sources/base.py) -
  round-trips detector labels (library tier, documented in
  [`labels`](labels.md)).
- `vtsearch.settings_io.sources.SettingsSource` - round-trips app
  settings (app tier; not part of `vtscore`). It lives outside the
  library because per-user settings persistence is an app concern, but
  it inherits the same `SyncSource` shape.

## Why it lives in `vtscore`

The ABC defines the contract: "a plugin with `load()` and `save()`
entry points backed by `_do_load()` / `_do_save()` hooks, configurable
via `PluginField`s, discoverable via the standard plugin registry." Both library-tier (labelsets) and app-tier (settings)
sources implement exactly that contract, so the abstract base lives at
the lowest tier that can host it. The concrete subclasses then live
wherever their data domain does - labels in `vtscore.labels`, settings
in `vtsearch.settings_io`.

## The class

`vtscore/sync/__init__.py` defines:

```python
from typing import Any, Generic, TypeVar
from vtscore.plugins import PluginBase, PluginField

LoadT = TypeVar("LoadT")
SaveT = TypeVar("SaveT")


class SyncSource(PluginBase, Generic[LoadT, SaveT]):
    """Bidirectional sync target.

    Subclasses set the standard PluginBase class attributes (``name``,
    ``display_name``, ``fields``, ...) and override the underscored
    template methods ``_do_load`` / ``_do_save``.
    """

    icon: str = "\U0001f504"        # default emoji: counter-clockwise arrows
    fields: list[PluginField]

    # Framework-owned wrappers: normalize field_values, then dispatch.
    def load(self, field_values: dict[str, Any]) -> LoadT:
        return self._do_load(self._normalize(field_values))

    def save(self, data: SaveT, /, field_values: dict[str, Any]) -> None:
        self._do_save(data, self._normalize(field_values))

    def peek_version(self, field_values: dict[str, Any]) -> Any | None: ...

    # Subclass hooks.
    def _do_load(self, field_values: dict[str, Any]) -> LoadT: ...
    def _do_save(self, data: SaveT, /, field_values: dict[str, Any]) -> None: ...
    def _do_peek_version(self, field_values: dict[str, Any]) -> Any | None: ...
```

### Why the public / hook split

`load` / `save` / `peek_version` are **framework-owned**. Each runs
`vtscore.plugins.normalize.normalize_field_values` over a *copy* of
`field_values` (the caller's dict is never mutated) before dispatching,
so every subclass body receives values that are whitespace-stripped,
required-checked, `template_vars`-substituted, and - for `url` and
`server_path` / `folder` field types - security-validated, with path
fields replaced by their confined, canonicalised form.

A subclass that overrides `load` or `save` directly **replaces that
wrapper**: its body then runs on raw input and the path confinement
never happens. Override `_do_load` / `_do_save` instead. `peek_version`
additionally swallows normalization errors and returns `None`, so an
unresolvable template can't crash a caller's freshness probe.

`_do_peek_version` is optional - the default returns `None`, meaning
"I can't cheaply check", and the caller falls back to an explicit sync.
Implement it when the backing store offers a cheap freshness token
(`st_mtime_ns` for a local file, `ETag` for an HTTP source).

The two type parameters are separate so the **load** and **save**
shapes can differ. This matters in practice: a `LabelsetSource` loads
raw label dicts (`list[dict[str, str]]`) but saves a full `LabelSet`
object, and a `SettingsSource` may load a flat dict from an external
source but save a more curated subset. Letting `LoadT != SaveT`
captures that without forcing either side into a least-common-shape.

The `save` / `_do_save` signature has the value as the first
**positional-only** parameter so subclasses can rename it according to
what they save (`labelset`, `settings`, …) without breaking the
override contract.

```python
def _do_save(self, labelset: LabelSet, /, field_values: dict[str, Any]) -> None: ...
```

Both `_do_load` and `_do_save` raise `NotImplementedError` by default -
a subclass must override them.

## Discovery and registration

`SyncSource` is a `PluginBase`, so concrete subclasses are discovered
via the standard sentinel-based registry. Each domain defines its own
sentinel and entry-point group:

| Subclass | Package | Sentinel | Entry-point group |
|----------|---------|----------|-------------------|
| `LabelsetSource` | `vtscore.labels.sources` | `LABELSET_SOURCE` | `vtscore.labelset_sources` |
| `SettingsSource` (app tier) | `vtsearch.settings_io.sources` | `SETTINGS_SOURCE` | `vtsearch.settings_sources` |

See [`plugins.md`](plugins.md) for how the registries are built and how
to register a third-party source via `importlib.metadata` entry points.

## Implementing a sync source

A minimal labelset source. The same shape applies to settings sources
- only the base class and sentinel change.

```python
# my_pkg/sources/sqlite_labelset.py
import sqlite3
from typing import Any

from vtscore.datasets.labelset import LabelSet
from vtscore.labels.sources.base import LabelsetSource
from vtscore.plugins import PluginField


class SqliteLabelsetSource(LabelsetSource):
    name = "sqlite"
    display_name = "SQLite Labelset"
    description = "Store labels in a SQLite database."
    fields = [
        PluginField(key="db_path", label="DB path", field_type="server_path"),
        PluginField(key="table", label="Table", field_type="text", default="labels"),
    ]

    def _do_load(self, field_values: dict[str, Any]) -> list[dict[str, str]]:
        # db_path is a server_path field, so it arrives already confined
        # to the current user's data directory.
        conn = sqlite3.connect(field_values["db_path"])
        rows = conn.execute(f"SELECT md5, label FROM {field_values['table']}").fetchall()
        conn.close()
        return [{"md5": md5, "label": lbl} for md5, lbl in rows]

    def _do_save(self, labelset: LabelSet, /, field_values: dict[str, Any]) -> None:
        conn = sqlite3.connect(field_values["db_path"])
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS {field_values['table']} "
            f"(md5 TEXT PRIMARY KEY, label TEXT)"
        )
        conn.executemany(
            f"INSERT OR REPLACE INTO {field_values['table']} VALUES (?, ?)",
            [(e.md5, e.label) for e in labelset.elements],
        )
        conn.commit()
        conn.close()


LABELSET_SOURCE = SqliteLabelsetSource()
```

Drop the module under `vtscore/labels/sources/sqlite/` or expose it as
an entry point - see [`plugins.md`](plugins.md) - and the standard
`get_labelset_source("sqlite")` / `list_labelset_sources()` accessors
pick it up.

## What this package does *not* do

- **It does not auto-sync.** The ABC defines the `load` / `save` pair.
  The caller (a route handler, an app-side hook, a CLI driver) decides
  when to call them. App-side wiring for the two existing source
  families - auto-export on vote change, auto-import on first load,
  circular-trigger prevention via a module-level `_syncing` flag under a
  re-entrant lock - lives in `vtscore/labels/sync.py` and
  `vtsearch/settings.py`, not here.
- **It does not persist anything.** `SyncSource` is a contract;
  storage is whatever the subclass implements. The standard caveats
  about persistence still apply - embeddings and trained model weights
  are in-memory artefacts, not labels. See `CLAUDE.md` "No Persisted
  Vectors or MLPs" if you're tempted to round-trip more than label
  identifiers + their `good`/`bad` flag.
- **It does not impose a transport.** `_do_load` / `_do_save` can talk to a
  filesystem, an HTTP API, a database, a message queue - anything the
  subclass cares to wire up. The framework only requires that the two
  methods exist.

## Cross-references

- [`plugins.md`](plugins.md) - registries, sentinels, entry points,
  inventory.
- [`labels.md`](labels.md) - the `LabelsetSource` subclass and the
  auto-sync helpers in `vtscore/labels/sync.py`.
- `vtsearch.settings_io.sources` (app tier) - the `SettingsSource`
  subclass with per-user template variables (`{username}`) and lazy,
  per-user auto-import. Its engine (freshness probing, the dirty-key
  contract, lock ordering) is documented in
  [EXTENDING-plugins.md § How the sync engine works](../../../docs/EXTENDING-plugins.md#how-the-sync-engine-works).
