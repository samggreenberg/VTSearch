# `vtscore.labels`

Label I/O - the two plugin families and the sync glue that move
detector labels between VTSearch and external systems. A *label
importer* is one-way (read labels from somewhere external, apply them
to the active detector). A *labelset source* is bidirectional: it
both loads labels from an external store and pushes them back whenever
votes change. Both produce / consume the same
[`LabelSet`](datasets.md#labelset) object defined in
`vtscore.datasets` - there is no separate label datatype.

## Contents

| Module | Concern |
|--------|---------|
| `vtscore/labels/importers/base.py` | The `LabelImporter` ABC |
| `vtscore/labels/importers/__init__.py` | Label-importer registry with auto-discovery |
| `vtscore/labels/importers/server_json_file/` | Read labels from a `.json` file on the server |
| `vtscore/labels/importers/server_csv_file/` | Read labels from a `.csv` file on the server |
| `vtscore/labels/sources/base.py` | The `LabelsetSource` ABC (a `SyncSource`) |
| `vtscore/labels/sources/__init__.py` | Labelset-source registry with auto-discovery |
| `vtscore/labels/sources/server_json_file/` | Bidirectional sync with a JSON file on the server |
| `vtscore/labels/json_format.py` | Shared shape checks for the label-JSON format (`require_label_object`, `extract_labels`) |
| `vtscore/labels/sync.py` | The sync glue: debounced push, synchronous pull, re-export guard |

`vtscore/labels/__init__.py` re-exports nothing; import from the
submodules (see [architecture.md](../architecture.md#import-paths-read-before-copy-pasting)).

## Label importers

`LabelImporter` (`vtscore/labels/importers/base.py`) is the ABC.
Subclasses declare `fields: list[PluginField]`, implement `run`, and
expose a module-level `LABEL_IMPORTER` sentinel. The registry
auto-discovers them.

```python
from vtscore.labels.importers import get_label_importer, list_label_importers

imp = get_label_importer("server_json_file")
labels = imp.run({"filepath": "/data/labels.json"})
# [{"md5": "abc...", "label": "good"}, {"md5": "def...", "label": "bad"}, ...]
```

`run(field_values)` returns a list of dicts; only `"md5"` and `"label"`
(values `"good"` or `"bad"`) are required, and any other label value is
skipped by the caller. The caller maps each dict's `md5` against
`medias[*]["md5"]` to apply the vote - the importer itself doesn't touch
dataset state. `get_label_importer(name)` returns `None` for an unknown
name.

`list_label_importers()` returns every registered importer, including
those with `hidden_from_picker = True`; filtering on that flag is the
caller's job (the app's picker does it).

Third-party importers register via the `vtscore.label_importers`
entry-point group; discovery and name-clash rules are in
[plugins.md](plugins.md).

### Built-in label importers

| Name               | Display name      | Notes                                                                          |
|--------------------|-------------------|--------------------------------------------------------------------------------|
| `server_json_file` | Server JSON File  | Reads a `LabelSet`-format JSON file on the server filesystem.                  |
| `server_csv_file`  | Server CSV File   | Reads a CSV with `md5,label` columns (optional `origin_name`, `filename`, `category`, JSON `origin`) from the server filesystem. |

### Custom-importer skeleton

```python
# my_pkg/postgres_label_importer.py
from vtscore.labels.importers.base import LabelImporter, PluginField

class PostgresLabelImporter(LabelImporter):
    name = "postgres"
    display_name = "PostgreSQL Query"
    description = "Import labels from a PostgreSQL query."
    fields = [
        PluginField("host",     "Host",     "text"),
        PluginField("database", "Database", "text"),
        PluginField("query",    "Query",    "text",
                           description="Must return md5 and label columns."),
    ]

    def run(self, field_values):
        import psycopg2
        conn = psycopg2.connect(host=field_values["host"],
                                database=field_values["database"])
        cur = conn.cursor()
        cur.execute(field_values["query"])
        return [{"md5": row[0], "label": row[1]} for row in cur.fetchall()]

LABEL_IMPORTER = PostgresLabelImporter()
```

`run_cli(field_values)` wraps any `field_type="file"` path argument in
`vtscore.plugins.uploads.CliUploadedFile` and delegates to `run`, so a
`run` written against the `UploadedFile` surface works from both the
CLI and a web upload. `add_cli_arguments(parser)` is auto-derived from
`fields`, so most importers work from the CLI with no extra code.

---

## Labelset sources

`LabelsetSource` (`vtscore/labels/sources/base.py`) extends
[`SyncSource[LoadT, SaveT]`](sync.md) for a detector's labelset. A
source can both *load* labels and *save* them back, so a detector
linked to a source auto-imports its labels on load and auto-exports
on every vote change. Standalone importers and exporters keep working
regardless of whether a source is active.

The generic parameters are `SyncSource[list[dict[str, str]],
LabelSet]`. The public methods are framework wrappers that normalize
`field_values` (strip, resolve templates, validate paths/URLs) and then
call the underscored hook - **subclasses override the hooks**:

| Public method | Override | Returns |
|---------------|----------|---------|
| `load(field_values)` | `_do_load` (required) | `list[{"md5": ..., "label": ...}]` - the raw label list, same shape as `LabelImporter.run`. |
| `load_full(field_values)` | `_do_load_full` (optional) | The full `LabelSet`, including any `detector_meta` block. The default wraps `_do_load` into a metadata-less `LabelSet`; override to surface `media_type` / `input_spec` / `threshold`. |
| `save(labelset, field_values)` | `_do_save` (required) | `None` - persist the labelset. |
| `peek_version(field_values)` | `_do_peek_version` (optional) | A cheap freshness token, or `None` ("can't tell"). See [sync.md](sync.md). |

Subclasses expose a module-level `LABELSET_SOURCE` sentinel for
auto-discovery; the `vtscore.labelset_sources` entry-point group
covers third-party plugins.

```python
from vtscore.labels.sources import get_labelset_source, list_labelset_sources

src = get_labelset_source("server_json_file")
labelset = src.load_full({"filepath": "/data/labels/my_detector.labels.json"})
src.save(labelset, {"filepath": "/tmp/copy.labels.json"})
```

### Built-in labelset sources

| Name               | Display name     | Notes                                                                |
|--------------------|------------------|----------------------------------------------------------------------|
| `server_json_file` | Server JSON File | Round-trips a `LabelSet` JSON file on the server filesystem.         |

### Template variables

Source `filepath` fields support two runtime templates:

| Template          | Resolved to                                          |
|-------------------|------------------------------------------------------|
| `{detector_id}`   | The active `DetectorContext.detector_id`.            |
| `{detector_name}` | The active `DetectorContext.name`.                   |

A field opts in by declaring `template_vars=("detector_id",
"detector_name")` on its `PluginField`. Substitution happens in the
`load` / `save` normalize pass and runs each value through
`vtscore.security.path_validation.sanitize_template_value`, so an
attacker-controlled detector name like `../../etc/passwd` cannot escape
the directory implied by an admin-configured template; the resolved
path is then re-validated.
`vtscore/labels/sources/server_json_file/__init__.py::resolve_filepath_for(field_values, *, detector_id, detector_name)`
does the same substitution + validation for a detector that is not the
active one (e.g. resolving old and new paths on rename).

### Custom-source skeleton

```python
# my_pkg/redis_labelset_source.py
from vtscore.datasets.labelset import LabelSet
from vtscore.labels.sources.base import LabelsetSource, PluginField

class RedisLabelsetSource(LabelsetSource):
    name = "redis"
    display_name = "Redis Key"
    description = "Sync detector labels with a Redis key."
    fields = [
        PluginField("host", "Host", "text", default="localhost"),
        PluginField("key",  "Key",  "text",
                    description="Supports {detector_id} and {detector_name}.",
                    template_vars=("detector_id", "detector_name")),
    ]

    def _do_load(self, field_values):
        import json, redis
        r = redis.Redis(host=field_values["host"])
        raw = r.get(field_values["key"])
        if not raw:
            return []
        return json.loads(raw).get("labels", [])

    def _do_save(self, labelset: LabelSet, field_values):
        import json, redis
        r = redis.Redis(host=field_values["host"])
        r.set(field_values["key"], json.dumps(labelset.to_dict()))

LABELSET_SOURCE = RedisLabelsetSource()
```

---

## Sync glue

`vtscore/labels/sync.py` wires labelset sources to the live detector
context (see [state.md](state.md)). Two entry points:

| Function                           | When called                                                  |
|------------------------------------|--------------------------------------------------------------|
| `sync_to_labelset_source()`        | Whenever votes change. Schedules a **debounced background push**. |
| `sync_from_labelset_source(detector_id=None)` | On detector load or on manual import. Pulls + applies labels synchronously; returns the imported label list, or `None` when no source is configured, the source is unknown, the load raised, or the source is empty. |

Both silently no-op when the detector has no `labelset_source`; a
`load`/`save` exception is logged, never raised.

### Debounced push

`sync_to_labelset_source` returns immediately. The actual push runs on
a background `threading.Timer` ~200ms (`_DEBOUNCE_DELAY`) after the
most recent call; further calls within the window cancel and restart
the timer, so a rapid voting burst collapses into a single sync run
that uses the **latest** captured contexts (latest wins). The pending
slot is keyed by `detector_id`, so two detectors voted on concurrently
never coalesce into each other's window. A push is skipped (not
retried) when the vote snapshot can't be proven consistent with the
active dataset; the next vote re-arms the timer.

`flush_pending_label_syncs()` drains the queue synchronously - used by
tests that need to assert the file was written, and by graceful
shutdown paths. An `atexit` hook also calls it so the most recent
vote's push survives normal interpreter exit (Ctrl-C, gunicorn
SIGQUIT, `sys.exit`). Hard kills (SIGKILL, `os._exit`) bypass `atexit`
and still drop the last ~200ms of work - accept this as the cost of
debouncing.

For test isolation, `reset_label_sync_for_tests()` *cancels* pending
syncs without firing them, so a sync scheduled under one test's
contexts can't fire after those contexts are gone.

### The `_syncing` guard

A module-level boolean (`_syncing`, guarded by the `_sync_lock` RLock)
is set for the whole apply pass of `sync_from_labelset_source`. Any
push that reaches the source during that window is skipped, so a
half-applied import is never written back to the source. The flag is
**module-level, not thread-local**: the timer thread re-checks it
inside `_sync_lock` immediately before calling `save`, so a push that
fires on another thread mid-import is suppressed too.

### Detector meta round-trip

`load_full` returns the full `LabelSet`, including any `detector_meta`
block. When `sync_from_labelset_source` sees one, it folds the
source's `input_spec` (and `media_type`, when the receiving detector
is missing one) into the receiving detector's on-disk JSON. The
source's `threshold` is intentionally **not** applied - the receiver
retrains its head from the imported labels and recomputes its own
threshold. Detector files only ever store origins and meta, never
embeddings or model weights (see
[architecture.md](../architecture.md#the-no-persisted-vectors-rule)).

### Configuration

A detector opts into source-based sync by setting
`DetectorContext.labelset_source` to:

```python
{
    "source_name": "server_json_file",
    "field_values": {"filepath": "/data/labels/{detector_name}.labels.json"},
}
```

`source_name` keys into `get_labelset_source`; `field_values` is
passed through to that source's `load_full` / `save`. A detector
without a linked source costs nothing on every vote.

---

## Relationship to `vtscore.datasets`

Labels are not a separate domain - they're `LabeledElement` /
`LabelSet` from [`vtscore.datasets`](datasets.md#labelset). Both
plugin families operate on (or produce) that type:

- A `LabelImporter` returns `list[{"md5", "label"}]` for compatibility with the legacy label format. Wrapping that into a `LabelSet` is `LabelSet.from_dict({"labels": result})`.
- A `LabelsetSource.save` consumes a `LabelSet`; its `load` returns the same raw list shape for symmetry with importers, while `load_full` returns the full `LabelSet` for sources that carry richer metadata.
- The on-disk JSON format both built-ins use is the dict produced by `LabelSet.to_dict()` - a superset of the legacy `{"labels": [{"md5", "label"}]}` shape, with optional `origin`, `origin_name`, `filename`, `category`, `metadata`, `region_box` per element and an optional top-level `detector_meta` block.
