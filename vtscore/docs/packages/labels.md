# `vtscore.labels`

Label I/O - the two plugin families and the sync glue that move
detector labels between VTSearch and external systems. A *label
importer* is one-way (read labels from somewhere external, apply them
to the active detector). A *labelset source* is bidirectional: it
both loads labels from an external store and pushes them back whenever
votes change. Both produce / consume the same
[`LabelSet`](datasets.md#labelset) object defined in
`vtscore.datasets` - there is no separate label datatype.

## Label importers

`LabelImporter` (`vtscore/labels/importers/base.py:68`) is the ABC.
Subclasses declare `fields: list[PluginField]`, implement `run`, and
expose a module-level `LABEL_IMPORTER` sentinel. The registry
auto-discovers them.

```python
from vtscore.labels.importers import (
    LabelImporter, LabelImporterField,
    get_label_importer, list_label_importers,
)

imp = get_label_importer("server_json_file")
labels = imp.run({"filepath": "/data/labels.json"})
# [{"md5": "abc...", "label": "good"}, {"md5": "def...", "label": "bad"}, ...]
```

`run(field_values)` returns a list of dicts; only `"md5"` and `"label"`
(values `"good"` or `"bad"`) are required. The route layer / CLI maps
each dict's `md5` against `medias[*]["md5"]` to apply the vote - the
importer itself doesn't touch dataset state.

`list_label_importers()` skips importers with `hidden_from_picker =
True` (the `holder` scaffold). They remain reachable via
`get_label_importer(name)`.

Third-party importers can register via the
`vtscore.label_importers` entry-point group; built-ins win on name
collisions.

### Built-in label importers

| Name               | Display name      | Notes                                                                          |
|--------------------|-------------------|--------------------------------------------------------------------------------|
| `server_json_file` | Server JSON File  | Reads a `LabelSet`-format JSON file on the server filesystem.                  |
| `server_csv_file`  | Server CSV File   | Reads a CSV with `md5,label` columns from the server filesystem.              |
| `holder`           | Holder Package    | Scaffold for the external Holder API; `hidden_from_picker=True` until wired.  |

### Custom-importer skeleton

```python
# my_pkg/postgres_label_importer.py
from vtscore.labels.importers import LabelImporter, LabelImporterField

class PostgresLabelImporter(LabelImporter):
    name = "postgres"
    display_name = "PostgreSQL Query"
    description = "Import labels from a PostgreSQL query."
    fields = [
        LabelImporterField("host",     "Host",     "text"),
        LabelImporterField("database", "Database", "text"),
        LabelImporterField("query",    "Query",    "text",
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

`run_cli(field_values)` defaults to `self.run(field_values)`; override
it only when `run` expects non-string objects (e.g. Werkzeug
`FileStorage` for file uploads). `add_cli_arguments(parser)` is
auto-derived from `fields`, so most importers work from the CLI with
no extra code.

---

## Labelset sources

`LabelsetSource` (`vtscore/labels/sources/base.py:30`) extends
[`SyncSource[LoadT, SaveT]`](sync.md) for a detector's labelset. A
source can both *load* labels and *save* them back, so a detector
linked to a source auto-imports its labels on load and auto-exports
on every vote change. Standalone importers and exporters keep working
regardless of whether a source is active.

The generic parameters are `SyncSource[list[dict[str, str]],
LabelSet]`:

- `load(field_values) -> list[{"md5": ..., "label": ...}]` - the raw label list (compatibility with `LabelImporter.run`).
- `load_full(field_values) -> LabelSet` - the full `LabelSet`, including any `detector_meta` block. The default implementation wraps `load()` into a metadata-less `LabelSet`; override to surface `media_type` / `input_spec` / `threshold` round-tripped through the source.
- `save(labelset: LabelSet, field_values) -> None` - persist the labelset.

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

Substitution happens at `load` / `save` time and runs each value
through `vtscore.security.path_validation.sanitize_template_value`, so
an attacker-controlled detector name like `../../etc/passwd` cannot
escape the directory implied by an admin-configured template
(`vtscore/labels/sources/server_json_file/__init__.py:117`).
`resolve_filepath_for(field_values, detector_id=..., detector_name=...)`
exposes the substitution as a pure function for flows that need to
resolve a path for a non-active detector (notably detector rename).

### Custom-source skeleton

```python
# my_pkg/redis_labelset_source.py
from vtscore.datasets.labelset import LabelSet
from vtscore.labels.sources import LabelsetSource, LabelsetSourceField

class RedisLabelsetSource(LabelsetSource):
    name = "redis"
    display_name = "Redis Key"
    description = "Sync detector labels with a Redis key."
    fields = [
        LabelsetSourceField("host", "Host", "text", default="localhost"),
        LabelsetSourceField("key",  "Key",  "text",
                            description="Supports {detector_id} and {detector_name}."),
    ]

    def load(self, field_values):
        import json, redis
        r = redis.Redis(host=field_values["host"])
        raw = r.get(field_values["key"])
        if not raw:
            return []
        return json.loads(raw).get("labels", [])

    def save(self, labelset: LabelSet, field_values):
        import json, redis
        r = redis.Redis(host=field_values["host"])
        r.set(field_values["key"], json.dumps(labelset.to_dict()))

LABELSET_SOURCE = RedisLabelsetSource()
```

---

## Sync glue

`vtscore/labels/sync.py` wires labelset sources to the live detector
context. Two entry points:

| Function                           | When called                                                  |
|------------------------------------|--------------------------------------------------------------|
| `sync_to_labelset_source()`        | Whenever votes change. Schedules a **debounced background push**. |
| `sync_from_labelset_source(detector_id=None)` | On detector load or on manual import. Pulls + applies labels synchronously. |

### Debounced push

`sync_to_labelset_source` returns immediately. The actual push runs on
a background `threading.Timer` ~200ms (`_DEBOUNCE_DELAY`) after the
most recent call; further calls within the window cancel and restart
the timer, so a rapid voting burst collapses into a single sync run
that uses the **latest** captured contexts (latest wins). A per-detector
`_pending_syncs` slot keyed by `detector_id` keeps two concurrent
detectors from coalescing into each other's window
(`vtscore/labels/sync.py:60`).

`flush_pending_label_syncs()` drains the queue synchronously - used by
tests that need to assert the file was written, and by graceful
shutdown paths. An `atexit` hook also calls it so the most recent
vote's push survives normal interpreter exit (Ctrl-C, gunicorn
SIGQUIT, `sys.exit`). Hard kills (SIGKILL, `os._exit`) bypass `atexit`
and still drop the last ~200ms of work - accept this as the cost of
debouncing.

For test isolation, `reset_label_sync_for_tests()` *cancels* pending
syncs without firing them, which is what the `reset_state` autouse
fixture wants between tests so a sync scheduled by one test's contexts
can't fire after those contexts are gone.

### The `_syncing` guard

A module-level boolean (`_syncing`, guarded by `_sync_lock`) prevents
circular re-export: while `sync_from_labelset_source` is mid-apply,
any `sync_to_labelset_source` triggered by the resulting `apply_label`
calls is silently skipped. Without this, importing a labelset would
immediately push it right back to the source - fine for a no-op
round-trip, but pathological for sources that timestamp or version
their writes.

The guard is **module-level, not thread-local**: a `sync_from` running
on one thread blocks a parallel `sync_to` push that fires on another
thread mid-import. The pending-push timer thread re-checks the flag
inside `_sync_lock` immediately before writing, so a flag set after
the timer fires but before the push runs still suppresses the push.

### Detector meta round-trip

`load_full` returns the full `LabelSet`, including any `detector_meta`
block. When `sync_from_labelset_source` sees one, it folds the
source's `input_spec` (and `media_type`, when the receiving detector
is missing one) into the receiving detector's on-disk JSON. The
source's `threshold` is intentionally **not** applied - the receiver
retrains its MLP from the imported labels and recomputes its own
threshold (`vtscore/labels/sync.py:296`). The detector files
themselves only ever store origins and meta, never embeddings or MLP
weights (CLAUDE.md "No Persisted Vectors").

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
passed through to that source's `load` / `save`. Both
`sync_to_labelset_source` and `sync_from_labelset_source` silently
no-op when the field is absent or empty, so a detector without a
linked source costs nothing on every vote.

---

## Relationship to `vtscore.datasets`

Labels are not a separate domain - they're `LabeledElement` /
`LabelSet` from [`vtscore.datasets`](datasets.md#labelset). Both
plugin families operate on (or produce) that type:

- A `LabelImporter` returns `list[{"md5", "label"}]` for compatibility with the legacy label format. Wrapping that into a `LabelSet` is `LabelSet.from_dict({"labels": result})`.
- A `LabelsetSource.save` consumes a `LabelSet`; its `load` returns the same raw list shape for symmetry with importers, while `load_full` returns the full `LabelSet` for sources that carry richer metadata.
- The on-disk JSON format both built-ins use is the dict produced by `LabelSet.to_dict()` - a superset of the legacy `{"labels": [{"md5", "label"}]}` shape, with optional `origin`, `origin_name`, `filename`, `category`, `metadata`, `region_box` per element and an optional top-level `detector_meta` block.
