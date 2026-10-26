# `vtscore.labels`

Label I/O - the two plugin families and the sync glue that move
detector labels between VTSearch and external systems. A *label
importer* is one-way (read labels from somewhere external, apply them
to the active detector). A *labelset source* is bidirectional: it
both loads labels from an external store and pushes a detector's labels
back whenever they change. Both produce / consume the same
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
| `vtscore/labels/binding.py` | Which source a detector syncs with: its registry binding, a context binding, or the deployment default |
| `vtscore/labels/sync.py` | The sync engine: paced push on every detector write, merging pull |

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
| `server_csv_file`  | Server CSV File   | Reads a CSV with `md5,label` columns (optional `origin_name`, `filename`, `category`, JSON `origin`, and a Good's JSON `region_box`) from the server filesystem. |

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
| `load_full(field_values)` | `_do_load_full` (optional) | The full `LabelSet`, including any `detector_meta` block. The default wraps `_do_load` into a metadata-less `LabelSet`; override to surface `media_type` / `input_spec`. |
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

## Sync engine

`vtscore/labels/sync.py` keeps a detector's labels current at its
linked source; `vtscore/labels/binding.py` decides which source that is.
The library guide's [When your `_do_save` and `_do_load_full`
run](../extending/labelset-sources.md#when-your-_do_save-and-_do_load_full-run)
is the source author's view of the same contract.

| Function | When called |
|----------|-------------|
| `note_detector_written(data)` | By `vtscore.detectors.store` on every detector-file write. Schedules a push; never raises. |
| `schedule_labelset_push(detector_name, *, detector_id="")` | Schedule a paced push of that detector's labels. A no-op when it syncs nowhere or its source is missing a field value. |
| `sync_to_labelset_source()` | Schedule a push for the active detector (every write already does; this is for library code that linked a source on a bare `DetectorContext`). |
| `sync_from_labelset_source(detector_id=None, *, prefer="source")` | The manual "sync now": pull and merge, the source winning a disagreement. Returns the label dicts the source held, or `None` when the detector syncs nowhere, a field value is missing, the read failed, or the source is empty. |
| `reconcile_labelset_source(detector_name, *, detector_id="")` | On detector load and when its link changes: pull and merge with the file winning, then schedule a push. Returns how many labels it added or changed; never raises. |

### Binding

`resolve_labelset_source(entry)` resolves, in order: the detector's
registry entry `labelset_source` (`{"source_name": "none"}` opts out),
a binding on its loaded `DetectorContext.labelset_source`, then the
deployment default `CoreConfig.default_labelset_source`. Non-empty
values in the default's `field_values` are fixed and overlay the
detector's own whenever it syncs to the default's source. The result
carries `missing_fields` (required fields with no value and no default);
a detector with any is not synced. `labelset_source_status(entry)` is
the same answer without field values, for listings.

```python
# Link a registered detector to a source (what the app's PUT route stores).
from vtscore.detectors.registry import update_detector

update_detector(
    detector_id,
    labelset_source={
        "source_name": "server_json_file",
        "field_values": {"filepath": "/data/labels/{detector_name}.labels.json"},
    },
)
```

### Push

The payload is the detector file's **whole labelset** - every dataset's
labels, not the active dataset's votes - plus `detector_meta`
(`media_type`, `input_spec`), read when the push runs. It goes out as every
labelset export does, through `labelset_for_export`
(`vtscore/datasets/labelset.py`): no threshold in `detector_meta`, and no
`rank_at_vote` / `score_at_vote` in any element's `vt:provenance`. Those were
measured by this server's embedder and head, so they mean nothing to a
receiver, which trains its own (#5131); the detector's own JSON keeps them.
Pacing is per detector name:

- pushes never overlap; a change during one sets a dirty flag and rides
  the next push;
- two pushes start at least `CoreConfig.labelset_source_push_interval`
  seconds apart (default 60), and never sooner than 200ms
  (`_DEBOUNCE_DELAY`) after the change that scheduled them;
- a payload whose digest matches what the source is known to hold (the
  last push, or the last pull) is skipped;
- a detector's first push in a process pulls first, so it never
  overwrites labels only the source held, and a source that cannot be
  read is not pushed to.

The push runs on a background timer as the detector's creator, with a
context carrying the detector's id and name so `{detector_*}` templates
resolve to it. `flush_pending_label_syncs()` runs every pending push now
(ignoring the pace) and waits for in-flight ones; an `atexit` hook does
the same on a normal exit, so only a hard kill loses what was waiting.
`reset_label_sync_for_tests()` *cancels* pending pushes instead.

### Pull

A pull reads `load_full()` and merges it into the detector file under
`label_sync_write_lock`: labels the source has and the file lacks are
appended, and on a disagreement the `prefer` side wins (`"local"` for
`reconcile_labelset_source`, `"source"` for the manual sync). No label
is removed. The source's `input_spec` (and `media_type`, when the file
has none) is folded in; its `threshold` is intentionally **not** - the
receiver retrains its head and derives its own (detector files store
origins and meta, never embeddings or weights; see
[architecture.md](../architecture.md#the-no-persisted-vectors-rule)).
A pull that changed the file updates the registry's `num_training`,
re-derives a loaded detector's votes (manual sync), and - through the
write - schedules a push of the merged labels.

---

## Relationship to `vtscore.datasets`

Labels are not a separate domain - they're `LabeledElement` /
`LabelSet` from [`vtscore.datasets`](datasets.md#labelset). Both
plugin families operate on (or produce) that type:

- A `LabelImporter` returns `list[{"md5", "label"}]` for compatibility with the legacy label format. Wrapping that into a `LabelSet` is `LabelSet.from_dict({"labels": result})`.
- A `LabelsetSource.save` consumes a `LabelSet`; its `load` returns the same raw list shape for symmetry with importers, while `load_full` returns the full `LabelSet` for sources that carry richer metadata.
- The on-disk JSON format both built-ins use is the dict produced by `LabelSet.to_dict()` - a superset of the legacy `{"labels": [{"md5", "label"}]}` shape, with optional `origin`, `origin_name`, `filename`, `category`, `metadata`, `region_box` per element and an optional top-level `detector_meta` block.

---

## Cross-references

- [`sync.md`](sync.md) - the `SyncSource` ABC (the public / hook split, `peek_version`) that `LabelsetSource` extends.
- [`datasets.md`](datasets.md#domain-objects) - `LabelSet`, `LabeledElement` and `Origin`, the types both families read and produce.
- [`detectors.md`](detectors.md#sync-helpers) - `label_sync_write_lock`, which a pull merges under, and the vote-to-labelset persist whose detector-file write schedules a push.
- [`config.md`](config.md#the-coreconfig-dataclass) - `default_labelset_source` and `labelset_source_push_interval`, the deployment-wide binding and pacing.
- [`../extending/label-importers.md`](../extending/label-importers.md) and [`../extending/labelset-sources.md`](../extending/labelset-sources.md) - writing a `LabelImporter` or a `LabelsetSource`; [`docs/EXTENDING-plugins.md`](../../../docs/EXTENDING-plugins.md#adding-a-labelset-source) has the app-side route and UI wiring.
