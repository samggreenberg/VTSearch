# Writing a `LabelsetSource`

A labelset source is a bidirectional sync target for one detector's
labels. Once linked to a detector, every change to the detector's labels
schedules a paced background push of its whole labelset to the source,
and loading the detector (or an explicit
`POST .../labelset-source/sync`) pulls the source's labels back in and
merges them. A deployment can link every detector to one source at once,
which keeps a copy of each detector's labels current outside the server
(#4913). Sources can also surface detector-level metadata (input spec,
media type, etc.) alongside the labels, so a detector imported from a
source comes back fully configured. Use a source for ongoing sync with
an external store (server JSON file, database, S3 object, an LMS
gradebook); use a [label importer](label-importers.md) for one-shot
pulls.

Subclass [`LabelsetSource`](../../labels/sources/base.py), which
inherits from `SyncSource[list[dict[str, str]], LabelSet]`. Implement
`_do_load(field_values)` and `_do_save(labelset, field_values)` — the
underscored template methods, **not** the public `load` / `save` —
and optionally override `_do_load_full(field_values)` to surface richer
metadata. The library auto-discovers sources under
`vtscore.labels.sources` (sentinel `LABELSET_SOURCE`) and walks the
`vtscore.labelset_sources` entry-point group.

**App-side counterpart:** [`docs/EXTENDING-plugins.md § Adding a
Labelset Source`](../../../docs/EXTENDING-plugins.md#adding-a-labelset-source)
covers the UI / route wiring. This guide focuses on the library API.

## Contents

- [The contract](#the-contract)
- [`_do_load` vs. `_do_load_full`](#_do_load-vs-_do_load_full)
- [Template variables: `{detector_id}` and `{detector_name}`](#template-variables-detector_id-and-detector_name)
- [How a detector is linked](#how-a-detector-is-linked)
- [When your `_do_save` and `_do_load_full` run](#when-your-_do_save-and-_do_load_full-run)
- [Entry-point registration](#entry-point-registration)
- [Worked example](#worked-example)
- [Testing pattern](#testing-pattern)

## The contract

`LabelsetSource` is a `SyncSource[list[dict[str, str]], LabelSet]`
subclass. The generic parameters describe the load and save shapes:
loading returns a list of label dicts; saving accepts a `LabelSet`
([`vtscore/datasets/labelset.py`](../../datasets/labelset.py)). The
two shapes differ intentionally - loading is the cheap path that the
manual-sync endpoint and detector-reload flow use, while saving gets
the full structured labelset including per-element origins and
optional detector metadata.

**Override the underscored template methods.** `load` / `save` /
`load_full` / `peek_version` are framework-owned wrappers: each
normalizes a copy of `field_values` (whitespace strip, template
substitution, URL / path validation - see [Framework-side
normalization](README.md#framework-side-normalization)) and then
dispatches to the matching `_do_*` hook. A subclass that overrides
`load` or `save` directly replaces that wrapper, so its body runs on
**unnormalized, unvalidated** input and the framework's path
confinement never happens.

Required:

| Member | Type | Purpose |
|--------|------|---------|
| `name: str` | class attr | Snake-case identifier |
| `display_name: str` | class attr | Human-readable label |
| `description: str` | class attr | One-sentence subtitle |
| `fields: list[PluginField]` | class attr | User-configurable inputs |
| `_do_load(field_values)` | method | Return a list of `{"md5", "label", ...}` dicts |
| `_do_save(labelset, field_values)` | method | Persist a `LabelSet` |

Optional:

| Member | Default | Purpose |
|--------|---------|---------|
| `icon: str` | family default | Emoji in the UI |
| `_do_load_full(field_values)` | wraps `_do_load()` | Return a `LabelSet` carrying `detector_meta` too |
| `_do_peek_version(field_values)` | returns `None` | Cheap freshness probe (`st_mtime_ns`, `ETag`) so callers can skip a full load |

Expose `LABELSET_SOURCE = YourSource()` at module level.

Callers still use the **public** names - `src.load(fv)`,
`src.save(labelset, fv)` - so tests and app code are unaffected by the
split.

## `_do_load` vs. `_do_load_full`

`_do_load()` returns the minimum - a list of label dicts the receiving
detector can apply. The default `_do_load_full()` wraps each dict in a
`LabeledElement` and returns a `LabelSet(elements)`, which is enough
for sources that only round-trip label/MD5 pairs.

Override `_do_load_full()` when your source carries detector-level
metadata you want the receiving detector to adopt - its input spec
(media type, clipper choice). The returned
`LabelSet.detector_meta` is folded into the detector's on-disk JSON
via `apply_detector_meta`
([`vtscore/detectors/input_spec.py`](../../detectors/input_spec.py))
on import. A `threshold` in `detector_meta` (an older VTSearch pushed
one) is intentionally **not** persisted - the receiving detector retrains
its head from the imported labels and computes its own threshold; remember the
[no-persisted-vectors-or-MLPs invariant](README.md#shared-rules-for-every-plugin).

The built-in server-JSON-file source
([`vtscore/labels/sources/server_json_file/__init__.py`](../../labels/sources/server_json_file/__init__.py))
is the canonical reference: `_do_load()` returns raw label dicts;
`_do_load_full()` reads the same file but parses `detector_meta` too.

## Template variables: `{detector_id}` and `{detector_name}`

Two template variables resolve at runtime from the active
`DetectorContext`:

- `{detector_id}` - stable internal identifier
- `{detector_name}` - user-facing detector name (may be renamed; the
  rename routes resolve both the old and the new path and move an on-disk
  file to follow the new name)

**Declare them on the field; don't substitute them yourself.** List them
in the field's `template_vars` and the framework substitutes and
sanitises them (rules in
[Template variables](README.md#template-variables)). For a
`server_path` field the same pass then confines the resolved path to
the user's data directory and writes it back, so
`field_values["filepath"]` in your `_do_load` / `_do_save` is already
substituted *and* confined:

```python
PluginField(
    key="filepath",
    label="Save to (server path)",
    field_type="server_path",
    placeholder=f"{DATA_DIR}/labels/{{detector_name}}.labels.json",
    template_vars=("detector_id", "detector_name"),
)
```

The framework resolves the templates against the detector being synced,
not whichever one the request has active: a push runs on a background
timer, so it scopes a context carrying that detector's id and name.

The one case that still needs manual work is resolving a template for a
detector under a name it does not have yet - the rename routes have to
resolve both the old and the new path to move the file. The
built-in source's `resolve_filepath_for()` helper does that by hand with
`sanitize_template_value` + `validate_server_filepath`, and is the
pattern to copy.

## How a detector is linked

A detector's link to a source is resolved in one place,
[`vtscore.labels.binding.resolve_labelset_source`](../../labels/binding.py),
which every push, pull and route reads:

1. the detector's own binding, stored on its registry entry under
   `labelset_source` (`{"source_name": ..., "field_values": {...}}`);
   `{"source_name": "none"}` opts it out;
2. otherwise a binding set in memory on its loaded `DetectorContext`
   (`ctx.labelset_source`) - the library-level way to link a source
   without a registry;
3. otherwise the deployment default, `CoreConfig.default_labelset_source`
   (the app's `default_labelset_source` server setting), which the
   detector inherits;
4. otherwise nothing.

Every non-empty value in the default's `field_values` is **fixed**: it is
laid over the detector's own values whenever the detector syncs to the
default's source, never stored per detector, and never sent to a
browser. The app's New Detector form asks only for your source's other
fields (`unfixed_fields`), so a source whose fields an admin fills in
completely costs users nothing. A required field that is neither fixed,
filled in, nor defaulted is reported as missing, and the detector is not
synced until it has a value.

## When your `_do_save` and `_do_load_full` run

The engine is [`vtscore/labels/sync.py`](../../labels/sync.py). You do not
call it; knowing what it does tells you what your source can assume.

**Pushes (`_do_save`).** Every write of a detector's JSON file schedules
one, so votes, label imports, corrections and renames all reach your
source. The `LabelSet` you receive is the detector file's **whole**
labelset - every dataset's labels, not just the active dataset's votes -
plus a `detector_meta` block (`media_type`, `input_spec`), read when the
push runs. It is references to labels, nothing measured from them: no
threshold, and no `rank_at_vote` / `score_at_vote` in an element's
`vt:provenance` (#5131), so the labels can rebuild the detector under
another embedder. Pushes are paced per detector:

- never two at once for one detector, so `_do_save` is not re-entered for
  the same detector (different detectors may push concurrently);
- at least `CoreConfig.labelset_source_push_interval` seconds (default 60)
  between the starts of two pushes, so a slow target or a burst of votes
  costs one call per window;
- a payload identical to what your source is known to hold is not sent
  again.

`_do_save` runs on a background thread, as the detector's creator (which
is the user a `server_path` field is confined to).
`flush_pending_label_syncs()` runs every pending push now, ignoring the
pace; tests use it, and an `atexit` hook calls it so the newest labels
reach your source on a normal shutdown (SIGKILL loses what was waiting).

**Pulls (`_do_load_full`).** A pull merges what your source holds into
the detector file: labels it has and the file lacks are added, and when
both hold one item with different labels the caller decides who wins -
the file when the detector loads or is linked (it is never older than
what it pushed), your source on an explicit sync. A pull never removes a
label. **A detector's first push in a process pulls first**, so a push
never overwrites labels only your source held - which is how a detector
this server lost comes back: re-create it under the same name and its
labels are restored from your source. If `_do_load_full` raises, that
push is skipped rather than risk overwriting what it could not read.

A pull that changes the file schedules a push of the merged labels;
that push is a no-op when the merge changed nothing your source lacks.
Raising from `_do_save` or `_do_load_full` is never fatal: the failure is
logged, the detector file stays the record, and the next change tries
again.

## Entry-point registration

In-tree:

```
vtscore/labels/sources/<your_source>/__init__.py
```

Out-of-tree:

```toml
[project]
name = "vtsearch-mylabelsetsource"
version = "0.1.0"
dependencies = ["vtsearch"]

[project.entry-points."vtscore.labelset_sources"]
my_source = "my_pkg.labelset_source:LABELSET_SOURCE"
```

After `pip install`, the source appears in `list_labelset_sources()`,
the `/api/labelset-sources` endpoint, and the detector-link UI.

## Worked example

A labelset source backed by S3, with per-detector keys derived from
`{detector_name}`. Every detector linked to it pushes its labels to a
dedicated S3 object and pulls them back on reload.

Note how little the bodies do: no `.strip()`, no required-field check,
no template substitution, no `sanitize_template_value`. Declaring
`template_vars` on the key field buys all of it, and the framework
wrapper has already applied it by the time `_do_load` runs.

```python
# my_pkg/labelset_source.py
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from vtscore.labels.sources.base import LabelsetSource, PluginField

if TYPE_CHECKING:
    from vtscore.datasets.labelset import LabelSet


class S3LabelsetSource(LabelsetSource):
    """Sync a detector's labels with a JSON object in an S3 bucket."""

    name = "s3"
    display_name = "S3 Object"
    description = "Sync detector labels with a JSON object in an S3 bucket."
    icon = "☁️"  # cloud
    fields = [
        PluginField(
            key="bucket",
            label="S3 Bucket",
            field_type="text",
            required=True,
        ),
        PluginField(
            key="key",
            label="Object Key",
            field_type="text",
            description="Supports the {detector_name} template.",
            placeholder="vtsearch/labels/{detector_name}.json",
            required=True,
            # Without this line the framework performs no substitution
            # and the object key literally contains "{detector_name}".
            template_vars=("detector_name",),
        ),
    ]

    def _do_load(self, field_values: dict[str, Any]) -> list[dict[str, str]]:
        import boto3  # noqa: PLC0415
        s3 = boto3.client("s3")
        try:
            obj = s3.get_object(Bucket=field_values["bucket"], Key=field_values["key"])
        except s3.exceptions.NoSuchKey:
            return []
        data = json.loads(obj["Body"].read())
        labels = data.get("labels")
        if not isinstance(labels, list):
            return []
        return [e for e in labels if isinstance(e, dict)]

    def _do_load_full(self, field_values: dict[str, Any]) -> LabelSet:
        from vtscore.datasets.labelset import LabelSet as _LabelSet  # noqa: PLC0415

        import boto3  # noqa: PLC0415
        s3 = boto3.client("s3")
        try:
            obj = s3.get_object(Bucket=field_values["bucket"], Key=field_values["key"])
        except s3.exceptions.NoSuchKey:
            return _LabelSet()
        data = json.loads(obj["Body"].read())
        if not isinstance(data, dict):
            return _LabelSet()
        return _LabelSet.from_dict(data)

    def _do_save(self, labelset: LabelSet, field_values: dict[str, Any]) -> None:
        import boto3  # noqa: PLC0415
        s3 = boto3.client("s3")
        s3.put_object(
            Bucket=field_values["bucket"],
            Key=field_values["key"],
            Body=json.dumps(labelset.to_dict(), indent=2).encode("utf-8"),
            ContentType="application/json",
        )


LABELSET_SOURCE = S3LabelsetSource()
```

And the `pyproject.toml`:

```toml
[project.entry-points."vtscore.labelset_sources"]
s3 = "my_pkg.labelset_source:LABELSET_SOURCE"
```

After `pip install`, link the source to a detector via `PUT
/api/detectors/<name>/labelset-source` with body
`{"source_name": "s3", "field_values": {"bucket": "my-bucket", "key":
"vtsearch/labels/{detector_name}.json"}}`, or make it every detector's
target with the `default_labelset_source` server setting. Every change
then schedules a paced push to S3; loading the detector calls
`load_full()` and merges the round-tripped labels (plus any
`detector_meta` the source carried).

## Testing pattern

Labelset-source tests live in `tests_lib/io/` and benefit from the
autouse fixtures that reset detector contexts. Use
[`vtscore.labels.sync.flush_pending_label_syncs`](../../labels/sync.py)
in tests that assert the scheduled push actually wrote, or
[`vtscore.labels.sync.reset_label_sync_for_tests`](../../labels/sync.py)
to drop pending pushes between tests.

```python
# tests_lib/io/test_s3_labelset_source.py
from unittest.mock import patch

from vtscore.datasets.labelset import LabeledElement, LabelSet
from vtscore.labels.sources import get_labelset_source, list_labelset_sources


class TestS3LabelsetSourceRegistration:
    def test_is_discoverable(self):
        names = [s.name for s in list_labelset_sources()]
        assert "s3" in names

    def test_fields(self):
        src = get_labelset_source("s3")
        assert {f.key for f in src.fields} == {"bucket", "key"}


class TestS3LabelsetSourceLoad:
    def test_returns_empty_for_missing_key(self):
        src = get_labelset_source("s3")

        class _ClientError(Exception): pass
        class _S3:
            class exceptions:
                NoSuchKey = _ClientError
            def get_object(self, Bucket, Key):
                raise _ClientError

        with patch("boto3.client", return_value=_S3()):
            assert src.load({"bucket": "b", "key": "no-key"}) == []


class TestS3LabelsetSourceSave:
    def test_writes_labelset_dict(self):
        src = get_labelset_source("s3")
        labelset = LabelSet([LabeledElement(md5="abc", label="good")])
        captured: dict = {}

        class _S3:
            def put_object(self, **kwargs):
                captured.update(kwargs)

        with patch("boto3.client", return_value=_S3()):
            src.save(labelset, {"bucket": "b", "key": "k.json"})

        assert captured["Bucket"] == "b"
        assert captured["Key"] == "k.json"
        assert b'"labels"' in captured["Body"]
```

The built-in [`vtscore/labels/sources/server_json_file/__init__.py`](../../labels/sources/server_json_file/__init__.py)
is the closest reference implementation - single file, single
`template_vars`-declaring field, both `_do_load()` and `_do_load_full()`
overrides, atomic-write `_do_save()`.
