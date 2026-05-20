# Writing a `LabelsetSource`

A labelset source is a bidirectional sync target for one detector's
labels. Once linked to a detector, every vote change debounces a
background push to the source, and a detector reload (or an explicit
`POST .../labelset-source/sync`) pulls the source's current state back
in. Sources can also surface detector-level metadata (input spec,
media type, etc.) alongside the labels, so a detector imported from a
source comes back fully configured. Use a source for ongoing sync with
an external store (server JSON file, database, S3 object, an LMS
gradebook); use a [label importer](label-importers.md) for one-shot
pulls.

Subclass [`LabelsetSource`](../../labels/sources/base.py)
([`vtscore/labels/sources/base.py:30`](../../labels/sources/base.py)),
which inherits from `SyncSource[list[dict[str, str]], LabelSet]`.
Implement `load(field_values)` and `save(labelset, field_values)`;
optionally override `load_full(field_values)` to surface richer
metadata. The library auto-discovers sources under
`vtscore.labels.sources` (sentinel `LABELSET_SOURCE`) and walks the
`vtscore.labelset_sources` entry-point group.

**App-side counterpart:** [`docs/EXTENDING-plugins.md § Adding a
Labelset Source`](../../../docs/EXTENDING-plugins.md#adding-a-labelset-source)
covers the UI / route wiring. This guide focuses on the library API.

## Contents

- [The contract](#the-contract)
- [`load` vs. `load_full`](#load-vs-load_full)
- [Template variables: `{detector_id}` and `{detector_name}`](#template-variables-detector_id-and-detector_name)
- [Circular-trigger prevention (`_syncing` guard)](#circular-trigger-prevention-_syncing-guard)
- [Entry-point registration](#entry-point-registration)
- [Worked example](#worked-example)
- [Testing pattern](#testing-pattern)

## The contract

`LabelsetSource` is a `SyncSource[list[dict[str, str]], LabelSet]`
subclass. The generic parameters describe the load and save shapes:
`load()` returns a list of label dicts; `save()` accepts a `LabelSet`
([`vtscore/datasets/labelset.py`](../../datasets/labelset.py)). The
two shapes differ intentionally — `load()` is the cheap path that the
manual-sync endpoint and detector-reload flow use, while `save()` gets
the full structured labelset including per-element origins and
optional detector metadata.

Required:

| Member | Type | Purpose |
|--------|------|---------|
| `name: str` | class attr | Snake-case identifier |
| `display_name: str` | class attr | Human-readable label |
| `description: str` | class attr | One-sentence subtitle |
| `fields: list[PluginField]` | class attr | User-configurable inputs |
| `load(field_values)` | method | Return a list of `{"md5", "label", ...}` dicts |
| `save(labelset, field_values)` | method | Persist a `LabelSet` |

Optional:

| Member | Default | Purpose |
|--------|---------|---------|
| `icon: str` | family default | Emoji in the UI |
| `load_full(field_values)` | wraps `load()` | Return a `LabelSet` carrying `detector_meta` too |

Expose `LABELSET_SOURCE = YourSource()` at module level.

## `load` vs. `load_full`

`load()` returns the minimum — a list of label dicts the receiving
detector can apply. The default `load_full()` wraps each dict in a
`LabeledElement` and returns a `LabelSet(elements)`, which is enough
for sources that only round-trip label/MD5 pairs.

Override `load_full()` when your source carries detector-level
metadata you want the receiving detector to adopt — input spec
(media type, embedder, clipper choice), threshold, etc. The returned
`LabelSet.detector_meta` is folded into the detector's on-disk JSON
via `apply_detector_meta`
([`vtscore/detectors/input_spec.py`](../../detectors/input_spec.py))
on import. Threshold is intentionally **not** persisted from
`detector_meta` — the receiving detector retrains its MLP from the
imported labels and computes its own threshold; remember the
[no-persisted-vectors-or-MLPs invariant](README.md#shared-rules-for-every-plugin).

The built-in server-JSON-file source
([`vtscore/labels/sources/server_json_file/__init__.py:61`](../../labels/sources/server_json_file/__init__.py))
is the canonical reference: `load()` returns raw label dicts;
`load_full()` reads the same file but parses `detector_meta` too.

## Template variables: `{detector_id}` and `{detector_name}`

Field values support two template variables resolved at runtime from
the active `DetectorContext`:

- `{detector_id}` — stable internal identifier
- `{detector_name}` — user-facing detector name (may be renamed; the
  rename endpoint resolves both old and new paths to detect orphaned
  files)

Always run substituted values through
`vtscore.security.sanitize_template_value`
([`vtscore/security/path_validation.py:82`](../../security/path_validation.py))
before splicing into a filesystem path, URL, or anything else
attacker-controllable; otherwise a detector named `../../etc/passwd`
would escape the admin-configured template:

```python
from vtscore.security.path_validation import sanitize_template_value

filepath = filepath.replace("{detector_name}", sanitize_template_value(name))
```

When resolving paths in `load()` / `save()`, also call
`validate_server_filepath(filepath)` so the final resolved path stays
inside `SERVER_ROOTS`. The built-in server-JSON source's
`resolve_filepath_for()` helper is a useful pattern to copy.

## Circular-trigger prevention (`_syncing` guard)

When a source's `load()` runs, every label it applies would normally
fire `sync_to_labelset_source()` and push the same labels back to the
source — an infinite loop. The framework prevents this with a
module-level `_syncing` flag in
[`vtscore/labels/sync.py:41`](../../labels/sync.py), coordinated by a
re-entrant `_sync_lock`. The flag is set for the duration of a
`sync_from_labelset_source()` import pass, and
`sync_to_labelset_source()` checks it (both at scheduling time and at
execution time) to suppress the push.

You don't have to do anything in your source to participate — the
guard wraps the apply pass, not your `load()` or `save()`. Just know
that:

- `save()` will not be called as a side effect of `load()` returning
  labels;
- two concurrent `sync_to`s from different threads serialize on
  `_sync_lock`;
- the `atexit` hook flushes the debounce queue at interpreter exit so
  the last 200ms of votes get pushed (SIGKILL bypasses atexit and
  drops that window).

If your source's `save()` is expensive, the 200ms debounce coalesces
rapid voting bursts into a single push; the in-flight write is held
by `_workers_lock` so `flush_pending_label_syncs()` (used by tests
and graceful shutdown) waits for it.

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

```python
# my_pkg/labelset_source.py
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from vtscore.labels.sources.base import LabelsetSource, LabelsetSourceField
from vtscore.security.path_validation import sanitize_template_value

if TYPE_CHECKING:
    from vtscore.datasets.labelset import LabelSet


def _resolve_key(field_values: dict[str, Any]) -> str:
    """Substitute {detector_name} into the user-provided key template."""
    key = (field_values.get("key") or "").strip()
    if not key:
        raise ValueError("S3 object key is required.")
    if "{detector_name}" in key:
        from vtscore.state.core import get_active_detector_context  # noqa: PLC0415

        ctx = get_active_detector_context()
        if ctx is None:
            raise ValueError("No active detector context for {detector_name} substitution.")
        key = key.replace("{detector_name}", sanitize_template_value(ctx.name))
    return key


class S3LabelsetSource(LabelsetSource):
    """Sync a detector's labels with a JSON object in an S3 bucket."""

    name = "s3"
    display_name = "S3 Object"
    description = "Sync detector labels with a JSON object in an S3 bucket."
    icon = "☁️"  # cloud
    fields = [
        LabelsetSourceField(
            key="bucket",
            label="S3 Bucket",
            field_type="text",
            required=True,
        ),
        LabelsetSourceField(
            key="key",
            label="Object Key",
            field_type="text",
            description="Supports {detector_name} template.",
            placeholder="vtsearch/labels/{detector_name}.json",
            required=True,
        ),
    ]

    def load(self, field_values: dict[str, Any]) -> list[dict[str, str]]:
        import boto3  # noqa: PLC0415
        bucket = field_values["bucket"]
        key = _resolve_key(field_values)
        s3 = boto3.client("s3")
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
        except s3.exceptions.NoSuchKey:
            return []
        data = json.loads(obj["Body"].read())
        labels = data.get("labels")
        if not isinstance(labels, list):
            return []
        return [e for e in labels if isinstance(e, dict)]

    def load_full(self, field_values: dict[str, Any]) -> LabelSet:
        from vtscore.datasets.labelset import LabelSet as _LabelSet  # noqa: PLC0415

        import boto3  # noqa: PLC0415
        bucket = field_values["bucket"]
        key = _resolve_key(field_values)
        s3 = boto3.client("s3")
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
        except s3.exceptions.NoSuchKey:
            return _LabelSet()
        data = json.loads(obj["Body"].read())
        if not isinstance(data, dict):
            return _LabelSet()
        return _LabelSet.from_dict(data)

    def save(self, labelset: LabelSet, field_values: dict[str, Any]) -> None:
        import boto3  # noqa: PLC0415
        bucket = field_values["bucket"]
        key = _resolve_key(field_values)
        s3 = boto3.client("s3")
        s3.put_object(
            Bucket=bucket,
            Key=key,
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
"vtsearch/labels/{detector_name}.json"}}`. Every vote then debounces a
push to S3; reloading the detector calls `load_full()` and applies
the round-tripped labels (plus any `detector_meta` the source
carried).

## Testing pattern

Labelset-source tests live in `tests_lib/io/` and benefit from the
autouse fixtures that reset detector contexts. Use
[`vtscore.labels.sync.flush_pending_label_syncs`](../../labels/sync.py)
in tests that assert the debounced push actually wrote, or
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
is the closest reference implementation — single file, single field,
both `load()` and `load_full()` overrides, atomic-write `save()`.
