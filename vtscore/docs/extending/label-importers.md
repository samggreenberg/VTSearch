# Writing a `LabelImporter`

A label importer pulls `(md5, label)` pairs from somewhere and applies
them to the active dataset / detector - one-shot, no ongoing sync. Use
it when you have a list of pre-existing labels (a CSV from an
analyst, a database query, a JSON file from a teammate) and want to
seed the detector's good / bad votes from that. For ongoing
bidirectional sync (auto-export on vote change, auto-import on detector
load), write a [labelset source](labelset-sources.md) instead.

Subclass [`LabelImporter`](../../labels/importers/base.py)
([`vtscore/labels/importers/base.py`](../../labels/importers/base.py)),
declare `fields`, and implement `run(field_values) -> list[dict]`. The
library auto-discovers importers under `vtscore.labels.importers`
(sentinel `LABEL_IMPORTER`) and walks the `vtscore.label_importers`
entry-point group.

**App-side counterpart:** [`docs/EXTENDING-plugins.md § Adding a Label
Importer`](../../../docs/EXTENDING-plugins.md#adding-a-label-importer).
This guide focuses on the library API.

## Contents

- [Importer vs. source: when to write which](#importer-vs-source-when-to-write-which)
- [The contract](#the-contract)
- [The returned label format](#the-returned-label-format)
- [Entry-point registration](#entry-point-registration)
- [Worked example](#worked-example)
- [Testing pattern](#testing-pattern)

## Importer vs. source: when to write which

| You want… | Build a… |
|-----------|---------|
| One-shot import of an existing label file / query | **`LabelImporter`** - `run()` returns a list of label dicts; the user triggers it manually |
| Auto-export as labels change, auto-import on detector load | **`LabelsetSource`** - `load()` / `save()` for bidirectional sync; can also surface detector metadata |
| Both - manual one-shot import AND ongoing sync from the same backend | Two classes that share helpers, or just a `LabelsetSource` (manual sync is `POST /api/detectors/<detector_ref>/labelset-source/sync`) |

Importers are simpler - no push pacing, no merging pulls, no atexit
flush. Reach for the source only when ongoing sync is the
actual requirement.

## The contract

`LabelImporter` is a `PluginBase` subclass. Required:

| Member | Type | Purpose |
|--------|------|---------|
| `name: str` | class attr | Snake-case identifier |
| `display_name: str` | class attr | Human-readable label |
| `description: str` | class attr | One-sentence subtitle |
| `fields: list[PluginField]` | class attr | User-configurable inputs |
| `run(field_values)` | method | Return a list of `{"md5", "label", ...}` dicts |

Optional:

| Member | Default | Purpose |
|--------|---------|---------|
| `icon: str` | `"🏷️"` | Emoji in the UI |
| `run_cli(field_values)` | delegates to `run()` | Override only when CLI values need different handling |
| `default_detector_name(field_values)` | the first `file` field's filename stem, else `"<display_name> detector"` | The name a detector created from this importer gets by default; the app prefills the New Detector › Trained form's name box with it, live. Keep it cheap: it runs on every form change |

Expose `LABEL_IMPORTER = YourImporter()` at module level.

A detector your importer creates (the Dashboard's New Detector › Trained, or
the CLI's `--import-labels-into … --create-detector`) keeps a record of the run
on its registry entry: your `name`, the field values it ran with, and the time
(`vtscore.detectors.import_record.build_import_record`). The fields kept follow
a dataset origin's policy: `"file"` and `"password"` fields are left out, and
`PluginField.include_in_origin` / `origin_serializer` tune any field as they do
for a dataset importer. Set `include_in_origin=False` on a field whose value
should not be written to disk.

## The returned label format

`run()` must return a list of dicts. The minimum shape:

```python
[
    {"md5": "<media-md5>", "label": "good"},
    {"md5": "<media-md5>", "label": "bad"},
    ...
]
```

Labels must be `"good"` or `"bad"`; any other value is counted as
skipped by the route handler. The route matches each entry against the
active dataset by `origin` + `origin_name` and by `md5` (the union of
both), falling back to `origin_name` or `filename` alone only when the
entry carries neither (`vtscore.state.media_lookup.resolve_media_ids`).
An entry that matches nothing but carries an `origin` is pulled in from
that origin by a background task and labelled once it lands
([`vtscore/datasets/ingest.py`](../../datasets/ingest.py)); one without
an `origin` is reported as unresolved. A label-importer's job is to
hand the route a list; the route's job is to apply what fits.

So return `origin` / `origin_name` alongside `md5` when your backend
knows them. A `metadata` dict is read for the vote's provenance
(`vtscore.datasets.vote_provenance.read_provenance`); the route ignores
any other key.

## Entry-point registration

In-tree:

```
vtscore/labels/importers/<your_importer>/__init__.py
```

Out-of-tree:

```toml
[project]
name = "vtsearch-mylabelimporter"
version = "0.1.0"
dependencies = ["vtsearch"]

[project.entry-points."vtscore.label_importers"]
my_importer = "my_pkg.label_importer:LABEL_IMPORTER"
```

After `pip install`, the importer appears in `list_label_importers()`,
the `/api/label-importers` endpoint, and the inventory.

## Worked example

A label importer that pulls labels from a Redis stream - each entry is
a small JSON blob `{"md5": "...", "label": "good", "annotator": "..."}`.

```python
# my_pkg/label_importer.py
from __future__ import annotations

import json
from typing import Any

from vtscore.labels.importers.base import LabelImporter, PluginField


class RedisStreamLabelImporter(LabelImporter):
    """Read labels from a Redis stream."""

    name = "redis_stream"
    display_name = "Redis Stream"
    description = "Import labels from a Redis stream of JSON entries."
    icon = "\U0001f5c4"  # file cabinet
    fields = [
        PluginField(
            key="redis_url",
            label="Redis URL",
            # Not "url": that field type is SSRF-checked by the framework and
            # accepts http(s) only, so a redis:// value would be rejected.
            # A plain text field gets no such check - vet the host yourself
            # if users you don't trust can set it.
            field_type="text",
            description="redis:// or rediss:// URL.",
            required=True,
        ),
        PluginField(
            key="stream_name",
            label="Stream name",
            field_type="text",
            default="vtsearch:labels",
            required=True,
        ),
        PluginField(
            key="count",
            label="Max entries",
            field_type="number",
            default="1000",
            min="1",
            max="100000",
            step="1",
            required=False,
        ),
    ]

    def run(self, field_values: dict[str, Any]) -> list[dict[str, str]]:
        try:
            import redis  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("Install 'redis' to use the redis_stream label importer.") from exc

        count = int(field_values.get("count") or 1000)
        client = redis.from_url(field_values["redis_url"])
        entries = client.xrange(field_values["stream_name"], count=count)

        labels: list[dict[str, str]] = []
        for _entry_id, fields_dict in entries:
            # XRANGE returns {b"json": b"..."} when the producer used XADD ... json '...'.
            raw = fields_dict.get(b"json")
            if raw is None:
                continue
            try:
                parsed = json.loads(raw)
            except (TypeError, ValueError):
                continue
            md5 = parsed.get("md5")
            label = parsed.get("label")
            if not md5 or label not in ("good", "bad"):
                continue
            row: dict[str, str] = {"md5": md5, "label": label}
            if "annotator" in parsed:
                row["annotator"] = parsed["annotator"]  # carried along; the import route ignores it
            labels.append(row)
        return labels


LABEL_IMPORTER = RedisStreamLabelImporter()
```

And the `pyproject.toml`:

```toml
[project.entry-points."vtscore.label_importers"]
redis_stream = "my_pkg.label_importer:LABEL_IMPORTER"
```

## Testing pattern

Label-importer tests live in `tests_lib/labels/`. Use the autouse
`reset_contexts` fixture to clear state between tests; mock the
external service so the test runs offline.

```python
# tests_lib/labels/test_redis_label_importer.py
from unittest.mock import patch

from vtscore.labels.importers import get_label_importer, list_label_importers


class TestRedisStreamRegistration:
    def test_is_discoverable(self):
        names = [imp.name for imp in list_label_importers()]
        assert "redis_stream" in names

    def test_metadata(self):
        imp = get_label_importer("redis_stream")
        assert imp.display_name
        keys = [f.key for f in imp.fields]
        assert set(keys) == {"redis_url", "stream_name", "count"}


class TestRedisStreamRun:
    def test_filters_invalid_entries(self):
        imp = get_label_importer("redis_stream")
        fake_entries = [
            (b"1-0", {b"json": b'{"md5": "abc", "label": "good"}'}),
            (b"2-0", {b"json": b'{"md5": "def", "label": "neither"}'}),  # bad label
            (b"3-0", {b"json": b'{"label": "good"}'}),                   # missing md5
            (b"4-0", {b"json": b"not-json"}),                            # parse error
            (b"5-0", {b"json": b'{"md5": "ghi", "label": "bad", "annotator": "alice"}'}),
        ]

        class _FakeRedis:
            def xrange(self, name, count): return fake_entries

        with patch("redis.from_url", return_value=_FakeRedis()):
            labels = imp.run({
                "redis_url": "redis://localhost",
                "stream_name": "test",
                "count": "10",
            })

        assert labels == [
            {"md5": "abc", "label": "good"},
            {"md5": "ghi", "label": "bad", "annotator": "alice"},
        ]
```

See [`vtscore/labels/importers/server_json_file/__init__.py`](../../labels/importers/server_json_file/__init__.py)
for a real working importer that reads from a JSON file, and
[`tests_lib/datasets/test_importers.py`](../../../tests_lib/datasets/test_importers.py)
for the general I/O test pattern.
