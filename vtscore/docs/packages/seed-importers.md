# `vtscore.seed_importers`

The **batch, verdict-free** sibling of
[datasource importers](datasource-importers.md). A datasource importer
fetches one exemplar the user picked by hand; a
[label importer](labels.md#label-importers) imports media that already carry a
`good` / `bad` verdict. A seed importer sits between the two: it returns a
*batch* of media with **no verdict attached** — items that are "close but
not quite" what the user is hunting for.

It exists so that "here is roughly the neighbourhood I'm interested in"
becomes an expressible way to start a detector, alongside a text query and
a hand-picked exemplar. The user labels from there.

Related docs: [`plugins.md`](plugins.md) for the registry machinery this
family is built on; [`detectors.md`](detectors.md) for what a detector does
with its examples; [`datasource-importers.md`](datasource-importers.md) for
the single-item sibling and the durable-origin contract they share.

## Contents

| Module | Concern |
|--------|---------|
| `vtscore/seed_importers/base.py` | `SeedImporter` ABC and the `SeedMediaItem` record |
| `vtscore/seed_importers/__init__.py` | The auto-discovering registry: `get_seed_importer`, `list_seed_importers` |

Nothing else: **no seed importer ships in-tree.** The family is an
extension point, so the registry is empty until a third-party package
registers one through the `vtscore.seed_importers` entry-point group.

---

## Seeds are queries, not labels

This is the whole reason the family is separate rather than a mode on an
existing one. Every item a seed importer produces is stored on the detector
as `{"type": "media", "value": <filename>, "labeled": false}`, and that flag
splits the two things a media example can mean:

| | Hand-picked exemplar | Imported seed |
|---|---|---|
| Stored as | `{type, value}` (+ `origin`) | same, plus `labeled: false` |
| Steers the first sort | yes | yes |
| Becomes a `good` `LabeledElement` | yes | **no** |
| Casts a good vote on load | yes | **no** |
| Counts toward `num_training` | yes | **no** |

Both halves of "no" go through one predicate,
`vtscore.detectors.media_seeding.is_labeled_example`, which
`labeled_elements_from_examples` and `seed_good_votes_from_examples` each
consult. An example with no `labeled` key is a verdict — that covers every
example predating this family, and every one the user picks by hand.

A seed still reaches the sort because the ranking path never looks at
labels: it reads the `example_media/` files directly and ranks the haystack
against the centroid of their embeddings. So a seeded detector opens on a
useful ordering while starting from zero training labels.

---

## The contract

```python
class SeedImporter(PluginBase):
    icon: str = "\U0001f331"            # seedling
    max_items: int = 100                # per-run batch cap

    def run(self, field_values: dict[str, Any]) -> list[SeedMediaItem]: ...
    def get_field_options(self, field_key, current_values) -> list[FieldOption]: ...
```

`field_values` arrives **already validated and normalised** — text
stripped, `url` and `server_path` fields security-checked by
`vtscore.plugins.normalize`, `file` fields as `UploadedFile` objects — so
`run` should not re-implement those checks.

Raise `ValueError` for bad user input (a malformed reference, an unknown
id). Any other exception is reported as an upstream/source failure. An
empty list is **not** an error: "nothing matched" is a real answer, and the
user is told so rather than being shown a failure.

`max_items` bounds one run. The route keeps the first `max_items` items and
reports the truncation, so a runaway query degrades to a short batch with a
visible warning instead of filling the example-media directory.

### `SeedMediaItem`

| Field | Notes |
|-------|-------|
| `data` | The item's raw bytes |
| `filename` | A human-meaningful name. **Its suffix matters** — it drives media-type inference and how downstream code decodes the saved file, so keep the source's real extension |
| `origin` | Optional durable `{"importer": ..., "params": {...}}` so the item can be re-fetched later |

`origin` follows exactly the contract described in
[`datasource-importers.md`](datasource-importers.md#fetchedmediaitem),
including the shortcut where a param named `path` resolves with no extra
code and any other param shape needs a matching `MediaSource` factory.

---

## Registration

Standard `vtscore.plugins` discovery. Declare an entry point in the
`vtscore.seed_importers` group:

```toml
[project.entry-points."vtscore.seed_importers"]
neighborhood = "my_pkg.seeds:SEED_IMPORTER"
```

```python
from vtscore.plugins import PluginField
from vtscore.seed_importers.base import SeedImporter, SeedMediaItem


class NeighborhoodSeedImporter(SeedImporter):
    """Seed a detector from a saved cluster of near-miss media."""

    fields = [
        PluginField(key="cluster_id", label="Cluster id", field_type="text"),
    ]

    def run(self, field_values):
        return [
            SeedMediaItem(data=blob, filename=name)
            for name, blob in fetch_cluster(field_values["cluster_id"])
        ]


SEED_IMPORTER = NeighborhoodSeedImporter()
```

An in-tree plugin would instead expose the same `SEED_IMPORTER` sentinel
from a sub-package of `vtscore/seed_importers/`.

Look one up, or list them all:

```python
from vtscore.seed_importers import get_seed_importer, list_seed_importers

for importer in list_seed_importers():
    print(importer.name, importer.max_items)
```

The web app renders each importer's `fields` as a dynamic form — the same
machinery the Add Dataset modal uses — behind its own tab in the New
Detector modal's Blank flow, and calls `POST /api/seed-import/<name>`,
which saves each returned item's bytes into the server-side example-media
directory. An install with no seed importers registered grows no tabs, so
the family costs nothing when unused.
