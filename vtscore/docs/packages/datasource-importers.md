# `vtscore.datasource_importers`

The **single-item** sibling of [dataset importers](datasets.md#importers).
A dataset importer's `run()` yields a whole collection; a datasource
importer's `fetch()` returns exactly one item's bytes.

It exists so a user supplying **exemplar media** - the image you want a
detector to find more of, the sound clip you want to seed a search with -
can pull it from the same kinds of places a whole dataset can come from:
a URL, a file already on the server, a third-party service. Without it,
exemplar media would be upload-only.

Related docs: [`plugins.md`](plugins.md) for the registry machinery this
family is built on; [`datasets.md`](datasets.md) for its bulk sibling and
for `MediaSource`, which is how an origin gets re-resolved later.

## Contents

| Module | Concern |
|--------|---------|
| `vtscore/datasource_importers/base.py` | `DataSourceImporter` ABC and the `FetchedMediaItem` record |
| `vtscore/datasource_importers/__init__.py` | The auto-discovering registry: `get_datasource_importer`, `list_datasource_importers` |
| `vtscore/datasource_importers/server_file/` | Pick one media file from the server's filesystem |
| `vtscore/datasource_importers/url_download/` | Download one media file from a public URL |

---

## The contract

```python
class DataSourceImporter(PluginBase):
    icon: str = "\U0001f4e5"            # inbox tray
    category: str = "services"          # picker tab: services | server | local | demo

    def fetch(self, field_values: dict[str, Any]) -> FetchedMediaItem: ...
    def get_field_options(self, field_key, current_values) -> list[FieldOption]: ...
```

`field_values` arrives **already validated and normalised** - text
stripped, `url` and `server_path` fields security-checked by
`vtscore.plugins.normalize`, `file` fields as `UploadedFile` objects. So
`fetch` should not re-implement those checks (though re-validating is
harmless and idempotent, which is what the built-in `server_file`
importer does so that direct library callers get the same confinement as
HTTP callers).

Raise `ValueError` for bad user input - a missing file, a malformed
reference. Any other exception is reported as an upstream/source
failure. The distinction reaches the user as a different message.

`category` deliberately reuses the dataset importers' category ids, so
both families share one tab bar in the example-media picker.

### `FetchedMediaItem`

| Field | Notes |
|-------|-------|
| `data` | The item's raw bytes |
| `filename` | A human-meaningful name. **Its suffix matters** - it drives media-type inference and how downstream code decodes the saved file, so keep the source's real extension |
| `origin` | Optional durable `{"importer": ..., "params": {...}}` so the item can be re-fetched later |

`origin` is the field worth thinking about. Set it when the item has a
**stable external identity** - a URL, a server path - so cross-dataset
label resolution and a deleted `example_media/` cache entry can both
re-derive the bytes. Leave it `None` when the bytes have no re-derivable
source and the saved snapshot is the only record.

There is a shortcut in how origins resolve: a param named **`path`** is
resolvable with no extra code, via the resolver's generic path fallback.
Any other param shape needs a matching `MediaSource` factory registered
under the importer's name. Both built-ins take the easy road - one uses
`path`, the other registers a URL-download source.

---

## Registration

Standard `vtscore.plugins` discovery. Expose a module-level
`DATASOURCE_IMPORTER` instance in a sub-package of
`vtscore/datasource_importers/`, or declare an entry point in the
`vtscore.datasource_importers` group.

```python
from vtscore.datasource_importers.base import DataSourceImporter, FetchedMediaItem
from vtscore.plugins import PluginField


class PastebinDataSourceImporter(DataSourceImporter):
    """Fetch a text snippet from a pastebin service."""

    name = "pastebin"
    display_name = "Pastebin"
    category = "services"
    fields = [
        PluginField(key="paste_id", label="Paste id", field_type="text", required=True),
    ]

    def fetch(self, field_values):
        data = ...  # download the snippet's bytes
        return FetchedMediaItem(data=data, filename=f"{field_values['paste_id']}.txt")


DATASOURCE_IMPORTER = PastebinDataSourceImporter()
```

Look one up, or list them all:

```python
from vtscore.datasource_importers import get_datasource_importer, list_datasource_importers

importer = get_datasource_importer("url_download")
item = importer.fetch({"url": "https://example.org/cat.jpg"})
```

The web app renders each importer's `fields` as a dynamic form - the
same machinery the Add Dataset modal uses - and calls
`POST /api/datasource-import/<name>`, which saves the fetched bytes into
the server-side example-media directory.

---

## Built-ins

| Name | Category | Notes |
|------|----------|-------|
| `server_file` | `server` | A `server_path` field. Re-validates through `validate_server_filepath` against the per-user base dir, then records the **validated** path as the origin's `path` param |
| `url_download` | `services` | A `url` field. The URL passes `vtscore.security.validate_url` (SSRF guard) at normalisation time and is re-checked on every redirect hop at fetch time |
