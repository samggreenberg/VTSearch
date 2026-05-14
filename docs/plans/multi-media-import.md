# Multi-media importing

Status: in progress (prototype landing alongside this doc).

## The problem

Today every dataset import has two orthogonal user-supplied inputs:

1. The importer's own `media_type` field — what type the user wants in the
   dataset (`audio`, `image`, `video`, `text`, `document`).
2. An optional `converters` field on a few importers (a comma-separated string
   of converter names) — extra source types to scan for and convert into the
   primary `media_type`.

For folder-shaped importers (`server_folder`, `local_folder`, `http_archive`)
this kind of works: the importer hands its directory to
`run_converters_on_folder()`, which globs for each converter's source-type
extensions and converts. The user only declares "convert videos to images"
once — the source-type discovery is implicit because the runner scans the
filesystem itself.

For service-style importers (ReCaller and hypothetical `DX`-style API
clients), the importer is the one that has to ask the upstream service for
media. There is no folder to glob over. If the user picks `media_type=image`
plus `converters=video2image,document2image`, the importer needs to know it
should request **images, videos, and documents** from the upstream service.
Today there is no convention for this — the importer would have to parse
`field_values["converters"]` itself, resolve each converter's `source_type`,
and union those types with `media_type`. None of the existing service-style
importer scaffolds do this; they only fetch the single `media_type`.

A second papercut: converters that take parameters (e.g. `video2image`'s
`n_clips`) accept those only via the constructor. The single registered
instance uses defaults. There is no way for a user to set "video → image at
30 clips" through the UI; the converter API surface only knows names.

## Goals

1. **One declaration per source type.** The user says "include videos via
   video2image at n_clips=30" once — they do not separately have to say
   "also include videos" and "also use video2image".
2. **Importers can iterate source types.** A service-style importer can
   ask `self.effective_source_specs(field_values)` and get back a list of
   `(source_type, converter_name, converter_params)` tuples to drive its
   upstream fetches.
3. **Converters carry their own parameters.** Each converter declares a
   `fields: list[PluginField]` (same `PluginField` already used by every
   plugin family). The frontend renders those fields inline per-row, and
   the values flow into `convert()` via a `params: dict` argument.
4. **External importers keep working.** Any current third-party
   `DatasetImporter` subclass continues to function as a single-type
   importer with bolt-on converters and converter defaults until the
   maintainer opts into the new model.

## Non-goals

- Multiple output media types in a single dataset. The dataset still has
  one `media_type`; multi-media-ness is on the **source** side. Every
  converter row produces the dataset's output type.
- Backwards-compat for existing `MediaConverter` subclasses outside this
  repo. There are none, so we break the converter ABC freely.

## Design

### MediaConverter ABC

Converters become `PluginBase`s with a `fields: list[PluginField]` class
attribute, just like every other plugin family. `convert()` gains a
`params: dict[str, Any]` argument. Class-level constants (e.g.
`Video2ImageMediaConverter.n_clips`) move into fields.

```python
class Video2ImageMediaConverter(MediaConverter):
    name = "video2image"
    display_name = "Video → Images"
    converter_description = "Extract frames from video files"
    fields = [
        PluginField(
            key="n_clips",
            label="Frames per video",
            field_type="text",
            default="10",
            description="Number of frames sampled (evenly spaced).",
        ),
    ]

    @property
    def source_type(self) -> str: return "video"
    @property
    def target_type(self) -> str: return "image"

    def convert(self, media: dict, params: dict) -> list[dict]:
        n_clips = int(params.get("n_clips") or 10)
        ...
```

`to_dict()` includes `fields` so the frontend can render parameter inputs
under each converter row. Default values are baked into fields, so an
importer that doesn't pass params (or passes an empty dict) gets the same
behaviour as today.

### Source specs

```python
@dataclass
class SourceSpec:
    source_type: str            # type_id (e.g. "video", "image")
    converter: str | None       # converter name, or None for "include directly"
    params: dict[str, Any]      # converter param values; ignored when converter is None
```

`DatasetImporter` gains a new class attribute and a helper:

```python
class DatasetImporter(PluginBase):
    multi_media: bool = False

    def effective_source_specs(self, field_values) -> list[SourceSpec]:
        ...
```

- When `multi_media=False` (default): the helper returns
  `[SourceSpec(field_values["media_type"], None, {})]` plus one
  `SourceSpec(source_type=conv.source_type, converter=conv.name, params={})`
  entry for each name in the legacy comma-separated `converters` field, so
  the helper is also useful to legacy importers that want to migrate to the
  new iteration style without flipping the flag yet.
- When `multi_media=True`: the helper parses `field_values["source_specs"]`
  — a list of dicts (or a JSON-string for multipart submissions) — into
  `SourceSpec`s. Validation: every spec must reference a real `source_type`,
  every named converter must exist and its `target_type` must equal the
  importer's chosen output media type, and at most one spec may have
  `converter=None`.

### Importer form schema

For `multi_media=True` importers, `to_dict()` emits one new compound field
in the serialised form:

```json
{
  "key": "source_specs",
  "label": "Include media",
  "field_type": "source_specs",
  "default": "[{\"source_type\":\"image\",\"converter\":null,\"params\":{}}]"
}
```

The frontend renders this as an "Include rows" repeater. The output media
type is a separate field on the importer (still called `media_type` for
continuity — its semantics narrow to "what type the dataset ends up
holding"). Each row picks a converter (filtered to those whose
`target_type` matches the output type) or "Include directly", and the
form renders the converter's own `PluginField`s inline.

Service-style importers (DX-style) iterate `self.effective_source_specs()`
in their `run()`:

```python
def run(self, field_values, medias, thin=False):
    for spec in self.effective_source_specs(field_values):
        records = self.list_records_for_source(spec.source_type, field_values)
        for record in records:
            raw = self.fetch_record(record, spec.source_type, field_values, thin=thin)
            if spec.converter is None:
                self._add_media(medias, raw)
            else:
                converter = get_converter(spec.converter)
                for out in converter.convert(raw, spec.params):
                    self._add_media(medias, out, target_type=spec.target_type)
```

### Runner update

`run_converters_on_folder()` grows a `converter_specs` parameter that
mirrors `SourceSpec` for converter rows (i.e. carries the per-converter
params), in addition to the existing `converter_names: list[str]` for the
legacy path. The body passes `params` into `converter.convert()`.

### Frontend

`dataset-importer-modal` renders:

- Old shape (most importers): unchanged — `media_type` select +
  `recursive` checkbox + whatever importer-specific fields exist.
- New shape (importers with `multi_media=True`): replaces the bare
  `media_type` field with an "Output media type" select plus an
  "Include …" repeater. Each repeater row:
  - Picks the **source type** (filtered to types that the chosen output
    type can be derived from — i.e. the output type itself + every source
    type for which a converter to that target exists).
  - When the source type ≠ output type, surfaces a converter dropdown
    (filtered to `target_type == output_type` and `source_type == source`)
    and the converter's parameter inputs inline.

The form submits `source_specs` as a JSON array. The Flask route
`/api/dataset/import/<name>` accepts `source_specs` alongside the legacy
fields and the existing `converters` pass-through.

## Shim semantics

| Importer flag | `media_type` field | `converters` field | `source_specs` field | Form UI |
|---------------|--------------------|--------------------|----------------------|---------|
| `multi_media=False` (default) | Yes (legacy) | Yes (legacy) | Ignored | Old "media_type + converters" |
| `multi_media=True`            | Yes (output type) | Ignored | Yes | New "output + Include rows" |

`effective_source_specs()` makes both paths converge inside the importer:
even a legacy importer can call it and get a useful spec list if it wants
to migrate its own `run()` to iterate. Once every in-tree importer flips,
we delete the legacy field handling.

### What unmigrated importers lose

Until a maintainer flips `multi_media=True`, their users:

- Cannot set converter parameters from the UI; the converter always runs
  with its declared `PluginField` defaults. (Previously the params lived
  on the converter class itself, so this is not a regression vs. main —
  parameters were never user-settable from the frontend at all.)
- Cannot mix-and-match per-source-type behaviour in one import (e.g.
  "video2image at n=10 and a second video2image row at n=30").

Neither is a behavioural regression for any current shipping importer.

## Migration checklist (for in-tree work after the prototype)

When migrating an importer to `multi_media=True`:

1. Set `multi_media = True` on the class.
2. Drop the `media_type` field's "scan filter" semantics — rename to
   `output_type` if the new meaning is clearer, otherwise leave alone.
3. Drop the `converters` field; the new repeater replaces it.
4. Rewrite `run()` to iterate `self.effective_source_specs(field_values)`.
   For folder-shaped importers this means calling a slimmed-down
   `run_converters_on_folder()` per spec; for service importers it means
   `list_records_for_source(spec.source_type, ...)`.
5. Update `build_origin()` to include the `source_specs` instead of the
   legacy `media_type` / `converters` pair.

## Scope landed in this PR

- ABC changes: `MediaConverter.fields`, `MediaConverter.convert(media,
  params)`, `DatasetImporter.multi_media`, `DatasetImporter.effective_source_specs()`,
  `SourceSpec` dataclass.
- `Video2ImageMediaConverter` migrated to expose `n_clips` as a
  `PluginField`. Other converters get an empty `fields=[]` (no
  user-visible change) and a no-op `params` arg.
- `ServerFolderDatasetImporter`, `ServerFilesDatasetImporter`,
  `LocalFolderDatasetImporter`, and `LocalFilesDatasetImporter` flipped
  to `multi_media=True`. Server-side `run()` / `run_chunked()` iterate
  `effective_source_specs()`; the lf-* importers are upload placeholders
  whose flow re-enters `server_folder`, so flipping the flag just lets
  the frontend render the Include-rows editor.
- `run_converters_on_folder()` accepts the new `SourceSpec` form.
- Source-specs editor rendered in all four pickers (sf-folder, sf-files
  via the generic form view, lf-folder, lf-files). Shared edit helpers
  in `dataset-importer-modal.component.ts`.
- Tests covering the new code path, the shim, and the `multi_media`
  flag on all migrated importers.

Still on the legacy shim (will migrate in followups):

- `pickle`, `combine_datasets`, `synthetic`, `http_archive`, `recaller`,
  `demo`. Each is mechanical — flip the flag and rewrite `run()` to
  iterate. ReCaller specifically waits on a working API client (see
  `plans/RCDatasetImporter.md`).

Followups (not in this PR):

- Once everything in-tree is migrated, delete the shim and the legacy
  `converters` form field, and rename `media_type` → `output_type`.
