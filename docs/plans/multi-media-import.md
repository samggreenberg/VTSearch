# Multi-media importing

Status: **Done.** The `multi_media` flag is gone (every importer is
multi-media-aware by construction), the framework owns conversion
end-to-end, and the legacy `converters` form-field path has been
deleted.  See "Final cleanup" at the bottom for the deletion checklist.

Any third-party `DatasetImporter` subclass still using the legacy
`converters` form field or `multi_media=False` semantics will need to
migrate; see the migration checklist below.

## The problem

Today every dataset import has two orthogonal user-supplied inputs:

1. The importer's own `media_type` field: what type the user wants in the
   dataset (`audio`, `image`, `video`, `text`, `document`).
2. An optional `converters` field on a few importers (a comma-separated string
   of converter names): extra source types to scan for and convert into the
   primary `media_type`.

For folder-shaped importers (`server_folder`, `local_folder`, `http_archive`)
this kind of works: the importer hands its directory to
`run_converters_on_folder()`, which globs for each converter's source-type
extensions and converts. The user only declares "convert videos to images"
once; the source-type discovery is implicit because the runner scans the
filesystem itself.

For service-style importers (ReCaller and hypothetical `DX`-style API
clients), the importer is the one that has to ask the upstream service for
media. There is no folder to glob over. If the user picks `media_type=image`
plus `converters=video2image,document2image`, the importer needs to know it
should request **images, videos, and documents** from the upstream service.
Today there is no convention for this; the importer would have to parse
`field_values["converters"]` itself, resolve each converter's `source_type`,
and union those types with `media_type`. None of the existing service-style
importer scaffolds do this; they only fetch the single `media_type`.

A second papercut: converters that take parameters (e.g. `video2image`'s
`n_clips`) accept those only via the constructor. The single registered
instance uses defaults. There is no way for a user to set "video → image at
30 clips" through the UI; the converter API surface only knows names.

## Goals

1. **One declaration per source type.** The user says "include videos via
   video2image at n_clips=30" once; they do not separately have to say
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
    description = "Extract frames from video files"
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
  (a list of dicts, or a JSON-string for multipart submissions) into
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
continuity; its semantics narrow to "what type the dataset ends up
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

- Old shape (most importers): unchanged. `media_type` select +
  `recursive` checkbox + whatever importer-specific fields exist.
- New shape (importers with `multi_media=True`): replaces the bare
  `media_type` field with an "Output media type" select plus an
  "Include …" repeater. Each repeater row:
  - Picks the **source type** (filtered to types that the chosen output
    type can be derived from (i.e. the output type itself + every source
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
  on the converter class itself, so this is not a regression vs. main;
  parameters were never user-settable from the frontend at all.)
- Cannot mix-and-match per-source-type behaviour in one import (e.g.
  "video2image at n=10 and a second video2image row at n=30").

Neither is a behavioural regression for any current shipping importer.

## Migration checklist (for in-tree work after the prototype)

When migrating an importer to `multi_media=True`:

1. Set `multi_media = True` on the class.
2. Drop the `media_type` field's "scan filter" semantics; rename to
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

## What shipped (framework-driven conversion)

Earlier rounds left importers to drive the converter loop themselves;
the docs example showed a `DXImporter` calling
`get_converter(spec.converter).convert(raw, spec.params)` inside its
`run()`.  That leak is now closed:

- New hook on `DatasetImporter`:
  `fetch_source_media(spec, field_values, thin=False) -> Iterator[dict]`.
  Subclasses yield raw media of `spec.source_type` and never touch the
  converter registry.
- The base-class `run()` now loops `effective_source_specs()`, calls
  `fetch_source_media()` once per spec, and runs
  `converter.convert(raw, spec.params)` itself when the spec declares a
  converter.  IDs and default origins are assigned by the framework
  exactly as before.
- The default `fetch_source_media()` delegates to
  `list_records()` + `fetch_records_bulk()`, so single-source-type
  service importers (which only ever pull one type per import) keep
  working without the new hook.
- The DX docs example was rewritten to use the new hook and no longer
  shows manual `get_converter()` calls.
- `recaller` migrated to be truly multi-source-type-aware: its old
  `list_records` / `fetch_record` / `_fetch_records_bulk_impl` trio was
  replaced by a single `fetch_source_media()` that filters
  `_rc_fetch_results` by `spec.source_type`.  A user can now build a
  single ReCaller-backed dataset that pulls in (say) images directly +
  videos converted to images + documents converted to images, with the
  framework running each converter.

## What shipped (bulk-fetch escape hatch)

Service-style importers had only one framework-driven fetch shape:
`fetch_source_media(spec)` called once per spec.  That works when the
backend serves one media type per query, but it forces N upstream calls
when a single query naturally returns mixed types.  Adding a second
fetch hook for that case:

- New optional hook on `DatasetImporter`:
  `fetch_all_source_media(specs, field_values, thin=False) -> Iterator[tuple[SourceSpec, dict]]`.
  Override this when one upstream call covers every spec; yield
  `(spec, raw_media)` pairs and the framework handles converter
  dispatch and ingestion exactly as it does for the per-spec hook.
- The base-class `run()` now calls `fetch_all_source_media()` once
  instead of looping `fetch_source_media()` itself.  The default
  `fetch_all_source_media()` delegates to `fetch_source_media()` per
  spec, so existing per-spec importers see no behavioural change.
- Converter lookups in `run()` are cached per-spec so a bulk importer
  that interleaves specs across yields doesn't re-resolve the same
  converter on every pair.
- Subclasses still never call `get_converter()` themselves; the
  framework owns conversion and ingestion regardless of which fetch
  hook is overridden.

## What shipped (latest round)

Flipped `multi_media = True` on the last six in-tree importers so the
in-tree set is uniformly off the legacy shim:

- `http_archive`: real migration.  `run()` / `run_chunked()` iterate
  `effective_source_specs()`; converter rows go through a thin
  `_run_converter_specs()` wrapper around the typed
  `run_converters_on_folder(converter_specs=...)` entry point.
  `build_origin()` / `build_cli_args()` serialise `source_specs` JSON
  instead of the legacy CSV `converters` field.
- `synthetic`, `recaller`: have a `media_type` field but only ever pull
  one source type per import, so `run()` keeps reading
  `field_values["media_type"]` directly.  Label updated to "Output
  Media Type".  No spec iteration added; there are no converter rows
  for these flows.
- `pickle`, `combine_datasets`, `demo`: no `media_type` field, no
  `converters` field, no spec iteration in `run()`.  The flag flip is
  purely to take them off the legacy shim; their custom / file-upload
  UI modes mean the multi-media editor never renders anyway, and
  `effective_source_specs()` is never called on them.

Tests: each migrated importer now asserts `multi_media=True`.  The
legacy synthesis branch of `effective_source_specs()` is still
exercised via a `_LegacyTestImporter` stand-in (a tiny in-test
`DatasetImporter` subclass with `multi_media=False`) so the shim stays
covered for external importers until removal.

## Final cleanup

The shim deletion landed.  Removed in one pass:

- `multi_media` class attribute on `DatasetImporter` (always implicit
  `True` now).  Frontend `ImporterInfo.multi_media`, the gate in
  `dataset-importer-modal.component.html`, and the `multi_media` check
  in the submit handler are gone; the Include-rows editor renders
  for every form-style importer.
- `_parse_legacy_specs()` and the legacy branch of
  `effective_source_specs()`.  The helper always parses
  `source_specs` now; missing/empty falls back to a single direct
  row.
- `converters` form-field passthrough in `vtsearch/routes/datasets/load.py`
  and the `extra_keys` tuples in `vtsearch/routes/datasets/staging.py`.
- `converter_names: list[str] | None` parameter of
  `run_converters_on_folder()`; converter rows always travel as typed
  `SourceSpec`s now.
- `_LegacyTestImporter` and `TestLegacyEffectiveSourceSpecs` /
  `TestImportAPIConverters` / `TestImporterMultiMediaFlagInToDict` /
  `TestMultiMediaImportersFlag` test classes; `multi_media` parameter
  on `tests_lib/datasets/test_build_origin.py::_make_importer`.
- `multi_media` entry in `.vulture-whitelist.py`.
- Doc updates in `CLAUDE.md`, `docs/EXTENDING-plugins.md`,
  `vtscore/docs/extending/dataset-importers.md`,
  `vtscore/docs/packages/datasets.md`, and `docs/vtscore-api.md`.

We intentionally did **not** rename `media_type` → `output_type`.
The rename touches importer fields, CLI flags, frontend form labels,
every existing dataset's persisted origin params (which would silently
break `reload_from_origin()` without a migration), and the entire
doc surface (for a name that already reads correctly when there is
exactly one media-type field per importer.  The class attribute
keeps the original name; only the user-visible label inside the
multi-media editor was already "Output Media Type" (no change).

### What this breaks for external importers

Any third-party `DatasetImporter` subclass that still relied on the
`multi_media=False` semantics (i.e. that declared a `converters`
form field and expected `run_converters_on_folder()` to be called for
it) needs to migrate:

1. Drop the `multi_media = True/False` line if present (it is no
   longer read).
2. Replace the `converters` form field with the multi-media editor
   (the framework injects `source_specs` automatically).  Rewrite
   `run()` to either:
   - For folder-shaped importers, call
     `run_converters_on_folder(folder_path, target_media_type=...,
     converter_specs=runnable_specs, ...)`.
   - For service-style importers, override `fetch_source_media(spec,
     field_values, thin=False)` (per-spec fetch) or
     `fetch_all_source_media(specs, field_values, thin=False)`
     (single bulk fetch returning typed pairs), and the base-class
     `run()` will run converters and ingest for you.
3. Update `build_origin()` to record `source_specs` (the framework
   auto-adds it via `_effective_extra_origin_keys()` for declarative
   `build_origin` impls).
