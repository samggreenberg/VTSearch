# Half media types (document, and the general model)

**Background.** `MediaType` names two orthogonal capabilities: `importable`
(a first-class ingestion category the user picks when importing) and
`embeddable` (derived from the embedder registry), plus `converts_to` (the
embeddable types a non-embeddable one converts into, first = default). That
gives two mirror-image **half types**: `document` is *convert-out* (importable,
no embedder; `converts_to = ["image", "text"]`) and `face` is *convert-in*
(embeddable, never imported natively; it only arises from `image2face`). The
model, its fields and the API surface are documented in
[`vtscore/docs/extending/media-types.md`](../../vtscore/docs/extending/media-types.md)
and [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md).

**Guiding principle for the open work:** surface the **ingestion category**
wherever the user differentiates (folder import, demo tabs — filter by
`importable`); surface the **embedded identity** only *downstream of
conversion* (the browse map, sort, detector-example pickers — filter by
`embeddable`). Never render a document by pretending its raw bytes are an
image — render it *as a document*.

<!-- item-sep -->

## Open work

<!-- item-sep -->

- **Convert-in output types in the folder importer.** The Add Dataset dialog's
  Multi-Dataset mode (#4703, `multi-output.util.ts`) already keys its rows off
  `importable` / `embeddable` / `converts_to`; this item and the next are owed
  only for the single-dataset form. There the folder importer's output
  media-type dropdown is populated from `all_folder_names()` (every registered
  type), so a *convert-in* type like `face` shows up as a native folder type
  even though it can't be scanned from files (empty `file_extensions`).
  Selecting it works — the `image2face` converter row is the real mechanism —
  but the "include face files directly" native row is a no-op. A cleaner UX
  would drive the output dropdown off `importable` (native scan types) *plus*
  convert-in targets reachable via a converter, instead of the raw registry
  list. Outside Multi-Dataset mode the frontend reads `importable` only to mark
  a convert-in dataset (`MediaTypeCapabilityService.isConvertIn`, for Show in
  photo).

<!-- item-sep -->

- **Import flow: require a converter for a non-embeddable folder.** When a
  folder import resolves to `document` (or any `embeddable == False` type),
  the import config should *require* the user pick a `converts_to` target
  (Image pages / Extracted text) instead of silently importing unembeddable
  media that can't be sorted or browsed. The clipper-chain already supports the
  converter step; this is the UX that makes the mandatory step visible.
  `import-config.component.ts` would read the served `converts_to` /
  `embeddable` to drive it (it reads neither today).

<!-- item-sep -->

- **Document preview: page navigation.** The bin-popup large preview now renders
  the document's **first** page (via `image_response`). A follow-up could let
  the preview paginate (render page N on demand) — `render_pdf_page_png` already
  takes a `page_index`; the route/frontend would need a `?page=` parameter.

<!-- item-sep -->
