# Half media types (document, and the general model)

**Background.** `MediaType` bundles two *orthogonal* capabilities that used to
be conflated:

1. **Ingestion identity** — a distinct kind of file the user recognises and
   picks when importing. `.pdf` is not `.png`; a folder scan tallies them
   separately and the user wants to differentiate and decide. **Every** type
   has this.
2. **Embeddability** — whether the type can be turned into a vector and
   therefore sorted / browsed / text-queried on its own. A type has this iff it
   registers an `embedder_*.py`.

`document` is the canonical **half type**: a first-class *ingestion* category
with **no embedder**. It must be converted to an embeddable type (image via
`document2image`, or text via `document2text`) before it is searchable. The
clipper-chain (`vtscore/datasets/clipper_chain.py`) already runs converter
steps at load time, so `document → image/text → embed` is a supported
pipeline — what was missing is that the *abstraction never named the gap*, so
each subsystem improvised. The two issues below are that improvisation showing
through.

## The model (shipped)

`MediaType` now names both capabilities explicitly, and every subsystem keys
off the right one:

- `MediaType.converts_to: list[str]` — embeddable `type_id`s a non-embeddable
  type can convert into (first = default). `document → ["image", "text"]`;
  empty for a directly-embeddable type.
- `MediaType.embeddable: bool` — derived from the embedder registry
  (`embedders_for_type(type_id)`), so it is `False` for a half type and `True`
  for image/audio/video/text automatically, with nothing to keep in sync.

Both are in `MediaType.to_dict()` → `GET /api/media-types`, and on the frontend
`MediaTypeInfo` (`embeddable?`, `converts_to?`). A "half type" is precisely
`embeddable == False and converts_to != []` — *a category that mandates a
conversion step*. The definition generalises past `document` (a future
`archive` / `email` type would fit the same mold).

**Guiding principle:** surface the **ingestion category** wherever the user
differentiates (folder import, demo tabs); surface the **embedded identity**
only *downstream of conversion* (the browse map, sort). Never render a
document by pretending its raw bytes are an image — render it *as a document*.

<!-- item-sep -->

## Open work

- **#2358 — full Document demo tab (convert-on-load).** Interim shipped: the
  UCSF demo is relabelled so its document→image provenance is explicit while it
  stays in the Image list. The structural finish is to move it to a real
  **Document** demo tab whose load applies a converter:

  - Move the `ucsf_documents_a` `DemoDataset` from `image/_demo_sources.py` to
    `DocumentMediaType.demo_datasets`, and implement
    `DocumentMediaType.load_demo_source("ucsf_documents", …)` to produce
    *document* (raw-PDF) clips (download via `download_ucsf_documents`).
  - Teach the demo picker (`…/pickers/demo/demo-picker.component.ts`) to handle
    a non-embeddable active tab: read `converts_to[0]`, load embedders for that
    *target* type, and pass `converter=document2image` (default; offer
    `document2text`) on load. Today `loadDemoEmbedders(mediaType)` assumes the
    tab type is embeddable and returns `[]` for `document`.
  - Reconcile demo-status caching: the converter changes the pickle cache key
    (`{dataset}__{converter}`), but `getDemoList(embedder, clipper)` does not
    pass a converter, so per-demo `ready / needs_embedding / needs_download`
    status must learn the converter dimension. Update
    `vtscore/datasets/demo_counts.py` if the entry's advertised count moves.

  The `document` `image_response` hook (below) already gives these document
  clips a real thumbnail/preview, so a Document tab renders correctly.

<!-- item-sep -->

- **Import flow: require a converter for a non-embeddable folder.** When a
  folder import resolves to `document` (or any `embeddable == False` type),
  the import config should *require* the user pick a `converts_to` target
  (Image pages / Extracted text) instead of silently importing unembeddable
  media that can't be sorted or browsed. The clipper-chain already supports the
  converter step; this is the UX that makes the mandatory step visible.
  `import-config.component.ts` reads the served `converts_to`/`embeddable` to
  drive it.

<!-- item-sep -->

- **Document preview: page navigation.** The bin-popup large preview now renders
  the document's **first** page (via `image_response`). A follow-up could let
  the preview paginate (render page N on demand) — `render_pdf_page_png` already
  takes a `page_index`; the route/frontend would need a `?page=` parameter.

<!-- item-sep -->

