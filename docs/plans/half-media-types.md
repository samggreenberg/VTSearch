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

`MediaType` now names the two orthogonal capabilities it used to conflate, and
every subsystem keys off the right one:

- `MediaType.importable: bool` — whether the type is a first-class *ingestion*
  category the user picks when importing (folder scan, file upload). Defaults
  `True`; a *convert-in* half type overrides it to `False`.
- `MediaType.embeddable: bool` — derived from the embedder registry
  (`embedders_for_type(type_id)`), so it is `False` for a *convert-out* half
  type and `True` for image/audio/video/text automatically, nothing to sync.
- `MediaType.converts_to: list[str]` — embeddable `type_id`s a non-embeddable
  type can convert into (first = default). `document → ["image", "text"]`;
  empty for a directly-embeddable type.

All three are in `MediaType.to_dict()` → `GET /api/media-types`, and on the
frontend `MediaTypeInfo` (`importable?`, `embeddable?`, `converts_to?`).

There are **two mirror-image half types**, split along the two axes:

- **Convert-out** (`importable && !embeddable && converts_to != []`):
  `document`. A category the user ingests but which mandates a conversion step
  before it can be searched. Generalises to a future `archive` / `email` type.
- **Convert-in** (`embeddable && !importable`): `face`. A category that is
  never imported natively — it only ever arises from converting *another* type
  (a face is cropped out of an image by `image2face`). It has its own embedder
  (FaceNet identity space) but no file extensions.

**Guiding principle:** surface the **ingestion category** wherever the user
differentiates (folder import, demo tabs — filter by `importable`); surface the
**embedded identity** only *downstream of conversion* (the browse map, sort,
detector-example pickers — filter by `embeddable`). Never render a document by
pretending its raw bytes are an image — render it *as a document*.

<!-- item-sep -->

## Open work

<!-- item-sep -->

- **Convert-in output types in the folder importer.** The importer's output
  media-type dropdown is populated from `all_folder_names()` (every registered
  type), so a *convert-in* type like `face` shows up as a native folder type
  even though it can't be scanned from files (empty `file_extensions`).
  Selecting it works — the `image2face` converter row is the real mechanism —
  but the "include face files directly" native row is a no-op. A cleaner UX
  would drive the output dropdown off `importable` (native scan types) *plus*
  convert-in targets reachable via a converter, instead of the raw registry
  list. Until then `importable` is surfaced but only filters the import-defaults
  settings and the detector-example demo tabs.

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
