# MediaCleaners: optional same-type cleanup gates before embedding

## Background

The core is shipped: `MediaCleaner` (a `MediaClipper` subclass whose `clean()`
is 1→1), its own registry, `kind: "cleaner"` chain steps, the Clean/Original
dual payload, `GET /api/cleaners`, the import-flow Cleanup checkbox list, and
the detail-viewer toggle. Metadata-only cleaning is shipped too: a gate may
clean by narrowing `clip_start` / `clip_end` / `clip_box` instead of rewriting
a payload, which is how the video gates work. The permanent documentation lives
in **`docs/EXTENDING-media.md` § Adding a Media Cleaner** (what to implement,
the `clean()` contract, the dual payload, cleaning by metadata) and
`docs/api/datasets.md` / `docs/api/medias.md` (the `cleaners` field and
`?variant=original`).

What remains is the rest of the **roster**: each entry below is a new
`MediaCleaner` subclass plus its `CLEANERS` registration and tests. Two design
points that still bear on the open work:

- **Cleaners run last, on the finished units.** Only cleaners matching the
  chain's *final* media type apply. Known cost: a letterboxed image fed to
  `ImageTilingClipper` tiles the letterbox before edge-trim can remove it. If
  that bites, add a per-cleaner `stage` property (`"pre_clip"` / `"post_clip"`,
  default post) — see Open questions.
- **Cleaners run in registration order**, with no user reordering, so every
  shipped cleaner should be order-insensitive in practice.

## Cleaner roster (open work, roughly by priority)

<!-- item-sep -->

- [ ] #5286 — Audio: loudness normalization

<!-- item-sep -->

- [ ] #5287 — Document: blank-page drop (blocked on a pre-convert stage)

<!-- item-sep -->

- [ ] #5288 — Text: boilerplate removal (deferred until there is a corpus to tune on)

<!-- item-sep -->

Deliberately deferred (revisit only with evidence): saliency-based image
cropping (can eat context the detector needs), audio noise reduction
(artifact risk).

## Open questions

- **Query-side consistency** — labeled-example replay applies cleaners via
  the chain trail, but ad-hoc example-media searches (dropping in a
  reference sound/image) don't pass through the same gates as the dataset
  did, so query and dataset vectors sit in slightly different
  distributions. Probably: run the dataset's enabled cleaners on query
  media at embed time.
- **Storage controls** — `original_*` retention doubles per-item storage
  when a cleaner mutates. Is a per-import "keep originals" toggle (default
  on) worth it, or is the mutating-items-only bound enough in practice?
  (Metadata-only gates sidestep this entirely — they store nothing.)
- **Pre-clip / pre-convert stage** — a per-cleaner `stage` property, wanted by
  the document blank-page gate and by tiling-vs-edge-trim ordering. Its design
  questions live in #5287.
