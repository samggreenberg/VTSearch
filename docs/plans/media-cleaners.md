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

- **Audio: loudness normalization** — peak or RMS normalize; quiet
  recordings embed worse with CLAP. Moderate value, low risk.

<!-- item-sep -->

- **Document: blank-page drop** — *blocked on a pre-convert stage.* The
  cleaner itself is easy (document in, thinner document out, via PyMuPDF),
  but it can never run as written: cleaners run last, on the chain's final
  media type, and a `document` dataset is not embeddable (`converts_to =
  ["image", "text"]`), so a document-typed gate behind a `document2*`
  converter fails `validate_chain`'s type check. It needs the `stage`
  property from Open questions — a `"pre_convert"` gate running on the
  document *before* the converter — so ship that first or drop the item.

<!-- item-sep -->

- **Text: boilerplate removal** *(future idea — good, but deferred)* —
  strip structured boilerplate that drowns out content: email headers
  (`Received:` / `From:` / `Subject:` blocks, quoted-reply chains,
  signatures), page headers/footers, navigation chrome. Worth doing once
  there's a corpus to tune against; needs per-domain heuristics, so it
  shouldn't gate the rest of the roster.

<!-- item-sep -->

Deliberately deferred (revisit only with evidence): saliency-based image
cropping (can eat context the detector needs), audio noise reduction
(artifact risk).

## Open work beyond the roster

<!-- item-sep -->

- **Persist the cleanup selection as an import default** — the embedder /
  clipper / source-spec pickers all remember the user's per-media-type choice
  via `ImportDefaultsService` (`snapshotImportConfig` →
  `maybeOfferSaveImportDefaults`), but `defaultCleanerSelection` deliberately
  ignores saved defaults and always seeds from each cleaner's
  `default_enabled`. Add `cleaners` to `ImportDefaultsForMediaType` and to the
  save-offer snapshot/equality checks so a user who always disables a gate
  isn't re-ticking it on every import.

<!-- item-sep -->

- **Thin imports lose their byte-savings on cleaned items** —
  `_relazify_reference_clips_stage` skips any item carrying an `original_*`
  snapshot, because `vtscore/media/lazy_clip.py` has no recipe that reproduces
  a cleaner's output from the source file; dropping the bytes would silently
  serve and re-embed the *uncleaned* original. So a reference (thin) import
  materializes both payloads for exactly the items some cleaner changed. Fix
  by teaching `lazy_clip` a cleaner recipe (replay the chain's cleaner steps on
  the source, keyed off the trail) — at which point a cleaned reference item
  can go back to storing only a recipe, and its "Original" is just the source
  file. (Metadata-only cleans are already exempt: they snapshot nothing, so a
  video-cleaned reference item stays lazy.)

<!-- item-sep -->

- **A video unit's crop box is invisible in the player** — `clip_box` reaches
  the frontend on the media payload and the server-rendered grid thumbnail is
  cropped to it, but `<video>` plays the uncropped file, so a letterbox-cropped
  item previews cropped and plays padded. Either crop the player's viewport
  with a CSS transform keyed off `clip_box`, or accept the divergence and say
  so in the UI (the trim half already shows up for free, since the player loops
  within `[clip_start, clip_end]`).

<!-- item-sep -->

- **Video labeled-example replay ignores the unit's window and box** —
  `replay_chain_on_file` writes the final unit to a tempfile and calls
  `embed_file`, which for video embeds the *whole* container: `clip_start` /
  `clip_end` / `clip_box` are dropped, so a re-embedded video label lands in a
  slightly different distribution from the dataset item it was taken from.
  This predates the cleaners (video tiles already replayed as whole files), but
  the metadata gates widen it from "the wrong slice" to "the wrong slice at the
  wrong framing". Fix by threading the media dict — not a bare path — into the
  video embed path so the sampler sees the window and the box.

<!-- item-sep -->

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
- **Pre-clip / pre-convert stage** — a per-cleaner `stage` property is now
  wanted by two items: tiling-vs-edge-trim ordering (does a tiled letterbox
  hurt enough to justify it?) and the document blank-page gate, which is
  blocked without a `"pre_convert"` placement. Open sub-questions: what
  "Original" means for a pre-clip-cleaned sub-clip, and whether pre-convert
  and pre-clip are one stage or two.
