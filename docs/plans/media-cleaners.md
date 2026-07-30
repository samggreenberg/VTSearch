# MediaCleaners: optional same-type cleanup gates before embedding

## Background

The core is shipped: `MediaCleaner` (a `MediaClipper` subclass whose `clean()`
is 1→1), its own registry, `kind: "cleaner"` chain steps, the Clean/Original
dual payload, `GET /api/cleaners`, the import-flow Cleanup checkbox list, and
the detail-viewer toggle. The permanent documentation lives in
**`docs/EXTENDING-media.md` § Adding a Media Cleaner** (what to implement, the
`clean()` contract, the dual payload) and `docs/api/datasets.md` /
`docs/api/medias.md` (the `cleaners` field and `?variant=original`).

What remains is the **roster**: each entry below is a new `MediaCleaner`
subclass plus its `CLEANERS` registration and tests. Two design points that
still bear on the open work:

- **Cleaners run last, on the finished units.** Only cleaners matching the
  chain's *final* media type apply. Known cost: a letterboxed image fed to
  `ImageTilingClipper` tiles the letterbox before edge-trim can remove it. If
  that bites, add a per-cleaner `stage` property (`"pre_clip"` / `"post_clip"`,
  default post) — see Open questions.
- **Cleaners run in registration order**, with no user reordering, so every
  shipped cleaner should be order-insensitive in practice.

## Cleaner roster (open work, roughly by priority)

<!-- item-sep -->

- **Image: solid-border trim** — crop near-solid white/black margins
  (letterbox, pillarbox, whitespace around logos). Promote
  `_trim_solid_edges` out of `thumbnail.py` into shared code with one
  implementation and two callers; the tuned caps (`_EDGE_TOL`,
  `_MAX_EDGE_TRIM`, `_MIN_EDGE_TRIM`) carry over as parameters.

<!-- item-sep -->

- **Audio: leading/trailing silence trim** — keep
  `[first_start, last_end]` of the non-silent intervals detected by the
  `SoundSilenceClipper._detect_segments` machinery (share it, don't copy).
  Same `top_db` / `pad` parameters. Redundant-but-harmless when combined
  with the silence clipper itself.

<!-- item-sep -->

- **Text: whitespace + de-hyphenation cleanup** — collapse whitespace
  runs, strip control characters, re-join words hyphen-broken across line
  breaks. Highest value on `document2text` output (PDF extraction junk).

<!-- item-sep -->

- **Text: markup strip** — remove HTML tags / markdown syntax so the
  embedder sees prose, not angle brackets.

<!-- item-sep -->

- **Video: letterbox bar crop** — run the edge-trim analysis on a few
  sampled frames and crop the consensus box from all frames.

<!-- item-sep -->

- **Video: leading/trailing black-frame trim** — the silence-trim analog
  (fade-ins, slates, tail cards).

<!-- item-sep -->

- **Audio: loudness normalization** — peak or RMS normalize; quiet
  recordings embed worse with CLAP. Moderate value, low risk.

<!-- item-sep -->

- **Document: blank-page drop** — still 1→1 (document in, thinner
  document out); helps everything downstream of `document2*`.

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
  file.

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
- **Pre-clip stage** — whether tiling-vs-edge-trim ordering hurts enough
  to justify a per-cleaner `stage` property (see Background above), and
  what "Original" means for a pre-clip-cleaned sub-clip if so.
