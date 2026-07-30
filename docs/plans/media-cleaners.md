# MediaCleaners: optional same-type cleanup gates before embedding

Proposed. Nothing here is implemented yet.

## Motivation

Embedders waste representational capacity on content-free regions: letterbox
bars around an image, leading/trailing silence in an audio clip, hyphenation
junk in PDF-extracted text. A **MediaCleaner** removes wasted regions from a
media item so the content embedder produces a more targeted vector.

Like a `MediaClipper`, a cleaner takes a media item of type X and returns
type X. The differences are **cardinality** and **use**:

| | Clipper | Converter | Cleaner |
|---|---|---|---|
| Type | X → X | X → Y | X → X |
| Cardinality | 1 → N | 1 → N | **1 → 1** |
| UI | pick **one** per import | routing step | **all optional gates**, independently toggleable |

A clipper breaks large media into manageable sub-items; a cleaner tightens
each item in place. The UI reflects that: clippers are a radio choice,
cleaners are a checkbox list that every item of that type passes through.

## Core design decisions

### Chain integration, not a new pipeline

The clipper chain (`vtscore/datasets/clipper_chain.py`) already runs ordered
load-time steps with per-output origin trails, content-hash disambiguation,
and deterministic cross-dataset replay. A cleaner is a chain step whose
`n_out` is always 1 — the trivial case of the existing disambiguation logic,
so label replay across datasets comes for free.

- New step kind `"cleaner"` alongside `"clipper"` / `"converter"` in
  `normalise_chain`, `validate_chain`, `apply_chain_to_clips`,
  `replay_chain_on_file`, and `parse_trail`. A distinct kind (rather than
  reusing `"clipper"`) is warranted because the runner treats cleaners
  specially: it preserves the original payload (below).
- The trail entry records the cleaner name + effective params +
  `content_hash` of the cleaned payload, so replay re-runs `clean()` and
  embeds the same bytes.

### Placement: cleaners run last, on the finished units

Enabled cleaners run **after the final clipper/converter step**, on the
units that will actually be embedded. Only cleaners matching the chain's
*final* media type apply (a document→text chain gets text cleaners, not
document cleaners). Rationale:

- The "Clean vs Original" dual payload (below) is exact per item: each
  embedded unit has a well-defined pre-clean version of *itself*. Running
  cleaners before a clipper would leave sub-clips with no meaningful
  original (slice boundaries computed on cleaned bytes don't map back).
- The stated goal is a better vector per embedded unit; tightening each
  final unit serves that directly (e.g. trimming per-tile silence).
- One simple mental model for the UI: "every imported item passes through
  these gates."

Known cost: a letterboxed image fed to `ImageTilingClipper` tiles the
letterbox before edge-trim can remove it. If that bites in practice, add a
per-cleaner `stage` property (`"pre_clip"` / `"post_clip"`, default post)
later — noted under Open questions, not in scope for v1.

Cleaners run in a fixed order (registration order) — no user reordering.
All shipped cleaners should be order-insensitive in practice.

### Dual payload: cleaned bytes embed, original bytes remain viewable

The cleaned payload becomes the **canonical** content: `media_bytes` /
`media_string`, `duration`, `file_size`, MD5, thumbnail, and embedding all
derive from it, so every existing consumer works unchanged. Additionally,
the chain runner snapshots the pre-clean payload the first time a cleaner
actually changes an item, under parallel keys:

- `original_media_bytes` / `original_media_string`
- `original_duration` (audio/video)

so users can view either the Clean or the Original version of an item.
Rules:

- The runner (not each cleaner) owns the snapshot: it compares payloads
  before/after `clean()` and stamps `original_*` only on real change. A
  no-op clean stores nothing — most cleaners no-op on most items, which
  bounds the storage cost well below a blanket 2×.
- With several cleaners in sequence, `original_*` is set once (before the
  first mutating cleaner) and never overwritten: it is the pre-*any*-clean
  payload of that unit.
- Dataset pickles persist `original_*` alongside the canonical payload —
  this is dataset content, not a cache, so it falls under the existing
  pickle exception to the no-persisted-artifacts rule. Embeddings are still
  derived only from the canonical (cleaned) payload.
- Serving: media routes stream the canonical payload by default and accept
  `?variant=original` to stream the original. The media list/detail payload
  gains a `has_original` flag; the detail viewer shows a Clean/Original
  toggle only when it is true.

### Failure convention

Same as clippers: undecodable or degenerate input returns the media
unchanged; a cleaner never aborts a load. Cleaners must also be
conservative by construction (caps like the thumbnail trimmer's
`_MAX_EDGE_TRIM` / `_MIN_EDGE_TRIM` carry over).

## ABC sketch

`vtscore/media/cleaner.py`:

```python
class MediaCleaner(MediaClipper):
    """1→1 cleanup step. Subclasses implement clean(); clip() wraps it."""

    @abstractmethod
    def clean(self, media: dict[str, Any]) -> dict[str, Any]:
        """Return a new media dict with wasted regions removed.

        Must return the media unchanged (or an equal copy) when there is
        nothing to clean or the payload can't be decoded.
        """

    def clip(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        return [self.clean(media)]
```

Subclassing `MediaClipper` reuses the whole descriptor stack —
`name`, `media_type`, `display_name`, `description`, `parameters`,
`creation_questions`, `with_params`, `to_dict` — with zero duplication.
Cleaners register in their own registry (`get_cleaner` / `list_cleaners`,
mirroring the clipper registry) so clipper choosers stay uncluttered; the
chain runner resolves `kind: "cleaner"` steps against this registry.

## UI treatment

- **Import flow**: below the clipper chooser, a "Cleanup" section for the
  dataset's final media type — one checkbox row per registered cleaner
  (display name + hover description), with a parameter disclosure like the
  clipper parameter editing. Defaults: all off, except cleaners that fix
  outright representation bugs (EXIF orientation) which default on.
- **Import row preview**: enabled cleaners render their
  `summary_template` lines alongside the clipper summary.
- **Detail viewer**: Clean/Original toggle when `has_original` is set.
- Desktop only, per repo policy.

## Cleaner roster (open work, roughly by priority)

<!-- item-sep -->

- **Core: `MediaCleaner` ABC + registry + chain integration** — the ABC,
  `kind: "cleaner"` chain support (validation, apply, replay, trail),
  dual-payload snapshot in the runner, `?variant=original` serving,
  `has_original` flag, import-flow Cleanup section, detail-viewer toggle.
  Everything below depends on this.

<!-- item-sep -->

- **Image: EXIF orientation normalize** — the embed path does *not* apply
  `exif_transpose` (only `vtscore/media/image/thumbnail.py` does), so
  embedders currently see portrait phone photos sideways while thumbnails
  show them upright. Bake the rotation into the canonical bytes. Cheapest,
  highest-confidence win; default **on**. (Near-bug-fix; could ship inside
  the Core slice.)

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

Deliberately deferred (revisit only with evidence): saliency-based image
cropping (can eat context the detector needs), audio noise reduction
(artifact risk), text boilerplate/header-footer removal (needs domain
heuristics).

## Open questions

- **Query-side consistency** — labeled-example replay applies cleaners via
  the chain trail, but ad-hoc example-media searches (dropping in a
  reference sound/image) would ideally pass through the same gates as the
  dataset did, or query and dataset vectors sit in slightly different
  distributions. Probably: run the dataset's enabled cleaners on query
  media at embed time.
- **Storage controls** — `original_*` retention doubles per-item storage
  when a cleaner mutates. Is a per-import "keep originals" toggle (default
  on) worth it, or is the mutating-items-only bound enough in practice?
- **Pre-clip stage** — whether tiling-vs-edge-trim ordering hurts enough
  to justify a per-cleaner `stage` property (see Placement above), and
  what "Original" means for a pre-clip-cleaned sub-clip if so.
