# Near-duplicate detection

Status: **Phase 1 shipped** — images + text near-dupe merge, opt-in at dataset
creation. Audio/video deferred (see Open follow-ups).

## Problem

We already collapse **exact** duplicates (identical MD5) at dataset load:
`collapse_duplicates` groups medias by MD5, keeps the first as a representative,
records every member's provenance in a `dupe_set` origin, and deletes the rest.

We want to also collapse **near**-duplicates: items that are the *same content*
under a different encoding / resize / recompression / trivial edit. This is
distinct from *semantic* similarity (what the CLAP/SigLIP/X-CLIP/E5 embeddings
measure). A perceptual hash answers "these *are* the same thing"; an embedding
answers "these *mean* the same thing". Near-dupe wants the former.

## Decisions (locked with the user)

1. **Scope (v1): images + text.** Ship the full flow (option → grouping →
   representative → export) for images and text. Audio (Chromaprint) and video
   (TMK / per-frame pHash) are deferred — different algorithms, new heavy deps.

2. **Signals.**
   - **Images:** classic **pHash** (DCT perceptual hash, 64-bit). Implemented
     dependency-free with PIL (already a dep) + a manual numpy DCT-II, so no
     `imagehash`/`scipy` dependency is added and `deptry` stays clean. Computed
     from `thumbnail_bytes` (already stored per image media); media with no
     decodable thumbnail are simply not grouped (false-negative, acceptable).
   - **Text:** **SimHash** (64-bit) over word k-shingles, hashed with
     `blake2b` (deterministic — *not* Python's salted `hash()`). Computed from
     `media_string`.

   Both reduce near-dupe to "two 64-bit hashes within Hamming distance ≤ K",
   giving one uniform grouping path for both modalities.

3. **Tight threshold + union-find (the user's point #2).** "pHashes are close"
   is not transitive (A≈B, B≈C, but A≉C). We pick a **tight** per-pair Hamming
   threshold so any positive is high-confidence (missed links are false
   *negatives*, which we tolerate), then take **connected components** over the
   closeness graph as if it were an equivalence relation. Thresholds (out of
   64 bits): images `K=4`, text `K=4`. Not user-exposed — no "equalness
   points" slider.

   These were calibrated empirically (see the measurements baked into
   `tests_lib/datasets/test_near_dupes.py`). For a structured ("photo-like")
   image, JPEG recompression and resize round-trips land at Hamming **0**
   while a different image is **~32** — a huge margin, so `K=4` is very safe.
   (Smooth gradients are a pathological pHash case — their low-frequency DCT
   coefficients cluster at the median so trivial edits flip many bits — but
   real media isn't smooth.) For a ~40-word document, a whitespace/case/
   encoding reflow hashes to **0** (the near-dup exact-MD5 misses), a single
   token edit ≈**4**, a multi-word edit ≥**8**, and an unrelated document
   ≥**30**. So text near-dup mainly catches reformatted/re-encoded copies and
   incidental single-token differences; larger edits are intentional
   false-negatives.

   *Caveat (logged, accepted for v1):* even a tight per-pair threshold doesn't
   fully kill transitive drift across a dense gradient (A~B~…~Z). It makes it
   rare rather than impossible. The fix if it ever bites is a cluster-diameter
   cap, which breaks the clean equivalence-relation model — not now.

4. **Representative selection (the user's point #3): `file_size` → centroid →
   id.** Within a group, prefer the largest `file_size` (a universal
   fidelity proxy that works for every media type, unlike "resolution"), then
   the item closest to the group's embedding centroid, then the lowest media id
   (deterministic tie-break — required by the test-suite flakiness rules).

5. **Optional at dataset creation (the user's point #4).** A single **"Merge
   near-duplicates"** checkbox in the importer modal's advanced section — no
   threshold knob. Exact MD5 dedup still runs unconditionally; near-dupe merge
   is the opt-in extra pass that runs immediately after it.

6. **Export the whole set (the user's point #5).** Exact-dupe export already
   expands a `dupe_set` representative into one `LabeledElement` per member
   (`expand_dupes=True` by default in `LabelSet.from_clips_and_votes`). By
   reusing the **same `dupe_set` origin structure** for near-dupes, export
   expansion comes for free. **Bug fixed along the way:** `collapse_duplicates`
   built member dicts *without* an `md5`, and `_clip_to_elements` did
   `m.get("md5", media["md5"])` — falling back to the *representative's* md5.
   Harmless for exact dupes (all members share one md5) but **wrong for
   near-dupes** (members have different md5s → exports/re-imports under the
   wrong hash). Members now carry their own `md5`.

## Where it runs

`vtscore/datasets/stages/finalize.py` gains `_collapse_near_duplicates_stage`,
called right after `_collapse_duplicates_stage` in the load pipeline, gated on
`ctx.merge_near_duplicates`. Because the result is baked into origins and saved
in the dataset pickle, reloads don't recompute it (the flag defaults off on the
reload path) — same lifecycle as exact dedup.

Near-dupe runs *after* exact dedup, so some group members are already `dupe_set`
representatives. When collapsing a near-dupe group, existing `dupe_set` members
lists are **merged** (flattened), never nested.

## Plumbing (mirrors `build_projection`)

- `DatasetContext.merge_near_duplicates` (transient slot, default `False`).
- `_run_origin_load_in_background(..., merge_near_duplicates=False)` sets it on
  the context before finalize.
- `_run_importer_in_background` pops `merge_near_duplicates` from `field_values`.
- `vtsearch/routes/datasets/load.py` reads it on the import-local-folder,
  import-local-files, and load-demo routes; the generic importer route carries
  it through `field_values`.
- Frontend: `mergeNearDuplicates` on the importer modal + `import-advanced`
  component/HTML checkbox + all three submission paths.

## What shipped

- `vtscore/state/near_dupes.py`: pHash, SimHash, banding + union-find,
  representative selection, `collapse_near_duplicates`.
- Per-member `md5` fix in `collapse_duplicates`.
- Pipeline + route + frontend wiring for the opt-in checkbox.
- Library-tier tests.

## Open follow-ups

- **Audio near-dupe** via Chromaprint/AcoustID (or constellation hashing).
  New external dep; not in v1.
- **Video near-dupe** via TMK+PDQF or per-keyframe pHash sequences. New dep.
- **Cluster-diameter cap** if transitive drift proves a problem in practice
  (see Decision 3 caveat).
- **Banding bucket blow-up:** a band bucket of many distinct-but-band-sharing
  hashes is verified pairwise (O(m²) within the bucket). Fine for v1 sizes;
  revisit with a smarter multi-index if it bites on huge datasets.
