# Near-duplicate detection

**Status: Phase 1 shipped** — images + text near-dupe merge, opt-in at dataset creation, with a vectorised/threaded hashing pass. Audio/video deferred (see Open follow-ups).

Near-duplicates are items that are the *same content* under a different encoding / resize / recompression / trivial edit (distinct from *semantic* similarity, which the embeddings measure). A perceptual hash answers "these *are* the same thing". We already collapse **exact** duplicates (identical MD5) at load via `collapse_duplicates`; near-dupe adds an opt-in pass right after it.

## Open follow-ups

- **Audio near-dupe** via Chromaprint/AcoustID (or constellation hashing).
  New external dep; not in v1.
- **Video near-dupe** via TMK+PDQF or per-keyframe pHash sequences. New dep.
- **Cluster-diameter cap** if transitive drift proves a problem in practice.
  *Context:* "pHashes are close" is not transitive (A≈B, B≈C, but A≉C). v1 relies
  on a **tight** per-pair Hamming threshold + connected components, which makes
  transitive drift across a dense gradient (A~B~…~Z) rare but not impossible. The
  fix if it ever bites is a cluster-diameter cap, which breaks the clean
  equivalence-relation model — deferred until it's a real problem.
- **Banding bucket blow-up:** a band bucket of many distinct-but-band-sharing
  hashes is verified pairwise (O(m²) within the bucket). Fine for v1 sizes;
  revisit with a smarter multi-index if it bites on huge datasets.

*GPU was considered and rejected for the hashing pass:* the per-item cost is image **decode** (no torch GPU path for arbitrary JPEG/PNG) and the 32×32 DCT is trivially cheap, so a device round-trip per item would lose. The CPU vectorise/thread wins (shipped below) are the right lever.

## What shipped

Core algorithm + wiring (`vtscore/state/near_dupes.py`):
- **Images:** classic **pHash** (64-bit DCT), dependency-free via PIL + a manual numpy DCT-II (no `imagehash`/`scipy`, `deptry` stays clean); computed from `thumbnail_bytes`, media with no decodable thumbnail simply not grouped.
- **Text:** **SimHash** (64-bit) over word k-shingles hashed with `blake2b` (deterministic, not salted `hash()`); computed from `media_string`.
- **Grouping:** tight per-pair Hamming threshold (images `K=4`, text `K=4`, out of 64; not user-exposed) → banding → union-find connected components. Calibrated empirically in `tests_lib/datasets/test_near_dupes.py`.
- **Representative selection:** largest `file_size` → closest to embedding centroid → lowest media id (deterministic tie-break).
- Per-member `md5` fix in `collapse_duplicates` (members previously fell back to the representative's md5 via `_clip_to_elements`; wrong for near-dupes where members have distinct md5s). Export expansion then comes free by reusing the `dupe_set` origin structure (`expand_dupes=True` in `LabelSet.from_clips_and_votes`).
- Pipeline: `_collapse_near_duplicates_stage` (`vtscore/datasets/stages/finalize.py`) runs after `_collapse_duplicates_stage`, gated on `ctx.merge_near_duplicates`; baked into origins + dataset pickle so reloads don't recompute (flag defaults off on reload). Existing `dupe_set` member lists are merged (flattened), never nested.
- Plumbing (mirrors `build_projection`): `DatasetContext.merge_near_duplicates`, threaded through `_run_origin_load_in_background` / `_run_importer_in_background` / `vtsearch/routes/datasets/load.py` (import-local-folder/files + load-demo + generic importer route).
- Frontend: **"Merge near-duplicates"** checkbox in the importer modal's advanced section (`mergeNearDuplicates` + `import-advanced` component/HTML + all three submission paths); no threshold knob.
- Library-tier tests.

Performance:
- **Progress driven across the hashing phase** — `_find_near_dup_groups` takes an `on_progress` and ticks every `_HASH_PROGRESS_STRIDE` items (was dead-air: the callback only ticked during the cheap `_collapse_group` loop, so the bar sat at its floor through the expensive part then filled in a burst).
- **Vectorised hashing (CPU, bit-identical).** `simhash_text` shingle×bit loops → single `np.unpackbits`/vote/`np.packbits` reduction; `_pack_bits` → `np.packbits`. Verified bit-for-bit against the old loops.
- **Threaded image decode.** Past `_THREAD_MIN_IMAGES`, pHash decode+resize (GIL released) runs on a small `ThreadPoolExecutor`; order-independent grouping gives identical results (`test_threaded_and_inline_hashing_agree`).

Not addressed: the **registry/pickle reload** path (`registry.py`, "Step 2 of 2 · Removing duplicates") never runs near-dupe — its "Removing duplicates" is the cheap exact-MD5 `collapse_duplicates`; any real cost there is the diversity-tree rebuild or pickle decompress, a separate concern.
