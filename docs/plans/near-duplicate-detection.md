# Near-duplicate detection

**Status:** Audio and video near-dupe detection remain deferred, plus the smaller follow-ups below; images + text near-dupe merge already ships (opt-in at dataset creation, vectorised/threaded hashing pass).

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

*GPU was considered and rejected for the hashing pass:* the per-item cost is image **decode** (no torch GPU path for arbitrary JPEG/PNG) and the 32×32 DCT is trivially cheap, so a device round-trip per item would lose. The CPU vectorise/thread path is the right lever.
