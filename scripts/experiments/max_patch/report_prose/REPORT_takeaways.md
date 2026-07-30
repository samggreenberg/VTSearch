## Take-aways

- **Region scoring is the big win; the tree is not.** The largest gap in the
  study is region-vote scoring vs whole-image scoring — DINOv3's global CLS
  vector is the *worst* arm, below even SigLIP, while the same embedder scored
  over its patches is the best. Once you score over patches, *how* you organise
  them (raw, k-means-pooled tree, or raw-patch tree) matters far less than the
  fact that you do.
- **Tree-free MaxPatch is the sweet spot.** With the geometry fix, MaxPatch
  scores over the whole-image vector *plus* every raw patch — so it already has
  a candidate at both ends of the scale range (a single patch for a small
  object, the full-image vector for a whole-scene one). That span is enough to
  make it the best arm at every scale, and it means the multi-scale *middle* a
  tree adds is not where the value is.
- **Neither tree beats tree-free scoring.** The production k-means tree (MaxHAC)
  loses to plain MaxPatch, and the raw-patch-leaf tree (MaxPatchHAC) only
  *numerically* edges MaxHAC (a trend, not significant) while still costing more
  than MaxPatch. A raw-patch-leaf tree does *rank* better than everything — so
  if a tree must exist, leaf it with raw patches, not k-means pools — but on the
  operating point the cleaner move is to delete the tree from ingest entirely.
- **More candidates is not free.** MaxPatchHAC's ~392-node pool does what it was
  designed to on large objects — its merged nodes improve large-object *recall*
  over pure raw patches — but the larger the pool, the heavier the tail of the
  max-over-N score, so it also raises *false positives*. The two cancel on large
  objects and the extra nodes are pure cost on mid-scale ones. Adding scale
  candidates helps recall and hurts precision; pick the pool size deliberately.
- **The tree's *merge order* barely matters; the candidate *set* does.** Denoising the merge order with per-image PCA (MaxPatchPcaHAC) changes the tree topology but leaves the outcome statistically unchanged — an image's score is a max-pool over *every* node, so it is insensitive to which patches merge in what order. The lever is the candidate set you pool over (raw patches + the whole-image row), not the structure of the tree built on it.
- **The scale crossover is real but already covered.** Raw patches beat pooled
  regions on small objects and the two converge on large ones (ρ = 0.50 between
  object size and the MaxPatch−MaxHAC gap) — the pre-registered hypothesis. But
  the whole-image row now inside MaxPatch covers the large end without a tree, so
  the crossover is a reason MaxPatch wins, not a reason to build multi-scale
  nodes.
- **Harness hygiene changed the answer.** The first run concluded MaxPatch
  "mis-calibrates on easy content"; that was a defect — a boxless Good vote
  trained on a vector the scorer never evaluated, and calibration bags collapsed
  in a geometry inference never used. With train/score geometry parity and
  calibration in inference geometry, MaxPatch is simply the best region-vote
  strategy. Worth remembering before trusting an operating-point result: check
  that every vector a vote trains on is a vector the scorer also scores.
