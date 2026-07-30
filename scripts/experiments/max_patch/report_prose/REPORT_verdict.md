## Verdict

**Ship tree-free MaxPatch as the DINOv3 region-vote strategy, and drop the HAC
tree from ingest. The tree does not earn its keep — the production k-means tree
(MaxHAC) loses to plain MaxPatch, and the new raw-patch-leaf tree (MaxPatchHAC)
does not beat MaxPatch either.** On the corrected harness, over 23 scale-band
Visual Genome categories × 3 seeds:

- **MaxPatch is the best arm** — ErrorCost **0.40** at t = 150, vs MaxHAC 0.46
  (paired Δ = −0.064, Holm p = 0.002) and MaxPatchHAC 0.44. It is the best or
  tied-best region style in **every** scale band and wins on both halves of the
  error (FPR 0.089, FNR 0.312).
- **The scale hypothesis holds for MaxPatch vs the production tree.** MaxPatch's
  edge over MaxHAC is largest on small objects and shrinks as objects grow
  (Spearman ρ = 0.50 between voted-box area and the MaxPatch−MaxHAC gap,
  p = 0.016): below leaf scale a raw patch is a near-pure object sample while the
  tree's smallest k-means-pooled leaf already blends the object with context; by
  whole-scene scale MaxHAC's pooled region catches up.
- **MaxPatchHAC lands between the two but beats neither convincingly.** It costs
  more than plain MaxPatch (Δ = +0.037, not significant) and only *numerically*
  edges the production MaxHAC (Δ = −0.027, Holm p = 0.064 — a trend, not
  significant). The one thing it clearly does best is **rank**: it has the
  highest average precision of any arm (AP 0.492 vs MaxPatch's 0.486). But its
  ~392-node multi-scale pool trades that ranking away at the operating point —
  it carries the **highest FPR** of the region styles (0.104 vs MaxPatch's
  0.089), because the max-over-N score has a heavier tail the more candidates N
  it pools. The result is a genuine double edge, visible per band: on large
  objects the merged nodes **improve recall** (FNR 0.21 vs MaxPatch's 0.26 in
  the above-4×-leaf band — the multi-scale idea working) but **over-fire** (FPR
  0.16 vs 0.11), netting to a tie; on the mid `patch_to_leaf` band — where a
  single raw patch is already the right candidate — the extra nodes are pure
  cost (worse FPR *and* FNR); on small objects it simply matches MaxPatch.
- **PCA-denoising the merge order is a no-op.** MaxPatchPcaHAC — MaxPatchHAC with the raw-patch merge *order* decided in a per-image 32-component PCA space (the `pca_dims` option ported from the HAC-tree-improvements branch) — is statistically indistinguishable from MaxPatchHAC (ErrorCost 0.420 vs 0.436, paired Δ = −0.015, Holm p = 1.0; AULC Δ = −0.001, p = 0.48). The tree topology genuinely changes, but an image's score is a max-pool over *every* node, so it is blind to which patches merged in what order. Like MaxPatchHAC, it does not beat plain MaxPatch (AULC Δ = +0.035 vs MaxPatch, p = 0.008).
- **Why the hybrid doesn't win:** the geometry fix already gave plain MaxPatch a
  whole-image row in its scored pool, so MaxPatch **already spans scales** — a
  single-patch candidate for small objects and the full-image vector for
  whole-scene ones. MaxPatchHAC's intermediate merged nodes add large-object
  recall but pay for it in false positives, and add nothing a raw patch or the
  whole-image row didn't already cover on small and mid objects.
- **Region scoring still matters more than any of this.** All three region
  styles crush whole-image scoring — DINOv3's global CLS vector (0.61) is the
  worst arm, below the SigLIP baseline (each region style beats CLS at
  p < 0.001) — so the win is region-vote scoring itself, not the choice of
  embedder or pooling.

### Plans for moving forward

1. **Adopt MaxPatch for DINOv3 region-vote scoring** (nearest-patch Good vote,
   whole-image + all-patch Bad flood, max-pool over the whole-image row + raw
   patches) and **remove the HAC tree build** (k-means leaves + O(k³) merges +
   ~24 stored region vectors per image) from the default ingest path. Expect a
   modest per-retrain scoring-cost increase (max-pooling ~197 rows vs ~24 nodes;
   milliseconds at session sizes — measure on the largest collections first).
2. **The multi-scale idea is not dead — its threshold is the problem.**
   MaxPatchHAC ranks best of all arms and recovers real large-object recall; it
   loses only at the operating point, to the many-node false-positive tail. If
   large-object recall is worth chasing, keep the raw-patch tree but pair it with
   a **max-pool-aware calibration** or a **softer pool** (top-k / log-sum-exp
   instead of hard max) to tame the tail — a ranking that good should be
   convertible into a better operating point. Absent that, plain MaxPatch is the
   simpler and better default.
3. **Follow-ups the data motivates:** a second boxed dataset (especially a
   small-object one like OpenLogo, once fetchable) to test generality; a
   rare-prevalence arm for the recall angle; and the `mean-of-patches-in-box`
   Good-vote variant the plan lists.
