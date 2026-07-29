## Verdict

**MaxPatch is the better region-vote strategy for the regime region votes exist
for — cluttered scenes with small, boxed objects — but it is not a safe blanket
replacement for MaxHAC, because its raw-patch max-pool mis-calibrates on easy,
centred, boxless content. Adopt MaxPatch for region-vote scoring on cluttered
collections (which lets the HAC tree be dropped there); keep MaxHAC / whole-image
as the default elsewhere, or gate raw-patch scoring on the presence of a real
sub-image region vote.** The two datasets give opposite answers, and that split
*is* the result — so the numbers are reported per dataset, never pooled.

**On Visual Genome (boxed, cluttered — the target regime):** MaxPatch wins
cleanly.

- ErrorCost 0.387 vs MaxHAC 0.489 at t = 150 (paired Δ = −0.102, Holm
  p < 0.001), and it wins on *both* halves of the error — FPR 0.200 vs 0.237 and
  FNR 0.188 vs 0.252 — as well as on threshold-free ranking (AP 0.498 vs 0.441).
- Both patch strategies dominate whole-image scoring. DINOv3's global **CLS**
  vector is the *worst* arm on Visual Genome (0.617), below even **SigLIP**
  (0.497) — the region machinery is what turns DINOv3 into a strong detector on
  cluttered scenes.
- The edge is **scale-driven**: sorting categories by object size (Figure 4),
  MaxPatch's advantage concentrates on the small, sub-leaf-scale categories and
  fades toward the large-region categories where the tree finally has a
  well-matched candidate (Spearman ρ = 0.57 between log-area and the
  MaxPatch−MaxHAC gap). This is the pre-registered hypothesis, confirmed on real
  annotation scales.
- **Timing:** MaxPatch and MaxHAC are statistically tied through the first ~50
  votes (cost@50 Δ = −0.040, p ≈ 0.3) and MaxPatch pulls decisively ahead as
  votes accumulate — flooding ~196 raw patches as negatives per Bad vote gives
  the classifier a denser negative manifold that compounds with evidence.

**On Caltech-101 (boxless, centred — the control):** MaxPatch *fails*, and the
failure is diagnostic.

- Every arm *ranks* the easy categories perfectly (AP = 1.000). MaxHAC,
  whole-image, and SigLIP also *threshold* cleanly (ErrorCost 0.030, 0.047,
  0.031). MaxPatch scores 0.686 — but with FPR = 0.000 and **FNR = 0.686**:
  perfect ranking, a broken operating point. Max-pooling over 196 raw patches
  compresses positive and negative scores together near the top, so the
  cross-calibrated threshold can only keep FPR at zero by rejecting most
  positives.
- The lesson: MaxPatch's weakness on easy content is **calibration, not
  ranking**. MaxHAC's smoothed 24-node region pool never shows this; it matches
  the whole-image control on easy data (no tax) and beats it on clutter.

### Plans for moving forward

1. **Adopt MaxPatch for region-vote scoring on cluttered / small-object
   collections**, where it beats MaxHAC on ranking *and* operating cost; there,
   the k-means-leaves + O(k³)-merges + ~24-vector-per-image HAC build can be
   removed from ingest.
2. **Do not blanket-replace whole-image scoring.** Either (a) gate raw-patch
   max-pool on a genuine sub-image region vote (fall back to pooled / whole-image
   when the voted box is near image scale), or (b) fix the threshold calibration
   for the compressed max-pool score distribution before making MaxPatch
   universal — the Caltech failure is an operating-point bug, not a ranking one,
   so a max-pool-aware calibration may recover it.
3. **Validate the runtime trade** on the largest collections (MaxPatch scores
   ~8× more rows per retrain — milliseconds at session sizes, but measure), and
   run the motivated follow-ups: a rare-prevalence (1 %) arm to confirm the
   recall win for rare-event search, and the OpenLogo / extreme-small-object
   regime once the dataset can be fetched.
