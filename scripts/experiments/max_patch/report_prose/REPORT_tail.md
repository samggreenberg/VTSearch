## Limitations (accepted for this run)

- **One dataset.** The corrected study is Visual Genome only. Caltech-101 was
  removed on purpose (boxless → it cannot exercise *region* voting), and
  OpenLogo — the extreme-small-logo regime — could not be fetched (the cluster's
  shared egress repeatedly failed on the 27k-file HF dataset). VG's scale-band
  categories still span the sub-patch → whole-scene range the crossover lives
  in, but a second boxed dataset (especially a small-object one) would test
  generality.
- **Natural prevalence only.** No 1 %-rare arm; the question here is object
  *scale*, not rarity. A rare-prevalence arm is the obvious follow-up for the
  rare-event recall angle.
- **MLP classifier only.** Every arm uses the production MLP; only the
  vote/score geometry differs.
- **Uniform-mean internal nodes for MaxPatchHAC.** The experiment carries no
  per-patch saliency, so MaxPatchHAC's internal-node vectors are the plain
  (uniform) L2-normalised mean of their member patches, where production's HAC
  pools are saliency-weighted. The tree structure and merge order (blended
  cosine + spatial, average linkage) are otherwise faithful.
- **Acquisition proxy is style-blind.** Autopilot ranks pool candidates by their
  whole-image vector under every style; only training and test scoring differ
  per style, so vote-order differences are attributable to the trained model,
  not a different acquisition rule.

## Reproducibility

All code is on branch `claude/max-patch-hac` (built on the corrected harness, PR
#2732). The report, figures, `metrics.json`, and `prepare_info.json` are
committed under `docs/experiments/max-patch/`; the full per-cell CSVs (too large
for the repo's file-size hook) and the cached embedding pickles stay on the Grid
under `/exp/$USER/max-patch/{results/cells,datadir/embeddings}`.

```bash
# 0. one GPU node, worktree env sourced (scripts/experiments/max_patch/).
#    prepare embeds VG once and selects scale-band categories by median voted box.
MAXPATCH_EXP=/exp/$USER/max-patch \
MAXPATCH_DATASETS=visual_genome_m \
MAXPATCH_EMBEDDERS=dinov3_patch,siglip  python prepare_data.py

# 1. the voting array (one SLURM task per dataset×embedder×category×seed; all
#    styles for a cell run inside it — max_hac / max_patch / max_patch_hac /
#    whole_image on DINOv3, whole_image on SigLIP).
MAXPATCH_N_SEEDS=5 bash launch_cells.sh

# 2. the report (per-dataset paired Wilcoxon, bootstrap-CI curves, the
#    voted-box scale scatter, captioned figures, and REPORT.md).
python analyze.py
```

### MaxPatchHAC in one paragraph

`build_patch_hac_tree` (`vtscore/eval/patch_styles.py`) takes the H×W raw patch
grid, makes every patch a leaf, and agglomeratively merges them (blended
cosine + spatial distance, average linkage) into a binary tree — `2·H·W − 1`
nodes plus a CLS whole-image node at index 0, ~392 for DINOv3's 14×14 grid.
`MaxPatchHacStyle` scores an image by max-pooling over every node, snaps a Good
region-vote to the node whose box best matches (multi-scale), and floods **every
node** on a Bad vote (symmetric with inference). Because the tree carries the CLS
node and the flood covers every scored row, it satisfies the train/score
geometry parity the corrected harness enforces — verified by the shared parity
tests plus a dedicated `TestMaxPatchHacStyle` (56 tests pass).
