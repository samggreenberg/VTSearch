# Max-Patch — MaxHAC vs MaxPatch vs whole-image (runner)

Code that runs the study designed in
[`docs/plans/max-patch-experiment.md`](../../../docs/plans/max-patch-experiment.md)
on the HLTCOE Grid and generates the report.  Image-only.

## The arms

Every arm is an `(embedder, style)` pair; styles are implemented in
`vtscore/eval/patch_styles.py` and threaded through the Autopilot voting
simulation via `simulate_voting_iterations(style=...)`.

| Embedder | Styles |
|---|---|
| `dinov2_patch` | `max_patch`, `max_patch_hac`, `max_patch_pca_hac`, `whole_image` (CLS control) |
| `dinov3_patch` | same |
| `siglip` | `whole_image` (standard baseline) |

- **max_patch** — the production pipeline (adopted in #2886): Good region-votes
  train on the *single raw patch* nearest the voted box, Bad votes flood the
  image-level vector + *every* raw patch (bag-weighted so a rejected image still
  counts once), images score by max-pooling the MLP over that same stack.
- **max_patch_hac** / **max_patch_pca_hac** — raw-patch-leaf HAC trees: snap a
  Good vote to the best-matching node, flood / max-pool every node. The PCA
  variant only changes the merge *ordering*.
- **whole_image** — single global vector for votes and scores.

> **`max_hac` is no longer runnable.** The original study's production arm
> (K-means-pooled HAC leaves, snap-to-node Good votes, CLS+leaf floods)
> delegated to production code that #2886 deleted when it adopted MaxPatch, so
> the arm was dropped from the grid rather than reimplemented. Its published
> numbers are in `docs/experiments/2026-07-29-max-patch/REPORT.md`, and `analyze.py` still
> labels `max_hac` rows found in archived result CSVs.

**Startup sort**: each cell's exemplar is a cropped positive (ground-truth box,
pre-embedded at prepare time); its full-image embedding is scored against the
dataset *in each style's own geometry* (whole-image cosine / max over patches /
max over tree nodes) and the Autopilot seed phase votes down that ranking.

## What each stage does

| Stage | Script | Output |
|---|---|---|
| 0 · prepare | `prepare_data.py` | Per-(dataset, embedder) pickles under `$VTSEARCH_DATA_DIR/embeddings/<ds>__<emb>.pkl` (with `patch_grid` for the DINOs), exemplar-crop vectors under `results/crops/`, and `prepare_info.json` (counts, selected categories, embed timings). |
| 1 · cells | `run_cells.py` | `results/cells/task_<i>.csv` — one SLURM-array task per `(dataset, embedder, category, seed)`, all styles inside. Per-step cost/FPR/FNR/AUROC/AP + train/score timings. |
| 2 · report | `summarize.py` | `results/REPORT.md` + `results/figures/` (deterministic from the CSVs). |

## Run it

```bash
export HF_TOKEN=hf_...   # required for the gated DINOv3 weights

# One-shot dependency chain:
bash queue_all.sh 240    # 240 = safe upper bound on array cells

# Or by hand, sized exactly:
sbatch ... --wrap "source ../../../gridenv.sh && cd $PWD && python prepare_data.py"
N=$(python run_cells.py --print-cells)      # after prepare
sbatch --array=0-$((N-1))%24 ... --wrap "... python run_cells.py"
python summarize.py
```

Default grid on a **boxed** dataset: 4 scale bands × 6 categories = 24
categories, so 3 embedders × 24 categories × 4 seeds = **288 array cells per
dataset** (all of an embedder's styles run inside its cell), 150 votes each.
Boxless datasets keep the old prevalence spread at `MAXPATCH_N_CATEGORIES`.

## Category selection: scale bands, not prevalence

The study's question is about object **scale**, so a boxed dataset's categories
are sampled to span scale on purpose — `MAXPATCH_N_PER_BAND` from each of the
four `SCALE_BANDS`, which straddle the two reference scales (one DINOv3 patch,
~0.51 % of image area; one HAC leaf, ~8.3 %, the smallest candidate the tree can
propose). Selecting by prevalence instead left scale coverage to chance, which
is how the first run ended up with only 5 categories above leaf scale — the
exact regime the hypothesis is about.

Two rules make the sample honest:

- **Scale means the *voted* box.** A category's scale is the median area of
  `region_box_for_category` — the **union** over every annotated instance,
  which is what a Good vote actually drags — never the median per-instance
  area. The two diverge sharply on multi-instance categories.
- **Near-frame votes are dropped.** A category whose median voted box exceeds
  `MAXPATCH_MAX_VOTED_AREA` is excluded: at that size a "region vote" is an
  image-level vote, and the cell would measure what the boxless Caltech-101 arm
  measured (what happens when the user ignores region voting) rather than what
  happens when the target is large. Drops are logged by name, never silent.

Within a band, ties break toward the lowest **union inflation**
(`voted_area / instance_area`) — categories whose vote is typically one clean
object rather than a union over scattered instances.

## Sizing knobs (env vars, read by `experiment_config.py`)

| Var | Default | Meaning |
|---|---|---|
| `MAXPATCH_DATASETS` | `visual_genome_m,openlogo_a,caltech101_m` | Demo datasets |
| `MAXPATCH_EMBEDDERS` | `dinov2_patch,dinov3_patch,siglip` | Embedders |
| `MAXPATCH_PATCH_STYLES` | `max_patch,max_patch_hac,max_patch_pca_hac,whole_image` | Styles run on patch embedders |
| `MAXPATCH_N_PER_BAND` | `6` | Categories per scale band (boxed datasets) |
| `MAXPATCH_MAX_VOTED_AREA` | `0.80` | Drop categories whose median voted box exceeds this |
| `MAXPATCH_N_CATEGORIES` | `6` | Categories for **boxless** datasets (spanning common→rare) |
| `MAXPATCH_N_SEEDS` | `4` | Seeds (paired across arms) |
| `MAXPATCH_MAX_STEPS` | `150` | Vote budget per trajectory |
| `MAXPATCH_EXEMPLAR_CANDIDATES` | `8` | Cropped exemplars pre-embedded per category |

## Notes / gotchas

- **DINOv3 is gated.** No `HF_TOKEN` → prepare skips `dinov3_patch` (loudly)
  and the report simply lacks those arms.
- **Memory**: `visual_genome_m` × `max_patch` flattens ~N×(14²–16²) patch rows;
  the style keeps the matrix fp16 and scores in chunks, but budget ~64G per
  task to be safe.
- **Pairing**: within a `(dataset, category, seed)` cell all of an embedder's
  styles share the same load, the same sim/test split, and the same exemplar.
  Across embedders the split and exemplar image are also identical (both are
  derived from seeds/ids, not vectors), so arm comparisons are paired
  everywhere.
- Results live under `/exp/$USER/max-patch/results`.
