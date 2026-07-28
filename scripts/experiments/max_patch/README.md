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
| `dinov2_patch` | `max_hac`, `max_patch`, `whole_image` (CLS control) |
| `dinov3_patch` | `max_hac`, `max_patch`, `whole_image` (CLS control) |
| `siglip` | `whole_image` (standard baseline) |

- **max_hac** — production pipeline: Good region-votes snap to the nearest HAC
  region-tree node, Bad votes flood the CLS + HAC leaves, images score by
  max-pooling the MLP over all ~2K region nodes.
- **max_patch** — no tree: Good region-votes train on the *single raw patch*
  nearest the voted box, Bad votes flood *every* raw patch (bag-weighted so a
  rejected image still counts once), images score by max-pooling the MLP over
  all H×W raw patches.
- **whole_image** — single global vector for votes and scores.

**Startup sort**: each cell's exemplar is a cropped positive (ground-truth box,
pre-embedded at prepare time); its full-image embedding is scored against the
dataset *in each style's own geometry* (whole-image cosine / max over region
nodes / max over patches) and the Autopilot seed phase votes down that ranking.

## What each stage does

| Stage | Script | Output |
|---|---|---|
| 0 · prepare | `prepare_data.py` | Per-(dataset, embedder) pickles under `$VTSEARCH_DATA_DIR/embeddings/<ds>__<emb>.pkl` (with `patch_grid`/`patch_regions` for the DINOs), exemplar-crop vectors under `results/crops/`, and `prepare_info.json` (counts, selected categories, embed timings). |
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

Default grid: 3 datasets × 3 embedders × 6 categories × 4 seeds = **216 array
cells** (all of an embedder's styles run inside its cell → 504 style-runs),
150 votes each.

## Sizing knobs (env vars, read by `experiment_config.py`)

| Var | Default | Meaning |
|---|---|---|
| `MAXPATCH_DATASETS` | `visual_genome_m,openlogo_a,caltech101_m` | Demo datasets |
| `MAXPATCH_EMBEDDERS` | `dinov2_patch,dinov3_patch,siglip` | Embedders |
| `MAXPATCH_PATCH_STYLES` | `max_hac,max_patch,whole_image` | Styles run on patch embedders |
| `MAXPATCH_N_CATEGORIES` | `6` | Categories per dataset (spanning common→rare) |
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
