# VTSBrowse UMAP parameter sweep (`scripts/experiments/umap_params/`)

Dev-only experiment harness for **Part 1** of `docs/plans/vtsbrowse-empirical-tuning.md`:
choose the VTSBrowse Stage-1 projection defaults (`n_neighbors`, `min_dist`, and
the `compact` boolean) **per embedder**, by a parameter sweep scored with a
ceiling-normalized taxonomy-separability metric, plus label-free structure
guards and multi-seed stability. Runs on the GRID GPU cluster.

Not shipped: this directory is outside the package, outside `deptry`'s prod
surface, and must not affect `./run-tests.sh`.

## What it does

1. **Embed once** (`prepare_dataset.py`, `prepare_fsd50k.py`) — build one cached
   `(ids, matrix, labels)` `.npz` per `(dataset, embedder)`. Audio reuses the
   existing CLAP demo pickles verbatim; every image dataset is re-embedded with
   `clip` / `siglip` / `siglip_l`. Matrices are L2-normalized (the ingest
   contract). This is the expensive step; the sweep re-fits over these caches.
2. **Sweep** (`sweep.py`) — for each matrix, fit UMAP over the
   `n_neighbors × min_dist` grid × 3 seeds (seeded *only here*; production stays
   unseeded), and score every fit **twice** — raw coordinates and
   post-`compact_layout` coordinates — so `compact` is a free axis. Emits one CSV
   row per `(params, seed, compact)` cell.
3. **Metric** (`metric.py`) — ceiling-normalized taxonomy separability + guards
   (see below).
4. **Summarize / plot / visualize** (`summarize.py`, `plots.py`, `visualize.py`)
   — per-embedder recommendation, the compaction verdict, the N-analysis, report
   figures, and taxonomy-colored 2-D scatter grids.

## The metric (plan §Metric)

- **Taxonomy separability.** Build a k-NN graph (k=20) of the 2-D layout. For
  each taxonomy node (class) with member set S, score each point by the fraction
  of its 2-D neighbors inside S; the node score is the **AUROC** of that fraction
  vs actual membership (≈1 ⇔ a clean local boundary). Average over nodes per
  taxonomy level, then average the levels. Position/scale invariant and
  multi-island tolerant.
- **Ceiling normalization.** Compute the same score on the original embedding
  matrix (cosine k-NN) and report the **ratio** 2-D / high-D — the part the UMAP
  params actually control, not the embedder's own class separability.
- **Guards (label-free):** k-NN recall (2-D↔high-D neighbor overlap),
  trustworthiness, continuity — veto layouts that fake class purity by shattering
  the space. Trustworthiness is subsampled (N≤2000) since it is O(N²).
- **Stability:** inter-seed neighbor-set agreement across the 3 seeds.

## Datasets

Image + audio only (the embedder scope). Cached demo sources plus two
deep-taxonomy additions:

| Media | Datasets | Taxonomy depth |
|-------|----------|----------------|
| Audio (clap) | ESC-50 S/M/L, GTZAN, **FSD50K eval** | ESC-50 2-level; FSD50K AudioSet (roots + classes) |
| Image (clip/siglip/siglip_l) | Places365 S/M/L, Caltech-256 S/M, **iNaturalist subset** | Places365 indoor-outdoor+scene; iNat 6-level lineage |

Deep-taxonomy build: FSD50K eval from Zenodo; iNaturalist = 156 species (3
kingdoms × 9 classes × 26 orders) streamed-selectively from the iNat2021 `val`
tarball. Prep scripts and the species selection live here.

## Running on the GRID

```bash
# one long GPU job: embed (idempotent) → sweep → summarize
sbatch scripts/experiments/umap_params/run_all.sbatch
# then, for the report:
python scripts/experiments/umap_params/plots.py
python scripts/experiments/umap_params/visualize.py esc50_l clap
```

Env (`/exp/sgreenberg/umap_env.sh`): the venv needs
`LD_LIBRARY_PATH=/cluster/apps/python/3.12.3/lib`; models live on
`/exp/scale26/.../models` (`VTSEARCH_MODELS_DIR` + `HF_HOME`) because the model
cache does not fit on the 50 GB `/exp/$USER` volume; results write to
`/exp/scale26/.../umap-sweep/`.

## Files

| File | Role |
|------|------|
| `common.py` | dataset roster, taxonomy builders, the parameter grid |
| `metric.py` | separability + guards + stability (pure numpy/sklearn) |
| `prepare_dataset.py`, `prepare_fsd50k.py` | embed-once → `.npz` caches |
| `sweep.py` | the grid × seeds × compact sweep → CSV rows |
| `summarize.py` | recommendations, compaction verdict, N-analysis → `summary.json` |
| `plots.py`, `visualize.py` | report figures + layout scatter grids |
| `build_report.py` | assemble the self-contained HTML report (figures inlined) → `docs/reports/` |
| `run_all.sbatch` | the end-to-end GRID job |
