## Limitations (accepted for this run)

- **Two datasets, not three.** OpenLogo (the extreme small-logo regime) was
  planned but the cluster's shared egress repeatedly failed to fetch the
  27k-file Hugging Face dataset (metadata-resolution stalls and hard HTTP
  errors on both GPU and CPU nodes). Visual Genome's small categories cover the
  sub-leaf-scale regime OpenLogo would have targeted (see Figure 4), so the
  scale story is still testable; a dedicated small-logo dataset would sharpen
  the extreme-small end.
- **Natural prevalence only.** Unlike the MLP-vs-SVM study, this run does not
  add a 1%-rare arm. The question here is about object *scale*, not rarity;
  a rare-prevalence arm is the obvious follow-up for the rare-event FNR angle.
- **MLP classifier only.** Every arm uses the production MLP; only the
  vote/score geometry differs. Whether the MaxHAC↔MaxPatch ordering holds under
  a different ranker is out of scope.
- **Acquisition proxy is style-blind.** The Autopilot pool-acquisition step
  ranks candidates by their whole-image vector under every style (matching the
  existing harness); only *training* and *test scoring* differ per style. This
  keeps vote-order differences attributable to the trained model rather than to
  a different acquisition rule, at the cost of not modelling a per-style
  acquisition order.
- **Exemplar leakage.** The startup exemplar image can land in the held-out
  test split; the optimism is tiny and identical across arms (they share the
  exemplar), so it does not bias the comparison.

## Reproducibility

All code is on branch `claude/max-patch-experiment-run`. The report, figures,
`metrics.json`, and `prepare_info.json` are committed under
`docs/experiments/max-patch/`; the full per-cell CSVs (240 cells × ~290 steps,
too large for the repo's file-size hook) and the cached embedding pickles stay
on the Grid at `/exp/$USER/max-patch/{results/cells,datadir/embeddings}` — point
`analyze.py` at them (`MAXPATCH_EXP=/exp/$USER/max-patch`) to regenerate every
table and figure.

```bash
# 0. one GPU node, worktree env sourced (scripts/experiments/max_patch/)
#    prepare embeds each (dataset, embedder) once and caches a cell pickle that
#    now carries patch_grid + patch_regions + gt regions (see the fix below).
MAXPATCH_EXP=/exp/$USER/max-patch \
MAXPATCH_DATASETS=caltech101_m,visual_genome_m \
MAXPATCH_EMBEDDERS=dinov3_patch,siglip \
MAXPATCH_N_CATEGORIES=12  python prepare_data.py

# 1. the voting array (one SLURM task per dataset×embedder×category×seed)
MAXPATCH_N_SEEDS=5 bash launch_cells.sh

# 2. the report (tables + captioned figures + metrics.json + REPORT.md)
python analyze.py
```

### The fix that made the study valid

The shipped harness copied the demo *cache* pickle, whose serializer persists
only `width`/`height`/`thumbnail_bytes` for images — it silently dropped
`patch_grid`, `patch_regions`, the ground-truth `regions`, and the multi-label
`categories`. Loaded back, every arm would have scored on the whole-image vector
alone and MaxHAC / MaxPatch / whole-image would have collapsed to a single
curve. The fix (a) runs the production patch back-fill (`embed_missing`) in
prepare so the side-channels exist, and (b) serializes the in-memory medias
directly (minus the bulky raster bytes the cell stage never reads) via
`scripts/experiments/max_patch/_cells_io.py`. The differentiated per-style
curves in this report are the evidence the fix works.
