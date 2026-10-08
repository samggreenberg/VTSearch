# A finer tile layer for small marks (#4415)

**Question.** The State of the App: Structural Document (#4392) found the
ranking's limit on its four weakest classes was Stage 1. Their missed
positives sat beyond the 2,000-page shortlist: small marks, 0.06–0.11 of the
page width, in text-dense 0.25 × 0.18 tiles. Does a second, finer tile layer
fix it? At what cost? Arms and verdicts were pre-registered on #4415.

**Answer: yes. Two tile layers ship, keeping the original projection.**

| FullMarks v5.0, 36 classes, tier `m`, end to end, held-out half | A: shipped (one layer) | **B: + fine layer, projection v1 (ships)** | C: + fine layer, refit v2 |
|---|---:|---:|---:|
| AP, no votes | 0.78 | 0.79 | 0.79 |
| AP, 10 clicks | 0.90 | **0.93** | 0.93 |
| **AP, 25 clicks** | 0.91 | **0.93** | 0.93 |
| paired vs A at 25 clicks (95%) | | **+0.022** [+0.007, +0.039] | +0.016 [+0.003, +0.031] |
| paired vs B at 25 clicks | | | −0.006 [−0.012, −0.001] |
| retrain, p90 | 3.1 s | 3.2 s | 3.0 s |
| tiles a page / tile RAM at 50,000 pages | ~57 / ~2.7 GiB | 178 / 8.5 GiB | 178 / 8.5 GiB |
| tiling at load, 16 threads | ~7 min | ~19 min | ~19 min |

- **B passed the AP bar and the retrain budget.**
- **It roughly triples tile memory and load-time tiling**, which sent the
  decision to the owner. They chose to ship two layers, but to refit the
  projection on both layers' tiles first.
- **That refit (C) scores 0.006 AP lower than B**, a small difference but a
  resolvable one. C was to ship only if it was at least as good, so **B ships:**
  v1, fit on coarse tiles only, applied to both layers.

## Where the fine layer helps

AP at 25 clicks; "beyond K" is test positives outside the shortlist at the
final click:

| class | A | B | beyond K, A → B |
|---|---:|---:|---|
| `ucsf/logo_bat_leaf` | 0.70 | **0.84** | 28 → 13 |
| `ucsf/logo_bw_oval_emblem` | 0.50 | **0.70** | 5 → 3 |
| `tobacco800/logo_afm90c00-first_1_0` | 0.80 | **0.92** | 13 → 3 |
| `tobacco800/logo_ald41a00-ernest_1` | 0.87 | **0.97** | 10 → 3 |

These are exactly the small-mark classes #4392 named. A 0.125 × 0.09 tile
holds the mark with far less of the surrounding text, so its VLAD describes
the mark. The returned set improves with the ranking: the #4367 line's F1 at
25 clicks goes 0.85 → 0.87.

## What changed

- **`structural_tiles.TILE_LAYERS`** now holds both layers: 0.25 × 0.18 and
  0.125 × 0.09, each with stride half a tile and at least 20 keypoints. A page
  scores its best tile across them.
- **`FIT_LAYERS`** records that the cached projection is fit on the coarse
  layer alone. That way `fit_tile_projection.py` regenerates v1 exactly,
  whatever the app applies it to. `PROJECTION_NAME` stays
  `tile_projection_v1`.
- **The fit script** pins pool workers to one BLAS thread and gives the SVD
  every thread.
- **`sota_documents.py`** gets `--tile-layers coarse` and `--projection` to
  reproduce arms A–C, and logs tiles a page and tile memory.

The v2 projection was a measured arm, not a shipped asset. Its cache file was
deleted after the verdict.

## Reproduce

```bash
source scripts/experiments/pile/pile_env.sh
python scripts/experiments/fullmarks/sota_documents.py --tier m --max-v 50 \
    --matrix /expscratch/$USER/fullmarks/votes-4162/matrix-m \
    --feature-cache /expscratch/$USER/fullmarks/features --out <dir>/armB      # B is the default now
#   A: add --tile-layers coarse
#   C: fit a both-layer projection under another name, then pass --projection <that name>
```

`measurements/` holds every arm's `steps.csv` and arm B's
`positives_final.csv`. Arm A is #4367's validation run, which used the same
code with the shipped tiling. Run directory: `/expscratch/sgreenberg/fine-4415/`.
Arm C's first run died when `/expscratch` filled up; its last 12 classes were
re-run as `armC2`.
