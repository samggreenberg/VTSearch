# Threshold-stability #2790 — data, schema & re-analysis guide

This is the durable guide to the **raw per-step data** produced by the VG boolean
head-strategy experiment (the study that traced the MLP decision-threshold "deep
spikes" down to *model over-flexibility on sparse positives*, and found a **linear
head fixes them**). Keep it so the data can be re-examined from new angles, drilled
into per case, and mined for the *next* problem.

## Where the data lives

| Artifact | Location | Notes |
|---|---|---|
| Per-class raw rows | `/exp/sgreenberg/threshold-stability/vg_bool/<embedder>/<head>/c<ID>/results.jsonl` | one file per (embedder, head, class); 900 rows each = 15 seeds × 60 steps |
| Consolidated dataset | `/exp/sgreenberg/threshold-stability/vg_bool_all.jsonl.gz` | all rows in one gz, written by the `vgfin` finalize job on completion |
| Manifest (coverage) | `/exp/sgreenberg/threshold-stability/vg_bool_manifest.txt` | row counts per (embedder, head) |
| **Durable local archive** | `/home/samiam/experiments/threshold-stability-2790/data/` | timestamped `.tgz` snapshots pulled off `/exp` (which is a volatile 50G quota, chronically ~98% full — do **not** treat `/exp` as durable) |

Each row is **self-describing** (carries `embedder`, `head`, `class`, `seed`, `t`),
so the consolidated file needs no join to the directory tree.

## Provenance / run config

Two-phase sweep on Visual Genome, `scripts/sod/sweep.py`, branch `claude/threshold-stability-2790`:

- **Phase 1** (`vgp1`, job 443276): heads `mlp`, `linear`
- **Phase 2** (`vgp2`, job 443277, `afterany` P1): heads `svm`, `reg-mlp`
- **Embedders**: `siglip` (SigLIP 1), `siglip2` (SigLIP 2)
- **Classes** (15): car clock dog cat bird umbrella bottle chair sign flower bus boat plate hat lamp
- **Per cell**: `--iterations 15` (seeds 0–14) × 60 labeling steps
- **Fixed params**: `--proposals whole --max-labels 60 --neg-multiple 40 --min-box-frac 0.03 --inclusion 0 --calibrate-count 2`
- Full grid = 2 embedders × 15 classes × 4 heads = **120 cells**, 900 rows each.

The four heads are a **flexibility ladder** (see `vtscore/eval/scoring_heads.py`):

| `head` | model | boundary | capacity control |
|---|---|---|---|
| `mlp` | production MLP | non-linear | `hidden_dim = clamp(n_train//3, 8, 64)` — grows with **total** votes |
| `reg-mlp` | narrow MLP | non-linear | `hidden_dim = clamp(4·n_good, 4, 64)` — grows with **good** votes only |
| `linear` | LogisticRegression | linear | logistic loss (this is **not** an SVM) |
| `svm` | SVC(kernel="linear") | linear | hinge / max-margin |

COCO/SigLIP-2 baseline (what VG is testing for generalization): deep-spike rate
`mlp` 0.055 vs `linear` 0.025 / `svm` 0.023 / `reg-mlp` 0.042; `linear`/`svm`
also won on cost & FNR at every budget. See `REPORT.md` / `SPARSE_POSITIVE_PLAN.md`.

## Row schema

Primary keys: **`(embedder, head, class, seed, t)`**.

**Identity / config**
- `dataset` — `vg`
- `class` — target object class
- `embedder` / `reg_name` — embedding model (whole-image ⇒ same)
- `proposal` / `proposal_slug` — `whole` (boolean detector; `maxpatch` would be region-voting)
- `region_voting` — `false` for all of these
- `head` — scoring-head arm (see ladder above)

**Pool sizes** (per class, constant across `t`)
- `n_pos_total` / `n_neg_total` — full positive / negative pool (negatives capped at available ≈ 96k for VG)
- `n_train_pos` / `n_train_neg` — train-split sizes
- `n_test` / `n_test_pos` — held-out test-set sizes (metrics are measured here)

**Per-step state**
- `t` — labeling step (1…60), i.e. click number
- `n_good` / `n_bad` — cumulative good / bad votes
- `phase` — autopilot phase: `good` → `bad` → `hard` → `new`
- `select_mode` — next-item selection (`top`, …)
- `calib_mode` — threshold calibration regime. **`cosine_coldstart`** = pre-MLP
  (good/bad phases, no learned head yet); once the learned head engages it switches
  to the conformal/cross-calibration mode. **Filter `calib_mode != cosine_coldstart`
  to isolate the learned-head regime** (that's where the deep spikes live).
- `threshold` — decision threshold chosen this step
- `stop_recommended` — whether the stop heuristic fired
- `compute_ms` — per-step wall time (ms)

**Metrics** (all on the held-out test set)
- `cost` — **FNR + FPR**, the headline metric (rare-event framing: weights missed
  needles and false alarms equally; preferred over `f1`, which under-imbalance
  undervalues FNR — the quirk that started this whole investigation)
- `fnr` — false-negative rate = needles missed
- `fpr` — false-positive rate
- `f1` — legacy metric, kept for comparison
- `oracle_cost` / `oracle_f1` — best achievable with an oracle-chosen threshold
  (the **gap `cost − oracle_cost` = calibration regret**, distinct from model error)
- `mean_iou` / `corloc` — localization metrics (meaningful for region tasks; ~inert for `whole`)

**Inert here** (region-voting/HAC-tree params carried by the row but unused for
`whole` boolean runs): `alpha`, `hac_k`, `leaf_seeding`, `leaf_assign`, `leaf_beta`,
`pca_dims`, `resolution`, `dinov3_model`, `negatives_exhaustive`, `neg_regions`,
`k`, `n_pos`, `n_neg_train`, `n_pos_exemplars`. See `scripts/sod/sweep.py` +
`vtscore/eval/region_curve.py` for exact definitions before relying on any of these.

## Re-analysis recipes

Load everything into pandas:

```python
import pandas as pd, json, gzip
rows = [json.loads(l) for l in gzip.open("vg_bool_all.jsonl.gz", "rt")]
df = pd.DataFrame(rows)
mlp_regime = df[df.calib_mode != "cosine_coldstart"]   # learned-head steps only
```

**Different perspectives**
- Cost/FNR/FPR vs `t`, grouped by `head` (× `embedder`): the headline arm comparison.
- Same, faceted by `class` — does the linear win hold everywhere, or is it carried by a few classes?
- `cost − oracle_cost` (calibration regret) vs `t` by head — separates *calibration*
  error from *model* error.
- `compute_ms` by head — the runtime cost of each arm (see the speed discussion in the report).
- `phase` / `calib_mode` transition timing — when does each cell leave cold-start?

**Individual cases**
- Pick a `(embedder, head, class, seed)` and plot `cost` vs `t`: you're looking at one
  simulated labeling session. Deep spikes = `Δcost > 0.1` between consecutive `t` in the
  learned-head regime.
- `df.sort_values("cost")` / largest `Δcost` to rank the worst single steps, then inspect
  the `threshold`, `n_good`, `n_bad`, `phase` at that step and the two around it.
- `scripts/experiments/threshold_stability/results_eval.py DIR LABEL [...]` gives the
  trace-free deep-spike rate + cost/FNR/FPR at budgets t∈{20,40,60} per arm directory.
- `deep_spikes.py`, `calib_conditions.py`, `handoff_quality.py`, `inspect_mech.py` —
  existing lenses (spike mechanism, calibration conditions, cold→warm handoff).

**Hunting for new problems** (leads this data can surface)
- Cells where `linear` *loses* to `mlp` — counter-examples to the fix.
- High `fnr` that never recovers (a stuck cut above the test positives) vs transient spikes.
- `stop_recommended` firing too early / too late relative to where `cost` actually bottoms.
- `oracle_cost` itself high & flat → the task is model/representation-bottlenecked, not a
  calibration problem (a different class of failure worth its own study).
- `compute_ms` outliers → per-step cost blowups.

## Known limitations

- **Trace-free**: rows are per-step *aggregates*; there is **no per-media-item detail**
  (which specific images spiked). Per-item case studies need a targeted re-run with
  labeling-trace capture on the interesting `(class, seed)` cells (traces are large —
  scope them narrowly and write to node-local scratch, not `/exp`).
- **Coverage**: classes with too few large-box positives (`min_box_frac 0.03`) may be
  skipped by the sweep; check the manifest for missing cells.
- The `whole`-image boolean setting only; region-voting ({DINOv2,DINOv3} × MaxPatch) is a
  separate planned round.
