# MLP hidden-layer sizing

**Status: shipped.** The one actionable finding — raising `MLP_HIDDEN_MIN`
4 → 8 — has been applied to `vtscore/config.py` (+ `docs/ML.md`,
`_auto_hidden_dim` docstring). Two optional sweep extensions remain open.

## Open follow-ups

- Optional: extend the sweep to audio (`esc50_s` / CLAP) and video
  (`ucf101_s` / X-CLIP) to confirm the plateau shape holds across every
  modality, not just image + text. The probe already supports `--dataset`.
- Optional: sweep with small vote counts (e.g. `--n-good 6 --n-bad 6`) to
  measure the floor's effect directly in the regime where it actually binds,
  rather than inferring it from the width-4 column at 100+100 votes.

## What shipped

- Raised `MLP_HIDDEN_MIN` 4 → 8 in `vtscore/config.py`. Only affects the
  tiny-vote regime: the floor binds when `n_train // 3 < 8` (fewer than ~24
  total votes); above that `_auto_hidden_dim` already returns ≥8. Cheap
  (8 neurons), strictly removes the observed underfit/instability at the start
  of a session.
- `MLP_HIDDEN_MAX = 32` left unchanged (it sits at the top of the stable
  plateau — correct).
- Updated `docs/ML.md` "4–32 neurons" → "8–32" and the `_auto_hidden_dim`
  docstring; the config/training package docs match.
- Added `scripts/overfitting_probe.py --sweep-hidden ...` alongside this doc;
  re-run to reproduce or extend any table below.

## Method (for reproducing / extending the sweep)

The detector is a small MLP (`vtscore/training/mlp.py`):
`Linear(input_dim, hidden_dim) → ReLU → Dropout(0.5) → Linear(hidden_dim, 1)`.
`hidden_dim` is auto-sized from the training-set size:
`max(MLP_HIDDEN_MIN, min(MLP_HIDDEN_MAX, n_train // 3))` — one neuron per three
votes, floored (≤12 votes) and capped (≥96 votes). Dropout 0.5,
`weight_decay=1e-4`, and inverse-frequency class weighting keep the small width
range from overfitting.

`scripts/overfitting_probe.py` fixes the split, sampled votes, and seeds and
varies **only** `hidden_dim`. Per width (5 seeds) it reports: `REAL_test`
(held-out ranking AUC on true-category labels — the capacity signal, its
across-seed std flags instability) and `NOISE_train` / `NOISE_test` (same model
on coin-flip labels — `NOISE_train → 1.0` is pure memorization, `NOISE_test ≈
0.5` confirms no width generalizes absent signal). Runs used `caltech101_s`
(image / SigLIP) and `20newsgroups_s` (text / E5), 100 good + 100 bad, inclusion 0.

## Results (informs the open sweeps)

**Images (SigLIP):** `REAL_test` is a perfect 1.000 at every width 4→256 — the
categories are linearly separable, so width is irrelevant here and this case
can't speak to whether 32 is ever too small.

**Text (E5) — the informative regime.** Held-out `REAL_test` vs width (5 seeds):

| hidden | religion | science | cars | NOISE_test (cars) | NOISE_train (cars) |
|-------:|---------:|--------:|-----:|------------------:|-------------------:|
| 4      | 0.900 ±0.15 | 0.975 | **0.684 ±0.23** | 0.489 | 0.698 |
| 8      | 0.977 | 0.974 | 0.949 | 0.506 | 0.948 |
| 16     | 0.977 | 0.978 | 0.956 | 0.503 | 0.950 |
| **32** | **0.977** | **0.980** | **0.959** | 0.512 | 0.976 |
| 64     | 0.977 | 0.981 | 0.960 | 0.513 | 0.999 |
| 128    | 0.976 | 0.982 | 0.960 | 0.517 | 1.000 |
| 256    | 0.976 | 0.982 | 0.959 | 0.518 | 1.000 |

Three regimes: **below ~8** underfits and is seed-unstable (width 4 gives `cars`
0.684 ±0.23); **~8 through 256** is a flat plateau (width-independent across 32×
— 32 sits on it with margin); **above 32** capacity only buys memorization
(`NOISE_train` → 1.000 by 64–128 while `NOISE_test` creeps up and `REAL_test`
dips a hair). Hence: keep `MLP_HIDDEN_MAX = 32`; the stable minimum is ~8, so the
old floor of 4 was the soft spot (now fixed).
