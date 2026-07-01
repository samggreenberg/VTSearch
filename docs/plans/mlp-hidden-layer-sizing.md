# MLP hidden-layer sizing

**Status: investigation done; one code change proposed, not yet made.** This
doc records an empirical capacity sweep of the detector MLP's hidden layer and
the one actionable finding it produced: the **lower bound** `MLP_HIDDEN_MIN = 4`
underfits and is unstable on harder tasks, and the evidence supports raising it
to `8`. The **upper bound** `MLP_HIDDEN_MAX = 32` is well-placed and should not
change. Nothing here has been applied to `vtscore/config.py`.

The sweep was produced by `scripts/overfitting_probe.py --sweep-hidden ...`
(added alongside this doc). Re-run it to reproduce or extend any table below.

## Background: how the hidden layer is sized today

The detector is a small MLP (`vtscore/training/mlp.py`):

```
Linear(input_dim, hidden_dim) -> ReLU -> Dropout(0.5) -> Linear(hidden_dim, 1)
```

`hidden_dim` is auto-sized from the training-set size
(`_auto_hidden_dim`, `vtscore/training/mlp.py`):

```python
max(MLP_HIDDEN_MIN, min(MLP_HIDDEN_MAX, n_train // 3))   # 4 .. 32
```

with `MLP_HIDDEN_MIN = 4`, `MLP_HIDDEN_MAX = 32` (`vtscore/config.py`). So the
width grows one neuron per three votes, floored at 4 (reached at ≤12 votes) and
capped at 32 (reached at ≥96 votes). The regularizers around it — dropout 0.5,
`weight_decay=1e-4`, inverse-frequency class weighting — are what let the width
range stay small without overfitting.

## Method

`scripts/overfitting_probe.py` holds the train/held-out split, the sampled
votes, and the seeds fixed and varies **only** `hidden_dim` (passed through the
new `train_model(..., hidden_dim=)` override). For each width it reports, over
5 seeds:

- `REAL_test` — held-out ranking AUC when items are labeled by their true
  category. This is the capacity signal (threshold-free, so it isolates ranking
  quality from threshold calibration). Its across-seed **std** flags instability.
- `NOISE_train` / `NOISE_test` — the same model trained on coin-flip labels.
  `NOISE_train → 1.0` as width grows is pure memorization; `NOISE_test ≈ 0.5`
  at every width confirms no capacity level can generalize a signal that isn't
  there (and its slow creep upward is the first whiff of overfitting).

Runs used `caltech101_s` (image / SigLIP) and `20newsgroups_s` (text / E5),
100 good + 100 bad votes, inclusion 0.

## Findings

### Images (SigLIP) — task is already linearly separable

`REAL_test` is a perfect 1.000 at **every** width from 4 to 256 (airplanes,
dolphin, flamingo all identical). SigLIP places these categories in a linearly
separable arrangement, so the hidden layer does no real work and width is
irrelevant. 32 is far more than enough; even 4 is perfect. This case can't
speak to whether 32 is ever *too small* — for that, see text.

### Text (E5) — the informative regime

Held-out `REAL_test` vs width (5 seeds); note the width-4 column:

| hidden | religion | science | cars | NOISE_test (cars) | NOISE_train (cars) |
|-------:|---------:|--------:|-----:|------------------:|-------------------:|
| 4      | 0.900 ±0.15 | 0.975 | **0.684 ±0.23** | 0.489 | 0.698 |
| 8      | 0.977 | 0.974 | 0.949 | 0.506 | 0.948 |
| 16     | 0.977 | 0.978 | 0.956 | 0.503 | 0.950 |
| **32** | **0.977** | **0.980** | **0.959** | 0.512 | 0.976 |
| 64     | 0.977 | 0.981 | 0.960 | 0.513 | 0.999 |
| 128    | 0.976 | 0.982 | 0.960 | 0.517 | 1.000 |
| 256    | 0.976 | 0.982 | 0.959 | 0.518 | 1.000 |

Three regimes, consistent across topics:

1. **Below ~8: underfit + unstable.** Width 4 gives `cars` only 0.684 held-out
   AUC with a **±0.23** seed std (religion: 0.900 ±0.15). The 4-neuron model
   can't reliably fit the boundary and swings wildly by seed.
2. **~8 through 256: flat plateau.** Once past underfitting, held-out AUC is
   width-independent across a 32× range. **32 sits on this plateau with margin.**
3. **Above 32: capacity only buys memorization.** `NOISE_train` reaches a perfect
   1.000 by width 64–128 while `NOISE_test` creeps 0.489 → 0.518 and `REAL_test`
   flattens/dips a hair. Going bigger than 32 is all downside (no real gain, more
   memorization, more variance, more compute).

### Conclusion

- **`MLP_HIDDEN_MAX = 32` is correct** — it's at the top of the stable plateau,
  big enough that no tested task is still improving, small enough to stay below
  where capacity starts rewarding noise. Leave it.
- **`MLP_HIDDEN_MIN = 4` is the soft spot** — 4 neurons underfits and destabilizes
  on harder text topics. The stable minimum in the data is ~8.

## Proposed change (not yet made)

Raise the floor in `vtscore/config.py`:

```python
MLP_HIDDEN_MIN = 8   # was 4
```

Scope and risk:

- Only affects the **tiny-vote regime**: the floor binds when `n_train // 3 < 8`,
  i.e. **fewer than ~24 total votes**. Above that, `_auto_hidden_dim` already
  returns ≥8 and nothing changes.
- In that regime held-out results are noisy regardless and the app's labeling
  indicators are already telling the user to keep voting, so the practical
  impact is small — but the change strictly removes the observed underfit/instability
  at the very start of a session at no cost (8 neurons is still trivially cheap).
- Update the `_auto_hidden_dim` docstring and the `docs/ML.md` "4–32 neurons"
  references to "8–32" if applied. Add/adjust any test that pins the floor.

## Open follow-ups

- [ ] Apply the `MLP_HIDDEN_MIN = 4 → 8` change (above) if/when someone touches
      training config; update `docs/ML.md` and `_auto_hidden_dim` docstring to match.
- [ ] Optional: extend the sweep to audio (`esc50_s` / CLAP) and video
      (`ucf101_s` / X-CLIP) to confirm the plateau shape holds across every
      modality, not just image + text. The probe already supports `--dataset`.
- [ ] Optional: sweep with small vote counts (e.g. `--n-good 6 --n-bad 6`) to
      measure the floor's effect directly in the regime where it actually binds,
      rather than inferring it from the width-4 column at 100+100 votes.
