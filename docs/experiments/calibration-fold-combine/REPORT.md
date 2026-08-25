# Pooled or averaged? Which of two contradictory docstrings is right

**Issues [#3115](https://github.com/samggreenberg/VTSearch/issues/3115) (the
combine rule) and [#3116](https://github.com/samggreenberg/VTSearch/issues/3116)
(the instruments) · one run, by construction · harness PR pending · Refs #2897**

> **Status: pre-registered, not yet run.** Everything below the *Design* heading
> was written before any cell existed and lives as module constants in
> `scripts/experiments/calibration/folds_combine_3115.py`, which applies it
> mechanically. Results replace the *Results* placeholder; nothing in *Design*
> is edited after seeing a number.

## The question

Two functions in this repo disagree about the same empirical fact.

`threshold_from_fold_orderings` — the cross-calibration path the live app calls
— **pools** every fold's held-out scores into one bag and takes a single
conformal quantile. Its docstring justifies this:

> *"All folds' scores live on the same sigmoid scale, so the pool is
> exchangeable enough for the quantile rule."*

`FoldAnchoredCut._combined_fold_quantile` does the opposite — one cut per fold,
each converted to **that fold's own** quantile, then averaged — and says it
keeps each fold's haystack specifically *"to read a cut's quantile in the scale
it was measured on"*, i.e. it is built on the premise that fold scores are
**not** directly comparable.

One of those premises is wrong. Nobody has measured which.

## Design

### Arms

Every arm re-cuts the **same already-trained fold prefix**, so the whole table
costs arithmetic on cached arrays and no extra fits. Per fold count `K`:

| Arm | Rule |
|---|---|
| `folds_k{K}_xcal` | **the pooled control** — `threshold_from_fold_orderings` verbatim, i.e. today's behaviour |
| `folds_k{K}_tmean` / `_tmedian` | per-fold conformal cuts, averaged in **score** space |
| `folds_k{K}_qmean` / `_qmedian` | per-fold cuts carried to the final model as quantiles of **their own fold's haystack**, averaged, realized on the final scores |
| `folds_k{K}_anchored` | production's fold-anchored rule (`FOLD_ANCHOR_COMBINE` = `qmean`) |
| `folds_k{K}_anchored_qmedian` | the **same fit**, re-cut under the robust combine |
| `folds_k{K}_blend` | the retired `cap50` mix-in, kept for continuity with #2897 |

There is deliberately no `qpooled`: a pooled cut has no single fold haystack to
read a quantile in, so that cell of the 2×2 does not exist.

### The contrast is factored, not lumped

`qmean` vs pooled is what #3115 literally asks for, but it moves two things at
once. It is reported as its legs first:

| Contrast | Isolates |
|---|---|
| `tmean − pooled` | pooling vs **averaging**, space held fixed |
| `qmean − tmean` | score space vs **quantile** space, combine held fixed — the comparability premise itself |
| `*median − *mean` | **contamination** — a degenerate fold is silently *inside* a pooled quantile, weighted 1/K by a mean, and ~ignored by a median |
| `anchored_qmedian − anchored` | the same contamination question on the **shipped** rule rather than the retired one |
| `qmean − pooled` | the total |

### Acceptance checks (identities, not results)

Two properties hold by construction, so a failure means the harness is
mis-wired rather than that a rule performed badly. A study whose only checks are
its own headline columns cannot tell those apart.

- **`k1_score_space_is_pooled`** — averaging one number is the identity, so at
  `K=1` the score-space arm must reproduce the pooled threshold **exactly**, on
  every step. This is independent of everything the run measures.
- **`median_collapses_below_k3`** — the mean and median of at most two numbers
  coincide, so every median contrast must be identically zero below `K=3`.

The second is also *why this has never been asked*: production ships
`calibrate_count = 2`, exactly where the question is structurally invisible.

### Decision rules

Fixed before the run, in `folds_combine_3115.py`:

- **Deep regime** is ≥ 100 votes; the verdict reads `K ≥ 3` only.
- A contrast is **called for a side** only when its paired mean over cells
  exceeds *both* twice its own standard error **and** the pre-registered
  `MARGIN = 0.005`. Both halves are load-bearing: these runs pool hundreds of
  autocorrelated steps per cell, so the standard error gets small enough to
  resolve differences four orders of magnitude below the margin — real, and no
  reason to change a shipped rule.
- Otherwise the contrast is reported **unresolved**, which on this question is a
  genuinely useful answer: *"the two docstrings' premises make no difference
  worth acting on"* settles the disagreement as decisively as either winning.
- **Exposure is reported beside every robustness claim.** `any_dropped_rate` is
  the fraction of steps where at least one fold was too degenerate to contribute
  a cut. A near-zero rate means the median arms were tested against a hazard
  this grid never presented, and their result must be read that way.

### Grid

| | |
|---|---|
| Fold counts | K ∈ {1, 2, 3, 4, 6, 8, 12, 16} — same as #2897, so the fold-count axis stays comparable |
| Live count | `calibrate_count = 2` (the trajectory users get; every arm scored on the same votes) |
| Environments | `visual_genome_m` × {`siglip`, `dinov3_patch`/`max_patch`}, `caltech101_m` × `siglip` |
| Voting | derived **per cell** (boxes **and** a patch grid), so `visual_genome_m × siglip` is binary — see *Errata inherited* |
| Head | unset → `PRODUCTION_HEAD` (the linear SVM) |
| Sizing | 4 seeds × 150 steps |
| Inclusion | 0 |

### Errata inherited from #2897, fixed here

- **The `voting` label was read from the dataset name.** Region voting needs
  boxes *and* a patch grid, so `visual_genome_m × siglip` runs whole-image and
  is a **binary** environment. #2897's report regrouped that cell into binary by
  hand while the analyzer's own column still called it region. It is now derived
  from `experiment_config.region_voting_for` — the same predicate the runner
  uses — so the next study does not have to know.
- **The launcher pinned `CALIB_HEAD=linear`**, which stopped being production at
  PR #3198. This run names no head.

## Results

_Pending — the run has not completed. This section is written from the analyzer's
`summary.json`, `agg/*.csv` and `figures/`, and not before._

## Reproducing

```bash
bash scripts/experiments/calibration/launch_folds_3115.sh
python scripts/experiments/calibration/analyze_folds_2897.py
python scripts/experiments/calibration/make_folds_3115_figs.py \
    --results "$CALIB_RESULTS" --out docs/experiments/calibration-fold-combine/figures
```

Analysis code, all in-tree: `scripts/experiments/calibration/folds_combine_3115.py`
(the contrast), `scripts/experiments/calibration/analyze_folds_2897.py` (the
driver and the fold-count axis),
`scripts/experiments/calibration/make_folds_3115_figs.py` (figures), and
`scripts/experiments/calibration/selftest_analyze_folds_2897.py`, which plants a
known answer — including the two acceptance identities — and checks the analyzer
recovers exactly it.
