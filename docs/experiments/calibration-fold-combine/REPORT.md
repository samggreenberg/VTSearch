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

### What this run structurally cannot see, and the gate that follows from it

Every arm is a **counterfactual re-cut** of a trajectory that lived under
production's rule, so the votes are held fixed. But the threshold is the rank
position Autopilot's Hard pick samples around: a different combine rule would
have collected *different votes*, and no screen holding the trajectory fixed can
see that. This is the same limit #2897's screen had, and there its live A/B
effect sank below the ship margin — a screen result is a **reason to book an
A/B, not a substitute for one**.

Pre-registered gate: **an A/B is booked only if the screen resolves a contrast
above `MARGIN` in the deep regime.** If the screen comes back unresolved, the
answer is "the disagreement does not matter enough to spend cluster time on",
and that is the finding.

### Grid

| | |
|---|---|
| Fold counts | K ∈ {1, 2, 3, 4, 6, 8, 12, 16} — same as #2897, so the fold-count axis stays comparable |
| Live count | `calibrate_count = 2` (the trajectory users get; every arm scored on the same votes) |
| Environment | **`vg_scale_any`** × {`siglip` → binary, `dinov3_patch`/`max_patch` → region} |
| Categories | all 12 classes; 300 positives each against one shared 3900-image negative pool |
| Prevalence | **7.1% in every cell**, by construction |
| Voting | derived **per cell** (boxes **and** a patch grid), never from the dataset name |
| Head | unset → `PRODUCTION_HEAD` (the linear SVM) |
| Sizing | 4 seeds × 150 steps → 96 cells |
| Inclusion | 0 |

#### Why not `visual_genome_m`

This study began on `visual_genome_m` + `caltech101_m`, the #2897 grid, and that
was the wrong instrument. Its selected categories run from **25** positives
(`banana`) to **1645** (`building`), and the thin end does not merely add noise:
the first two cells of the first attempt completed in seconds as **header-only
CSVs with zero rows** — `ball`, 51 positives in 4193 media, never reached a
trainable step. Such a file is non-empty, parses cleanly and passes
`find -size 0`, so it counts as a present cell while contributing nothing.

A threshold *is* a quantile of the calibration set. A grid whose calibration sets
differ 60-fold in size is confounding the axis the study reads, and a difference
between two combine rules could be a prevalence difference wearing a disguise.

`vg_scale_any` is #3156's hand-checked scale dataset with the box-size band
collapsed away (`class@band → class`), derived from the built `vg_scale` pickle
so it is provably the same images, boxes and label corrections. Its exclusion
semantics are preserved and are the part that matters: the 300 media that hold
one of the 12 classes at a size outside every band stay **evaluable on nothing**,
rather than becoming negatives — scoring them as negatives would penalise a
detector for finding a real bus, which is what #3156 exists to prevent.

Two further gains fall out. Both voting modes now come from **one** dataset, so
the mode contrast is no longer confounded with the dataset the way #2897's
caltech-vs-VG split was; and `caltech101_m` is dropped rather than kept as a
second binary environment, since a second *environment* is what the
pre-registered A/B gate below is for.

#### The cost of that choice, stated up front

Uniform 7.1% prevalence is what makes the combine and space legs clean, and it is
also the thing most likely to make the **contamination** legs read null: a
degenerate (single-class) fold is the hazard `qmedian` is robust to, and a
well-populated grid presents fewer of them. The exposure term is therefore
reported beside every robustness claim, and the cold-start windows — where even a
7.1% dataset yields folds with one or two positives — are where that exposure
lives. If `any_dropped_rate` comes back near zero in every window, the median
legs are untested rather than refuted, and the report must say so.

### Errata inherited from #2897, fixed here

- **The `voting` label was read from the dataset name.** Region voting needs
  boxes *and* a patch grid, so `visual_genome_m × siglip` runs whole-image and
  is a **binary** environment. #2897's report regrouped that cell into binary by
  hand while the analyzer's own column still called it region. It is now derived
  from `experiment_config.region_voting_for` — the same predicate the runner
  uses — so the next study does not have to know.
- **The launcher pinned `CALIB_HEAD=linear`**, which stopped being production at
  PR #3198. This run names no head.
- **Header-only cells were invisible.** A cell whose simulation never reached a
  trainable step writes its header and nothing else — non-empty, parseable, zero
  rows. The analyzer now counts and names them alongside zero-byte and
  unreadable cells, so "96/96 cells" cannot quietly mean something else.

## Results

_Pending — the run has not completed. This section is written from the analyzer's
`summary.json`, `agg/*.csv` and `figures/`, and not before._

## Reproducing

```bash
bash scripts/experiments/calibration/launch_folds_3115.sh prepare   # cpu; reads the pile in place
bash scripts/experiments/calibration/launch_folds_3115.sh size      # time ONE cell first
bash scripts/experiments/calibration/launch_folds_3115.sh arms      # array + analysis
python scripts/experiments/calibration/make_folds_3115_figs.py \
    --results "$CALIB_RESULTS" --out docs/experiments/calibration-fold-combine/figures
```

The run reads the shared pre-embedded pile (`scripts/experiments/pile/pile_env.sh`),
so no cell re-embeds and no stage needs a GPU. `size` exists because this run's
per-cell cost is **not** #2897's — Kmax fold fits now carry an anchored EM per
fold — and quoting a previous grid's seconds is exactly how #3129 produced a
90-minute overestimate.

Analysis code, all in-tree: `scripts/experiments/calibration/folds_combine_3115.py`
(the contrast), `scripts/experiments/calibration/analyze_folds_2897.py` (the
driver and the fold-count axis),
`scripts/experiments/calibration/make_folds_3115_figs.py` (figures), and
`scripts/experiments/calibration/selftest_analyze_folds_2897.py`, which plants a
known answer — including the two acceptance identities — and checks the analyzer
recovers exactly it.
