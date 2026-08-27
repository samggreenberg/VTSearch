# Calibration study runner (issues #2781, #2799, #2836)

Measures **calibration regret** — the extra `FPR + FNR` cost the trained
(cross-calibrated conformal) threshold pays versus the *oracle* threshold for the
same ranking — across the region-voting and binary-voting arms, decomposes it
into rule inefficiency vs calibration→test shift, hunts the runaway-threshold
bug, tests whether re-pooling can save the raw-patch tree, and checks the
Inclusion budget under grouped calibration. Design: `docs/plans/calibration-experiment.md`.

## Arms

| Dataset | Embedder | Style(s) | Calibration |
|---|---|---|---|
| `visual_genome_m` | `siglip`, `siglip_l` | `whole_image` | row-wise |
| `visual_genome_m` | `dinov3_patch` | `max_patch`, `max_patch_pca_hac` | grouped (bag max-pool) |
| `caltech101_m` | `siglip`, `siglip_l` | `whole_image` | row-wise |

The `max_patch_pca_hac` arm additionally emits two **remedial re-pools** of its
own per-node scores (`topk` k=4, `pnorm` extreme-value normalisation), each with
its own recalibrated threshold, tagged in the `pool_variant` column.

## Stages

1. **`prepare_data.py`** — ensures a per-`(dataset, embedder)` pickle + exemplar
   crops for every arm. Reuses the Max-Patch pickles/crops where the pair
   coincides (VG×{siglip,dinov3_patch}, Caltech×siglip); only embeds the missing
   `siglip_l` pairs.
2. **`run_cells.py`** — one SLURM-array task per `(dataset, embedder, category,
   seed)` cell; runs every style for the embedder, emitting the calibration
   metrics (`_CALIBRATION_COLUMNS`) to `results/cells/task_<idx>.csv` and the
   inclusion sweep (`_INCLUSION_SWEEP_COLUMNS`) to `task_<idx>__sweep.csv`.
3. **`analyze.py`** — concatenates the cells, computes the pre-registered
   deliverables, writes `results/summary.json`, `results/agg/*.csv`,
   `results/figures/*.png`, and a `results/REPORT.md` draft.

Under `CALIB_SAFE_THRESHOLDS=1` each step also emits one row per **cut variant**
(`gmm_variant`; `_SAFE_GMM_VARIANTS`) and a per-(step, geometry) **cut
decomposition** frame (`_CUT_DIAGNOSTIC_COLUMNS`) to `task_<idx>__cutdiag.csv`.
Two alternative analyzers read those: `analyze_safe.py` (the #2799 safe-on/off
question) and `analyze_cut.py` (the #2836 question of *which* cut and *why*).

`theory_bench.py` is standalone and needs no dataset: it scores the same cut
rules against a generative model of region voting whose exact rate-optimal cut is
computable, so it can attribute a rule's error to the loss, the fitted family, or
the sample size. Run it with `python theory_bench.py --reps 40`.

## Running on the Grid

```bash
cd /exp/$USER/projects/vts-calib/scripts/experiments/calibration
bash launch_all.sh          # reuse-symlink -> prepare (GPU) -> cells -> analyze
bash launch_safe.sh         # the #2799 safe-threshold sizing, analyze_safe.py
bash launch_cut.sh          # the #2836 cut-rule study: theory bench + analyze_cut.py
bash launch_anchored.sh     # the #2852 anchored-mixture study, analyze_anchored.py
```

Both study launchers are thin wrappers over `launch_all.sh` that flip the
pre-registered knobs and point `CALIB_EXP` somewhere the other studies' outputs
are not.

Each analyzer has a self-test that runs it on fabricated cells with a planted
answer, so a sign error is caught before an overnight run rather than after:
`python selftest_analyze_ab.py`, `python selftest_analyze_cut.py`.

Every analyzer discovers its input through `_cells_io.main_frame_files` /
`side_frame_files` rather than globbing `task_*.csv` itself. That glob also
matches the side frames (`__sweep`, `__cutdiag`, `__cutincl`), which are separate
long-format tables — concatenating one into the main frame yields a ragged
DataFrame whose extra rows enter every aggregate silently. Add a new side frame's
suffix to `SIDE_FRAME_SUFFIXES` and every analyzer is correct by construction.

`launch_all.sh` points `VTSEARCH_DATA_DIR` at the Max-Patch datadir so the shared
embeddings pickles and demo data are read in place (the `siglip_l` pickles land
alongside them harmlessly), and writes all study output under
`/exp/$USER/calibration`.

## Fixed config (pre-registered)

`inclusion=0` (cost = FPR + FNR), `sim_fraction=0.5`, `calibrate_count=2`,
`calibration_fraction=0.5`, `safe_thresholds=False`, MLP trainer, 150 votes,
4 seeds. Env knobs mirror the `MAXPATCH_*` set under the `CALIB_*` prefix.

## Safe-threshold GMM study (issue #2799)

```bash
cd /exp/$USER/projects/vts-calib/scripts/experiments/calibration
bash launch_safe.sh      # safe_thresholds ON, VG only, 30 votes, 8 seeds
```

`launch_safe.sh` re-drives the same pipeline with `CALIB_SAFE_THRESHOLDS=1`:
every step then emits one extra row per safe-threshold GMM variant
(`gmm_variant` column — fit geometry x cut rule x fit space, plus an
`xcal_only` control), and the analyze stage runs `analyze_safe.py` instead of
`analyze.py`. Results land under `/exp/$USER/calibration-safe`, reusing the
shared Max-Patch pickles/crops in place.

## Anchored-mixture study (issue #2852)

```bash
cd /exp/$USER/projects/vts-calib/scripts/experiments/calibration
bash launch_anchored.sh  # safe+anchored ON, VG only, 300 votes (deep regime), 4 seeds
```

`launch_anchored.sh` additionally sets `CALIB_ANCHORED=1`: every step then
emits one row per anchored-mixture arm — the label-anchored family
(`anchored_w{W}_{rule}`: anchored EM on the final model's haystack scores with
the voted items' scores clamped to their labelled component), the fold-anchored
"cross-LabeledGMM" family (`fold_anchored_w{W}_{rule}_{combine}`: per-fold
anchored fits on honest held-out anchors, rank-transferred back to the final
scale), and the `rank_transfer` attribution arm — all step-paired against the
`pooled_mid` (shipped blend) and `xcal_only` controls. The sweep grid is
`CALIB_ANCHORED_WEIGHTS` × `CALIB_ANCHORED_RULES` ×
`CALIB_ANCHORED_FOLD_COMBINES` (see `experiment_config.py`). Analyzer:
`analyze_anchored.py` (H1–H4 verdicts + paired tables); self-test:
`python selftest_analyze_anchored.py`. Results land under
`/exp/$USER/calibration-anchored`. Design and pre-registered decision rules:
`docs/plans/population-anchored-calibration.md`.

Cost note: the fold-anchored arms score the sim set once per calibration fold
per step (`calibrate_count=2` → two extra scoring passes); disable them with
`CALIB_ANCHORED_FOLD_ARMS=0` for a cheap label-anchored-only run.

## Cut-rule × Inclusion study (issue #2865)

```bash
cd /exp/$USER/projects/vts-calib/scripts/experiments/calibration
python selftest_analyze_cutincl.py   # planted answer; run before the array
bash launch_incl_2865.sh             # reuses the #2861 prepare stage
python analyze_cutincl.py
```

Asks **which cut rule should answer the Inclusion knob**, which neither
calibration run could: both scored every arm at inclusion 0, and inclusion 0 is
the one setting where the rule choice cannot matter. Shipping the measured
`κ=0.3, mid` verbatim therefore made the knob a *no-op* for every detector with
usable folds — a midpoint of two component means never looks at the cost weights
inclusion arrives as. `mid_tilt` (#2868) restored the tilt while reproducing the
measured arm bit-for-bit at 0; this run prices that tilt.

`CALIB_CUT_INCL_KS` turns on a side frame (`_CUT_INCLUSION_COLUMNS` →
`task_<idx>__cutincl.csv`): one row per (step, fold-anchored arm, inclusion `k`),
each scored under the cost weights of **its own** `k` and against the oracle at
that same `k`, so regret is comparable along the knob as well as across arms.
The arms are `CALIB_ANCHORED_WEIGHTS` × `CALIB_ANCHORED_RULES` ×
`CALIB_ANCHORED_FOLD_COMBINES`, with `q_tilt` additionally expanded over
`CALIB_CUT_INCL_QTILT_STEPS` (its step size is a free parameter, so the sweep
has to *fit* it rather than assume one).

Cost note: nearly free. The per-fold anchored EM depends on none of the swept
axes, so one fit per anchor weight serves the whole (rule × combine × `k`) grid —
the same no-refit re-cut the app does on a slider drag, which also makes the
sweep measure the object production actually re-cuts.

The analyzer reports the issue's two decision numbers: paired regret at each `k`
against the shipped rule (bootstrapped over **cells**, since consecutive steps of
one trajectory share a model), and how much of the knob survives as **distinct
admitted sets** — a rule that moves the threshold without moving the included set
has fixed nothing. It gates on *pointwise* regret rather than pooled, because an
arm can win on average across the knob while being worse everywhere a user parks
the slider; and it counts an arm harmed only past a `HARM_TOLERANCE` of 0.01
(the margin PR #2891 pre-registered), since a bare significance test rejects even
a perfect arm across ~100 intervals on multiplicity alone.

A note on the candidate set, because the issue's own list has a redundancy in it:
its **candidate 2** ("drop the mixture-weight factor: `lam = fnr/fpr` instead of
`(fnr/fpr)·(w_lo/w_hi)`") describes what the existing `rate` rule *already*
computes — the prior-odds factor in `rate`'s `lam` cancels the `w_lo/w_hi` inside
`_rate_cut`'s `offset` exactly, so `rate` is prior-free and its interior root is
invariant to the mixture weights at every inclusion (pinned by
`tests_lib/detectors/test_cut_inclusion_sweep.py::TestRateIsPriorFree`). The rule
that genuinely retains the priors is `cross_tilt`, added as the literal reading
of candidate 2 so the issue's text gets priced too. Candidate 4 ("keep `mid`") is
the `mid` arm, the honest null.

## Fold-count study (issue #2897)

```bash
cd /exp/$USER/projects/vts-calib/scripts/experiments/calibration
python selftest_analyze_folds_2897.py   # planted answer; run before the array
bash launch_folds_2897.sh               # the screen: every K in one run
bash launch_folds_2897_ab.sh 2 8        # then the live A/B, once K* is named
```

Re-prices production's `calibrate_count=2`: what does raising the number of
cross-calibration folds cost in wall clock, and what does it buy in oracle
regret? Both voting modes, since the calibrators differ (VG region voting runs
the bag-aware calibrator, Caltech binary voting the row-wise one).

`CALIB_FOLD_COUNTS=1,2,3,4,6,8,16` makes every step train `max(counts)` folds
and emit one `folds_k{K}_xcal` row per count — plus `folds_k{K}_blend` (the
retired `cap50` mix-in) and `folds_k{K}_anchored` (production's fold-anchored
rule, and the only arm in which K moves *both* halves of the threshold) — each
carrying that K's regret and its measured `fold_seconds`. This is cheap and **exact** rather than
approximate because the folds are nested: each is an independent stratified draw
off one `RandomState(42)` at a per-fold size that ignores the count, so the K
folds a live `calibrate_count=K` run trains are the first K of these. Every K is
therefore paired within the step, the arm at `K == CALIB_CALIBRATE_COUNT`
reproduces the step's own conformal cut, and the extra folds cannot perturb the
live trajectory (all three asserted in
`tests_lib/detectors/test_fold_count_variant_rows.py`).

Price is set by the grid's **maximum**, not its length: `Kmax - calibrate_count`
extra fold fits per step. Size it from one real cell before submitting.

What the screen cannot see is acquisition feedback — K also steers the rank
position Autopilot's Hard pick samples around — which is why
`launch_folds_2897_ab.sh` runs one full simulation per fold count, each living
at its own K. Pass those arm dirs to the analyzer
(`python analyze_folds_2897.py /exp/$USER/calibration-folds-2897-ab-k8`) to get
the `screen_agrees` check. Analyzer: `analyze_folds_2897.py`; design and
pre-registered decision rules: `docs/experiments/calibration-fold-count/REPORT.md`.

Not to be confused with the older `analyze_folds.py` / `launch_folds_2861.sh`,
which moved the fold count to 4 only to unlock the anchored `qmean`/`qmedian`
combine question — it measures no cost and covers region voting only.

## Good Mining: sweeping the Autopilot **opening** (issue #3267)

Getting enough Goods looks like what separates a VTSearch run that works from
one that fails, and the *opening* is where Goods come from. Today it is fixed:
the top of the seed sort until 3 positives, that sort's cutoff until 4
negatives, then the learned Hard sort ever after.

Both of those phases are the **same operation** — a rank-space `hard` select
against a cut drawn on the seed sort — at two different cuts. The Good phase's
`top` select is that select against a cut placed above every score; the Bad
phase's is against the sort's own fitted GMM, split at the production midpoint.
So the opening collapses to a list of rounds, each naming *how many clicks* and
*where on the sort*, which is what `CALIB_STARTUP_SCHEDULE` sweeps.

```bash
GM_STAGE=live bash launch_good_mining.sh    # coco_val + visual_genome_m
GM_STAGE=bands bash launch_good_mining.sh   # vg_box_small/medium/large
bash analyse_good_mining.sh                 # once every arm drains: everything
python selftest_analyze_startup.py          # planted-answer check on the analyzer
python selftest_curves.py                   # ...and on the quality-over-clicks pair
```

`analyse_good_mining.sh` is the whole analysis in one command, because the pieces
have to agree. To re-run only the analyzer, give it the zero-click anchor
yourself — without it the quality curves have no `t=0` and the far right has
nothing to be compared against:

```bash
python text_baseline.py --results "$CALIB_EXP/results" --out "$OUT/text_baseline.csv"
GM_TEXT_BASELINE="$OUT/text_baseline.csv" GM_OUT="$OUT" python analyze_startup.py
```

Grammar (full reference: [`vtscore/eval/startup_schedule.py`](../../../vtscore/eval/startup_schedule.py)):
`<g|b|n><count>@<top|mid|k[-]N|q<frac>>`, comma-separated. `g3` stays until 3
goods exist, `b4` until 4 bads, `n8` for 8 clicks; `@top` cuts above every score,
`@mid` at the shipped GMM midpoint, `@k-3` at that GMM split under inclusion −3,
`@q0.05` at the sort's 5th rank percentile. `g3@top,b4@mid` is today's opening
and is *required* to reproduce a default run click for click.

**Two arms are load-bearing.**

- `deep_first` (`n10@q0.35,n6@mid`) is the **falsifier**: it opens below the good
  mass and must mine *fewer* positives. If it does not, depth is not the
  mechanism and no other number in the run is interpretable — `analyze_startup.py`
  withholds the verdict rather than reporting one.
- `flat_mid` (`n16@mid`) is the **length-matched control**. Every banded arm
  spends 16 opening clicks against `prod`'s ~7, so a win over `prod` alone could
  be "spend more clicks before training". Read each banded arm against both.

**Expect the `k` family to be partly inert, and check rather than assume.** How
far a given inclusion moves the pick is a property of the fitted mixture, not of
the inclusion: on a steep sort the whole usable range can land inside a couple of
rank percent, so a grid that looks well spread in `k` can be nearly a point in
the space the picks actually live in. That is exactly why the `q` family exists
beside it — `q` establishes whether *position* is the mechanism, `k` asks whether
the app's existing Inclusion knob is a usable handle on it. The analyzer reads
each arm's realized `startup_cut_percentile` and reports "measured nothing"
rather than "the lever does nothing".

### Quality over clicks — the standard figure pair

Every arm's whole point is what the *user's detector* is worth after N clicks, so
`analyze_startup.py` emits that as its headline figure and delegates the drawing
to [`curves.py`](curves.py), which is the one implementation every simulated-user
study shares (see the `grid-experiments` skill for the rule):

- `cost_vs_clicks.png` / `average_precision_vs_clicks.png` — **the averages**: a
  panel per dataset, a line per arm, meaned over every seed and category, with an
  inter-quartile band.
- `cost_vs_clicks_runs__<dataset>.png` — **the individuals**: a panel per arm
  holding every one of that arm's seeds on that dataset as its own line,
  coloured by the category's prevalence.

**Click 0 is the free text sort.** There is no detector at the far left, so each
curve is anchored on what typing the query got for nothing (`text_baseline.py`,
per cell) and that level is carried across the panel as a dotted line. The far
left is what typing was worth, the far right is what clicking was worth, and the
crossing between them — reported as a table in `REPORT_startup.md` — is how many
clicks it took before the detector was worth more than the query.

The coverage strip under each panel is the denominator: a starved cell trains no
detector and emits no metric row, so an arm that starves on a third of its grid
would otherwise have its mean computed over the two thirds that worked and look
*better* for it. The mean is dashed wherever it describes fewer than 95% of the
arm's cells; only a solid segment is a level worth quoting.

To redraw them without re-running the analysis:

```bash
python curves.py --results "$CALIB_EXP/results" \
  --arms prod,top_long,easy_med_hard,band_wide,incl_k,incl_k_wide,flat_mid,deep_first \
  --baseline "$OUT/text_baseline.csv" --prevalence "$CALIB_EXP/results/prepare_info.json" \
  --out "$OUT/figures"
```

### The pick log

`CALIB_EMIT_PICKS=1` (the default) writes `task_*__picks.csv`: **one row per
click**, carrying what was picked, whether it turned out to be a positive, and
where on the seed sort it came from. The main frame cannot answer this study's
questions — it starts at the first *trainable* step, so the opening, which is
the whole subject, is exactly the part it does not record. A cell whose opening
never found both classes emits no main row at all; that is a result about that
arm (the starvation regime), and the analyzer counts those cells rather than
dropping them.

### Sizing

`live` is 2 datasets × `siglip` × (6 COCO + 4 bands × 3 VG) categories × 6 seeds
× 8 arms ≈ 860 cells at `CALIB_MAX_STEPS=100`. `bands` triples the dataset count.
Per the [GRID playbook](../GRID-PLAYBOOK.md), time **one real cell** before
submitting `bands` — do not extrapolate from the `live` stage's slowest cell.
