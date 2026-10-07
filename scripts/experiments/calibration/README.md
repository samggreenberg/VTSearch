# Calibration study runner (issues #2781, #2799, #2836)

Measures **calibration regret** — the extra `FPR + FNR` cost the trained
(cross-calibrated conformal) threshold pays versus the *oracle* threshold for the
same ranking — across the region-voting and binary-voting arms, decomposes it
into rule inefficiency vs calibration→test shift, hunts the runaway-threshold
bug, tests whether re-pooling can save the raw-patch tree, and checks the
Inclusion budget under grouped calibration. Design: `docs/plans/calibration-experiment.md`.

## Which file belongs to which study

<!-- BEGIN INDEX -->

**This directory is flat on purpose.** ~120 scripts, ~20 concluded studies, one
namespace. #3409 proposed `git mv`-ing each concluded study into a subdirectory;
that was declined, because the cost is real and the benefit is not. These are
archival cluster scripts: the launcher `cd`s here and every file reaches its
neighbours by bare sibling import (`import common`), so a subdirectory breaks
imports in ~50 files and paths in 32 shell launchers, with **no test or type
coverage to catch a mistake** (`pyrightconfig.json` excludes `scripts/`, no
pytest reads them). Roughly 110 backticked references in concluded `REPORT.md`s
name these paths, and a report is a record of what was run at a path that
existed, not a document to rewrite when the tree is tidied. And the cross-study
imports the flat namespace enables are mostly *deliberate*: `analyze_folds_3314`
builds on `analyze_folds_2897` because #3314 is the follow-up to #2897, and
`analyze_transfer` reuses `analyze_cut`'s decomposition because #2883 is the
last link of the #2836 chain. Splitting them would force duplication or a shared
layer that re-flattens what the split just separated.

What the subdirectories were actually wanted for is **knowing which file belongs
to which study**, and that is a table. `scripts/check-calibration-index.py`
(a `run-tests.sh` gate) requires this index to be *total* in both directions:
every `.py` and `.sh` here appears in it exactly once, and every file it names
exists. So a new study's files cannot land unclassified, and a deletion cannot
leave a phantom row.

### The shared layer

Everything below the studies. Edit these with the care of a library, not a
script: a change here moves every study's numbers at once, and several are what
a report's committed-figure requirement points at.

| Files | What they are |
|---|---|
| `common.py`, `experiment_config.py` | Env/`sys.path` setup (call `setup_env()` before importing anything under `vtscore`) and the pre-registered grid. |
| `_cells_paths.py` | Which files in a `cells/` directory are a cell's **main** frame (`main_frame_files`, `side_frame_files`, `SIDE_FRAME_SUFFIXES`). Import-free on purpose — no pandas — so the csv-and-stdlib figure scripts share the rule with the pandas analyzers instead of each re-typing it. |
| `_cells_io.py` | Cell-pickle I/O, the one cell reader (`load_cells` → `(frame, provenance)`, and `describe_load` to print it), the opening assertion, and `load_arm` — the per-arm loader six callers across five studies use (it lived in `analyze_spikes.py` until #3409, and that module re-exports the name). `load_arm` sets a floor-era run's spot-check rows (`phase == "check"`, past `max_steps`) apart unless called with `keep_check=True` (#4364). Re-exports `_cells_paths`' discovery functions. |
| `_rank_metrics.py` | A precision floor's line read off where the positives sit in a ranking (#4357): the top-*K* set each floor keeps, its precision, shortfall and recall, the best cut's recall at the floor, and AP. The one definition the click-0 text baseline and the State of the App analyzer both read with, over a cell's rank frames (`task_NNNN__rankframes.csv`, `CALIB_RANK_FRAME_STEPS`) or a text sort's scores. |
| `prepare_data.py`, `run_cells.py`, `analyze.py`, `launch_all.sh`, `launch_cells.sh` | The three-stage pipeline (#2781). Every study launcher is a wrapper that flips pre-registered knobs over these and re-points `CALIB_EXP`. |
| `noop.py` | The analyze step for a launcher whose analysis runs separately. |
| `study_paths.py` | `require_study_dir` — the check an **analyzer** owes its `--exp` (#4001). The finished study output dirs were deleted on 2026-09-18, so a stale default now dies on whatever file it opens first; this names the directory and the record instead. Readers only: a launcher `mkdir -p`s its study dir, and requiring one would break the re-run that made the delete acceptable. |
| `objective.py` | **Which metric a decision reads** (#4584): the objective, `fbeta` (F-beta of the withheld half above the threshold the app holds, at the row's own beta), on any frame that carries a beta, and `cost` only on one that does not (`CALIB_BETA=off`, a Cost-era run). `with_objective` fills the `fbeta` columns on cells written before the runner emitted them, from their `precision` and `recall`. The A/B analyzer, the curves, the stopping point, the overview and the viewer all default through it. |
| `curves.py`, `selftest_curves.py` | The standard quality-over-clicks figure pair every simulated-user study owes. One implementation; do not write it again. |
| `stopping.py`, `selftest_stopping.py` | **Stopping point and stopping cost** (#3560): where the app's own stopping rules fired on each trajectory, and what the detector cost there — the number a user actually leaves with, as against the "final cost" at a click budget nobody chose. Reads the `phase` column every run since 2026-07-31 already emits, so it enriches finished studies without a re-run. Handles the three things that make a naive average wrong: the rules **flap**, they **often never fire**, and the runs excluded by "average over the ones that stopped" are exactly the slow ones. Holds each stop against the run's own best, reads through a mid-session prompted check, and is what the State of the App analyzer's stopping block calls; `tests_lib/meta/test_calibration_stopping.py` runs the self-test in the suite. |
| `preflight_knobs.py` | **Check 12 of the launch preflight** (`../preflight.sh`): every knob this run pins - the detector's and the session's (the opening, the preference, the spot check and its walk, acquisition) - against what the app ships, so a divergence passes only when declared with `--diverges <knob>`, and a value the harness would refuse fails outright (#4549). Imported under the run's own environment, from the worktree the jobs import. |
| `harvest_headroom.py` | **Sizing a deep grid from its DEEPEST arm** (#3611): the positives-per-class that keep the most aggressive arm's harvest under a study's pre-registered compression bar at its planned horizon, read off a short pilot wave. Runs as check 16c of the launch preflight (`../preflight.sh --require-harvest-headroom`), and standalone (`--pilot <study>/bin`) to size the next pile. The per-arm harvest of a study that has already RUN is a different question, and a per-study script (`harvest_3547.py`) answers it. |
| `viewer.py`, `selftest_viewer.py` | The interactive `viewer.html` every study's report links to. `--reskin` pushes a template change onto committed pages. `--default-metric` / `--hide-metrics` set the metric a page opens on and take retired ones off its menu, at build time or on a reskin. |
| `make_bench_html.py` | A study's `report.html` reading copy, generated from its own `REPORT.md`. |
| `bench_cells.py` | Pure-pandas reading and pairing of overview-benchmark cells, shared by the bench analyzers. |

### The studies

Newest last, matching an `ls` of [`docs/experiments/`](../../../docs/experiments/).
A study's report is the record of what it found; the files here are how it was
produced.

| Study | Files |
|---|---|
| **#2781 Calibration regret** — [report](../../../docs/experiments/2026-07-31-calibration/REPORT.md) | The base pipeline itself; see the shared layer above. |
| **#2799 Safe thresholds** — [report](../../../docs/experiments/2026-08-03-safe-thresholds/REPORT.md) | `launch_safe.sh`, `launch_safe_ab.sh`, `launch_ab_cells.sh`, `analyze_safe.py`, `analyze_ab.py`, `selftest_analyze_ab.py` |
| **#2836 GMM cut rule** — [report](../../../docs/experiments/2026-08-04-gmm-cut/REPORT.md) | `launch_cut.sh`, `launch_tail_2881.sh`, `theory_bench.py`, `analyze_cut.py`, `selftest_analyze_cut.py`, `make_cut_2846_fig.py` |
| **#2841 Mix-in schedule** — [report](../../../docs/experiments/2026-08-04-mixin-schedule/REPORT.md) | `launch_mixin.sh`, `build_coco_pickle.py`, `analyze_mixin.py`, `selftest_analyze_mixin.py` |
| **#2852 Anchored mixture**, and the #2861 anchor-mass sweep — [report](../../../docs/experiments/2026-08-05-population-anchored-calibration/REPORT.md) | `launch_anchored.sh`, `analyze_anchored.py`, `selftest_analyze_anchored.py`, `launch_rate_2861.sh`, `launch_folds_2861.sh`, `analyze_rate.py`, `analyze_folds.py`, `selftest_analyze_rate.py`, `make_rate_figs.py`, `theory_kappa_bench.py` |
| **#2877 / #2905 Acquisition–reporting decoupling** — [report](../../../docs/experiments/2026-08-07-acquisition-inclusion/REPORT.md) | `launch_acq_incl.sh`, `launch_acq_incl_vg.sh`, `launch_acq_2877.sh`, `launch_acq_region_2905.sh`, `analyze_acq.py`, `selftest_analyze_acq.py`, `per_env_acq_2877.py`, `status_acq_2877.sh`, `bump_acq_2877_throttle.sh`, `probe_acq_divergence.sh` |
| **#3319 Acquisition-offset frontier** (past −4, half steps, the deep regime; shipped −4) — [report](../../../docs/experiments/2026-08-07-acquisition-inclusion/REPORT_3319.md) | `launch_acq_3319.sh`, `frontier_3319.py` — and the #2877 row's analyzer, whose arm table this study widens through `ACQ_ANALYZE_ARMS` rather than forking |
| **#3547 Deep-session acquisition offset** (does the optimum move DEEPER through a session; runs on the `vg_scale_deep` pile) — [plan](../../../docs/experiments/2026-08-07-acquisition-inclusion/PLAN_3547.md) | `launch_acq_3547.sh`, `frontier_3547.py`, `check_prefix_3547.py`, `harvest_3547.py`, `spike_timing_3547.py`, `spike_examples_3547.py`, `figures_3547.py` — and the #2877 row's analyzer. The first is the evidence that a 400-step trajectory strictly EXTENDS the 100-step one, which is why this study runs ONE wave and reads both horizons off it; the second is the positives-exhaustion measurement on #3319's cells that motivated the deeper pile. the spike-examples script prints the literal trajectory rows around a cell's first deep spike — the evidence that `n_good` sits frozen through an exhaustion spike; the figures script draws the report's five figures from the CSVs the frontier analyzer's `--csv` writes, so the tables and the pictures cannot drift apart |
| **#2847 Do the MLP-era spikes survive?** — [report](../../../docs/experiments/2026-08-07-spike-check-2847/REPORT.md) | `launch_spike_2847.sh`, `analyze_spikes.py`, `selftest_analyze_spikes.py` |
| **#2897 Fold count** — [report](../../../docs/experiments/2026-08-12-calibration-fold-count/REPORT.md) | `launch_folds_2897.sh`, `launch_folds_2897_ab.sh`, `chain_folds_2897_ab.sh`, `status_folds_2897.sh`, `analyze_folds_2897.py`, `selftest_analyze_folds_2897.py` |
| **Overview benchmark** (production defaults across the pile) — [report](../../../docs/experiments/2026-08-12-overview-bench/REPORT.md) | `launch_bench.sh`, `launch_errdump.sh`, `launch_horizon.sh`, `analyze_bench.py`, `analyze_bench_interaction.py`, `analyze_horizon.py`, `make_bench_figs.py`, `error_report.py`, `make_error_sheets.py`, `label_noise.py`, `text_baseline.py` |
| **#2808 Linear-head convergence** — [report](../../../docs/experiments/2026-08-19-linhead-convergence-2808/REPORT.md) | `launch_linhead_2808.sh`, `make_linhead_figs.py` (analysis reuses `analyze_spikes.py`) |
| **#2865 Cut rule × Inclusion** — [report](../../../docs/experiments/2026-08-21-inclusion-cut-rule/REPORT.md) | `launch_incl_2865.sh`, `analyze_cutincl.py`, `selftest_analyze_cutincl.py`, `make_cutincl_figs.py` |
| **#2883 Is `transfer` bias or variance?** — [report](../../../docs/experiments/2026-08-24-transfer-2883/REPORT.md) | `launch_transfer_2883.sh`, `analyze_transfer.py`, `selftest_analyze_transfer.py` |
| **#3115 Fold combine rule** — [report](../../../docs/experiments/2026-08-25-calibration-fold-combine/REPORT.md) | `launch_folds_3115.sh`, `folds_combine_3115.py`, `make_folds_3115_figs.py` |
| **#3156 Scale / the descriptive map** — [report](../../../docs/experiments/2026-08-25-vg-scale/REPORT.md) | `launch_scale.sh`, `analyse_all.sh`, `analyze_scale.py`, `analyze_overview.py`, `analyze_phases.py`, `analyze_tail_overlap.py`, `figures_scale.py`, `figures_overview.py`, `figures_trajectory.py`, `pick_sheets.py` |
| **#4051 Size-generalisation matrix** (train at one band, test at all three) — no report yet; [PREREG](../../../docs/experiments/2026-09-20-size-bands-4051/PREREG.md) | `launch_bands.sh`. #3156 answered the band question on the DIAGONAL, because `scale_core._evaluable` excludes an image holding the class at another size from the cell — so the other six squares were not merely unmeasured, they were unaddressable. #4047 made them reachable (`CALIB_TEST_BANDS`); this row is the run. The band contrast is the same paired statistic #3156 already reports, now with an off-diagonal, so the analysis reuses that row's analyzers rather than forking them |
| **#3679 Re-measure of #3156's band effect** on the rebuilt 25-class construction ([report](../../../docs/experiments/2026-09-07-band-remeasure-3679/REPORT.md)) | `measure_bands_3679.py` — emits every number that report quotes as one JSON, from both runs' cells, through the #3156 row's own analyzer rather than a second statistic. It lives here and not beside the report *because* it imports that module: deptry scans `docs/`, so a sibling import from a report directory reads as an undeclared dependency and blocks the suite for every branch (#3745, #3747) |
| **#3287 Calibration fraction** — [report](../../../docs/experiments/2026-08-27-calibration-fraction-3287/REPORT.md) | `launch_calfrac_3287.sh`, `run_3292.sh`, `analyze_calfrac.py`, `selftest_analyze_calfrac.py` |
| **#3267 Good Mining (the Autopilot opening)** — [report](../../../docs/experiments/2026-08-27-good-mining-3267/REPORT.md) | `launch_good_mining.sh`, `analyse_good_mining.sh`, `analyze_startup.py`, `selftest_analyze_startup.py`, `make_startup_sheets.py`, `probe_startup_cuts.py` |
| **#3310 / #3314 Fold count, cost and benefit** — [report](../../../docs/experiments/2026-08-28-calibration-fold-count-3310/REPORT.md) | `launch_folds_3314.sh`, `status_folds_3314.sh`, `analyze_folds_3314.py`, `selftest_analyze_folds_3314.py`, `scratch_folds_3310.py` |
| **#3312 Voted-media exclusion floor** — [plan](../../../docs/experiments/2026-08-28-voted-exclusion-3308/PLAN.md) (no report yet) | `launch_exclusion_3308.sh`, `analyze_exclusion.py`, `selftest_analyze_exclusion.py` |
| **#3196 Inclusion knob under the linear SVM head** — [report](../../../docs/experiments/2026-08-29-inclusion-knob-3196/REPORT.md) | `launch_incl_3196.sh`, `analyze_incl_3196.py`, `selftest_analyze_incl_3196.py`, `figures_incl_3196.py` |
| **#3197 Why the SVM head beats the logistic one** — [report](../../../docs/experiments/2026-09-22-svm-vs-logistic-3197/REPORT.md) (Stage A, analyzers and figures live in [`../svm_vs_logistic/`](../svm_vs_logistic/)) | `launch_svmlog_3197.sh` |
| **#3329 Is the 2-component mixture a good fit?** — [report](../../../docs/experiments/2026-08-30-fit-quality-3329/REPORT.md) | `launch_fitq_3329.sh`, `analyze_fitq_3329.py`, `selftest_analyze_fitq_3329.py`, `figures_fitq_3329.py`, `worked_cell_3329.py` |
| **#3796 Calibration-split noise floor** — [plan](../../../docs/experiments/2026-09-11-calibration-seed-3796/PLAN.md) | `launch_calseed_3796.sh`, `analyze_calseed_3796.py`, `selftest_analyze_calseed_3796.py`. The grid is inverted — the cell seed is HELD and the Train/Calibrate split is redrawn — so the draws are a cell axis (`CALIB_CALIBRATION_SEEDS`) rather than the arm-per-value shape #3287 needed: #3794 put a `calibration_seed` column on every row, and an arm per draw would be twenty arrays whose only difference is already written down |
| **#3557 Sign-dependent (hinge) cut rule** — [plan](../../../docs/experiments/2026-09-22-hinge-tilt-3557/PLAN.md) | `launch_hinge_3557.sh`, `analyze_hinge_3557.py`, `selftest_analyze_hinge_3557.py`, `figures_hinge_3557.py`. Two run-level arms (live rule `mid_tilt` vs `hinge`, `CALIB_LIVE_CUT_RULE`) because the acquisition cut re-cuts the same estimator below the hinge's seam; each arm also carries the paired re-cut frame |
| **#3551 Tuning the retired `rare` / `corridor` blend schedules** — [plan](../../../docs/experiments/2026-09-22-blend-endpoints-3551/PLAN.md) | `launch_blend_3551.sh`, `analyze_blend_3551.py`, `selftest_analyze_blend_3551.py`, `analyze_blend_ab_3551.py`. The schedule is now only the fused cut's fold fallback, so the screen splits every schedule row by `shipped_provenance` into Q1 (the fallback it would ship) and Q2 (a replacement for the fused cut), and gates on the shipped fallback reproducing the live threshold; `ab` runs promoted arms per voting mode (`AB_MODE`) |
| **#4184 The calibration ladder on COCO Better** (one cost curve per rung of the *Hold The Line* deck, for its slide) — [plan](../../../docs/experiments/2026-09-25-progression-4184/PLAN.md), [report](../../../docs/experiments/2026-09-25-progression-4184/REPORT.md) | `launch_progression_4184.sh`, `analyze_progression_4184.py`, `selftest_analyze_progression_4184.py`, and for the #4201 prevalence arm (`CALIB_HAYSTACK_PREVALENCE`) `compare_prevalence_4201.py` (the rungs at two prevalences, paired per cell) and `figure_prevalence_4201.py` (the report's two figures). Seven run-level arms, one per rung; the four retired thresholds are `CALIB_LIVE_THRESHOLD` (`vtscore/eval/live_threshold_rules.py`), because a rung's line picks its own questions and a re-cut of another rung's trajectory would not. The 1% re-run (#4519) and today's app at each preset (#4548, rungs `r8_labels_b025` and `r8_labels_b4`) feed the deck's slide 32, `slides/figs/src/make-follow-suit-fig.py`, from `docs/experiments/2026-10-05-ladder-fbeta-4519/presets/progression_curve.csv` |
| **#4114 The converged logistic head in the loop, on COCO Better** — [report](../../../docs/experiments/2026-09-27-logreg-head-4114/REPORT.md) (analyzers, examples and figures live in [`../svm_vs_logistic/`](../svm_vs_logistic/)) | `launch_logreg_4114.sh`. Arms `svm` / `lrconv` (head `linear_logreg`) / `linear` |

| **#4220 Can a precision-floor promise be kept from the app's votes?** (the #4223 ruling's objective, priced offline) — [report](../../../docs/experiments/2026-09-28-precision-frames-4220/REPORT.md) | `analyze_pframes_4220.py`, `selftest_analyze_pframes_4220.py`, `figure_pframes_4220.py`. Reads the per-cell precision frames `task_NNNN__pframes.npz` that `CALIB_PFRAME_STEPS` makes the cell runner record (test truth, pool, votes, fold held-out votes and haystacks); prices P(y\|score) estimators (in-sample, fold-rank, fold-raw × logistic/isotonic × point/bootstrap lower bound, ± EM prior shift) on recall at a precision floor X and the violation rate, in the same-prevalence and shifted-corpus scenarios |
| **#4222 Stay in TextTop until G Goods, in the low-prevalence world** (does a deeper text opening mine the positives a precision promise needs?) — [report](../../../docs/experiments/2026-09-28-textgood-4222/REPORT.md) | `launch_textgood_4222.sh`, `analyze_textgood_4222.py`, `selftest_analyze_textgood_4222.py`, `figure_textgood_4222.py`. Today's app with the opening's Good round at G = 3 (production) / 6 / 10 / 20, at COCO Better's default 0.44% and positives thinned to 0.1% (`CALIB_TARGET_PREVALENCE`); precision frames on. Reports harvest, calibration positives against the #4220 gate, the #4220 estimator's promises, and AP paired against the G = 3 control, then applies the decision rule fixed in #4222 |
| **#4224 Rank frames for precision-floor studies off the GRID** — [data and schema](../../../docs/experiments/2026-09-28-rank-frames/README.md) | `export_rank_frames.py`. Cuts the #4220/#4222 precision frames down to the ranks of each test corpus's positives, plus the shipped estimator's cuts (as shipped, and with a consistent reference pool) at X = 25/50/75%, so random-verification and audit-sampling studies can run from CSVs in the repo |
| **#4257 How many audit votes does an honest precision-floor promise cost?** — [report](../../../docs/experiments/2026-09-28-random-verification/REPORT.md) | `analyze_random_verification.py`, `selftest_analyze_random_verification.py`. Reads the #4224 rank frames and prices random verification: the user audits a uniform sample of a candidate top k, and a Clopper–Pearson lower bound decides the promise. One-round, sequential-shrinking and stratified rules against the stored estimators and against reading the top K; the audit draws are hypergeometric and the only randomness. Pure pandas/numpy/scipy, runs anywhere in about a minute, and writes the report's tables and figures |
| **#4267 How big should the spot check's candidate be at a low floor?** (the owner's 10% preset, and "the candidate grows as the floor falls") — [report](../../../docs/experiments/2026-09-29-floor-candidate-4267/REPORT.md) | `analyze_floor_candidate_4267.py`, `selftest_analyze_floor_candidate_4267.py`. Reuses the #4257 analyzer's audit simulation on the #4224 rank frames to price one-round and shrinking spot checks from a top 32 to a top 512 at X = 10–50%, and the schedule the owner picked from them (start at the top 128 at 10% and the top 64 at 25%, 5 picks a round, halving to 32) at all five presets, against reading the top 32 yourself. Pure pandas/numpy/scipy, about 30 seconds |
| **#4383 Where should the precision floor's line go on a corpus of any size?** (today's line is a fixed top 32 / 128) — [report](../../../docs/experiments/2026-09-30-line-estimate-4383/REPORT.md) | `analyze_line_estimate_4383.py`, `selftest_analyze_line_estimate_4383.py`, `figure_line_estimate_4383.py`. Reads the #4220 precision frames of the default arm at 0.1 / 0.44 / 5%, resamples each test half to 32, 320, 3,200 items, the whole half and a x10 bootstrap, and prices line rules that see only the corpus's scores, the session's votes and a few uniform audit picks per rank band: today's fixed count and spot check, an adaptive band search, a monotone curve on audits, the vote-anchored mixture and the #4220 posterior with and without a logit recalibration on the audits, and (#4389) the smaller of today's count and the mixture's, the no-vote line. Scores each on the returned set's precision, shortfall, meets-P, recall ÷ oracle, F1 ÷ best F1 and votes, with cluster SEs; ~10 min on 60 CPUs |
| **#3827 What does `_GMM_MAX_SAMPLES` change, and save, at the sizes it binds?** (every bench sort is under the 50k cap; a GUI Find reaches ~250k, a CLI Find 2M+) | `analyze_gmm_cap_3827.py`, `figure_gmm_cap_3827.py` — [report](../../../docs/experiments/2026-10-01-gmm-cap-3827/REPORT.md). Reads the #4383 study's #4220 precision frames (0.1 / 0.44 / 5%), bootstraps each test half and the session pool to 100k, 250k and 2M items (a resample: what it measures is the sampling noise a cap-sized fit adds on the bench's score distribution), and fits the app's own mixtures with the cap patched to 5k, 10k, 25k, 50k (shipped), 100k or none: the no-vote line (`mixture_count`'s `gmm` rule and the shipped `min-fixed-gmm`) and the fold-anchored cut (inclusion 0, and the acquisition offset). Paired against the uncapped fit: share of lines that moved, relative change in the kept count, change in F1 and shortfall with cluster SEs, how far the cut moved (items per million), and seconds per fit. Sharded (`--shard K/N`, then `--merge OUT`) |
| **#4411 Could the line be drawn at an F-beta optimum instead of a precision floor?** (the owner's 2026-10-01 question: let users self-categorize by F-beta) | `analyze_fbeta_line_4411.py`, `figure_fbeta_line_4411.py`. On the #4383 frames, worlds, sizes and draws, at beta = 0.5 / 1 / 2: the mixture's argmax F-beta (`fb-gmm`), the band walk with an F-beta stop and the mixture's positive count (`fb-walk`), every band audited with the count from the audits (`fb-bands`), beside the shipped floor mechanism at the preset the preference maps to (`floor-walk`, `floor-novote`; beta 0.5 -> 90%, 1 -> 50%, 2 -> 10%). Scored as the returned set's F-beta over the best F-beta of any cut, with precision, recall, votes and cluster SEs, paired against the floor's walk. The F-beta optimum needs the corpus's total positives, which the mixture over-counts 5-30x on sparse corpora (#3827); this reads what that costs |
| **#4523 What do Test mode's targets and budgets cost, and do its ranges hold?** (the Test autopilot's stop rule, priced before the view ships) — [report](../../../docs/experiments/2026-10-05-line-test-4523/REPORT.md) | `launch_line_test_4523.sh` (3 worlds x 3 presets, siglip binary, `CALIB_SAVE_TEST_SCORES` + `CALIB_LINE_TEST`), `analyze_line_test_4523.py`, `selftest_analyze_line_test_4523.py`, `figure_line_test_4523.py`. Replays `vtscore.eval.line_test_arm` from each cell's saved withheld-half snapshot over precision widths 0.15 / 0.20 / 0.30 x per-phase budgets 20 / 40 / 80 x several Test seeds (thinning the 5% world's withheld half, reading document frames with no model), checks the replay against the in-run rows, and summarises picks, stops, coverage, widths, estimate errors and verdict agreement with cluster SEs; `pick.csv` names the cheapest grid point that covers at 93%. ~0.2 s per replay |
| **#4560 How hard should the pooled prior pull the bands above the line?** (Lean's farthest-shallow preset under-reads precision) — [report](../../../docs/experiments/2026-10-06-pooled-taper-4560/REPORT.md) | `price_pooled_taper_4560.py`. Swaps `line_test.pooled_weight` for a taper (flat = today, linear, half, weak) and runs the #4523 replay on its saved snapshots with the rest of the command line, so every taper is replayed on the same sessions, seeds and grid point |
| **#4287 Does the shipped opening change what the spot check returns?** (#4282's `g3@top,b4@mid,g20+dry1/16@top` against today's g3) — [report](../../../docs/experiments/2026-09-29-floor-opening-4287/REPORT.md) | `compare_floor_opening_4287.py`. Reads the #4267 analyzer's `best_attempt.csv` for two openings' rank frames (exported from #4222's grids) and adds absolute recall plus paired per-session differences (unchecked top-K precision, oracle recall at X) with SEs. Pure pandas/numpy, seconds |
| **#4256 Calibrate the precision floor on learned-sort votes only?** (does a provenance filter keep the promise) — [report](../../../docs/experiments/2026-09-28-provenance-4256/REPORT.md) | `analyze_provenance_4256.py`, `selftest_analyze_provenance_4256.py`, `figure_provenance_4256.py`. Reads precision frames that name each calibration vote (`fold_cal_vote`) and the cell's pick log, and runs the library estimator with all votes or only those picked at or after the first learned-phase pick, under the shipped and a consistent reference pool; arms from the #4222 launcher under a separate base. `--shard I/N` + `--merge` split the ~16 s-a-cell analysis over an array. Verdict: no - learned-only evidence avoids broken promises only by abstaining, and still breaks 83-100% with a consistent pool |
| **#4582 The Cost-era studies re-scored at F-beta 1/4, 1 and 4** (no new runs: the surviving cells, the committed viewer pages' per-run series, and #3826's per-sort tables) — [report](../../../docs/experiments/2026-10-06-cost-fbeta-rescore-4582/REPORT.md) | `rescore_fbeta_4582.py` (reproduces each report's Δcost with its own pairing and window, then scores F-beta), `figures_4582.py` (the report's three figures) |
| **#4583 The calibration split and fold count under the labels line, at the three presets** (closed-loop: fraction {0.3, 0.5} x count {2, 4} x beta {1/4, 1, 4}, 1% pool; one fold and a region arm added) — [report](../../../docs/experiments/2026-10-07-calsplit-4583/REPORT.md) | `launch_calsplit_4583.sh` (twelve arms, eight cells per array task to stay under the submit cap), `analyze_calsplit_4583.py` (the paired contrasts per beta, and the per-run σ of the Δ-objective #4584 sizes with), `selftest_analyze_calsplit_4583.py` (planted answers), `figures_4583.py` (the report's three figures) |
| **#4584 The objective's σ per cell, by preset** (what `preflight.sh --resolve-delta` sizes a balance A/B with) — [note](../../../docs/experiments/2026-10-07-objective-sigma-4584/README.md) | `sigma_objective_4584.py` (per-cell sd of the paired Δ-objective in `analyze_ab`'s all-steps window) |
| **#3546 What should Autopilot's acquisition cut be?** (today's line - 4 is a rank pin near the 98.5th percentile; eight rules at three presets) — [report](../../../docs/experiments/2026-10-07-acquisition-cut-3546/REPORT.md) | `launch_acqcut_3546.sh` (24 arms: today's cut, at the line, offsets -2 and -6, a 98.5th-percentile pin, the old Inclusion 0 - 4 origin, target pick precision 0.5 and 0.25), `analyze_acqcut_3546.py` (each rule against today's cut: objective, Goods, Hard picks, where the cut sits), `figures_3546.py` |
| **#4222 The dry stop: text sort until G Goods or until it runs dry** (does #4254's adaptive opening beat a fixed G at both 0.44% and 0.1%, and feed #4216's starved small hunts?) — [report](../../../docs/experiments/2026-09-29-drystop-4222/REPORT.md) | `analyze_drystop_4222.py`, `selftest_analyze_drystop_4222.py`, `figure_drystop_4222.py`. Arms `pool/g<G>[d<W>]` from the #4222 launcher above (`<world>-g20d8` = `g20+dry1/8@top,b4@mid`); reports how each opening ended (met G / ran dry / never), Good votes by checkpoint, the share of hunts finding < 3 positives per size band paired against g3, and AP / oracle cost paired against g3 per band (a no-detector session scores AP 0); applies the rule fixed on #4222. Verdict: `g3@top,g20+dry1/16@top,b4@mid` (+0.052 / +0.032 AP at vote 150); a bare dry stop quits early and fails 4-5x more sessions |
| **#4197 Sibling fixation: diversify the opening's text-sort walk?** (the text query surfaces a sibling class - keyboards for "laptop" - that gets voted Bad again and again) — [report](../../../docs/experiments/2026-09-29-opening-diversity-4197/REPORT.md) | `analyze_siblings_4197.py`. Arms from the #4222 launcher (`<world>-div<TT>k<K>` sets `CALIB_OPENING_DIVERSITY=0.TT/K`: pass over text-sort candidates at cosine >= tau to >= k Bads); read with the dry-stop analyzer (AP, final cost, per class) plus this script's share of each phase's Bad clicks that hold the class's sibling. Verdict: not shipped - the fixation is in the opening, and the knob gains ~0.0035 cost |
<!-- END INDEX -->

## Arms

| Dataset | Embedder | Style(s) | Calibration |
|---|---|---|---|
| `visual_genome_m` | `siglip`, `siglip_l` | `whole_image` | row-wise |
| `visual_genome_m` | `dinov3_patch` | `max_patch` | grouped (bag max-pool) |
| `caltech101_m` | `siglip`, `siglip_l` | `whole_image` | row-wise |

Two #2781 arms are **off by default** now that their questions are closed, and a
study that wants either adds it back explicitly (and declares the divergence to
`preflight.sh`):

- `max_patch_pca_hac` (`CALIB_PATCH_STYLES`) — the raw-patch tree geometry. It
  lost the Max-Patch study at the operating point (PR #2749) and #2886 removed
  the tree it delegates to from ingest, so carrying it doubled the GPU cost of
  every patch cell to measure a geometry production does not have.
- `topk` / `pnorm` (`CALIB_REPOOL_VARIANTS`) — the **remedial re-pools** of that
  arm's own per-node scores (`topk` k=4, `pnorm` extreme-value normalisation),
  each with its own recalibrated threshold, tagged in the `pool_variant` column.
  Both failed (`docs/plans/set-scorer-experiment.md`), and every analyzer filters
  `pool_variant` back down to the base rows, so the arms produced rows nothing
  reads.

## Stages

1. **`prepare_data.py`** — ensures a per-`(dataset, embedder)` pickle + exemplar
   crops for every arm. Reuses the Max-Patch pickles/crops where the pair
   coincides (VG×{siglip,dinov3_patch}, Caltech×siglip); only embeds the missing
   `siglip_l` pairs.
2. **`run_cells.py`** — one SLURM-array task per `(dataset, embedder, category,
   seed)` cell; runs every style for the embedder, emitting the calibration
   metrics (`CALIBRATION_COLUMNS`) to `results/cells/task_<idx>.csv` and the
   inclusion sweep (`INCLUSION_SWEEP_COLUMNS`) to `task_<idx>__sweep.csv`.
   `--print-cells` prints the array size. `--print-paired-cells` prints how many
   cells a trajectory A/B of the grid pairs on, one per style, which is what
   `../preflight.sh --resolve-delta` sizes against (#4111).
3. **`analyze.py`** — concatenates the cells, computes the pre-registered
   deliverables, writes `results/summary.json`, `results/agg/*.csv`,
   `results/figures/*.png`, and a `results/REPORT.md` draft.

On the fused threshold path (`CALIB_SAFE_THRESHOLDS`, **on by default** since
#3400 because the app has had no switch since #2799) each step also emits one row
per **cut variant** (`gmm_variant`; `_SAFE_GMM_VARIANTS`) and a per-(step,
geometry) **cut decomposition** frame (`CUT_DIAGNOSTIC_COLUMNS`) to
`task_<idx>__cutdiag.csv`.
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

Every analyzer discovers *and reads* its input through `_cells_io`:
`main_frame_files` / `side_frame_files` for discovery, `load_cells` for the read.
Nothing globs `task_*.csv` itself. A bare glob also matches the side frames
(`__sweep`, `__cutdiag`, `__cutincl`, `__picks`, `__fitq`), which are separate
long-format tables — concatenating one into the main frame yields a ragged
DataFrame whose extra rows enter every aggregate silently.

`main_frame_files` excludes side frames **structurally**, on the `__` in the
stem, not on a list of known suffixes. The list shape is one a human has to
remember to extend and twice did not: `__picks` (#3267) and `__fitq` (#3329)
were both added to `run_cells.py` without it, and `bench_cells.py` — which had
its own private copy of the list — was reading the per-click pick log into four
bench analyzers' metric frames as a result. `SIDE_FRAME_SUFFIXES` survives as
the registry `side_frame_files` reads and as documentation; a meta-test holds it
to what `run_cells.py` actually writes, and holds every script in this directory
to going through `_cells_io`.

`load_cells(cells_dir)` returns `(frame, provenance)` and is the only reader.
Eight analyzers used to have their own, diverging on the three guards a grid run
needs and nothing else — zero-byte skip, unreadable catch, header-only count —
with four of the eight having none of them. The provenance names all three
separately, because they are different facts: a zero-byte or unreadable cell is
data loss, while a header-only cell is a *starved* cell (the simulator emits no
row before one Good and one Bad vote coexist), which is a legitimate result and
the extreme of the regime several of these studies are about. `describe_load`
formats them into one line so two reports mean the same thing by "N of M cells".

`launch_all.sh` points `VTSEARCH_DATA_DIR` at the Max-Patch datadir so the shared
embeddings pickles and demo data are read in place (the `siglip_l` pickles land
alongside them harmlessly), and writes all study output under
`/exp/$USER/calibration`.

## Fixed config (pre-registered)

`inclusion=0` (cost = FPR + FNR), `sim_fraction=0.5`, `calibrate_count=2`,
150 votes, 4 seeds. Env knobs mirror the `MAXPATCH_*` set under the `CALIB_*`
prefix.

#2781 also pre-registered `calibration_fraction=0.5` and the auto-sized MLP
head. Both now follow the app instead: the head is the shipped linear SVM
(`CALIB_HEAD` unset; `CALIB_HEAD=mlp` recovers the historical arm), and the
fraction resolves per space through `production_split_for` (0.3 single-vector,
0.5 patch) unless `CALIB_CALIBRATION_FRACTION` pins it.

`safe_thresholds` was pre-registered `False` here and is **`True` now** (#3400):
#2781 pre-registered the unfused control while it was still a shipped path, and
#2799 removed the switch from the app. A default is only "what a user gets" for
as long as the app agrees, so this one follows the app and a study wanting the
control sets `CALIB_SAFE_THRESHOLDS=0` and declares the divergence.

## Safe-threshold GMM study (issue #2799)

```bash
cd /exp/$USER/projects/vts-calib/scripts/experiments/calibration
bash launch_safe.sh      # safe_thresholds ON, VG only, 30 votes, 8 seeds
```

`launch_safe.sh` re-drives the same pipeline on the fused path (pinned with
`CALIB_SAFE_THRESHOLDS=1`, which is also the default since #3400):
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

`CALIB_CUT_INCL_KS` turns on a side frame (`CUT_INCLUSION_COLUMNS` →
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
pre-registered decision rules: `docs/experiments/2026-08-12-calibration-fold-count/REPORT.md`.

Not to be confused with the older `analyze_folds.py` / `launch_folds_2861.sh`,
which moved the fold count to 4 only to unlock the anchored `qmean`/`qmedian`
combine question — it measures no cost and covers region voting only.

## Supervised skyline / training regret (issue #3322)

```bash
CALIB_SKYLINE_ARMS=skyline_train_full ./launch_cells.sh
```

Splits the frame's `oracle_cost` into a **learnability floor** and the headroom
the interactive loop left on the table:

```
cost = skyline_oracle_cost + training_regret + regret
```

`skyline_train_full` trains the same head, through the same trainer, on the
**entire sim split with full ground-truth labels**, and scores the untouched
test split. That answers the routing question no other column answers: when a
cell is expensive, buy a better *embedder* (high floor) or a better
*acquisition loop* (high `training_regret`)? A stuck run with a low floor was a
findable class the loop missed; one with a high floor was never learnable in
that space.

Add `skyline_test_xfit` for the cross-fitted test-side bracket partner. It is
**never** a naive train-on-test fit — a ~769-parameter head on a test set of
comparable size shatters near-arbitrary labelings and would report `d / n_test`
under the name "learnability" — so it folds the test split and scores each item
with a head that never saw it, the SVM analogue of `honest_test_oracle`.

Nearly free: the skyline is vote-independent, so it costs **one extra fit per
arm per cell**, not one per click. Both arms emit a single `t = 0` row tagged in
`gmm_variant`, and the four decomposition columns
(`skyline_oracle_cost{,_honest}`, `training_regret{,_honest}`) are filled on
every row of the run so the identity holds within a row.

Scoped to the **whole-image** column in v1: a patch column's skyline needs a
supervision decision (GT boxes vs. a multiple-instance problem) that is still
open on #3321, so the harness warns and skips there rather than improvising one.
See [`docs/EVAL.md`](../../../docs/EVAL.md) for the full read, and
`tests_lib/detectors/test_skyline_arm.py` for the telescope, vote-independence
and cross-fitting checks.

## Voted-media exclusion floor (issue #3312)

```bash
cd /exp/$USER/projects/vts-exclusion-3312/scripts/experiments/calibration
python selftest_analyze_exclusion.py          # planted answer; run before the array
bash launch_exclusion_3308.sh prepare         # stage 0, ONCE, shared by every arm
bash launch_exclusion_3308.sh baseline        # the click-0 text-sort anchor
bash launch_exclusion_3308.sh size A 0        # time ONE cell per stage AND per geometry
bash launch_exclusion_3308.sh size B 12
bash launch_exclusion_3308.sh arms            # both stages, then one cross-arm analyze
```

Prices PR #3311: the #3308 exclusion drops the voted media from every haystack
the fold-anchored estimator fits on, and ships behind a floor
(`EXCLUSION_MIN_REMAINDER = 60`) that switches it off when too little of the
collection would be left. **Both numbers behind that floor are synthetic**, which
is what this study exists to fix.

The arm axis is one number — `CALIB_EXCLUDE_VOTED`, the smallest remainder at
which the exclusion still fires — so the arms are ordered and need no sentinel:
`off` (= `inf`, the pre-#3308 baseline), `always` (= 0, no floor), a numeric
floor such as `250`, and **unset**, which resolves through the app's own
`resolve_exclusion_floor` and is therefore the incumbent. Unset is deliberate:
pinning `60` would freeze the arm against a constant that can move underneath
the study.

Two stages, because the two questions live in different regimes, and
`CALIB_SIM_FRACTION` is the instrument that separates them — it sets the
haystack the threshold is fitted on, and therefore the votes-to-haystack ratio
the effect is bounded by. **Stage A** (`sim_fraction=0.5`, 150 clicks) is
production scale, where the remainder never falls below ~1950 and the floor is
inert *by construction*: `always`, the app arm and `f250` are the same estimator
there, so only `off` vs the app arm is a contrast. **Stage B**
(`sim_fraction=0.10`, 380 clicks) drives the remainder 419 → 40, so each arm
switches its exclusion off at a different, known step and a difference is
attributable to the floor rather than to the arm.

These are full runs, not paired re-cuts: the floor sets the threshold, which
sets the acquisition cut, which sets the next vote — the same reason
`calibrate_count` (#2897) and `calibration_fraction` (#3287) each needed live
A/Bs after their screens.

Two validity checks run before any verdict, both reported in `REPORT_exclusion.md`.
The **trap check** asserts that two arms whose floors agree above some remainder
produce *identical* thresholds above it — they are the same estimator there, so
anything under 1.0 means an arm ran under the wrong environment. The **floor
regime** table reconstructs, from `n_remainder` alone, where each arm's
exclusion was actually live, so an arm that never excludes (or always does)
cannot be mistaken for a contrast about the floor.

Preflight gained a check for this study's own design error: a horizon that
outruns its haystack does not fail, it silently *truncates*, which would make
`max_steps` a property of the dataset rather than of the design.

Analyzer: `analyze_exclusion.py`. Design and pre-registered decision rules:
`docs/experiments/2026-08-28-voted-exclusion-3308/PLAN.md`.

## Good Mining: sweeping the Autopilot **opening** (issue #3267)

Getting enough Goods looks like what separates a VTSearch run that works from
one that fails, and the *opening* is where Goods come from. Today it is fixed:
the top of the seed sort until 3 positives, that sort's cutoff until 4
negatives, then the learned Hard sort ever after. (Since #4282 it adds a third
round, the top of the sort again until 20 positives or 16 misses in a row;
this section describes the study as it ran.)

Both of those phases are the **same operation** — a rank-space `hard` select
against a cut drawn on the seed sort — at two different cuts. The Good phase's
`top` select is that select against a cut placed above every score; the Bad
phase's is against the sort's own fitted GMM, split at the production midpoint.
So the opening collapses to a list of rounds, each naming *how many clicks* and
*where on the sort*, which is what `CALIB_STARTUP_SCHEDULE` sweeps.

```bash
GM_STAGE=live bash launch_good_mining.sh    # coco_val + visual_genome_m
GM_STAGE=bands bash launch_good_mining.sh   # vg_box_small/medium/large: cached pickles only (see below)
bash analyse_good_mining.sh                 # once every arm drains: everything
python selftest_analyze_startup.py          # planted-answer check on the analyzer
python selftest_curves.py                   # ...and on the quality-over-clicks pair
```

The `bands` stage runs only where its prepared pickles are still cached: the
`vg_box_*` datasets were unregistered when the Visual Genome machinery was
retired (#4038), so `prepare_data.py` can no longer build them fresh.

`analyse_good_mining.sh` is the whole analysis in one command, because the pieces
have to agree. To re-run only the analyzer, give it the zero-click anchor
yourself — without it the quality curves have no `t=0` and the far right has
nothing to be compared against:

```bash
python text_baseline.py --results "$CALIB_EXP/results" --out "$OUT/text_baseline.csv"
GM_TEXT_BASELINE="$OUT/text_baseline.csv" GM_OUT="$OUT" python analyze_startup.py
```

Grammar (full reference: [`vtscore/eval/startup_schedule.py`](../../../vtscore/eval/startup_schedule.py)):
`<g|b|n><count>[+dry<m>/<w>]@<top|mid|k[-]N|q<frac>>`, comma-separated. `g3`
stays until 3 goods exist, `b4` until 4 bads, `n8` for 8 clicks; `+dry1/8` also
ends a `g` or `n` round once its last 8 picks held fewer than 1 good (#4222's
adaptive stop, e.g. `g20+dry1/8@top`); `@top` cuts above every score,
`@mid` at the shipped GMM midpoint, `@k-3` at that GMM split under inclusion −3,
`@q0.05` at the sort's 5th rank percentile. `g3@top,b4@mid,g20+dry1/16@top` is today's
opening (#4282; `g3@top,b4@mid` before it) and is *required* to reproduce a
default run click for click.

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
  panel per dataset, a line per arm, averaged over every seed and category, with an
  inter-quartile band.
- `cost_vs_clicks_runs__<dataset>.png` — **the individuals**: a panel per arm
  holding every one of that arm's seeds on that dataset as its own line,
  coloured by the category's prevalence.

**Click 0 is the free text sort.** There is no detector at the far left, so each
curve *begins* on what typing the query got for nothing (`text_baseline.py`, per
cell) — its own leftmost point, in its own colour, rather than a rule across the
panel. The far left is what typing was worth, the far right is what clicking was
worth, and how many clicks it took to overtake the query is reported as a number
in `REPORT_startup.md`'s crossover table rather than eyeballed off a crossing.

**Coverage is the denominator.** A starved cell trains no detector and emits no
metric row, so an arm that starves on a third of its grid would otherwise have
its mean computed over the two thirds that worked and look *better* for it. The
mean is dashed wherever it describes fewer than 95% of the arm's cells; only a
solid segment is a level worth quoting.

A coverage strip under the panel draws that fraction outright, but only when the
dashing does not already tell it: on a healthy grid every arm ramps to full
inside the first handful of clicks and then holds a flat 100% line across the
rest of the axis, so the strip is drawn only when a shortfall reaches past
`CURVE_STRIP_SPAN` (default 25%) of the click axis, or when coverage falls back
after reaching full. Suppressed, the panel title names the click from which
every arm is fully measured. `coverage` is a column of the emitted CSV either
way; set `CURVE_STRIP_SPAN=0` to get the strip unconditionally.

### The interactive viewer

`analyze_startup.py` also writes `viewer.html` — a single self-contained page
carrying **every** slice of the run, which the report links to. The PNGs above
answer the questions the analyzer asked; the viewer is what lets a reader ask
their own without a re-run.

| Control | Choices |
|---|---|
| dataset | one, all (averaged), or each (its own line or panel) |
| category | one, all (averaged), or each |
| embedder | any **non-empty** subset — **one panel each, never averaged** |
| arms | any **non-empty** subset |
| seeds | averaged, or every seed its own line |
| metric | cost, precision, recall, F1, the objective (F-beta at the run's balance; the page opens on it when the run drew its line at one, #4584), FPR, FNR, average precision, AUROC |
| draw › oracle threshold | off (default), or the cheating-threshold line dotted beside the solid performance line |
| draw › overlay on one chart | off (default), one chart per varying dimension with its ±1 SD shadow; on, all of them on one chart in distinct hues, shadows off |

**Overlay is the shadow/comparison trade, made explicit.** Off, every varying
dimension becomes its own chart holding exactly one bold line, so the shaded
spread underneath it is readable and colour carries no meaning. On, they all
land on one chart in distinct hues with the shadows dropped — because two
translucent bands over one another are a third shape nobody can read the overlap
of. Embedders overlay like anything else: the ban that keeps them out of one
*number* is a ban on pooling, and two lines pool nothing. Past the palette's 8
contrast-checked slots hues are generated on the golden angle, and the page says
when that happened.

**Four reference quantities, drawn as what each one is.** Two lines — the
performance the loop achieved (solid) and the same model at the **oracle
threshold** (dotted, same hue, behind the checkbox; the gap between them is the
calibration regret) — and two points, notched into the margins: the free **text
sort** at the left and the supervised **skyline** (issue #3322) at the right.
Nothing joins the text-sort notch to the curve: no detector exists in between, so
the gap is the honest drawing. The notches are marks in the margin rather than
rules across the panel because each is a level that holds at one x, and a rule
would claim it holds at every x — which for the skyline would read as "the
learnability floor was reachable at click 3".

The oracle is offered on every metric that is a statement about **one cut**.
The harness emits only the oracle cut's cost and its FPR/FNR, so precision,
recall and F1 there are reconstructed from those rates and the split's class
counts — the same confusion matrix in a different unit. `average_precision` and
`auroc` integrate over every threshold, so re-cutting cannot move them; the box
disables itself and says so rather than drawing one line twice.

Both chip controls are non-empty by construction — the last remaining chip is
locked, with a tooltip, rather than snapping silently back — because an empty
selection has no honest rendering: the page would either go blank or fall back
to "all", and a reader who missed that would take a chart of everything for a
chart of nothing.

The per-seed payload is packed to a byte budget by coarsening the *click* axis,
never by dropping runs or metrics, and the page says which grid it got. Build it
standalone with:

```bash
python viewer.py --results "$CALIB_EXP/results" \
  --arms prod,top_long,easy_med_hard,band_wide,incl_k,incl_k_wide,flat_mid,deep_first \
  --baseline "$OUT/text_baseline.csv" --out "$OUT/viewer.html"
python selftest_viewer.py     # planted-answer check on the codec and the pooling
```

The skyline notch needs skyline rows, so it appears only for a run launched with
`CALIB_SKYLINE_ARMS` (see [Supervised skyline / training regret](#supervised-skyline--training-regret-issue-3322)
above). `viewer.py` reads them straight out of the cell CSVs, because
`analyze_spikes.load_arm` — which every analyzer goes through — filters
`gmm_variant`-tagged rows out by design, and a skyline reads ground-truth labels
the app can never see. A run without them still builds; the page just has no
floor, and the builder says so. `--no-skyline` skips the pass.

A study that finished before #3322 can still get a floor without re-running its
loop. The skyline is vote-independent, so a second, cheaper pass over the same
cells measures the same quantity — and `--skyline-results` reads it from that
pass's results root while the curves keep coming from the original one:

```bash
CALIB_EXP=/expscratch/$USER/<study>-skyline CALIB_SKYLINE_ARMS=skyline_train_full \
  CALIB_VGSCALE_EMBEDDERS=<the whole-image columns> bash launch_scale.sh cells
python viewer.py --results "$ORIGINAL" --arms results=prod \
  --skyline-results /expscratch/$USER/<study>-skyline \
  --baseline "$OUT/text_baseline.csv" --out "$OUT/viewer.html"
```

Point the second root at a pass over the **same cells** — symlink the original
run's `prepare_info.json` and `crops` into it, as `launch_scale.sh size` does,
so the two agree on what each cell is. Rows from a foreign grid are dropped
(they land by `(dataset, embedder, category)`), but rows from the same grid at a
different configuration would be taken at face value. Re-running the loop
instead would work too, and costs more than the floor: it **replaces** the
performance rows the report's tables were read off.

To redraw the PNGs without re-running the analysis:

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
