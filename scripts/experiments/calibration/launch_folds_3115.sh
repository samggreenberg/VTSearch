#!/usr/bin/env bash
# #3115 (+ #3116) fold combine-rule study: pooled or averaged cross-calibration?
#
#   bash launch_folds_3115.sh prepare   # stage 0 (cpu, reads the pile in place)
#   bash launch_folds_3115.sh size      # time ONE cell before committing
#   bash launch_folds_3115.sh arms      # the array + the analysis step
#
# Two functions in this repo disagree about the same empirical fact.
# `threshold_from_fold_orderings` - the path the live app calls - POOLS every
# fold's held-out scores into one bag and takes a single conformal quantile,
# because "all folds' scores live on the same sigmoid scale".
# `FoldAnchoredCut._combined_fold_quantile` takes one cut per fold and averages
# them in QUANTILE space, specifically so no cross-scale averaging of raw cuts
# ever happens - i.e. on the premise that fold scores are NOT comparable.  One
# of those is wrong, and nobody has measured which.
#
# The two issues are one run by construction and must not be split: #3116's
# honest reference and its `sd(threshold)` are what make #3115's result
# readable, and a combine-rule answer read through #2897's broken decomposition
# would reproduce that study's confusion rather than resolve it.
#
# WHY K>=3 IS THE POINT: at production's `calibrate_count=2` the mean and the
# median of two numbers coincide, so the contamination question is structurally
# invisible there.  The grid must reach past it; the analyzer reads its verdict
# from K>=3 only, and treats the K<3 rows as a control that must read zero.
#
# Cost note, MEASURED rather than argued.  Every arm re-cuts the SAME already-
# trained fold prefix, so a combine rule costs arithmetic on cached arrays and
# no fits - and the anchored pair shares one EM per fold (`_fold_count_arms`),
# because `FoldAnchoredCut` reads the combine at cut time.  What the K grid
# actually costs is training Kmax folds once per step (the arms are nested
# prefixes of that one calibration, not K separate ones), which `fold_seconds`
# reports at K=Kmax:
#
#   binary cell: 0.15 s/step of a 20m33s cell   (~2%)
#   region cell: 19.7 s/step of a 1h32m cell    (~31%)
#
# Both cells are dominated by the harness's per-step Autopilot simulation, which
# a #2897-shaped run pays identically.  Note that summing `fold_seconds` ACROSS
# the K grid double-counts - the prefixes share their fold models - so the sum is
# not the cost of the sweep and reading it as one overstates the region cell by
# 3.6x.  Halving Kmax to 8 would save ~14% of a region cell and cost the study
# half its K axis, which is the axis the growth-with-K mechanism lives on.
#
# Design + pre-registered decision rules:
#   docs/experiments/calibration-fold-combine/REPORT.md
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- NOTHING WAS SUBMITTED" >&2' ERR

MODE="${1:-arms}"

export VTS_REPO="${VTS_REPO:-/exp/$USER/projects/vts-folds-3115}"
WT="$VTS_REPO"
HERE="$WT/scripts/experiments/calibration"

# /exp is a shared quota; the fold-count arms emit ~8 arms x 8 counts per step,
# so the cells dir runs to a few GB.  Keep it on the 500G scratch.
export CALIB_EXP="${CALIB_EXP:-/expscratch/$USER/folds-combine-3115}"
export CALIB_RESULTS="${CALIB_RESULTS:-$CALIB_EXP/results}"

# --- science knobs ---
# Safe thresholds are not a switch any more (#2799): fusion is always on, and
# every arm that reads a cut in a fold's own distribution - each `q*` rule, and
# both anchored rules - needs the per-fold haystacks only this path supplies.
export CALIB_SAFE_THRESHOLDS=1
export CALIB_ANALYZE=analyze_folds_2897.py

# CALIB_HEAD is deliberately UNSET: `head=None` resolves to
# `voting_iterations.PRODUCTION_HEAD`, the linear SVM a live detector has
# trained since PR #3198.  #2897's launcher pinned `CALIB_HEAD=linear` because
# the logistic head was production then; carrying that pin forward would measure
# a combine rule on a detector nobody has.  That is the failure preflight check
# 12 exists for.
#
# CALIB_BLEND_SCHEDULE likewise unset: an explicit schedule overrides the app's
# per-mode default (#2841), and it reaches only the RETIRED `blend` arm.

# --- the grid this study exists to sweep ---
# Same K grid as #2897, so the fold-count axis stays directly comparable with
# that report and the combine contrast is measured across an order of magnitude
# of K rather than at one arbitrary count.  The pooling argument predicts a gap
# that GROWS with K - a pooled quantile estimates the quantile of a mixture of K
# half-trained models, and that mixture widens - so a flat curve refutes the
# mechanism whatever the level says.  Only a grid this wide can show that.
export CALIB_FOLD_COUNTS="${CALIB_FOLD_COUNTS:-1,2,3,4,6,8,12,16}"
# The run still LIVES at production's 2, so the trajectory is the one users get
# and every arm is scored against the same votes.
export CALIB_CALIBRATE_COUNT=2

# --- environments ---
# ONE dataset, `vg_scale_any`: #3156's hand-checked scale set with the box-size
# band collapsed away (12 classes x 300 positives against one shared 3900-image
# negative pool, so the evaluable pool is 4200 at 7.1% prevalence, IDENTICAL in
# every cell).
#
# It replaces `visual_genome_m`, which this study started on and which is the
# wrong instrument for a calibration question.  Its selected categories run from
# 25 positives (`banana`) to 1645 (`building`), and the thin end does not merely
# add noise - it produces cells with **no trainable step at all**: the first two
# cells of the first attempt came back as header-only CSVs, `ball` (51 positives)
# among them.  A threshold is a quantile of the calibration set, so a grid whose
# calibration sets differ 60-fold in size is confounding the very axis the study
# reads.  Uniform prevalence also means a difference between two combine rules
# cannot be a prevalence difference wearing a disguise.
#
# Both voting modes come from this ONE dataset, so the mode contrast is no
# longer confounded with the dataset the way #2897's caltech-vs-VG split was.
# The premise is still asserted per CELL, not per dataset: region voting needs
# ground-truth boxes (dataset) AND a patch grid (embedder), so
# `vg_scale_any x siglip` is BINARY and `vg_scale_any x dinov3_patch` is REGION.
#
# All 12 classes, not a scale-stratified subset: the band is exactly what this
# dataset collapsed, so there is no scale axis left to stratify on.  With 12
# categories of identical count, `prevalence` mode returns all of them.
export CALIB_DATASETS="${CALIB_DATASETS:-vg_scale_any}"
export CALIB_VGSCALE_EMBEDDERS="${CALIB_VGSCALE_EMBEDDERS:-siglip,dinov3_patch}"
export CALIB_CATEGORY_MODE="${CALIB_CATEGORY_MODE:-prevalence}"
export CALIB_N_CATEGORIES="${CALIB_N_CATEGORIES:-12}"
export CALIB_PATCH_STYLES="${CALIB_PATCH_STYLES:-max_patch}"
export CALIB_REPOOL_VARIANTS=""

# --- sizing ---
# Long enough to cross the cold start - where the calibration set is a handful
# of scores and the combine rule plausibly matters most - into the regime where
# the quantile is stable and it should not.  The shape across that crossing is
# the deliverable, so it has to be inside the window.
export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-150}"
# 4 seeds is also what makes `sd(threshold)` (#3116) computable at all: it is
# taken ACROSS seeds at a fixed step, so a single-seed run has no instrument.
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-4}"

# The cells train a linear head on cached embeddings: no GPU work anywhere in
# this study, and `launch_cells.sh` defaults to `--partition=gpu --gres=gpu:v100:1`,
# where the `4gpu_tier` QOS would cap the array at 2 concurrent tasks.  Both must
# be named, not just the partition: the flag is dropped rather than passed as
# `--gres=none`, which this cluster's submit filter rewrites and then rejects.
export CALIB_PARTITION=cpu
export CALIB_GRES=none

# Concurrency is capped by the `cpu_limit` QOS: cpu=240 with 2 charged per task
# (=120), so asking for more only parks the excess behind your own array.
# --- MEASURED on this grid, 2026-08-25, by `size` (jobs 538980/538983) ---
#
#   cell   1  visual_genome_m x siglip       whole_image  20m33s   594 MB
#   cell  92  visual_genome_m x dinov3_patch max_patch  1h32m11s  5.52 GB
#
# The region cell is 4.5x the binary one and 10x its memory, so IT sets both
# knobs; sizing from the binary cell alone would have picked a memory limit the
# region arm cannot run in.
#
# 9G x 105 = 945G of the 1074G per-user allowance under QOS `cpu_limit` (88%),
# which is what preflight's check 8 will accept - it fails at 90%, because an
# array claiming the whole allowance parks your OWN later jobs behind it in
# QOSMaxMemoryPerUser (#3129, three times in one evening).
#
# 9G, not 8G: the region cell peaked at 5.7 GB, so 8G leaves 29% headroom on a
# 1.5-hour cell and an OOM there is a lost cell, not a slow one.  Not more than
# 9G either - that memory is *dataset* load, essentially identical for every
# VG x dinov3_patch cell (the same 4193 medias), so the category-to-category
# variance this is buying headroom against is small.
#
# 105, and specifically NOT below ~100: the array runs binary cells first
# (indices 0-91), so the critical path is "a region cell that starts once those
# free their slots" ~= 21m + 92m ~= 113m.  Concurrency at or above 100 keeps
# every region cell inside that one wave; at 90 the last few region cells start
# only after a SECOND wave and the array stretches to ~205m.  Higher than 105
# buys nothing - the first wave is already all binary cells.
export CALIB_MEM="${CALIB_MEM:-9G}"
export CALIB_CPUS=1
export CALIB_TIME="${CALIB_TIME:-12:00:00}"
export CALIB_CONC="${CALIB_CONC:-105}"
export CALIB_ANALYZE_MEM="${CALIB_ANALYZE_MEM:-48G}"
export CALIB_ANALYZE_TIME="${CALIB_ANALYZE_TIME:-2:00:00}"

# Pin the BLAS pools to one thread each.  sklearn's `GaussianMixture` spawns a
# pool per fit, and this study's dominant cost is the anchored arm's per-fold EM
# - summed over the K grid, 52 fits per step at Kmax=16 - so #2883's lesson
# (`lessons/2026-08-24-a-login-node-timing-nearly-cut-an-arm.md`) predicts the
# pool costing more than the arithmetic at these sizes.
#
# MEASURED, and the prediction does NOT hold on a compute node.  Cell 1
# (visual_genome_m x siglip, whole_image, 150 steps) came back at **20m33s**
# unpinned and **20m31s** pinned - two seconds apart, i.e. nothing.  #2883's 36x
# was a LOGIN-node effect, where the box is shared and already oversubscribed;
# a cpu-partition task with `--cpus-per-task=1` does not reproduce it.
#
# The pins stay anyway, on the other half of that lesson: 120 concurrent cells
# each spawning a node-sized pool oversubscribes whatever node they land on,
# which is antisocial whether or not it is slow for us.  They are exported HERE
# rather than inside a mode so `size` and `arms` measure and run under the SAME
# environment - a cell timed with different threading than the array will use is
# a guess with a unit attached, which is the part of the lesson that did apply.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

# Read the pre-embedded pile in place: no re-embed, no GPU, no model download.
source "$WT/scripts/experiments/pile/pile_env.sh"

LOGS="$CALIB_EXP/logs"
mkdir -p "$LOGS" "$CALIB_RESULTS/cells" "$CALIB_RESULTS/crops"

ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"

require_jobid() {
  local id="$1" what="$2"
  if ! [[ "$id" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $what was REFUSED by sbatch (no job id came back)." >&2
    echo "       Nothing downstream can run; fix the submission and re-launch." >&2
    exit 1
  fi
}

case "$MODE" in
  prepare)
    # Stage 0 on the CPU partition: every pair is already in the pile, so this
    # loads each pickle, re-derives the selected categories and writes the
    # startup-exemplar vectors.  No model is constructed.
    P=$(sbatch --parsable --job-name=f3115-prep --mem=24G --cpus-per-task=2 \
      --time=1:30:00 --partition=cpu --export=ALL \
      --output="$LOGS/prepare-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python prepare_data.py")
    require_jobid "$P" "prepare"
    echo "prepare job: $P  ->  $LOGS/prepare-$P.out"
    ;;

  size)
    # Time ONE cell before committing to the array.  The Kmax=16 fold budget is
    # ~8x production's calibration work per step and the anchored arm fits an EM
    # per fold, so this run's per-cell cost is NOT #2897's - and quoting a
    # previous grid's seconds is how #3129 produced a 90-minute overestimate.
    IDX="${2:-0}"
    # Sizing cells go in a dir named for the GRID they belong to.  A cell index
    # means something different under a different dataset, so a shared `sizing/`
    # silently mixes two grids' timings under filenames that look comparable -
    # `task_0001.csv` from a 4193-media dataset beside `task_0000.csv` from a
    # 7749-media one, with nothing in either saying so.  This study changed its
    # dataset mid-flight and hit exactly that; preflight's "one study, one
    # results dir" check covers `results/` and not this.
    SIZING="$CALIB_EXP/sizing/${CALIB_DATASETS//,/_}"
    S=$(sbatch --parsable --job-name=f3115-size --mem="$CALIB_MEM" --cpus-per-task="$CALIB_CPUS" \
      --time="$CALIB_TIME" --partition=cpu --export=ALL \
      --output="$LOGS/size-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && time python run_cells.py --index $IDX --outdir $SIZING")
    require_jobid "$S" "size"
    echo "size job: $S (cell $IDX)  ->  $LOGS/size-$S.out   cells -> $SIZING"
    ;;

  arms)
    if [[ ! -f "$CALIB_RESULTS/prepare_info.json" ]]; then
      echo "ERROR: no prepare_info.json at $CALIB_RESULTS - run '$0 prepare' first." >&2
      exit 1
    fi
    # The premise, asserted rather than assumed: `region_voting` is a REQUEST,
    # and a boxed dataset on a single-vector embedder silently runs binary
    # (#2877).  This study reports its verdict per voting mode, so a region arm
    # that is secretly binary would not break anything - it would just answer a
    # different question under the region heading.
    # Preflight's checks are mostly PYTHON: it imports vtscore to assert the
    # region-voting premise and to compare every pinned knob against its shipped
    # constant.  The login shell has no venv on a non-interactive ssh, where the
    # system python is old enough that `X | None` raises at import time - so all
    # three of those checks came back FAIL for a reason that has nothing to do
    # with the run.  Loud, and therefore survivable; the dangerous version of
    # this is preflight reporting `ok` without having looked (#2905).  Activate
    # the venv first so the checks actually run.
    # shellcheck disable=SC1091
    source "$WT/gridenv.sh" >/dev/null 2>&1 || {
      echo "ERROR: could not activate the venv at $WT/gridenv.sh" >&2; exit 1
    }
    if [[ -x "$WT/scripts/experiments/preflight.sh" ]]; then
      # No `--diverges`: this run pins nothing off-production.  The combine rule
      # is the axis it sweeps and the shipped rule is one of the arms, so even
      # that is a comparison rather than a divergence.
      # --require-min-positives is the control for what actually went wrong on
      # the first attempt: `visual_genome_m x siglip x ball` (51 positives) ran
      # to completion and wrote a header-only CSV, which every "N/N cells" count
      # reports as a present cell.  100 is a floor this grid clears by 3x - every
      # class has exactly 300 - so it is a tripwire for a future edit that
      # repoints the run at a thinner dataset, not a constraint on this one.
      bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --need-gb 20 \
        --require-region-voting vg_scale_any:dinov3_patch \
        --require-min-positives 100 \
        --job-name cal-cells --mem "$CALIB_MEM" --conc "$CALIB_CONC" || {
        echo "preflight FAILED" >&2; [[ "${PREFLIGHT_SKIP:-0}" == "1" ]] || exit 1
      }
    fi
    exec bash "$HERE/launch_cells.sh"
    ;;

  *)
    echo "usage: $0 {prepare|size [cell]|arms}" >&2
    exit 2
    ;;
esac
