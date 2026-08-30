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
#   docs/experiments/2026-08-25-calibration-fold-combine/REPORT.md
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- NOTHING WAS SUBMITTED" >&2' ERR

MODE="${1:-arms}"

export VTS_REPO="${VTS_REPO:-/exp/$USER/projects/vts-folds-3115}"
WT="$VTS_REPO"
HERE="$WT/scripts/experiments/calibration"

# /exp is a shared quota; the fold-count arms emit ~8 arms x 8 counts per step,
# so the cells dir runs to a few GB.  Keep it on the 500G scratch.
# One study, one dir: this is the confound-breaking grid, not the first run's.
export CALIB_EXP="${CALIB_EXP:-/expscratch/$USER/folds-combine-3115-modes}"
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
# The region half is the PAIR `siglip+dinov3_patch` (#3278).  This grid exists
# to SEPARATE the embedder from the voting mode -- `siglip/whole` vs
# `dinov3/whole` is the embedder at fixed mode, `dinov3/whole` vs
# `dinov3/max_patch` is the mode at fixed embedder -- and bare `dinov3_patch`
# would have put a third axis through both of them: DINOv3 has no text tower, so
# every DINOv3 cell (in EITHER style) opens on three random known-goods while
# every SigLIP cell opens on a typed query.  The first contrast would then be
# "embedder AND opening" and the second would be clean, which is worse than both
# being dirty because only one of them looks confounded.  With the pair all four
# corners open on the same SigLIP text sort of the same 12 classes.
export CALIB_VGSCALE_EMBEDDERS="${CALIB_VGSCALE_EMBEDDERS:-siglip,siglip+dinov3_patch}"
export CALIB_REQUIRE_OPENING=text
# A paired arm cannot fall back to the known-good start (`run_cells.py` raises),
# so every selected category must have a typed query.  Selection filters to the
# eligible ones before it picks, replacing rather than dropping.
export CALIB_REQUIRE_SEED_QUERY=1
export CALIB_CATEGORY_MODE="${CALIB_CATEGORY_MODE:-prevalence}"
export CALIB_N_CATEGORIES="${CALIB_N_CATEGORIES:-12}"
# BOTH styles on the patch embedder, which is what breaks the mode-vs-embedder
# confound the first run shipped with.  That run held exactly two cells -
# `siglip x whole_image` (all the "binary" rows) and `dinov3_patch x max_patch`
# (all the "region" rows) - so voting mode and embedder moved together and its
# per-mode headline was equally a per-embedder one.
#
# `dinov3_patch x whole_image` is the missing corner, and it is a genuine BINARY
# cell: under `whole_image` a Bad vote contributes one row rather than ~197, so
# `_flood_context` finds no flooding, `cal_groups` stays None and the row-wise
# calibrator runs.  It costs almost nothing - the style runs inside the existing
# cell on the already-loaded pickle.
#
#   siglip/whole  vs  dinov3/whole      -> the EMBEDDER, at fixed voting mode
#   dinov3/whole  vs  dinov3/max_patch  -> the VOTING MODE, at fixed embedder
#
# (`siglip x max_patch` is the impossible fourth corner: no patch grid.)
export CALIB_PATCH_STYLES="${CALIB_PATCH_STYLES:-whole_image,max_patch}"
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
# --- MEASURED on THIS grid, 2026-08-25, by `size` (jobs 539432/539433) ---
#
#   cell  0  vg_scale_any x siglip        whole_image  20m55s  0.53 GB
#   cell 48  vg_scale_any x dinov3_patch  max_patch     >1h    7.12 GB
#
# Sized on `vg_scale_any`, not carried over from the `visual_genome_m` grid this
# study started on: that one's region cell peaked at 5.52 GB and this one is
# already at 7.12 GB, so reusing the old number would have under-provisioned
# every region cell by 29%.  1.85x the medias (7749 vs 4193) barely moved the
# BINARY cell (20m33s -> 20m55s) but moved the region cell's memory a lot, which
# is why both have to be measured rather than scaled.
#
# The region cell sets both knobs; sizing from the binary cell alone would pick a
# limit the region arm cannot run in (0.53 GB vs 7.12 GB is a 13x difference).
#
# 12G: the region peak is 7.12 GB, held flat across three sstat samples a minute
# apart, so that IS the peak and not a still-climbing reading.  12G leaves 41%
# headroom; an OOM here is a lost cell, not a slow one.
#
# 80: 80 x 12G = 960G of the 1074G per-user allowance under QOS `cpu_limit`
# (89%), just inside preflight check 8's 90% failure line.  An array claiming the
# whole allowance parks your OWN later jobs behind it in QOSMaxMemoryPerUser
# (#3129, three times in one evening).  The array runs binary cells first
# (indices 0-47), so the critical path is a region cell starting once those free
# their slots: ~21m + one region cell.
export CALIB_MEM="${CALIB_MEM:-12G}"
export CALIB_CPUS=1
export CALIB_TIME="${CALIB_TIME:-12:00:00}"
export CALIB_CONC="${CALIB_CONC:-80}"
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
        --require-region-voting vg_scale_any:siglip+dinov3_patch \
        --require-min-positives 100 \
        --contrasts-voting-modes \
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
