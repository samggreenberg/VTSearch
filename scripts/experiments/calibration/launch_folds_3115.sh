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
# Cost note: the sweep is nearly free.  Every arm re-cuts the SAME already-
# trained fold prefix, so a combine rule costs arithmetic on cached arrays and
# no fits - and the anchored pair shares one EM per fold (`_fold_count_arms`),
# because `FoldAnchoredCut` reads the combine at cut time.  The marginal cost
# over a plain #2897-shaped run is the extra fold fits Kmax already implies.
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
# Both voting modes, with the premise asserted per CELL rather than per dataset.
# Region voting needs BOTH halves - ground-truth boxes (dataset) and a patch grid
# (embedder) - so `visual_genome_m x siglip` is a BINARY environment however
# boxed its dataset is; that is the trap behind #2877, #2905 and #2897's own
# errata.  `visual_genome_m x dinov3_patch` is the only region arm here, and it
# is where the bag-aware calibrator lives - with a much smaller effective
# calibration set, which is where a combine rule has the most room to matter.
export CALIB_DATASETS="${CALIB_DATASETS:-visual_genome_m,caltech101_m}"
export CALIB_VG_EMBEDDERS="${CALIB_VG_EMBEDDERS:-siglip,dinov3_patch}"
export CALIB_CALTECH_EMBEDDERS="${CALIB_CALTECH_EMBEDDERS:-siglip}"
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

# Concurrency is capped by the `cpu_limit` QOS: cpu=240 with 2 charged per task
# (=120), so asking for more only parks the excess behind your own array.
export CALIB_MEM="${CALIB_MEM:-8G}"
export CALIB_CPUS=1
export CALIB_TIME="${CALIB_TIME:-12:00:00}"
export CALIB_CONC="${CALIB_CONC:-120}"
export CALIB_ANALYZE_MEM="${CALIB_ANALYZE_MEM:-48G}"
export CALIB_ANALYZE_TIME="${CALIB_ANALYZE_TIME:-2:00:00}"

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
    S=$(sbatch --parsable --job-name=f3115-size --mem="$CALIB_MEM" --cpus-per-task="$CALIB_CPUS" \
      --time="$CALIB_TIME" --partition=cpu --export=ALL \
      --output="$LOGS/size-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && time python run_cells.py --index $IDX --outdir $CALIB_EXP/sizing")
    require_jobid "$S" "size"
    echo "size job: $S (cell $IDX)  ->  $LOGS/size-$S.out"
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
    if [[ -x "$WT/scripts/experiments/preflight.sh" ]]; then
      # No `--diverges`: this run pins nothing off-production.  The combine rule
      # is the axis it sweeps and the shipped rule is one of the arms, so even
      # that is a comparison rather than a divergence.
      bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --need-gb 20 \
        --require-region-voting visual_genome_m:dinov3_patch \
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
