#!/usr/bin/env bash
# #3551: the `rare` and `corridor` blend schedules were retired on guessed
# endpoints.  Tune them against TODAY's shipped threshold stack.
#
#   bash launch_blend_3551.sh prepare        # stage 0, ONCE (cpu, reads the pile in place)
#   bash launch_blend_3551.sh size [cell]    # time ONE cell before committing
#   bash launch_blend_3551.sh baseline       # the click-0 text anchor
#   bash launch_blend_3551.sh screen         # the tuning screen (one array)
#   bash launch_blend_3551.sh ab ARM [ARM..] # A/B trajectories for promoted arms
#   bash launch_blend_3551.sh analyze        # re-analyse finished cells
#
# WHAT "TODAY'S STACK" MEANS FOR A SCHEDULE.  #2841 tuned the blend when it WAS
# the shipped threshold.  Since #2861/#2863 the shipped threshold is the
# fold-anchored fused cut, unconditionally, and the schedule blend is only its
# FALLBACK - the steps with no usable calibration folds (fewer than two votes of
# a class), where production feeds it NO_GOOD_THRESHOLD as the x-cal side
# (`_fused_threshold`).  So a schedule now answers two separate questions, and
# the screen measures both off one trajectory:
#
#   Q1 fallback: on fallback steps, which schedule should combine the GMM cut
#      with the "admit nothing" sentinel?  Here `rare` behaves nothing like it
#      did in #2841 - a fallback step holds ONE vote of its rarer class (no row
#      exists before both classes do), so every rare ramp with lo >= 1 is pure
#      GMM there - and the corridor clamps the sentinel to a point between the
#      midpoint and the upper component mean.
#   Q2 replacement: on fused steps, would a TUNED blend of the raw x-cal cut and
#      the GMM midpoint beat the fused cut?  #2864 found `cap50` tied fusion on
#      COCO binary and beat it on caltech101, on a stack four threshold changes
#      ago, so on binary voting this is a live question, not a formality.
#
# Every schedule row carries `shipped_provenance`, so the two are never pooled.
# The shipped fallback schedule's own row must reproduce the base row on every
# fallback step bit-for-bit; the analyzer refuses to report otherwise.
#
# Design + pre-registered decision rules:
#   docs/experiments/2026-09-22-blend-endpoints-3551/PLAN.md
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- NOTHING WAS SUBMITTED" >&2' ERR

MODE="${1:-screen}"

export VTS_REPO="${VTS_REPO:-/expscratch/$USER/worktrees/vts-blend-3551}"
WT="$VTS_REPO"
HERE="$WT/scripts/experiments/calibration"
BASE="${BLEND_BASE:-/expscratch/$USER/blend-3551}"

# --- science knobs -----------------------------------------------------------
# The SHIPPED threshold path, named: it is the premise of both questions.
export CALIB_SAFE_THRESHOLDS=1
# CALIB_HEAD unset -> the production linear SVM.  CALIB_CALIBRATION_FRACTION
# unset -> the app's per-space split.  The acquisition offset is left at the
# shipped constant (-4, #3319): ~half of #2841's gain was acquisition feedback,
# and holding it at production is what keeps these arms comparable to #2841's.
# CALIB_BLEND_SCHEDULE unset -> the app's per-mode fallback (screen); the `ab`
# mode sets it per arm.

# The tuning grid.  Parametric names (`family:key=value`) are resolved by
# `blend_schedules.parse_parametric_schedule`; a typo fails the cell.
#   references: the two shipped fallbacks, #2841's baseline ramp, the controls.
REFS="cap50,slow_cap50,prod,cap80,pure_gmm,pure_xcal"
#   rare: #2841's 1->8 was a guess.  Both endpoints swept, over a range wide
#   enough that the best point cannot sit on its edge unnoticed (#2861's lesson:
#   an optimum at the bottom of the grid is a boundary, not an optimum).
RARE=""
for lo in 0 1 2 4; do
  for hi in 4 8 16 32 64; do
    (( hi > lo )) && RARE="$RARE,rare:lo=$lo:hi=$hi"
  done
done
#   rare, capped: `cap50`'s win was SPREAD control; a rare ramp that also keeps a
#   GMM share is the fair head-to-head with it.
for hi in 4 8 16 32; do RARE="$RARE,rare:lo=1:hi=$hi:cap=0.5"; done
for hi in 8 16 32; do RARE="$RARE,rare:lo=1:hi=$hi:cap=0.8"; done
#   corridor: width as a fraction of the way from the GMM cut to each mean.
#   #2841's `corridor` is w=1 (kept by its registry name); w=0 is `pure_gmm`.
CORR="corridor"
for w in 0.05 0.1 0.2 0.3 0.5 0.75; do CORR="$CORR,corridor:w=$w"; done
#   the ramped corridor, discontinuity fixed (it now holds its width past hi).
CORR="$CORR,corridor_ramp"
for w in 0.2 0.5; do CORR="$CORR,corridor_ramp:w=$w"; done
export CALIB_SCHEDULE_VARIANTS="$REFS$RARE,$CORR"

# --- environments ------------------------------------------------------------
# The three classic pile datasets only - none depends on vg_scale, DocMarks or
# coco_quarry.  Per voting mode, never pooled (#2841 measured the modes to want
# different curves, and PRODUCTION_SCHEDULE_BY_MODE is mode-gated):
#   region = {visual_genome_m, coco_val} x siglip+dinov3_patch, max_patch
#   binary = {visual_genome_m, coco_val, caltech101_m} x siglip
# The region arm is the PAIR (#3278): bare dinov3_patch has no text tower and
# would open on three random known-goods while every siglip arm opens on a
# typed query.  caltech101_m is boxless, so it can only be binary.
export CALIB_DATASETS=visual_genome_m,coco_val,caltech101_m
export CALIB_VG_EMBEDDERS=siglip+dinov3_patch,siglip
export CALIB_COCO_EMBEDDERS=siglip+dinov3_patch,siglip
export CALIB_CALTECH_EMBEDDERS=siglip
export CALIB_PATCH_STYLES=max_patch
export CALIB_REPOOL_VARIANTS=
export CALIB_REQUIRE_OPENING=text
export CALIB_REQUIRE_SEED_QUERY=1
# Caltech's prevalence-spread default is 6 categories; 12 gives the third binary
# environment the same order of cells as the other two.
export CALIB_N_CATEGORIES="${CALIB_N_CATEGORIES:-12}"

# --- sizing: the calseed noise floor -----------------------------------------
# #3796: 70% of the cell-to-cell cost variance is the calibration DRAW, the one
# axis studies do not vary, and pairing cannot lift it between trajectories.
# So the draw is a CELL AXIS here (two draws, production's 42 first), which
# makes every contrast an average over splits rather than a statement about
# split 42.  Screen rows are paired within a step (same model, same split, same
# test scores), so the draw does not enter their difference; the A/B arms'
# trajectories diverge, so it does, and the `ab` mode doubles the seeds.
export CALIB_CALIBRATION_SEEDS="${CALIB_CALIBRATION_SEEDS:-42,0}"
export CALIB_CELL_ORDER=calibration_seed
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-4}"
# 150 votes: the fallback lives in the cold start, but the replacement question
# is the #2864 one, whose binary gap tracked positives and needs depth to show.
export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-150}"

# --- ops ---------------------------------------------------------------------
# No GPU: cells train the linear head on cached pile embeddings.  BOTH knobs are
# named (#2897: an empty --gres is rewritten and rejected by the submit filter).
export CALIB_PARTITION=cpu
export CALIB_GRES=none
export CALIB_CPUS=1
# Measured by `size` on THIS pile, 2026-09-22, not carried from #2865:
#   cell   0  visual_genome_m x siglip+dinov3_patch (region)  7m25s  5.7 GB
#   cell 335  caltech101_m x siglip (binary)                  0m46s  1.4 GB
# 8G is 40% over the region peak (an OOM is a LOST cell); 1h is 8x the slowest.
export CALIB_MEM="${CALIB_MEM:-8G}"
export CALIB_TIME="${CALIB_TIME:-1:00:00}"
# The cpu_limit QOS (cpu=240, mem=1.1T, 2 CPUs charged per task) is PER USER and
# two peer sessions share it; %50 x 2 = 100 CPUs and 500G leaves them room.
export CALIB_CONC="${CALIB_CONC:-50}"
# Word-split on purpose by launch_cells.sh (`python $ANALYZE`), which is how the
# chained analyzer gets the click-0 anchor.
export CALIB_ANALYZE="analyze_blend_3551.py --baseline $BASE/text_baseline.csv"
export CALIB_ANALYZE_MEM="${CALIB_ANALYZE_MEM:-64G}"
export CALIB_ANALYZE_TIME="${CALIB_ANALYZE_TIME:-2:00:00}"

export VTS_PILE="${VTS_PILE:-/expscratch/$USER/vts-cache}"
export VTSEARCH_DATA_DIR="$VTS_PILE/datadir"
export VTSEARCH_MODELS_DIR="$VTS_PILE/models"
export HF_HOME="$VTS_PILE/models"

require_jobid() {
  if ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $2 was REFUSED by sbatch (no job id came back)." >&2
    exit 1
  fi
}

set_exp() {
  export CALIB_EXP="$BASE/$1"
  export CALIB_RESULTS="$CALIB_EXP/results"
  export CALIB_JOB_NAME="blend3551-$1"
  mkdir -p "$CALIB_EXP/logs" "$CALIB_RESULTS/cells"
  ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
}

# One prepare serves every grid: the arms differ only in the schedule knobs,
# which prepare never reads.  Each grid gets its own CALIB_EXP (one study, one
# dir) with prepare_info.json and crops copied in.
link_prepare() {
  local src="$BASE/prepare/results"
  [[ -f "$src/prepare_info.json" ]] || { echo "ERROR: run '$0 prepare' first" >&2; exit 1; }
  cp -n "$src/prepare_info.json" "$CALIB_RESULTS/"
  [[ -d "$src/crops" && ! -e "$CALIB_RESULTS/crops" ]] && cp -r "$src/crops" "$CALIB_RESULTS/crops"
  return 0
}

run_preflight() {
  [[ -x "$WT/scripts/experiments/preflight.sh" ]] || return 0
  # Declared divergences: the calibration draw is a cell axis (the calseed
  # sizing above), and an A/B arm pins its fallback schedule on purpose.
  local div="calibration_seed"
  [[ -n "${CALIB_BLEND_SCHEDULE:-}" ]] && div="$div,blend_schedule"
  for arm in visual_genome_m:siglip+dinov3_patch coco_val:siglip+dinov3_patch; do
    bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --need-gb 20 \
      --require-region-voting "$arm" --diverges "$div" \
      --job-name "$CALIB_JOB_NAME" --mem "$CALIB_MEM" --conc "$CALIB_CONC" || {
      echo "preflight FAILED ($arm)" >&2
      [[ "${PREFLIGHT_SKIP:-0}" == "1" ]] || exit 1
    }
  done
}

case "$MODE" in
  prepare)
    set_exp prepare
    P=$(sbatch --parsable --job-name=blend3551-prep --mem=32G --cpus-per-task=2 \
      --time=1:30:00 --partition=cpu --export=ALL \
      --output="$CALIB_EXP/logs/prepare-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python prepare_data.py")
    require_jobid "$P" "prepare"
    echo "prepare job: $P -> $CALIB_EXP/logs/prepare-$P.out"
    ;;

  size)
    set_exp sizing
    link_prepare
    IDX="${2:-0}"
    S=$(sbatch --parsable --job-name="blend3551-size-$IDX" --mem="$CALIB_MEM" --cpus-per-task=1 \
      --time="$CALIB_TIME" --partition=cpu --export=ALL \
      --output="$CALIB_EXP/logs/size-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && /usr/bin/time -v python run_cells.py --index $IDX --outdir $CALIB_EXP/results/cells")
    require_jobid "$S" "size"
    echo "size job: $S (cell $IDX) -> $CALIB_EXP/logs/size-$S.out"
    ;;

  screen)
    set_exp screen
    link_prepare
    run_preflight
    exec bash "$HERE/launch_cells.sh"
    ;;

  ab)
    shift
    [[ $# -ge 1 ]] || { echo "usage: $0 ab ARM [ARM...]" >&2; exit 2; }
    # A/B arms live under the named fallback schedule; nothing is re-cut.
    export CALIB_SCHEDULE_VARIANTS=
    export CALIB_N_SEEDS="${AB_N_SEEDS:-8}"
    for arm in "$@"; do
      (
        set_exp "ab-${arm//[:=]/_}"
        export CALIB_BLEND_SCHEDULE="$arm"
        link_prepare
        run_preflight
        bash "$HERE/launch_cells.sh"
      ) || exit 1
    done
    ;;

  baseline)
    # The click-0 text-sort anchor for the quality-over-clicks figures.  It is
    # draw-independent (no detector exists at click 0), so one file serves
    # every calibration draw; the analyzer replicates it per draw.
    set_exp prepare
    T=$(sbatch --parsable --job-name=blend3551-baseline --mem=32G --cpus-per-task=2 \
      --time=1:00:00 --partition=cpu --export=ALL \
      --output="$BASE/prepare/logs/baseline-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python text_baseline.py --results $BASE/prepare/results --out $BASE/text_baseline.csv")
    require_jobid "$T" "text baseline"
    echo "baseline job: $T -> $BASE/text_baseline.csv"
    ;;

  analyze)
    set_exp "${2:-screen}"
    A=$(sbatch --parsable --job-name="$CALIB_JOB_NAME-analyze" --mem="$CALIB_ANALYZE_MEM" \
      --cpus-per-task=4 --time="$CALIB_ANALYZE_TIME" --partition=cpu --export=ALL \
      --output="$CALIB_EXP/logs/analyze-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python $CALIB_ANALYZE --results $CALIB_RESULTS --out $CALIB_EXP/analysis --baseline $BASE/text_baseline.csv")
    require_jobid "$A" "analyze"
    echo "analyze job: $A"
    ;;

  *)
    echo "usage: $0 {prepare|size [cell]|baseline|screen|ab ARM..|analyze [grid]}" >&2
    exit 2
    ;;
esac
