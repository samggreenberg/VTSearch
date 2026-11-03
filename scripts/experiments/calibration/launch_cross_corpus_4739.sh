#!/usr/bin/env bash
# #4739: price the Coverage Atlas walk on a cross-corpus Find, at every preset, with the two domain-shift flags.
#
#   bash launch_cross_corpus_4739.sh prepare              # stage 0, ONCE for every leg (cpu, reads the pile in place)
#   bash launch_cross_corpus_4739.sh baseline             # the typed query's set at today's lines, per cell, ONCE
#   CC4739_LEG=L bash launch_cross_corpus_4739.sh size [idx] ARM   # time ONE cell of leg L
#   CC4739_LEG=L bash launch_cross_corpus_4739.sh preflight [ARM..]
#   bash launch_cross_corpus_4739.sh plan                 # print each leg's and arm's knobs; submits nothing
#   bash launch_cross_corpus_4739.sh status               # cells written per leg and arm, against the grid size
#
# Each leg's arms are submitted in chunks by drive_chunks.sh (MaxJobCount, #4701), one driver a leg:
#
#   L=screens_equal; CC4739_LEG=$L LAUNCHER=$PWD/launch_cross_corpus_4739.sh \
#     BASE=/expscratch/$USER/cross-corpus-4739/$L JOB_PREFIX=cc4739-$L \
#     RUNGS="$(bash launch_cross_corpus_4739.sh arms)" DRAIN_BY="..." bash drive_chunks.sh
#
# THE ARMS, each at beta 1/4, 1 and 4 (`_b025`, `_b1`, `_b4`), each preset on its own sessions:
#
#   ctl_*     today's app
#   nowalk_*  New takes the Hard pick instead of the Coverage Atlas's walk (`CALIB_NEW_WALK=hard`, #4671's arm),
#             the phase machine unchanged (Span still gates Done, and without the walk it never goes green)
#
# THE LEGS: what the run Finds on beside the withheld half (#4960). The votes, the pool, the atlas and every
# in-corpus column are the same in every leg (the Find corpus is only ever scored), so a leg is a second read of
# one set of trajectories, and the in-corpus columns must match across legs cell for cell.
#
#   screens0       CALIB_FIND_CORPUS=apptitude CALIB_FIND_MIX=0      the withheld half alone: the flags' no-shift read
#   screens_equal  CALIB_FIND_CORPUS=apptitude CALIB_FIND_MIX=equal  the withheld half plus Apptitude's screens
#                                                                     (#4356) as extra haystack; the screens are
#                                                                     3,141, fewer than the half's ~11,400, so the
#                                                                     draw takes all of them (a share of ~0.28)
#   screens_only   CALIB_FIND_CORPUS=apptitude CALIB_FIND_MIX=only   the screens alone: no positive, so F-beta is NaN
#                                                                     and the read is the count returned among them
#   o365           CALIB_FIND_CORPUS=o365_better                     o365_better's cell of the same name (107 pair,
#                                                                     #4815); waits on #4817's build
#
# The flags (`domain_shift_report` against the sim pool's atlas, `evidence_coverage_report` from the votes) are
# read at CALIB_FIND_FLAGS_AT, the clicks the analyzer reads points at.
#
# Everything else is today's production, on #4671's bench: COCO Better, binary SigLIP, 144 cells x 5 seeds,
# 150 votes, the user's pool at 1% (the withheld half at 0.44%). Every leg and arm runs on one frozen commit, so
# the control pairs with the test arm on (category, seed) and the legs share their trajectories.
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- NOTHING WAS SUBMITTED" >&2' ERR

MODE="${1:-}"

export VTS_REPO="${VTS_REPO:-/expscratch/$USER/worktrees/vts-4739-run}"
WT="$VTS_REPO"
HERE="$WT/scripts/experiments/calibration"
ROOT="${CC4739_ROOT:-/expscratch/$USER/cross-corpus-4739}"
LEGS="screens0 screens_equal screens_only o365"
LEG="${CC4739_LEG:-}"
BASE="$ROOT/${LEG:-none}"
export CALIB_HAYSTACK_PREVALENCE=0.01

PRESETS="b025 b1 b4"
ALL_ARMS=""
for arm in ctl nowalk; do
  for p in $PRESETS; do ALL_ARMS="$ALL_ARMS ${arm}_$p"; done
done
ALL_ARMS="${ALL_ARMS# }"

# --- science knobs: production everywhere the arm table does not say otherwise ---------------------------
export CALIB_SAFE_THRESHOLDS=1
unset CALIB_HEAD CALIB_BLEND_SCHEDULE CALIB_EXCLUDE_VOTED CALIB_CALIBRATION_SEEDS CALIB_LIVE_CUT_RULE
unset CALIB_CALIBRATION_FRACTION CALIB_CALIBRATE_COUNT CALIB_OPENING_DIVERSITY CALIB_LIVE_THRESHOLD
unset CALIB_ACQ_RANK_PERCENTILE CALIB_ACQ_P_CROSSING CALIB_ACQ_ORIGIN CALIB_ACQ_TARGET_P CALIB_ACQ_INCLUSION_OFFSET
unset CALIB_WALK_SHAPE CALIB_BAND_SHARE CALIB_SPOT_CHECK CALIB_SIGMA_FLOOR CALIB_MORE_WALK
unset CALIB_STARTUP_SCHEDULE CALIB_SAVE_TEST_SCORES CALIB_SAVE_TEST_SCORES_AT
export CALIB_SCHEDULE_VARIANTS=""
export CALIB_REPOOL_VARIANTS=""
export CALIB_FOLD_COUNTS=""
export CALIB_ANCHORED=0
export CALIB_CUT_INCL_KS=""

# --- environment (#4668's bench) ---------------------------------------------------------------------------
export CALIB_DATASETS=coco_better
export CALIB_COCO_BETTER_EMBEDDERS=siglip
export CALIB_CATEGORY_MODE=all
export CALIB_PATCH_STYLES=max_patch
export CALIB_REQUIRE_OPENING=text
export CALIB_REQUIRE_SEED_QUERY=1

# --- the Find corpus (#4960) -------------------------------------------------------------------------------
# Exported by leg_env only: experiment_config refuses CALIB_FIND_FLAGS_AT without a corpus, and prepare and the
# baseline run with none.
FLAGS_AT="${CALIB_FIND_FLAGS_AT:-10,25,50,100,150}"
unset CALIB_FIND_CORPUS CALIB_FIND_MIX CALIB_FIND_FLAGS_AT
leg_env() {
  export CALIB_FIND_FLAGS_AT="$FLAGS_AT"
  case "$1" in
    screens0) export CALIB_FIND_CORPUS=apptitude CALIB_FIND_MIX=0 ;;
    screens_equal) export CALIB_FIND_CORPUS=apptitude CALIB_FIND_MIX=equal ;;
    screens_only) export CALIB_FIND_CORPUS=apptitude CALIB_FIND_MIX=only ;;
    o365) export CALIB_FIND_CORPUS=o365_better ;;
    *) echo "unknown leg '$1'; expected one of: $LEGS" >&2; exit 2 ;;
  esac
  return 0
}

# --- sizing: 720 paired runs an arm resolve ~0.010 / 0.006 / 0.0075 of the objective at 2 SE (#4584) ----------
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-5}"
export CALIB_CELL_ORDER="${CALIB_CELL_ORDER:-seed}"
export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-150}"
export CALIB_CELLS_GZIP="${CALIB_CELLS_GZIP:-1}"

# --- ops ---------------------------------------------------------------------------------------------------
export CALIB_PARTITION=cpu
export CALIB_GRES=none
export CALIB_CPUS=1
export CALIB_MEM="${CALIB_MEM:-4G}"
export CALIB_TIME="${CALIB_TIME:-1:30:00}"
export CALIB_ANALYZE=noop.py
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

# shellcheck disable=SC1091
source "$WT/scripts/experiments/pile/pile_env.sh"

require_jobid() {
  if ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $2 was REFUSED by sbatch (no job id came back)." >&2
    exit 1
  fi
}

require_leg() {
  [[ -n "$LEG" ]] || { echo "ERROR: set CC4739_LEG to one of: $LEGS" >&2; exit 2; }
  leg_env "$LEG"
}

# The run's dir under the leg (or, for the shared prepare, under ROOT).
set_dir() {
  export CALIB_EXP="$1"
  export CALIB_RESULTS="$CALIB_EXP/results"
  mkdir -p "$CALIB_EXP/logs" "$CALIB_RESULTS/cells"
  ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
}

set_exp() {
  require_leg
  set_dir "$BASE/$1"
  export CALIB_JOB_NAME="cc4739-$LEG-$1"
}

link_prepare() {
  local src="$ROOT/prepare/results"
  [[ -f "$src/prepare_info.json" ]] || { echo "ERROR: run '$0 prepare' first" >&2; exit 1; }
  cp -n "$src/prepare_info.json" "$CALIB_RESULTS/"
  [[ -d "$src/crops" && ! -e "$CALIB_RESULTS/crops" ]] && cp -r "$src/crops" "$CALIB_RESULTS/crops"
  return 0
}

# The arm table: each arm names only what differs from today's app.
rung_env() {
  unset CALIB_BETA CALIB_NEW_WALK
  RUNG_DIVERGES=""
  local arm="${1%_*}" preset="${1##*_}"
  case "$preset" in
    b025) export CALIB_BETA=0.25 ;;
    b1) export CALIB_BETA=1 ;;
    b4) export CALIB_BETA=4 ;;
    *) echo "unknown preset in '$1'; expected one of: $ALL_ARMS" >&2; exit 2 ;;
  esac
  case "$arm" in
    ctl) ;;
    nowalk) export CALIB_NEW_WALK=hard ;;
    *) echo "unknown arm '$1'; expected one of: $ALL_ARMS" >&2; exit 2 ;;
  esac
  [[ "$CALIB_BETA" != "1" ]] && RUNG_DIVERGES="$RUNG_DIVERGES,beta"
  [[ -n "${CALIB_NEW_WALK:-}" ]] && RUNG_DIVERGES="$RUNG_DIVERGES,new_walk"
  RUNG_DIVERGES="${RUNG_DIVERGES#,}"
  return 0
}

run_preflight() {
  local div=()
  [[ -n "$RUNG_DIVERGES" ]] && div=(--diverges "$RUNG_DIVERGES")
  bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --need-gb 5 \
    "${div[@]}" --job-name "$CALIB_JOB_NAME" --mem "$CALIB_MEM" --conc 10
}

case "$MODE" in
  arms) echo "$ALL_ARMS" ;;

  legs) echo "$LEGS" ;;

  prepare)
    set_dir "$ROOT/prepare"
    P=$(sbatch --parsable --job-name=cc4739-prep --mem=32G --cpus-per-task=2 \
      --time=1:30:00 --partition=cpu --export=ALL \
      --output="$CALIB_EXP/logs/prepare-%j.out" \
      --wrap="source $WT/scripts/grid/gridenv.sh && $ENVX && cd $HERE && python prepare_data.py")
    require_jobid "$P" "prepare"
    echo "prepare job: $P -> $CALIB_EXP/logs/prepare-$P.out"
    ;;

  size)
    IDX="${2:-0}"
    ARM="${3:?size needs an arm}"
    require_leg
    rung_env "$ARM"
    set_dir "$BASE/sizing-$ARM-$IDX"
    link_prepare
    S=$(sbatch --parsable --job-name="cc4739-size-$LEG-$ARM-$IDX" --mem="$CALIB_MEM" --cpus-per-task=1 \
      --time="$CALIB_TIME" --partition=cpu --export=ALL \
      --output="$CALIB_EXP/logs/size-%j.out" \
      --wrap="source $WT/scripts/grid/gridenv.sh && $ENVX && cd $HERE && /usr/bin/time -v python run_cells.py --index $IDX --outdir $CALIB_RESULTS/cells")
    require_jobid "$S" "size"
    echo "size job: $S (leg $LEG, cell $IDX, $ARM) -> $CALIB_EXP/logs/size-$S.out"
    ;;

  baseline)
    # The typed query's own set at today's per-preset line (#4603), what every arm shows before a detector. The
    # in-corpus read; on the Find corpus the analyzer reads the runs' own typed-query rows instead.
    set_dir "$ROOT/prepare"
    T=$(sbatch --parsable --job-name=cc4739-baseline --mem=32G --cpus-per-task=2 \
      --time=1:00:00 --partition=cpu --export=ALL \
      --output="$CALIB_EXP/logs/baseline-%j.out" \
      --wrap="source $WT/scripts/grid/gridenv.sh && $ENVX && cd $HERE && python text_baseline.py --results $CALIB_RESULTS --out $ROOT/text_baseline.csv")
    require_jobid "$T" "text baseline"
    echo "baseline job: $T -> $ROOT/text_baseline.csv"
    ;;

  preflight)
    shift
    for arm in ${*:-$ALL_ARMS}; do
      (
        rung_env "$arm"
        set_exp "$arm"
        link_prepare
        echo "=== $LEG/$arm: find_corpus=$CALIB_FIND_CORPUS mix=${CALIB_FIND_MIX:-only} diverges=${RUNG_DIVERGES:-none}"
        run_preflight
      ) || exit 1
    done
    ;;

  plan)
    for leg in ${LEG:-$LEGS}; do
      (
        leg_env "$leg"
        for arm in $ALL_ARMS; do
          rung_env "$arm"
          printf '%-14s %-11s beta=%-5s new_walk=%-6s find=%s mix=%-6s flags_at=%s diverges=%s\n' "$leg" "$arm" \
            "$CALIB_BETA" "${CALIB_NEW_WALK:-atlas}" "$CALIB_FIND_CORPUS" "${CALIB_FIND_MIX:-only}" \
            "$CALIB_FIND_FLAGS_AT" "${RUNG_DIVERGES:-none}"
        done
      )
    done
    ;;

  status)
    for leg in ${LEG:-$LEGS}; do
      for arm in $ALL_ARMS; do
        d="$ROOT/$leg/$arm/results/cells"
        [[ -d "$d" ]] || continue
        n="$(find "$d" -maxdepth 1 -name 'task_[0-9][0-9][0-9][0-9].csv*' -size +0 2>/dev/null | wc -l)"
        q="$(squeue -u "$USER" -h -n "cc4739-$leg-$arm" -r -o %i 2>/dev/null | wc -l)"
        printf '%-14s %-11s %5s cells written  %4s tasks queued/running\n' "$leg" "$arm" "$n" "$q"
      done
    done
    ;;

  *)
    echo "usage: [CC4739_LEG=L] $0 {arms|legs|prepare|baseline|size [idx] ARM|preflight [ARM..]|plan|status}" >&2
    exit 2
    ;;
esac
