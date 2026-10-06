#!/usr/bin/env bash
# #4523: price Test mode's stop rule - the Test arm on the withheld half.
#
#   bash launch_line_test_4523.sh plan            # the arm table; submits nothing
#   bash launch_line_test_4523.sh prepare [ARM..] # link a finished prepare (once per arm)
#   bash launch_line_test_4523.sh arms [ARM..]    # submit; default: every arm
#   bash launch_line_test_4523.sh status
#
# An arm is <world>-<beta>: the world is COCO Better's default pool (natural,
# 0.44% positive), positives thinned to 0.1% (p0.001: test and pool both at the
# target, #4222's knob), or the Train pool's negatives thinned so positives are
# 5% of it (h0.05: the withheld half stays at 0.44%, #4303's knob; the study
# thins the withheld half itself in the replay).  The beta is the app's balance
# preset (b025, b1, b4; #4448): every session runs at that F-beta balance.
#
# Everything else is today's app: SigLIP whole-image binary voting on every
# coco_better class, the text opening, the shipped cut.  Two knobs record what
# the study reads: CALIB_SAVE_TEST_SCORES keeps the withheld half's scores and
# labels at the last ordinary click with the labels' class model (so the Test
# can be replayed over a grid of targets and budgets without retraining), and
# CALIB_LINE_TEST runs the Test arm in-run at the app's default budgets (the
# replay must reproduce those rows exactly, which the analysis checks).
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- NOTHING WAS SUBMITTED" >&2' ERR

MODE="${1:-}"
export VTS_REPO="${VTS_REPO:-/expscratch/$USER/worktrees/vts-4523}"
WT="$VTS_REPO"
HERE="$WT/scripts/experiments/calibration"
BASE="${LINE_TEST_BASE:-/expscratch/$USER/line-test-4523}"
# A finished siglip prepare on this grid: the 2026-10-05 State of the App's (same classes, same pickles).
PREPARE_SRC="${PREPARE_SRC:-/expscratch/$USER/state-of-the-app/2026-10-05-b1/results}"

WORLDS="natural p0.001 h0.05"
BETAS="0.25 1 4"

# --- science knobs: today's app --------------------------------------------------
export CALIB_SAFE_THRESHOLDS=1
unset CALIB_HEAD CALIB_BLEND_SCHEDULE CALIB_EXCLUDE_VOTED CALIB_CALIBRATION_SEEDS CALIB_LIVE_CUT_RULE
unset CALIB_LIVE_THRESHOLD CALIB_CALIBRATION_FRACTION CALIB_ACQ_INCLUSION_OFFSET CALIB_HAYSTACK_PREVALENCE
unset CALIB_OPENING_DIVERSITY CALIB_STARTUP_SCHEDULE CALIB_MIN_PRECISION CALIB_TEST_BANDS CALIB_SPOT_CHECK
unset CALIB_WALK_PICKS CALIB_WALK_TOL CALIB_WALK_FINE CALIB_WALK_GUARD CALIB_WALK_SHAPE
export CALIB_SCHEDULE_VARIANTS="" CALIB_REPOOL_VARIANTS="" CALIB_FOLD_COUNTS="" CALIB_ANCHORED=0 CALIB_CUT_INCL_KS=""
export CALIB_SKYLINE_ARMS=""
export CALIB_DATASETS=coco_better CALIB_COCO_BETTER_EMBEDDERS=siglip CALIB_CATEGORY_MODE=all CALIB_PATCH_STYLES=max_patch
export CALIB_REQUIRE_OPENING=text CALIB_REQUIRE_SEED_QUERY=1
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-3}" CALIB_CELL_ORDER="${CALIB_CELL_ORDER:-seed}" CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-150}"
export CALIB_CELLS_GZIP=1
# What the study reads (see the header).
export CALIB_SAVE_TEST_SCORES=1 CALIB_LINE_TEST=1
export CALIB_RANK_FRAME_STEPS=""

# --- ops -------------------------------------------------------------------------
export CALIB_PARTITION=cpu CALIB_GRES=none CALIB_CPUS=1
export CALIB_MEM="${CALIB_MEM:-4G}" CALIB_TIME="${CALIB_TIME:-1:30:00}"
# 9 arrays x %13 = 117 tasks, 2 CPUs charged each, under the 240-CPU per-user cap.
export CALIB_CONC="${CALIB_CONC:-13}"
export CALIB_ANALYZE=noop.py
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# shellcheck disable=SC1091
source "$WT/gridenv.sh"
# shellcheck disable=SC1091
source "$WT/scripts/experiments/pile/pile_env.sh"

beta_tag() { python3 -c "import sys; b=float(sys.argv[1]); print('b' + (f'{b:g}'.replace('.', '') if b < 1 else f'{b:g}'))" "$1"; }

arm_env() {  # arm = <world>-<betatag>, e.g. natural-b1, p0.001-b025, h0.05-b4
  local world="${1%-*}" tag="${1##*-}"
  unset CALIB_HAYSTACK_PREVALENCE CALIB_TARGET_PREVALENCE
  DIVERGES=""
  case "$world" in
    natural) ;;
    p*) export CALIB_TARGET_PREVALENCE="${world#p}"; DIVERGES="target_prevalence" ;;
    h*) export CALIB_HAYSTACK_PREVALENCE="${world#h}"; DIVERGES="haystack_prevalence" ;;
    *) echo "unknown world '$world'" >&2; exit 2 ;;
  esac
  case "$tag" in
    b025) export CALIB_BETA=0.25 ;;
    b1) export CALIB_BETA=1 ;;
    b4) export CALIB_BETA=4 ;;
    *) echo "unknown beta tag '$tag'" >&2; exit 2 ;;
  esac
  export CALIB_EXP="$BASE/$1" CALIB_RESULTS="$BASE/$1/results" CALIB_JOB_NAME="lt4523-$1"
  mkdir -p "$CALIB_EXP/logs" "$CALIB_RESULTS/cells"
}

all_arms() { for w in $WORLDS; do for b in $BETAS; do echo "$w-$(beta_tag "$b")"; done; done; }

case "$MODE" in
  plan)
    for a in $(all_arms); do
      arm_env "$a"
      printf '%-14s beta=%-5s target_prevalence=%-8s haystack_prevalence=%-8s diverges=%s\n' "$a" \
        "$CALIB_BETA" "${CALIB_TARGET_PREVALENCE:-natural}" "${CALIB_HAYSTACK_PREVALENCE:-natural}" "${DIVERGES:-none}"
    done
    ;;
  prepare)
    [[ -f "$PREPARE_SRC/prepare_info.json" ]] || { echo "no prepare at $PREPARE_SRC" >&2; exit 1; }
    shift
    for a in ${*:-$(all_arms)}; do
      arm_env "$a"
      cp -n "$PREPARE_SRC/prepare_info.json" "$CALIB_RESULTS/"
      [[ -d "$PREPARE_SRC/crops" && ! -e "$CALIB_RESULTS/crops" ]] && cp -r "$PREPARE_SRC/crops" "$CALIB_RESULTS/crops"
    done
    echo "prepare linked into ${*:-$(all_arms | wc -l) arms} under $BASE"
    ;;
  arms)
    shift
    for a in ${*:-$(all_arms)}; do
      (
        arm_env "$a"
        [[ -f "$CALIB_RESULTS/prepare_info.json" ]] || { echo "run '$0 prepare' first" >&2; exit 1; }
        echo "=== $a: beta=$CALIB_BETA target_prevalence=${CALIB_TARGET_PREVALENCE:-natural} haystack_prevalence=${CALIB_HAYSTACK_PREVALENCE:-natural}"
        div=()
        [[ -n "$DIVERGES" ]] && div=(--diverges "$DIVERGES")
        bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --need-gb 3 "${div[@]}" \
          --job-name "$CALIB_JOB_NAME" --mem "$CALIB_MEM" --conc "$CALIB_CONC" || {
          echo "preflight FAILED ($a)" >&2
          [[ "${PREFLIGHT_SKIP:-0}" == "1" ]] || exit 1
        }
        bash "$HERE/launch_cells.sh"
      ) || exit 1
    done
    ;;
  status)
    for a in $(all_arms); do
      n="$(find "$BASE/$a/results/cells" -maxdepth 1 -name 'task_[0-9][0-9][0-9][0-9].csv*' -size +0 2>/dev/null | wc -l)"
      q="$(squeue -u "$USER" -h -n "lt4523-$a" -o %i 2>/dev/null | wc -l)"
      printf '%-14s %5s cells written  %3s queued/running jobs\n' "$a" "$n" "$q"
    done
    ;;
  *)
    echo "usage: $0 {plan|prepare|arms [ARM..]|status}" >&2
    exit 2
    ;;
esac
