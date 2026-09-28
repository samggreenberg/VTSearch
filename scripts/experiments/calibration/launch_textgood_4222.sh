#!/usr/bin/env bash
# #4222: "stay in TextTop until G Goods" - does a deeper text opening mine the
# positives a low-prevalence session starves for?
#
#   bash launch_textgood_4222.sh plan               # the arm table; submits nothing
#   bash launch_textgood_4222.sh prepare            # link the #4184 prepare output (once)
#   bash launch_textgood_4222.sh arms [ARM..]       # submit; default: every arm
#   bash launch_textgood_4222.sh status
#
# Today's app (the #4184 r7 rung: shipped cut, 70/30 split, acquisition offset)
# with ONE thing changed: the opening's Good round.  Production opens
# `g3@top,b4@mid` - the top of the typed-query sort until 3 Goods, then its
# midpoint until 4 Bads.  The arms raise G; the control leaves the schedule
# UNSET so it is the app itself, not a re-spelling of it.
#
# Two worlds, both low: COCO Better's natural pool (0.44% positive, ~1 in 230)
# and positives thinned to 0.1% (~1 in 1,000; ~23 positives a cell, ~11 in the
# pool).  No rich arm - the owner's point is the low-prevalence world.  At 0.1%
# a large G cannot be met; the round then walks the text sort for the whole
# session, and that failure is part of what is being measured.
#
# Every cell records precision frames at votes 25/50/100/150, so the #4220
# estimator (fold-rank + logistic + lower bound + EM, gated on calibration
# positives) can be priced on each arm's votes.
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- NOTHING WAS SUBMITTED" >&2' ERR

MODE="${1:-}"
export VTS_REPO="${VTS_REPO:-/expscratch/$USER/worktrees/vts-4222}"
WT="$VTS_REPO"
HERE="$WT/scripts/experiments/calibration"
BASE="${TEXTGOOD_BASE:-/expscratch/$USER/textgood-4222}"
PREPARE_SRC="${PREPARE_SRC:-/expscratch/$USER/progression-4184/prepare/results}"

GOODS="3 6 10 20"
WORLDS="natural p0.001"

# --- science knobs: today's app, as the #4184 r7 rung ran it ---------------------
export CALIB_SAFE_THRESHOLDS=1
unset CALIB_HEAD CALIB_BLEND_SCHEDULE CALIB_EXCLUDE_VOTED CALIB_CALIBRATION_SEEDS CALIB_LIVE_CUT_RULE
unset CALIB_LIVE_THRESHOLD CALIB_CALIBRATION_FRACTION CALIB_ACQ_INCLUSION_OFFSET CALIB_HAYSTACK_PREVALENCE
export CALIB_SCHEDULE_VARIANTS="" CALIB_REPOOL_VARIANTS="" CALIB_FOLD_COUNTS="" CALIB_ANCHORED=0 CALIB_CUT_INCL_KS=""
export CALIB_DATASETS=coco_better CALIB_COCO_BETTER_EMBEDDERS=siglip CALIB_CATEGORY_MODE=all CALIB_PATCH_STYLES=max_patch
export CALIB_REQUIRE_OPENING=text CALIB_REQUIRE_SEED_QUERY=1
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-5}" CALIB_CELL_ORDER="${CALIB_CELL_ORDER:-seed}" CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-150}"
export CALIB_CELLS_GZIP=1
export CALIB_PFRAME_STEPS="${CALIB_PFRAME_STEPS:-25,50,100,150}"

# --- ops -------------------------------------------------------------------------
export CALIB_PARTITION=cpu CALIB_GRES=none CALIB_CPUS=1
export CALIB_MEM="${CALIB_MEM:-4G}" CALIB_TIME="${CALIB_TIME:-1:30:00}"
# 8 arrays x %15 = 120 tasks = the whole 240-CPU per-user limit (2 charged each).
export CALIB_CONC="${CALIB_CONC:-15}"
export CALIB_ANALYZE=noop.py
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# shellcheck disable=SC1091
source "$WT/scripts/experiments/pile/pile_env.sh"

arm_env() {  # arm = <world>-g<G>
  local world="${1%%-*}" g="${1##*-g}"
  unset CALIB_STARTUP_SCHEDULE CALIB_TARGET_PREVALENCE
  DIVERGES=""
  if [[ "$g" != "3" ]]; then
    export CALIB_STARTUP_SCHEDULE="g${g}@top,b4@mid"
    DIVERGES="startup_schedule"
  fi
  case "$world" in
    natural) ;;
    p*) export CALIB_TARGET_PREVALENCE="${world#p}"; DIVERGES="${DIVERGES:+$DIVERGES,}target_prevalence" ;;
    *) echo "unknown world '$world'" >&2; exit 2 ;;
  esac
  export CALIB_EXP="$BASE/$1" CALIB_RESULTS="$BASE/$1/results" CALIB_JOB_NAME="tg4222-$1"
  mkdir -p "$CALIB_EXP/logs" "$CALIB_RESULTS/cells"
}

all_arms() { for w in $WORLDS; do for g in $GOODS; do echo "$w-g$g"; done; done; }

case "$MODE" in
  plan)
    for a in $(all_arms); do
      arm_env "$a"
      printf '%-16s schedule=%-18s target_prevalence=%-8s diverges=%s\n' "$a" \
        "${CALIB_STARTUP_SCHEDULE:-app default}" "${CALIB_TARGET_PREVALENCE:-natural}" "${DIVERGES:-none}"
    done
    ;;
  prepare)
    [[ -f "$PREPARE_SRC/prepare_info.json" ]] || { echo "no prepare at $PREPARE_SRC" >&2; exit 1; }
    for a in $(all_arms); do
      arm_env "$a"
      cp -n "$PREPARE_SRC/prepare_info.json" "$CALIB_RESULTS/"
      [[ -d "$PREPARE_SRC/crops" && ! -e "$CALIB_RESULTS/crops" ]] && cp -r "$PREPARE_SRC/crops" "$CALIB_RESULTS/crops"
    done
    echo "prepare linked into $(all_arms | wc -l) arms under $BASE"
    ;;
  arms)
    shift
    for a in ${*:-$(all_arms)}; do
      (
        arm_env "$a"
        [[ -f "$CALIB_RESULTS/prepare_info.json" ]] || { echo "run '$0 prepare' first" >&2; exit 1; }
        echo "=== $a: schedule=${CALIB_STARTUP_SCHEDULE:-app default} target_prevalence=${CALIB_TARGET_PREVALENCE:-natural}"
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
      q="$(squeue -u "$USER" -h -n "tg4222-$a" -o %i 2>/dev/null | wc -l)"
      printf '%-16s %5s cells written  %3s queued/running jobs\n' "$a" "$n" "$q"
    done
    ;;
  *)
    echo "usage: $0 {plan|prepare|arms [ARM..]|status}" >&2
    exit 2
    ;;
esac
