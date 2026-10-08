#!/usr/bin/env bash
# #4668: the F-beta era as a cumulative build-up, each step's gain in the objective
# at every preset, on COCO Better with the user's pool at 1%.
#
#   bash launch_buildup_4668.sh prepare          # stage 0, ONCE (cpu, reads the pile in place)
#   bash launch_buildup_4668.sh size [idx] RUNG  # time ONE cell
#   bash launch_buildup_4668.sh plan             # print each rung's knobs; submits nothing
#   bash launch_buildup_4668.sh baseline         # the typed query's set at every line, per cell
#   bash launch_buildup_4668.sh rungs [RUNG..]   # submit rung arrays (default: all sixteen)
#   bash launch_buildup_4668.sh status           # cells written per rung, against the grid size
#
# WHAT A RUNG IS.  Each rung adds ONE shipped F-beta-era change to the rung
# before it, so two adjacent rungs differ in exactly that change, and the last
# shipped rung is today's app.  Not the deck's Calibration ladder (#4184, #4519):
# those rules replaced each other, and in F-beta the first of them is a cliff
# (#4582).  The owner picked this build-up on 2026-10-08.
#
#   b1_xcal        cross-calibration on today's folds, asking at its own line, no check
#   b2_labels_*    the labels line at the preset (#4452), its absolute spread floor,
#                  Autopilot asking at line - 4 (the app 10-02..10-07), no check
#   b3_floor_*     + the relative spread floor (#4492)
#   b4_check_*     + the weak-separation check (#4496)
#   b5_app_*       + even-odds asking (#3546, #4632): TODAY'S APP
#   (b6)           + the typed query's per-preset line (#4136, #4603): b5's
#                  sessions re-scored before the hand-over, offline, no rung here
#   b7_walk_*      + the More walk on the detector's top (#4637), PRICED, NOT SHIPPED;
#                  scored with the typed query on screen until Hard
#
# `*` is the preset: b025 / b1 / b4 run CALIB_BETA 0.25 / 1 / 4, each scored at
# its own beta.  b1_xcal runs the Inclusion arm (CALIB_BETA=off), whose line
# ignores the preference, so one set of sessions serves every preset.
#
# Everything else is held at today's production in every rung: the linear-SVM
# head, the SigLIP embedder, the dry-stop opening, Autopilot with the Coverage
# Atlas, the 70/30 split (a null under the labels line, #4583), the calibration
# draw.  #4490's Bads-KDE line changes no session, so it is not a rung.
#
# EVERY RUNG IS ITS OWN CALIB_EXP: a rung's line decides what Autopilot asks, so
# a rung is a trajectory.  The rungs pair on (category, seed).
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- NOTHING WAS SUBMITTED" >&2' ERR

MODE="${1:-}"

# Cells import from a FROZEN worktree at the launch commit (frozen-run lesson,
# #4534): the branch's own worktree keeps moving while the arrays run.
export VTS_REPO="${VTS_REPO:-/expscratch/$USER/worktrees/vts-buildup-4668-run}"
WT="$VTS_REPO"
HERE="$WT/scripts/experiments/calibration"
BASE="${BUILDUP_BASE:-/expscratch/$USER/buildup-4668}"
# The user's pool at 1% positive, as #4519 / #4548 / #4583 / #3546; the withheld
# half Find searches stays at the bench's own 0.44%.
export CALIB_HAYSTACK_PREVALENCE=0.01

PRESETS="b025 b1 b4"
ALL_RUNGS="b1_xcal"
for step in labels floor check app walk; do
  for p in $PRESETS; do ALL_RUNGS="$ALL_RUNGS b_${step}_$p"; done
done
# Number the steps the way the issue does (b2..b7); the dir names carry them.
ALL_RUNGS="$(sed -e 's/b_labels_/b2_labels_/g; s/b_floor_/b3_floor_/g; s/b_check_/b4_check_/g; s/b_app_/b5_app_/g; s/b_walk_/b7_walk_/g' <<<"$ALL_RUNGS")"

# --- science knobs -------------------------------------------------------------
# The fused path must be on: b1's retired rule replaces its cut and reads the
# fold models it computes (`live_threshold_rules`).
export CALIB_SAFE_THRESHOLDS=1
unset CALIB_HEAD CALIB_BLEND_SCHEDULE CALIB_EXCLUDE_VOTED CALIB_CALIBRATION_SEEDS CALIB_LIVE_CUT_RULE
unset CALIB_CALIBRATION_FRACTION CALIB_CALIBRATE_COUNT CALIB_STARTUP_SCHEDULE CALIB_OPENING_DIVERSITY
unset CALIB_ACQ_RANK_PERCENTILE CALIB_ACQ_P_CROSSING CALIB_ACQ_ORIGIN CALIB_WALK_SHAPE CALIB_BAND_SHARE
export CALIB_SCHEDULE_VARIANTS=""
export CALIB_REPOOL_VARIANTS=""
export CALIB_FOLD_COUNTS=""
export CALIB_ANCHORED=0
export CALIB_CUT_INCL_KS=""

# --- environment (as #4519, so the rungs read beside its ladder) ----------------
export CALIB_DATASETS=coco_better
export CALIB_COCO_BETTER_EMBEDDERS=siglip
export CALIB_CATEGORY_MODE=all
export CALIB_PATCH_STYLES=max_patch
export CALIB_REQUIRE_OPENING=text
export CALIB_REQUIRE_SEED_QUERY=1

# --- sizing --------------------------------------------------------------------
# 144 cells x 5 seeds = 720 paired runs a rung.  With the objective's sigma
# (#4584: 0.13 / 0.08 / 0.10 at beta 1/4, 1, 4) a step resolves about
# 0.010 / 0.006 / 0.0075 at 2 SE.
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-5}"
export CALIB_CELL_ORDER="${CALIB_CELL_ORDER:-seed}"
export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-150}"
export CALIB_CELLS_GZIP="${CALIB_CELLS_GZIP:-1}"

# --- ops -----------------------------------------------------------------------
export CALIB_PARTITION=cpu
export CALIB_GRES=none
export CALIB_CPUS=1
export CALIB_MEM="${CALIB_MEM:-4G}"
export CALIB_TIME="${CALIB_TIME:-1:30:00}"
# 16 rungs x %6 = 96 tasks = 192 charged CPUs of the 240-CPU per-user cap, so
# every rung advances seed by seed together and ~48 CPUs stay free for peers.
export CALIB_CONC="${CALIB_CONC:-6}"
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

set_exp() {
  export CALIB_EXP="$BASE/$1"
  export CALIB_RESULTS="$CALIB_EXP/results"
  export CALIB_JOB_NAME="bu4668-$1"
  mkdir -p "$CALIB_EXP/logs" "$CALIB_RESULTS/cells"
  ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
}

link_prepare() {
  local src="$BASE/prepare/results"
  [[ -f "$src/prepare_info.json" ]] || { echo "ERROR: run '$0 prepare' first" >&2; exit 1; }
  cp -n "$src/prepare_info.json" "$CALIB_RESULTS/"
  [[ -d "$src/crops" && ! -e "$CALIB_RESULTS/crops" ]] && cp -r "$src/crops" "$CALIB_RESULTS/crops"
  return 0
}

# The rung table.  Each rung names only what differs from today's app; anything
# not set here stays unset = production.
rung_env() {
  unset CALIB_BETA CALIB_LIVE_THRESHOLD CALIB_ACQ_INCLUSION_OFFSET CALIB_ACQ_TARGET_P
  unset CALIB_SPOT_CHECK CALIB_SIGMA_FLOOR CALIB_MORE_WALK
  RUNG_DIVERGES=""
  local step="${1%_*}" preset="${1##*_}"
  case "$1" in
    b1_xcal)
      # Iteration 1's line ("Grade My Own Homework") on the Inclusion arm, which
      # draws no preference line; its acquisition aims at the reporting cut.
      export CALIB_BETA=off CALIB_LIVE_THRESHOLD=xcal_mincost CALIB_ACQ_INCLUSION_OFFSET=0 CALIB_SPOT_CHECK=off
      ;;
    b[23457]_*_b025 | b[23457]_*_b1 | b[23457]_*_b4)
      case "$preset" in
        b025) export CALIB_BETA=0.25 ;;
        b1) export CALIB_BETA=1 ;;
        b4) export CALIB_BETA=4 ;;
      esac
      case "$step" in
        b2_labels) export CALIB_SIGMA_FLOOR=absolute CALIB_ACQ_TARGET_P=off CALIB_SPOT_CHECK=off ;;
        b3_floor) export CALIB_ACQ_TARGET_P=off CALIB_SPOT_CHECK=off ;;
        b4_check) export CALIB_ACQ_TARGET_P=off ;;
        b5_app) ;;
        b7_walk) export CALIB_MORE_WALK=detector ;;
        *) echo "unknown step '$step' in rung '$1'" >&2; exit 2 ;;
      esac
      ;;
    *) echo "unknown rung '$1'; expected one of: $ALL_RUNGS" >&2; exit 2 ;;
  esac
  [[ -n "${CALIB_BETA:-}" && "$CALIB_BETA" != "1" ]] && RUNG_DIVERGES="$RUNG_DIVERGES,beta"
  [[ -n "${CALIB_LIVE_THRESHOLD:-}" ]] && RUNG_DIVERGES="$RUNG_DIVERGES,live_threshold"
  [[ -n "${CALIB_ACQ_INCLUSION_OFFSET:-}" ]] && RUNG_DIVERGES="$RUNG_DIVERGES,acq_offset"
  [[ -n "${CALIB_ACQ_TARGET_P:-}" ]] && RUNG_DIVERGES="$RUNG_DIVERGES,acq_target_p"
  [[ -n "${CALIB_SPOT_CHECK:-}" ]] && RUNG_DIVERGES="$RUNG_DIVERGES,spot_check"
  [[ -n "${CALIB_SIGMA_FLOOR:-}" ]] && RUNG_DIVERGES="$RUNG_DIVERGES,sigma_floor"
  [[ -n "${CALIB_MORE_WALK:-}" ]] && RUNG_DIVERGES="$RUNG_DIVERGES,more_walk"
  RUNG_DIVERGES="${RUNG_DIVERGES#,}"
  return 0
}

run_preflight() {
  [[ -x "$WT/scripts/experiments/preflight.sh" ]] || return 0
  local div=()
  [[ -n "$RUNG_DIVERGES" ]] && div=(--diverges "$RUNG_DIVERGES")
  # Gzipped, sixteen rungs write ~2 GB (11,520 cells x ~190 KB).
  bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --need-gb 5 \
    "${div[@]}" --job-name "$CALIB_JOB_NAME" --mem "$CALIB_MEM" --conc "$CALIB_CONC" || {
    echo "preflight FAILED ($CALIB_JOB_NAME)" >&2
    [[ "${PREFLIGHT_SKIP:-0}" == "1" ]] || exit 1
  }
}

case "$MODE" in
  prepare)
    set_exp prepare
    P=$(sbatch --parsable --job-name=bu4668-prep --mem=32G --cpus-per-task=2 \
      --time=1:30:00 --partition=cpu --export=ALL \
      --output="$CALIB_EXP/logs/prepare-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python prepare_data.py")
    require_jobid "$P" "prepare"
    echo "prepare job: $P -> $CALIB_EXP/logs/prepare-$P.out"
    ;;

  size)
    IDX="${2:-0}"
    RUNG="${3:?size needs a rung}"
    rung_env "$RUNG"
    set_exp "sizing-$RUNG-$IDX"
    link_prepare
    S=$(sbatch --parsable --job-name="bu4668-size-$RUNG-$IDX" --mem="$CALIB_MEM" --cpus-per-task=1 \
      --time="$CALIB_TIME" --partition=cpu --export=ALL \
      --output="$CALIB_EXP/logs/size-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && /usr/bin/time -v python run_cells.py --index $IDX --outdir $CALIB_RESULTS/cells")
    require_jobid "$S" "size"
    echo "size job: $S (cell $IDX, $RUNG) -> $CALIB_EXP/logs/size-$S.out"
    ;;

  baseline)
    # The typed query's own set before any detector shows, at every line the
    # rungs need: the mixture midpoint the app held until #4590, and each
    # preset's line today (#4603).  One file serves every rung.
    # The midpoint file is the same script with the app's own opt-out
    # (VTSEARCH_TEXT_SORT_CUT=gmm_midpoint): its text_* columns are the midpoint.
    set_exp prepare
    for which in today midpoint; do
      rule=""
      [[ "$which" == "midpoint" ]] && rule="export VTSEARCH_TEXT_SORT_CUT=gmm_midpoint && "
      out="$BASE/text_baseline.csv"
      [[ "$which" == "midpoint" ]] && out="$BASE/text_baseline_midpoint.csv"
      T=$(sbatch --parsable --job-name="bu4668-baseline-$which" --mem=32G --cpus-per-task=2 \
        --time=1:00:00 --partition=cpu --export=ALL \
        --output="$CALIB_EXP/logs/baseline-$which-%j.out" \
        --wrap="source $WT/gridenv.sh && $ENVX && ${rule}cd $HERE && python text_baseline.py --results $CALIB_RESULTS --out $out")
      require_jobid "$T" "text baseline ($which)"
      echo "baseline job ($which): $T -> $out"
    done
    ;;

  rungs)
    shift
    RUNGS="${*:-$ALL_RUNGS}"
    for rung in $RUNGS; do
      (
        rung_env "$rung"
        set_exp "$rung"
        link_prepare
        echo "=== $rung: diverges=${RUNG_DIVERGES:-none} -> $CALIB_EXP"
        run_preflight
        bash "$HERE/launch_cells.sh"
      ) || exit 1
    done
    ;;

  plan)
    for rung in $ALL_RUNGS; do
      rung_env "$rung"
      printf '%-16s beta=%-5s live=%-13s acq_off=%-4s acq_tp=%-4s check=%-5s floor=%-9s walk=%-9s diverges=%s\n' \
        "$rung" "${CALIB_BETA:-1}" "${CALIB_LIVE_THRESHOLD:-shipped}" "${CALIB_ACQ_INCLUSION_OFFSET:--4}" \
        "${CALIB_ACQ_TARGET_P:-0.5}" "${CALIB_SPOT_CHECK:-weak}" "${CALIB_SIGMA_FLOOR:-relative}" \
        "${CALIB_MORE_WALK:-seed}" "${RUNG_DIVERGES:-none}"
    done
    ;;

  status)
    for rung in $ALL_RUNGS; do
      set_exp "$rung"
      n="$(find "$CALIB_RESULTS/cells" -maxdepth 1 -name 'task_[0-9][0-9][0-9][0-9].csv*' -size +0 2>/dev/null | wc -l)"
      z="$(find "$CALIB_RESULTS/cells" -maxdepth 1 -name 'task_[0-9][0-9][0-9][0-9].csv*' -size 0 2>/dev/null | wc -l)"
      q="$(squeue -u "$USER" -h -n "$CALIB_JOB_NAME" -r -o %i 2>/dev/null | wc -l)"
      printf '%-16s %5s cells written  %3s zero-byte  %4s tasks queued/running\n' "$rung" "$n" "$z" "$q"
    done
    echo "expected per rung: 144 cells x $CALIB_N_SEEDS seeds = $((144 * CALIB_N_SEEDS))"
    ;;

  *)
    echo "usage: $0 {prepare|size [idx] RUNG|plan|baseline|rungs [RUNG..]|status}" >&2
    exit 2
    ;;
esac
