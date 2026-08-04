#!/usr/bin/env bash
# Mix-in schedule study (#2841): how safe should safe-thresholds be?
#
# Two phases, launched separately:
#
#   bash launch_mixin.sh screen      Phase 1. One run on the *production*
#                                    trajectory, scoring every registered
#                                    schedule counterfactually at each step
#                                    (one extra metric row each, tagged
#                                    `schedule`).  Cheap - many schedules for
#                                    the price of one run - but blind to
#                                    acquisition feedback, so it ranks
#                                    candidates rather than deciding.
#
#   bash launch_mixin.sh ab NAME...  Phase 2. One full trajectory per named
#                                    schedule, so the blend's effect on
#                                    Autopilot's Hard pick is included.  This
#                                    is the verdict; the screen only picks who
#                                    gets to run.
#
# Arms: VG x dinov3_patch x max_patch is the production *region-voting* path
# (the one live decisions read); VG x siglip and COCO x siglip/siglip2 x
# whole_image are *binary voting*.  #2841 explicitly allows the two to want
# different curves, so both are carried at full weight.
#
# Everything reuses cached embeddings: the VG pickles come from the Max-Patch
# run, and COCO is assembled from the #2790 sweep cache by build_coco_pickle.py.
# No embedder is ever loaded.
set -uo pipefail

MODE="${1:-screen}"; shift || true

WT="${VTS_REPO:-/exp/$USER/projects/vts-mixin2841}"
HERE="$WT/scripts/experiments/calibration"
export VTS_REPO="$WT"
export CALIB_EXP="${CALIB_EXP:-/exp/$USER/mixin-2841}"
LOGS="$CALIB_EXP/logs"; mkdir -p "$LOGS"

# --- the pre-registered grid (identical across both phases) ---
export CALIB_DATASETS="${CALIB_DATASETS:-visual_genome_m,coco_val}"
export CALIB_VG_EMBEDDERS="${CALIB_VG_EMBEDDERS:-siglip,dinov3_patch}"
export CALIB_COCO_EMBEDDERS="${CALIB_COCO_EMBEDDERS:-siglip,siglip2}"
# Tree-free MaxPatch only: #2749 shipped it and dropped the HAC tree, so the
# HAC styles would measure a path no user is on.
export CALIB_PATCH_STYLES="${CALIB_PATCH_STYLES:-max_patch}"
export CALIB_SAFE_THRESHOLDS=1
export CALIB_HEAD="${CALIB_HEAD:-linear}"
export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-30}"
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-12}"
export CALIB_SWEEP_KS="${CALIB_SWEEP_KS:-0}"

PARTITION="${CALIB_PARTITION:-cpu}"
MEM="${CALIB_MEM:-24G}"
CPUS="${CALIB_CPUS:-4}"
TIME="${CALIB_TIME:-4:00:00}"
CONC="${CALIB_CONC:-60}"

# VTS_REPO must ride along or common.setup_env() imports whichever worktree its
# default points at - the #2799 trap that cost an hour.
BASE_ENV="export VTS_REPO=$WT CALIB_EXP=$CALIB_EXP \
CALIB_DATASETS=$CALIB_DATASETS CALIB_VG_EMBEDDERS=$CALIB_VG_EMBEDDERS \
CALIB_COCO_EMBEDDERS=$CALIB_COCO_EMBEDDERS CALIB_PATCH_STYLES=$CALIB_PATCH_STYLES \
CALIB_SAFE_THRESHOLDS=1 CALIB_HEAD=$CALIB_HEAD CALIB_MAX_STEPS=$CALIB_MAX_STEPS \
CALIB_N_SEEDS=$CALIB_N_SEEDS CALIB_SWEEP_KS=$CALIB_SWEEP_KS"

cell_count() {
  local envx="$1"
  (cd "$HERE" && source "$WT/gridenv.sh" >/dev/null 2>&1; eval "$envx"; \
   python run_cells.py --print-cells 2>/dev/null | tail -1)
}

submit_arm() {
  # submit_arm <results-dir> <extra-env> <job-tag>
  local results="$1" extra="$2" tag="$3"
  local envx="$BASE_ENV CALIB_RESULTS=$results $extra"
  mkdir -p "$results/cells"
  # Every arm writes its own results dir but reads the *one* prepare stage, so
  # the category selection and startup exemplars are identical across arms by
  # construction - otherwise arms would differ by more than their schedule.
  local prepared="$CALIB_EXP/results"
  [[ -f "$prepared/prepare_info.json" ]] || { echo "ERROR: run prepare_data.py first" >&2; return 1; }
  ln -sfn "$prepared/prepare_info.json" "$results/prepare_info.json"
  ln -sfn "$prepared/crops" "$results/crops"
  local n; n=$(cell_count "$envx")
  if ! [[ "$n" =~ ^[0-9]+$ ]] || [[ "$n" -eq 0 ]]; then
    echo "ERROR: no cells for $tag (prepare not run?)" >&2; return 1
  fi
  # Which array indices still need running.  A cell that produced no CSV either
  # never ran or died - this run lost ~950 cells to a transient ENOSPC on the
  # shared /exp volume - so resubmitting only the gaps costs minutes instead of
  # redoing a whole arm.  An arm with no output yet yields the full range.
  local spec missing
  missing=$(cd "$HERE" && python - "$results/cells" "$n" <<'PY'
import sys
from pathlib import Path
cells, n = Path(sys.argv[1]), int(sys.argv[2])
have = {int(p.stem.split("_")[1]) for p in cells.glob("task_*.csv") if "__sweep" not in p.name}
print(",".join(str(i) for i in range(n) if i not in have))
PY
)
  if [[ -z "$missing" ]]; then
    echo "ERROR: arm $tag is already complete ($n/$n cells)" >&2
    return 1
  fi
  spec="$missing"

  # The cluster caps total queued jobs (MaxJobCount), and one arm is >1300 array
  # tasks, so a batch of arms can be refused partway through.  Fail loudly rather
  # than print an empty job id and march on.  sbatch writes an informational
  # "Set partition to cpu" line to *stderr* even on success, so stderr must stay
  # out of the job-id capture - folding it in makes every success look like a
  # failure, which is exactly what silently skipped two arms here.
  local j err
  err=$(mktemp)
  j=$(sbatch --parsable --job-name="mix-$tag" --array="$spec"%"$CONC" \
    --mem="$MEM" --cpus-per-task="$CPUS" --time="$TIME" --partition="$PARTITION" \
    --output="$LOGS/$tag-%A_%a.out" \
    --wrap="source $WT/gridenv.sh && $envx && cd $HERE && python run_cells.py" 2>"$err")
  if [[ -z "$j" || ! "$j" =~ ^[0-9]+$ ]]; then
    echo "ERROR: sbatch refused arm $tag: $(cat "$err")" >&2
    rm -f "$err"
    return 1
  fi
  rm -f "$err"
  echo "$j"
}

case "$MODE" in
  screen)
    RESULTS="$CALIB_EXP/results-screen"
    # The screen lives on the production trajectory and re-cuts every schedule
    # on it; `prod`'s own variant row must reproduce the live blend exactly,
    # which analyze_mixin.py asserts before reporting anything.
    J=$(submit_arm "$RESULTS" "CALIB_SCHEDULE_VARIANTS=all" "screen") || exit 1
    echo "screen array: $J  -> $RESULTS"
    A=$(sbatch --parsable --dependency=afterany:"$J" --job-name=mix-analyze \
      --mem=16G --cpus-per-task=4 --time=0:40:00 --partition=cpu \
      --output="$LOGS/analyze-screen-%j.out" \
      --wrap="source $WT/gridenv.sh && $BASE_ENV CALIB_RESULTS=$RESULTS && cd $HERE && python analyze_mixin.py --mode screen")
    echo "analyze: $A"
    ;;
  ab)
    [[ $# -ge 1 ]] || { echo "usage: launch_mixin.sh ab SCHEDULE [SCHEDULE...]" >&2; exit 1; }
    JOBS=()
    for sched in "$@"; do
      RESULTS="$CALIB_EXP/results-ab/$sched"
      J=$(submit_arm "$RESULTS" "CALIB_BLEND_SCHEDULE=$sched" "ab-$sched") || exit 1
      echo "  arm $sched: $J -> $RESULTS"
      JOBS+=("$J")
    done
    DEP=$(IFS=:; echo "${JOBS[*]}")
    A=$(sbatch --parsable --dependency=afterany:"$DEP" --job-name=mix-analyze-ab \
      --mem=24G --cpus-per-task=4 --time=1:00:00 --partition=cpu \
      --output="$LOGS/analyze-ab-%j.out" \
      --wrap="source $WT/gridenv.sh && $BASE_ENV CALIB_RESULTS=$CALIB_EXP/results-ab && cd $HERE && python analyze_mixin.py --mode ab --arms $(IFS=,; echo "$*")")
    # Chained GRID-side on purpose: a local watcher dies with the VPN, this does not.
    echo "analyze: $A"
    ;;
  *)
    echo "unknown mode '$MODE' (expected: screen | ab)" >&2; exit 1
    ;;
esac
