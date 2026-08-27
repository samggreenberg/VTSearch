#!/usr/bin/env bash
# The scale study (#3156): one class list, three box-size bands.
#
#   bash launch_scale.sh prepare      # exemplar crops for all 36 cells
#   bash launch_scale.sh size 0,40    # time one whole-image and one patch cell
#   bash launch_scale.sh cells        # the full array
#   bash launch_scale.sh status
#
# The question is whether detection cost rises as the target shrinks, and
# whether region voting's advantage depends on target size. `vg_box_*` cannot
# answer it: those sets band each category by its median box, so their
# vocabularies are disjoint and a small-vs-large gap confounds size with class
# identity. Here the twelve classes are held fixed and only the band moves,
# paired on identical negatives at identical prevalence.
#
# Shipped defaults only, as in launch_bench.sh: the contrast under test is the
# BAND, so any other knob left non-default would be a second, uncontrolled one.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"

source "$WT/gridenv.sh"
source "$WT/scripts/experiments/pile/pile_env.sh"

export VTS_REPO="$WT"
export CALIB_EXP="${CALIB_EXP:-/expscratch/$USER/scale-3156}"
export CALIB_RESULTS="${CALIB_RESULTS:-$CALIB_EXP/results}"

export CALIB_DATASETS="${CALIB_DATASETS:-vg_scale}"
# siglip is the shipped default and votes binary; siglip2_l is the premium
# whole-image end; dinov3_patch is the only patch-capable embedder in the pile
# and is what makes region voting real.
#
# All three cells are already embedded in the pile, so a column costs training
# and inference only -- no encoder time. And the encoder here is a BLOCKING
# factor, not the contrast: the question is whether the band effect holds
# across encoders, so a second whole-image encoder replicates the finding
# rather than competing with the first. ("siglip vs siglip2_l is unresolvable
# on cost at three seeds" is a fact about comparing the two encoders, and says
# nothing about whether each one shows the same size penalty.)
export CALIB_VGSCALE_EMBEDDERS="${CALIB_VGSCALE_EMBEDDERS:-siglip,siglip2_l,dinov3_patch}"
# Every category is a designated cell; selecting a subset would discard the
# design, and prevalence-spreading is meaningless when prevalence is 0.0250
# everywhere by construction.
export CALIB_CATEGORY_MODE=all
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-3}"
export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-150}"

export CALIB_REPOOL_VARIANTS=""
export CALIB_SCHEDULE_VARIANTS=""
export CALIB_FOLD_COUNTS=""
export CALIB_PATCH_STYLES="${CALIB_PATCH_STYLES:-max_patch}"

LOGS="$CALIB_EXP/logs"
mkdir -p "$LOGS" "$CALIB_RESULTS/cells"

# Sized from the array that actually ran this grid (job 539790, 324 tasks):
# MaxRSS peaked at 8.73G, typical ~7G, on the dinov3_patch/max_patch cells that
# dominate the memory profile. 64G was a guess carried over from a different
# study and it is 7x the measured peak -- which is not free: memory is the
# binding per-user quota here (cpu_limit, ~1074G), so an oversized --mem caps
# concurrency and parks your own later jobs behind the array in
# QOSMaxMemoryPerUser. 16G keeps ~2x headroom over the measured peak and stays
# above preflight's 12G floor for patch cells.
MEM="${CALIB_MEM:-16G}"
CPUS="${CALIB_CPUS:-6}"
TIME="${CALIB_TIME:-6:00:00}"
PARTITION="${CALIB_PARTITION:-cpu}"
CONC="${CALIB_CONC:-24}"
JOB_NAME="${CALIB_JOB_NAME:-scale-$(basename "$CALIB_EXP")}"

ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS"
ENVX="$ENVX VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
ENVX="$ENVX VTS_REPO=$VTS_REPO CALIB_DATASETS=$CALIB_DATASETS"
ENVX="$ENVX CALIB_VGSCALE_EMBEDDERS=$CALIB_VGSCALE_EMBEDDERS CALIB_CATEGORY_MODE=$CALIB_CATEGORY_MODE"
ENVX="$ENVX CALIB_N_SEEDS=$CALIB_N_SEEDS CALIB_MAX_STEPS=$CALIB_MAX_STEPS"
ENVX="$ENVX CALIB_REPOOL_VARIANTS= CALIB_SCHEDULE_VARIANTS= CALIB_FOLD_COUNTS="
ENVX="$ENVX CALIB_PATCH_STYLES=$CALIB_PATCH_STYLES"

# A submission is not a launch: --parsable returns an EMPTY id when the submit
# filter refuses the job (#2897 lost both arms exactly this way).
submit() {
  local name="$1"; shift
  local J
  J=$(sbatch --parsable "$@") || { echo "SUBMIT FAILED for $name" >&2; return 1; }
  if [[ "$J" =~ ^[0-9]+$ ]]; then
    echo "$J" > "$LOGS/.jobid_$name"
    echo "$name -> job $J"
  else
    echo "$name SUBMIT FAILED (empty job id) — NOT LAUNCHED" >&2
    return 1
  fi
}

case "${1:-status}" in
prepare)
  # The dataset's own review coverage is a precondition, not a detail: a cell
  # whose reviewed images have been rebuilt away is measuring labels nobody
  # checked (see scripts/experiments/lessons/).
  ( cd "$WT/scripts/experiments/pile" && python check_review_coverage.py ) || {
    echo "REVIEW COVERAGE FAILED — refusing to launch a study on it" >&2; exit 3; }
  # Assert the arm the study is FOR actually exists. A patch embedder on a
  # dataset the config forgot to mark boxed runs whole_image without complaint,
  # which is indistinguishable from success in every output.
  ( cd "$HERE" && python - <<'PYCHK' ) || { echo "REGION VOTING IS OFF for the patch arm — refusing to launch" >&2; exit 4; }
import os
import sys
sys.path.insert(0, os.getcwd())
import experiment_config as cfg
ds = cfg.DATASETS[0]
bad = [e for e in cfg.embedders_for_dataset(ds)
       if cfg.is_patch_embedder(e) and not cfg.region_voting_for(ds, e)]
if bad:
    print(f"  {ds}: patch embedders {bad} would run whole_image", file=sys.stderr)
    raise SystemExit(1)
for e in cfg.embedders_for_dataset(ds):
    print(f"  {ds} x {e}: styles={cfg.styles_for(ds, e)} region_voting={cfg.region_voting_for(ds, e)}")
PYCHK
  echo "CALIB_EXP=$CALIB_EXP  datasets=$CALIB_DATASETS  embedders=$CALIB_VGSCALE_EMBEDDERS  seeds=$CALIB_N_SEEDS"
  submit prepare --job-name=scale-prep --mem=96G --cpus-per-task=8 \
    --time=3:00:00 --partition="$PARTITION" --export=ALL \
    --output="$LOGS/prepare-%j.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python prepare_data.py"
  ;;

size)
  IDXS="${2:?usage: launch_scale.sh size <comma-separated cell indices>}"
  SIZE_RESULTS="$CALIB_EXP/sizing"
  mkdir -p "$SIZE_RESULTS/cells"
  ln -sfn "$CALIB_RESULTS/prepare_info.json" "$SIZE_RESULTS/prepare_info.json"
  ln -sfn "$CALIB_RESULTS/crops" "$SIZE_RESULTS/crops"
  for idx in ${IDXS//,/ }; do
    submit "size$idx" --job-name="scale-size$idx" --mem="$MEM" --cpus-per-task="$CPUS" \
      --time=2:00:00 --partition="$PARTITION" --export=ALL \
      --output="$LOGS/size-$idx-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && export CALIB_RESULTS=$SIZE_RESULTS && cd $HERE && time python run_cells.py --index $idx"
  done
  ;;

cells)
  N=$(cd "$HERE" && python run_cells.py --print-cells 2>/dev/null | tail -1)
  if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" -eq 0 ]]; then
    echo "ERROR: could not determine cell count (got '$N')" >&2; exit 1
  fi
  echo "cells: $N (array 0-$((N-1))%$CONC on $PARTITION)"
  PATCH_FLAG=""
  case "$CALIB_VGSCALE_EMBEDDERS" in *_patch*) PATCH_FLAG="--patch" ;; esac
  bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --arms prod \
    --job-name "$JOB_NAME" --mem "$MEM" --conc "$CONC" $PATCH_FLAG || {
    echo "PREFLIGHT FAILED" >&2; exit 2; }
  submit cells --job-name="$JOB_NAME" --array="0-$((N-1))%$CONC" \
    --mem="$MEM" --cpus-per-task="$CPUS" --time="$TIME" \
    --partition="$PARTITION" --export=ALL \
    --output="$LOGS/cells-%A_%a.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python run_cells.py"
  ;;

redo)
  # Re-run a specific set of array indices. A failed task leaves its PREVIOUS
  # output in place, so a partial re-run silently mixes two runs' cells unless
  # the stale files are deleted first -- delete them, then pass the indices
  # here. `--mem` must be sized from a cell of the SAME KIND that failed: a
  # max_patch cell peaks near 9G where a whole_image one peaks under 1G, and
  # sizing the array from the wrong kind is what produced the OOMs this
  # subcommand exists to repair.
  IDXS="${2:?usage: launch_scale.sh redo <comma-separated indices>}"
  echo "re-running indices: $IDXS (mem=$MEM)"
  submit redo --job-name="$JOB_NAME-redo" --array="$IDXS%$CONC" \
    --mem="$MEM" --cpus-per-task="$CPUS" --time="$TIME" \
    --partition="$PARTITION" --export=ALL \
    --output="$LOGS/redo-%A_%a.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python run_cells.py"
  ;;

status)
  echo "=== queue ==="
  squeue -u "$USER" -o "%.10i %.16j %.9T %.11M %.6D %R" | grep -E "scale|JOBID" || true
  echo "=== cells written ==="
  ls "$CALIB_RESULTS/cells" 2>/dev/null | wc -l
  ;;
*)
  echo "usage: launch_scale.sh {prepare|size <idx>|cells|redo <idx-list>|status}" >&2; exit 1 ;;
esac
