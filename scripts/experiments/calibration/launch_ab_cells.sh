#!/usr/bin/env bash
# Submit both arms of the #2799 A/B (safe_thresholds ON and OFF) plus the two
# analyzers.  Run by launch_safe_ab.sh as an afterok dependency of prepare -
# the cell count is only known once prepare has selected the categories.
set -uo pipefail

WT="${VTS_REPO:-/exp/$USER/projects/vts-safe2799}"
HERE="$WT/scripts/experiments/calibration"
ON_EXP="${CALIB_AB_ON_EXP:-/exp/$USER/calibration-safe-linear}"
OFF_EXP="${CALIB_AB_OFF_EXP:-/exp/$USER/calibration-off-linear}"
ON_RES="$ON_EXP/results"
OFF_RES="$OFF_EXP/results"
LOGS="$ON_EXP/logs"
mkdir -p "$LOGS" "$OFF_RES/cells"

# Both arms must enumerate the identical (dataset, embedder, category, seed)
# cells, so the OFF run reuses the ON run's prepare output verbatim.
cp -f "$ON_RES/prepare_info.json" "$OFF_RES/prepare_info.json"

PARTITION="${CALIB_PARTITION:-cpu}"
GRES="${CALIB_CELL_GRES:-none}"
GRES_ARG=(--gres="$GRES")
[[ "$GRES" == "none" || -z "$GRES" ]] && GRES_ARG=()
MEM="${CALIB_MEM:-24G}"
CPUS="${CALIB_CPUS:-4}"
TIME="${CALIB_TIME:-3:00:00}"
CONC="${CALIB_CONC:-16}"

SHARED="export CALIB_DATASETS=$CALIB_DATASETS CALIB_VG_EMBEDDERS=$CALIB_VG_EMBEDDERS \
CALIB_PATCH_STYLES=$CALIB_PATCH_STYLES CALIB_REPOOL_VARIANTS='${CALIB_REPOOL_VARIANTS:-}' \
CALIB_MAX_STEPS=$CALIB_MAX_STEPS CALIB_N_SEEDS=$CALIB_N_SEEDS CALIB_HEAD=$CALIB_HEAD \
CALIB_SWEEP_KS=$CALIB_SWEEP_KS VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR \
VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"

N=$(cd "$HERE" && eval "$SHARED"; export CALIB_EXP="$ON_EXP" CALIB_RESULTS="$ON_RES"; \
    python run_cells.py --print-cells 2>/dev/null | tail -1)
if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" -eq 0 ]]; then
  echo "ERROR: could not determine cell count (got '$N'); did prepare finish?" >&2
  exit 1
fi
echo "cells per arm: $N (array 0-$((N-1))%$CONC on partition=$PARTITION gres=$GRES)"

submit_arm() {  # $1=name $2=exp dir $3=safe_thresholds flag
  sbatch --parsable --job-name="ab-$1" --array=0-$((N-1))%$CONC \
    "${GRES_ARG[@]}" --mem="$MEM" --cpus-per-task="$CPUS" --time="$TIME" --partition="$PARTITION" \
    --export=ALL --output="$LOGS/cells-$1-%A_%a.out" \
    --wrap="source $WT/gridenv.sh && $SHARED && export CALIB_EXP=$2 CALIB_RESULTS=$2/results CALIB_SAFE_THRESHOLDS=$3 && cd $HERE && python run_cells.py"
}

ON_JOB=$(submit_arm on "$ON_EXP" 1)
OFF_JOB=$(submit_arm off "$OFF_EXP" 0)
echo "safe-ON cells array:  $ON_JOB"
echo "safe-OFF cells array: $OFF_JOB"

# Within-run GMM variant contrasts (#2799 as pre-registered) on the ON arm.
A1=$(sbatch --parsable --dependency=afterany:$ON_JOB --job-name=ab-an-safe --mem=16G \
  --cpus-per-task=4 --time=0:40:00 --partition=cpu --export=ALL --output="$LOGS/analyze-safe-%j.out" \
  --wrap="source $WT/gridenv.sh && $SHARED && export CALIB_EXP=$ON_EXP CALIB_RESULTS=$ON_RES && cd $HERE && python analyze_safe.py")
echo "analyze_safe (ON):    $A1"

# The ship decision: ON vs OFF, paired per cell.
A2=$(sbatch --parsable --dependency=afterany:$ON_JOB:$OFF_JOB --job-name=ab-an-ab --mem=16G \
  --cpus-per-task=4 --time=0:40:00 --partition=cpu --export=ALL --output="$LOGS/analyze-ab-%j.out" \
  --wrap="source $WT/gridenv.sh && $SHARED && export CALIB_EXP=$ON_EXP CALIB_RESULTS=$ON_RES CALIB_AB_ON=$ON_RES CALIB_AB_OFF=$OFF_RES && cd $HERE && python analyze_ab.py")
echo "analyze_ab (ON/OFF):  $A2"

echo "$ON_JOB $OFF_JOB" > "$LOGS/.ab_jobids"
echo "Reports -> $ON_RES/REPORT.md and $ON_RES/REPORT_AB.md"
