#!/usr/bin/env bash
# The training-domain summary's power against the full report, on the #3348 bench (issue #5568)
#
#   bash launch_summary_power_5568.sh run      # 15 tasks: embedder x seed
#   bash launch_summary_power_5568.sh status
#
# Reads #3348's matrices (launch_node_models_3348.sh prepare) in place: same
# 5 embedders, 5 datasets, 3 seeds and splits.
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- check what was submitted" >&2' ERR

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"

source "$WT/scripts/grid/gridenv.sh"

export VTS_REPO="$WT"
MATS="${SUMMARY_MATS:-/expscratch/$USER/atlas-node-3348/mats}"
EXP="${SUMMARY_EXP:-/expscratch/$USER/summary-5568}"
RESULTS="$EXP/results"
LOGS="$EXP/logs"
mkdir -p "$RESULTS" "$LOGS" "$EXP/analysis"

DATASETS="${SUMMARY_DATASETS:-coco_better,o365_better,coco_val,caltech101_m,visual_genome_m}"
EMBEDDERS="${SUMMARY_EMBEDDERS:-siglip,dinov3_patch,clip,clip_l,siglip2_l}"
SEEDS="${SUMMARY_SEEDS:-0,1,2}"
CPUS=4

ENVX="export VTS_REPO=$VTS_REPO OMP_NUM_THREADS=$CPUS MKL_NUM_THREADS=$CPUS OPENBLAS_NUM_THREADS=$CPUS CUDA_VISIBLE_DEVICES="

case "${1:-status}" in
run)
  i=0; : > "$RESULTS/run_map.csv"
  for sd in ${SEEDS//,/ }; do for emb in ${EMBEDDERS//,/ }; do
    echo "$i,$emb,$sd" >> "$RESULTS/run_map.csv"; i=$((i + 1)); done; done
  J=$(sbatch --parsable --job-name=summary5568 --array="0-$((i-1))" --mem=24G --cpus-per-task="$CPUS" \
    --time=1:00:00 --partition=cpu --export=ALL --output="$LOGS/run-%A_%a.out" \
    --wrap="source $WT/scripts/grid/gridenv.sh && $ENVX && cd $HERE && \
      L=\$(awk -F, -v i=\$SLURM_ARRAY_TASK_ID '\$1==i' $RESULTS/run_map.csv) && test -n \"\$L\" && \
      python run_summary_power_5568.py --embedder \$(cut -d, -f2 <<<\"\$L\") --seed \$(cut -d, -f3 <<<\"\$L\") \
        --datasets $DATASETS --mats $MATS --out $RESULTS")
  [[ "$J" =~ ^[0-9]+$ ]] || { echo "SUBMIT FAILED -- NOT LAUNCHED" >&2; exit 1; }
  echo "$J" > "$LOGS/.jobid_run"; echo "run -> job $J"
  ;;
status)
  echo "exp:   $EXP"
  echo "queue: $(squeue -u "$USER" -h -n summary5568 -o %i | wc -l) run"
  echo "run:   $(find "$RESULTS" -name 'shift_*.csv' | wc -l) of 15"
  ;;
*)
  echo "usage: $0 {run|status}" >&2; exit 1 ;;
esac
