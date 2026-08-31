#!/usr/bin/env bash
# The rest of the #3329 inventory: parts B and C (issue #3329, PREREG-part2.md)
#
#   bash launch_structure_3329.sh list      # index -> cell map
#   bash launch_structure_3329.sh cells     # the 25-cell array
#   bash launch_structure_3329.sh shift     # the 5 cross-dataset tasks
#   bash launch_structure_3329.sh status
#
# Part 1 measured the fit that decides every threshold, and rode the click loop
# to do it.  These families are fitted ONCE per dataset, so the axis that buys
# information is dataset x embedder, not clicks -- 25 cells, plus one
# cross-dataset task per embedder for the domain-shift guard's POWER, which no
# single-dataset cell can see.
#
# Shipped defaults everywhere: the atlas is built with the same k=3 and
# auto_max_depth call production uses, so the null under test is the shipped
# estimator's own.
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- NOTHING WAS SUBMITTED" >&2' ERR

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"

source "$WT/gridenv.sh"
source "$WT/scripts/experiments/pile/pile_env.sh"

export VTS_REPO="$WT"
export STRUCT_EXP="${STRUCT_EXP:-/expscratch/$USER/struct-3329}"
RESULTS="$STRUCT_EXP/results"
LOGS="$STRUCT_EXP/logs"
mkdir -p "$RESULTS" "$LOGS" "$STRUCT_EXP/analysis"

# BLAS pinned so a cell's cost is the same whether one or twenty run at once.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

DATASETS="${STRUCT_DATASETS:-vg_scale_any,coco_val,caltech101_m,vg_box_large,visual_genome_m}"
EMBEDDERS="${STRUCT_EMBEDDERS:-siglip,dinov3_patch,clip,clip_l,siglip2_l}"
# Seeds move the build/holdout SPLIT, which is the only randomness these
# families have. Without them every statistic is a point estimate off one split
# and the report cannot say whether a difference between two embedders is real.
SEEDS="${STRUCT_SEEDS:-0,1,2}"

# Measured on the caltech101_m/siglip smoke run (838 items, 40s) and scaled for
# the largest dataset in the grid; the projection stage is the long pole and is
# capped at PROJECTION_MAX_N=3000 points, so cost is near-flat above that.
MEM="${STRUCT_MEM:-16G}"
CPUS="${STRUCT_CPUS:-2}"
TIME="${STRUCT_TIME:-2:00:00}"
JOB_NAME="${STRUCT_JOB_NAME:-struct-3329}"

ENVX="export STRUCT_EXP=$STRUCT_EXP VTS_REPO=$VTS_REPO"
ENVX="$ENVX VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
ENVX="$ENVX OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES="

cells_list() {
  local i=0
  for sd in ${SEEDS//,/ }; do
    for ds in ${DATASETS//,/ }; do
      for emb in ${EMBEDDERS//,/ }; do
        echo "$i $ds $emb $sd"
        i=$((i + 1))
      done
    done
  done
}

submit() {
  local name="$1"; shift
  local J
  J=$(sbatch --parsable "$@") || { echo "SUBMIT FAILED for $name" >&2; return 1; }
  if [[ "$J" =~ ^[0-9]+$ ]]; then
    echo "$J" > "$LOGS/.jobid_$name"; echo "$name -> job $J"
  else
    echo "$name SUBMIT FAILED (empty job id) — NOT LAUNCHED" >&2; return 1
  fi
}

case "${1:-status}" in
list)
  cells_list | while read -r i ds emb sd; do printf "  %3d  %s / %s  seed=%s\n" "$i" "$ds" "$emb" "$sd"; done
  echo "$(cells_list | wc -l) cells; datasets=$DATASETS embedders=$EMBEDDERS"
  ;;

cells)
  N=$(cells_list | wc -l)
  echo "cells: $N (array 0-$((N-1)))"
  # The cell map is written beside the results BEFORE the array exists, so the
  # analysis reports "25 of 25" off the grid that actually ran rather than off a
  # literal typed into a report later.
  cells_list | awk '{print $1","$2","$3","$4}' | sed '1i index,dataset,embedder,seed' > "$RESULTS/cell_map.csv"
  # The task resolves its cell from cell_map.csv, NOT by re-deriving it: the map
  # is what the analysis reads, so an array task and the report cannot disagree
  # about which cell an index was. Parsing the pretty `list` output instead
  # would also have split `ds / emb` on the separator.
  submit cells --job-name="$JOB_NAME" --array="0-$((N-1))" \
    --mem="$MEM" --cpus-per-task="$CPUS" --time="$TIME" --partition=cpu --export=ALL \
    --output="$LOGS/cell-%A_%a.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && \
      DS=\$(awk -F, -v i=\$SLURM_ARRAY_TASK_ID 'NR>1 && \$1==i {print \$2}' $RESULTS/cell_map.csv) && \
      EMB=\$(awk -F, -v i=\$SLURM_ARRAY_TASK_ID 'NR>1 && \$1==i {print \$3}' $RESULTS/cell_map.csv) && \
      SD=\$(awk -F, -v i=\$SLURM_ARRAY_TASK_ID 'NR>1 && \$1==i {print \$4}' $RESULTS/cell_map.csv) && \
      test -n \"\$DS\" -a -n \"\$EMB\" -a -n \"\$SD\" && \
      python structure_fits_3329.py --dataset \$DS --embedder \$EMB --seed \$SD --out $RESULTS"
  ;;

shift)
  for emb in ${EMBEDDERS//,/ }; do
    submit "shift_$emb" --job-name="$JOB_NAME-shift" --mem=32G --cpus-per-task="$CPUS" \
      --time="$TIME" --partition=cpu --export=ALL \
      --output="$LOGS/shift-$emb-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && \
        python domain_shift_3329.py --embedder $emb --datasets $DATASETS --out $RESULTS"
  done
  ;;

status)
  echo "exp:    $STRUCT_EXP"
  echo "queue:  $(squeue -u "$USER" -h -n "$JOB_NAME" -o %i | wc -l) cells, $(squeue -u "$USER" -h -n "$JOB_NAME-shift" -o %i | wc -l) shift"
  echo "cells:  $(find "$RESULTS" -name 'struct_*.csv' 2>/dev/null | wc -l) of $(cells_list | wc -l)"
  echo "shift:  $(find "$RESULTS" -name 'domainshift_*.csv' 2>/dev/null | wc -l) of $(tr ',' '\n' <<<"$EMBEDDERS" | wc -l)"
  echo "empty:  $(find "$RESULTS" -name '*.csv' -size 0 2>/dev/null | wc -l) zero-byte"
  ;;

*)
  echo "usage: $0 {list|cells|shift|status}" >&2; exit 1 ;;
esac
