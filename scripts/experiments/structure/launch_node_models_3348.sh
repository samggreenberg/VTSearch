#!/usr/bin/env bash
# Candidate atlas node models, scored the #3329 part-2 way (issue #3348)
#
#   bash launch_node_models_3348.sh prepare   # 25 tasks: pile cell -> compact matrix
#   bash launch_node_models_3348.sh run       # 15 tasks: embedder x seed, afterok:prepare
#   bash launch_node_models_3348.sh status
#
# The grid is #3329's 5 embedders x 3 seeds, but NOT its 5 datasets: the
# dinov3_patch cells of vg_scale_any and vg_box_large are gone with the VG
# machinery (#4038) and cannot be rebuilt, so coco_better and o365_better take
# their places.  The shipped model is re-run on this grid as `grid_mean_shipped`,
# so every comparison is on the same cells.
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- check what was submitted" >&2' ERR

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"

source "$WT/gridenv.sh"
source "$WT/scripts/experiments/pile/pile_env.sh"

export VTS_REPO="$WT"
EXP="${NODE_EXP:-/expscratch/$USER/atlas-node-3348}"
MATS="$EXP/mats"
RESULTS="$EXP/results"
LOGS="$EXP/logs"
mkdir -p "$MATS" "$RESULTS" "$LOGS" "$EXP/analysis"

DATASETS="${NODE_DATASETS:-coco_better,o365_better,coco_val,caltech101_m,visual_genome_m}"
EMBEDDERS="${NODE_EMBEDDERS:-siglip,dinov3_patch,clip,clip_l,siglip2_l}"
SEEDS="${NODE_SEEDS:-0,1,2}"
TIME="${NODE_TIME:-4:00:00}"
MEM="${NODE_MEM:-24G}"
CPUS="${NODE_CPUS:-4}"

ENVX="export VTS_REPO=$VTS_REPO CALIB_EXP=$EXP"
ENVX="$ENVX VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
ENVX="$ENVX OMP_NUM_THREADS=$CPUS MKL_NUM_THREADS=$CPUS OPENBLAS_NUM_THREADS=$CPUS CUDA_VISIBLE_DEVICES="

submit() {
  local name="$1"; shift
  local J
  J=$(sbatch --parsable "$@") || { echo "SUBMIT FAILED for $name" >&2; return 1; }
  if [[ "$J" =~ ^[0-9]+$ ]]; then
    echo "$J" > "$LOGS/.jobid_$name"; echo "$name -> job $J"
  else
    echo "$name SUBMIT FAILED (empty job id) -- NOT LAUNCHED" >&2; return 1
  fi
}

case "${1:-status}" in
prepare)
  i=0; : > "$MATS/prep_map.csv"
  for ds in ${DATASETS//,/ }; do for emb in ${EMBEDDERS//,/ }; do
    echo "$i,$ds,$emb" >> "$MATS/prep_map.csv"; i=$((i + 1)); done; done
  # A patch cell streams through iter_medias, but a one-shot pickle is still
  # read whole: coco_better's 7.5 GB cell sets the memory.
  submit prepare --job-name=node3348-prep --array="0-$((i-1))%12" --mem=40G --cpus-per-task=2 \
    --time=1:00:00 --partition=cpu --export=ALL --output="$LOGS/prep-%A_%a.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && \
      L=\$(awk -F, -v i=\$SLURM_ARRAY_TASK_ID '\$1==i' $MATS/prep_map.csv) && test -n \"\$L\" && \
      python prepare_matrices_3348.py --dataset \$(cut -d, -f2 <<<\"\$L\") --embedder \$(cut -d, -f3 <<<\"\$L\") --out $MATS"
  ;;

run)
  i=0; : > "$RESULTS/run_map.csv"
  for sd in ${SEEDS//,/ }; do for emb in ${EMBEDDERS//,/ }; do
    echo "$i,$emb,$sd" >> "$RESULTS/run_map.csv"; i=$((i + 1)); done; done
  DEP=()
  [[ -f "$LOGS/.jobid_prepare" ]] && DEP=(--dependency="afterok:$(cat "$LOGS/.jobid_prepare")")
  submit run --job-name=node3348 --array="0-$((i-1))" "${DEP[@]}" --mem="$MEM" --cpus-per-task="$CPUS" \
    --time="$TIME" --partition=cpu --export=ALL --output="$LOGS/run-%A_%a.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && \
      L=\$(awk -F, -v i=\$SLURM_ARRAY_TASK_ID '\$1==i' $RESULTS/run_map.csv) && test -n \"\$L\" && \
      python run_node_models_3348.py --embedder \$(cut -d, -f2 <<<\"\$L\") --seed \$(cut -d, -f3 <<<\"\$L\") \
        --datasets $DATASETS --mats $MATS --out $RESULTS"
  ;;

status)
  echo "exp:   $EXP"
  echo "queue: $(squeue -u "$USER" -h -n node3348-prep -o %i | wc -l) prepare, $(squeue -u "$USER" -h -n node3348 -o %i | wc -l) run"
  echo "mats:  $(find "$MATS" -name '*.npz' | wc -l) of 25"
  echo "run:   $(find "$RESULTS" -name 'shift_*.csv' | wc -l) of 15"
  ;;

*)
  echo "usage: $0 {prepare|run|status}" >&2; exit 1 ;;
esac
