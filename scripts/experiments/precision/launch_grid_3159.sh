#!/usr/bin/env bash
# #3159: what does the float16 patch grid cost region voting?
#
#   bash launch_grid_3159.sh build              # fp32-grid cells (1 GPU, pinned to the pile's node)
#   bash launch_grid_3159.sh verify             # fp32 cell -> fp16 cast == the pile cell, bit for bit. MUST pass
#   bash launch_grid_3159.sh drift              # score / rank drift on the pooled region score (CPU)
#   bash launch_grid_3159.sh prepare            # bench: per-arm category selection + exemplars
#   bash launch_grid_3159.sh verify-pairing     # bench: identical categories + exemplars. MUST pass
#   bash launch_grid_3159.sh size <arm> <idx>   # time one real cell first
#   bash launch_grid_3159.sh cells              # both arms' arrays
#   bash launch_grid_3159.sh status
#   bash launch_grid_3159.sh analyze
#
# Two arms, and only the patch-row dtype differs between them:
#
#   fp16  the PUBLISHED pile cells, symlinked read-only, scored as shipped
#   fp32  this study's rebuild with PATCH_ROW_DTYPE=float32: stored grid,
#         flattened region matrix and harness stack all float32
#
# The bench is the #3143 design (paired by (dataset, embedder, category, seed),
# production defaults, SE over cells) pointed at the region arm #3143 excluded.
# The embedder is the PAIR `siglip+dinov3_patch`: the app's real opening is a
# typed text sort, which DINOv3 has no tower for, so SigLIP ranks the query and
# DINOv3 does all the learning.
#
# verify-pairing: verify_pairing.py looks for crops under the PAIRED name and
# finds none (prepare writes them under the learn half). #3159 compared the two
# arms' crops/*.json and *.npz directly instead: identical, bit for bit. The opening is therefore identical in both arms
# by construction, and everything after it reads the grid under test.
set -euo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- NOTHING WAS SUBMITTED" >&2' ERR

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
CALIB="$WT/scripts/experiments/calibration"

source "$WT/gridenv.sh"

export VTS_REPO="$WT"
export VTS_GRID_STUDY="${VTS_GRID_STUDY:-/expscratch/$USER/patchgrid-3159}"
export VTS_PILE="${VTS_PILE:-/expscratch/$USER/vts-cache}"
BENCH_ROOT="${VTS_BENCH_ROOT:-$VTS_GRID_STUDY/bench}"
LOGS="$VTS_GRID_STUDY/logs"
mkdir -p "$LOGS"

DATASETS="${VTS_GRID_DATASETS:-visual_genome_m,coco_val,caltech101_m}"
ARMS="fp16,fp32"
# The pile's dinov3_patch cells were all built on rack7n03 (sacct-recovered,
# see their provenance sidecars).  Same node, same kernels: then the verify
# step can demand BIT identity instead of arguing about a tolerance.
GPU_NODE="${VTS_GPU_NODE:-rack7n03}"
# The embed BATCH SIZE is part of the forward: dinov3_patch's grid is not
# batch-invariant.  The pile cells predate pile_config's per-embedder batch
# table and were embedded at the then-default 32; at today's 64 a rebuild
# differs from them on ~1.3% of grid elements after the cast (some by several
# float16 ULPs), at 32 it matches bit for bit (probe on caltech101_m: 0 of
# 126M elements differ).  CPU dispatch (ATEN_CPU_CAPABILITY) made no difference.
BATCH="${VTS_GRID_BATCH:-32}"

# --- the bench grid ---------------------------------------------------------
# Region voting needs boxes, so the bench runs the two boxed datasets only;
# caltech101_m is in the drift analysis (it still max-pools over patches) but
# a boxless cell runs whole-image styles (experiment_config.styles_for).
export CALIB_DATASETS="${CALIB_DATASETS:-visual_genome_m,coco_val}"
export CALIB_VG_EMBEDDERS="siglip+dinov3_patch"
export CALIB_COCO_EMBEDDERS="siglip+dinov3_patch"
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-24}"
export CALIB_N_PER_BAND="${CALIB_N_PER_BAND:-4}"
export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-150}"
export CALIB_REQUIRE_OPENING="text"
export CALIB_REQUIRE_SEED_QUERY=1
export CALIB_REPOOL_VARIANTS=""
export CALIB_SCHEDULE_VARIANTS=""
export CALIB_FOLD_COUNTS=""
export CALIB_PATCH_STYLES="max_patch"
MEM="${CALIB_MEM:-14G}"  # sizing cell MaxRSS 9.6G
CPUS="${CALIB_CPUS:-2}"  # a cell is single-threaded (user ~= real)
TIME="${CALIB_TIME:-4:00:00}"
CONC="${CALIB_CONC:-16}"

arm_env() {
  local arm="$1"
  local exp="$BENCH_ROOT/$arm"
  local dtype="float16"; [[ "$arm" == fp32 ]] && dtype="float32"
  echo "export CALIB_EXP=$exp CALIB_RESULTS=$exp/results" \
       "VTSEARCH_DATA_DIR=$VTS_GRID_STUDY/piles/$arm/datadir" \
       "VTSEARCH_MODELS_DIR=$VTS_PILE/models HF_HOME=$VTS_PILE/models" \
       "VTS_REPO=$WT VTS_PATCH_ROW_DTYPE=$dtype" \
       "CALIB_DATASETS=$CALIB_DATASETS CALIB_VG_EMBEDDERS=$CALIB_VG_EMBEDDERS" \
       "CALIB_COCO_EMBEDDERS=$CALIB_COCO_EMBEDDERS" \
       "CALIB_N_SEEDS=$CALIB_N_SEEDS CALIB_N_PER_BAND=$CALIB_N_PER_BAND CALIB_MAX_STEPS=$CALIB_MAX_STEPS" \
       "CALIB_REQUIRE_OPENING=$CALIB_REQUIRE_OPENING CALIB_REQUIRE_SEED_QUERY=$CALIB_REQUIRE_SEED_QUERY" \
       "CALIB_REPOOL_VARIANTS= CALIB_SCHEDULE_VARIANTS= CALIB_FOLD_COUNTS=" \
       "CALIB_PATCH_STYLES=$CALIB_PATCH_STYLES"
}

submit() {
  local name="$1"; shift
  local J
  J=$(sbatch --parsable "$@") || { echo "SUBMIT FAILED for $name" >&2; return 1; }
  if [[ "$J" =~ ^[0-9]+$ ]]; then
    echo "$name -> job $J"
  else
    echo "$name SUBMIT FAILED (empty job id) -- NOT LAUNCHED" >&2
    return 1
  fi
}

# The fp16 arm reads the published cells in place.  Symlinks, never copies:
# nothing here may write into the shared pile, and a copy would be a second
# "published" cell for someone to find later.
link_fp16_arm() {
  local dd="$VTS_GRID_STUDY/piles/fp16/datadir/embeddings"
  mkdir -p "$dd"
  for ds in ${DATASETS//,/ }; do
    for emb in dinov3_patch siglip; do
      ln -sfn "$VTS_PILE/datadir/embeddings/${ds}__${emb}.pkl" "$dd/${ds}__${emb}.pkl"
    done
  done
  # The fp32 arm shares the SigLIP (opening) cells too: the opening is not
  # under test, and one file for both arms makes that true by construction.
  local d32="$VTS_GRID_STUDY/piles/fp32/datadir/embeddings"
  mkdir -p "$d32"
  for ds in ${DATASETS//,/ }; do
    ln -sfn "$VTS_PILE/datadir/embeddings/${ds}__siglip.pkl" "$d32/${ds}__siglip.pkl"
  done
  [[ -f "$VTS_GRID_STUDY/piles/fp16/provenance.json" ]] || \
    echo '{"arm": "fp16", "patch_row_dtype": "float16", "cells": "symlinks to the published pile"}' \
      > "$VTS_GRID_STUDY/piles/fp16/provenance.json"
}

case "${1:-status}" in
build)
  GPU_TYPE="$(sinfo -h -N -n "$GPU_NODE" -o "%G" | head -1 | sed -E 's/.*gpu:([A-Za-z0-9_.-]+):.*/\1/')"
  submit build --job-name=grid3159-build --partition=gpu --gres="gpu:${GPU_TYPE}:1" \
    --nodelist="$GPU_NODE" --cpus-per-task=8 --mem=48G --time=4:00:00 \
    --output="$LOGS/build-%j.out" \
    --wrap="source $WT/gridenv.sh && export HF_HOME=$VTS_PILE/models ATEN_CPU_CAPABILITY=avx2 VTSEARCH_EMBED_BATCH_SIZE=$BATCH VTSEARCH_TORCH_THREADS=8 OMP_NUM_THREADS=8 && cd $HERE && python build_grid_3159.py build --force --datasets $DATASETS && python build_grid_3159.py adopt-cls --datasets $DATASETS"
  ;;

verify)
  submit verify --job-name=grid3159-verify --partition=cpu --ntasks=1 --cpus-per-task=2 --mem=24G \
    --time=1:00:00 --output="$LOGS/verify-%j.out" \
    --wrap="source $WT/gridenv.sh && cd $HERE && python build_grid_3159.py verify --datasets $DATASETS"
  ;;

drift)
  submit drift --job-name=grid3159-drift --partition=cpu --ntasks=1 --cpus-per-task=8 --mem=48G \
    --time=6:00:00 --output="$LOGS/drift-%j.out" \
    --wrap="source $WT/gridenv.sh && export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 && cd $HERE && python grid_drift_3159.py --datasets $DATASETS --out $VTS_GRID_STUDY/drift"
  ;;

prepare)
  link_fp16_arm
  for arm in ${ARMS//,/ }; do
    exp="$BENCH_ROOT/$arm"
    mkdir -p "$exp/logs" "$exp/results/cells"
    [[ -f "$VTS_GRID_STUDY/piles/$arm/provenance.json" ]] || { echo "FAIL: arm $arm has no provenance.json" >&2; exit 2; }
    submit "prep-$arm" --job-name="grid3159-prep-$arm" --mem=32G --cpus-per-task=4 --ntasks=1 \
      --time=2:00:00 --partition=cpu --output="$exp/logs/prepare-%j.out" \
      --wrap="source $WT/gridenv.sh && $(arm_env "$arm") && cd $HERE && python run_cells_3159.py --prepare"
  done
  ;;

verify-pairing)
  cd "$HERE" && python verify_pairing.py --arms "$ARMS" --root "$BENCH_ROOT"
  ;;

size)
  ARM="${2:?usage: size <arm> <idx>}"; IDX="${3:?usage: size <arm> <idx>}"
  exp="$BENCH_ROOT/$ARM"; S="$exp/sizing"
  mkdir -p "$S/cells" "$exp/logs"
  ln -sfn "$exp/results/prepare_info.json" "$S/prepare_info.json"
  ln -sfn "$exp/results/crops" "$S/crops"
  submit "size-$ARM-$IDX" --job-name=grid3159-size --mem="$MEM" --cpus-per-task="$CPUS" --ntasks=1 \
    --time=2:00:00 --partition=cpu --output="$exp/logs/size-$IDX-%j.out" \
    --wrap="source $WT/gridenv.sh && $(arm_env "$ARM") && export CALIB_RESULTS=$S && cd $HERE && time python run_cells_3159.py --index $IDX"
  ;;

cells)
  for arm in ${ARMS//,/ }; do
    exp="$BENCH_ROOT/$arm"
    mkdir -p "$exp/logs" "$exp/results/cells"
    N=$(cd "$CALIB" && env $(arm_env "$arm" | sed 's/^export //') python run_cells.py --print-cells 2>/dev/null | tail -1)
    if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" -eq 0 ]]; then
      echo "ERROR: could not determine cell count for $arm (got '$N')" >&2; exit 1
    fi
    echo "$arm: $N cells (array 0-$((N-1))%$CONC)"
    submit "cells-$arm" --job-name="grid3159-$arm" --array="0-$((N-1))%$CONC" \
      --mem="$MEM" --cpus-per-task="$CPUS" --ntasks=1 --time="$TIME" --partition=cpu \
      --output="$exp/logs/cells-%A_%a.out" \
      --wrap="source $WT/gridenv.sh && $(arm_env "$arm") && cd $HERE && python run_cells_3159.py"
  done
  ;;

status)
  squeue -u "$USER" -o "%.12i %.22j %.9T %.10M %R" | grep -E 'grid3159|JOBID' || echo "(no grid3159 jobs)"
  for arm in ${ARMS//,/ }; do
    d="$BENCH_ROOT/$arm/results/cells"
    n=$(ls "$d" 2>/dev/null | grep -c '^task_[0-9]*\.csv$' || true)
    echo "  $arm: $n cells written"
  done
  ;;

analyze)
  cd "$HERE"
  python analyze_bench_precision.py --arms "$ARMS" --root "$BENCH_ROOT" | tee "$BENCH_ROOT/analyze_bench_precision.txt"
  python analyze_grid_3159.py --root "$BENCH_ROOT" --out "$VTS_GRID_STUDY/bench_analysis"
  ;;

*)
  echo "usage: $0 {build|verify|drift|prepare|verify-pairing|size|cells|status|analyze}" >&2; exit 1
  ;;
esac
