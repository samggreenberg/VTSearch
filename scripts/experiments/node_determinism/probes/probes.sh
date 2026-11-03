#!/usr/bin/env bash
# #4110's per-op probes: one job a node, every env variant in a fresh process
# (the pins are read when numpy and torch load).
#
#   bash probes.sh submit NODE [NODE..]    # e.g. rack7n11 rack2n13 (cpu partition)
#   PROBE_SBATCH="--partition=gpu --gres=gpu:v100:1" bash probes.sh submit rack8n03
#   bash probes.sh collate                 # tables -> $PROBE_DIR/tables
#
# Outputs: $PROBE_DIR/{ops,mkl,text,cost}/<node>__<variant>.json.  The text probe
# loads SigLIP's text tower, so it needs the pile's models dir (pile_env.sh).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VTS_REPO="${VTS_REPO:-/expscratch/$USER/worktrees/vts-4110-run}"
PROBE_DIR="${PROBE_DIR:-/expscratch/$USER/det-4110/probes}"
NPY="AVX512F AVX512_SKX"

body() {
  cat <<BODY
set -uo pipefail
source $VTS_REPO/scripts/grid/gridenv.sh >/dev/null 2>&1
source $VTS_REPO/scripts/experiments/pile/pile_env.sh
unset NPY_DISABLE_CPU_FEATURES MKL_CBWR ATEN_CPU_CAPABILITY
export VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR HF_HOME CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CALIB_EXP=$PROBE_DIR/calib CALIB_RESULTS=$PROBE_DIR/calib/results CALIB_DATASETS=coco_better CALIB_COCO_BETTER_EMBEDDERS=siglip
mkdir -p $PROBE_DIR/{ops,mkl,text,cost}
o() { PROBE_OUT=$PROBE_DIR/ops python $HERE/probe_ops.py "\$@"; }
m() { PROBE_OUT=$PROBE_DIR/mkl python $HERE/probe_mkl.py "\$@"; }
t() { PROBE_OUT=$PROBE_DIR/text python $HERE/probe_text.py "\$@"; }
c() { PROBE_OUT=$PROBE_DIR/cost OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python $HERE/probe_cost.py "\$@"; }
o base; NPY_DISABLE_CPU_FEATURES="$NPY" o npy
m base; MKL_CBWR=COMPATIBLE m mkl_compat; MKL_CBWR=AVX m mkl_avx; MKL_CBWR=AVX2,STRICT m mkl_avx2_strict
ATEN_CPU_CAPABILITY=avx2 m aten2; MKL_ENABLE_INSTRUCTIONS=AVX2 ATEN_CPU_CAPABILITY=avx2 m mklinstr_aten2
t base; NPY_DISABLE_CPU_FEATURES="$NPY" MKL_CBWR=COMPATIBLE t pins
NPY_DISABLE_CPU_FEATURES="$NPY" MKL_CBWR=COMPATIBLE ATEN_CPU_CAPABILITY=avx2 t pins_aten2
c base; NPY_DISABLE_CPU_FEATURES="$NPY" c npy; MKL_CBWR=COMPATIBLE c mklcompat
NPY_DISABLE_CPU_FEATURES="$NPY" MKL_CBWR=COMPATIBLE ATEN_CPU_CAPABILITY=avx2 c all
BODY
}

case "${1:-}" in
  submit)
    shift
    mkdir -p "$PROBE_DIR/logs"
    for n in "$@"; do
      # shellcheck disable=SC2086
      sbatch --parsable --job-name=probe4110 --nodelist="$n" --cpus-per-task=4 --mem=8G --time=1:00:00 \
        ${PROBE_SBATCH:---partition=cpu} --output="$PROBE_DIR/logs/probe-$n-%j.out" --wrap="$(body)"
    done
    ;;
  collate)
    source "$VTS_REPO/scripts/grid/gridenv.sh" >/dev/null 2>&1
    python "$HERE/collate_probes.py" --probes "$PROBE_DIR" --out "$PROBE_DIR/tables"
    ;;
  *)
    echo "usage: $0 {submit NODE..|collate}" >&2
    exit 2
    ;;
esac
