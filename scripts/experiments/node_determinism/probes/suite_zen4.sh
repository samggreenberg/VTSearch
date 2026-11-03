#!/usr/bin/env bash
# #4938: the app's own test suite on a Zen 4 CPU with a numpy that bundles OpenBLAS 0.3.34,
# under the Haswell pin, with and without the OPENBLAS_L2_SIZE that keeps its blocking.
#
#   bash suite_zen4.sh submit [NODE]     # default rack4n01 (EPYC 9534, an L40S node); both arms
#
# Arms: bug = OPENBLAS_L2_SIZE exported empty (OpenBLAS detects the 1 MB L2 and re-tunes the
# forced Haswell core); fix = gridenv.sh's default.  numpy comes first on PYTHONPATH from
# $NP_DIR (numpy_pins.sh install).  Output: $OUT_DIR/<arm>/{pytest.log,junit.xml}.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="${VTS_REPO:-$(cd "$HERE/../../../.." && pwd)}"
NUMPY="${NUMPY:-/expscratch/$USER/det-4930/np/np-2.5.x}"
OUT_DIR="${OUT_DIR:-/expscratch/$USER/det-4938/suite}"

run() {
  local arm=$1 out="$OUT_DIR/$1"
  mkdir -p "$out"
  export PYTHONPATH="$NUMPY"
  unset OPENBLAS_L2_SIZE OPENBLAS_CORETYPE NPY_DISABLE_CPU_FEATURES
  [ "$arm" = bug ] && export OPENBLAS_L2_SIZE=
  cd "$WT" && source scripts/grid/gridenv.sh >/dev/null 2>&1
  echo "host=$(hostname -s) cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)" \
    "numpy=$(python -c 'import numpy; print(numpy.__version__)')" \
    "CORETYPE=$OPENBLAS_CORETYPE L2=[${OPENBLAS_L2_SIZE-unset}]"
  CUDA_VISIBLE_DEVICES='' python -m pytest tests tests_lib -q --tb=line --no-header -n 12 --dist loadgroup \
    -p no:cacheprovider --continue-on-collection-errors --junitxml="$out/junit.xml" >"$out/pytest.log" 2>&1
  echo "pytest exit $?"
  tail -5 "$out/pytest.log"
}

case "${1:-}" in
  submit)
    node="${2:-rack4n01}"
    mkdir -p "$OUT_DIR"
    for arm in bug fix; do
      sbatch --parsable --job-name="suite4938-$arm" --nodelist="$node" --partition=gpu --gres=gpu:l40s:1 \
        --cpus-per-task=12 --mem=48G --time=1:00:00 --output="$OUT_DIR/$arm.out" "$0" run "$arm"
    done
    ;;
  run) run "$2" ;;
  *)
    echo "usage: $0 {submit [NODE]|run bug|fix}" >&2
    exit 2
    ;;
esac
