#!/usr/bin/env bash
# #4930: does a numpy pin take every numpy version off its AVX-512 kernels on every
# CPU?  One job a node; each (numpy, pin) runs in a fresh process, numpy put first on
# PYTHONPATH from a `uv pip install --target` dir under $NP_DIR.
#
#   bash numpy_pins.sh install                     # numpy 2.2.6 .. 2.5.x into $NP_DIR
#   bash numpy_pins.sh submit NODE [NODE..]        # cpu partition
#   PROBE_SBATCH="--partition=gpu --gres=gpu:l40s:1" bash numpy_pins.sh submit rack4n01
#
# Outputs: $PROBE_DIR/<node>__<numpy>__<pin>.json (probe_numpy_pins.py).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VTS_REPO="${VTS_REPO:-$(cd "$HERE/../../../.." && pwd)}"
PROBE_DIR="${PROBE_DIR:-/expscratch/$USER/det-4930/probes}"
NP_DIR="${NP_DIR:-/expscratch/$USER/det-4930/np}"
VERSIONS=(2.2.6 "2.3.*" 2.4.0 "2.4.*" "2.5.*")
# name=value; "-" leaves the variable unset.
PINS=(
  "base=-"
  "old=AVX512F AVX512_SKX"
  "old_icl=AVX512F AVX512_SKX AVX512_ICL"
  "old_all=AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR"
  "v4=X86_V4"
  "v4_icl_spr=X86_V4 AVX512_ICL AVX512_SPR"
)

body() {
  cat <<BODY
set -uo pipefail
source $VTS_REPO/gridenv.sh >/dev/null 2>&1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENBLAS_CORETYPE=Haswell PROBE_OUT=$PROBE_DIR
mkdir -p $PROBE_DIR
for d in $NP_DIR/np-*; do
  v=\$(basename "\$d"); v=\${v#np-}
$(for p in "${PINS[@]}"; do
    name=${p%%=*} val=${p#*=}
    if [ "$val" = - ]; then
      echo "  (unset NPY_DISABLE_CPU_FEATURES; PYTHONPATH=\$d:\$PYTHONPATH python $HERE/probe_numpy_pins.py \${v}__$name)"
    else
      echo "  (NPY_DISABLE_CPU_FEATURES='$val' PYTHONPATH=\$d:\$PYTHONPATH python $HERE/probe_numpy_pins.py \${v}__$name)"
    fi
  done)
done
BODY
}

case "${1:-}" in
  install)
    source "$VTS_REPO/gridenv.sh" >/dev/null 2>&1
    for v in "${VERSIONS[@]}"; do
      uv pip install -q --python "$(command -v python)" --target "$NP_DIR/np-${v//\*/x}" "numpy==$v"
    done
    ;;
  submit)
    shift
    mkdir -p "$PROBE_DIR/logs"
    for n in "$@"; do
      # shellcheck disable=SC2086
      sbatch --parsable --job-name=probe4930 --nodelist="$n" --cpus-per-task=1 --mem=4G --time=0:30:00 \
        ${PROBE_SBATCH:---partition=cpu} --output="$PROBE_DIR/logs/probe-$n-%j.out" --wrap="$(body)"
    done
    ;;
  body) body ;;
  *)
    echo "usage: $0 {install|submit NODE..|body}" >&2
    exit 2
    ;;
esac
