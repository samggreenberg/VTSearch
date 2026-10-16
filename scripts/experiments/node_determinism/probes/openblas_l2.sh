#!/usr/bin/env bash
# #4938: does OPENBLAS_L2_SIZE=2048 stop OpenBLAS 0.3.34's Zen 4 override (a segfault in
# float64 GEMM and a different float32 sum order under OPENBLAS_CORETYPE=Haswell), and does
# it leave every older OpenBLAS's bits alone?  One job a node; each (numpy, env, op) runs in
# its own process, so a crash is recorded against its op.
#
#   bash openblas_l2.sh submit NODE [NODE..]        # cpu partition
#   PROBE_SBATCH="--partition=gpu --gres=gpu:l40s:1" bash openblas_l2.sh submit rack4n01
#
# numpy comes from `uv pip install --target` dirs under $NP_DIR (see numpy_pins.sh install);
# 2.2.6 is the GRID venv's own.  Output: $PROBE_DIR/<node>.tsv, one line a process, the
# hash or CRASH(<exit code>).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VTS_REPO="${VTS_REPO:-$(cd "$HERE/../../../.." && pwd)}"
PROBE_DIR="${PROBE_DIR:-/expscratch/$USER/det-4938/l2}"
NP_DIR="${NP_DIR:-/expscratch/$USER/det-4930/np}"
VERSIONS="${VERSIONS:-2.2.6 2.4.x 2.5.1 2.5.x}"
# OPS_OVERRIDE="dgemm_scipy 128 768 ..." (space-separated triples) replaces the list below.
OPS=(
  "dgemm 128 64" "dgemm 128 256" "dgemm 128 300" "dgemm 128 320" "dgemm 128 512" "dgemm 128 768" "dgemm 128 1024"
  "sgemm 64 256" "sgemm 64 321" "sgemm 64 400" "sgemm 64 512" "sgemm 64 768" "sgemm 64 1024"
  "ssyrk 500 768" "dsyrk 200 768" "sgemv 4000 768" "dgemv 4000 768"
)
if [ -n "${OPS_OVERRIDE:-}" ]; then
  read -r -a _w <<<"$OPS_OVERRIDE"
  OPS=()
  for ((i = 0; i < ${#_w[@]}; i += 3)); do OPS+=("${_w[i]} ${_w[i + 1]} ${_w[i + 2]}"); done
fi

body() {
  cat <<BODY
set -uo pipefail
source $VTS_REPO/gridenv.sh >/dev/null 2>&1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENBLAS_CORETYPE=Haswell
mkdir -p $PROBE_DIR
out=$PROBE_DIR/\$(hostname -s).tsv
: > \$out
for v in $VERSIONS; do
  d=$NP_DIR/np-\$v
  for l2 in - 2048; do
    while read -r op m k; do
      if [ "\$l2" = - ]; then
        r=\$( (unset OPENBLAS_L2_SIZE; PYTHONPATH=\$d:\$PYTHONPATH python $HERE/probe_blas_ops.py \$op \$m \$k) 2>/dev/null); rc=\$?
      else
        r=\$(OPENBLAS_L2_SIZE=\$l2 PYTHONPATH=\$d:\$PYTHONPATH python $HERE/probe_blas_ops.py \$op \$m \$k 2>/dev/null); rc=\$?
      fi
      [ \$rc -eq 0 ] || r="\$(hostname -s)\tnumpy=\$v\tOPENBLAS_CORETYPE=Haswell OPENBLAS_L2_SIZE=\$l2\t\$op\t\$m\t\$k\tCRASH(\$rc)"
      printf '%b\n' "\$r" >> \$out
    done <<'LIST'
$(printf '%s\n' "${OPS[@]}")
LIST
  done
done
BODY
}

case "${1:-}" in
  submit)
    shift
    mkdir -p "$PROBE_DIR/logs"
    for n in "$@"; do
      # shellcheck disable=SC2086
      sbatch --parsable --job-name=l2-4938 --nodelist="$n" --cpus-per-task=1 --mem=4G --time=0:30:00 \
        ${PROBE_SBATCH:---partition=cpu} --output="$PROBE_DIR/logs/$n-%j.out" --wrap="$(body)"
    done
    ;;
  body) body ;;
  *)
    echo "usage: $0 {submit NODE..|body}" >&2
    exit 2
    ;;
esac
