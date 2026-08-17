#!/usr/bin/env bash
# Embedding-precision study (issue #3143): build one side pile per precision arm.
#
#   bash launch_precision.sh arms            # submit every arm (one GPU job each)
#   bash launch_precision.sh arms fp16_l40s  # just one arm
#   bash launch_precision.sh size fp32_l40s  # time ONE arm before committing to all
#   bash launch_precision.sh status
#   bash launch_precision.sh check           # provenance + cell agreement, post-run
#
# Nothing here writes to the shared pile.  Each arm gets its own pile root under
# $VTS_PRECISION_STUDY, and weights are read from the shared pile's models dir so
# no arm re-downloads (and so a download cannot land on /exp's 50G quota).
#
# The GPU type is PINNED PER ARM and deliberately not auto-picked: the cross-GPU
# arm is the experiment's control, so letting pick_gpu.py choose would collapse
# the treatment and the control into one confounded difference.  This is the one
# launcher in the tree that should not call it (#3144 / PR #3150 argue the
# opposite for every launcher whose GPU choice is incidental — here it is not).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"

source "$WT/gridenv.sh"

export VTS_REPO="$WT"
export VTS_PRECISION_STUDY="${VTS_PRECISION_STUDY:-/expscratch/$USER/precision-3143}"
export VTS_PILE="${VTS_PILE:-/expscratch/$USER/vts-cache}"
EMBEDDERS="${VTS_PRECISION_EMBEDDERS:-siglip,siglip2_l}"
DATASET="${VTS_PRECISION_DATASET:-visual_genome_m}"

LOGS="$VTS_PRECISION_STUDY/logs"
mkdir -p "$LOGS" "$VTS_PRECISION_STUDY/results"

# Sized from the pile builds: a whole-image cell of 4193 medias peaks ~1.1 GB
# (GRID-PLAYBOOK), so 16G is already generous.  Keeping it small matters twice
# over here: memory is the binding per-user quota, and a fat request wedges the
# job off idle GPUs whose node RAM is already reserved.
MEM="${VTS_PRECISION_MEM:-16G}"
CPUS="${VTS_PRECISION_CPUS:-8}"
TIME="${VTS_PRECISION_TIME:-2:00:00}"

ARMS_ALL="fp32_l40s fp32_v100 fp16_l40s fp16_v100 autocast_l40s bf16_l40s"

arm_gpu() {
  case "$1" in
    *_l40s) echo l40s ;;
    *_v100) echo v100 ;;
    *) echo "unknown GPU for arm $1" >&2; return 1 ;;
  esac
}

# Spend the allocation the job actually holds: the image processor's
# resize/normalise runs on the CPU between decode and forward, and
# VTSEARCH_TORCH_THREADS defaults to 1, which stalls the GPU behind it.
BUILDENV="export VTS_REPO=$WT VTS_PRECISION_STUDY=$VTS_PRECISION_STUDY VTS_PILE=$VTS_PILE"
BUILDENV="$BUILDENV VTSEARCH_TORCH_THREADS=$CPUS OMP_NUM_THREADS=$CPUS MKL_NUM_THREADS=$CPUS"

# A submission is not a launch: --parsable returns an EMPTY id when the submit
# filter refuses the job, which looks exactly like a queued job (#2897 lost both
# arms this way, and #2905's --gres=none refusal did it again).
submit_arm() {
  local arm="$1" gpu jid
  gpu="$(arm_gpu "$arm")"
  jid=$(sbatch --parsable \
    --job-name="prec-$arm" \
    --partition=gpu \
    --gres="gpu:${gpu}:1" \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$TIME" \
    --output="$LOGS/$arm-%j.out" \
    --wrap "source $WT/gridenv.sh && $BUILDENV && cd $HERE && python build_arm.py --arm $arm --dataset $DATASET --embedders $EMBEDDERS") || {
      echo "SUBMIT FAILED for $arm" >&2; return 1; }
  if [[ ! "$jid" =~ ^[0-9]+$ ]]; then
    echo "$arm SUBMIT FAILED (empty job id) — NOT LAUNCHED" >&2
    return 1
  fi
  echo "$jid" > "$LOGS/.jobid_$arm"
  echo "$arm -> job $jid  (gpu:$gpu, log: $LOGS/$arm-$jid.out)"
}

preflight() {
  # The checkable mistakes, before any GPU time is spent.
  local fail=0
  if [[ ! -d "$VTS_PILE/datadir/visual_genome" ]]; then
    echo "FAIL: shared demo source $VTS_PILE/datadir/visual_genome missing — a build would" >&2
    echo "      silently re-download a truncated dataset instead of failing." >&2
    fail=1
  fi
  for e in ${EMBEDDERS//,/ }; do
    if [[ ! -f "$VTS_PILE/datadir/embeddings/${DATASET}__${e}.pkl" ]]; then
      echo "FAIL: no published fp32 cell ${DATASET}__${e}.pkl to reproduce against" >&2
      fail=1
    fi
  done
  if [[ ! -d "$VTS_PILE/models" ]]; then
    echo "FAIL: shared models dir $VTS_PILE/models missing — arms would re-download weights" >&2
    fail=1
  fi
  # Space on the mount the study is actually on, not its parent.
  local avail
  avail=$(df -BG --output=avail "$VTS_PRECISION_STUDY" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
  if [[ "${avail:-0}" -lt 20 ]]; then
    echo "FAIL: only ${avail}G free on $(df --output=target "$VTS_PRECISION_STUDY" | tail -1)" >&2
    fail=1
  fi
  # A zero-byte cell from a previous incident counts as "done" to the resume
  # path, so it would be skipped rather than rebuilt.
  local zero
  zero=$(find "$VTS_PRECISION_STUDY/piles" -name '*.pkl' -size 0 2>/dev/null | wc -l)
  if [[ "$zero" -gt 0 ]]; then
    echo "FAIL: $zero zero-byte cell(s) under $VTS_PRECISION_STUDY/piles — resume would SKIP them:" >&2
    find "$VTS_PRECISION_STUDY/piles" -name '*.pkl' -size 0 2>/dev/null >&2
    fail=1
  fi
  # A job name already in the queue breaks every per-name query, including the
  # completion waiter.
  local dupes
  dupes=$(squeue -u "$USER" -h -o "%j" | grep -c '^prec-' || true)
  if [[ "${dupes:-0}" -gt 0 ]]; then
    echo "FAIL: $dupes prec-* job(s) already queued; the completion waiter cannot tell the runs apart" >&2
    squeue -u "$USER" -o "%.10i %.16j %.9T" | grep 'prec-' >&2 || true
    fail=1
  fi
  [[ "$fail" -eq 0 ]] || { echo "PREFLIGHT FAILED" >&2; return 2; }
  echo "preflight ok: shared pile readable, ${avail}G free, no name collision"
}

case "${1:-status}" in
arms)
  shift
  ARMS="${*:-$ARMS_ALL}"
  preflight
  echo "study=$VTS_PRECISION_STUDY dataset=$DATASET embedders=$EMBEDDERS"
  for arm in $ARMS; do submit_arm "$arm"; done
  echo
  echo "watch:  bash $0 status"
  echo "verify: bash $0 check"
  ;;

size)
  # Time ONE arm before committing to all six. Sizing from a real cell rather
  # than a guess is the difference between an ETA and a number made up.
  ARM="${2:?usage: launch_precision.sh size <arm>}"
  preflight
  submit_arm "$ARM"
  ;;

status)
  echo "=== queue ==="
  squeue -u "$USER" -o "%.10i %.16j %.9T %.11M %.6D %R" | grep -E 'prec-|JOBID' || echo "(no prec-* jobs)"
  echo "=== cells built ==="
  for arm in $ARMS_ALL; do
    n=$(ls "$VTS_PRECISION_STUDY/piles/$arm/datadir/embeddings"/*.pkl 2>/dev/null | wc -l)
    z=$(find "$VTS_PRECISION_STUDY/piles/$arm/datadir/embeddings" -name '*.pkl' -size 0 2>/dev/null | wc -l)
    prov=$([[ -f "$VTS_PRECISION_STUDY/piles/$arm/provenance.json" ]] && echo "provenance" || echo "-")
    printf '  %-16s %s/%s cells  %-10s %s\n' "$arm" "$n" "$(echo "$EMBEDDERS" | tr ',' ' ' | wc -w)" "$prov" \
      "$([[ "$z" -gt 0 ]] && echo "** $z ZERO-BYTE **" || echo "")"
  done
  ;;

check)
  cd "$HERE" && python check_arms.py
  ;;

*)
  echo "usage: $0 {arms [arm...]|size <arm>|status|check}" >&2; exit 1
  ;;
esac
