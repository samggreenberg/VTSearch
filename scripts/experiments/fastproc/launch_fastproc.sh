#!/usr/bin/env bash
# Image-processor study (issue #3146): build one side pile per backend/device arm.
#
#   bash launch_fastproc.sh arms             # submit every arm (one GPU job each)
#   bash launch_fastproc.sh arms tv_cuda     # just one arm
#   bash launch_fastproc.sh size tv_cpu      # time ONE arm before committing to all
#   bash launch_fastproc.sh status
#   bash launch_fastproc.sh check            # provenance + cell agreement, post-run
#
# Nothing here writes to the shared pile.  Each arm gets its own pile root under
# $VTS_FASTPROC_STUDY, and weights are read from the shared pile's models dir so
# no arm re-downloads (and so a download cannot land on /exp's 50G quota).
#
# Every arm is pinned to ONE NODE, not a GPU type.  #3160 established that
# `gres/gpu:v100` is two different devices and that the swap alone moves
# siglip2_l fp32 by 1.5e-4 -- the size of the effect this study measures.  A
# type request would let SLURM reintroduce that as an unlabelled arm difference.
set -euo pipefail

# `set -e` can abort anywhere, including mid-preflight, which is how a launcher
# once exited 1 with no output and no job submitted
# (lessons/2026-08-17-a-launcher-that-exits-1-with-no-output.md).
trap 'echo "ABORTED: $0 line $LINENO exited $? -- NOTHING WAS SUBMITTED" >&2' ERR

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"

source "$WT/gridenv.sh"

export VTS_REPO="$WT"
export VTS_FASTPROC_STUDY="${VTS_FASTPROC_STUDY:-/expscratch/$USER/fastproc-3146}"
export VTS_PILE="${VTS_PILE:-/expscratch/$USER/vts-cache}"
EMBEDDERS="${VTS_FASTPROC_EMBEDDERS:-siglip,siglip2_l}"
DATASET="${VTS_FASTPROC_DATASET:-visual_genome_m}"
NODE="${VTS_FASTPROC_NODE:-rack4n01}"
GPU="${VTS_FASTPROC_GPU:-l40s}"

LOGS="$VTS_FASTPROC_STUDY/logs"
mkdir -p "$LOGS" "$VTS_FASTPROC_STUDY/results"

# A whole-image cell of 4193 medias peaks ~1.1 GB (GRID-PLAYBOOK), so 16G is
# generous.  Memory is the binding per-user quota and a fat request wedges the
# job off idle GPUs whose node RAM is already reserved.
MEM="${VTS_FASTPROC_MEM:-16G}"
CPUS="${VTS_FASTPROC_CPUS:-8}"
TIME="${VTS_FASTPROC_TIME:-2:00:00}"

ARMS_ALL="tv_cpu tv_cpu_rep pil_cpu tv_cuda"

# CPU threads are held FIXED across arms on purpose.  The treatment moves work
# between the CPU and the GPU, so a thread count that varied with the arm would
# be a second, unlabelled treatment -- and the CPU arms are precisely the ones a
# thread change would flatter.
BUILDENV="export VTS_REPO=$WT VTS_FASTPROC_STUDY=$VTS_FASTPROC_STUDY VTS_PILE=$VTS_PILE"
BUILDENV="$BUILDENV VTSEARCH_TORCH_THREADS=$CPUS OMP_NUM_THREADS=$CPUS MKL_NUM_THREADS=$CPUS"

# A submission is not a launch: --parsable returns an EMPTY id when the submit
# filter refuses the job, which looks exactly like a queued job (#2897 lost both
# arms this way, #2905's --gres=none refusal did it again).
submit_arm() {
  local arm="$1" jid
  jid=$(sbatch --parsable \
    --job-name="fproc-$arm" \
    --partition=gpu \
    --gres="gpu:${GPU}:1" \
    --nodelist="$NODE" \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$TIME" \
    --output="$LOGS/$arm-%j.out" \
    --wrap "source $WT/gridenv.sh && $BUILDENV && cd $HERE && python build_arm.py --arm $arm --dataset $DATASET --embedders $EMBEDDERS") || {
      echo "SUBMIT FAILED for $arm" >&2; return 1; }
  if [[ ! "$jid" =~ ^[0-9]+$ ]]; then
    echo "$arm SUBMIT FAILED (empty job id) -- NOT LAUNCHED" >&2
    return 1
  fi
  echo "$jid" > "$LOGS/.jobid_$arm"
  echo "$arm -> job $jid  (gpu:$GPU node:$NODE, log: $LOGS/$arm-$jid.out)"
}

preflight() {
  local fail=0
  if [[ ! -d "$VTS_PILE/datadir/visual_genome" ]]; then
    echo "FAIL: shared demo source $VTS_PILE/datadir/visual_genome missing -- a build would" >&2
    echo "      silently re-download a truncated dataset instead of failing." >&2
    fail=1
  fi
  for e in ${EMBEDDERS//,/ }; do
    if [[ ! -f "$VTS_PILE/datadir/embeddings/${DATASET}__${e}.pkl" ]]; then
      echo "FAIL: no published cell ${DATASET}__${e}.pkl to reproduce against" >&2
      fail=1
    fi
  done
  if [[ ! -d "$VTS_PILE/models" ]]; then
    echo "FAIL: shared models dir $VTS_PILE/models missing -- arms would re-download weights" >&2
    fail=1
  fi
  # The pinned node must exist and be usable, or every arm queues forever on a
  # --nodelist that SLURM will never satisfy.
  if ! sinfo -h -n "$NODE" -o "%N" | grep -q "$NODE"; then
    echo "FAIL: pinned node $NODE is not a known node" >&2
    fail=1
  elif ! sinfo -h -n "$NODE" -o "%G" | grep -q "$GPU"; then
    echo "FAIL: pinned node $NODE does not offer gpu:$GPU (has: $(sinfo -h -n "$NODE" -o '%G'))" >&2
    fail=1
  fi
  # Space on the mount the study is actually on, not its parent.
  local avail
  avail=$(df -BG --output=avail "$VTS_FASTPROC_STUDY" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
  if [[ "${avail:-0}" -lt 20 ]]; then
    echo "FAIL: only ${avail}G free on $(df --output=target "$VTS_FASTPROC_STUDY" | tail -1)" >&2
    fail=1
  fi
  # A zero-byte cell counts as "done" to the resume path, so it would be skipped.
  local zero
  mkdir -p "$VTS_FASTPROC_STUDY/piles"
  zero=$(find "$VTS_FASTPROC_STUDY/piles" -name '*.pkl' -size 0 2>/dev/null | wc -l || true)
  if [[ "$zero" -gt 0 ]]; then
    echo "FAIL: $zero zero-byte cell(s) under $VTS_FASTPROC_STUDY/piles -- resume would SKIP them:" >&2
    find "$VTS_FASTPROC_STUDY/piles" -name '*.pkl' -size 0 2>/dev/null >&2
    fail=1
  fi
  # A job name already in the queue breaks every per-name query, including the
  # completion waiter.
  local dupes
  dupes=$(squeue -u "$USER" -h -o "%j" | grep -c '^fproc-' || true)
  if [[ "${dupes:-0}" -gt 0 ]]; then
    echo "FAIL: $dupes fproc-* job(s) already queued; the completion waiter cannot tell the runs apart" >&2
    squeue -u "$USER" -o "%.10i %.16j %.9T" | grep 'fproc-' >&2 || true
    fail=1
  fi
  [[ "$fail" -eq 0 ]] || { echo "PREFLIGHT FAILED" >&2; return 2; }
  echo "preflight ok: shared pile readable, node $NODE has gpu:$GPU, ${avail}G free, no name collision"
}

case "${1:-status}" in
arms)
  shift
  ARMS="${*:-$ARMS_ALL}"
  preflight
  echo "study=$VTS_FASTPROC_STUDY dataset=$DATASET embedders=$EMBEDDERS node=$NODE"
  for arm in $ARMS; do submit_arm "$arm"; done
  echo
  echo "watch:  bash $0 status"
  echo "verify: bash $0 check"
  ;;

size)
  # Time ONE arm before committing to all four.  Sizing from a real cell rather
  # than a guess is the difference between an ETA and a number made up.
  ARM="${2:?usage: launch_fastproc.sh size <arm>}"
  preflight
  submit_arm "$ARM"
  ;;

status)
  echo "=== queue ==="
  squeue -u "$USER" -o "%.10i %.16j %.9T %.11M %.6D %R" | grep -E 'fproc-|JOBID' || echo "(no fproc-* jobs)"
  echo "=== cells built ==="
  n_want=$(echo "$EMBEDDERS" | tr ',' ' ' | wc -w)
  for arm in $ARMS_ALL; do
    n=$(ls "$VTS_FASTPROC_STUDY/piles/$arm/datadir/embeddings"/*.pkl 2>/dev/null | wc -l || true)
    z=$(find "$VTS_FASTPROC_STUDY/piles/$arm/datadir/embeddings" -name '*.pkl' -size 0 2>/dev/null | wc -l || true)
    prov=$([[ -f "$VTS_FASTPROC_STUDY/piles/$arm/provenance.json" ]] && echo "provenance" || echo "-")
    printf '  %-12s %s/%s cells  %-10s %s\n' "$arm" "$n" "$n_want" "$prov" \
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
