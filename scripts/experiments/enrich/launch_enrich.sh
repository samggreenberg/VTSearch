#!/usr/bin/env bash
# Description-enrichment default study (issue #3127): one GPU chunk per media type.
#
#   bash launch_enrich.sh size            # time ONE cheap cell before committing
#   bash launch_enrich.sh chunks          # submit every chunk
#   bash launch_enrich.sh chunks image    # just one chunk
#   bash launch_enrich.sh status
#   bash launch_enrich.sh wait            # blocks until the queue drains, then summarises
#
# Four chunks, one per media type, because the cost here is the *dataset load*
# (download + embed) and a chunk that stays in one process reuses the encoder it
# already paid for.  The audio chunk runs a second pass with `--embedder clap`:
# #3077's claim that enrichment flips sign between the two CLAP checkpoints is
# the premise this whole issue rests on, so it gets re-measured, not assumed.
#
# 4 chunks is also exactly the 4gpu_tier per-user GPU cap, so the whole study is
# one wave with nothing re-queued behind itself.
set -euo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- CHECK WHAT WAS SUBMITTED" >&2' ERR

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"

source "$WT/gridenv.sh"
export VTS_REPO="$WT"

EXP="${VTS_ENRICH_EXP:-/expscratch/$USER/enrich-3127}"
LOGS="$EXP/logs"
mkdir -p "$LOGS" "$EXP/results" "$EXP/data"

MEM="${VTS_ENRICH_MEM:-24G}"
CPUS="${VTS_ENRICH_CPUS:-8}"
TIME="${VTS_ENRICH_TIME:-6:00:00}"
JOB_PREFIX="enr3127"

# Every eval dataset whose media type has a text-capable default embedder.
# `face` (default `face`, no text tower) and `document` (no embedders at all)
# cannot text-sort, so the setting is inert for them -- and the vggface2 eval
# datasets are *image* media type, so they are in the image chunk.
DS_AUDIO="esc50_s esc50_m esc50_l"
DS_IMAGE="caltech101_s caltech101_m caltech256_a enrico_m enrico_a rico_screen2words_m rico_screen2words_a rvl_cdip_m rvl_cdip_a vggface2_faces_s vggface2_faces_m visual_genome_s visual_genome_m"
DS_TEXT="20newsgroups_s 20newsgroups_m 20newsgroups_l"
DS_VIDEO="ucf101_s ucf101_m ucf101_l"

CHUNKS_ALL="audio image text video"

chunk_datasets() {
  case "$1" in
    audio) echo "$DS_AUDIO" ;;
    image) echo "$DS_IMAGE" ;;
    text)  echo "$DS_TEXT" ;;
    video) echo "$DS_VIDEO" ;;
    *) echo "unknown chunk: $1" >&2; return 1 ;;
  esac
}

RUNENV="export VTS_REPO=$WT VTSEARCH_DATA_DIR=$EXP/data VTSEARCH_MODELS_DIR=${VTS_ENRICH_MODELS:-/exp/$USER/projects/VTSearch/data/models}"
RUNENV="$RUNENV VTSEARCH_TORCH_THREADS=$CPUS OMP_NUM_THREADS=$CPUS MKL_NUM_THREADS=$CPUS"

# The GPU choice belongs to the scheduler, not to whatever `VTS_GPU` the
# interactive shell exports for the app launcher (~/.bashrc pins v100 there, and
# pick_gpu.py honours an explicit pin without querying anything).  A study's GPU
# is incidental: this comparison is paired *inside* one process on one card, so
# nothing here wants a particular part -- only the one that can start now.
pick_gpu() {
  if [[ -n "${VTS_ENRICH_GPU:-}" ]]; then echo "$VTS_ENRICH_GPU"; return; fi
  env -u VTS_GPU python3 "$WT/scripts/slurm/pick_gpu.py" "$@"
}

submit_chunk() {
  local chunk="$1" ds gpu jid cmd
  ds="$(chunk_datasets "$chunk")"
  gpu="$(pick_gpu --need "${VTS_ENRICH_NEED:-1}")"
  cmd="cd $HERE && python run_enrich.py --exp $EXP --wrappers --datasets $ds"
  if [[ "$chunk" == "audio" ]]; then
    # The #3077 premise control, on the same node and the same medias.
    cmd="$cmd && python run_enrich.py --exp $EXP --wrappers --embedder clap --datasets $ds"
  fi
  jid=$(sbatch --parsable \
    --job-name="$JOB_PREFIX-$chunk" \
    --partition=gpu \
    --gres="gpu:${gpu}:1" \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$TIME" \
    --output="$LOGS/$chunk-%j.out" \
    --wrap "source $WT/gridenv.sh && $RUNENV && $cmd") || {
      echo "SUBMIT FAILED for $chunk" >&2; return 1; }
  # A submission is not a launch: --parsable returns an EMPTY id when the submit
  # filter refuses the job, which looks exactly like a queued job.
  if [[ ! "$jid" =~ ^[0-9]+$ ]]; then
    echo "$chunk SUBMIT FAILED (empty job id) -- NOT LAUNCHED" >&2
    return 1
  fi
  echo "$jid" > "$LOGS/.jobid_$chunk"
  echo "$chunk -> job $jid  (gpu:$gpu, $(echo "$ds" | wc -w) datasets, log: $LOGS/$chunk-$jid.out)"
}

preflight() {
  local fail=0 avail dupes zero
  # Space on the mount the study is actually on, not its parent: the demo
  # downloads (~3.5 GB) and their .dl_* spool land in $EXP/data.
  avail=$(df -BG --output=avail "$EXP" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
  if [[ "${avail:-0}" -lt 30 ]]; then
    echo "FAIL: only ${avail}G free on $(df --output=target "$EXP" | tail -1); the demo sources need ~10G" >&2
    fail=1
  fi
  # A zero-byte cell counts as "done" to the resume path, so it would be skipped.
  zero=$(find "$EXP/results" -name '*.csv' -size 0 2>/dev/null | wc -l || true)
  if [[ "${zero:-0}" -gt 0 ]]; then
    echo "FAIL: $zero zero-byte cell(s) under $EXP/results -- resume would SKIP them:" >&2
    find "$EXP/results" -name '*.csv' -size 0 2>/dev/null >&2
    fail=1
  fi
  # A job name already in the queue breaks the per-name completion waiter.
  dupes=$(squeue -u "$USER" -h -o "%j" | grep -c "^$JOB_PREFIX-" || true)
  if [[ "${dupes:-0}" -gt 0 ]]; then
    echo "FAIL: $dupes $JOB_PREFIX-* job(s) already queued; the waiter cannot tell the runs apart" >&2
    squeue -u "$USER" -o "%.10i %.16j %.9T" | grep "$JOB_PREFIX-" >&2 || true
    fail=1
  fi
  # The worktree the cluster will actually import must be the one we committed.
  if [[ -n "$(git -C "$WT" status --porcelain)" ]]; then
    echo "WARN: $WT has uncommitted changes -- the run measures code that is not on the branch" >&2
  fi
  [[ "$fail" -eq 0 ]] || { echo "PREFLIGHT FAILED" >&2; return 2; }
  echo "preflight ok: ${avail}G free on $EXP, no zero-byte cells, no name collision"
}

case "${1:-status}" in
size)
  # Time ONE real cell before committing to the rest.  esc50_s is the cheapest
  # cell that still downloads a source and loads an encoder.
  preflight
  DS="${2:-esc50_s}"
  gpu="$(pick_gpu)"
  jid=$(sbatch --parsable --job-name="$JOB_PREFIX-size" --partition=gpu \
    --gres="gpu:${gpu}:1" --cpus-per-task="$CPUS" --mem="$MEM" --time=2:00:00 \
    --output="$LOGS/size-%j.out" \
    --wrap "source $WT/gridenv.sh && $RUNENV && cd $HERE && python run_enrich.py --exp $EXP --wrappers --datasets $DS")
  [[ "$jid" =~ ^[0-9]+$ ]] || { echo "SIZE SUBMIT FAILED (empty job id)" >&2; exit 1; }
  echo "size cell $DS -> job $jid (gpu:$gpu, log: $LOGS/size-$jid.out)"
  ;;

chunks)
  shift
  preflight
  CHUNKS="${*:-$CHUNKS_ALL}"
  echo "exp=$EXP mem=$MEM cpus=$CPUS time=$TIME"
  export VTS_ENRICH_NEED="$(echo "$CHUNKS" | wc -w)"
  for c in $CHUNKS; do submit_chunk "$c"; done
  echo
  echo "watch: bash $0 status   |   block: bash $0 wait"
  ;;

control)
  # A non-default embedder, on datasets of its own media type.  The audio chunk
  # carries `clap` inline because that control is the issue's premise; this is
  # for the ones a result asks for afterwards -- e.g. `bge` on the text corpora,
  # which says whether enrichment's damage there belongs to the media type or to
  # E5's asymmetric query:/passage: convention.
  shift
  EMB="${1:?usage: launch_enrich.sh control <embedder> <dataset>...}"
  shift
  DS="${*:?usage: launch_enrich.sh control <embedder> <dataset>...}"
  gpu="$(pick_gpu)"
  jid=$(sbatch --parsable --job-name="$JOB_PREFIX-ctl-$EMB" --partition=gpu \
    --gres="gpu:${gpu}:1" --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME" \
    --output="$LOGS/ctl-$EMB-%j.out" \
    --wrap "source $WT/gridenv.sh && $RUNENV && cd $HERE && python run_enrich.py --exp $EXP --wrappers --embedder $EMB --datasets $DS")
  [[ "$jid" =~ ^[0-9]+$ ]] || { echo "CONTROL SUBMIT FAILED (empty job id)" >&2; exit 1; }
  echo "control $EMB -> job $jid (gpu:$gpu, log: $LOGS/ctl-$EMB-$jid.out)"
  ;;

status)
  echo "=== queue ==="
  squeue -u "$USER" -o "%.10i %.18j %.9T %.11M %.6D %R" | grep -E "$JOB_PREFIX-|JOBID" || echo "(no $JOB_PREFIX-* jobs)"
  echo "=== cells ==="
  # `ls glob | wc -l` is a pipefail landmine: with no matches `ls` exits 2 and
  # takes the whole script with it.
  printf '  %s cell CSVs\n' "$(find "$EXP/results" -name '*.csv' 2>/dev/null | wc -l)"
  for c in $CHUNKS_ALL; do
    want=$(chunk_datasets "$c" | wc -w)
    have=0
    for d in $(chunk_datasets "$c"); do
      [[ -s "$EXP/results/default__$d.csv" ]] && have=$((have + 1)) || true
    done
    printf '  %-6s %s/%s\n' "$c" "$have" "$want"
  done
  have=0
  for d in $DS_AUDIO; do [[ -s "$EXP/results/clap__$d.csv" ]] && have=$((have + 1)) || true; done
  printf '  %-6s %s/%s\n' "clap" "$have" "$(echo "$DS_AUDIO" | wc -w)"
  z=$(find "$EXP/results" -name '*.csv' -size 0 2>/dev/null | wc -l || true)
  [[ "${z:-0}" -gt 0 ]] && echo "  ** $z ZERO-BYTE cells **" || true
  ;;

wait)
  # Completion is a question about files on disk, not about a live process:
  # this loop can die with the VPN without taking the run with it.
  until [ "$(squeue -u "$USER" -h -n "$JOB_PREFIX-audio,$JOB_PREFIX-image,$JOB_PREFIX-text,$JOB_PREFIX-video" -o %i | wc -l)" -eq 0 ]; do
    sleep 120
  done
  echo "QUEUE DRAINED"
  bash "$0" status
  ;;

*)
  echo "usage: $0 {size [dataset]|chunks [chunk...]|control <embedder> <dataset>...|status|wait}" >&2; exit 1
  ;;
esac
