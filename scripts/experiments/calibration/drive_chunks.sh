#!/usr/bin/env bash
# Run several run-level arms' cells in small chunks, so the queue never holds more than a few hundred of the
# cluster's job records (#4668, #4701: MaxJobCount counts every array task, and sixteen whole 720-task arrays
# held 85% of the GRID's 10,000 on 2026-10-08).
#
#   LAUNCHER=/abs/launch_X.sh BASE=/expscratch/$USER/X JOB_PREFIX=x RUNGS="a_b025 a_b1 ..." \
#     DRAIN_BY="2026-10-09 20:00" nohup bash drive_chunks.sh > $BASE/drive.out 2>&1 &
#
# The launcher must define rung_env ARM, set_exp ARM (setting CALIB_EXP, CALIB_RESULTS, CALIB_JOB_NAME =
# "$JOB_PREFIX-ARM", ENVX, WT, HERE) and link_prepare, and a `plan` mode that only prints - sourcing it with
# `plan` loads them. Run each arm's preflight before starting this.
#
# Per arm: at most PER_RUNG tasks running. An arm gets its next CHUNK of missing indices (in index order, so a
# seed-major grid completes seed by seed) once its queued chunk has no pending task left; the chunk's throttle
# is what the arm's running tasks leave of PER_RUNG. An index is done when its task_NNNN.csv.gz exists
# (header-only = starved, still done). One that leaves the queue with no file is retried once, then logged as
# FAILED and skipped; a zero-byte file is removed before its retry (resume would read it as done).
#
# Exits on: every arm done ("done"), or DRAIN_BY ("deadline"). Stop by hand: kill $(cat $BASE/drive.pid).
# Log: $BASE/drive.log.
set -uo pipefail
: "${LAUNCHER:?LAUNCHER (absolute path of the arm launcher)}" "${BASE:?BASE}" "${JOB_PREFIX:?JOB_PREFIX}" "${RUNGS:?RUNGS}"
: "${DRAIN_BY:?DRAIN_BY (a date -d string: when to stop submitting)}"
PER_RUNG=${PER_RUNG:-6}
CHUNK=${CHUNK:-48}
N=${N:-720}
DRAIN_AT=$(date -d "$DRAIN_BY" +%s)
echo $$ > "$BASE/drive.pid"
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$BASE/drive.log"; }
declare -A TRIES
log "start (pid $$) per_rung=$PER_RUNG chunk=$CHUNK n=$N arms: $RUNGS"

done_set() {  # the main frames only (task_NNNN.csv[.gz]); side frames carry a `__`
  find "$BASE/$1/results/cells" -maxdepth 1 \( -name 'task_[0-9][0-9][0-9][0-9].csv' -o -name 'task_[0-9][0-9][0-9][0-9].csv.gz' \) \
    -size +0 2>/dev/null | xargs -r -n1 basename | cut -c6-9 | sed -E 's/^0+([0-9])/\1/' | sort -n -u
}
queued_set() {
  squeue -u "$USER" -h -r -n "$JOB_PREFIX-$1" -o "%K" 2>/dev/null | grep -E '^[0-9]+$' | sort -n -u
}

while :; do
  all_done=1
  for r in $RUNGS; do
    d=$(done_set "$r")
    q=$(queued_set "$r")
    ndone=$(grep -c . <<<"$d")
    ((ndone >= N)) && continue
    all_done=0
    running=$(squeue -u "$USER" -h -r -n "$JOB_PREFIX-$r" -t R -o %i 2>/dev/null | wc -l)
    pending=$(squeue -u "$USER" -h -r -n "$JOB_PREFIX-$r" -t PD -o %i 2>/dev/null | wc -l)
    ((pending > 0)) && continue
    list=()
    while read -r i; do
      [[ -z "$i" ]] && continue
      if [[ -n "${TRIES[$r:$i]:-}" ]] && ((TRIES[$r:$i] >= 2)); then continue; fi
      list+=("$i")
      ((${#list[@]} >= CHUNK)) && break
    done < <(comm -23 <(seq 0 $((N - 1)) | sort -u) <(printf '%s\n%s\n' "$d" "$q" | grep -E '^[0-9]+$' | sort -u) | sort -n)
    ((${#list[@]} == 0)) && continue
    for i in "${list[@]}"; do
      TRIES[$r:$i]=$((${TRIES[$r:$i]:-0} + 1))
      ((TRIES[$r:$i] == 2)) && log "$r: retrying index $i"
      cell=$(printf 'task_%04d' "$i")
      for z in "$BASE/$r/results/cells/$cell.csv" "$BASE/$r/results/cells/$cell.csv.gz"; do
        if [[ -f "$z" && ! -s "$z" ]]; then
          log "$r: removing zero-byte $(basename "$z")"
          rm -f "$z"
        fi
      done
    done
    thr=$((PER_RUNG - running))
    ((thr < 1)) && thr=1
    spec=$(
      IFS=,
      echo "${list[*]}"
    )
    out=$(bash -c "source $LAUNCHER plan >/dev/null 2>&1; rung_env $r; set_exp $r; link_prepare; \
      sbatch --parsable --job-name=\"\$CALIB_JOB_NAME\" --array=$spec%$thr --mem=\"\$CALIB_MEM\" --cpus-per-task=1 \
        --time=\"\$CALIB_TIME\" --partition=cpu --export=ALL --output=\"\$CALIB_EXP/logs/cells-%A_%a.out\" \
        --wrap=\"source \$WT/gridenv.sh && \$ENVX && cd \$HERE && python run_cells.py\"" 2>&1 | tail -1)
    if [[ "$out" =~ ^[0-9]+$ ]]; then
      log "$r: submitted ${#list[@]} (${list[0]}..${list[-1]}) %$thr -> $out  [done $ndone]"
    else
      log "$r: submit REFUSED: $out"
      for i in "${list[@]}"; do TRIES[$r:$i]=$((TRIES[$r:$i] - 1)); done
    fi
  done
  if ((all_done)); then
    log "done"
    break
  fi
  if (($(date +%s) >= DRAIN_AT)); then
    log "deadline: stopping (queued tasks keep running)"
    break
  fi
  # Raise each arm's pending chunk throttle as its older tasks finish (never above PER_RUNG in total).
  for r in $RUNGS; do
    jobs=$(squeue -u "$USER" -h -n "$JOB_PREFIX-$r" -t PD -o "%F" 2>/dev/null | sort -u)
    [[ -z "$jobs" ]] && continue
    running=$(squeue -u "$USER" -h -r -n "$JOB_PREFIX-$r" -t R -o %i 2>/dev/null | wc -l)
    for j in $jobs; do
      want=$((PER_RUNG - running + $(squeue -j "$j" -h -r -t R -o %i 2>/dev/null | wc -l)))
      ((want < 1)) && want=1
      scontrol update JobId="$j" ArrayTaskThrottle="$want" >/dev/null 2>&1
    done
  done
  sleep 60
done
gave_up=0
for k in "${!TRIES[@]}"; do
  ((TRIES[$k] >= 2)) || continue
  r=${k%%:*}
  i=${k##*:}
  cell=$(printf 'task_%04d' "$i")
  if [[ ! -s "$BASE/$r/results/cells/$cell.csv.gz" && ! -s "$BASE/$r/results/cells/$cell.csv" ]]; then
    log "FAILED twice: $k"
    gave_up=$((gave_up + 1))
  fi
done
log "exit (gave up on $gave_up)"
