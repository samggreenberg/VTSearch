#!/usr/bin/env bash
# #4428: sweep the acquisition factor below (and above) the shipped 0.5, on dev's head ee6605262 (frozen
# worktree vts-sweep-run). Arms (Binary SigLIP, coco_better 144 cells):
#   150 clicks, 5 seeds: b0.5-x0.25, b1-x0.25, b2-x0.25   (the shipped x0.5 and x1.0 are today's #4409 arms)
#   300 clicks, 3 seeds, beta 1: b1-x0.25-c300, b1-xship-c300 (the shipped default, x0.5), b1-x0.75-c300
# Each explicit factor runs with CALIB_ACQ_INCLUSION_OFFSET=0 (the harness's rule); the shipped arm runs the
# default (its no-estimate fallback is line - 4). Stop it: kill driver3.pid FIRST, then scancel sweep-*.
set -uo pipefail
R=/expscratch/sgreenberg/acq-sweep-4428
SOTA=/expscratch/sgreenberg/worktrees/vts-sweep-run/scripts/experiments/state_of_app
echo $$ > $R/driver3.pid
log() { echo "$(date '+%F %T') $*" >> $R/driver3.log; }
export CALIB_MEM=4G CALIB_CPUS=1 CALIB_CONC=230 CALIB_CELL_ORDER=seed CALIB_TIME=2:00:00
declare -a ARMS JOBS
# arm  beta  factor(or ship)  steps  seeds
ARMS=("b0.5-x0.25 0.5 0.25 150 5" "b1-x0.25 1 0.25 150 5" "b2-x0.25 2 0.25 150 5"
      "b1-x0.25-c300 1 0.25 300 3" "b1-xship-c300 1 ship 300 3" "b1-x0.75-c300 1 0.75 300 3")
for spec in "${ARMS[@]}"; do
  read -r arm B X STEPS N <<<"$spec"
  export CALIB_EXP=$R/$arm CALIB_BETA=$B CALIB_JOB_NAME=sweep-$arm SOTA_DATE=sweep-$arm SOTA_SEEDS=$N SOTA_MAX_STEPS=$STEPS
  unset CALIB_MIN_PRECISION CALIB_ACQ_INCLUSION_OFFSET CALIB_ACQ_P_CROSSING
  if [[ $X != ship ]]; then export CALIB_ACQ_INCLUSION_OFFSET=0 CALIB_ACQ_P_CROSSING=$X; fi
  if (( STEPS > 150 )); then
    export SOTA_RANK_FRAME_STEPS=1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,175,200,225,250,275,300
  else
    export SOTA_RANK_FRAME_STEPS=1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150
  fi
  mkdir -p $CALIB_EXP
  out=$(cd "$SOTA" && srun --ntasks=1 -p cpu --mem=8G -c 2 -t 60 --quiet bash launch.sh prepare 2>&1)
  PJ=$(grep -oP 'prepare -> job \K[0-9]+' <<<"$out"); [[ -n "$PJ" ]] || { log "$arm prepare FAILED: $(tail -2 <<<"$out")"; exit 1; }
  while [[ -n "$(squeue -j $PJ -h 2>/dev/null)" ]]; do sleep 20; done
  [[ "$(sacct -j $PJ -n -X --format=State | tr -d ' ')" == COMPLETED ]] || { log "$arm prepare failed"; exit 1; }
  IDX=$(python3 -c "import sys; n=int(sys.argv[1]); print(','.join(f'{s*288}-{s*288+143}' for s in range(n)))" "$N")
  while (( $(squeue -u $USER -h -r 2>/dev/null | wc -l) > 1200 )); do sleep 60; done   # the ~2000-job queue cap
  out=$(cd "$SOTA" && bash launch.sh redo "$IDX" 2>&1)
  J=$(grep -oP 'redo -> job \K[0-9]+' <<<"$out"); [[ -n "$J" ]] || { log "$arm SUBMIT FAILED: $(tail -1 <<<"$out")"; exit 1; }
  JOBS+=("$J"); log "$arm (beta $B, acq $X, $STEPS clicks) seeds 0-$((N-1)) -> job $J"
done
for J in "${JOBS[@]}"; do while [[ -n "$(squeue -j $J -h 2>/dev/null)" ]]; do sleep 120; done; done
log "all arms drained: $(for J in "${JOBS[@]}"; do sacct -j $J -n -X --format=State | sort | uniq -c | tr -s ' ' | paste -sd' '; done | paste -sd';')"
log "done"
