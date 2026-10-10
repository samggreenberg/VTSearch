#!/usr/bin/env bash
# State of the App: Face (#4762): the face path as it ships, on FHIBE (#4699).
#
# The owner's settings (2026-10-10), the face review's standing recipe:
#   * the #4699 baseline's 562 people (<release>-derived/runs/2026-10-09-k4/identities.txt);
#   * sessions that start from 1 photo and from 4 (the example opening), two session sets;
#   * FaceNet on image2face crops at both stored sizes: fhibe_faces_1024 and fhibe_faces_640;
#   * one session set per preset (beta 1/4, 1, 4), 150 votes, the full-label ceiling, shipped
#     defaults otherwise; one seed (people are the replication).
# So a review is six run dirs, <date>-sota-k<K>-b<beta>, each 1,124 cells (562 people x 2 sizes),
# and 6,744 runs a seed. Each run dir is an ordinary fhibe/launch.sh run (its grid symlinked
# from the baseline's prepared grid for that K), but the cells of all six go through one pool.
#
#   sota_face.sh dirs                     the six run dirs, their grid shapes and preflights
#   sota_face.sh tasks [N]                write the task list (K beta index); N = the first N
#                                         people of each size only (a smoke run), else all
#   sota_face.sh pack [JOBS]              submit the task list as JOBS (default 1) V100 pack jobs
#                                         of CALIB_PACK_PAR (default 38) cells at a time
#   sota_face.sh baseline                 click 0: the example sort at K = 1 and 4 (example_baseline.py)
#   sota_face.sh status                   cells written per run dir
#
# Launch from a FROZEN worktree: every task imports the worktree this script sits in.
# Env: SOTA_FACE_DATE (default today), SOTA_FACE_TASKS (the task list's path).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FHIBE_ROOT="${VTS_FHIBE_ROOT:-/expscratch/$USER/fhibe}"
RUNS="$FHIBE_ROOT/${VTS_FHIBE_RELEASE:-fhibe-full-resolution-674a7dcf}-derived/runs"
D="${SOTA_FACE_DATE:-$(date +%Y-%m-%d)}"
BASE="${SOTA_FACE_BASE:-2026-10-09}"
KS=(1 4)
BETAS=(0.25 1 4)
N_PEOPLE=562
TASKS="${SOTA_FACE_TASKS:-$RUNS/$D-sota-tasks.txt}"

btag() { python3 -c "import sys; b=float(sys.argv[1]); print(f'{b:g}'.replace('.', '') if b < 1 else f'{b:g}')" "$1"; }

# The settings one run dir takes, for launch.sh: K and beta vary, the rest is the recipe.
run_env() {  # K BETA
  export FHIBE_EXAMPLES="$1" FHIBE_MAX_EXAMPLES=4 FHIBE_BETA="$2"
  export FHIBE_RUN="$D-sota" FHIBE_IDENTITIES="$RUNS/$BASE-k4/identities.txt"
  export FHIBE_DATASETS=fhibe_faces_1024,fhibe_faces_640 FHIBE_MAX_STEPS=150
  # Dense rank frames: every vote to 10, every 5 after (the skill's F1/F-beta curve recipe).
  export CALIB_RANK_FRAME_STEPS="1,2,3,4,5,6,7,8,9,10,$(seq -s, 15 5 150)"
  export CALIB_JOB_NAME="sotaface-k$1-b$(btag "$2")"
}

case "${1:-}" in
dirs)
  for k in "${KS[@]}"; do
    for b in "${BETAS[@]}"; do
      (
        run_env "$k" "$b"
        dir="$RUNS/$FHIBE_RUN-k$k-b$(btag "$b")"
        mkdir -p "$dir/results/cells" "$dir/logs"
        chmod 700 "$dir"
        ln -sfn "$RUNS/$BASE-k$k/results/prepare_info.json" "$dir/results/prepare_info.json"
        ln -sfn "$RUNS/$BASE-k$k/results/crops" "$dir/results/crops"
        bash "$HERE/launch.sh" shape | tail -2
      )
    done
  done
  ;;
tasks)
  # Person-major, so a grid stopped early holds the same people in every session set; a cell
  # already written is skipped (run_cells does not skip it).
  n="${2:-$N_PEOPLE}"
  : >"$TASKS"
  skipped=0
  for ((i = 0; i < n; i++)); do
    for size in 0 1; do
      for k in "${KS[@]}"; do
        for b in "${BETAS[@]}"; do
          idx=$((size * N_PEOPLE + i))
          if [[ -s "$RUNS/$D-sota-k$k-b$(btag "$b")/results/cells/$(printf 'task_%04d.csv' "$idx")" ]]; then
            skipped=$((skipped + 1))
          else
            echo "$k $b $idx" >>"$TASKS"
          fi
        done
      done
    done
  done
  chmod 600 "$TASKS"
  echo "$(wc -l <"$TASKS") tasks -> $TASKS ($skipped already written)"
  ;;
pack)
  jobs="${2:-1}"
  [[ -s "$TASKS" ]] || { echo "no task list at $TASKS (run tasks first)" >&2; exit 3; }
  PAR="${CALIB_PACK_PAR:-38}"
  LOGS="$RUNS/$D-sota-logs"
  mkdir -p "$LOGS" && chmod 700 "$LOGS"
  THREADS="OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2"
  # Every job reads the whole list and claims each task before running it (an atomic mkdir), so a
  # job that starts late, behind the per-user GPU limit, takes what is left rather than a fixed shard.
  claims="$LOGS/claims-$(date +%s)"
  mkdir -p "$claims"
  shard="$LOGS/tasks.txt"
  cp "$TASKS" "$shard"
  for ((j = 0; j < jobs; j++)); do
    J=$(sbatch --parsable --job-name="sotaface-pack$j" --partition=gpu --gres="${CALIB_PACK_GRES:-gpu:v100:1}" \
      --mem="$((PAR * 3))G" --cpus-per-task="$((PAR * 2))" --time="${CALIB_PACK_TIME:-12:00:00}" \
      --export=ALL --output="$LOGS/pack$j-%j.out" \
      --wrap="export CUDA_VISIBLE_DEVICES= $THREADS SOTA_FACE_DATE=$D SOTA_FACE_CLAIMS=$claims && \
xargs -P $PAR -L 1 bash $HERE/sota_face.sh cell <$shard; echo pack done") || { echo "pack $j SUBMIT FAILED" >&2; exit 1; }
    [[ "$J" =~ ^[0-9]+$ ]] || { echo "pack $j SUBMIT FAILED (empty job id)" >&2; exit 1; }
    echo "pack $j -> job $J (claims from $(wc -l <"$shard") cells, $PAR at a time)"
  done
  ;;
cell)  # K BETA INDEX: one cell, in its run dir's environment (called by pack)
  k="$2"; b="$3"; idx="$4"
  if [[ -n "${SOTA_FACE_CLAIMS:-}" ]]; then
    mkdir "$SOTA_FACE_CLAIMS/k$k-b$b-$idx" 2>/dev/null || exit 0  # another job has it
  fi
  run_env "$k" "$b"
  ENVX=$(bash -c "source $HERE/launch.sh status >/dev/null 2>&1; echo \"\$ENVX\"")
  source "$HERE/../../../gridenv.sh" >/dev/null 2>&1
  eval "$ENVX"
  dir="$RUNS/$FHIBE_RUN-k$k-b$(btag "$b")"
  cd "$HERE/../calibration"
  python run_cells.py --index "$idx" >"$dir/logs/cell-$idx.out" 2>&1 || echo "FAILED k$k b$b $idx"
  ;;
baseline)  # click 0: the example sort at K = 1 and 4, in the K=1 beta-1 run's environment
  run_env 1 1
  ENVX=$(bash -c "source $HERE/launch.sh status >/dev/null 2>&1; echo \"\$ENVX\"")
  source "$HERE/../../../gridenv.sh" >/dev/null 2>&1
  eval "$ENVX"
  out="$RUNS/$D-sota-baseline"
  mkdir -p "$out" && chmod 700 "$out"
  cd "$HERE"
  python example_baseline.py --results "$RUNS/$D-sota-k1-b1/results" --ks 1,4 --out-dir "$out" \
    --check "$RUNS/$D-sota-k{k}-b1/results/cells"
  ;;
status)
  for k in "${KS[@]}"; do
    for b in "${BETAS[@]}"; do
      dir="$RUNS/$D-sota-k$k-b$(btag "$b")"
      echo "k$k b$(btag "$b"): $(ls "$dir/results/cells" 2>/dev/null | grep -c '^task_[0-9]*\.csv$' || true) cells"
    done
  done
  ;;
*)
  echo "usage: sota_face.sh {dirs | tasks [N] | pack [JOBS] | baseline | status}" >&2; exit 1 ;;
esac
