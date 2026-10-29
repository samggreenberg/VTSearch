#!/usr/bin/env bash
# State of the App: Face (#4762): the face path as it ships, on FHIBE (#4699).
#
# The owner's settings (2026-10-10), the face review's standing recipe:
#   * the #4699 baseline's 562 people (<release>-derived/runs/2026-10-09-k4/identities.txt);
#   * sessions that start from 1 photo and from 4 (the example opening), two session sets;
#   * the face path on image2face crops at both stored sizes: fhibe_faces_ik_1024 and fhibe_faces_ik_640
#     (the face provider's, the arm `ik`, since #5320; FaceNet's fhibe_faces_{1024,640} before it);
#   * one session set per preset (beta 1/4, 1, 4), 150 votes, the full-label ceiling, shipped
#     defaults otherwise; one seed (people are the replication).
# So a review is six run dirs, <date>-sota-k<K>-b<beta>, each 1,124 cells (562 people x 2 sizes),
# and 6,744 runs a seed. Each run dir is an ordinary fhibe/launch.sh run (its grid symlinked
# from the baseline's prepared grid for that K), but the cells of all six go through one pool.
#
#   sota_face.sh prepare                  an arm's own prepared grids, one per K (SOTA_FACE_ARM only)
#   sota_face.sh dirs                     the six run dirs, their grid shapes and preflights
#   sota_face.sh tasks [N]                write the task list (K beta index); N = the first N
#                                         people of each size only (a smoke run), else all;
#                                         SOTA_FACE_SIZES="1024" keeps one size (#4778)
#   sota_face.sh pack [JOBS]              submit the task list as JOBS (default 1) V100 pack jobs
#                                         of CALIB_PACK_PAR (default 38) cells at a time;
#                                         CALIB_PARTITION=cpu packs them on the cpu partition instead
#   sota_face.sh baseline                 click 0: the example sort at K = 1 and 4 (example_baseline.py)
#   sota_face.sh status                   cells written per run dir
#
# Launch from a FROZEN worktree: every task imports the worktree this script sits in.
# Env: SOTA_FACE_DATE (default today), SOTA_FACE_TASKS (the task list's path), SOTA_FACE_TAG (the run
# dirs' tag, default `sota`: <date>-<tag>-k<K>-b<beta>; a study on the same recipe takes its own, e.g.
# #4776's `4776`), SOTA_FACE_CLAIMS_DIR (join an earlier `pack`'s task pool, say cpu packs beside the
# V100 ones, so the two never run the same cell).
#
# SOTA_FACE_ARM=<tag> picks the face arm, fhibe_faces_<tag>_{1024,640}: `ik` (the default, #5321) is the
# face provider's own faces and vectors, the path as it ships since #5320. The retired MTCNN + FaceNet
# arms still run on their existing cells: SOTA_FACE_ARM= (empty) for the plain crops, and #4777's roll
# and sim, the crops turned upright. Its run dirs are <date>-sota-<tag>-k<K>-b<beta>,
# its grids are prepared on its own cells (`prepare`, then `dirs`), and it keeps the review's
# identity file, so every session pairs with the plain arm's by person, size, K and beta.
#
# CALIB_SPOT_CHECK (end | off) runs a study's arm without the app's weak-separation check (#4778);
# `dirs` declares it to the preflight and `pack` pins it in every cell's environment.
#
# SOTA_FACE_BETAS="1" runs fewer presets than the three (#4780). SOTA_FACE_BASE names the prepared
# grids (<base>-k<K>) that `prepare` writes and `dirs` links, so an arm on other vectors under the same
# dataset names (VTS_PILE=<a private pile>, #4780's ArcFace) prepares its own instead of sharing one.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FHIBE_ROOT="${VTS_FHIBE_ROOT:-/expscratch/$USER/fhibe}"
RUNS="$FHIBE_ROOT/${VTS_FHIBE_RELEASE:-fhibe-full-resolution-674a7dcf}-derived/runs"
D="${SOTA_FACE_DATE:-$(date +%Y-%m-%d)}"
ARM="${SOTA_FACE_ARM-ik}"
TAG="${SOTA_FACE_TAG:-sota}"
SET="$D-$TAG${ARM:+-$ARM}"  # the run dirs' common prefix
# The prepared grids the run dirs link to: the #4699 baseline's for the plain arm, the arm's own otherwise.
BASE="${SOTA_FACE_BASE:-${ARM:+$D-prep-$ARM}}"
BASE="${BASE:-2026-10-09}"
IDENTITIES="${SOTA_FACE_IDENTITIES:-$RUNS/2026-10-09-k4/identities.txt}"
DATASETS="fhibe_faces${ARM:+_$ARM}_1024,fhibe_faces${ARM:+_$ARM}_640"
KS=(1 4)
read -r -a BETAS <<<"${SOTA_FACE_BETAS:-0.25 1 4}"  # a study may run fewer presets (#4780: beta 1)
N_PEOPLE=562
# The task index runs size-major in the datasets' order: 0..561 the 1024 px crops, 562..1123 the 640 px.
SIZE_NAMES=(1024 640)
TASKS="${SOTA_FACE_TASKS:-$RUNS/$SET-tasks.txt}"

# The run dir's balance tag, as launch.sh writes it (0.25 -> 025, 1 -> 1, 4 -> 4). A lookup, not a
# python call: the task list asks for it 6,744 times.
btag() {
  case "$1" in
    0.25) echo 025 ;;
    1) echo 1 ;;
    4) echo 4 ;;
    *) python3 -c "import sys; b=float(sys.argv[1]); print(f'{b:g}'.replace('.', '') if b < 1 else f'{b:g}')" "$1" ;;
  esac
}

# The settings one run dir takes, for launch.sh: K and beta vary, the rest is the recipe.
run_env() {  # K BETA
  export FHIBE_EXAMPLES="$1" FHIBE_MAX_EXAMPLES=4 FHIBE_BETA="$2"
  export FHIBE_RUN="$SET" FHIBE_IDENTITIES="$IDENTITIES"
  export FHIBE_DATASETS="$DATASETS" FHIBE_MAX_STEPS=150
  # Dense rank frames: every vote to 10, every 5 after (the skill's F1/F-beta curve recipe).
  export CALIB_RANK_FRAME_STEPS="1,2,3,4,5,6,7,8,9,10,$(seq -s, 15 5 150)"
  export CALIB_JOB_NAME="${TAG}face${ARM:+-$ARM}-k$1-b$(btag "$2")"
}

case "${1:-}" in
prepare)
  [[ -n "$ARM" ]] || { echo "the plain arm links the #4699 baseline's grids; prepare is for SOTA_FACE_ARM" >&2; exit 2; }
  for k in "${KS[@]}"; do
    FHIBE_EXAMPLES="$k" FHIBE_MAX_EXAMPLES=4 FHIBE_RUN="$BASE" FHIBE_IDENTITIES="$IDENTITIES" \
      FHIBE_DATASETS="$DATASETS" bash "$HERE/launch.sh" prepare
  done
  ;;
dirs)
  for k in "${KS[@]}"; do
    for b in "${BETAS[@]}"; do
      (
        run_env "$k" "$b"
        dir="$RUNS/$FHIBE_RUN-k$k-b$(btag "$b")"
        mkdir -p "$dir/results/cells" "$dir/logs"
        chmod 700 "$dir"
        [[ -s "$RUNS/$BASE-k$k/results/prepare_info.json" ]] || { echo "no prepared grid at $RUNS/$BASE-k$k" >&2; exit 3; }
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
  read -r -a sizes <<<"${SOTA_FACE_SIZES:-1024 640}"
  : >"$TASKS"
  skipped=0
  for ((i = 0; i < n; i++)); do
    for size in 0 1; do
      [[ " ${sizes[*]} " == *" ${SIZE_NAMES[size]} "* ]] || continue
      for k in "${KS[@]}"; do
        for b in "${BETAS[@]}"; do
          idx=$((size * N_PEOPLE + i))
          cell="$RUNS/$SET-k$k-b$(btag "$b")/results/cells/$(printf 'task_%04d.csv' "$idx")"
          if [[ -s "$cell" || -s "$cell.gz" ]]; then  # either spelling (CALIB_CELLS_GZIP)
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
  LOGS="$RUNS/$SET-logs"
  mkdir -p "$LOGS" && chmod 700 "$LOGS"
  THREADS="OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2"
  # Every job reads the whole list and claims each task before running it (an atomic mkdir), so a
  # job that starts late, behind the per-user GPU limit, takes what is left rather than a fixed shard.
  claims="${SOTA_FACE_CLAIMS_DIR:-$LOGS/claims-$(date +%s)}"
  mkdir -p "$claims"
  shard="$LOGS/tasks.txt"
  [[ -n "${SOTA_FACE_CLAIMS_DIR:-}" && -s "$shard" ]] || cp "$TASKS" "$shard"
  echo "claims: $claims"
  # The V100 nodes by default (#4762), the GPU hidden; on the cpu partition, the per-user cap applies.
  if [[ "${CALIB_PARTITION:-gpu}" == gpu ]]; then
    WHERE=(--partition=gpu --gres="${CALIB_PACK_GRES:-gpu:v100:1}")
  else
    WHERE=(--partition="$CALIB_PARTITION")
  fi
  for ((j = 0; j < jobs; j++)); do
    J=$(sbatch --parsable --job-name="${TAG}face${ARM:+-$ARM}-pack$j" "${WHERE[@]}" \
      --mem="$((PAR * 3))G" --cpus-per-task="$((PAR * 2))" --time="${CALIB_PACK_TIME:-12:00:00}" \
      --export=ALL --output="$LOGS/pack$j-%j.out" \
      --wrap="export CUDA_VISIBLE_DEVICES= $THREADS SOTA_FACE_DATE=$D SOTA_FACE_TAG=$TAG SOTA_FACE_ARM=$ARM SOTA_FACE_BASE=$BASE \
SOTA_FACE_IDENTITIES=$IDENTITIES SOTA_FACE_CLAIMS=$claims CALIB_SPOT_CHECK=${CALIB_SPOT_CHECK:-} \
SOTA_FACE_RUNNER=${SOTA_FACE_RUNNER:-} HARD_FACE_SWAP=${HARD_FACE_SWAP:-} && \
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
  source "$HERE/../../grid/gridenv.sh" >/dev/null 2>&1
  eval "$ENVX"
  dir="$RUNS/$FHIBE_RUN-k$k-b$(btag "$b")"
  cd "$HERE/../calibration"
  # SOTA_FACE_RUNNER: a study's own cell runner in place of run_cells.py (#5376's swapped examples).
  python "${SOTA_FACE_RUNNER:-run_cells.py}" --index "$idx" >"$dir/logs/cell-$idx.out" 2>&1 || echo "FAILED k$k b$b $idx"
  ;;
baseline)  # click 0: the example sort at K = 1 and 4, in the K=1 beta-1 run's environment
  run_env 1 1
  ENVX=$(bash -c "source $HERE/launch.sh status >/dev/null 2>&1; echo \"\$ENVX\"")
  source "$HERE/../../grid/gridenv.sh" >/dev/null 2>&1
  eval "$ENVX"
  out="$RUNS/$SET-baseline"
  mkdir -p "$out" && chmod 700 "$out"
  cd "$HERE"
  python example_baseline.py --results "$RUNS/$SET-k1-b1/results" --ks 1,4 --out-dir "$out" \
    --check "$RUNS/$SET-k{k}-b1/results/cells"
  ;;
status)
  for k in "${KS[@]}"; do
    for b in "${BETAS[@]}"; do
      dir="$RUNS/$SET-k$k-b$(btag "$b")"
      echo "k$k b$(btag "$b"): $(ls "$dir/results/cells" 2>/dev/null | grep -cE '^task_[0-9]+\.csv(\.gz)?$' || true) cells"
    done
  done
  ;;
*)
  echo "usage: sota_face.sh {prepare | dirs | tasks [N] | pack [JOBS] | baseline | status}" >&2; exit 1 ;;
esac
