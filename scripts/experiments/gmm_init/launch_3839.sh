#!/usr/bin/env bash
# What to do about the anchored refits that still exit on max_iter (#3839).
#
#   bash launch_3839.sh gate       # the #3585 fold corpus through every arm in arms_3839.py, sharded
#   AB_ARMS=cap2000 bash launch_3839.sh ab   # the trajectory A/B for the gate's pick
#   AB_ARMS=cap2000 bash launch_3839.sh abanalyze  # pair it with #3840's v_ll1e-8 grid
#   bash launch_3839.sh analyse    # the tables and figures
#   bash launch_3839.sh status
#
# NO CAPTURE AND NO NEW CORPUS.  The 84 cells #3585 captured are every array
# the shipped fold path saw; #3825 gated its stopping rules on them without
# re-running a cell, and so does this.  The trajectory A/B runs only the arm the
# gate picks, on #3840's validation cells (seeds 0-8), and pairs it with that
# study's `v_ll1e-8` grid - the shipped rule on the same code - rather than
# running a second baseline; #3840 is also what sized it.
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- NOTHING WAS SUBMITTED" >&2' ERR

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
EXP="${MAXITER_EXP:-/expscratch/$USER/maxiter-3839}"
CORPUS="${CORPUS:-/expscratch/$USER/gmm-3585/corpus}"
ANALYSIS="$EXP/analysis"
SHARDS="${GATE_SHARDS:-12}"
LOGS="$EXP/logs"
mkdir -p "$LOGS" "$ANALYSIS"

DEP_ARG=()
[[ -n "${DEP:-}" ]] && DEP_ARG=(--dependency="$DEP")
# BLAS pinned so refit_seconds prices the loop rather than a thread pool (#3166).
PIN="export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"

submit() {
  local name="$1"; shift
  local J
  J=$(sbatch --parsable "$@") || { echo "SUBMIT FAILED for $name" >&2; return 1; }
  [[ "$J" =~ ^[0-9]+$ ]] || { echo "$name SUBMIT FAILED (empty job id)" >&2; return 1; }
  echo "$J" > "$LOGS/.jobid_$name"
  echo "$name -> job $J"
}

case "${1:-status}" in
gate)
  [[ -d "$CORPUS" ]] || { echo "no corpus at $CORPUS" >&2; exit 3; }
  submit gate --job-name=maxiter-3839-gate "${DEP_ARG[@]}" --array="0-$((SHARDS-1))" \
    --partition=cpu --mem=8G --cpus-per-task=1 --time=3:00:00 --output="$LOGS/gate-%A_%a.out" \
    --wrap="source $WT/gridenv.sh && $PIN && cd $HERE && python gate_3839.py --corpus $CORPUS --out $ANALYSIS --shard \$SLURM_ARRAY_TASK_ID/$SHARDS"
  ;;

ab)
  # The trajectory A/B for the arm the gate recommends, on #3840's validation
  # cells (seeds 0-8 of the #3585 environments).  Its baseline is #3840's
  # `v_ll1e-8` grid - the shipped rule, the same dev, the same cells - so only
  # the candidate is run here.  AB_ARMS names it.
  : "${AB_ARMS:?set AB_ARMS to the candidate arm(s)}"
  CALIB_EXP="$EXP" CALIB_N_SEEDS="${CALIB_N_SEEDS:-9}" CALIB_CONC="${CALIB_CONC:-16}" CALIB_CPUS=1 CALIB_MEM=7G \
    CALIB_JOB_NAME=maxiter-3839 AB_RUNNER=run_cells_arm_3839.py AB_ARMS="$AB_ARMS" \
    bash "$HERE/launch_3825.sh" ab
  ;;

abanalyze)
  # Pair the candidate with #3840's `v_ll1e-8` grid (the shipped rule, same dev,
  # same seeds 0-8) through the standard analyzer, then pool it.
  : "${AB_ARMS:?set AB_ARMS to the candidate arm(s)}"
  OFF="${AB_OFF:-/expscratch/$USER/abres-3840/ab_ll1e-8/results}"
  CALIB="$WT/scripts/experiments/calibration"
  for arm in $AB_ARMS; do
    submit "abanalyze_$arm" --job-name="maxiter-3839-abanalyze-$arm" "${DEP_ARG[@]}" --partition=cpu --mem=48G \
      --cpus-per-task=2 --time=2:00:00 --output="$LOGS/abanalyze-$arm-%j.out" \
      --wrap="source $WT/gridenv.sh && export CALIB_AB_ON=$EXP/ab_$arm/results CALIB_AB_OFF=$OFF CALIB_AB_OUT=$ANALYSIS/ab_$arm && mkdir -p $ANALYSIS/ab_$arm && cd $CALIB && python analyze_ab.py && cd $HERE && python ab_summary_3839.py --paired $ANALYSIS/ab_$arm/agg/ab_paired_cells.csv --out $ANALYSIS/ab_$arm"
  done
  ;;

analyse)
  submit analyse --job-name=maxiter-3839-analyse "${DEP_ARG[@]}" --partition=cpu --mem=16G --cpus-per-task=2 \
    --time=1:00:00 --output="$LOGS/analyse-%j.out" \
    --wrap="source $WT/gridenv.sh && cd $HERE && python analyze_3839.py --analysis $ANALYSIS --shards $SHARDS && python figures_3839.py --analysis $ANALYSIS"
  ;;

status)
  echo "shards written: $(ls "$ANALYSIS"/gate3839_cuts.*.csv 2>/dev/null | wc -l)/$SHARDS"
  squeue -u "$USER" -h -o "%.14i %.28j %.8T %.10M" | grep maxiter || true
  ;;

*)
  echo "usage: $0 {gate|analyse|status}" >&2
  exit 1
  ;;
esac
