#!/usr/bin/env bash
# The validation grid for the A/B cells-to-resolution curve (#3840).
#
#   bash launch_3840.sh ab         # seeds 0..8 of `baseline` and `ll1e-8`, plus a same-code repeat
#   bash launch_3840.sh frames     # collapse every grid to the compact per-step frame
#   bash launch_3840.sh analyse    # the curve, the breakdowns, the check against the prediction
#   bash launch_3840.sh abfigures  # quality-over-clicks pair + viewer for the validation grid
#   bash launch_3840.sh status
#
# WHY A RUN AT ALL.  The curve itself is re-analysis: subsample the paired-cell
# frames #3585 and #3825 already hold.  But a curve estimated from 114 cells and
# checked against those same 114 cells is a bootstrap checked against itself.
# So the prediction is written down first (docs/experiments/
# 2026-09-22-ab-resolution-3840/PLAN.md, committed before this submits) and a
# grid of the size it names is run on cells the curve never saw.
#
# WHAT RUNS.  #3825's own pair - the parameter rule it replaced (`baseline`)
# against the rule it shipped (`ll1e-8`) - through #3825's own runner, on the
# #3585 environments at CALIB_N_SEEDS=9.  Seeds 0-1 are the cells #3825 ran on
# 2026-09-13; seeds 2-8 (399 paired cells) are new and are the validation.  The
# re-run of seeds 0-1 is not waste: against the 2026-09-13 grids it says whether
# eleven days of dev moved a trajectory, and `rep` - the same arm on the same
# code a second time - says whether the harness is deterministic at all, which
# is the assumption every "the arm caused this Δ" reading rests on.
#
# CPU only.  A 2-seed grid was 10.5 task-hours and 21 minutes wall at %40 on
# 2026-09-13 (job 648961, MaxRSS 5.2 GB), so this is ~105 task-hours.
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- check what was submitted" >&2' ERR

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
EXP="${ABRES_EXP:-/expscratch/$USER/abres-3840}"
G=/expscratch/$USER
LOGS="$EXP/logs"
mkdir -p "$LOGS"

# The grids the curve is built from (2026-09-13) and the ones this issue adds.
EXISTING=(
  "sklearn=$G/gmm-3585/ab_baseline/results"
  "native=$G/gmm-3585/ab_native/results"
  "baseline=$G/anchem-3825/ab_baseline/results"
  "ll1e-3=$G/anchem-3825/ab_ll1e-3/results"
  "ll1e-6=$G/anchem-3825/ab_ll1e-6/results"
  "ll1e-8=$G/anchem-3825/ab_ll1e-8/results"
)
NEW=(
  "v_baseline=$EXP/ab_baseline/results"
  "v_ll1e-8=$EXP/ab_ll1e-8/results"
  "v_rep=$EXP/rep/ab_baseline/results"
)

case "${1:-status}" in
ab)
  CALIB_EXP="$EXP" CALIB_N_SEEDS=9 CALIB_CONC="${CALIB_CONC:-30}" CALIB_JOB_NAME=abres-3840 \
    AB_ARMS="baseline ll1e-8" bash "$WT/scripts/experiments/gmm_init/launch_3825.sh" ab
  CALIB_EXP="$EXP/rep" CALIB_N_SEEDS=2 CALIB_CONC="${CALIB_CONC:-30}" CALIB_JOB_NAME=abres-3840-rep \
    AB_ARMS="baseline" bash "$WT/scripts/experiments/gmm_init/launch_3825.sh" ab
  ;;

frames)
  DEP_ARG=()
  [[ -n "${DEP:-}" ]] && DEP_ARG=(--dependency="$DEP")
  ARGS=""
  for spec in "${EXISTING[@]}" "${NEW[@]}"; do ARGS="$ARGS --grid $spec"; done
  J=$(sbatch --parsable --job-name=abres-3840-frames "${DEP_ARG[@]}" --partition=cpu --mem=48G --cpus-per-task=2 \
    --time=2:00:00 --output="$LOGS/frames-%j.out" \
    --wrap="source $WT/gridenv.sh && cd $HERE && python frames_3840.py $ARGS --out $EXP/frames/steps.csv.gz")
  [[ "$J" =~ ^[0-9]+$ ]] || { echo "frames SUBMIT FAILED" >&2; exit 1; }
  echo "frames -> job $J"
  ;;

analyse)
  DEP_ARG=()
  [[ -n "${DEP:-}" ]] && DEP_ARG=(--dependency="$DEP")
  J=$(sbatch --parsable --job-name=abres-3840-analyse "${DEP_ARG[@]}" --partition=cpu --mem=16G --cpus-per-task=2 \
    --time=2:00:00 --output="$LOGS/analyse-%j.out" \
    --wrap="source $WT/gridenv.sh && cd $HERE && python resolution_3840.py --steps $EXP/frames/steps.csv.gz --out $EXP/analysis && python figures_3840.py --analysis $EXP/analysis")
  [[ "$J" =~ ^[0-9]+$ ]] || { echo "analyse SUBMIT FAILED" >&2; exit 1; }
  echo "analyse -> job $J"
  ;;

abfigures)
  # The quality-over-clicks pair and the viewer every simulated-user study owes,
  # for the validation grid.  Click 0 is the free text sort, computed on the
  # validation cells themselves (seeds 0-8), not borrowed from #3585's 2 seeds.
  CALIB="$WT/scripts/experiments/calibration"
  ARMS_ROOT="$EXP/arms"
  mkdir -p "$ARMS_ROOT" "$EXP/analysis/figures"
  ln -sfn "$EXP/ab_baseline/results" "$ARMS_ROOT/baseline"
  ln -sfn "$EXP/ab_ll1e-8/results" "$ARMS_ROOT/ll1e-8"
  source "$WT/scripts/experiments/pile/pile_env.sh"
  J=$(sbatch --parsable --job-name=abres-3840-abfig --partition=cpu --mem=24G --cpus-per-task=2 --time=3:00:00 \
    --output="$LOGS/abfig-%j.out" \
    --wrap="source $WT/gridenv.sh && export VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME CALIB_RESULTS=$EXP/ab_baseline/results CALIB_DATASETS=visual_genome_m,caltech101_m,coco_val CALIB_VG_EMBEDDERS=siglip,dinov3_patch CALIB_CALTECH_EMBEDDERS=siglip,dinov3_patch CALIB_COCO_EMBEDDERS=siglip,dinov3_patch CALIB_PATCH_STYLES=whole_image,max_patch CALIB_CATEGORY_MODE=all CALIB_N_SEEDS=9 CALIB_CELL_ORDER=seed && cd $CALIB && python text_baseline.py --results $EXP/ab_baseline/results --out $EXP/analysis/text_baseline.csv && python curves.py --results $ARMS_ROOT --arms baseline,ll1e-8 --out $EXP/analysis/figures --baseline $EXP/analysis/text_baseline.csv && python viewer.py --results $ARMS_ROOT --arms baseline=baseline,ll1e-8=ll1e-8 --out $EXP/analysis/viewer.html --title 'The #3840 validation grid: parameter rule vs ll1e-8' --baseline $EXP/analysis/text_baseline.csv")
  [[ "$J" =~ ^[0-9]+$ ]] || { echo "abfigures SUBMIT FAILED" >&2; exit 1; }
  echo "abfigures -> job $J"
  ;;

status)
  for d in "$EXP/ab_baseline" "$EXP/ab_ll1e-8" "$EXP/rep/ab_baseline"; do
    echo "$d: $(ls "$d/results/cells" 2>/dev/null | grep -c -v __) task files"
  done
  squeue -u "$USER" -h -o "%.10i %.28j %.8T %.10M" | grep abres || true
  ;;

*)
  echo "usage: $0 {ab|frames|analyse|status}" >&2
  exit 1
  ;;
esac
