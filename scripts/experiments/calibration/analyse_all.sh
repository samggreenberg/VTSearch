#!/usr/bin/env bash
# Everything the #3156 map needs, chained GRID-side so it survives a
# disconnect. Each step is independent and non-fatal: a step that fails must
# not cost the ones after it, and its absence has to be visible in the log
# rather than inferred from a missing file.
set -u
WT=/exp/sgreenberg/projects/vts-sd-pair2
EXP=${CALIB_EXP:-/expscratch/sgreenberg/scale-3156-fixed}
# Overridable so this can be smoke-tested against a partial grid without
# leaving a half-finished REPORT where the real one belongs -- a stale report
# in the expected place is worse than no report at all.
OUT=${ANALYSE_OUT:-$EXP/analysis}
FIGS=$OUT/figures
mkdir -p "$OUT" "$FIGS"

source "$WT/gridenv.sh"
source "$WT/scripts/experiments/pile/pile_env.sh"
export VTS_REPO="$WT"
cd "$WT/scripts/experiments/calibration" || exit 2
# Unbuffered: a step that prints nothing for ten minutes is indistinguishable
# from a hung one, and nobody is watching this job when it runs.
export PYTHONUNBUFFERED=1

step() {
  local name="$1"; shift
  echo; echo "=== $name  ($(date +%H:%M)) ==="
  if "$@"; then echo "--- $name OK"; else echo "--- $name FAILED (rc=$?)"; fi
}

CELLS=$(ls "$EXP/results/cells" 2>/dev/null | grep -c '^task_[0-9]*\.csv$')
echo "cells present: $CELLS / 6480"

step overview      bash -c "python analyze_overview.py --exp $EXP > $OUT/REPORT_overview.txt 2>&1; tail -5 $OUT/REPORT_overview.txt"
step scale         bash -c "python analyze_scale.py --exp $EXP > $OUT/REPORT_scale.txt 2>&1; tail -5 $OUT/REPORT_scale.txt"
step phases       bash -c "python analyze_phases.py --exp $EXP > $OUT/REPORT_phases.txt 2>&1; tail -8 $OUT/REPORT_phases.txt"
step tail_overlap  bash -c "python analyze_tail_overlap.py --exp $EXP --metric average_precision > $OUT/REPORT_tail.txt 2>&1; tail -6 $OUT/REPORT_tail.txt"
step figs_overview python figures_overview.py --exp "$EXP" --out "$FIGS"
step figs_scale    python figures_scale.py --exp "$EXP" --out "$FIGS"
step curves        python figures_trajectory.py --exp "$EXP" --out "$FIGS" --metric cost,average_precision

# Sessions across the landscape rather than one cherry-picked cell: three
# classes that behaved differently in the preview, at both ends of the size
# axis, two seeds each -- so a claim about "how it does" can be checked by eye
# on a case nobody chose after seeing the result.
for cat in "bird@small" "bird@large" "stop sign@small" "stop sign@large" "backpack@medium" "clock@medium"; do
  for seed in 0 1; do
    slug=$(echo "$cat" | tr ' @' '__')
    step "session_${slug}_s${seed}" python pick_sheets.py --exp "$EXP" \
      --category "$cat" --seed "$seed" --clicks 12 --out "$FIGS/session_${slug}_s${seed}.jpg"
  done
done

# What a Good vote drags, at both ends of the size axis.
cd "$WT/scripts/experiments/pile" || exit 2
for cat in "bird@small" "bird@large" "stop sign@small" "backpack@medium"; do
  slug=$(echo "$cat" | tr ' @' '__')
  step "boxes_${slug}" python box_sheets.py --dataset vg_scale --category "$cat" \
    --n 8 --order area --out "$FIGS/boxes_${slug}.jpg"
done

echo; echo "=== done $(date +%H:%M) ==="
ls -la "$OUT" "$FIGS" | tail -40
