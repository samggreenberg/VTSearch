#!/usr/bin/env bash
# Everything the #3156 map needs, chained GRID-side so it survives a
# disconnect. Each step is independent and non-fatal: a step that fails must
# not cost the ones after it, and its absence has to be visible in the log
# rather than inferred from a missing file.
set -u
# The worktree is an ARGUMENT, not a constant: this chain is submitted by
# whichever run is being analysed, and a hardcoded path silently analyses one
# run's cells with another run's code. `VTS_REPO` is what every other script
# here reads, so it is what this reads too.
WT=${VTS_REPO:-/exp/sgreenberg/projects/vts-vgmap-3276}
EXP=${CALIB_EXP:-/expscratch/sgreenberg/scale-3156-map}
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

# Each step is bounded as well as non-fatal. "Non-fatal" only covers a step that
# EXITS non-zero; a step that never exits takes the whole job with it, and one
# did: `figures_trajectory` sat 50 minutes on its first figure here while the
# identical command on more data finished in four elsewhere. The cause was never
# established, so the control is a bound rather than a fix.
STEP_TIMEOUT="${STEP_TIMEOUT:-1800}"
step() {
  local name="$1"; shift
  echo; echo "=== $name  ($(date +%H:%M)) ==="
  local rc=0
  timeout -s KILL "$STEP_TIMEOUT" "$@" || rc=$?
  if [[ "$rc" -eq 0 ]]; then
    echo "--- $name OK"
  elif [[ "$rc" -eq 137 ]]; then
    echo "--- $name TIMED OUT after ${STEP_TIMEOUT}s - the rest of the chain continues"
  else
    echo "--- $name FAILED (rc=$rc)"
  fi
}

# The denominator comes from the run, never from this file. It was written here
# as a literal 6480, which is a property of ONE grid (three columns, sixty
# seeds); read as "complete" for the 3600-cell grid that replaced it, the same
# literal would have been wrong in the direction that HIDES missing cells.
# `launch_scale.sh` records the shape it launched beside the results.
CELLS=$(ls "$EXP/results/cells" 2>/dev/null | grep -c '^task_[0-9]*\.csv$')
SHAPE="$EXP/results/grid_shape.json"
EXPECT=$(python - "$SHAPE" <<'PY' 2>/dev/null || true
import json, sys

try:
    print(int(json.load(open(sys.argv[1]))["n_cells"]))
except Exception:
    pass
PY
)
if [[ -n "$EXPECT" ]]; then
  echo "cells present: $CELLS / $EXPECT"
  [[ "$CELLS" == "$EXPECT" ]] || echo "!! INCOMPLETE GRID - every number below is drawn from $CELLS cells"
else
  echo "cells present: $CELLS (no grid_shape.json, so the expected count is unknown)"
fi

# The zero-click anchor, and the interactive viewer that hangs off it.
#
# Both are here rather than left to the report's prose because they answer the
# questions a reader has AFTER the report: the anchor says what typing the query
# got for free, so a curve can be read as "did the clicking earn its keep at
# all"; the viewer carries every (band, category, embedder, metric, seed) slice,
# so a reader can ask their own question instead of asking for a re-run. The
# report links `viewer.html`, so a chain that skips this leaves a dead link.
#
# The anchor is cached - it re-embeds one query per category and re-reads every
# pickle, which makes it the slow step here and its answer does not change
# between analysis runs. Delete the CSV to force a recompute.
TEXT_BASELINE="$OUT/text_baseline.csv"
if [[ -s "$TEXT_BASELINE" ]]; then
  echo "reusing $TEXT_BASELINE ($(wc -l < "$TEXT_BASELINE") rows) - delete it to recompute"
else
  step text_baseline python text_baseline.py --results "$EXP/results" --out "$TEXT_BASELINE"
fi

# `results=prod` names the directory and labels it: this study has ONE arm - the
# shipped configuration - and `results` is where it sits, not what it is.
step viewer python viewer.py --results "$EXP" --arms results=prod \
  --baseline "$TEXT_BASELINE" --out "$OUT/viewer.html" \
  --title "VTSearch on vg_scale: quality over clicks" \
  --subtitle "Twelve classes x three target-size bands x five embedder columns, shipped defaults (#3156, #3276)"

step overview      bash -c "python analyze_overview.py --exp $EXP --baseline $TEXT_BASELINE > $OUT/REPORT_overview.txt 2>&1; tail -5 $OUT/REPORT_overview.txt"
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
