#!/usr/bin/env bash
# Fold-count cost/benefit study (#3314): what is true right now, from disk.
#
# Deliberately not a `squeue` wrapper.  A drained queue and a finished study look
# identical from the scheduler's side; the difference is whether the cells are
# there.  So this counts cells against the expected total, names the zero-byte
# ones (which resume would skip, silently analysing a hole) and the header-only
# ones (which resume would skip too, and which no `-size 0` test can see), and
# says whether each stage's output exists.
#
# Safe to run at any time, from anywhere, as often as you like.
#
#   bash status_folds_3314.sh [--write]
#     --write  also drops the same text at $BASE/STATUS.md
set -uo pipefail

WT="${VTS_REPO:-/exp/$USER/projects/vts-folds-3314}"
BASE="${FOLDS3314_BASE:-/expscratch/$USER/folds-3314}"
HERE="$WT/scripts/experiments/calibration"
PREP="$BASE/prepare/results"

cells_in() {
  # task_NNNN.csv only: `__sweep`, `__cutdiag`, `__cutincl` and `__picks` are
  # sidecars of the same cell, so a loose glob quintuples the count.
  find "$1" -name 'task_[0-9][0-9][0-9][0-9].csv' -size +0 2>/dev/null | wc -l
}

render() {
  echo "# Fold-count cost/benefit (#3314) - status at $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo
  echo "base: $BASE"
  echo "repo: $WT @ $(git -C "$WT" log --oneline -1 2>/dev/null || echo '?')"
  echo

  echo "## Queue"
  local q
  q=$(squeue -u "$USER" -h -o "%.10i %.18j %.2t %.10M %R" 2>/dev/null)
  [[ -z "$q" ]] && echo "  (empty - nothing of yours is queued or running)" || echo "$q"
  echo

  echo "## Stage 0"
  [[ -s "$PREP/prepare_info.json" ]] && echo "  OK      $PREP/prepare_info.json" || echo "  MISSING $PREP/prepare_info.json"
  [[ -s "$BASE/analysis/text_baseline.csv" ]] &&
    echo "  OK      text baseline ($(($(wc -l < "$BASE/analysis/text_baseline.csv") - 1)) rows) - the click-0 anchor" ||
    echo "  MISSING $BASE/analysis/text_baseline.csv  <- curves refuse to draw without it"
  echo

  local expected
  expected=$( (cd "$HERE" && CALIB_RESULTS="$PREP" python run_cells.py --print-cells 2>/dev/null | tail -1) || echo "?")

  for stage in screen ab-k* ; do
    for d in "$BASE"/$stage; do
      [[ -d "$d" ]] || continue
      local res="$d/results"
      echo "## $(basename "$d")"
      echo "  cells: $(cells_in "$res/cells") / $expected"
      # Zero-byte and header-only both count as "done" to resume, and only one
      # of them is visible to `find -size 0`.
      local z h
      z=$(find "$res/cells" -name 'task_[0-9][0-9][0-9][0-9].csv' -size 0 2>/dev/null | wc -l)
      h=$(find "$res/cells" -name 'task_[0-9][0-9][0-9][0-9].csv' -size +0 2>/dev/null \
            -exec sh -c '[ "$(wc -l < "$1")" -le 1 ]' _ {} \; -print 2>/dev/null | wc -l)
      echo "  ZERO-BYTE: $z   HEADER-ONLY: $h   <- delete the first before any resume; count the second in the report"
      echo "  bytes: $(du -sh "$res/cells" 2>/dev/null | cut -f1)"
      for f in "$res/summary_folds3314.json" "$res/REPORT_folds3314.md"; do
        [[ -s "$f" ]] && echo "  OK      $f" || echo "  pending $f"
      done
      if [[ -s "$res/summary_folds3314.json" ]]; then
        python - "$res/summary_folds3314.json" <<'PY' 2>/dev/null || true
import json
import sys

s = json.load(open(sys.argv[1]))
g = s.get("gate") or {}
print(f"    cost model: {s.get('cost_model')}")
print(f"    gate_open={g.get('gate_open')} k_best={g.get('k_best')}@{g.get('k_best_geometry')} "
      f"schedule={g.get('schedule')}")
print(f"    reason: {g.get('reason')}")
PY
      fi
      echo
    done
  done

  echo "## Disk"
  df -h "$BASE" 2>/dev/null | tail -1 | sed 's/^/  /'
}

if [[ "${1:-}" == "--write" ]]; then
  mkdir -p "$BASE"
  render | tee "$BASE/STATUS.md"
else
  render
fi
