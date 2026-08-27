#!/usr/bin/env bash
# Fold-count study (#2897): what is actually true right now, from files on disk.
#
# Deliberately not a `squeue` wrapper.  A drained queue and a finished study
# look identical from the scheduler's side; the difference is whether the cells
# are there.  So this counts cells against the expected total, names the
# zero-byte ones (which `resume` would skip, silently analysing a hole), and
# says whether each stage's output file exists.
#
# Safe to run at any time, from anywhere, as often as you like.
#
# Usage: source folds_env.sh && bash status_folds_2897.sh [--write]
#        --write  also drops the same text at $CALIB_EXP/STATUS.md
set -uo pipefail

EXP="${CALIB_EXP:?CALIB_EXP unset - source folds_env.sh first}"
RES="${CALIB_RESULTS:-$EXP/results}"
WT="${VTS_REPO:-}"

render() {
  echo "# Fold-count study (#2897) - status at $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo
  echo "exp:  $EXP"
  echo "repo: $WT @ $(git -C "$WT" log --oneline -1 2>/dev/null || echo '?')"
  echo

  echo "## Queue"
  local q
  q=$(squeue -u "$USER" -h -o "%.10i %.18j %.2t %.10M %R" 2>/dev/null)
  if [[ -z "$q" ]]; then echo "  (empty - nothing of yours is queued or running)"; else echo "$q"; fi
  echo

  echo "## Screen cells"
  local expected produced empty
  # A cell is task_NNNN.csv; task_NNNN__cutdiag.csv / __sweep.csv are its
  # sidecars, so match the four-digit form exactly or the count triples.
  expected=$( (cd "$WT/scripts/experiments/calibration" && python run_cells.py --print-cells 2>/dev/null | tail -1) || echo "?")
  produced=$(find "$RES/cells" -name 'task_[0-9][0-9][0-9][0-9].csv' -size +0 2>/dev/null | wc -l)
  empty=$(find "$RES/cells" -name 'task_[0-9][0-9][0-9][0-9].csv' -size 0 2>/dev/null | wc -l)
  echo "  produced: $produced / $expected"
  echo "  ZERO-BYTE: $empty   <- delete these before any resume; they count as done"
  if ((empty > 0)); then
    find "$RES/cells" -name 'task_[0-9][0-9][0-9][0-9].csv' -size 0 2>/dev/null | head -10 | sed 's/^/    /'
  fi
  echo "  bytes: $(du -sh "$RES/cells" 2>/dev/null | cut -f1)"
  echo

  echo "## Screen outputs"
  for f in "$RES/summary.json" "$RES/REPORT.md"; do
    if [[ -s "$f" ]]; then echo "  OK      $f"; else echo "  MISSING $f"; fi
  done
  if [[ -s "$RES/summary.json" ]]; then
    python - "$RES/summary.json" <<'PY' 2>/dev/null || true
import json, sys
s = json.load(open(sys.argv[1]))
for mode, b in (s.get("by_voting") or {}).items():
    print(f"    {mode}: recommended_k={b.get('h3_recommended_k')} "
          f"kept_production={b.get('h3_kept_production')} "
          f"best_ignoring_cost={b.get('best_k_ignoring_cost')} "
          f"sd_thr_falls={b.get('h4_sd_threshold_falls_at_best_k')}")
PY
  fi
  echo

  echo "## Live A/B arms"
  local found=0
  for d in "$(dirname "$EXP")"/calibration-folds-2897-ab-k*; do
    [[ -d "$d" ]] || continue
    found=1
    local n
    n=$(find "$d/results/cells" -name 'task_[0-9][0-9][0-9][0-9].csv' -size +0 2>/dev/null | wc -l)
    echo "  $(basename "$d"): $n cells, report $( [[ -s "$d/results/REPORT.md" ]] && echo present || echo absent )"
  done
  ((found)) || echo "  (none yet - the chain launches these only if the screen names a K* != 2)"
  echo

  echo "## Disk"
  df -h "$EXP" 2>/dev/null | tail -1 | sed 's/^/  /'
}

if [[ "${1:-}" == "--write" ]]; then
  render | tee "$EXP/STATUS.md"
else
  render
fi
