#!/usr/bin/env bash
# Where is the #2877 pile run?
#
# Counts CELL FILES per (half, arm) against the expected total, not `squeue`.
# A drained queue with missing cells means failures, and that is invisible in a
# job listing -- which is why the completion signal here is a file on disk.
#
#   bash status_acq_2877.sh
set -uo pipefail
BASE="${ACQ_BASE:-/expscratch/$USER/acq-2877}"
SEEDS="${ACQ_SEEDS:-16}"
NENV="${ACQ_NENV:-12}"
WANT=$(( SEEDS * NENV ))

printf '%-4s %-9s %7s %7s %7s %7s  %s\n' half arm cells want zero hdr pct
total=0
for half in bin reg; do
  [[ -d "$BASE/$half" ]] || continue
  for arm in prod acq_m1 acq_m2 acq_m3 acq_m4 acq_p2 rank_pin; do
    d="$BASE/$half/$arm/results/cells"
    [[ -d "$d" ]] || continue
    # Main frames only: a cell also writes __sweep / __cutdiag / __cutincl /
    # __picks, and counting those reports 5x progress.
    n=$(find "$d" -maxdepth 1 -name 'task_*.csv' ! -name '*__*' | wc -l)
    # A zero-byte file counts as "done" to resume and must be deleted before
    # one; a header-only file parses cleanly and passes `-size 0`, so it is
    # counted separately rather than trusted.
    z=$(find "$d" -maxdepth 1 -name 'task_*.csv' ! -name '*__*' -size 0 | wc -l)
    h=0
    for f in $(find "$d" -maxdepth 1 -name 'task_*.csv' ! -name '*__*' -size -2k); do
      [[ "$(wc -l < "$f")" -le 1 ]] && h=$((h + 1))
    done
    total=$((total + n))
    printf '%-4s %-9s %7d %7d %7d %7d  %5.1f%%\n' "$half" "$arm" "$n" "$WANT" "$z" "$h" \
      "$(awk -v a="$n" -v b="$WANT" 'BEGIN{print 100*a/b}')"
  done
done
echo
echo "total cells: $total of $(( 14 * WANT ))"
echo
squeue -u "$USER" -h -O "Name:26,State:12" | sort | uniq -c
echo
df -h "$BASE" | tail -1
