#!/usr/bin/env bash
# Fold-count study (#2897), stage 2: the live A/B that closes what the screen
# cannot see.
#
# The screen (launch_folds_2897.sh) holds the trajectory fixed and re-cuts every
# K against the same votes.  That is exactly what makes it cheap, and exactly
# what makes it blind to acquisition feedback: the threshold is the rank
# position Autopilot's Hard pick samples around, so a run that genuinely lives
# at a different K collects DIFFERENT votes from step one, and the regret it
# ends at is not the regret the screen attributed to that K.
#
# So this runs one full simulation per fold count, each living at its own K.
# Changing K changes the splits, the fold models and therefore the trajectory,
# so these arms are NOT paired: the contrast that survives is each run's own
# `Delta vs its K=2 counterfactual` (the screen rows ride along, at no extra
# cost beyond the arm's own Kmax), compared across runs as an unpaired
# difference of deltas.
#
# Run it only AFTER the screen has named a candidate K*: two arms (2 and K*)
# answer the shipped question; more only if the screen's curve is flat enough
# that K* is genuinely ambiguous.
#
# Usage: bash launch_folds_2897_ab.sh <K> [<K> ...]     e.g. bash launch_folds_2897_ab.sh 2 8
set -uo pipefail

(($# >= 1)) || { echo "usage: $0 <K> [<K> ...]" >&2; exit 1; }

HERE="$(cd "$(dirname "$0")" && pwd)"

for K in "$@"; do
  [[ "$K" =~ ^[0-9]+$ ]] || { echo "ERROR: '$K' is not a fold count" >&2; exit 1; }

  # One study, one CALIB_EXP: each arm is a different trajectory, so each gets
  # its own results dir.  Sharing one would interleave two grids' cells under
  # indistinguishable task indices and there would be no way to separate them
  # afterwards.
  #
  # CALIB_AB_BASE picks the parent.  It is not cosmetic: /exp/$USER is a 50 G
  # quota that these studies keep filling, and an arm that runs out of disk
  # loses its *late* steps - the ones the saturation question lives in.  Point
  # it at a roomy mount and the arms land there instead.
  export CALIB_EXP="${CALIB_AB_BASE:-/exp/$USER}/calibration-folds-2897-ab-k$K"
  export CALIB_RESULTS="$CALIB_EXP/results"

  # The arm LIVES at K: the acquisition feedback the screen cannot see is the
  # whole point of this stage.
  export CALIB_CALIBRATE_COUNT="$K"
  # Carry the K=2 counterfactual along inside each arm, so every run can be
  # differenced against its own control rather than across machines.
  export CALIB_FOLD_COUNTS="2,$K"

  echo "=== A/B arm: calibrate_count=$K -> $CALIB_EXP ==="
  bash "$HERE/launch_folds_2897.sh" || { echo "arm K=$K FAILED to submit" >&2; exit 1; }
done

echo
echo "Submitted ${#} arm(s).  A submission is not a launch: confirm each arm came"
echo "back with a numeric job id above, and that cells begin appearing under"
echo "/exp/$USER/calibration-folds-2897-ab-k*/results/cells, before quoting an ETA."
