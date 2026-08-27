#!/usr/bin/env bash
# Fold-count study (#2897), stage 1: the counterfactual screen.
#
# Production cross-calibrates at 2 folds.  Nobody has re-asked whether that is
# the right number since the calibration stack was rebuilt (cross-partially-
# labelled GMM, anchored mixtures, the blend schedule), so this measures the
# cost and the benefit of raising it across the whole grid at once.
#
# The screen is cheap because the folds are NESTED: each is an independent
# stratified draw off one RandomState(42) at a per-fold size that ignores the
# count, so the K folds a live `calibrate_count=K` run would train are exactly
# the first K of the Kmax folds trained here.  One run therefore yields every
# K's threshold - byte-identical to that K's own run, for the same votes - plus
# every K's measured wall clock, paired within the step.  Price is set by the
# grid's MAXIMUM (Kmax - 2 extra fold fits per step), not by its length.
#
# Both voting modes, because #2897 asks for both and the calibrators differ:
# visual_genome_m is region voting on the bag-aware (grouped) calibrator,
# caltech101_m is binary voting on the row-wise one.
#
# What this run structurally CANNOT see: K also steers acquisition (the
# threshold is the rank position Autopilot's Hard pick samples around), so a
# larger K would have collected different votes.  That is launch_folds_2897_ab.sh.
#
# Design + pre-registered decision rules: docs/experiments/calibration-fold-count/REPORT.md
#
# Usage: bash launch_folds_2897.sh
set -uo pipefail

export CALIB_EXP="${CALIB_EXP:-/exp/$USER/calibration-folds-2897}"
export CALIB_RESULTS="${CALIB_RESULTS:-$CALIB_EXP/results}"

# The shipped path: safe thresholds on, production linear head.  The fold count
# is a knob on the cross-calibration term, but what a user gets is that term
# after the blend, so both arms are emitted per K (folds_k{K}_xcal / _blend).
export CALIB_SAFE_THRESHOLDS="${CALIB_SAFE_THRESHOLDS:-1}"
export CALIB_HEAD="${CALIB_HEAD:-linear}"
export CALIB_BLEND_SCHEDULE="${CALIB_BLEND_SCHEDULE:-prod}"
export CALIB_ANALYZE=analyze_folds_2897.py

# --- The grid this study exists to sweep. ---
# Log-ish spacing from "no cross-calibration at all" through production's 2 to a
# count well past any plausible knee, so saturation is visible rather than
# assumed.  Kmax=16 costs 14 extra fold fits per step: confirm the per-cell
# seconds on ONE cell before submitting the array, and fall back to
# CALIB_FOLD_COUNTS=1,2,3,4,6,8 if 16 does not fit the window.
export CALIB_FOLD_COUNTS="${CALIB_FOLD_COUNTS:-1,2,3,4,6,8,12,16}"
# The run still LIVES at production's 2, so the trajectory is the one users get
# and every K is scored against the same votes.
export CALIB_CALIBRATE_COUNT="${CALIB_CALIBRATE_COUNT:-2}"

# Both voting modes; one whole-image embedder each keeps the cell count down so
# the Kmax fold budget goes into seeds and steps instead of arms.  The region
# side additionally runs the production patch arm, which is where the bag-aware
# calibrator (and its much smaller effective calibration set) lives.
# The region side is the PAIR `siglip+dinov3_patch` (#3278).  This study reports
# per voting mode because the calibrators differ (bag-aware vs row-wise), and
# with bare `dinov3_patch` the region side would also have been the only side
# opening on three random known-goods -- DINOv3 has no text tower.  regret(K) is
# read across the cold start, where the calibration set is a handful of scores
# and therefore most sensitive to which items the opening put in front of the
# user, so the difference lands squarely inside the deliverable.
export CALIB_DATASETS="${CALIB_DATASETS:-visual_genome_m,caltech101_m}"
export CALIB_VG_EMBEDDERS="${CALIB_VG_EMBEDDERS:-siglip,siglip+dinov3_patch}"
export CALIB_CALTECH_EMBEDDERS="${CALIB_CALTECH_EMBEDDERS:-siglip}"
# The preflight below runs BEFORE prepare (this is a full chain), so check 14
# reports itself skipped there and run_cells.py is what asserts this per cell.
export CALIB_REQUIRE_OPENING=text
# A paired arm cannot fall back to the known-good start (`run_cells.py` raises),
# so every selected category must have a typed query.  Selection filters to the
# eligible ones before it picks, replacing rather than dropping.
export CALIB_REQUIRE_SEED_QUERY=1
export CALIB_PATCH_STYLES="${CALIB_PATCH_STYLES:-max_patch}"
export CALIB_REPOOL_VARIANTS="${CALIB_REPOOL_VARIANTS:-}"

# Long enough to cross the cold start into the regime where the calibration set
# is big enough that extra folds should stop mattering - the shape of the
# regret(K) curve across that crossing IS the deliverable, so it must be inside
# the window.  4 seeds x the standard category sets.
export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-150}"
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-4}"

# Kmax=16 is ~8x production's calibration work per step.  Generous, because a
# cell that times out is a cell whose *late* steps - the ones that answer the
# saturation question - are the ones missing.
export CALIB_TIME="${CALIB_TIME:-12:00:00}"
export CALIB_PARTITION="${CALIB_PARTITION:-cpu}"
export CALIB_GRES="${CALIB_GRES:-none}"
export CALIB_MEM="${CALIB_MEM:-8G}"
export CALIB_CPUS="${CALIB_CPUS:-1}"
export CALIB_CONC="${CALIB_CONC:-120}"

# The gate, not a reminder: refuses a results dir already holding another grid's
# cells, a low actual mount, zero-byte cells resume would skip, or a VTS_REPO
# that is not what was committed.
WT="${VTS_REPO:-/exp/$USER/projects/vts-calib}"
if [[ -x "$WT/scripts/experiments/preflight.sh" ]]; then
  bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --need-gb 6 || {
    echo "preflight FAILED" >&2; [[ "${PREFLIGHT_SKIP:-0}" == "1" ]] || exit 1
  }
fi

exec bash "$(dirname "$0")/launch_all.sh"
