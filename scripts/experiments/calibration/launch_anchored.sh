#!/usr/bin/env bash
# Anchored-mixture calibration study (#2852) on the HLTCOE Grid.
#
# Thin wrapper over launch_all.sh that flips the pre-registered knobs for the
# population-anchored-calibration experiment (design + decision rules:
# docs/plans/population-anchored-calibration.md): safe_thresholds ON (the
# anchored arms ride the variant-row path and need the shipped blend +
# xcal_only controls emitted alongside), CALIB_ANCHORED=1, the production
# linear head, Visual Genome region voting with the production max_patch arm
# plus a whole_image single-vector control, and 300 voting steps - the deep
# regime where the ongoing owner-side experiment found the naive GMM still
# competitive with x-cal.  Results land under /exp/$USER/calibration-anchored
# so the #2781/#2799 outputs are untouched; the shared Max-Patch pickles/crops
# are reused in place.
#
# The sweep grid (anchor weights x cut rules x fold combines) comes from
# CALIB_ANCHORED_WEIGHTS / CALIB_ANCHORED_RULES / CALIB_ANCHORED_FOLD_COMBINES
# (see experiment_config.py for the registered defaults).
#
# Usage: bash launch_anchored.sh
set -uo pipefail

export CALIB_EXP="${CALIB_EXP:-/exp/$USER/calibration-anchored}"
export CALIB_RESULTS="${CALIB_RESULTS:-$CALIB_EXP/results}"

export CALIB_SAFE_THRESHOLDS=1
export CALIB_ANCHORED=1
export CALIB_ANALYZE=analyze_anchored.py
# The head the live detector ships since #2790/#2809 - the estimator is only
# worth measuring on the model users actually get.
export CALIB_HEAD="${CALIB_HEAD:-linear}"

# Visual Genome region voting; production patch arm + single-vector control.
# The region arm is the PAIR `siglip+dinov3_patch` (#3278): the anchored
# estimator is characterised HERE against the whole-image control, and bare
# `dinov3_patch` cannot open the way that control does -- no text tower means
# three random known-goods where the control gets a typed query.  An anchored
# mixture is fitted to the score distribution the votes produce, and the opening
# is what produces the first of them, so a per-mode reading of this study would
# have been part voting mode and part start.  The pair holds the opening fixed
# and leaves the geometry as the only difference.
export CALIB_DATASETS="${CALIB_DATASETS:-visual_genome_m}"
export CALIB_VG_EMBEDDERS="${CALIB_VG_EMBEDDERS:-siglip,siglip+dinov3_patch}"
# Asserted per cell by run_cells.py, and by preflight check 14 wherever this
# chain reaches a preflight with prepare already staged.  `launch_all.sh`
# submits the array from a grid job, so the per-cell guard is the one that
# always runs; --export=ALL carries this to it.
export CALIB_REQUIRE_OPENING=text
# A paired arm cannot fall back to the known-good start (`run_cells.py` raises),
# so every selected category must have a typed query.  Selection filters to the
# eligible ones before it picks, replacing rather than dropping.
export CALIB_REQUIRE_SEED_QUERY=1
export CALIB_PATCH_STYLES="${CALIB_PATCH_STYLES:-max_patch}"
# No raw-patch tree arm -> nothing to re-pool.
export CALIB_REPOOL_VARIANTS="${CALIB_REPOOL_VARIANTS:-}"

# The deep-vote regime is the object of study: checkpoints to 300 votes, so
# every window of the plan's {20,50,100,200,300} grid is populated.
export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-300}"
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-4}"

# 300-step trajectories + one sim-set scoring pass per calibration fold per
# step (the fold-anchored arms) -> long cells.
export CALIB_TIME="${CALIB_TIME:-8:00:00}"

exec bash "$(dirname "$0")/launch_all.sh"
