#!/usr/bin/env bash
# Safe-threshold GMM study (#2799) on the HLTCOE Grid.
#
# Thin wrapper over launch_all.sh that flips the pre-registered #2799 knobs:
# safe_thresholds ON, Visual Genome region voting only, the production
# `max_patch` patch style plus a `whole_image` single-vector control, 30 voting
# steps (the GMM has authority only below ~20 votes), 8 seeds (cells are ~5x
# cheaper than the 150-step #2781 cells, and the small-vote regime is noisy).
# Results land under /exp/$USER/calibration-safe so the #2781 outputs are
# untouched; the shared Max-Patch pickles/crops are reused in place.
#
# Design + pre-registered decision rules: docs/plans/safe-threshold-gmm-experiment.md
#
# Usage: bash launch_safe.sh
set -uo pipefail

export CALIB_EXP="${CALIB_EXP:-/exp/$USER/calibration-safe}"
export CALIB_RESULTS="${CALIB_RESULTS:-$CALIB_EXP/results}"

export CALIB_SAFE_THRESHOLDS=1
export CALIB_ANALYZE=analyze_safe.py

# Visual Genome region voting only; production patch arm + single-vector control.
export CALIB_DATASETS="${CALIB_DATASETS:-visual_genome_m}"
export CALIB_VG_EMBEDDERS="${CALIB_VG_EMBEDDERS:-siglip,dinov3_patch}"
export CALIB_PATCH_STYLES="${CALIB_PATCH_STYLES:-max_patch}"
# No raw-patch tree arm -> nothing to re-pool.
export CALIB_REPOOL_VARIANTS="${CALIB_REPOOL_VARIANTS:-}"

# The 6-20-vote ramp window is the object of study; beyond ~20 votes the blend
# is pure cross-cal and the #2781 study already covers it.
export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-30}"
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-8}"

# Short trajectories -> short cells.
export CALIB_TIME="${CALIB_TIME:-1:30:00}"

exec bash "$(dirname "$0")/launch_all.sh"
