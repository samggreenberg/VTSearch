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
# Usage: bash launch_safe.sh
set -uo pipefail

export CALIB_EXP="${CALIB_EXP:-/exp/$USER/calibration-safe}"
export CALIB_RESULTS="${CALIB_RESULTS:-$CALIB_EXP/results}"

export CALIB_SAFE_THRESHOLDS=1
export CALIB_ANALYZE=analyze_safe.py
# The head a live detector actually has, because the blend's authority is only worth
# measuring on the model users actually get.  NOT `linear`: that was production
# when this launcher was written, and PR #3198 moved `PRODUCTION_HEAD` to the
# linear SVM, so the pin outlived the thing it was pinning -- preflight check 12
# has been failing on it since.  Named rather than left unset, which is what
# `launch_transfer_2883.sh` settled on: the run's head is then readable from the
# launcher instead of from a default three modules away.
export CALIB_HEAD="${CALIB_HEAD:-linear_svm}"

# Visual Genome region voting only; production patch arm + single-vector control.
# The region arm is the PAIR `siglip+dinov3_patch` (#3278).  The control is what
# says a winning GMM variant does not regress the no-max-pool geometry, and a
# control that opens differently from the arm it controls is not one: DINOv3 has
# no text tower, so bare `dinov3_patch` starts on three random known-goods while
# the SigLIP control starts on a typed query.  The blend this study measures has
# authority only in the first ~20 votes, which is exactly the stretch the
# opening determines.
export CALIB_DATASETS="${CALIB_DATASETS:-visual_genome_m}"
export CALIB_VG_EMBEDDERS="${CALIB_VG_EMBEDDERS:-siglip,siglip+dinov3_patch}"
export CALIB_REQUIRE_OPENING=text
# A paired arm cannot fall back to the known-good start (`run_cells.py` raises),
# so every selected category must have a typed query.  Selection filters to the
# eligible ones before it picks, replacing rather than dropping.
export CALIB_REQUIRE_SEED_QUERY=1
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
