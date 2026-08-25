#!/usr/bin/env bash
# Fold combine-rule study (#3115), run together with #3116's instruments.
#
# Two functions in this repo disagree about the same empirical fact.
# `threshold_from_fold_orderings` - the path the live app calls - POOLS every
# fold's held-out scores into one bag and takes a single conformal quantile,
# because "all folds' scores live on the same sigmoid scale".
# `FoldAnchoredCut._combined_fold_quantile` takes one cut per fold and averages
# them in QUANTILE space, specifically so no cross-scale averaging of raw cuts
# ever happens - i.e. on the premise that fold scores are NOT comparable.  One
# of those is wrong and nobody has measured which.
#
# The two issues are one run by construction, and must not be split: #3116's
# honest reference and sd(threshold) are what make #3115's result readable, and
# a combine-rule answer read through #2897's broken decomposition would
# reproduce that study's confusion rather than resolve it.
#
# Cheap for the same reason the fold-count screen is: every arm re-cuts the SAME
# already-trained fold prefix.  Adding a combine rule costs arithmetic on cached
# arrays and no fits - the anchored pair even shares one EM (see
# `_fold_count_arms`), so the price of this study over a plain #2897 re-run is
# effectively the extra fold fits Kmax already implies.
#
# WHY K>=3 IS THE WHOLE POINT: at production's calibrate_count=2 the mean and
# the median of two numbers coincide, so the contamination question is
# structurally invisible there.  The grid must reach past it, and the analyzer
# reads its verdict from K>=3 only.
#
# Design + decision rules: docs/experiments/calibration-fold-combine/REPORT.md
#
# Usage: bash launch_folds_3115.sh
set -uo pipefail

export CALIB_EXP="${CALIB_EXP:-/exp/$USER/calibration-folds-3115}"
export CALIB_RESULTS="${CALIB_RESULTS:-$CALIB_EXP/results}"

# The shipped path.  Safe thresholds are not a switch any more (#2799): fusion
# is always on, and the arms that read a cut in a fold's own distribution -
# every `q*` rule, and both anchored rules - need the per-fold haystacks that
# only this path supplies.
export CALIB_SAFE_THRESHOLDS="${CALIB_SAFE_THRESHOLDS:-1}"
export CALIB_ANALYZE=analyze_folds_2897.py

# CALIB_HEAD deliberately UNSET: it resolves to `PRODUCTION_HEAD`, today the
# linear SVM.  #2897's launcher pinned `linear`, which was production when it
# was written and was not by the time #3129 ran a launcher with the same pin -
# the failure preflight check 12 now blocks.  A study that does not sweep the
# head must not name one.
#
# CALIB_BLEND_SCHEDULE likewise unset: an explicit schedule overrides the app's
# per-mode default (#2841), and it only reaches the RETIRED `blend` arm anyway.

# --- The grid this study exists to sweep. ---
# Same K grid as #2897, so the fold-count axis of this run is directly
# comparable with that report's, and so the combine contrast is measured across
# a full order of magnitude of K rather than at one arbitrary count.  The
# pooling argument predicts a gap that GROWS with K (a pooled quantile estimates
# the quantile of a mixture of K half-trained models, and that mixture widens);
# a flat curve refutes the mechanism whatever the level says, which only a grid
# this wide can show.
export CALIB_FOLD_COUNTS="${CALIB_FOLD_COUNTS:-1,2,3,4,6,8,12,16}"
# The run still LIVES at production's 2, so the trajectory is the one users get
# and every arm is scored against the same votes.
export CALIB_CALIBRATE_COUNT="${CALIB_CALIBRATE_COUNT:-2}"

# Both voting modes, asserted per CELL rather than per dataset: region voting
# needs boxes AND a patch grid, so `visual_genome_m x siglip` is a BINARY
# environment however boxed its dataset is (#2877/#2905, and #2897's own
# errata).  `dinov3_patch`/`max_patch` is the only region arm here, and it is
# where the bag-aware calibrator - and its much smaller effective calibration
# set - lives, which is where a combine rule has the most room to matter.
export CALIB_DATASETS="${CALIB_DATASETS:-visual_genome_m,caltech101_m}"
export CALIB_VG_EMBEDDERS="${CALIB_VG_EMBEDDERS:-siglip,dinov3_patch}"
export CALIB_CALTECH_EMBEDDERS="${CALIB_CALTECH_EMBEDDERS:-siglip}"
export CALIB_PATCH_STYLES="${CALIB_PATCH_STYLES:-max_patch}"
export CALIB_REPOOL_VARIANTS="${CALIB_REPOOL_VARIANTS:-}"

# Long enough to cross the cold start, where the calibration set is a handful of
# scores and the combine rule plausibly matters most, into the regime where the
# quantile is stable and it should not.  The shape across that crossing is the
# deliverable, so it has to be inside the window.
export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-150}"
# 4 seeds is also what makes sd(threshold) (#3116) computable at all: it is
# taken ACROSS seeds at a fixed step, so a single-seed run has no instrument.
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-4}"

export CALIB_TIME="${CALIB_TIME:-12:00:00}"
export CALIB_PARTITION="${CALIB_PARTITION:-cpu}"
export CALIB_GRES="${CALIB_GRES:-none}"
export CALIB_MEM="${CALIB_MEM:-8G}"
export CALIB_CPUS="${CALIB_CPUS:-1}"
export CALIB_CONC="${CALIB_CONC:-120}"

# The gate, not a reminder: refuses a results dir already holding another grid's
# cells, a low actual mount, zero-byte cells resume would skip, a job name
# already in use, or a VTS_REPO that is not what was committed.
WT="${VTS_REPO:-/exp/$USER/projects/vts-folds-3115}"
if [[ -x "$WT/scripts/experiments/preflight.sh" ]]; then
  bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --need-gb 6 \
    --mem "$CALIB_MEM" --conc "$CALIB_CONC" || {
    echo "preflight FAILED" >&2; [[ "${PREFLIGHT_SKIP:-0}" == "1" ]] || exit 1
  }
fi

exec bash "$(dirname "$0")/launch_all.sh"
