#!/usr/bin/env bash
# Folds addendum to the #2861 anchor-mass sweep: 4 calibration folds instead of
# production's 2.
#
# The one item left from the pre-registered #2860 follow-up.  Two folds make the
# fold-anchored `qmean` and `qmedian` combine arms byte-identical, so the
# combine question cannot be asked at production's `calibrate_count`; and the
# fold count itself plausibly interacts with the anchor mass, because each fold
# holds out half the votes at K=2 but three quarters at K=4 (fewer anchors per
# fold, more folds to average).
#
# This is a run-level A/B, NOT a paired arm: changing K changes the splits, the
# per-fold models and therefore the trajectory.  The contrast that survives that
# is each run's own `Δ vs xcal_only`, since the control moves with the
# trajectory.  Scoped to VG × siglip (whole_image) — the cheap region-voting
# environment — so it fits the window; the same κ grid as the main run.
set -uo pipefail

export VTS_REPO=/exp/sgreenberg/projects/vts-rate-2861
WT="$VTS_REPO"
HERE="$WT/scripts/experiments/calibration"

export CALIB_EXP="/exp/$USER/anchor-folds-2861"
export CALIB_RESULTS="$CALIB_EXP/results"

export CALIB_SAFE_THRESHOLDS=1
export CALIB_ANCHORED=1
export CALIB_ANALYZE=analyze_anchored.py
export CALIB_HEAD=linear

# --- the one thing that differs from the main run ---
export CALIB_CALIBRATE_COUNT=4

export CALIB_DATASETS=visual_genome_m
export CALIB_VG_EMBEDDERS=siglip
export CALIB_PATCH_STYLES=max_patch
export CALIB_REPOOL_VARIANTS=

export CALIB_ANCHORED_WEIGHTS=0.01,0.03,0.1,0.3,0.5,1,2,3
export CALIB_ANCHORED_RULES=mid,rate
export CALIB_ANCHORED_FOLD_ARMS=1
# Now meaningful: at K=4 these are four distinct fold quantiles to combine.
export CALIB_ANCHORED_FOLD_COMBINES=qmean,qmedian
export CALIB_ANCHORED_CHECKPOINTS=20,50,100,200,300

export CALIB_BLEND_SCHEDULE=prod
export CALIB_SCHEDULE_VARIANTS=prod,slow_cap50,cap50,pure_gmm,pure_xcal

export CALIB_MAX_STEPS=300
export CALIB_N_SEEDS=4

export CALIB_PARTITION=cpu
export CALIB_GRES=none
export CALIB_MEM=8G
export CALIB_CPUS=1
# Four fold models per step instead of two, so ~2x the per-step scoring work.
export CALIB_TIME=3:00:00
export CALIB_CONC=120

export VTSEARCH_DATA_DIR="$CALIB_EXP/datadir"
export VTSEARCH_MODELS_DIR="/exp/$USER/max-patch/models"
export HF_HOME="/exp/$USER/.cache/huggingface"

mkdir -p "$CALIB_EXP/logs" "$CALIB_RESULTS/cells" "$CALIB_RESULTS/crops"
[[ -f "$CALIB_RESULTS/prepare_info.json" ]] || { echo "ERROR: no prepare_info.json" >&2; exit 1; }

if [[ -x "$WT/scripts/experiments/preflight.sh" ]]; then
  bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --need-gb 4 || {
    echo "preflight FAILED" >&2; [[ "${PREFLIGHT_SKIP:-0}" == "1" ]] || exit 1
  }
fi

exec bash "$HERE/launch_cells.sh"
