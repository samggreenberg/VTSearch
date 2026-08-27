#!/usr/bin/env bash
# #2861 anchor-mass (kappa) boundary sweep + environment generalization.
#
# Follow-up to the #2852/#2860 anchored-mixture run, whose winner
# (fold_anchored_w1_rate) sat at the BOTTOM EDGE of the tested kappa grid
# {1,3,10,30,100}.  This run:
#   (a) extends the grid two decades DOWN - {0.01 .. 3} - so the optimum is
#       interior rather than a boundary, with kappa=1 and kappa=3 replicated
#       from the prior run for continuity;
#   (b) repeats the whole sweep in SIX environments (3 datasets x 4 embedders
#       x 2 voting modes x 2 styles) so "kappa* = 1" can be checked for
#       stability rather than asserted from one arm pair.
#
# Trajectory schedule is pinned to `prod` (the 6->20 ramp) - the same
# trajectory generator the #2860 run lived under, so the replicated kappa
# points are true replicates.  Today's shipped schedules (slow_cap50 region /
# cap50 binary) ride along as free counterfactual rows.
#
# Prepare is REUSED (no GPU stage): VG + COCO from the #2841 mixin run,
# caltech101_m derived locally (boxless -> whole-image exemplars, no model).
set -uo pipefail

export VTS_REPO=/exp/sgreenberg/projects/vts-rate-2861
WT="$VTS_REPO"
HERE="$WT/scripts/experiments/calibration"

export CALIB_EXP="/exp/$USER/anchor-rate-2861"
export CALIB_RESULTS="$CALIB_EXP/results"

# --- science knobs ---
export CALIB_SAFE_THRESHOLDS=1
export CALIB_ANCHORED=1
export CALIB_ANALYZE=analyze_anchored.py
export CALIB_HEAD=linear

# Six environments.  Order matters: the array is enumerated
# dataset -> embedder -> category -> seed, so the LONG arm (VG dinov3 max_patch,
# ~75 min/cell) is listed first and starts in the first concurrency wave.
# The region environment is the PAIR `siglip+dinov3_patch` (#3278).  "kappa* is
# stable across six environments" is a claim about ENVIRONMENTS, so an
# environment has to differ from its neighbours in the things that name it --
# dataset, embedder, voting mode -- and not also in how the run started.  Bare
# `dinov3_patch` has no text tower, which would have made the single region
# environment the only one of the six opening on three random known-goods.
export CALIB_DATASETS=visual_genome_m,coco_val,caltech101_m
export CALIB_VG_EMBEDDERS=siglip+dinov3_patch,siglip
export CALIB_REQUIRE_OPENING=text
# A paired arm cannot fall back to the known-good start (`run_cells.py` raises),
# so every selected category must have a typed query.  Selection filters to the
# eligible ones before it picks, replacing rather than dropping.
export CALIB_REQUIRE_SEED_QUERY=1
export CALIB_COCO_EMBEDDERS=siglip,siglip2
export CALIB_CALTECH_EMBEDDERS=siglip,siglip_l
export CALIB_PATCH_STYLES=max_patch
export CALIB_REPOOL_VARIANTS=

# --- the sweep this run exists for ---
# 8 anchor masses spanning 0.01 -> 3 (two decades below the #2860 winner,
# overlapping it at 1 and 3).  Cost parity with the prior run is kept by
# dropping qmedian, which is byte-identical to qmean at CALIBRATE_COUNT=2.
export CALIB_ANCHORED_WEIGHTS=0.01,0.03,0.1,0.3,0.5,1,2,3
export CALIB_ANCHORED_RULES=mid,rate
export CALIB_ANCHORED_FOLD_ARMS=1
export CALIB_ANCHORED_FOLD_COMBINES=qmean
export CALIB_ANCHORED_CHECKPOINTS=20,50,100,200,300

# Trajectory pinned to the #2860 run's schedule; the shipped ones scored free.
export CALIB_BLEND_SCHEDULE=prod
export CALIB_SCHEDULE_VARIANTS=prod,slow_cap50,cap50,slow,pure_gmm,pure_xcal

export CALIB_MAX_STEPS=300
export CALIB_N_SEEDS=4

# --- ops: cpu partition (dodges the 4-GPU QOS cap).
# The binding constraint is the `cpu_limit` QOS: cpu=240 and mem=1100000M PER
# USER.  The #2860 run asked for 4 cpus + 24G a cell, which capped it at 60
# concurrent tasks - but `sacct -o TotalCPU,Elapsed` on that run shows
# TotalCPU == Elapsed, i.e. a cell is single-threaded and three of those four
# cpus did nothing, and its peak RSS was 5.4G, not 24G.  1 cpu + 8G buys 130
# concurrent instead, which is what lets all 92 long dinov3 cells run in one
# wave rather than two.
export CALIB_PARTITION=cpu
export CALIB_GRES=none
export CALIB_MEM=8G
export CALIB_CPUS=1
export CALIB_TIME=3:00:00
export CALIB_CONC=130

export VTSEARCH_DATA_DIR="$CALIB_EXP/datadir"
export VTSEARCH_MODELS_DIR="/exp/$USER/max-patch/models"
export HF_HOME="/exp/$USER/.cache/huggingface"

LOGS="$CALIB_EXP/logs"
mkdir -p "$LOGS" "$CALIB_RESULTS/cells" "$CALIB_RESULTS/crops"

if [[ ! -f "$CALIB_RESULTS/prepare_info.json" ]]; then
  echo "ERROR: no prepare_info.json at $CALIB_RESULTS" >&2; exit 1
fi

if [[ -x "$WT/scripts/experiments/preflight.sh" ]]; then
  bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --need-gb 6 || {
    echo "preflight FAILED" >&2; [[ "${PREFLIGHT_SKIP:-0}" == "1" ]] || exit 1
  }
fi

exec bash "$HERE/launch_cells.sh"
