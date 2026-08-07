#!/usr/bin/env bash
# #2847: do the MLP-era cost spikes survive today's stack?
#
# Issue #2847 (MatthewELucio) reran the #2790 threshold study on COCO `cat` with
# a SigLIP2 whole-image embedder and found the error curve still blips - single
# steps where the operating cost jumps to 0.25-0.68 while the *oracle* cost for
# the same ranking stays flat around 0.05.  Two things have changed since:
#
#   * the detector head is now **linear** (logistic), not the auto-sized MLP
#     (#2790 / #2809);
#   * the threshold is now the **fold-anchored GMM cut** - anchor mass 0.3,
#     `mid_tilt` rule, `qmean` combine (#2852 / #2861 / #2865) - not the bare
#     cross-calibrated conformal quantile.
#
# So "are we better?" has two candidate causes, and a run that only measures
# today's stack cannot tell them apart - nor tell a real fix from a harness that
# never showed the phenomenon.  This launches the 2x2:
#
#   A_mlp_xcal   mlp    + conformal only   <- the #2847-era configuration
#   B_mlp_fused  mlp    + fold-anchored    <- threshold change alone
#   C_lin_xcal   linear + conformal only   <- head change alone
#   D_lin_fused  linear + fold-anchored    <- TODAY'S PRODUCTION
#
# Arm A is the positive control: if it does not spike, nothing else here means
# anything.  Both knobs steer acquisition (the threshold feeds Autopilot's Hard
# pick), so each arm is its own trajectory - arms are paired at the
# (category, seed) level on summary statistics, never step-by-step.
#
# Grid: coco_val x siglip2 x whole_image (binary voting), 19 categories x 5
# seeds x 100 steps - `cat` at the issue's own 5 seeds, the other 18 categories
# for breadth.  Prepare is REUSED from the #2841 mixin run (no GPU stage).
set -uo pipefail

export VTS_REPO=${VTS_REPO:-/exp/$USER/projects/vts-spike-2847}
WT="$VTS_REPO"
HERE="$WT/scripts/experiments/calibration"

export CALIB_EXP="/exp/$USER/spike-2847"

# --- environment: the issue's own arm ---
export CALIB_DATASETS=coco_val
export CALIB_COCO_EMBEDDERS=siglip2
export CALIB_MAX_STEPS=100
export CALIB_N_SEEDS=${CALIB_N_SEEDS:-5}
export CALIB_ANCHORED=0
export CALIB_SCHEDULE_VARIANTS=
export CALIB_REPOOL_VARIANTS=

# --- ops: cpu partition; a cell is single-threaded (#2861 lesson).  The real
# cap is the `cpu_limit` QOS (cpu=240 per user, 2 charged per task) => 120
# concurrent tasks whatever %N says.  Four arms share that pool.
export CALIB_PARTITION=cpu
export CALIB_GRES=none
export CALIB_MEM=${CALIB_MEM:-8G}
export CALIB_CPUS=1
export CALIB_TIME=${CALIB_TIME:-2:00:00}
export CALIB_CONC=${CALIB_CONC:-30}

export VTSEARCH_DATA_DIR="$CALIB_EXP/datadir"
export VTSEARCH_MODELS_DIR="/exp/$USER/max-patch/models"
export HF_HOME="/exp/$USER/.cache/huggingface"

# The cells array's afterany analyze step is not wanted per-arm - the analysis
# is cross-arm and runs once, by hand, after all four drain.
export CALIB_ANALYZE=${CALIB_ANALYZE:-noop.py}

mkdir -p "$CALIB_EXP/logs"

if [[ ! -f "$CALIB_EXP/results/prepare_info.json" ]]; then
  echo "ERROR: no prepare_info.json at $CALIB_EXP/results" >&2; exit 1
fi

if [[ -x "$WT/scripts/experiments/preflight.sh" ]]; then
  bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --need-gb 4 || {
    echo "preflight FAILED" >&2; [[ "${PREFLIGHT_SKIP:-0}" == "1" ]] || exit 1
  }
fi

# arm -> "HEAD SAFE_THRESHOLDS"
declare -A ARMS=(
  [A_mlp_xcal]="mlp 0"
  [B_mlp_fused]="mlp 1"
  [C_lin_xcal]="linear 0"
  [D_lin_fused]="linear 1"
)

for arm in A_mlp_xcal B_mlp_fused C_lin_xcal D_lin_fused; do
  read -r head safe <<<"${ARMS[$arm]}"
  export CALIB_HEAD="$head"
  export CALIB_SAFE_THRESHOLDS="$safe"
  export CALIB_RESULTS="$CALIB_EXP/results/$arm"
  mkdir -p "$CALIB_RESULTS/cells"
  # Each arm's runner reads prepare_info + crops from its own results dir.
  ln -sfn "$CALIB_EXP/results/prepare_info.json" "$CALIB_RESULTS/prepare_info.json"
  ln -sfn "$CALIB_EXP/results/crops" "$CALIB_RESULTS/crops"
  echo "=== arm $arm (head=$head safe_thresholds=$safe) -> $CALIB_RESULTS"
  bash "$HERE/launch_cells.sh" || echo "ARM $arm SUBMIT FAILED" >&2
done
