#!/usr/bin/env bash
# GMM cut-point study (#2836) on the HLTCOE Grid.
#
# Same sizing and arms as the #2799 safe-threshold run (production linear head,
# Visual Genome region voting, `max_patch` + a `whole_image` single-vector
# control, 30 steps, 12 seeds), because this study re-cuts the *same* per-step
# models and its contrasts are paired within a step -- reusing the sizing keeps
# it directly comparable to the numbers #2836 is trying to explain.
#
# Two independent halves, launched together:
#   theory bench (cpu, ~15 min)  -> the rules against a known truth; no dataset
#   cells array  (cpu, ~1 h)     -> the rules on real VG scores + the oracle
#                                   decomposition, analyzed by analyze_cut.py
#
# The theory bench does not depend on prepare, so it is submitted immediately
# rather than chained behind the embedding stage.
#
# Design + pre-registered predictions: docs/plans/gmm-cut-theory-experiment.md
#
# Usage: bash launch_cut.sh
set -uo pipefail

WT="${VTS_REPO:-/exp/$USER/projects/vts-calib}"
HERE="$WT/scripts/experiments/calibration"

export CALIB_EXP="${CALIB_EXP:-/exp/$USER/calibration-cut}"
export CALIB_RESULTS="${CALIB_RESULTS:-$CALIB_EXP/results}"

export CALIB_SAFE_THRESHOLDS=1
export CALIB_ANALYZE=analyze_cut.py
# The head the live detector ships since #2790/#2809 - the cut's authority is
# only worth measuring on the model users actually get.
export CALIB_HEAD="${CALIB_HEAD:-linear}"

# Visual Genome region voting only; production patch arm + single-vector control.
# The control is not decoration here: `calculate_gmm_threshold` also backs the
# cosine/text sort, so a winning rule has to not regress the no-max-pool geometry.
export CALIB_DATASETS="${CALIB_DATASETS:-visual_genome_m}"
export CALIB_VG_EMBEDDERS="${CALIB_VG_EMBEDDERS:-siglip,dinov3_patch}"
export CALIB_PATCH_STYLES="${CALIB_PATCH_STYLES:-max_patch}"
# No raw-patch tree arm -> nothing to re-pool.
export CALIB_REPOOL_VARIANTS="${CALIB_REPOOL_VARIANTS:-}"
# The inclusion sweep is a different question; keep the side frame tiny.
export CALIB_SWEEP_KS="${CALIB_SWEEP_KS:-0}"

# The 6-20-vote ramp window is the object of study; beyond ~20 votes the blend
# is pure cross-cal and the cut has no authority left.
export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-30}"
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-12}"

# Cells train a linear head on CPU in ~3 min each, which dodges the 4-GPU QOS
# cap entirely (see the #2799 run).
export CALIB_PARTITION="${CALIB_PARTITION:-cpu}"
export CALIB_GRES="${CALIB_GRES:-none}"
export CALIB_MEM="${CALIB_MEM:-24G}"
export CALIB_CPUS="${CALIB_CPUS:-4}"
export CALIB_CONC="${CALIB_CONC:-40}"
export CALIB_TIME="${CALIB_TIME:-1:30:00}"

LOGS="$CALIB_EXP/logs"
mkdir -p "$LOGS"

# --- The theory half: independent of any dataset, so it runs now. ---
T=$(sbatch --parsable --job-name=cut-theory --mem=8G --cpus-per-task=4 \
  --time=1:00:00 --partition=cpu --export=ALL --output="$LOGS/theory-%j.out" \
  --wrap="source $WT/gridenv.sh && export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS && cd $HERE && python theory_bench.py --reps ${CUT_THEORY_REPS:-40}")
echo "theory bench job: $T   -> $CALIB_RESULTS/theory/"

# --- The data half: the standard prepare -> cells -> analyze chain. ---
exec bash "$HERE/launch_all.sh"
