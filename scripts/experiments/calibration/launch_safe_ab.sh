#!/usr/bin/env bash
# Safe-threshold study (#2799), the ship-decision A/B: safe_thresholds ON vs OFF
# on the **production linear head**, on the HLTCOE Grid.
#
# launch_safe.sh measures the GMM *variants* inside one safe-on run (which cut
# rule / fit geometry is best).  This launcher answers the deployment question
# instead - "should safe_thresholds be forced on for every VTSearch user?" - by
# running two otherwise-identical simulations, one with the blend and one
# without, and pairing them per (arm, category, seed) cell.  Two runs are needed
# because the blended threshold feeds Autopilot's Hard pick, so the arms label
# different items and a within-step counterfactual cannot see that feedback.
#
# Both runs train the linear (logistic) head the live detector ships since
# #2790/#2809, so the numbers are the ones users would actually get.
#
# Chains: prepare (GPU, ON dir) -> {ON cells array, OFF cells array} ->
#         analyze_safe.py (ON, variant contrasts) + analyze_ab.py (ON vs OFF).
#
# Usage: bash launch_safe_ab.sh
set -uo pipefail

WT="${VTS_REPO:-/exp/$USER/projects/vts-safe2799}"
HERE="$WT/scripts/experiments/calibration"
MAXPATCH="/exp/$USER/max-patch"

export CALIB_AB_ON_EXP="${CALIB_AB_ON_EXP:-/exp/$USER/calibration-safe-linear}"
export CALIB_AB_OFF_EXP="${CALIB_AB_OFF_EXP:-/exp/$USER/calibration-off-linear}"

# Shared pre-registered knobs.
# Visual Genome region voting only: the production max_patch arm plus a
# single-vector whole_image control.
export CALIB_DATASETS="${CALIB_DATASETS:-visual_genome_m}"
export CALIB_VG_EMBEDDERS="${CALIB_VG_EMBEDDERS:-siglip,dinov3_patch}"
export CALIB_PATCH_STYLES="${CALIB_PATCH_STYLES:-max_patch}"
export CALIB_REPOOL_VARIANTS="${CALIB_REPOOL_VARIANTS:-}"
export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-30}"
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-8}"
export CALIB_HEAD="${CALIB_HEAD:-linear}"
# The inclusion-budget sweep is #2781's question, not this one: keep one k so
# the side CSVs stay tiny (/exp is a shared 50G volume).
export CALIB_SWEEP_KS="${CALIB_SWEEP_KS:-0}"

export VTSEARCH_DATA_DIR="${VTSEARCH_DATA_DIR:-$MAXPATCH/datadir}"
export VTSEARCH_MODELS_DIR="${VTSEARCH_MODELS_DIR:-$MAXPATCH/models}"
export HF_HOME="${HF_HOME:-/exp/$USER/.cache/huggingface}"

LOGS="$CALIB_AB_ON_EXP/logs"
mkdir -p "$LOGS" "$CALIB_AB_ON_EXP/results/crops" "$CALIB_AB_ON_EXP/results/cells" \
         "$CALIB_AB_OFF_EXP/results/crops" "$CALIB_AB_OFF_EXP/results/cells"

# Reuse the Max-Patch exemplar crops (the embeddings pickles are read in place
# from the shared datadir); both runs must see the identical exemplars.
for exp in "$CALIB_AB_ON_EXP" "$CALIB_AB_OFF_EXP"; do
  for base in visual_genome_m__siglip__crops visual_genome_m__dinov3_patch__crops; do
    for ext in npz json; do
      src="$MAXPATCH/results/crops/$base.$ext"
      dst="$exp/results/crops/$base.$ext"
      [[ -e "$src" && ! -e "$dst" ]] && ln -s "$src" "$dst"
    done
  done
done

SHARED="export CALIB_DATASETS=$CALIB_DATASETS CALIB_VG_EMBEDDERS=$CALIB_VG_EMBEDDERS \
CALIB_PATCH_STYLES=$CALIB_PATCH_STYLES CALIB_REPOOL_VARIANTS='$CALIB_REPOOL_VARIANTS' \
CALIB_MAX_STEPS=$CALIB_MAX_STEPS CALIB_N_SEEDS=$CALIB_N_SEEDS CALIB_HEAD=$CALIB_HEAD \
CALIB_SWEEP_KS=$CALIB_SWEEP_KS VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR \
VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME \
CALIB_AB_ON_EXP=$CALIB_AB_ON_EXP CALIB_AB_OFF_EXP=$CALIB_AB_OFF_EXP VTS_REPO=$WT"

# --- Stage 0: prepare, once, into the ON run's results dir (the OFF run copies
# its prepare_info.json so both enumerate the identical cells). ---
P=$(sbatch --parsable --job-name=ab-prep --gres="${CALIB_GRES:-gpu:v100:1}" \
  --mem="${CALIB_PREP_MEM:-32G}" --cpus-per-task=6 --time="${CALIB_PREP_TIME:-3:00:00}" \
  --partition=gpu --export=ALL --output="$LOGS/prepare-%j.out" \
  --wrap="source $WT/gridenv.sh && $SHARED && export CALIB_EXP=$CALIB_AB_ON_EXP CALIB_RESULTS=$CALIB_AB_ON_EXP/results && cd $HERE && python prepare_data.py")
echo "prepare job: $P"

# --- Stage 1: a tiny launcher job submits both arrays + both analyzers once the
# cell count is known (it is not, until prepare has selected the categories). ---
S=$(sbatch --parsable --dependency=afterok:$P --job-name=ab-launch --mem=4G --cpus-per-task=1 \
  --time=0:20:00 --partition=cpu --export=ALL --output="$LOGS/launch-%j.out" \
  --wrap="source $WT/gridenv.sh && $SHARED && cd $HERE && bash launch_ab_cells.sh")
echo "A/B launcher job (after prepare): $S"
echo "ON  -> $CALIB_AB_ON_EXP/results   (REPORT.md, REPORT_AB.md)"
echo "OFF -> $CALIB_AB_OFF_EXP/results"
echo "Logs -> $LOGS"
