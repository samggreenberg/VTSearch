#!/usr/bin/env bash
# Acquisition/reporting decoupling, second environment — issue #2877.
#
# PR #2876 measured this on `coco_val x siglip2 x whole_image` (BINARY voting)
# and found an interior optimum at acq_inclusion_offset = -3, which #2878 then
# shipped as the default in both the app and the harness.  The plan that scoped
# that run pre-registered a second environment conditional on the first being
# positive: `visual_genome_m x siglip`, REGION voting.  It was positive, so this
# is that check.
#
# Why it is not a formality: #2861 showed this family of answers does not always
# transfer between voting modes.  Under region voting a media's score is a max
# over ~24 region-node scores, so the Bad mode is an extreme-value statistic and
# the score distribution the acquisition cut is realized on has a different
# shape.  docs/ML.md also records that the fused threshold's gain tracks how
# many POSITIVE anchors the regime supplies, and that it wins clearly on region
# voting while being a dead heat on COCO binary voting — the acquisition cut
# rides the same fitted mixture via `mid_tilt`, so region voting is plausibly
# where the offset does MORE.  That is a hypothesis, not a result.
#
# Arms are #2876's verbatim so the two environments are directly comparable:
#
#   prod      k_acq =  0   control - the pre-#2876 coupled behaviour
#   acq_m1    k_acq = -1
#   acq_m2    k_acq = -2
#   acq_m3    k_acq = -3   the shipped default
#   acq_m4    k_acq = -4   far end
#   acq_p2    k_acq = +2   FALSIFICATION arm - must make positives WORSE
#   rank_pin  cut pinned at the conformal path's own pool percentile (0.959)
#
# `prod` MUST name its 0 explicitly: the default is now -3, so a launcher copied
# from an older template would run seven arms of the same thing.
#
# Prepare is REUSED (no GPU stage): the VG siglip pickle from the #2749
# max-patch run and the exemplar crops + category selection from the #2861
# anchor-rate run, so the categories are the same 23 that study used.
set -uo pipefail

export VTS_REPO=${VTS_REPO:-/exp/$USER/projects/vts-acq-vg}
WT="$VTS_REPO"
HERE="$WT/scripts/experiments/calibration"

export CALIB_EXP="/exp/$USER/acq-vg"

# --- environment: VG region voting, one embedder ---
# region_voting is implied by the dataset (REGION_VOTING_BY_DATASET), and siglip
# is not a patch embedder, so the only style is whole_image — the exemplar is a
# dragged ground-truth box and a media's score is a max over its region nodes.
export CALIB_DATASETS=visual_genome_m
export CALIB_VG_EMBEDDERS=siglip
export CALIB_MAX_STEPS=100
# 24, not #2876's 8.  The issue says to size for the COST contrast rather than
# the positives one, and 8 seeds is the wrong answer HERE even though it was the
# right answer on COCO: VG region-voting costs sit near 0.43 instead of 0.137 and
# are correspondingly noisier, so the 8-seed pilot (kept at results_8seed/) put a
# 95% CI of [-0.014, +0.019] on the k=-3 cost delta against a ship-rule tolerance
# of +0.01.  That is a null too wide to certify, and it cannot separate "the
# offset is free on region voting" from "it costs something" - opposite shipping
# decisions.  Tripling the seeds is a precision fix on the pre-registered
# endpoint, NOT a wider arm grid.
export CALIB_N_SEEDS=${CALIB_N_SEEDS:-24}
export CALIB_HEAD=linear
export CALIB_SAFE_THRESHOLDS=1
export CALIB_ANCHORED=0
export CALIB_SCHEDULE_VARIANTS=
export CALIB_REPOOL_VARIANTS=
# CALIB_BLEND_SCHEDULE deliberately unset: the run must live under whatever
# schedule production picks for its voting mode (#2849 made that per-mode, so
# region voting gets `slow` where the COCO run got `cap50`).  Pinning one here
# would measure the offset under a schedule no region-voting user runs.

# --- ops: cpu partition, single-threaded cells; the real cap is the cpu_limit
# QOS (cpu=240/user, 2 charged per task) => 120 concurrent whatever %N says.
export CALIB_PARTITION=cpu
export CALIB_GRES=none
export CALIB_MEM=${CALIB_MEM:-8G}
export CALIB_CPUS=1
export CALIB_TIME=${CALIB_TIME:-3:00:00}
export CALIB_CONC=${CALIB_CONC:-16}

export VTSEARCH_DATA_DIR="$CALIB_EXP/datadir"
export VTSEARCH_MODELS_DIR="/exp/$USER/max-patch/models"
export HF_HOME="/exp/$USER/.cache/huggingface"

# Analysis is cross-arm and runs once, by hand, after all seven drain.
export CALIB_ANALYZE=${CALIB_ANALYZE:-noop.py}

RESULTS_ROOT="${RESULTS_ROOT:-$CALIB_EXP/results}"
mkdir -p "$CALIB_EXP/logs" "$RESULTS_ROOT"

if [[ ! -f "$CALIB_EXP/results/prepare_info.json" ]]; then
  echo "ERROR: no prepare_info.json at $CALIB_EXP/results" >&2; exit 1
fi

if [[ -x "$WT/scripts/experiments/preflight.sh" ]]; then
  bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --need-gb 4 || {
    echo "preflight FAILED" >&2; [[ "${PREFLIGHT_SKIP:-0}" == "1" ]] || exit 1
  }
fi

# arm -> "ACQ_INCLUSION_OFFSET ACQ_RANK_PERCENTILE"  ("-" = unset)
declare -A ARMS=(
  [prod]="0 -"
  [acq_m1]="-1 -"
  [acq_m2]="-2 -"
  [acq_m3]="-3 -"
  [acq_m4]="-4 -"
  [acq_p2]="2 -"
  [rank_pin]="0 0.959"
)

for arm in prod acq_m1 acq_m2 acq_m3 acq_m4 acq_p2 rank_pin; do
  read -r inc pct <<<"${ARMS[$arm]}"
  export CALIB_ACQ_INCLUSION_OFFSET=""
  export CALIB_ACQ_RANK_PERCENTILE=""
  [[ "$inc" != "-" ]] && export CALIB_ACQ_INCLUSION_OFFSET="$inc"
  [[ "$pct" != "-" ]] && export CALIB_ACQ_RANK_PERCENTILE="$pct"
  export CALIB_RESULTS="$RESULTS_ROOT/$arm"
  mkdir -p "$CALIB_RESULTS/cells"
  ln -sfn "$CALIB_EXP/results/prepare_info.json" "$CALIB_RESULTS/prepare_info.json"
  ln -sfn "$CALIB_EXP/results/crops" "$CALIB_RESULTS/crops"
  echo "=== arm $arm (acq_inclusion_offset='${CALIB_ACQ_INCLUSION_OFFSET}' acq_rank_percentile='${CALIB_ACQ_RANK_PERCENTILE}')"
  bash "$HERE/launch_cells.sh" || echo "ARM $arm SUBMIT FAILED" >&2
done
