#!/usr/bin/env bash
# #2865 cut-rule x inclusion sweep: which rule should answer the Inclusion knob?
#
# The #2861 anchor-mass run moved production to `kappa=0.3, mid`, and `mid` is
# the midpoint of two component means - it never looks at the cost weights the
# Inclusion knob arrives as.  Shipping it verbatim made the knob a NO-OP for
# every detector with usable calibration folds.  #2868 shipped `mid_tilt` (the
# measured midpoint at inclusion 0, rate-rule tilt away from it) to restore the
# knob without disturbing the measured operating point, but the TILT ITSELF has
# never been measured: both calibration runs scored every arm at INCLUSION 0,
# which is the one inclusion where the rule choice cannot matter.
#
# This run sweeps the candidates across the whole knob.  Two decision numbers,
# per `analyze_cutincl.py`:
#   (a) paired regret at each k against the incumbent, scored at that k;
#   (b) how much of the knob survives as DISTINCT ADMITTED SETS - a rule that
#       moves the threshold without moving the included set has fixed nothing.
#
# Cost note: the sweep is nearly free.  The per-fold anchored EM does not depend
# on the cut rule, the combine, or the inclusion, so one fit per anchor weight
# serves the whole (rule x combine x k) grid - the same no-refit re-cut the app
# does when the user drags the slider.  The marginal cost over a plain #2861-shaped
# run is arithmetic plus one oracle sweep per k.
#
# Prepare is REUSED from the #2861 run (no GPU stage).  Point CALIB_EXP at that
# run's directory, or symlink its `results/prepare_info.json` + `results/crops/`
# into a fresh one.
set -uo pipefail

export VTS_REPO=/exp/sgreenberg/projects/vts-incl-2865
WT="$VTS_REPO"
HERE="$WT/scripts/experiments/calibration"

export CALIB_EXP="/exp/$USER/cut-incl-2865"
export CALIB_RESULTS="$CALIB_EXP/results"

# --- science knobs ---
export CALIB_SAFE_THRESHOLDS=1
export CALIB_ANCHORED=1
export CALIB_ANALYZE=analyze_cutincl.py
export CALIB_HEAD=linear

# --- environments ---
# The issue asks for "at least the two region-voting environments where fusion
# actually pays".  Only the `dinov3_patch` arms actually region-vote (a boxed
# dataset paired with a single-vector embedder silently degrades to binary; see
# experiment_config.REGION_VOTING_BY_DATASET), and `visual_genome_m` is the only
# prepared dataset with a stored patch grid - so the two region-voting cells are
# its two patch STYLES, not two datasets.  COCO rides along as the binary-voting
# control: the knob is a user-facing control on every detector, so a rule that
# restores it on region voting while wrecking binary is not shippable.
export CALIB_DATASETS=visual_genome_m,coco_val
export CALIB_VG_EMBEDDERS=dinov3_patch
export CALIB_COCO_EMBEDDERS=siglip2
export CALIB_PATCH_STYLES=max_patch,max_patch_pca_hac
export CALIB_REPOOL_VARIANTS=

# --- the sweep this run exists for ---
# kappa is PINNED at the shipped 0.3: this run is about the cut rule, and
# re-opening the anchor mass would confound the two (and multiply the arms by 8).
export CALIB_ANCHORED_WEIGHTS=0.3
# The #2865 candidate set.  Note what is NOT here: the issue's candidate 2
# ("drop the mixture-weight factor, lam = fnr/fpr") turned out to describe what
# `rate` already computes - the prior-odds factor in rate's lam cancels the
# w_lo/w_hi inside _rate_cut's offset exactly, so rate is prior-free and its
# interior root is invariant to the mixture weights at every inclusion.
# `cross_tilt` is the rule that genuinely retains the priors, i.e. candidate 2's
# literal formula, and it is here so the issue's text gets priced too.
# Candidate 4 ("keep mid, tilt elsewhere") is `mid`: the honest null.
export CALIB_ANCHORED_RULES=mid,mid_tilt,rate,cross_tilt,q_tilt
export CALIB_ANCHORED_FOLD_ARMS=1
export CALIB_ANCHORED_FOLD_COMBINES=qmean
export CALIB_ANCHORED_CHECKPOINTS=20,50,100,200,300

# The knob's nominal range is [-10, 10].  Sweeping every stop would be honest
# but the ends are where the quantile pins at 0/1 for every rule, so the grid is
# dense in the middle (where users sit) and reaches the ends to catch a rule
# that only saturates there.
export CALIB_CUT_INCL_KS=-10,-6,-4,-3,-2,-1,0,1,2,3,4,6,10

# `q_tilt`'s step size is a FREE PARAMETER with no principled value, so it has
# to be fitted rather than assumed.  Sweeping it is what makes candidate 3 a
# real candidate instead of a rule with a magic number in it; a `q_tilt` that
# only wins at one hand-picked step is not a result.
export CALIB_CUT_INCL_QTILT_STEPS=0.005,0.01,0.02,0.04,0.08

# Trajectory: shipped defaults, so the steps the arms are re-cut on are the ones
# users actually generate.  The counterfactual schedules ride along free.
export CALIB_BLEND_SCHEDULE=
export CALIB_SCHEDULE_VARIANTS=slow_cap50,cap50

export CALIB_MAX_STEPS=300
export CALIB_N_SEEDS=4

# --- ops: cpu partition (dodges the 4-GPU QOS cap), sized off the #2861 run,
# whose cells were single-threaded with a 5.4G peak RSS.  This run adds ~13 k
# values of arithmetic per step over that one and no new scoring passes, so the
# per-cell cost is essentially unchanged; the array is much smaller (one
# embedder per dataset), so a 3 h limit has ample headroom.
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
  echo "ERROR: no prepare_info.json at $CALIB_RESULTS" >&2
  echo "       Reuse the #2861 run's prepare stage - symlink its results/prepare_info.json" >&2
  echo "       and results/crops/ in, or point CALIB_EXP at that run directly." >&2
  exit 1
fi

if [[ -x "$WT/scripts/experiments/preflight.sh" ]]; then
  bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --need-gb 6 || {
    echo "preflight FAILED" >&2; [[ "${PREFLIGHT_SKIP:-0}" == "1" ]] || exit 1
  }
fi

exec bash "$HERE/launch_cells.sh"
