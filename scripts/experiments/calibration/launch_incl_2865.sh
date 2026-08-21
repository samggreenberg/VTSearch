#!/usr/bin/env bash
# #2865 cut-rule x inclusion sweep: which rule should answer the Inclusion knob?
#
#   bash launch_incl_2865.sh prepare   # stage 0 (cpu, reads the pile in place)
#   bash launch_incl_2865.sh size      # time ONE cell of the slowest arm
#   bash launch_incl_2865.sh arms      # the array + the analysis step
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
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- NOTHING WAS SUBMITTED" >&2' ERR

MODE="${1:-arms}"

export VTS_REPO=/exp/sgreenberg/projects/vts-incl-2865
WT="$VTS_REPO"
HERE="$WT/scripts/experiments/calibration"

# /exp is a 50G quota shared with every other study; the cut-inclusion side
# frame is ~120 rows per step per cell, so the cells dir runs to a few GB.
export CALIB_EXP="/expscratch/$USER/cut-incl-2865"
export CALIB_RESULTS="$CALIB_EXP/results"

# --- science knobs ---
export CALIB_SAFE_THRESHOLDS=1
export CALIB_ANCHORED=1
export CALIB_ANALYZE=analyze_cutincl.py

# CALIB_HEAD is deliberately UNSET: `head=None` resolves to
# `voting_iterations.PRODUCTION_HEAD`, the linear SVM a live detector trains
# since PR #3198.  The #2861/#2864 runs this one follows pinned `CALIB_HEAD=linear`
# because the logistic head was production *then*; carrying that pin forward
# would measure the cut rule on a head no user has.  See the report's fidelity
# section - the arms here are paired within a step, so the head choice moves
# every arm together and the *contrast* is the same object either way.

# --- environments ---
# The issue asks for "at least the two region-voting environments where fusion
# actually pays".  Region voting needs BOTH halves - ground-truth boxes (the
# dataset) and a patch grid (the embedder) - so the two are
# `visual_genome_m x dinov3_patch` and `coco_val x dinov3_patch`; the pile
# gained the second one after this launcher was first drafted, which is why it
# no longer reaches for two patch STYLES of one dataset instead.
# Each dataset also rides its `siglip` binary arm: the knob is a user-facing
# control on every detector, so a rule that restores it on region voting while
# wrecking binary voting is not shippable.  `siglip` (not `siglip2_l`) because
# it is the shipped default embedder.
# Order matters: the array enumerates dataset -> embedder -> category -> seed,
# so the long dinov3 arms are listed first and start in the first wave.
export CALIB_DATASETS=visual_genome_m,coco_val
export CALIB_VG_EMBEDDERS=dinov3_patch,siglip
export CALIB_COCO_EMBEDDERS=dinov3_patch,siglip
# `max_patch` only: it *is* the production patch pipeline.  The HAC hybrids are
# experiment-only arms and #2886 removed the tree from ingest, so scoring them
# here would double the run to price geometry no user gets.
export CALIB_PATCH_STYLES=max_patch
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
# embedder per dataset), so the limit has ample headroom.  `size` re-measures
# it rather than trusting this paragraph.
export CALIB_PARTITION=cpu
export CALIB_GRES=none
# Measured, not guessed: `size` timed one cell of each of the four arms at
# 75 min / 83 min (the two dinov3 region arms, MaxRSS 5.3G) and 8-9 min (the two
# siglip binary arms).  8G covers the peak; 4 h covers the slowest cell twice
# over.  Concurrency is capped by the `cpu_limit` QOS, which is cpu=240 with 2
# charged per task (=120) and mem=1100000M (=134 at 8G) - so 120 is the cpu cap,
# and asking for more only parks the excess behind your own array.
export CALIB_MEM=8G
export CALIB_CPUS=1
export CALIB_TIME=4:00:00
export CALIB_CONC=120
# ~28 k cut-inclusion rows per cell x 336 cells = ~9.4 M rows for one analyzer.
export CALIB_ANALYZE_MEM=48G
export CALIB_ANALYZE_TIME=2:00:00

# Read the pre-embedded pile in place: no re-embed, no GPU, no model download.
export VTS_PILE="${VTS_PILE:-/expscratch/$USER/vts-cache}"
export VTSEARCH_DATA_DIR="$VTS_PILE/datadir"
export VTSEARCH_MODELS_DIR="$VTS_PILE/models"
export HF_HOME="$VTS_PILE/models"

LOGS="$CALIB_EXP/logs"
mkdir -p "$LOGS" "$CALIB_RESULTS/cells" "$CALIB_RESULTS/crops"

ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"

require_jobid() {
  local id="$1" what="$2"
  if ! [[ "$id" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $what was REFUSED by sbatch (no job id came back)." >&2
    exit 1
  fi
}

case "$MODE" in
  prepare)
    # Stage 0 on the CPU partition: every pair is already in the pile, so this
    # loads each pickle, re-derives the selected categories and writes the
    # startup-exemplar vectors.  No model is constructed.
    P=$(sbatch --parsable --job-name=incl-prep --mem=24G --cpus-per-task=2 \
      --time=1:30:00 --partition=cpu --export=ALL \
      --output="$LOGS/prepare-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python prepare_data.py")
    require_jobid "$P" "prepare"
    echo "prepare job: $P  ->  $LOGS/prepare-$P.out"
    ;;

  size)
    # Time ONE cell of the slowest arm before committing to the whole array.
    # Cell 0 is the first (dataset, embedder) pair listed, which is the long one.
    IDX="${2:-0}"
    S=$(sbatch --parsable --job-name=incl-size --mem="$CALIB_MEM" --cpus-per-task="$CALIB_CPUS" \
      --time="$CALIB_TIME" --partition=cpu --export=ALL \
      --output="$LOGS/size-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && time python run_cells.py --index $IDX --outdir $CALIB_EXP/sizing")
    require_jobid "$S" "size"
    echo "size job: $S (cell $IDX)  ->  $LOGS/size-$S.out"
    ;;

  arms)
    if [[ ! -f "$CALIB_RESULTS/prepare_info.json" ]]; then
      echo "ERROR: no prepare_info.json at $CALIB_RESULTS - run '$0 prepare' first." >&2
      exit 1
    fi
    # Both region arms get their premise asserted, not assumed: `region_voting`
    # is a REQUEST, and a boxed dataset on a single-vector embedder silently runs
    # binary (#2877).  Preflight takes one arm at a time, so it runs twice.
    if [[ -x "$WT/scripts/experiments/preflight.sh" ]]; then
      for arm in visual_genome_m:dinov3_patch coco_val:dinov3_patch; do
        bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --need-gb 20 \
          --require-region-voting "$arm" \
          --job-name cal-cells --mem "$CALIB_MEM" --conc "$CALIB_CONC" || {
          echo "preflight FAILED ($arm)" >&2; [[ "${PREFLIGHT_SKIP:-0}" == "1" ]] || exit 1
        }
      done
    fi
    exec bash "$HERE/launch_cells.sh"
    ;;

  *)
    echo "usage: $0 {prepare|size [cell]|arms}" >&2
    exit 2
    ;;
esac
