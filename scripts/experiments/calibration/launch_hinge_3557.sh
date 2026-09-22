#!/usr/bin/env bash
# #3557: price a sign-dependent ("hinge") tilt - cross_tilt below k=0, mid_tilt at and above.
#
#   bash launch_hinge_3557.sh prepare        # stage 0, ONCE, shared by both arms (cpu, reads the pile)
#   bash launch_hinge_3557.sh list           # which array index is which cell
#   bash launch_hinge_3557.sh size [idx]     # time ONE cell before committing
#   bash launch_hinge_3557.sh incumbent      # arm A: live rule = the SHIPPED mid_tilt
#   bash launch_hinge_3557.sh hinge          # arm B: live rule = the guarded hinge
#   bash launch_hinge_3557.sh analyze        # both-arm analysis, after both drain
#
# Plan + pre-registered decision rule: docs/experiments/2026-09-22-hinge-tilt-3557/PLAN.md
#
# WHY TWO RUN-LEVEL ARMS AND NOT ONE RE-CUT.  #2865 priced cross_tilt as a
# RE-CUT of the incumbent's trajectory - the reporting line only.  A shipped
# hinge is not only a reporting line: the acquisition cut re-cuts the SAME
# estimator at k + ACQUISITION_INCLUSION_OFFSET (= -4, below the seam at every
# reporting k < 4), so shipping the hinge changes which media Autopilot votes
# on from the first fitted step.  Arm A's re-cut frame is the paired,
# reporting-only price (every rule on one trajectory); arm B is what a user of
# the shipped hinge would actually get, paired to arm A on cell identity.
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- NOTHING WAS SUBMITTED" >&2' ERR

MODE="${1:-}"

export VTS_REPO="${VTS_REPO:-/expscratch/$USER/worktrees/vts-hinge-3557}"
WT="$VTS_REPO"
HERE="$WT/scripts/experiments/calibration"

# /expscratch: ~160 cut-inclusion rows per step per cell x 300 steps runs to
# several GB per arm, and /exp/$USER is a 50G mount.
BASE="${HINGE3557_BASE:-/expscratch/$USER/hinge-3557}"
PREP="$BASE/prepare/results"

# --- the sweep this run exists for -------------------------------------------
# Every integer stop of the knob's nominal range, plus two FRACTIONAL stops just
# below the seam.  The literal hinge breaks nesting iff q_cross(0-) < q_mid, and
# an integer grid only sees k=-1 against k=0; -0.5 and -0.25 are real operating
# points (#3319: the knob is continuous; the acquisition offset may be
# fractional) and put the seam itself on the frame.
export CALIB_CUT_INCL_KS="${CALIB_CUT_INCL_KS:--10,-9,-8,-7,-6,-5,-4,-3,-2,-1,-0.5,-0.25,0,1,2,3,4,5,6,7,8,9,10}"
# mid_tilt is the incumbent and must come first (preflight check 12 wants the
# shipped rule in the set).  `mid` is the INSTRUMENT CHECK (it must come back
# inert).  `rate` and `cross_tilt` are #2865's two readings, re-measured on
# today's stack.  The three hinges: `hinge` (guarded, the candidate),
# `hinge_raw` (the issue's literal rule, here to COUNT seam violations) and
# `hinge_cont` (cross's slope on mid's location: the decomposition arm).
export CALIB_ANCHORED_RULES="${CALIB_ANCHORED_RULES:-mid_tilt,mid,rate,cross_tilt,hinge,hinge_raw,hinge_cont}"
# kappa PINNED at the shipped 0.3: this run is about the cut rule.
export CALIB_ANCHORED_WEIGHTS="${CALIB_ANCHORED_WEIGHTS:-0.3}"
export CALIB_ANCHORED_FOLD_ARMS=1
export CALIB_ANCHORED_FOLD_COMBINES=qmean
export CALIB_ANCHORED_CHECKPOINTS="${CALIB_ANCHORED_CHECKPOINTS:-20,50,100,200,300}"

# --- science knobs -----------------------------------------------------------
export CALIB_SAFE_THRESHOLDS=1
export CALIB_ANCHORED=1
# Everything else UNSET so it resolves to production and check 12 has something
# true to check: head (#3198), acquisition offset (-4, #3319), blend schedule,
# calibration fraction, exclusion floor.  CALIB_LIVE_CUT_RULE is set per ARM.
unset CALIB_HEAD CALIB_ACQ_INCLUSION_OFFSET CALIB_BLEND_SCHEDULE CALIB_CALIBRATION_FRACTION CALIB_EXCLUDE_VOTED
export CALIB_SCHEDULE_VARIANTS=""
export CALIB_REPOOL_VARIANTS=""
# The per-arm stock analyzer is switched OFF: #3319's launcher chained an
# `analyze` job that fired when the arrays drained and silently overwrote the
# study's own analysis with a default-arm table.  The both-arm analysis is the
# `analyze` mode below, into its own directory.
export CALIB_ANALYZE=noop.py

# --- environments --------------------------------------------------------------
# Three datasets that depend on nothing another session is rebuilding today
# (not vg_scale, not DocMarks, not coco_quarry):
#   visual_genome_m x {siglip+dinov3_patch @ max_patch (REGION), siglip (binary)}
#   coco_val        x {siglip+dinov3_patch @ max_patch (REGION), siglip (binary)}
#   caltech101_m    x  siglip (binary; boxless, so there is no region arm)
# The first four are #2865's environments exactly, so its table is the prior;
# caltech is new and is the one environment none of #2865's numbers came from.
# Region arms are the PAIR (#3278) so every arm opens on a typed query.
export CALIB_DATASETS="${CALIB_DATASETS:-visual_genome_m,coco_val,caltech101_m}"
export CALIB_VG_EMBEDDERS="${CALIB_VG_EMBEDDERS:-siglip+dinov3_patch,siglip}"
export CALIB_COCO_EMBEDDERS="${CALIB_COCO_EMBEDDERS:-siglip+dinov3_patch,siglip}"
export CALIB_CALTECH_EMBEDDERS="${CALIB_CALTECH_EMBEDDERS:-siglip}"
export CALIB_REQUIRE_OPENING=text
export CALIB_REQUIRE_SEED_QUERY=1
# `max_patch` only: the production patch pipeline (PRODUCTION_PATCH_STYLE).
export CALIB_PATCH_STYLES=max_patch

# --- sizing ------------------------------------------------------------------
# 300 steps, as #2865: its deep band (> 100 votes) is where its cross_tilt
# numbers were read, and #3319's lesson is that a trajectory knob is reported by
# its trajectory - arm B's speed contrast needs the long horizon.
export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-300}"
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-4}"
# Seed-major: a truncated array loses its last SEEDS uniformly across every
# environment rather than its last categories entirely.
export CALIB_CELL_ORDER="${CALIB_CELL_ORDER:-seed}"

# No GPU work: cells train a small head on cached pile embeddings.
export CALIB_PARTITION=cpu
export CALIB_GRES=none
export CALIB_MEM="${CALIB_MEM:-8G}"
export CALIB_CPUS=1
export CALIB_TIME="${CALIB_TIME:-5:00:00}"
# MODEST ON PURPOSE.  Several other studies share this user's `cpu_limit` QOS
# (cpu=240 with 2 charged per task, mem ~1074G) today.  25 per arm x 2 arms =
# 50 tasks = 100 CPUs and 400G: under half of either ceiling.
export CALIB_CONC="${CALIB_CONC:-25}"
export CALIB_ANALYZE_MEM="${CALIB_ANALYZE_MEM:-96G}"
export CALIB_ANALYZE_TIME="${CALIB_ANALYZE_TIME:-4:00:00}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

# Read the pre-embedded pile in place: no re-embed, no GPU, no model download.
# shellcheck disable=SC1091
source "$WT/scripts/experiments/pile/pile_env.sh"

require_jobid() {
  local id="$1" what="$2"
  if ! [[ "$id" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $what was REFUSED by sbatch (no job id came back)." >&2
    exit 1
  fi
}

link_prepare() {
  local rd="$1"
  mkdir -p "$rd/cells"
  [[ -e "$rd/prepare_info.json" ]] || ln -s "$PREP/prepare_info.json" "$rd/prepare_info.json"
  [[ -e "$rd/crops" ]] || ln -s "$PREP/crops" "$rd/crops"
}

activate_venv() {
  # shellcheck disable=SC1091
  source "$WT/gridenv.sh" >/dev/null 2>&1 || {
    echo "ERROR: could not activate the venv at $WT/gridenv.sh" >&2; exit 1
  }
}

run_preflight() {
  local job="$1"; shift
  [[ -x "$WT/scripts/experiments/preflight.sh" ]] || return 0
  # `local`: bash scoping is dynamic, so an unqualified loop variable here named
  # `arm` overwrote submit_arm's own `arm` - the first launch named its array
  # after the last region premise it checked.
  local premise
  for premise in visual_genome_m:siglip+dinov3_patch coco_val:siglip+dinov3_patch; do
    bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --need-gb 30 \
      --reuse-prepare "$PREP" \
      --require-region-voting "$premise" \
      "$@" \
      --job-name "$job" --mem "$CALIB_MEM" --conc "$CALIB_CONC" || {
      echo "preflight FAILED for $job ($premise)" >&2
      [[ "${PREFLIGHT_SKIP:-0}" == "1" ]] || exit 1
    }
  done
}

# One arm = one live rule = one full trajectory = one CALIB_EXP.
submit_arm() {
  local arm="$1"; shift
  local -a div=("$@")
  if [[ ! -f "$PREP/prepare_info.json" ]]; then
    echo "ERROR: no prepare_info.json at $PREP - run '$0 prepare' first." >&2
    exit 1
  fi
  activate_venv
  export CALIB_EXP="$BASE/$arm"
  export CALIB_RESULTS="$CALIB_EXP/results"
  mkdir -p "$CALIB_EXP/logs"
  link_prepare "$CALIB_RESULTS"
  run_preflight "hinge3557-$arm" "${div[@]}"
  echo "=== arm $arm (live cut rule=${CALIB_LIVE_CUT_RULE:-default/production}) -> $CALIB_EXP ==="
  CALIB_JOB_NAME="hinge3557-$arm" bash "$HERE/launch_cells.sh" || {
    echo "arm $arm FAILED to submit" >&2; exit 1
  }
  local id
  id="$(cat "$CALIB_EXP/logs/.cells_jobid" 2>/dev/null || true)"
  require_jobid "$id" "arm $arm's cells array"
  echo "arm $arm array: $id"
}

case "$MODE" in
  prepare)
    export CALIB_EXP="$BASE/prepare"
    export CALIB_RESULTS="$PREP"
    mkdir -p "$BASE/prepare/logs" "$PREP/cells" "$PREP/crops"
    ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
    P=$(sbatch --parsable --job-name=hinge3557-prep --mem="${HINGE3557_PREP_MEM:-12G}" --cpus-per-task=2 \
      --time=1:30:00 --partition=cpu --export=ALL \
      --output="$BASE/prepare/logs/prepare-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python prepare_data.py")
    require_jobid "$P" "prepare"
    echo "prepare job: $P  ->  $BASE/prepare/logs/prepare-$P.out"
    ;;

  list)
    activate_venv
    export CALIB_EXP="$BASE/incumbent" CALIB_RESULTS="$PREP"
    (cd "$HERE" && python - <<'PYLIST'
import json

import common

common.setup_env()
import experiment_config as cfg
from run_cells import _categories_by_dataset

info = json.loads((common.RESULTS / "prepare_info.json").read_text())
cells = cfg.array_cells(_categories_by_dataset(info))
print(f"{len(cells)} cells per arm (order={cfg.CELL_ORDER})")
for i, c in enumerate(cells):
    styles = cfg.styles_for(c["dataset"], c["embedder"])
    print(f"  {i:4d}  {c['dataset']}/{c['embedder']}  styles={','.join(styles)}  {c['category']}  seed={c['seed']}")
PYLIST
    )
    ;;

  size)
    # Time ONE cell under the shipped arm before committing; the region pair
    # cell is the critical path (#2865: 75-83 min at 300 steps on that stack).
    IDX="${2:-0}"
    export CALIB_EXP="$BASE/sizing"
    export CALIB_RESULTS="$PREP"
    SIZING="$BASE/sizing/cell-$IDX"
    mkdir -p "$BASE/sizing/logs" "$SIZING"
    ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME OMP_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$MKL_NUM_THREADS OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
    S=$(sbatch --parsable --job-name=hinge3557-size --mem="$CALIB_MEM" --cpus-per-task="$CALIB_CPUS" \
      --time="$CALIB_TIME" --partition=cpu --export=ALL \
      --output="$BASE/sizing/logs/size-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && time python run_cells.py --index $IDX --outdir $SIZING")
    require_jobid "$S" "size (cell $IDX)"
    echo "size job: $S (cell $IDX)  ->  $BASE/sizing/logs/size-$S.out"
    ;;

  incumbent)
    # ARM A.  CALIB_LIVE_CUT_RULE UNSET: the harness resolves the app's own
    # FOLD_ANCHOR_CUT_RULE, so this arm cannot drift from production.
    unset CALIB_LIVE_CUT_RULE
    submit_arm incumbent
    ;;

  hinge)
    # ARM B.  Deliberately off-production, and therefore DECLARED.
    export CALIB_LIVE_CUT_RULE=hinge
    submit_arm hinge --diverges live_cut_rule
    ;;

  analyze)
    activate_venv
    mkdir -p "$BASE/logs" "$BASE/analysis"
    export CALIB_EXP="$BASE" CALIB_RESULTS="$BASE/incumbent/results"
    AENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS"
    DEP=()
    [[ -n "${HINGE3557_DEPEND:-}" ]] && DEP=(--dependency="afterany:${HINGE3557_DEPEND}")
    A=$(sbatch --parsable "${DEP[@]}" --job-name=hinge3557-analyze --mem="$CALIB_ANALYZE_MEM" \
      --cpus-per-task=4 --time="$CALIB_ANALYZE_TIME" --partition=cpu --export=ALL \
      --output="$BASE/logs/analyze-%j.out" \
      --wrap="source $WT/gridenv.sh && $AENVX && cd $HERE && python analyze_hinge_3557.py --incumbent $BASE/incumbent/results --hinge $BASE/hinge/results --out $BASE/analysis")
    require_jobid "$A" "analyze"
    echo "analyze job: $A -> $BASE/logs/analyze-$A.out"
    ;;

  *)
    echo "usage: $0 {prepare|list|size [idx]|incumbent|hinge|analyze}" >&2
    exit 2
    ;;
esac
