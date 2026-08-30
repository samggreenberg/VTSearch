#!/usr/bin/env bash
# #3196: does the Inclusion knob still have authority under the linear SVM head?
#
#   bash launch_incl_3196.sh prepare        # stage 0, ONCE, shared by both heads
#   bash launch_incl_3196.sh list           # which array index is which cell
#   bash launch_incl_3196.sh size [idx]     # time ONE cell before committing
#   bash launch_incl_3196.sh svm            # arm A: the SHIPPED head
#   bash launch_incl_3196.sh linear         # arm B: the head the SVM replaced
#   bash launch_incl_3196.sh analyze        # cross-head analysis, after both drain
#
# Design + pre-registered decision rules:
#   docs/experiments/inclusion-knob-3196/PLAN.md
#
# WHY THE HEAD IS TWO RUNS AND NOT TWO ARMS.  The fold-anchored threshold drives
# Autopilot's `hard` pick, so a different head collects DIFFERENT VOTES from its
# first retrain onward.  There is no trajectory the two heads share, so there is
# no `CALIB_EXP` they can share either (preflight check 1).  Everything the cut
# sweep itself varies - rule, k, q_tilt step - is eval-only and rides one
# trajectory: the per-fold anchored EM does not depend on the cut rule or the
# inclusion, so one fit per step serves the whole (rule x k) grid.  That is the
# same no-refit re-cut the app does when the user drags the slider, which is why
# 21 stops x 9 arms is arithmetic rather than 189 runs.
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- NOTHING WAS SUBMITTED" >&2' ERR

MODE="${1:-}"

export VTS_REPO="${VTS_REPO:-/exp/$USER/projects/vts-incl-3196}"
WT="$VTS_REPO"
HERE="$WT/scripts/experiments/calibration"

# /expscratch, not /exp: each arm emits ~28 k cut-inclusion rows per step per
# style, which runs to several GB of cells per arm, and /exp/$USER is a 50G mount.
BASE="${INCL3196_BASE:-/expscratch/$USER/incl-3196}"
PREP="$BASE/prepare/results"

# --- the sweep this run exists for -------------------------------------------
# The knob's whole nominal range, every stop.  #2865 sampled 13 of 21 and could
# see THAT a rule saturates but not WHERE the flat band begins, which is the
# question here: a band at the ends is a slider users never reach, a band around
# 0 is a slider that does nothing.
export CALIB_CUT_INCL_KS="${CALIB_CUT_INCL_KS:--10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10}"
# The #2865 candidate set, unchanged so the two studies' tables are comparable.
# `mid` is here as the INSTRUMENT CHECK, not as a candidate: it never reads the
# cost weights, so it must come back inert, and a `mid` that moves means the
# flatness measure is broken before any headline number is read.
export CALIB_ANCHORED_RULES="${CALIB_ANCHORED_RULES:-mid,mid_tilt,rate,cross_tilt,q_tilt}"
# `q_tilt`'s step is a FREE PARAMETER with no derivation (`FOLD_ANCHOR_QTILT_STEP`
# is a placeholder).  Sweeping it is what makes it a candidate rather than a rule
# with a magic number in it.
export CALIB_CUT_INCL_QTILT_STEPS="${CALIB_CUT_INCL_QTILT_STEPS:-0.005,0.01,0.02,0.04,0.08}"
# kappa PINNED at the shipped 0.3 (#2861): this run is about the knob, and
# re-opening the anchor mass would confound the two and multiply the arms by 5.
export CALIB_ANCHORED_WEIGHTS="${CALIB_ANCHORED_WEIGHTS:-0.3}"
export CALIB_ANCHORED_FOLD_ARMS=1
export CALIB_ANCHORED_FOLD_COMBINES=qmean
export CALIB_ANCHORED_CHECKPOINTS="${CALIB_ANCHORED_CHECKPOINTS:-25,60,100,150}"

# --- science knobs -----------------------------------------------------------
# The SHIPPED threshold path: the anchored arms need the per-fold haystacks
# `_safe_threshold_for_step` supplies, and `docs/ML.md` is explicit that fusion
# is not a setting the app has.
export CALIB_SAFE_THRESHOLDS=1
export CALIB_ANCHORED=1
# Everything else deliberately UNSET, so it resolves to production and check 12
# has something true to check: CALIB_BLEND_SCHEDULE (the app's per-mode default,
# #2841), CALIB_CALIBRATION_FRACTION (the per-space default, #3287),
# CALIB_ACQ_INCLUSION_OFFSET (the shipped -3, #2877/#3320).  CALIB_HEAD is set
# per ARM below and nowhere else.
export CALIB_ANALYZE="${CALIB_ANALYZE:-analyze_cutincl.py}"

# --- environment -------------------------------------------------------------
# ONE dataset, `vg_scale` (#3156): 12 hand-checked classes at three box-size
# bands = 36 designated cells, 100 positives and prevalence 0.0250 in EVERY one.
# A threshold IS a quantile of a calibration set, so a grid whose cells differ
# 60-fold in positives would confound the swept axis with prevalence; and the
# band is the SEPARABILITY LADDER the mechanism under test runs on (the tilt
# dies when the rate root stays interior, which is what a cleanly separated
# haystack produces).  See the PLAN for why this replaces the issue's COCO +
# visual_genome_m pair.
export CALIB_DATASETS="${CALIB_DATASETS:-vg_scale}"
# Every band-suffixed category IS the experiment; re-banding an already-banded
# set would discard it.
export CALIB_CATEGORY_MODE="${CALIB_CATEGORY_MODE:-all}"
# The region half is the PAIR (#3278): DINOv3 has no text tower, so a bare
# `dinov3_patch` arm would open on three random known-goods while every SigLIP
# arm opens on a typed query - a SEEDING contrast hidden inside the voting-mode
# contrast.
export CALIB_VGSCALE_EMBEDDERS="${CALIB_VGSCALE_EMBEDDERS:-siglip,siglip+dinov3_patch}"
export CALIB_REQUIRE_OPENING=text
export CALIB_REQUIRE_SEED_QUERY=1
# BOTH styles on the pair, so voting mode is separable from embedder:
#   siglip/whole  vs  pair/whole      -> the EMBEDDER, at fixed voting mode
#   pair/whole    vs  pair/max_patch  -> the VOTING MODE, at fixed embedder
# Without the middle corner "region voting keeps its knob" cannot be told apart
# from "DINOv3 keeps its knob" (#3115's confound, #3258's).
export CALIB_PATCH_STYLES="${CALIB_PATCH_STYLES:-whole_image,max_patch}"
export CALIB_REPOOL_VARIANTS=""

# --- sizing ------------------------------------------------------------------
# 150 steps: the flat band is read in the DEEP regime (n_votes >= 100), where the
# anchored mixture has enough labels for the cut rule rather than the anchor
# supply to be what is measured, and 150 leaves a real deep band without paying
# for #2865's 300.
export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-150}"
# Seed-major: a truncated array then loses its last SEEDS uniformly across every
# environment (a power problem a report can state) rather than its last
# CATEGORIES entirely (a design failure).  #3287's argument.
export CALIB_CELL_ORDER="${CALIB_CELL_ORDER:-seed}"
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-4}"

# No GPU work: the cells train a small head on cached pile embeddings.
# `launch_cells.sh` defaults to `--partition=gpu --gres=gpu:v100:1`, where the
# `4gpu_tier` QOS caps the array at 2 concurrent tasks.  BOTH must be named -
# the flag is dropped rather than passed as `--gres=none`, which this cluster's
# submit filter rewrites and then rejects (#2897 lost both A/B arms to that,
# with an empty job id as the only symptom).
export CALIB_PARTITION=cpu
export CALIB_GRES=none

# MEASURED on THIS grid by `size` before either arm went in - see the report's
# ops section.  No prior grid's seconds transfer: this cell has no fold grid
# (#3314's had 8 folds) but carries a 21-stop x 9-arm re-cut per step that no
# earlier cell paid for.
export CALIB_MEM="${CALIB_MEM:-16G}"
export CALIB_CPUS=1
export CALIB_TIME="${CALIB_TIME:-12:00:00}"
# BOTH ARMS RUN AT ONCE, so the width below is per arm and the footprint is
# twice it.  1074G is the per-user allowance under QOS `cpu_limit`; 2 x 28 x 16G
# = 896G is 83% of it, under preflight check 8's 90% line, and leaves room for
# the analyze steps.  (`cpu_limit` also charges 2 CPUs per task, so 56 concurrent
# tasks is 112 of the 240-CPU ceiling - memory binds first here, as usual.)
export CALIB_CONC="${CALIB_CONC:-28}"
# The cross-head frame is ~24 M rows over both arms; 16G/40min is where such an
# analysis dies, after the cells have already been paid for.
export CALIB_ANALYZE_MEM="${CALIB_ANALYZE_MEM:-64G}"
export CALIB_ANALYZE_TIME="${CALIB_ANALYZE_TIME:-3:00:00}"

# Pin the BLAS pools to one thread each: dozens of concurrent cells each
# spawning a node-sized pool oversubscribes whatever node they land on.
# Exported HERE, not inside a mode, so `size` measures under the SAME
# environment the array will run in
# (`lessons/2026-08-24-a-login-node-timing-nearly-cut-an-arm.md`).
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
    echo "       Nothing downstream can run; fix the submission and re-launch." >&2
    exit 1
  fi
}

link_prepare() {
  local rd="$1"
  mkdir -p "$rd/cells"
  [[ -e "$rd/prepare_info.json" ]] || ln -s "$PREP/prepare_info.json" "$rd/prepare_info.json"
  [[ -e "$rd/crops" ]] || ln -s "$PREP/crops" "$rd/crops"
}

# Preflight's checks are mostly PYTHON: it imports vtscore to assert the
# region-voting premise and to compare every pinned knob against its shipped
# constant.  A non-interactive login shell has no venv, and the system python is
# old enough that `X | None` raises at import time - so those checks come back
# FAIL for a reason unrelated to the run.  Activate the venv first.
activate_venv() {
  # shellcheck disable=SC1091
  source "$WT/gridenv.sh" >/dev/null 2>&1 || {
    echo "ERROR: could not activate the venv at $WT/gridenv.sh" >&2; exit 1
  }
}

run_preflight() {
  local job="$1"; shift
  [[ -x "$WT/scripts/experiments/preflight.sh" ]] || return 0
  bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --need-gb 30 \
    --require-min-positives 100 \
    --reuse-prepare "$PREP" \
    --require-region-voting vg_scale:siglip+dinov3_patch \
    --contrasts-voting-modes --patch \
    "$@" \
    --job-name "$job" --mem "$CALIB_MEM" --conc "$CALIB_CONC" || {
    echo "preflight FAILED for $job" >&2
    [[ "${PREFLIGHT_SKIP:-0}" == "1" ]] || exit 1
  }
}

# One arm = one head = one full trajectory = one CALIB_EXP.
submit_arm() {
  local arm="$1"; shift          # svm | linear
  local -a div=("$@")            # --diverges ... , or nothing
  if [[ ! -f "$PREP/prepare_info.json" ]]; then
    echo "ERROR: no prepare_info.json at $PREP - run '$0 prepare' first." >&2
    exit 1
  fi
  activate_venv
  export CALIB_EXP="$BASE/$arm"
  export CALIB_RESULTS="$CALIB_EXP/results"
  mkdir -p "$CALIB_EXP/logs"
  link_prepare "$CALIB_RESULTS"
  run_preflight "incl3196-$arm" "${div[@]}"
  echo "=== arm $arm (head=${CALIB_HEAD:-default/production}) -> $CALIB_EXP ==="
  CALIB_JOB_NAME="incl3196-$arm" bash "$HERE/launch_cells.sh" || {
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
    P=$(sbatch --parsable --job-name=incl3196-prep --mem=24G --cpus-per-task=2 \
      --time=1:30:00 --partition=cpu --export=ALL \
      --output="$BASE/prepare/logs/prepare-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python prepare_data.py")
    require_jobid "$P" "prepare"
    echo "prepare job: $P  ->  $BASE/prepare/logs/prepare-$P.out"
    echo "both arms enumerate from $PREP/prepare_info.json"
    ;;

  list)
    activate_venv
    export CALIB_EXP="$BASE/svm" CALIB_RESULTS="$PREP"
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
    # Time ONE cell before committing to 2 x 288.  A cell runs EVERY style of its
    # embedder in one task (they share the loaded pickle), so the pair cell
    # carries BOTH geometries and is the critical path; sizing off the binary
    # cell alone picks a limit the pair cells cannot run in (#3255 lost 74 of 108
    # cells to exactly that).  Size BOTH, and size them under the SHIPPED head -
    # the arm whose numbers the study is read off.
    IDX="${2:-0}"
    export CALIB_EXP="$BASE/sizing"
    export CALIB_RESULTS="$PREP"
    SIZING="$BASE/sizing/cell-$IDX"
    mkdir -p "$BASE/sizing/logs" "$SIZING"
    ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME OMP_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$MKL_NUM_THREADS OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
    S=$(sbatch --parsable --job-name=incl3196-size --mem="$CALIB_MEM" --cpus-per-task="$CALIB_CPUS" \
      --time="$CALIB_TIME" --partition=cpu --export=ALL \
      --output="$BASE/sizing/logs/size-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && time python run_cells.py --index $IDX --outdir $SIZING")
    require_jobid "$S" "size (cell $IDX)"
    echo "size job: $S (cell $IDX)  ->  $BASE/sizing/logs/size-$S.out"
    ;;

  svm)
    # ARM A, the one the product question is read off.  CALIB_HEAD is left UNSET
    # so the harness resolves `PRODUCTION_HEAD` itself: naming `linear_svm` here
    # would pin what is production TODAY and go stale silently the next time the
    # head moves, which is exactly how #2865's launcher came to run the previous
    # head hours after the SVM shipped.  Nothing diverges, so no --diverges.
    unset CALIB_HEAD
    submit_arm svm
    ;;

  linear)
    # ARM B, the reference: what the knob did under the head the SVM replaced.
    # Deliberately off-production, and therefore DECLARED - preflight check 12
    # fails an undeclared divergence from a shipped constant.
    export CALIB_HEAD=linear
    submit_arm linear --diverges head
    ;;

  analyze)
    # Cross-head analysis.  Run after BOTH arms drain; each arm has already run
    # the stock `analyze_cutincl.py` on its own cells (chained by
    # launch_cells.sh), which is the per-arm liveness table.  This step is the
    # part that needs both: the paired head contrast, and H4's offset collapse.
    activate_venv
    export CALIB_EXP="$BASE" CALIB_RESULTS="$BASE/svm/results"
    mkdir -p "$BASE/logs" "$BASE/analysis"
    AENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
    A=$(sbatch --parsable --job-name=incl3196-analyze --mem="$CALIB_ANALYZE_MEM" \
      --cpus-per-task=4 --time="$CALIB_ANALYZE_TIME" --partition=cpu --export=ALL \
      --output="$BASE/logs/analyze-%j.out" \
      --wrap="source $WT/gridenv.sh && $AENVX && cd $HERE && python analyze_incl_3196.py --svm $BASE/svm/results --linear $BASE/linear/results --out $BASE/analysis")
    require_jobid "$A" "the cross-head analyze step"
    echo "cross-head analyze: $A  ->  $BASE/analysis"
    ;;

  *)
    echo "usage: $0 {prepare|list|size [idx]|svm|linear|analyze}" >&2
    exit 2
    ;;
esac
