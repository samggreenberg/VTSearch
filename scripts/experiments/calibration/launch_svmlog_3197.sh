#!/usr/bin/env bash
# #3197 Stage B: why does the linear SVM head beat the logistic one?  The heads
# inside the Autopilot loop.
#
#   bash launch_svmlog_3197.sh datadir     # ONCE: private unit-norm datadir (see make_datadir.py)
#   bash launch_svmlog_3197.sh prepare     # ONCE: category selection, shared by every arm
#   bash launch_svmlog_3197.sh list        # which array index is which cell
#   bash launch_svmlog_3197.sh size [idx]  # time ONE cell before committing
#   bash launch_svmlog_3197.sh <arm>       # one arm = one head = one CALIB_EXP
#   bash launch_svmlog_3197.sh all         # every arm in ARMS
#
# Arms (each is its own trajectory - the head drives Autopilot's `hard` pick,
# so two heads never share a vote sequence and can only be paired on the CELL):
#
#   svm      the SHIPPED head (CALIB_HEAD unset -> PRODUCTION_HEAD), C = 1
#   linear   the logistic head the SVM replaced (balanced BCE, Adam, <=200
#            epochs, patience 10, label smoothing 0.05, weight decay 1e-4)
#   mlp      the retired auto-sized MLP, for reference (the issue asks for it)
#   linconv  the logistic head run to convergence (2000 epochs, no early stop):
#            the in-loop half of "is it the loss, or the fit?"
#   svmc01   the SVM at C = 0.1  } the in-loop half of "is it just the
#   svmc10   the SVM at C = 10   } regularisation strength?"
#
# Stage A (the heads on FIXED vote sets, including these arms' own Autopilot
# vote sets replayed) is scripts/experiments/svm_vs_logistic/stage_a.py.
# Design and decision rules: docs/experiments/2026-09-22-svm-vs-logistic-3197/.
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- check what was submitted" >&2' ERR

MODE="${1:-}"

export VTS_REPO="${VTS_REPO:-/expscratch/$USER/worktrees/vts-svmlog-3197}"
WT="$VTS_REPO"
HERE="$WT/scripts/experiments/calibration"
BASE="${SVMLOG_BASE:-/expscratch/$USER/svmlog-3197/stageB}"
PREP="$BASE/prepare/results"
DATADIR="${SVMLOG_DATADIR:-/expscratch/$USER/svmlog-3197/datadir}"
ARMS="${SVMLOG_ARMS:-svm linear mlp linconv svmc01 svmc10}"

# --- environment -------------------------------------------------------------
# Three pile datasets that depend on none of vg_scale, FullMarks or coco_better.
# Whole-image embedders only: the head question is the same Linear(D, 1) under
# region voting, but max-pooling and per-bag flooding weights would add two more
# moving parts to a study whose point is to have as few as possible.  Region
# voting is a scoped-out follow-up, not an oversight (see the report).
export CALIB_DATASETS="${CALIB_DATASETS:-caltech101_m,coco_val,visual_genome_m}"
export CALIB_CALTECH_EMBEDDERS="${CALIB_CALTECH_EMBEDDERS:-siglip,siglip2_l}"
export CALIB_COCO_EMBEDDERS="${CALIB_COCO_EMBEDDERS:-siglip,siglip2_l}"
export CALIB_VG_EMBEDDERS="${CALIB_VG_EMBEDDERS:-siglip,siglip2_l}"

# --- science knobs -----------------------------------------------------------
# The shipped threshold path (fold-anchored fusion).  Everything else UNSET so
# it resolves to production; CALIB_HEAD and the head's own fit knobs are set
# per ARM below and nowhere else.
export CALIB_SAFE_THRESHOLDS=1
# The inclusion sweep side-frame is not read by this study; k = 0 alone keeps it
# small (the base rows are unaffected either way).
export CALIB_SWEEP_KS="${CALIB_SWEEP_KS:-0}"
export CALIB_EMIT_PICKS=1           # Stage A replays these vote sequences
export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-150}"
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-5}"
# Seed-major: a truncated array loses its last seeds uniformly, not whole
# categories (#3287's argument).
export CALIB_CELL_ORDER="${CALIB_CELL_ORDER:-seed}"

# --- resources ---------------------------------------------------------------
# CPU only: every head here is milliseconds-to-seconds on CPU, and two peer
# sessions need the GPUs.
export CALIB_PARTITION=cpu
export CALIB_GRES=none
export CALIB_CPUS=1
export CALIB_MEM="${CALIB_MEM:-6G}"
export CALIB_TIME="${CALIB_TIME:-4:00:00}"
export CALIB_CONC="${CALIB_CONC:-40}"
export CALIB_ANALYZE="${CALIB_ANALYZE:-noop.py}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# shellcheck disable=SC1091
source "$WT/scripts/experiments/pile/pile_env.sh"
# The pile's models, but THIS study's datadir: symlinks to the pile's cells plus
# a unit-norm copy of the one cell that is not (make_datadir.py says why).
export VTSEARCH_DATA_DIR="$DATADIR"

require_jobid() {
  if ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $2 was REFUSED by sbatch (no job id came back)." >&2
    exit 1
  fi
}

activate_venv() {
  # shellcheck disable=SC1091
  source "$WT/gridenv.sh" >/dev/null 2>&1 || { echo "ERROR: no venv via $WT/gridenv.sh" >&2; exit 1; }
}

link_prepare() {
  local rd="$1"
  mkdir -p "$rd/cells"
  [[ -e "$rd/prepare_info.json" ]] || ln -s "$PREP/prepare_info.json" "$rd/prepare_info.json"
  [[ -e "$rd/crops" ]] || ln -s "$PREP/crops" "$rd/crops"
}

# Per-arm knobs.  Each arm exports exactly the knobs that make it that arm, and
# declares them to preflight.
arm_env() {
  unset CALIB_HEAD VTSEARCH_SVM_HEAD_C VTSEARCH_TRAIN_EPOCHS VTSEARCH_TRAIN_PATIENCE
  DIVERGES=""
  case "$1" in
    svm) ;;  # CALIB_HEAD left UNSET on purpose: resolves to PRODUCTION_HEAD
    linear) export CALIB_HEAD=linear; DIVERGES="head" ;;
    mlp) export CALIB_HEAD=mlp; DIVERGES="head" ;;
    linconv)
      export CALIB_HEAD=linear VTSEARCH_TRAIN_EPOCHS=2000 VTSEARCH_TRAIN_PATIENCE=0
      DIVERGES="head,train_epochs,train_patience" ;;
    svmc01) export VTSEARCH_SVM_HEAD_C=0.1; DIVERGES="svm_head_c" ;;
    svmc10) export VTSEARCH_SVM_HEAD_C=10; DIVERGES="svm_head_c" ;;
    *) echo "unknown arm $1" >&2; exit 1 ;;
  esac
}

submit_arm() {
  local arm="$1"
  [[ -f "$PREP/prepare_info.json" ]] || { echo "ERROR: run '$0 prepare' first" >&2; exit 1; }
  activate_venv
  arm_env "$arm"
  export CALIB_EXP="$BASE/$arm"
  export CALIB_RESULTS="$CALIB_EXP/results"
  mkdir -p "$CALIB_EXP/logs"
  link_prepare "$CALIB_RESULTS"
  local -a div=()
  [[ -n "$DIVERGES" ]] && div=(--diverges "$DIVERGES")
  if [[ -x "$WT/scripts/experiments/preflight.sh" ]]; then
    bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --need-gb 10 \
      --reuse-prepare "$PREP" "${div[@]}" \
      --job-name "svmlog-$arm" --mem "$CALIB_MEM" --conc "$CALIB_CONC" || {
      echo "preflight FAILED for $arm" >&2
      [[ "${PREFLIGHT_SKIP:-0}" == "1" ]] || exit 1
    }
  fi
  echo "=== arm $arm (head=${CALIB_HEAD:-<production>} C=${VTSEARCH_SVM_HEAD_C:-<shipped>}" \
       "epochs=${VTSEARCH_TRAIN_EPOCHS:-<shipped>} patience=${VTSEARCH_TRAIN_PATIENCE:-<shipped>}) -> $CALIB_EXP ==="
  CALIB_JOB_NAME="svmlog-$arm" bash "$HERE/launch_cells.sh" || { echo "arm $arm FAILED to submit" >&2; exit 1; }
  local id
  id="$(cat "$CALIB_EXP/logs/.cells_jobid" 2>/dev/null || true)"
  require_jobid "$id" "arm $arm's cells array"
  echo "arm $arm array: $id"
}

case "$MODE" in
  datadir)
    mkdir -p "$BASE/logs"
    D=$(sbatch --parsable --job-name=svmlog-datadir --mem=16G --cpus-per-task=1 --time=0:30:00 \
      --partition=cpu --output="$BASE/logs/datadir-%j.out" \
      --wrap="source $WT/gridenv.sh && python $WT/scripts/experiments/svm_vs_logistic/make_datadir.py --out $DATADIR")
    require_jobid "$D" "datadir"
    echo "datadir job: $D"
    ;;
  prepare)
    export CALIB_EXP="$BASE/prepare" CALIB_RESULTS="$PREP"
    mkdir -p "$BASE/prepare/logs" "$PREP/cells" "$PREP/crops"
    ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
    P=$(sbatch --parsable --job-name=svmlog-prep --mem=24G --cpus-per-task=2 --time=1:30:00 \
      --partition=cpu --export=ALL --output="$BASE/prepare/logs/prepare-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python prepare_data.py")
    require_jobid "$P" "prepare"
    echo "prepare job: $P -> $BASE/prepare/logs/prepare-$P.out"
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
    print(f"  {i:4d}  {c['dataset']}/{c['embedder']}  {c['category']}  seed={c['seed']}")
PYLIST
    )
    ;;
  size)
    IDX="${2:-0}"
    ARM="${3:-svm}"
    arm_env "$ARM"
    export CALIB_EXP="$BASE/sizing" CALIB_RESULTS="$PREP"
    SIZING="$BASE/sizing/cell-$ARM-$IDX"
    mkdir -p "$BASE/sizing/logs" "$SIZING"
    ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
    S=$(sbatch --parsable --job-name=svmlog-size --mem="$CALIB_MEM" --cpus-per-task=1 --time="$CALIB_TIME" \
      --partition=cpu --export=ALL --output="$BASE/sizing/logs/size-$ARM-$IDX-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && /usr/bin/time -v python run_cells.py --index $IDX --outdir $SIZING")
    require_jobid "$S" "size"
    echo "size job: $S (arm $ARM, cell $IDX)"
    ;;
  all)
    for a in $ARMS; do submit_arm "$a"; done
    ;;
  svm|linear|mlp|linconv|svmc01|svmc10)
    submit_arm "$MODE"
    ;;
  *)
    sed -n 2,27p "$0"; exit 1 ;;
esac
