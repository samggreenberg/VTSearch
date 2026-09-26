#!/usr/bin/env bash
# #3959: the Gaussian-process head at GRID scale, on the calibration harness.
#
#   bash launch_grid_3959.sh prepare <env>          # ONCE per env: category selection, shared by every arm
#   bash launch_grid_3959.sh list <env>             # which array index is which cell
#   bash launch_grid_3959.sh size <env> <arm> [idx] # time ONE cell before committing
#   bash launch_grid_3959.sh <env> <arm>            # one (env, arm) = one CALIB_EXP = one array
#   bash launch_grid_3959.sh <env> all              # every arm in ARMS
#   bash launch_grid_3959.sh status
#
# Envs (the ranking has room to move on both; the #3954 pilot's caltech101_m had
# AUROC 1.00 on every cell and could not separate heads by ranking):
#
#   better   coco_better: every class x band cell (CALIB_CATEGORY_MODE=all), 100
#            positives against 9,900 shared negatives -> ~1% prevalence by design
#   natural  coco_val at its NATURAL per-category prevalence, 12 categories
#            spread common -> rare
#
# both with the shipped whole-image embedder (`siglip`) and the premium one
# (`siglip2_l`), each opening on its own text sort.
#
# Arms (each is its own trajectory - the head and the cut drive Autopilot's Hard
# pick - so arms pair on the CELL, never on the step):
#
#   app                 the SHIPPED detector: linear-SVM head, fold-anchored fusion
#   app_xcal            the same head, plain cross-calibration cut (safe thresholds OFF)
#   app_gmm             the same head, the GMM midpoint on its own haystack (live `gmm_mid`)
#   gp_rbf_anch         RBF GP, the GP-NATIVE cut: the shipped fold-anchored estimator
#                       on the GP's OWN calibration folds (CALIB_STANDALONE_CUT=anchored)
#   gp_rbf_xcal         RBF GP, plain cross-calibration raw cut (the pilot's `gp_rbf`)
#   gp_rbf_gmm          RBF GP, the GMM midpoint on its own haystack (live `gmm_mid`)
#   gp_dot_anch         dot-product GP (Bayesian linear), GP-native cut
#   gp_rbf_anch_maxvar  RBF GP, GP-native cut, max-posterior-spread Hard pick
#
# The three rules each run under both heads, so `gp_* - app_*` at the SAME rule
# is the head with the rule held fixed; `app - app_xcal` is what the shipped rule
# is worth on the shipped head (the pilot found it NEGATIVE on caltech101_m).
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- check what was submitted" >&2' ERR

MODE="${1:-}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
export VTS_REPO="$WT"
CAL="$WT/scripts/experiments/calibration"
ROOT="${GPGRID_ROOT:-/expscratch/$USER/gp-grid-3959}"
ARMS="${GPGRID_ARMS:-app app_xcal app_gmm gp_rbf_anch gp_rbf_xcal gp_rbf_gmm gp_dot_anch gp_rbf_anch_maxvar}"

# --- environment -------------------------------------------------------------
env_env() {
  unset CALIB_CATEGORY_MODE CALIB_N_CATEGORIES CALIB_COCO_BETTER_EMBEDDERS CALIB_COCO_EMBEDDERS
  case "$1" in
    better)
      export CALIB_DATASETS=coco_better
      export CALIB_COCO_BETTER_EMBEDDERS=siglip,siglip2_l
      export CALIB_CATEGORY_MODE=all
      export CALIB_N_SEEDS="${GPGRID_BETTER_SEEDS:-2}" ;;
    natural)
      export CALIB_DATASETS=coco_val
      export CALIB_COCO_EMBEDDERS=siglip,siglip2_l
      export CALIB_N_CATEGORIES="${GPGRID_NATURAL_CATEGORIES:-12}"
      export CALIB_N_SEEDS="${GPGRID_NATURAL_SEEDS:-8}" ;;
    *) echo "unknown env '$1' (better | natural)" >&2; exit 1 ;;
  esac
  ENV_NAME="$1"
  BASE="$ROOT/$1"
  PREP="$BASE/prepare/results"
}

# --- science knobs (everything unset resolves to production) ----------------
export CALIB_SAFE_THRESHOLDS=1
export CALIB_REQUIRE_OPENING=text
export CALIB_REQUIRE_SEED_QUERY=1
export CALIB_SWEEP_KS="${CALIB_SWEEP_KS:-0}"
export CALIB_EMIT_PICKS=1
export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-150}"
# Seed-major: a truncated array loses its last seeds uniformly, not whole categories.
export CALIB_CELL_ORDER=seed
export CALIB_REPOOL_VARIANTS="" CALIB_SCHEDULE_VARIANTS="" CALIB_FOLD_COUNTS=""

# --- resources ---------------------------------------------------------------
# CPU only: every head here fits in milliseconds on <=150 votes; the cost is
# scoring the ~5k-item haystack per step.
export CALIB_PARTITION=cpu
export CALIB_GRES=none
export CALIB_CPUS=1
export CALIB_MEM="${CALIB_MEM:-8G}"
export CALIB_TIME="${CALIB_TIME:-4:00:00}"
export CALIB_CONC="${CALIB_CONC:-30}"
export CALIB_ANALYZE="${CALIB_ANALYZE:-noop.py}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# shellcheck disable=SC1091
source "$WT/scripts/experiments/pile/pile_env.sh"

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
  unset CALIB_TRAINER CALIB_STRATEGY CALIB_STANDALONE_CUT CALIB_LIVE_THRESHOLD
  export CALIB_SAFE_THRESHOLDS=1
  DIVERGES=""
  case "$1" in
    app) ;;
    app_xcal) export CALIB_SAFE_THRESHOLDS=0; DIVERGES="safe_thresholds" ;;
    app_gmm) export CALIB_LIVE_THRESHOLD=gmm_mid; DIVERGES="live_threshold" ;;
    gp_rbf_anch)
      export CALIB_TRAINER=gp_rbf CALIB_STANDALONE_CUT=anchored; DIVERGES="trainer,standalone_cut" ;;
    gp_rbf_xcal)
      export CALIB_TRAINER=gp_rbf CALIB_SAFE_THRESHOLDS=0; DIVERGES="trainer,safe_thresholds" ;;
    gp_rbf_gmm)
      export CALIB_TRAINER=gp_rbf CALIB_LIVE_THRESHOLD=gmm_mid; DIVERGES="trainer,live_threshold" ;;
    gp_dot_anch)
      export CALIB_TRAINER=gp_dot CALIB_STANDALONE_CUT=anchored; DIVERGES="trainer,standalone_cut" ;;
    gp_rbf_anch_maxvar)
      export CALIB_TRAINER=gp_rbf CALIB_STANDALONE_CUT=anchored CALIB_STRATEGY=autopilot_maxvar
      DIVERGES="trainer,standalone_cut,strategy" ;;
    *) echo "unknown arm $1" >&2; exit 1 ;;
  esac
}

submit_arm() {
  local arm="$1"
  [[ -f "$PREP/prepare_info.json" ]] || { echo "ERROR: run '$0 prepare $ENV_NAME' first" >&2; exit 1; }
  activate_venv
  arm_env "$arm"
  export CALIB_EXP="$BASE/$arm"
  export CALIB_RESULTS="$CALIB_EXP/results"
  mkdir -p "$CALIB_EXP/logs"
  link_prepare "$CALIB_RESULTS"
  local -a div=()
  [[ -n "$DIVERGES" ]] && div=(--diverges "$DIVERGES")
  bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --need-gb 10 \
    --reuse-prepare "$PREP" "${div[@]}" --require-text-seed \
    --job-name "gp3959-$ENV_NAME-$arm" --mem "$CALIB_MEM" --conc "$CALIB_CONC" || {
    echo "preflight FAILED for $ENV_NAME/$arm" >&2
    [[ "${PREFLIGHT_SKIP:-0}" == "1" ]] || exit 1
  }
  echo "=== $ENV_NAME/$arm (trainer=${CALIB_TRAINER:-app} strategy=${CALIB_STRATEGY:-autopilot}" \
       "cut=${CALIB_STANDALONE_CUT:-raw} safe=${CALIB_SAFE_THRESHOLDS} live=${CALIB_LIVE_THRESHOLD:-shipped}) -> $CALIB_EXP ==="
  CALIB_JOB_NAME="gp3959-$ENV_NAME-$arm" bash "$CAL/launch_cells.sh" || { echo "$arm FAILED to submit" >&2; exit 1; }
  local id
  id="$(cat "$CALIB_EXP/logs/.cells_jobid" 2>/dev/null || true)"
  require_jobid "$id" "$ENV_NAME/$arm's cells array"
  echo "$ENV_NAME/$arm array: $id"
}

case "$MODE" in
  prepare)
    env_env "${2:?usage: $0 prepare <env>}"
    activate_venv
    export CALIB_EXP="$BASE/prepare" CALIB_RESULTS="$PREP"
    mkdir -p "$CALIB_EXP/logs" "$PREP"
    J=$(sbatch --parsable --job-name="gp3959-prep-$ENV_NAME" --mem=32G --cpus-per-task=4 --time=2:00:00 \
      --partition=cpu --export=ALL --output="$CALIB_EXP/logs/prepare-%j.out" \
      --wrap="source $WT/gridenv.sh && cd $CAL && python prepare_data.py")
    require_jobid "$J" "prepare"
    echo "prepare $ENV_NAME: job $J -> $PREP"
    ;;
  baseline)
    # The click-0 anchor every curve starts from: the typed query's own sort,
    # cut at its GMM line, scored on each cell's test half.
    env_env "${2:?usage: $0 baseline <env>}"
    activate_venv
    export CALIB_EXP="$BASE/prepare" CALIB_RESULTS="$PREP"
    dep=()
    [[ -n "${GPGRID_AFTER:-}" ]] && dep=(--dependency="afterok:$GPGRID_AFTER")
    J=$(sbatch --parsable --job-name="gp3959-text-$ENV_NAME" --mem=16G --cpus-per-task=2 --time=2:00:00 \
      "${dep[@]}" --partition=cpu --export=ALL --output="$CALIB_EXP/logs/text-baseline-%j.out" \
      --wrap="source $WT/gridenv.sh && cd $CAL && python text_baseline.py --results $PREP --out $PREP/text_baseline.csv")
    require_jobid "$J" "text baseline"
    echo "text baseline $ENV_NAME: job $J -> $PREP/text_baseline.csv"
    ;;
  list)
    env_env "${2:?usage: $0 list <env>}"
    activate_venv
    export CALIB_RESULTS="$PREP"
    (cd "$CAL" && python - <<'PY'
import json, os, sys
sys.path.insert(0, ".")
import experiment_config as cfg
info = json.load(open(os.path.join(os.environ["CALIB_RESULTS"], "prepare_info.json")))
cats = {ds: {e: v["selected_categories"] for e, v in info["datasets"][ds].items()} for ds in cfg.DATASETS}
for i, c in enumerate(cfg.array_cells(cats)):
    print(i, c["dataset"], c["embedder"], c["category"], c["seed"])
PY
    )
    ;;
  size)
    env_env "${2:?usage: $0 size <env> <arm> [idx]}"
    arm="${3:?usage: $0 size <env> <arm> [idx]}"
    idx="${4:-0}"
    activate_venv
    arm_env "$arm"
    export CALIB_EXP="$BASE/sizing/$arm" CALIB_RESULTS="$BASE/sizing/$arm/results"
    mkdir -p "$CALIB_EXP/logs"
    link_prepare "$CALIB_RESULTS"
    dep=()
    [[ -n "${GPGRID_AFTER:-}" ]] && dep=(--dependency="afterok:$GPGRID_AFTER")
    J=$(sbatch --parsable --job-name="gp3959-size-$ENV_NAME-$arm" --mem=16G --cpus-per-task=1 --time=4:00:00 \
      "${dep[@]}" --partition=cpu --export=ALL --output="$CALIB_EXP/logs/size-$idx-%j.out" \
      --wrap="source $WT/gridenv.sh && cd $CAL && /usr/bin/time -v python run_cells.py --index $idx")
    require_jobid "$J" "size"
    echo "size $ENV_NAME/$arm idx $idx: job $J (read Elapsed + MaxRSS off sacct)"
    ;;
  status)
    squeue -u "$USER" -h -o "%.10i %.36j %.9T %.11M" | grep gp3959 || echo "(nothing queued)"
    for d in "$ROOT"/*/*/results/cells; do
      [[ -d "$d" ]] || continue
      echo "$(ls "$d" | grep -cE '^task_[0-9]+\.csv$') cells  ${d#$ROOT/}"
    done
    ;;
  better|natural)
    env_env "$MODE"
    target="${2:?usage: $0 <env> <arm|all>}"
    if [[ "$target" == all ]]; then
      for a in $ARMS; do submit_arm "$a"; done
    else
      submit_arm "$target"
    fi
    ;;
  *)
    sed -n '2,9p' "$0"; exit 1 ;;
esac
