#!/usr/bin/env bash
# #4110: is a calibration cell bit-identical across the GRID's two CPU types,
# today and with the numeric-library pins?
#
#   bash launch_4110.sh plan            # each arm's node type and env; submits nothing
#   bash launch_4110.sh arms [ARM..]    # submit arm arrays (default: all four)
#   bash launch_4110.sh status          # cells written per arm
#   bash launch_4110.sh analyze         # pair the arms cell by cell (analyze_4110.py)
#
# WHY.  #3840 ran the same arm twice on one commit and 3 of 114 cells differed.
# All 36 cells with any difference had run on an AMD node in one run and an
# Intel node in the other; none of the 60 same-vendor pairs differed at all.
# #4481's OPENBLAS_CORETYPE=Haswell pin came later, but per-op probes on
# today's dev still find three CPU-dependent paths on a cell's way:
#   - numpy's AVX-512 float64 exp/log (the anchored EM) and AVX-512 argsort
#     (tie order), on the Xeon Silver 4114 nodes only;
#   - torch's bundled MKL, whose sgemv/sgemm/VML pick kernels by CPU.
# NPY_DISABLE_CPU_FEATURES and MKL_CBWR=COMPATIBLE made every probed op
# bit-identical across the two node types.  This run asks whether that holds
# for whole cells, and whether today's dev drifts without it.
#
# ARMS.  The app as shipped on one frozen commit, binary SigLIP, coco_better,
# 144 classes x 1 seed, 150 votes, the user's pool at 1% (#4864's grid).  The
# arms differ only in where they run and two env vars:
#   i_base  Intel nodes (Xeon Silver 4114, AVX-512), today's env
#   a_base  AMD nodes (EPYC 7713, AVX2), today's env.  The numpy pin is a no-op
#           on a CPU without AVX-512 (probed), so this is also AMD + numpy pin.
#   i_npy   Intel, numpy pin only (NPY_DISABLE_CPU_FEATURES)
#   i_pin   Intel, numpy pin + MKL_CBWR=COMPATIBLE
#   a_pin   AMD, numpy pin + MKL_CBWR=COMPATIBLE
#   i_full  Intel, numpy + MKL pins + ATEN_CPU_CAPABILITY=avx2
#   a_full  AMD, the same three (the ATen pin is a no-op on an AVX2-only CPU)
#   i_ship, a_ship  the pins as shipped: no pin exported, so run_cells.py sets them itself.
#           Run from the branch (VTS_REPO=<its worktree>) with CELLS_EVERY=24; each
#           cell must match i_full/a_full bit for bit.
#   a_rep   a_base again: the same-node-type repeat every other pair is read against
# The ATen pin was added after i_pin~a_pin still differed from vote 1: the cell
# embeds its typed query live (SigLIP's text tower on the CPU), whose LayerNorm,
# softmax and GELU run ATen's AVX-512 kernels on the Xeons and AVX2 on the EPYCs.
# Pairs read: a_base~a_rep (is a cell deterministic on one node type?), i_base~a_base (does dev drift?), i_npy~a_base (is the numpy pin
# enough?), i_pin~a_pin (do both pins close it?).
# CELLS_EVERY=2 runs every other class (72 cells an arm): enough to count
# divergent cells, at half the Intel-node hours.
#
# All four arms run from one frozen worktree (VTS_REPO), never the one being edited.
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- check what was submitted" >&2' ERR

MODE="${1:-}"

export VTS_REPO="${VTS_REPO:-/expscratch/$USER/worktrees/vts-4110-run}"
WT="$VTS_REPO"
HERE="$WT/scripts/experiments/calibration"
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${DET_BASE:-/expscratch/$USER/det-4110}"
LADDER="${LADDER_BASE:-/expscratch/$USER/progression-4184-h0.01}"

ALL_ARMS="i_base a_base i_npy i_pin a_pin a_rep i_full a_full i_ship a_ship"
CELLS_EVERY="${CELLS_EVERY:-2}"
PIN_NPY="${PIN_NPY:-AVX512F AVX512_SKX}"
PIN_MKL="${PIN_MKL:-COMPATIBLE}"
PIN_ATEN="${PIN_ATEN:-avx2}"

# --- science knobs: the app as shipped ----------------------------------------
export CALIB_SAFE_THRESHOLDS=1
unset CALIB_HEAD CALIB_BLEND_SCHEDULE CALIB_EXCLUDE_VOTED CALIB_CALIBRATION_SEEDS CALIB_LIVE_CUT_RULE
unset CALIB_LIVE_THRESHOLD CALIB_ACQ_INCLUSION_OFFSET CALIB_SPOT_CHECK CALIB_MIN_PRECISION
unset CALIB_CALIBRATION_FRACTION CALIB_CALIBRATE_COUNT CALIB_BETA CALIB_SPLIT_ROUNDING
export CALIB_SCHEDULE_VARIANTS=""
export CALIB_REPOOL_VARIANTS=""
export CALIB_FOLD_COUNTS=""
export CALIB_ANCHORED=0
export CALIB_CUT_INCL_KS=""
export CALIB_HAYSTACK_PREVALENCE=0.01

# --- environment (#4864's binary grid, one seed) ------------------------------
export CALIB_DATASETS=coco_better
export CALIB_COCO_BETTER_EMBEDDERS=siglip
export CALIB_CATEGORY_MODE=all
export CALIB_PATCH_STYLES=max_patch
export CALIB_REQUIRE_OPENING=text
export CALIB_REQUIRE_SEED_QUERY=1
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-1}"
export CALIB_CELL_ORDER="${CALIB_CELL_ORDER:-seed}"
export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-150}"
export CALIB_CELLS_GZIP="${CALIB_CELLS_GZIP:-1}"

# --- ops -----------------------------------------------------------------------
# #4864 (on mixed nodes): a bundle of 8 took 26-47 min, < 1.2 GB.  The Xeons run
# the scoring ~3x slower than the EPYCs, so bundles are 4 here.
BUNDLE="${BUNDLE:-4}"
MEM="${CALIB_MEM:-3G}"
TIME="${CALIB_TIME:-4:00:00}"
CONC="${CALIB_CONC:-6}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OPENBLAS_CORETYPE="${OPENBLAS_CORETYPE:-Haswell}"

# shellcheck disable=SC1091
source "$WT/scripts/experiments/pile/pile_env.sh"

require_jobid() {
  [[ "$1" =~ ^[0-9]+$ ]] || {
    echo "ERROR: $2 was REFUSED by sbatch (no job id came back)." >&2
    exit 1
  }
}

arm_env() { # sets the node type and the pin env for arm $1
  case "$1" in
    i_*) CONSTRAINT=intel ;;
    a_*) CONSTRAINT=amd ;;
    *)
      echo "unknown arm '$1'; expected one of: $ALL_ARMS" >&2
      exit 2
      ;;
  esac
  case "$1" in
    *_npy) PINX="export NPY_DISABLE_CPU_FEATURES='$PIN_NPY' && unset MKL_CBWR ATEN_CPU_CAPABILITY" ;;
    *_pin) PINX="export NPY_DISABLE_CPU_FEATURES='$PIN_NPY' MKL_CBWR=$PIN_MKL && unset ATEN_CPU_CAPABILITY" ;;
    *_full) PINX="export NPY_DISABLE_CPU_FEATURES='$PIN_NPY' MKL_CBWR=$PIN_MKL ATEN_CPU_CAPABILITY=$PIN_ATEN" ;;
    *_base | *_rep) PINX="unset NPY_DISABLE_CPU_FEATURES MKL_CBWR ATEN_CPU_CAPABILITY" ;;
    *_ship) PINX="unset NPY_DISABLE_CPU_FEATURES MKL_CBWR ATEN_CPU_CAPABILITY OPENBLAS_CORETYPE" ;;
  esac
}

set_exp() {
  export CALIB_EXP="$BASE/$1"
  export CALIB_RESULTS="$CALIB_EXP/results"
  export CALIB_JOB_NAME="det4110-$1"
  mkdir -p "$CALIB_EXP/logs" "$CALIB_RESULTS/cells"
  ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
}

link_prepare() {
  local src="$LADDER/prepare/results"
  [[ -f "$src/prepare_info.json" ]] || {
    echo "ERROR: no prepare at $src" >&2
    exit 1
  }
  cp -n "$src/prepare_info.json" "$CALIB_RESULTS/"
  [[ -d "$src/crops" && ! -e "$CALIB_RESULTS/crops" ]] && cp -r "$src/crops" "$CALIB_RESULTS/crops"
  return 0
}

grid_size() {
  (cd "$HERE" && source "$WT/gridenv.sh" >/dev/null 2>&1 && eval "$ENVX" && python run_cells.py --print-cells 2>/dev/null | tail -1)
}

case "$MODE" in
  plan)
    for a in $ALL_ARMS; do
      arm_env "$a"
      printf '%-7s constraint=%-6s %s\n' "$a" "$CONSTRAINT" "$PINX"
    done
    echo "bundle=$BUNDLE cells/task, conc=%$CONC per arm, mem=$MEM, time=$TIME"
    echo "repo=$WT ($(git -C "$WT" rev-parse --short HEAD))"
    ;;

  arms)
    shift
    ARMS="${*:-$ALL_ARMS}"
    for a in $ARMS; do
      (
        arm_env "$a"
        set_exp "$a"
        link_prepare
        N=$(grid_size)
        [[ "$N" =~ ^[0-9]+$ && "$N" -gt 0 ]] || {
          echo "ERROR: cell count '$N' for $a" >&2
          exit 1
        }
        # Cells run: every CELLS_EVERY-th index, M of them; bundled task j runs picks j, j+T, ...
        M=$(((N + CELLS_EVERY - 1) / CELLS_EVERY))
        T=$(((M + BUNDLE - 1) / BUNDLE))
        if [[ -x "$WT/scripts/experiments/preflight.sh" ]]; then
          bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --need-gb 3 --job-name "$CALIB_JOB_NAME" \
            --mem "$MEM" --conc "$CONC" --array-tasks "$T" --reuse-prepare "$LADDER/prepare/results" || {
            echo "preflight FAILED ($CALIB_JOB_NAME)" >&2
            [[ "${PREFLIGHT_SKIP:-0}" == "1" ]] || exit 1
          }
        fi
        # Each task records its node, CPU and the kernels it resolved, so a pair can be read by hardware.
        # A copy, so editing this tree mid-run cannot change what a later task runs.
        cp "$SELF/node_stamp.py" "$CALIB_EXP/node_stamp.py"
        body="python $CALIB_EXP/node_stamp.py >> $CALIB_EXP/logs/nodes.tsv; for k in \$(seq 0 $((BUNDLE - 1))); do p=\$((SLURM_ARRAY_TASK_ID + k * $T)); [ \$p -lt $M ] || continue; i=\$((p * $CELLS_EVERY)); python run_cells.py --index \$i --outdir $CALIB_RESULTS/cells || echo \"cell \$i FAILED\"; done"
        J=$(sbatch --parsable --job-name="$CALIB_JOB_NAME" --array="0-$((T - 1))%$CONC" --mem="$MEM" --cpus-per-task=1 \
          --time="$TIME" --partition=cpu --constraint="$CONSTRAINT" --export=ALL \
          --output="$CALIB_EXP/logs/cells-%A_%a.out" \
          --wrap="source $WT/gridenv.sh && $ENVX && $PINX && cd $HERE && $body")
        require_jobid "$J" "$a"
        echo "$a: $M of $N cells (every $CELLS_EVERY) in $T tasks of $BUNDLE on $CONSTRAINT -> job $J"
      ) || exit 1
    done
    ;;

  status)
    for a in $ALL_ARMS; do
      d="$BASE/$a/results/cells"
      n="$(find "$d" -maxdepth 1 -name 'task_[0-9][0-9][0-9][0-9].csv*' -size +0 2>/dev/null | wc -l)"
      q="$(squeue -u "$USER" -h -r -n "det4110-$a" -o %i 2>/dev/null | wc -l)"
      printf '%-7s %4s cells written  %3s tasks queued/running\n' "$a" "$n" "$q"
    done
    ;;

  analyze)
    J=$(sbatch --parsable --job-name=det4110-analyze --partition=cpu --mem=8G --cpus-per-task=1 --time=1:00:00 \
      --output="$BASE/logs/analyze-%j.out" \
      --wrap="source $WT/gridenv.sh && python $SELF/analyze_4110.py --base $BASE --out $BASE/analysis")
    require_jobid "$J" "analyze"
    echo "analyze job $J -> $BASE/analysis"
    ;;

  *)
    echo "usage: $0 {plan|arms [ARM..]|status|analyze}" >&2
    exit 2
    ;;
esac
