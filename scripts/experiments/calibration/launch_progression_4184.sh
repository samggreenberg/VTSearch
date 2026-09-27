#!/usr/bin/env bash
# #4184: the deck's calibration ladder as one progression of cost curves.
#
#   bash launch_progression_4184.sh prepare         # stage 0, ONCE (cpu, reads the pile in place)
#   bash launch_progression_4184.sh size [idx] [RUNG]  # time ONE cell (default: cell 0, r7_acq4)
#   bash launch_progression_4184.sh plan            # print each rung's knobs; submits nothing
#   bash launch_progression_4184.sh baseline        # the click-0 text anchor, shared by every rung
#   bash launch_progression_4184.sh rungs [RUNG..]  # submit rung arrays (default: all seven)
#   bash launch_progression_4184.sh status          # cells written per rung, against the grid size
#   bash launch_progression_4184.sh analyze         # the cross-rung analysis, after every rung drains
#
# Plan: docs/experiments/2026-09-25-progression-4184/PLAN.md
#
# WHAT A RUNG IS.  `slides/decks/hold-the-line.deck` walks the threshold through
# a sequence of iterations, each repairing what the last starved on.  Each rung
# below is the app AS THE DECK DRAWS IT at that point - not as it historically
# shipped (#4184 accepts that; the real order interleaved these with two head
# swaps).  Everything a slide does not discuss is held at today's production:
# the linear-SVM head, the SigLIP embedder, the text-sort opening, autopilot
# with the Coverage Atlas, the pinned calibration draw.  So two adjacent rungs
# differ in exactly the one thing the slide between them changes.
#
#   r1_xcal      Grading Your Own Homework   per-fold min-cost cut, averaged   50/50  offset 0
#   r2_gmm       Oops! All Haystack          GMM midpoint on the haystack      50/50  offset 0
#   r3_blend     Cross Examination           r1 + r2 under corridor20          50/50  offset 0
#   r4_rawmean   The Rank & File, build b    anchored folds, raw cuts averaged 50/50  offset 0
#   r5_anchored  The Rank & File, build d    anchored + quantile transfer      50/50  offset 0
#   r6_split70   Train More, Check Less      ... at 70/30                      70/30  offset 0
#   r7_acq4      Second Cut                  ... acquisition at k-4 = TODAY    70/30  offset -4
#
# The Preference slides between r6 and r7 (walk, tilt) change nothing at
# inclusion 0 - `mid_tilt` IS `mid` there, bit for bit - so they are not a rung;
# the tilt is what makes r7's offset possible at all (`mid` is constant in k).
# The Exploration slides are not a rung either: the owner kept the atlas on in
# every rung, so all seven explore the same way today's app does.
#
# EVERY RUNG IS ITS OWN CALIB_EXP.  A rung's line decides what autopilot asks
# next, so a rung is a trajectory, never a re-cut of another rung's trajectory -
# and `_cells_io.load_arm` reads base rows only, so each rung must be a run-level
# arm for the figures to see it at all.  The rungs pair on (category, seed): same
# split, same test set, same opening, different line from the first fitted step.
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- NOTHING WAS SUBMITTED" >&2' ERR

MODE="${1:-}"

export VTS_REPO="${VTS_REPO:-/expscratch/$USER/worktrees/vts-4184}"
WT="$VTS_REPO"
HERE="$WT/scripts/experiments/calibration"
BASE="${PROGRESSION_BASE:-/expscratch/$USER/progression-4184}"

ALL_RUNGS="r1_xcal r2_gmm r3_blend r4_rawmean r5_anchored r6_split70 r7_acq4"

# --- science knobs -------------------------------------------------------------
# The fused path must be on: every retired rule replaces ITS cut and reads the
# fold models and haystack scores it computes (`live_threshold_rules`).
export CALIB_SAFE_THRESHOLDS=1
# Held at production in EVERY rung, so they are unset and resolve there: the
# head (linear SVM, #3198), the blend schedule (corridor20 on binary - which is
# also what r3 blends under), the exclusion floor, the calibration draw, the
# live fold-anchored cut rule.  The three rung knobs are set per rung below.
unset CALIB_HEAD CALIB_BLEND_SCHEDULE CALIB_EXCLUDE_VOTED CALIB_CALIBRATION_SEEDS CALIB_LIVE_CUT_RULE
unset CALIB_LIVE_THRESHOLD CALIB_CALIBRATION_FRACTION CALIB_ACQ_INCLUSION_OFFSET
# No paired re-cut families: every rung is a trajectory, and the side frames
# would only multiply the rows the analyzer has to skip.
export CALIB_SCHEDULE_VARIANTS=""
export CALIB_REPOOL_VARIANTS=""
export CALIB_FOLD_COUNTS=""
export CALIB_ANCHORED=0
export CALIB_CUT_INCL_KS=""

# --- environment ---------------------------------------------------------------
# COCO Better (`coco_better` in code, #4183), every one of its 144 class@band
# cells, BINARY voting only: the progression sits before the deck's Regions
# section, so SigLIP alone - no patch column, no region arm.
export CALIB_DATASETS=coco_better
export CALIB_COCO_BETTER_EMBEDDERS=siglip
export CALIB_CATEGORY_MODE=all
export CALIB_PATCH_STYLES=max_patch
# Declared, not left to whether each cell happens to have a query (#3278).
# `EXPERIMENT_QUERIES["coco_better"]` covers every cell.
export CALIB_REQUIRE_OPENING=text
export CALIB_REQUIRE_SEED_QUERY=1

# --- sizing --------------------------------------------------------------------
# Five seeds, as the issue proposed, and the A/B arithmetic agrees: the curve is a
# mean over 144 x 5 = 720 paired cells per rung, so an adjacent-rung difference
# carries a paired SE near 0.04 / sqrt(720) ~ 0.0015 - resolving a 0.004 step at
# 2 SE (GRID skill, #3840).  Each seed also averages the per-click mean over 144
# cells, which is what makes the drawn curve smooth rather than a hairball.
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-5}"
# Seed-major, so a truncated array loses whole seeds uniformly, never whole classes.
export CALIB_CELL_ORDER="${CALIB_CELL_ORDER:-seed}"
# 150 votes: the deck's session is "a few minutes, twenty-seven questions", and
# r7's measured worth is speed (#3319) - both live well inside 150.
export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-150}"

# --- ops -----------------------------------------------------------------------
export CALIB_PARTITION=cpu
export CALIB_GRES=none
export CALIB_CPUS=1
# NOT SIZED ON THIS GRID.  The nearest measurement is #3551's caltech101_m x
# siglip binary cell at 150 votes: 0m46s, 1.4 GB.  A coco_better cell holds
# ~11,000 images to caltech's ~8,700, so expect the same order, and the retired
# rungs do no extra training.  Run `size` on r1 and r7 and read MaxRSS and
# Elapsed off `sacct` before trusting these or raising CONC.
export CALIB_MEM="${CALIB_MEM:-4G}"
export CALIB_TIME="${CALIB_TIME:-1:00:00}"
# 7 rungs x %15 = 105 tasks, 210 charged CPUs of the per-user cpu_limit QOS
# (cpu=240, 2 charged per task) and 420G of its ~1.1T memory.  Lower it if a
# peer study is running.
export CALIB_CONC="${CALIB_CONC:-15}"
# The per-rung analyzer is OFF: a rung has nothing to say alone, and a chained
# default analyzer is what overwrote #3319's study analysis.  `analyze` below
# reads every rung at once.
export CALIB_ANALYZE=noop.py

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

# Read the pre-embedded pile in place: no re-embed, no GPU, no model download.
# shellcheck disable=SC1091
source "$WT/scripts/experiments/pile/pile_env.sh"

require_jobid() {
  if ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $2 was REFUSED by sbatch (no job id came back)." >&2
    exit 1
  fi
}

set_exp() {
  export CALIB_EXP="$BASE/$1"
  export CALIB_RESULTS="$CALIB_EXP/results"
  export CALIB_JOB_NAME="prog4184-$1"
  mkdir -p "$CALIB_EXP/logs" "$CALIB_RESULTS/cells"
  ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
}

# One prepare serves every rung: the rungs differ only in threshold knobs, which
# prepare never reads.  Binary SigLIP needs no exemplar crops, so this is just
# the grid's prepare_info.json.
link_prepare() {
  local src="$BASE/prepare/results"
  [[ -f "$src/prepare_info.json" ]] || { echo "ERROR: run '$0 prepare' first" >&2; exit 1; }
  cp -n "$src/prepare_info.json" "$CALIB_RESULTS/"
  [[ -d "$src/crops" && ! -e "$CALIB_RESULTS/crops" ]] && cp -r "$src/crops" "$CALIB_RESULTS/crops"
  return 0
}

# The rung table.  Sets the three rung knobs and names the divergences
# preflight must see declared.  Anything not set here stays unset = production.
rung_env() {
  unset CALIB_LIVE_THRESHOLD CALIB_CALIBRATION_FRACTION CALIB_ACQ_INCLUSION_OFFSET
  RUNG_DIVERGES=""
  case "$1" in
    r1_xcal) export CALIB_LIVE_THRESHOLD=xcal_mincost ;;
    r2_gmm) export CALIB_LIVE_THRESHOLD=gmm_mid ;;
    r3_blend) export CALIB_LIVE_THRESHOLD=blend ;;
    r4_rawmean) export CALIB_LIVE_THRESHOLD=anchored_rawmean ;;
    r5_anchored | r6_split70 | r7_acq4) ;;
    *) echo "unknown rung '$1'; expected one of: $ALL_RUNGS" >&2; exit 2 ;;
  esac
  # 50/50 until "Train More, Check Less" moves it.  Unset from r6 on, which the
  # harness resolves to production's per-space split: 0.3 held out for SigLIP.
  case "$1" in
    r1_* | r2_* | r3_* | r4_* | r5_*) export CALIB_CALIBRATION_FRACTION=0.5 ;;
  esac
  # Acquisition aims at the reporting cut until "Second Cut".  r1-r4 ignore the
  # offset anyway (their rules have no inclusion-aware form, and the harness
  # drops the fit the offset would re-cut); pinning it there too keeps the
  # launcher honest about what every rung ran with.
  case "$1" in
    r7_acq4) ;;
    *) export CALIB_ACQ_INCLUSION_OFFSET=0 ;;
  esac
  [[ -n "${CALIB_LIVE_THRESHOLD:-}" ]] && RUNG_DIVERGES="$RUNG_DIVERGES,live_threshold"
  [[ -n "${CALIB_CALIBRATION_FRACTION:-}" ]] && RUNG_DIVERGES="$RUNG_DIVERGES,calibration_fraction"
  [[ -n "${CALIB_ACQ_INCLUSION_OFFSET:-}" ]] && RUNG_DIVERGES="$RUNG_DIVERGES,acq_offset"
  RUNG_DIVERGES="${RUNG_DIVERGES#,}"
  return 0
}

run_preflight() {
  [[ -x "$WT/scripts/experiments/preflight.sh" ]] || return 0
  local div=()
  [[ -n "$RUNG_DIVERGES" ]] && div=(--diverges "$RUNG_DIVERGES")
  bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --need-gb 10 \
    "${div[@]}" --job-name "$CALIB_JOB_NAME" --mem "$CALIB_MEM" --conc "$CALIB_CONC" || {
    echo "preflight FAILED ($CALIB_JOB_NAME)" >&2
    [[ "${PREFLIGHT_SKIP:-0}" == "1" ]] || exit 1
  }
}

grid_size() {
  (cd "$HERE" && source "$WT/gridenv.sh" >/dev/null 2>&1 && eval "$ENVX" && python run_cells.py --print-cells 2>/dev/null | tail -1)
}

case "$MODE" in
  prepare)
    set_exp prepare
    P=$(sbatch --parsable --job-name=prog4184-prep --mem=32G --cpus-per-task=2 \
      --time=1:30:00 --partition=cpu --export=ALL \
      --output="$CALIB_EXP/logs/prepare-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python prepare_data.py")
    require_jobid "$P" "prepare"
    echo "prepare job: $P -> $CALIB_EXP/logs/prepare-$P.out"
    ;;

  size)
    IDX="${2:-0}"
    RUNG="${3:-r7_acq4}"
    rung_env "$RUNG"
    set_exp "sizing-$RUNG"
    link_prepare
    S=$(sbatch --parsable --job-name="prog4184-size-$RUNG-$IDX" --mem="$CALIB_MEM" --cpus-per-task=1 \
      --time="$CALIB_TIME" --partition=cpu --export=ALL \
      --output="$CALIB_EXP/logs/size-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && /usr/bin/time -v python run_cells.py --index $IDX --outdir $CALIB_RESULTS/cells")
    require_jobid "$S" "size"
    echo "size job: $S (cell $IDX, $RUNG) -> $CALIB_EXP/logs/size-$S.out"
    echo "then: sacct -j $S --format=Elapsed,MaxRSS,State"
    ;;

  baseline)
    # The click-0 anchor: the typed query's own ranking, before any vote.  No
    # detector exists at click 0, so it is rung-independent - one file serves
    # all seven, and every curve starts from the same point by construction.
    set_exp prepare
    T=$(sbatch --parsable --job-name=prog4184-baseline --mem=32G --cpus-per-task=2 \
      --time=1:00:00 --partition=cpu --export=ALL \
      --output="$CALIB_EXP/logs/baseline-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python text_baseline.py --results $CALIB_RESULTS --out $BASE/text_baseline.csv")
    require_jobid "$T" "text baseline"
    echo "baseline job: $T -> $BASE/text_baseline.csv"
    ;;

  rungs)
    shift
    RUNGS="${*:-$ALL_RUNGS}"
    for rung in $RUNGS; do
      (
        rung_env "$rung"
        set_exp "$rung"
        link_prepare
        echo "=== $rung: live_threshold=${CALIB_LIVE_THRESHOLD:-shipped}" \
          "calibration_fraction=${CALIB_CALIBRATION_FRACTION:-production}" \
          "acq_offset=${CALIB_ACQ_INCLUSION_OFFSET:-production} -> $CALIB_EXP"
        run_preflight
        bash "$HERE/launch_cells.sh"
      ) || exit 1
    done
    ;;

  plan)
    # Submits nothing: what each rung will run with, and what preflight must
    # see declared.  Read this before `rungs`.
    for rung in $ALL_RUNGS; do
      rung_env "$rung"
      printf '%-12s live_threshold=%-17s calibration_fraction=%-11s acq_offset=%-11s diverges=%s\n' \
        "$rung" "${CALIB_LIVE_THRESHOLD:-shipped}" "${CALIB_CALIBRATION_FRACTION:-production}" \
        "${CALIB_ACQ_INCLUSION_OFFSET:-production}" "${RUNG_DIVERGES:-none}"
    done
    ;;

  status)
    for rung in $ALL_RUNGS; do
      rung_env "$rung"
      set_exp "$rung"
      n="$(find "$CALIB_RESULTS/cells" -maxdepth 1 -name 'task_[0-9][0-9][0-9][0-9].csv' -size +0 2>/dev/null | wc -l)"
      z="$(find "$CALIB_RESULTS/cells" -maxdepth 1 -name 'task_[0-9][0-9][0-9][0-9].csv' -size 0 2>/dev/null | wc -l)"
      q="$(squeue -u "$USER" -h -n "$CALIB_JOB_NAME" -o %i 2>/dev/null | wc -l)"
      printf '%-12s %5s cells written  %3s zero-byte  %3s queued/running jobs\n' "$rung" "$n" "$z" "$q"
    done
    echo "expected per rung: 144 cells x $CALIB_N_SEEDS seeds = $((144 * CALIB_N_SEEDS)) (confirm with run_cells.py --print-cells)"
    ;;

  analyze)
    ARMS=""
    for rung in $ALL_RUNGS; do ARMS="$ARMS,$BASE/$rung/results=$rung"; done
    set_exp analysis
    A=$(sbatch --parsable --job-name=prog4184-analyze --mem="${CALIB_ANALYZE_MEM:-32G}" \
      --cpus-per-task=4 --time="${CALIB_ANALYZE_TIME:-2:00:00}" --partition=cpu --export=ALL \
      --output="$CALIB_EXP/logs/analyze-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python analyze_progression_4184.py --arms ${ARMS#,} --baseline $BASE/text_baseline.csv --out $CALIB_EXP")
    require_jobid "$A" "analyze"
    echo "analyze job: $A -> $CALIB_EXP"
    ;;

  *)
    echo "usage: $0 {prepare|size [idx] [RUNG]|plan|baseline|rungs [RUNG..]|status|analyze}" >&2
    exit 2
    ;;
esac
