#!/usr/bin/env bash
# Is `transfer` a bias or a variance? (#2883) on the HLTCOE Grid.
#
# The #2836 chain's last link dominates the corrected (#3187) decomposition --
# +0.037 of a +0.056 total on the production arm's ramp, 67 %.  #2883 asks what
# it scales with before anyone proposes a remedy.  This run answers that with
# nine new variant rows per step, all of which are **re-cuts of the same per-step
# model against the same test scores**: four subsample levels of the sim set (the
# learning curve), two variance-reduced readings of the same target, two
# label-free bagged-fit arms, and a cross-fitted reference point recorded beside
# the sample-minimum one the decomposition uses today.
#
# So it costs what `launch_tail_2881.sh` cost plus ~20 % of a cell (measured, on
# a compute node: `decomposition_cuts` goes 0.20s -> 0.52s per step-geometry).
# No GPU, no new embeddings, no extra training.
#
# **Its own CALIB_EXP.** The grid differs from #3130's -- nine new variants and
# six new diagnostic columns per step -- which makes it a different study, and
# sharing a results dir is how two grids get analysed as one.
#
# Pre-registration -- read it before reading the output:
#   docs/experiments/transfer-2883/PREREG.md
#
# The theory bench is deliberately not launched: #2883 item 3 is explicit that it
# has mis-sized this family twice at full power and any bench-only result here is
# a hypothesis, not evidence.
#
# Usage: bash launch_transfer_2883.sh
set -uo pipefail

WT="${VTS_REPO:-/exp/$USER/projects/vts-transfer-2883}"
HERE="$WT/scripts/experiments/calibration"

export CALIB_EXP="${CALIB_EXP:-/exp/$USER/transfer-2883/run}"
export CALIB_RESULTS="${CALIB_RESULTS:-$CALIB_EXP/results}"

export CALIB_SAFE_THRESHOLDS=1
export CALIB_ANALYZE=analyze_cut.py

# The head a live detector actually has.  NOT `linear`: `launch_cut.sh` and
# `launch_tail_2881.sh` both pin that, which was production until #3198 moved
# PRODUCTION_HEAD to the linear SVM.  Preflight check 12 fails on a stale pin --
# leaving it unset would work too, but naming it is what makes the run's head
# readable from the launcher instead of from a default three modules away.
export CALIB_HEAD="${CALIB_HEAD:-linear_svm}"

# Unchanged from #3130 so the contrast is direct: this study is measuring the
# same term, on the same arms, that the corrected decomposition reports.
# The `whole_image` control is the other half of every claim here -- it has no
# max-pool, so if the transfer story is really about a max-pooled Bad tail rather
# than about sample size, the two arms have to disagree.
export CALIB_DATASETS="${CALIB_DATASETS:-visual_genome_m}"
# The region arm is the PAIR `siglip+dinov3_patch` (#3278).  Every claim here is
# "the two arms have to disagree" -- if transfer is about a max-pooled Bad tail
# rather than about sample size, the whole_image control must not show it -- and
# a disagreement is only evidence when the arms differ in one thing.  Bare
# `dinov3_patch` has no text tower, so it would also differ in how it started,
# and #3267 puts the largest measured effects in exactly that.
export CALIB_VG_EMBEDDERS="${CALIB_VG_EMBEDDERS:-siglip,siglip+dinov3_patch}"
export CALIB_REQUIRE_OPENING=text
# A paired arm cannot fall back to the known-good start (`run_cells.py` raises),
# so every selected category must have a typed query.  Selection filters to the
# eligible ones before it picks, replacing rather than dropping.
export CALIB_REQUIRE_SEED_QUERY=1
export CALIB_PATCH_STYLES="${CALIB_PATCH_STYLES:-max_patch}"
export CALIB_REPOOL_VARIANTS="${CALIB_REPOOL_VARIANTS:-}"
export CALIB_SWEEP_KS="${CALIB_SWEEP_KS:-0}"

export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-30}"
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-12}"

export CALIB_PARTITION="${CALIB_PARTITION:-cpu}"
export CALIB_GRES="${CALIB_GRES:-none}"
export CALIB_MEM="${CALIB_MEM:-12G}"
export CALIB_CPUS="${CALIB_CPUS:-4}"
export CALIB_CONC="${CALIB_CONC:-40}"
export CALIB_TIME="${CALIB_TIME:-2:00:00}"

# One thread per cell.  The bagged-fit arm runs 16 EM fits per step-geometry and
# sklearn's default threading makes each of them *slower*, not faster, at this
# size -- 0.58s vs 0.016s for the same fit when measured on a loaded login node
# against a compute node with BLAS pinned to one thread.  40 concurrent cells
# each spawning a thread pool is also how a shared node gets oversubscribed.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

# The pile (#3121): the durable home for embeddings, and the only reason this
# study needs no GPU stage at all.
source "$WT/scripts/experiments/pile/pile_env.sh"
export HF_HOME="${HF_HOME:-$VTSEARCH_MODELS_DIR}"
export CUDA_VISIBLE_DEVICES=""

LOGS="$CALIB_EXP/logs"
mkdir -p "$LOGS"

# #3130's restaged prepare: same datasets, same embedders, same 23 categories,
# same exemplar crops.  Prepare selects categories and cuts exemplar crops --
# neither depends on the head -- so switching to `linear_svm` does not invalidate
# it, and reusing it skips a GPU stage that has nothing to do.
REUSE="${TRANSFER_REUSE_PREPARE:-/exp/$USER/tail2881-prepare/results}"

# The gate, not a reminder.  `--reuse-prepare` is the one that matters here: the
# crops are symlinks into whichever study generated them, `readlink -f` will
# happily recreate a dangling link, and a study that is archived between runs
# leaves exactly that behind (#2881).
PREFLIGHT_ARGS=(--exp "$CALIB_EXP" --arms siglip,siglip+dinov3_patch
                --job-name cal-cells --mem "$CALIB_MEM" --conc "$CALIB_CONC")
[[ -d "$REUSE" ]] && PREFLIGHT_ARGS+=(--reuse-prepare "$REUSE")
bash "$WT/scripts/experiments/preflight.sh" "${PREFLIGHT_ARGS[@]}" || exit 1

if [[ ! -f "$REUSE/prepare_info.json" ]]; then
  echo "ERROR: no reusable prepare at $REUSE" >&2
  echo "       This study deliberately has no GPU stage; restage prepare or point" >&2
  echo "       TRANSFER_REUSE_PREPARE at a finished run with the same arms." >&2
  exit 1
fi

mkdir -p "$CALIB_RESULTS/cells" "$CALIB_RESULTS/crops"
cp "$REUSE/prepare_info.json" "$CALIB_RESULTS/prepare_info.json"
for f in "$REUSE"/crops/*; do
  dst="$CALIB_RESULTS/crops/$(basename "$f")"
  # cp -P would copy the symlink itself, whose relative target would not resolve
  # from here; resolve it to the real file instead.
  [[ -e "$dst" ]] || ln -s "$(readlink -f "$f")" "$dst"
done
echo "reused prepare from $REUSE"

# Not `exec`: this study needs a second analysis step chained after the first,
# and the cells launcher's own analyze job is the thing it has to depend on.
CELLS_OUT="$(bash "$HERE/launch_cells.sh")"
echo "$CELLS_OUT"
CUT_JOB="$(sed -n 's/^analyze: \([0-9][0-9]*\)$/\1/p' <<<"$CELLS_OUT" | tail -1)"
if [[ -z "$CUT_JOB" ]]; then
  echo "ERROR: could not read the analyze job id from launch_cells.sh; not chaining" >&2
  echo "       -> run \`python analyze_transfer.py\` by hand once the cells finish." >&2
  exit 1
fi

# `afterany`, matching the cut analyzer's own dependency: a run with some failed
# cells still has a decomposition worth reading, and the analyzers report what
# they dropped rather than refusing to run.
X=$(sbatch --parsable --dependency=afterany:"$CUT_JOB" --job-name=cal-transfer \
  --mem="${CALIB_ANALYZE_MEM:-24G}" --cpus-per-task=4 --time="${CALIB_ANALYZE_TIME:-0:40:00}" \
  --partition=cpu --export=ALL --output="$LOGS/transfer-%j.out" \
  --wrap="source $WT/gridenv.sh && export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME && cd $HERE && python analyze_transfer.py")
echo "transfer analyze: $X (after $CUT_JOB)"
echo "Pre-registration -> $WT/docs/experiments/transfer-2883/PREREG.md"
echo "Report           -> $CALIB_RESULTS/REPORT_TRANSFER.md"
