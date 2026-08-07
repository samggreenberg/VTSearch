#!/usr/bin/env bash
# One-constant EVT tail-alpha cut (#2881) on the HLTCOE Grid.
#
# The tail rules ride along inside the ordinary cut-study cells array: they read
# the EVT fit that every step already computes, and their cuts are closed form,
# so this run costs the same as `launch_cut.sh` and produces seven more variant
# rows per step.  Same arms, sizing and seeds as #2836/#2846 -- the contrasts are
# paired within a step, so reusing the sizing keeps them directly comparable to
# the numbers those two reports produced.
#
# **Its own CALIB_EXP.** The grid differs from #2846's (seven new variants per
# step), which makes it a different study; sharing a results dir is how two grids
# get analysed as one.
#
# Pre-registration -- read it before reading the output:
#   docs/experiments/gmm-cut/PREREG-2881.md
#
# The theory bench is deliberately not launched.  Its CANDIDATE_RULES list does
# not carry the tail family, and #2846's recommendation was to stop trusting the
# bench on this family after it mis-sized it twice in the same direction.  Set
# TAIL_WITH_THEORY=1 to launch it anyway (it will measure the crossing rules).
#
# Usage: bash launch_tail_2881.sh
set -uo pipefail

WT="${VTS_REPO:-/exp/$USER/projects/vts-calib}"
HERE="$WT/scripts/experiments/calibration"

export CALIB_EXP="${CALIB_EXP:-/exp/$USER/calibration-tail2881}"
export CALIB_RESULTS="${CALIB_RESULTS:-$CALIB_EXP/results}"

export CALIB_SAFE_THRESHOLDS=1
export CALIB_ANALYZE=analyze_cut.py
export CALIB_HEAD="${CALIB_HEAD:-linear}"

# Visual Genome region voting; production patch arm + the single-vector control.
# The control is the ship gate's other half: `calculate_gmm_threshold` also backs
# the cosine/text sort, which has no max-pool and so no extreme-value tail at all.
export CALIB_DATASETS="${CALIB_DATASETS:-visual_genome_m}"
export CALIB_VG_EMBEDDERS="${CALIB_VG_EMBEDDERS:-siglip,dinov3_patch}"
export CALIB_PATCH_STYLES="${CALIB_PATCH_STYLES:-max_patch}"
export CALIB_REPOOL_VARIANTS="${CALIB_REPOOL_VARIANTS:-}"
export CALIB_SWEEP_KS="${CALIB_SWEEP_KS:-0}"

export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-30}"
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-12}"

export CALIB_PARTITION="${CALIB_PARTITION:-cpu}"
export CALIB_GRES="${CALIB_GRES:-none}"
export CALIB_MEM="${CALIB_MEM:-24G}"
export CALIB_CPUS="${CALIB_CPUS:-4}"
export CALIB_CONC="${CALIB_CONC:-40}"
export CALIB_TIME="${CALIB_TIME:-1:30:00}"

MAXPATCH="/exp/$USER/max-patch"
export VTSEARCH_DATA_DIR="${VTSEARCH_DATA_DIR:-$MAXPATCH/datadir}"
export VTSEARCH_MODELS_DIR="${VTSEARCH_MODELS_DIR:-$MAXPATCH/models}"
export HF_HOME="${HF_HOME:-/exp/$USER/.cache/huggingface}"

LOGS="$CALIB_EXP/logs"
mkdir -p "$LOGS"

# The gate, not a reminder.  It also resolves `import vtscore` the way a job does
# and fails if it lands outside VTS_REPO -- which is exactly how #2846 ran a
# fresh worktree against the shared checkout with a correct PYTHONPATH.  A branch
# that only *changes* behaviour (this one adds seven rules, so it would in fact
# have been caught by the missing-symbol crash, but the next one may not) would
# otherwise produce a clean, plausible, wrong table.
bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --arms siglip,dinov3_patch || exit 1

if [[ -n "${TAIL_WITH_THEORY:-}" ]]; then
  T=$(sbatch --parsable --job-name=tail-theory --mem=8G --cpus-per-task=4 \
    --time="${TAIL_THEORY_TIME:-3:00:00}" --partition=cpu --export=ALL --output="$LOGS/theory-%j.out" \
    --wrap="source $WT/gridenv.sh && export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS && cd $HERE && python theory_bench.py --reps ${TAIL_THEORY_REPS:-40}")
  echo "theory bench job: $T   -> $CALIB_RESULTS/theory/"
fi

# The arms are identical to #2799's, so its prepare output (category counts in
# prepare_info.json, exemplar-crop symlinks into the Max-Patch results) is
# reusable verbatim -- there is no new embedding to do, and seeding from it skips
# a GPU stage that would otherwise queue behind the 4-GPU QOS cap for nothing.
REUSE="${TAIL_REUSE_PREPARE:-/exp/$USER/calibration-safe-linear/results}"
if [[ -f "$REUSE/prepare_info.json" ]]; then
  mkdir -p "$CALIB_RESULTS/cells" "$CALIB_RESULTS/crops"
  cp "$REUSE/prepare_info.json" "$CALIB_RESULTS/prepare_info.json"
  for f in "$REUSE"/crops/*; do
    dst="$CALIB_RESULTS/crops/$(basename "$f")"
    # cp -P would copy the symlink itself, whose relative target would not
    # resolve from here; resolve it to the real file instead.
    [[ -e "$dst" ]] || ln -s "$(readlink -f "$f")" "$dst"
  done
  echo "reused prepare from $REUSE"
  exec bash "$HERE/launch_cells.sh"
fi

echo "no reusable prepare at $REUSE; running the full chain"
exec bash "$HERE/launch_all.sh"
