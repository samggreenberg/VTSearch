#!/usr/bin/env bash
# #3292: does #3287's `calibration_fraction` optimum survive a change of FAMILY?
#
# Runs the #3287 grid on two CLIP checkpoints, unattended, as one SLURM chain:
#
#   bash run_3292.sh chain      # submit everything; nothing else needs a terminal
#   bash run_3292.sh status     # where the chain is, from files on disk
#
# WHY A CHAIN AND NOT A TERMINAL.  Every watcher this project has run from a
# laptop has eventually died with the VPN (`lessons/`, and the grid-experiments
# skill says it plainly: make completion a question about a FILE, not about a
# live process).  So the pile build, both prepares, both baselines and both
# five-arm grids are submitted as SLURM jobs chained on `afterok`, and the only
# thing a human has to do afterwards is read `status`.
#
# WHY TWO CHECKPOINTS.  The question is whether 0.3 follows *single-vector
# geometry* or just *the SigLIP lineage*, and one CLIP arm cannot answer it:
# leaving SigLIP changes the family AND the capacity at the same time, which is
# the identical confound #3115 and the pile's own `siglip -> siglip2_l` note
# warn about.  Two capacities of one lineage separate them:
#
#   clip    ViT-B/32, 512-d   the checkpoint the app already ships
#   clip_l  ViT-L/14, 768-d   dimension-matched to `siglip`
#
# Agreement between them reads as CLIP's family.  Disagreement reads as
# capacity, and would mean #3290's constant keys on neither "single-vector" nor
# "family" -- a more useful answer than either, and only visible with both arms.
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- check what was SUBMITTED" >&2' ERR

MODE="${1:-status}"

export VTS_REPO="${VTS_REPO:-/exp/$USER/projects/vts-clip-3292}"
WT="$VTS_REPO"
HERE="$WT/scripts/experiments/calibration"
PILE_DIR="$WT/scripts/experiments/pile"

#: The two arms, in the order they run.  `clip` first: it is the cheaper
#: encoder, so a mistake in the shared plumbing surfaces sooner.
ARMS="${CLIP3292_ARMS:-clip clip_l}"
ROOT="${CLIP3292_ROOT:-/expscratch/$USER}"
LOGS="$ROOT/calfrac-3292-chain"
mkdir -p "$LOGS"

base_for() { echo "$ROOT/calfrac-3287-$1"; }

# MEMORY.  #3289 measured 0.94 GB for a `siglip` binary cell and the launcher's
# single-vector default is 4G on the back of it.  That default does not transfer
# here: `run_cells.py` calls `embed_text_query` per cell, so the encoder's
# weights sit in every cell's RSS, and CLIP ViT-L/14 is ~428M params against
# SigLIP-base's ~200M.  8G keeps the same study footprint by halving the
# per-arm concurrency (12 x 5 arms x 8G = 480G, identical to 24 x 5 x 4G), so
# this buys headroom for free rather than trading throughput for it.  An OOM
# here is a LOST cell, not a slow one.
export CALIB_MEM="${CALIB_MEM:-8G}"
export CALIB_CONC="${CALIB_CONC:-12}"

require_jobid() {
  local id="$1" what="$2"
  if ! [[ "$id" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $what was REFUSED by sbatch (no job id came back)." >&2
    exit 1
  fi
}

case "$MODE" in
  chain)
    # --- stage 1: the pile cells (GPU) ----------------------------------
    # Builds `vg_scale`, not `vg_scale_any`.  The study reads the derived cell,
    # but a derived cell is a RELABEL of its parent pickle and refuses to build
    # without one -- and no `vg_scale__*` parent is on disk for any embedder, so
    # naming the derived dataset alone would just fail.  `build_pile.py` pulls
    # the derived cell into any run that carries its parent, which is exactly
    # the coupling #3281 shipped a stale cell for want of.
    # Smoke-tests both encoders BEFORE embedding 4200 images with them.  The
    # thing that can silently go wrong is the TEXT tower: with no vector the
    # study falls back to the known-good opening and stops being comparable to
    # the SigLIP runs at all (#3278), and nothing downstream raises.
    P=$(sbatch --parsable --job-name=clip3292-pile \
      --partition=gpu --gres=gpu:v100:1 --mem=32G --cpus-per-task=4 --time=3:00:00 \
      --export=ALL --output="$LOGS/pile-%j.out" \
      --wrap="source $WT/gridenv.sh && cd $PILE_DIR && \
        python smoke_clip_3292.py && \
        python build_pile.py --embedders clip,clip_l --datasets vg_scale && \
        python build_pile.py --verify --embedders clip,clip_l --datasets vg_scale,vg_scale_any")
    require_jobid "$P" "the pile build"
    echo "pile build: $P  ->  $LOGS/pile-$P.out"

    # --- stage 2: the two studies (CPU driver, afterok) -----------------
    # `afterok`, not `afterany`: a failed pile build means the cells this study
    # reads do not exist, and a grid launched onto missing cells produces 240
    # failures rather than an answer.
    D=$(sbatch --parsable --dependency="afterok:$P" --job-name=clip3292-driver \
      --partition=cpu --mem=4G --cpus-per-task=1 --time=12:00:00 \
      --export=ALL --output="$LOGS/driver-%j.out" \
      --wrap="bash $HERE/run_3292.sh drive")
    require_jobid "$D" "the study driver"
    echo "study driver: $D (afterok:$P)  ->  $LOGS/driver-$D.out"
    echo
    echo "Nothing else needs a terminal.  Read progress with:"
    echo "  bash $HERE/run_3292.sh status"
    ;;

  drive)
    # Runs INSIDE a batch job.  Serial on purpose: the two studies would each
    # claim 480G of a 1074G allowance, and two at once would also blow the CPU
    # cap (240 cpus, 2 charged per task).  Serially they are ~12 min each.
    # shellcheck disable=SC1091
    source "$WT/gridenv.sh"
    for EMB in $ARMS; do
      BASE="$(base_for "$EMB")"
      echo "=== $EMB -> $BASE ==="
      export CALFRAC_BASE="$BASE"
      export CALIB_VGSCALE_EMBEDDERS="$EMB"

      bash "$HERE/launch_calfrac_3287.sh" prepare | tee "$LOGS/$EMB-prepare.txt"
      # Wait on the FILE the next stage actually needs, not on the queue: a
      # drained queue with no prepare_info.json is a failure, and only the file
      # can tell the two apart.
      for _ in $(seq 1 360); do
        [[ -f "$BASE/prepare/results/prepare_info.json" ]] && break
        sleep 20
      done
      if [[ ! -f "$BASE/prepare/results/prepare_info.json" ]]; then
        echo "FAILED: $EMB prepare produced no prepare_info.json in 2h" >&2
        continue
      fi

      bash "$HERE/launch_calfrac_3287.sh" baseline | tee "$LOGS/$EMB-baseline.txt"
      bash "$HERE/launch_calfrac_3287.sh" arms     | tee "$LOGS/$EMB-arms.txt"

      # Drain before starting the next arm, so the two studies never overlap.
      for _ in $(seq 1 720); do
        n=$(squeue -u "$USER" -h -o %j | grep -c "^cal-cells-f" || true)
        [[ "$n" -eq 0 ]] && break
        sleep 60
      done
      echo "=== $EMB cells drained: $(find "$BASE" -path "*/results/cells/*.pkl" | wc -l) cell files ==="
    done
    echo "=== driver done ==="
    ;;

  status)
    for EMB in $ARMS; do
      BASE="$(base_for "$EMB")"
      printf '%-8s ' "$EMB"
      if [[ ! -d "$BASE" ]]; then echo "not started"; continue; fi
      n=$(find "$BASE" -path "*/results/cells/*" -name "*.pkl" 2>/dev/null | wc -l)
      rep="$BASE/analysis/REPORT_calfrac.md"
      printf 'cells=%s/240  report=%s\n' "$n" "$([[ -f $rep ]] && echo yes || echo no)"
    done
    echo "--- queue ---"
    squeue -u "$USER" -o '%.10i %.18j %.8T %.10M %.6D %R' 2>/dev/null | head -20
    ;;

  *)
    echo "usage: $0 {chain|drive|status}" >&2
    exit 2
    ;;
esac
