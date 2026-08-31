#!/usr/bin/env bash
# Build DocMarks at full scale on the GRID (#3343).
#
#   bash launch_docmarks.sh probe          # can I reach every source?
#   bash launch_docmarks.sh build          # stage 1: sources + clustering (CPU)
#   bash launch_docmarks.sh status         # queue + the real signal on disk
#   bash launch_docmarks.sh embed s        # stage 5: cells for one tier (GPU)
#
# WHY ONE LONG CPU JOB.  The binding cost is not compute, it is wall-clock
# against a shared public API: UCSF fetch+render measured 0.75 pages/s, so
# 200k distractors is ~74 h no matter how much hardware is pointed at it.
# Parallelising the pull across jobs would only be rude to UCSF and get the
# job rate-limited.  Resume is free (atomic downloads, rendered pages skipped
# when present, stable Solr cursor order), so a killed job restarts where it
# stopped -- which is what makes a multi-day reservation safe.
#
# SOURCES default to spods,ucsf, NOT the full four.  StaVer and Tobacco800 are
# Kaggle-hosted and blocked without a token; the UCSF pull is the 74 h pole and
# is entirely independent of them.  Start the pole now, add the anchors later:
# the sources are cached, so a rebuild with all four re-runs only clustering
# and manifest writing.  Set VTS_DOCMARKS_SOURCES once the token is in place.
#
# RAR EXTRACTOR.  SPODS is RAR4 and no compute node here ships bsdtar, 7z,
# unar or unrar.  A static 7-Zip 25.01 (x64) is installed at ~/.local/bin/7zz
# with a `7z` symlink beside it -- that directory is prepended to PATH below,
# because a login shell has it but an sbatch job does not.
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- NOTHING WAS SUBMITTED" >&2' ERR

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"

source "$WT/gridenv.sh"
source "$WT/scripts/experiments/pile/pile_env.sh"
export VTS_REPO="$WT"
export PATH="$HOME/.local/bin:$PATH"

export VTS_DOCMARKS_RAW="${VTS_DOCMARKS_RAW:-/expscratch/$USER/docmarks/raw}"
export VTS_DOCMARKS_OUT="${VTS_DOCMARKS_OUT:-/expscratch/$USER/docmarks/corpus}"

SOURCES="${VTS_DOCMARKS_SOURCES:-spods,ucsf}"
DISTRACTORS="${VTS_DOCMARKS_DISTRACTORS:-200000}"
LETTERHEAD="${VTS_DOCMARKS_LETTERHEAD:-2000}"
ROSTER_ARG=""
[ -n "${VTS_DOCMARKS_ROSTER:-}" ] && ROSTER_ARG="--roster ${VTS_DOCMARKS_ROSTER}"

# The pull is one stream; the CPUs are for rendering PDFs to 150 dpi PNGs
# behind it.  MEM is the runbook number -- clustering holds every mark at once.
MEM="${VTS_DOCMARKS_MEM:-16G}"
CPUS="${VTS_DOCMARKS_CPUS:-8}"
TIME="${VTS_DOCMARKS_TIME:-4-00:00:00}"
PARTITION="${VTS_DOCMARKS_PARTITION:-cpu}"
JOB_NAME="${VTS_DOCMARKS_JOB_NAME:-docmarks-build}"

LOGS="$VTS_DOCMARKS_OUT/logs"
mkdir -p "$LOGS" "$VTS_DOCMARKS_RAW" "$VTS_DOCMARKS_OUT"

cmd="${1:-status}"

case "$cmd" in
probe)
  cd "$HERE" && exec python build_corpus.py --probe
  ;;

build)
  bash "$WT/scripts/experiments/preflight.sh" --exp "$VTS_DOCMARKS_OUT" \
    --job-name "$JOB_NAME" --mem "$MEM" --conc 1 || exit 1

  RUNNER="/exp/$USER/.docmarks-build.$$.sh"
  cat > "$RUNNER" <<RUNNER_EOF
#!/usr/bin/env bash
set -uo pipefail
source "$WT/gridenv.sh"
source "$WT/scripts/experiments/pile/pile_env.sh"
export VTS_REPO="$WT"
export PATH="\$HOME/.local/bin:\$PATH"
export VTS_DOCMARKS_RAW="$VTS_DOCMARKS_RAW"
export VTS_DOCMARKS_OUT="$VTS_DOCMARKS_OUT"
export CUDA_VISIBLE_DEVICES=
cd "$HERE"
echo "node=\$(hostname) job=\$SLURM_JOB_ID start=\$(date -Is)"
echo "sources=$SOURCES distractors=$DISTRACTORS letterhead=$LETTERHEAD"
python -u build_corpus.py \\
  --sources "$SOURCES" \\
  --ucsf-distractors $DISTRACTORS \\
  --ucsf-letterhead-per-author $LETTERHEAD \\
  $ROSTER_ARG
echo "exit=\$? end=\$(date -Is)"
RUNNER_EOF
  chmod +x "$RUNNER"

  sbatch --job-name="$JOB_NAME" --partition="$PARTITION" \
    --mem="$MEM" --cpus-per-task="$CPUS" --time="$TIME" \
    --output="$LOGS/build-%j.out" --error="$LOGS/build-%j.out" \
    --wrap="bash $RUNNER"
  ;;

status)
  squeue -u "$USER" -n "$JOB_NAME" -o "%.10i %.16j %.8T %.12M %.12l %R"
  echo
  echo "raw:    $(du -sh "$VTS_DOCMARKS_RAW" 2>/dev/null | cut -f1)"
  echo "out:    $(du -sh "$VTS_DOCMARKS_OUT" 2>/dev/null | cut -f1)"
  echo "pages:  $(find "$VTS_DOCMARKS_RAW" -name "*.png" 2>/dev/null | wc -l) rendered"
  echo "free:   $(df -h "$VTS_DOCMARKS_OUT" | tail -1 | awk "{print \$4}")"
  tail -5 "$(ls -t "$LOGS"/build-*.out 2>/dev/null | head -1)" 2>/dev/null
  ;;

embed)
  tier="${2:-s}"
  GRES="gpu:$(python3 "$WT/scripts/slurm/pick_gpu.py"):1"
  RUNNER="/exp/$USER/.docmarks-embed-$tier.$$.sh"
  cat > "$RUNNER" <<RUNNER_EOF
#!/usr/bin/env bash
set -uo pipefail
source "$WT/gridenv.sh"
source "$WT/scripts/experiments/pile/pile_env.sh"
export VTS_REPO="$WT"
export VTS_DOCMARKS_OUT="$VTS_DOCMARKS_OUT"
cd "$HERE"
echo "node=\$(hostname) job=\$SLURM_JOB_ID tier=$tier start=\$(date -Is)"
python -u embed_corpus.py --tier $tier --embedders "${VTS_DOCMARKS_EMBEDDERS:-sift_vlad,siglip}"
RUNNER_EOF
  chmod +x "$RUNNER"
  sbatch --job-name="docmarks-embed-$tier" --partition=gpu --gres="$GRES" \
    --mem="${VTS_DOCMARKS_EMBED_MEM:-32G}" --cpus-per-task=4 \
    --time="${VTS_DOCMARKS_EMBED_TIME:-12:00:00}" \
    --output="$LOGS/embed-$tier-%j.out" --error="$LOGS/embed-$tier-%j.out" \
    --wrap="bash $RUNNER"
  ;;

*)
  echo "usage: $0 {probe|build|status|embed <tier>}" >&2; exit 2 ;;
esac
