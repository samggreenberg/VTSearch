#!/usr/bin/env bash
# Build FullMarks at full scale on the GRID (#3343).
#
#   bash launch_fullmarks.sh probe          # can I reach every source?
#   bash launch_fullmarks.sh build          # stage 1: sources + clustering (CPU)
#   bash launch_fullmarks.sh slate          # stage 2: the human audit bundle (CPU)
#   bash launch_fullmarks.sh siglip         # stage 2b: the audit's second opinion (GPU)
#   bash launch_fullmarks.sh status         # queue + the real signal on disk
#   bash launch_fullmarks.sh embed s        # stage 5: cells for one tier (GPU)
#
# WHY ONE LONG CPU JOB.  The binding cost is not compute, it is wall-clock
# against a shared public API: UCSF fetch+render measured 0.75 pages/s, so
# 200k distractors is ~74 h no matter how much hardware is pointed at it.
# Parallelising the pull across jobs would only be rude to UCSF and get the
# job rate-limited.  Resume is free (atomic downloads, rendered pages skipped
# when present, stable Solr cursor order), so a killed job restarts where it
# stopped -- which is what makes a multi-day reservation safe.
#
# SOURCES is all four.  StaVer and Tobacco800 were Kaggle-blocked until #3343
# taught the credential gate about ~/.kaggle/access_token (the file "Create New
# Token" actually writes); with that, all four probe OK.  If a token ever goes
# missing again, VTS_FULLMARKS_SOURCES=spods,ucsf still runs the 74 h pole -- the
# anchors fold in later off cached sources for the price of a re-cluster.
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

export VTS_FULLMARKS_RAW="${VTS_FULLMARKS_RAW:-/expscratch/$USER/fullmarks/raw}"
export VTS_FULLMARKS_OUT="${VTS_FULLMARKS_OUT:-/expscratch/$USER/fullmarks/corpus}"

SOURCES="${VTS_FULLMARKS_SOURCES:-spods,staver,tobacco800,ucsf}"
DISTRACTORS="${VTS_FULLMARKS_DISTRACTORS:-200000}"
LETTERHEAD="${VTS_FULLMARKS_LETTERHEAD:-2000}"
ROSTER_ARG=""
[ -n "${VTS_FULLMARKS_ROSTER:-}" ] && ROSTER_ARG="--roster ${VTS_FULLMARKS_ROSTER}"
# Building into a corpus that already exists must say which tier promise it
# makes (#3903): VTS_FULLMARKS_PIN_TIERS=<build_report.json> keeps the tiers,
# VTS_FULLMARKS_NEW_VERSION=1 declares a new corpus version.
TIER_ARG=""
[ -n "${VTS_FULLMARKS_PIN_TIERS:-}" ] && TIER_ARG="--pin-tiers ${VTS_FULLMARKS_PIN_TIERS}"
[ "${VTS_FULLMARKS_NEW_VERSION:-0}" = "1" ] && TIER_ARG="$TIER_ARG --new-version"

# The pull is one stream; the CPUs are for rendering PDFs behind it.
#
# MEM is 64G, not the runbook's 16G, and the number is measured rather than
# inherited.  The smoke build (job 602791: 1088 SPODS + 453 UCSF pages, 160
# letterhead candidates) peaked at MaxRSS 6.5 GB -- already 41% of 16G on a
# corpus 1/130th the size.  Clustering holds every mark in memory at once, and
# the mark count is driven by the letterhead candidates, which go from 160 to
# 8 authors x 2000 = 16,000 in the real build.  16G would be a coin flip three
# days into an unattended job; on the `cpu` partition, where nodes sit idle and
# the per-user cap is ~1 TB, the usual "an over-fat --mem wedges you off idle
# GPUs" tradeoff simply does not apply.  Over-provision here.
#
# CPUS is 8 but the smoke burned ~1.0 core: SPODS mask decomposition is
# single-threaded and the UCSF pull is one polite stream.  The headroom is for
# PDF rendering, which is the only parallel stage; do not read 8 as measured.
MEM="${VTS_FULLMARKS_MEM:-64G}"
CPUS="${VTS_FULLMARKS_CPUS:-8}"
TIME="${VTS_FULLMARKS_TIME:-4-00:00:00}"
PARTITION="${VTS_FULLMARKS_PARTITION:-cpu}"
JOB_NAME="${VTS_FULLMARKS_JOB_NAME:-fullmarks-build}"

LOGS="$VTS_FULLMARKS_OUT/logs"
mkdir -p "$LOGS" "$VTS_FULLMARKS_RAW" "$VTS_FULLMARKS_OUT"

cmd="${1:-status}"

case "$cmd" in
probe)
  cd "$HERE" && exec python build_corpus.py --probe
  ;;

build)
  # build_corpus refuses this itself, but only once the job starts; saying so
  # here costs no queue wait.
  if [ -f "$VTS_FULLMARKS_OUT/build_report.json" ] && [ -z "$TIER_ARG" ]; then
    echo "REFUSED: $VTS_FULLMARKS_OUT already holds a finished build (build_report.json)." >&2
    echo "  set VTS_FULLMARKS_PIN_TIERS=$VTS_FULLMARKS_OUT/build_report.json to keep its tiers," >&2
    echo "  or VTS_FULLMARKS_NEW_VERSION=1 to declare a new corpus version (#3903)." >&2
    exit 2
  fi
  bash "$WT/scripts/experiments/preflight.sh" --exp "$VTS_FULLMARKS_OUT" \
    --job-name "$JOB_NAME" --mem "$MEM" --conc 1 || exit 1

  RUNNER="/exp/$USER/.fullmarks-build.$$.sh"
  cat > "$RUNNER" <<RUNNER_EOF
#!/usr/bin/env bash
set -uo pipefail
source "$WT/gridenv.sh"
source "$WT/scripts/experiments/pile/pile_env.sh"
export VTS_REPO="$WT"
export PATH="\$HOME/.local/bin:\$PATH"
export VTS_FULLMARKS_RAW="$VTS_FULLMARKS_RAW"
export VTS_FULLMARKS_OUT="$VTS_FULLMARKS_OUT"
export CUDA_VISIBLE_DEVICES=
cd "$HERE"
echo "node=\$(hostname) job=\$SLURM_JOB_ID start=\$(date -Is)"
echo "sources=$SOURCES distractors=$DISTRACTORS letterhead=$LETTERHEAD"
python -u build_corpus.py \\
  --sources "$SOURCES" \\
  --ucsf-distractors $DISTRACTORS \\
  --ucsf-letterhead-per-author $LETTERHEAD \\
  $ROSTER_ARG $TIER_ARG
echo "exit=\$? end=\$(date -Is)"
RUNNER_EOF
  chmod +x "$RUNNER"

  sbatch --job-name="$JOB_NAME" --partition="$PARTITION" \
    --mem="$MEM" --cpus-per-task="$CPUS" --time="$TIME" \
    --output="$LOGS/build-%j.out" --error="$LOGS/build-%j.out" \
    --wrap="bash $RUNNER"
  ;;

slate)
  # The human pass, as one bundle to take away.  Both sheet sets in one job
  # because they are one sitting: the merge slate fixes the *partition* (which
  # proposals are one mark) and membership fixes the *instances* (which crops
  # really are that mark), and doing them a week apart means re-deriving the
  # same context twice.  Neither touches a GPU and neither refetches a source --
  # this reads the built corpus and writes PNGs -- so it is minutes, not hours.
  #
  # "One sitting" is true only once there is a roster.  Membership walks the
  # classes carrying `on_roster`, and build_corpus stamps that only under
  # `--roster`, so on a freshly built corpus the second half has nothing to
  # render.  It used to render nothing quietly -- an empty verdicts.jsonl, a tar
  # that succeeded, exit 0 -- so a reviewer who took the bundle away believed
  # both passes were in it (#3601).  make_audit_slate.py now exits 3 for that
  # case; the bundle then holds the merge sheets alone and the log says why.
  #
  # MEM is 32G: rendering opens full-resolution scans (SPODS is A4 at 300 dpi,
  # ~2,476x3,480) one at a time, but membership walks every instance of every
  # class, so the page cache is the cost rather than any one image.
  RUNNER="/exp/$USER/.fullmarks-slate.$$.sh"
  cat > "$RUNNER" <<RUNNER_EOF
#!/usr/bin/env bash
set -uo pipefail
source "$WT/gridenv.sh"
source "$WT/scripts/experiments/pile/pile_env.sh"
export VTS_REPO="$WT"
export VTS_FULLMARKS_OUT="$VTS_FULLMARKS_OUT"
export CUDA_VISIBLE_DEVICES=
cd "$HERE"
echo "node=\$(hostname) job=\$SLURM_JOB_ID start=\$(date -Is)"
python -u make_audit_slate.py --task merge      --corpus "$VTS_FULLMARKS_OUT" || exit 1

SHEETS="audit/merge"
python -u make_audit_slate.py --task membership --corpus "$VTS_FULLMARKS_OUT"
rc=\$?
if [ \$rc -eq 0 ]; then
  SHEETS="audit/merge audit/membership"
elif [ \$rc -ne 3 ]; then
  exit 1
fi

TARBALL="audit/fullmarks-audit-\$(date +%Y%m%d).tar.gz"
cd "$VTS_FULLMARKS_OUT" && tar czf "\$TARBALL" \$SHEETS || exit 1
echo "bundle: $VTS_FULLMARKS_OUT/\$TARBALL"
echo "  contains: \$SHEETS"
[ \$rc -eq 3 ] && echo "  MEMBERSHIP NOT RENDERED -- no roster yet; re-run 'slate' after stage 2"
echo "end=\$(date -Is)"
RUNNER_EOF
  chmod +x "$RUNNER"

  sbatch --job-name="fullmarks-slate" --partition="$PARTITION" \
    --mem="${VTS_FULLMARKS_SLATE_MEM:-32G}" --cpus-per-task=2 \
    --time="${VTS_FULLMARKS_SLATE_TIME:-02:00:00}" \
    --output="$LOGS/slate-%j.out" --error="$LOGS/slate-%j.out" \
    --wrap="bash $RUNNER"
  ;;

siglip)
  # The audit's second opinion, on a semantic embedder (#3600).  One GPU job,
  # because it is the only part of the audit that needs a card: it embeds every
  # class instance once and caches the vectors, after which the re-render, the
  # analysis and every later slate read the cache on `cpu`.
  #
  # ~1.5k crops of a few hundred pixels is minutes on any card here, so this
  # does not get a type pin -- but VTS_GPU in the grid ~/.bashrc would silently
  # supply one, so it is cleared for the same reason `embed` clears it.
  if [ -n "${VTS_FULLMARKS_GPU:-}" ]; then
    GRES="gpu:$VTS_FULLMARKS_GPU:1"
  else
    GRES="gpu:$(env -u VTS_GPU python3 "$WT/scripts/slurm/pick_gpu.py"):1"
  fi
  echo "gres: $GRES"
  RUNNER="/exp/$USER/.fullmarks-siglip.$$.sh"
  cat > "$RUNNER" <<RUNNER_EOF
#!/usr/bin/env bash
set -uo pipefail
source "$WT/gridenv.sh"
source "$WT/scripts/experiments/pile/pile_env.sh"
export VTS_REPO="$WT"
export VTS_FULLMARKS_OUT="$VTS_FULLMARKS_OUT"
cd "$HERE"
echo "node=\$(hostname) job=\$SLURM_JOB_ID start=\$(date -Is)"
python -u siglip_audit.py --embed   --corpus "$VTS_FULLMARKS_OUT" || exit 1
python -u siglip_audit.py --analyze --corpus "$VTS_FULLMARKS_OUT" || exit 1
# Re-render both similarity passes against the cache, on the same node while it
# is already warm.  The merge slate is re-ordered and re-numbered, so the old
# merges.txt no longer refers to the same classes -- it is kept beside the new
# one rather than overwritten.
[ -f "$VTS_FULLMARKS_OUT/audit/merge/merges.txt" ] && \
  cp "$VTS_FULLMARKS_OUT/audit/merge/merges.txt" "$VTS_FULLMARKS_OUT/audit/merge/merges.phash.txt"
DESC="\${VTS_FULLMARKS_AUDIT_EMBEDDER:-siglip2_l}"
python -u make_audit_slate.py --task merge   --corpus "$VTS_FULLMARKS_OUT" --descriptor "\$DESC" || exit 1
python -u make_audit_slate.py --task cluster --corpus "$VTS_FULLMARKS_OUT" --descriptor "\$DESC" || exit 1
cd "$VTS_FULLMARKS_OUT" && tar czf "audit/fullmarks-audit-siglip-\$(date +%Y%m%d).tar.gz" \
  audit/merge audit/cluster audit/siglip/pairs.json audit/siglip/splits.json audit/siglip/query_check.json
echo "bundle: $VTS_FULLMARKS_OUT/audit/fullmarks-audit-siglip-\$(date +%Y%m%d).tar.gz"
echo "exit=\$? end=\$(date -Is)"
RUNNER_EOF
  chmod +x "$RUNNER"

  sbatch --job-name="fullmarks-siglip" --partition=gpu --gres="$GRES" \
    --mem="${VTS_FULLMARKS_SIGLIP_MEM:-32G}" --cpus-per-task=4 \
    --time="${VTS_FULLMARKS_SIGLIP_TIME:-02:00:00}" \
    --output="$LOGS/siglip-%j.out" --error="$LOGS/siglip-%j.out" \
    --wrap="bash $RUNNER"
  ;;

status)
  squeue -u "$USER" -n "$JOB_NAME" -o "%.10i %.16j %.8T %.12M %.12l %R"
  echo
  echo "raw:    $(du -sh "$VTS_FULLMARKS_RAW" 2>/dev/null | cut -f1)"
  echo "out:    $(du -sh "$VTS_FULLMARKS_OUT" 2>/dev/null | cut -f1)"
  echo "pdfs:   $(find "$VTS_FULLMARKS_RAW/ucsf/pdf" -name "*.pdf" 2>/dev/null | wc -l) fetched"
  echo "pages:  $(find "$VTS_FULLMARKS_OUT/images" -type f 2>/dev/null | wc -l) rendered"
  echo "free:   $(df -h "$VTS_FULLMARKS_OUT" | tail -1 | awk "{print \$4}")"
  echo "slate:  $(ls "$VTS_FULLMARKS_OUT"/audit/merge/slate_*.png 2>/dev/null | wc -l) sheet(s), $(ls "$VTS_FULLMARKS_OUT"/audit/merge/pairs_*.png 2>/dev/null | wc -l) pair sheet(s)"
  tail -5 "$(ls -t "$LOGS"/build-*.out 2>/dev/null | head -1)" 2>/dev/null
  ;;

embed)
  tier="${2:-s}"
  # `VTS_GPU` is set in the grid ~/.bashrc, and pick_gpu honours it by design
  # ("VTS_GPU is set; not querying the scheduler") -- which here would inherit
  # a v100 pin the embed stage never chose.  GRID-PLAYBOOK sec.2 is explicit
  # that a named type is a pin that outlives its reason in both directions, and
  # v100 has already cost 2.3x on a siglip2_l embed while L40S nodes idled.
  # Clear it so the scheduler actually gets asked.  VTS_FULLMARKS_GPU overrides.
  if [ -n "${VTS_FULLMARKS_GPU:-}" ]; then
    GRES="gpu:$VTS_FULLMARKS_GPU:1"
  else
    GRES="gpu:$(env -u VTS_GPU python3 "$WT/scripts/slurm/pick_gpu.py"):1"
  fi
  echo "gres: $GRES"
  RUNNER="/exp/$USER/.fullmarks-embed-$tier.$$.sh"
  cat > "$RUNNER" <<RUNNER_EOF
#!/usr/bin/env bash
set -uo pipefail
source "$WT/gridenv.sh"
source "$WT/scripts/experiments/pile/pile_env.sh"
export VTS_REPO="$WT"
export VTS_FULLMARKS_OUT="$VTS_FULLMARKS_OUT"
cd "$HERE"
echo "node=\$(hostname) job=\$SLURM_JOB_ID tier=$tier start=\$(date -Is)"
python -u embed_corpus.py --tier $tier --embedders "${VTS_FULLMARKS_EMBEDDERS:-sift_vlad,siglip}"
RUNNER_EOF
  chmod +x "$RUNNER"
  sbatch --job-name="fullmarks-embed-$tier" --partition=gpu --gres="$GRES" \
    --mem="${VTS_FULLMARKS_EMBED_MEM:-32G}" --cpus-per-task=4 \
    --time="${VTS_FULLMARKS_EMBED_TIME:-12:00:00}" \
    --output="$LOGS/embed-$tier-%j.out" --error="$LOGS/embed-$tier-%j.out" \
    --wrap="bash $RUNNER"
  ;;

*)
  echo "usage: $0 {probe|build|slate|siglip|status|embed <tier>}" >&2; exit 2 ;;
esac
