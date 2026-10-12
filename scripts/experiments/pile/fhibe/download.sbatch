#!/usr/bin/env bash
#SBATCH --job-name=fhibe-dl
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=download-%j.log
#
# Download FHIBE Full Resolution from Mozilla Data Collective (#4699) and extract it.
#
# Submit from this directory (it finds mdc_stream.py beside it):
#   cd scripts/experiments/pile/fhibe && FHIBE_ROOT=/expscratch/$USER/fhibe \
#     sbatch --output=/expscratch/$USER/fhibe/download-%j.log download.sbatch
#
# The archive is streamed straight into tar, so the 136 GB tarball and the
# 136 GB extract never coexist on /expscratch (314 GB free on 2026-10-09).
# mdc_stream.py makes the stream resumable (R2 drops long transfers).
#
# FHIBE may not be redistributed: this directory is chmod 700 and must stay out
# of /exp/scale26 (group-shared). On a consent-revocation re-release, delete the
# old fhibe-full-resolution-<hash>/ dir and rerun this job.
#
# Needs: the owner's MDC API key in ~/.config/mdc/api_key (mode 600, never
# printed), and the dataset terms accepted on the MDC site.
set -euo pipefail
set +x
ROOT=${FHIBE_ROOT:-/expscratch/$USER/fhibe}
HERE=$SLURM_SUBMIT_DIR
mkdir -p "$ROOT" && chmod 700 "$ROOT"
DATASET_ID=cmur4xkmm00e3nv073rtjriqo
API=https://mozilladatacollective.com/api
KEY_FILE=$HOME/.config/mdc/api_key
umask 077
[[ -s $KEY_FILE ]] || { echo "FAIL: no API key at $KEY_FILE"; exit 2; }
echo "$(date '+%F %T') start on $(hostname); free on /expscratch: $(df -BG --output=avail $ROOT | tail -1)"

ask_url() {  # POST for a fresh signed URL; prints "filename<TAB>url"
  curl -sS --fail-with-body -m 60 -X POST -H "Authorization: Bearer $(<"$KEY_FILE")" \
    "$API/datasets/$DATASET_ID/download" \
  | python3 -c 'import json,sys; d=json.load(sys.stdin)
if "downloadUrl" not in d: sys.exit("API said: %s" % d)
print(d["filename"]+"\t"+d["downloadUrl"])'
}

resp=$(ask_url) || { echo "FAIL: download request refused (terms not accepted on the site, or bad key)"; exit 3; }
fname=${resp%%$'\t'*}; ver=${fname%.tar.gz}; dest=$ROOT/$ver
echo "$(date '+%F %T') streaming $fname -> $dest"
rm -rf "$dest.partial"; mkdir -p "$dest.partial"
# mdc_stream.py resumes dropped connections by byte range, so tar sees one
# unbroken stream. The archive was packed on a Mac: skip its xattr headers and ._ AppleDouble files.
python3 "$HERE/mdc_stream.py" "$DATASET_ID" "$ROOT/$ver.sha256" "$ROOT/$ver.name" \
  | tar -xz --warning=no-unknown-keyword --exclude='._*' -C "$dest.partial" \
  || { echo "FAIL: stream or extract"; exit 4; }

# Verify against filepaths.csv: every listed img and json must exist.
csv=$(find "$dest.partial" -maxdepth 6 -name filepaths.csv | head -1)
[[ -n $csv ]] || { echo "FAIL: no filepaths.csv in the archive"; exit 5; }
python3 - "$csv" <<'PY'
import csv, os, sys
p = sys.argv[1]; base = os.path.dirname(p)
rows = list(csv.DictReader(open(p)))
miss = [r[c] for r in rows for c in ("img", "json") if not os.path.exists(os.path.join(base, r[c]))]
print(f"filepaths.csv rows: {len(rows)}; missing files: {len(miss)}")
sys.exit(1 if miss or len(rows) != 10901 else 0)
PY
[[ -e $dest ]] && { echo "FAIL: $dest already exists; not overwriting"; exit 6; }
mv "$dest.partial" "$dest"
cat > "$ROOT/MANIFEST.txt" <<M
source: Mozilla Data Collective, dataset $DATASET_ID (FHIBE Full Resolution)
terms: https://fairnessbenchmark.ai.sony/legal/terms-of-use-fhibe-dataset
version (archive): $fname
sha256 (archive stream): $(cat "$ROOT/$ver.sha256")
downloaded: $(date '+%F %T %Z') by $USER, job $SLURM_JOB_ID
extracted to: $dest
filepaths.csv: ${csv/$dest.partial/$dest}
M
echo "$(date '+%F %T') DONE: $(du -sh "$dest" | cut -f1) in $dest"
