#!/usr/bin/env bash
#SBATCH -J shots-5391
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o /expscratch/sgreenberg/runs/5391/check-%x-%j.log
# Arg 1: arm name (renders go to $R/<arm>); full check.sh in Chrome for Testing 153.0.8010.12.
set -euo pipefail
ARM=$1
WT=/expscratch/sgreenberg/worktrees/vts-5391-shots${2:-}
R=/expscratch/sgreenberg/runs/5391
module load "$VTS_MODULE"
source "$VTS_VENV/bin/activate"
export TMPDIR=$R/tmp-$ARM; mkdir -p "$TMPDIR"
export PLAYWRIGHT_BROWSERS_PATH=/expscratch/sgreenberg/cache/ms-playwright
export CHROMIUM_PATH=${CHROMIUM_PATH:-/expscratch/sgreenberg/cache/cft-153/chrome-linux64/chrome}
echo "FONTCONFIG_FILE=${FONTCONFIG_FILE:-unset}"
export VTSEARCH_MODELS_DIR=/expscratch/sgreenberg/vtsearch-data/models
export NG_CLI_ANALYTICS=false
echo "node $(hostname) $(date) commit $(git -C $WT rev-parse --short HEAD) arm $ARM"
grep -m1 "model name" /proc/cpuinfo
"$CHROMIUM_PATH" --version
curl -sf -o /dev/null http://localhost:5000/ && { echo "port 5000 busy on this node"; exit 3; }
# The frontend build is shared by both arms; build once, under a lock.
( flock 9; [ -f $R/.built$ARM ] || { cd $WT/frontend && npm ci --no-audit --no-fund && npm run build:prod && cd $WT/scripts/screenshots && npm ci --no-audit --no-fund && touch $R/.built$ARM; } ) 9>$R/.build.lock
cd $WT/scripts/screenshots
rm -rf $R/$ARM; mkdir -p $R/$ARM
sed -e "s|^TMP=\"\$(mktemp -d)\"|TMP=$R/$ARM|" -e "s|^trap 'rm -rf \"\$TMP\"' EXIT||" check.sh > check-keep-$ARM.sh
chmod +x check-keep-$ARM.sh
echo "=== check.sh start $(date)"
set +e
./check-keep-$ARM.sh; rc=$?
echo "=== check.sh rc=$rc $(date)"
rm -f check-keep-$ARM.sh
exit $rc
