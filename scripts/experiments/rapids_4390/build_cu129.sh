#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=3:00:00
#SBATCH --output=/expscratch/sgreenberg/rapids-4390/logs/%x-%j.log
# #4390 Turing+ path: a fresh scratch venv through `bash scripts/install.sh cu129`, unmodified.
set -uo pipefail
D=/expscratch/sgreenberg/rapids-4390; V=$D/venv-cu129; W=/expscratch/sgreenberg/worktrees/vts-4390
source /etc/profile.d/modules.sh; module load python/3.12.3
export PIP_CACHE_DIR=/expscratch/sgreenberg/cache/pip VTSEARCH_VERBOSE=1
[ -e "$V" ] && { echo "exists: $V"; exit 2; }
python3 -m venv "$V" && source "$V/bin/activate"
echo "[$(date +%T)] $(hostname) python=$(command -v python)"
cd "$W" && bash scripts/install.sh cu129; rc=$?
echo "[$(date +%T)] install.sh rc=$rc"
pip freeze --all > $D/freeze-cu129.txt; pip check > $D/check-cu129.txt 2>&1
grep -iE '^(torch|pandas|cuml|cudf|numpy|cupy|nvidia|onnxruntime|rmm|cuvs|pylibraft)' $D/freeze-cu129.txt
echo "== pip check"; cat $D/check-cu129.txt
exit $rc
