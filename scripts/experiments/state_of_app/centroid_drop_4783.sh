#!/usr/bin/env bash
# #4783: the same drops on the State of the App's Binary Photo opening (fhibe/centroid_drop_4783.sh has the
# rules): state_of_app/centroid_count_4759.sh with the run dir, job names and rules of #4783.
#
#   centroid_drop_4783.sh {dirs SEEDS | launch SEEDS [FIRST [LAST]]}
#
# Launch from a FROZEN worktree. Env: CC4783_DATE (default 2026-10-11), CC4783_RULES.
set -euo pipefail
export CC_STUDY=4783 CC4759_DATE="${CC4783_DATE:-2026-10-11}"
export CC4759_RULES="${CC4783_RULES:-midpoint count_gap_floor count_gap_floor_skip25 count_gap_floor_skip50 count_gap_floor_rank5}"
exec bash "$(cd "$(dirname "$0")" && pwd)/centroid_count_4759.sh" "$@"
