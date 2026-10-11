#!/usr/bin/env bash
# #4779: the same corpora on the State of the App's Binary Photo opening (fhibe/centroid_unvoted_4779.sh has the
# rules): state_of_app/centroid_count_4759.sh with the run dir, job names and rules of #4779.
#
#   centroid_unvoted_4779.sh {dirs SEEDS | launch SEEDS [FIRST [LAST]]}
#
# Launch from a FROZEN worktree. Env: CC4779_DATE (default 2026-10-11), CC4779_RULES.
set -euo pipefail
export CC_STUDY=4779 CC4759_DATE="${CC4779_DATE:-2026-10-11}"
export CC4759_RULES="${CC4779_RULES:-midpoint count_gap_floor_skip25_b1 count_gap_floor_skip25_b1~pool count_gap_floor_skip25_b1~unvoted}"
exec bash "$(cd "$(dirname "$0")" && pwd)/centroid_count_4759.sh" "$@"
