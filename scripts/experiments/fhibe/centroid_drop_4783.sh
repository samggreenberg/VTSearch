#!/usr/bin/env bash
# #4783: price drops for the Goods' centroid that look past a standout first photo, on the FHIBE cells.
# Since #4759 the line never keeps fewer than the media above the widest drop in the top 11
# (count_gap_floor); when a person's best photo stands far above their others, that drop is right after it.
# The candidates (CENTROID_DROPS) run as tagged rows against count_gap_floor, as #4759's did: this is
# centroid_count_4759.sh with the run dir, job names and rules of #4783.
#
#   centroid_drop_4783.sh {dirs K | variants K | launch K | pack K | size K IDX}
#
# Launch from a FROZEN worktree. Env: CC4783_DATE (default 2026-10-11), CC4783_RULES.
set -euo pipefail
export CC_STUDY=4783 CC4759_DATE="${CC4783_DATE:-2026-10-11}"
export CC4759_RULES="${CC4783_RULES:-midpoint count_gap_floor count_gap_floor_skip25 count_gap_floor_skip50 count_gap_floor_rank5}"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/centroid_count_4759.sh" "$@"
