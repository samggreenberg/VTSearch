#!/usr/bin/env bash
# #4779: price the Goods' centroid's line drawn over the unvoted media, on the FHIBE cells.
# The app draws the centroid's line over every media of the labelled collection, so the Goods it
# was built from count as what stands out. The app's rule runs as tagged rows over three corpora:
# the withheld half (a Find elsewhere), the whole pool (~pool, the app before #4779) and the pool
# less the voted media (~unvoted, the fix). This is centroid_count_4759.sh with the run dir, job
# names and rules of #4779.
#
#   centroid_unvoted_4779.sh {dirs K | variants K | launch K | pack K | size K IDX}
#
# Launch from a FROZEN worktree. Env: CC4779_DATE (default 2026-10-11), CC4779_RULES.
set -euo pipefail
export CC_STUDY=4779 CC4759_DATE="${CC4779_DATE:-2026-10-11}"
export CC4759_RULES="${CC4779_RULES:-midpoint count_gap_floor_skip25_b1 count_gap_floor_skip25_b1~pool count_gap_floor_skip25_b1~unvoted}"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/centroid_count_4759.sh" "$@"
