# 2026-08-17 — the report cited a script nobody committed (#3129)

**Cost:** ~30 min to recover the code off a cluster worktree; it would have been
the whole analysis if that worktree had been cleaned up.

**What broke.** The overview-bench report described "an env-gated per-media dump"
and cited `scripts/experiments/calibration/label_noise.py` as the script behind
its most checkable finding (VG's `sky` labels are missing). Neither was in the
tree: the dump hook was an uncommitted edit to `vtscore/eval/voting_iterations.py`
and the two analysis scripts were untracked files, all living only in
`/exp/sgreenberg/projects/vts-bench-err`. The report's headline label-noise claim
could not be reproduced, extended, or applied to the next dataset — and when the
owner asked for *more* examples, step one was archaeology on a scratch checkout.

**The doc gate said nothing, by design.** `scripts/check-docs.py` exempted
`docs/experiments/` from its PATH invariant, on the grounds that a run record
cites the cluster scratch dir and the throwaway scripts that drove it. The scratch
dirs never needed that exemption — they are absolute paths, or relative ones like
`agg/x.csv` whose first component is not a tracked top-level directory, so the
path check ignores them anyway. The exemption was only ever covering *scripts*,
which is exactly the thing that must not be throwaway.

**Prevented.** The exemption is gone (`PATH_SKIP_PREFIXES` is now `docs/plans/`
alone, since a plan legitimately names files its work has not created yet). An
experiment report can now only cite analysis code that exists in the tree;
`tests_lib/meta/test_docs_gate.py` pins both halves — the citation fires, the
scratch paths stay quiet. The three real violations this surfaced were
`scripts/sod/`, which lives in the `evaluation-framework` repo and is now an
allowlist entry with that reason.

**Related, still advice:** a dump file is rewritten at every step, so reading one
from a running job gives you a mid-run state that looks exactly like a result.
The `glasses` cell's numbers went into a draft at step ~90 of 150 (129 false
positives) and were wrong by the end (2,863). Wait for the job, then read.
