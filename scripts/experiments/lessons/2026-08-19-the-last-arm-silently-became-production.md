# 2026-08-19 — the last arm in the list silently became "production" (#2808)

**Study:** #2808 linear-head convergence. **Cost:** near-miss, caught before the
run finished. Would have printed every "production" figure against the
counterfactual arm.

`analyze_spikes.py` was written for #2847's fixed 2×2, where the production arm
genuinely was the last one listed (`D_lin_fused`). #2808 reused the analyzer
through a new `SPIKE_ARMS` environment override with arms ordered
`C_mlp, A_shipped, B_converged` — control first, so the no-verdict guard reads
the right arm. That put **`B_converged`, the counterfactual, in the last slot**,
and `PRODUCTION_ARM` defaulted to it.

Nothing about this fails. The analyzer runs, the tables populate, and every
sentence that says "production" describes the arm that is not shipped. The error
is invisible in the output precisely because the output is well-formed.

**The generalisation:** when a constant becomes configurable, its *defaults*
carry the old configuration's assumptions. `ARMS[0]` and `ARMS[-1]` were correct
as long as the arm list was a literal; the moment it came from the environment,
position stopped meaning role.

**Status: prevented.** `analyze_spikes.py` now warns loudly when `SPIKE_ARMS` is
set without `SPIKE_PRODUCTION_ARM` / `SPIKE_CONTROL_ARM`, naming the arm it
picked, and #2808's launcher passes both explicitly.
