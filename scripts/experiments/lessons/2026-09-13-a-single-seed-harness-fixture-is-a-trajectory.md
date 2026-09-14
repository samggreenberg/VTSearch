# 2026-09-13 — A single-seed harness fixture is a trajectory, and a threshold change re-rolls it

**#3585.** Replacing the unanchored GMM fit broke two `tests_lib`
fixtures that have nothing to do with mixtures:

| Test | What it asserts | How it failed |
|---|---|---|
| `test_cost_decreases_over_time_for_overlapping_data` | late/early cost ≤ 1.15 | 1.1526 — over by 0.0026 |
| `test_negative_training_regret_is_not_clamped` | some step out-ranks the skyline | the fixture's one negative step (−0.027) went to zero |

Neither was a regression. **The threshold is not an output**: Autopilot's Hard
phase picks the unlabelled item nearest the decision boundary, so *any* change
to the threshold rule sends a simulated run down a different path from its
second Hard pick onward. A fixture that holds its claim by a hair on one seed
stops holding it, and the failure reads exactly like the thing the test was
written to catch — "cost stopped improving", "the harness started clamping".

**Cost:** ~40 minutes, most of it spent looking for a bug in the fit.

**What it looks like when it is really a regression.** Sweep the seed. Here the
ratio spanned 0.71–1.15 across four seeds under *both* arms; the failing seed
sat at 1.03 of a 1.15 bound beforehand, and one of the others moved 0.06 in the
candidate's favour. A change that is real moves the seeds together.

**Prevented?** Partly, and only for these two.
`test_cost_decreases_over_time_for_overlapping_data` now pools three seeds — a
claim about a trend needs more than one trajectory, which its own docstring
always said. `test_negative_training_regret_is_not_clamped` moved to a fixture
seed with five negative steps instead of one and asserts three, so it fails
while there is still margin to see why.

**Still only advice:** the harness has other single-seed fixtures asserting
soft inequalities, and the next threshold change will find them. If it finds
several, the fix is a shared "pool over seeds" helper rather than another
paragraph here.
