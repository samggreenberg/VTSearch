# The click-15 spot check already waits for a detector, now that the hold is 0 (#5530)

**2026-11-04.** The 2026-11-03 Binary Photo review (#5525) found Autopilot's first prompted spot check at a median of
click 15. That fell inside #4736's query hold (30 labels), so in three sessions in four the check's votes went to
the typed query's set before the app showed a detector. #5530 asked for an A/B: arm A is dev as shipped, and arm B
holds the check until the app shows a detector.

#5556 (merged in #5562) then set the hold to 0. A typed-query detector now gives the trained head from the label
quota: 3 Goods and 4 Bads, or a Good and 16 Bads (`label_quota.py`). This note re-reads the question with hold 0.
**Arm B is now the same as arm A in every session, so no A/B was run.**

**Why, from the code.** A prompted check can start only once Autopilot has left its opening (`good`, `bad`,
`more`):
- The weak-separation prompt is gated on `flow.phase not in _OPENING_PHASES` (`voting_iterations.py`, `weak_phase="learned"`, the app's arm).
- The check before Done needs all three lights green, which `next_phase` reaches only after the opening.

The opening ends only when the label counts already meet the quota. `bad` and `more` need 3 Goods and 4 Bads.
The one exception is a Good walk that ran dry, and that takes 16 misses in a row, which is the second quota (a
Good and 16 Bads). So by the first check pick, the app is already showing the trained head.

**From the review's own picks.** The hold changes what Find shows, never what Autopilot asks (#4736). #5556's
confirmation runs gave the same picks and stop at hold 0 and hold 40. So the review's sessions are the hold-0 app's.
`check_quota_5530.py` reads each session's pick log and compares two clicks. One is the first prompted check pick
(`check_kind` `weak` or `done`). The other is the first pick at which the counts meet the quota.

| | beta ¼ | beta 1 | beta 4 |
|---|---:|---:|---:|
| sessions | 11,520 | 11,520 | 11,520 |
| prompted (first check: `done` / `weak`) | 10,967 (6,107 / 4,860) | 10,962 (5,945 / 5,017) | 10,887 (6,053 / 4,834) |
| first check pick, median click (quartiles) | 15 (15, 26) | 16 (15, 27) | 15 (15, 26) |
| labels meet the quota, median click (quartiles) | 7 (7, 17) | 7 (7, 17) | 7 (7, 17) |
| **first check pick before the quota** | **0** | **0** | **0** |
| fewest clicks from the quota to the check (median) | 6 (8) | 6 (8) | 6 (8) |
| first check pick under 30 labels (inside the old hold) | 8,437 (77%) | 8,427 (77%) | 8,434 (77%) |

The prompted counts and the click quartiles match the review's table, so this read counts the same checks. In 77%
of sessions the check landed inside the 30-label hold, the "three in four" #5530 was filed on. With hold 0, the
detector those sessions see at the check is the trained head, so the share is now 0. 318 sessions a preset never
meet the quota, and none of them is prompted.

**So the early check is fine as shipped.** The issue raised two concerns. The check "audits a set the user hasn't
trained yet", and its votes "are spent before the first detector appears". With hold 0, neither is true in any
session: every check audits the trained head, at least 6 clicks after it first appears. Arm B's gate ("not until
the app shows a detector") never binds, so it would reproduce arm A pick for pick. No app change.

**Not asked here.** More than half of the first checks are the check *before Done* (`done`, median click 15), not the
weak-separation prompt (`weak`, median click 20). All three lights are green by click 15 in those sessions. Whether
a check that early is the best time to spend 20-30 votes is a timing question. It is not a hold question, and
this note does not price it.

## Files

- `scripts/experiments/state_of_app/check_quota_5530.py`: the read above. Run as
  `check_quota_5530.py <run>/results/cells OUT.csv 8` over `/expscratch/sgreenberg/state-of-the-app/2026-11-03-b{025,1,4}`,
  in about 2 minutes a preset on 8 CPUs.
- `measurements/summary.txt`: its printed summary for the three presets.
