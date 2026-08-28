# 2026-08-28 — a figure that hung, and three fixes that were not it (#3156)

**Study:** #3156 `vg_scale` map.
**Cost:** two hours of a dedicated node, a 30-minute timeout inside the chained
analysis, three unnecessary code changes, and the map landing without its
headline figures until the morning after.

**What broke.** `figures_trajectory.py` placed the endpoint labels on its
averaged curves with:

```python
y = value
while placed and abs(y - placed[-1]) < gap:
    y = placed[-1] + gap
```

Once `placed[-1] + gap == placed[-1]` in floating point, `y` stops moving while
the condition stays true. A **live lock**: no error, no output, no memory
growth, 99% CPU forever. `ends` is sorted ascending, so the whole construct is a
`max()` and never needed to be a loop at all:

```python
y = value if not placed else max(value, placed[-1] + gap)
```

16 figures in 72s, from not finishing.

**The expensive part was not the bug.** It was three consecutive explanations
offered instead of a diagnosis, each real, each defensible, none of them the
cause:

| explanation | true? | the cause? |
|---|---|---|
| `value_at` rescanned each run per step — quadratic | yes | no |
| one matplotlib `Line2D` artist per run, thousands per panel | yes | no |
| node contention from the 71-task array | no | no |

The third is the one to be embarrassed by. Two standalone runs 90 seconds apart
returned `RC=137` at 5428 runs and `RC=0` at 5537 — *more* data succeeding — and
that was quoted as evidence **for** contention. It was evidence that the model
was wrong. A live lock is a coin flip against any timeout, and a coin flip fits
every story.

Each fix also made the real diagnosis harder: sampling the spaghetti was sold to
the owner as a deliberate trade against his "show me all the seeds" request, and
it was solving a problem that did not exist.

**What actually worked, in four minutes:**

```bash
PYTHONFAULTHANDLER=1 timeout -s ABRT 240 python figures_trajectory.py ...
# Current thread ...
#   File ".../figures_trajectory.py", line 199 in figure_average
```

**The general form.** *When something hangs rather than fails, get the stack
before forming a theory.* A failure carries an error to reason from; a hang
carries nothing, so every explanation fits it equally well and the plausible
ones crowd out the true one. Reach for `faulthandler` first, not fourth.

**A second, smaller lesson from the same incident.** `analyse_all.sh` claimed
each step was non-fatal so one failure could not cost the others — but that only
ever covered a step that **exits** non-zero. A step that never exits takes the
whole job. Steps now run under `timeout -s KILL` (`STEP_TIMEOUT`, 30 min), and a
timed-out step is reported distinctly from a failed one. **Non-fatal is not the
same as bounded**, and for an unattended chain only the second one is worth
anything.
