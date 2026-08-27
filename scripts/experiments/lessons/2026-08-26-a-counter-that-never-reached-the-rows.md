# 2026-08-26 — a counter that never reached the rows, and a check that was a no-op (#3267)

**Study:** #3267 Good Mining, the Autopilot-opening sweep.
**Cost:** 20 minutes of a 5760-cell array, caught because a contact-sheet script
was smoke-tested on live cells before the run was trusted. Had it not been, the
study's central comparison would have been wrong and nothing in the data would
have said so.

A startup schedule deliberately **will not finish while one vote class is still
empty** — handing a learned Hard sort a one-class labelset leaves the selector
picking at random. The harness keeps voting on the last round and counts those
clicks in `StartupState.extended_clicks`, whose docstring says, correctly:

> A non-zero value on an arm is a finding, not noise: the opening as written did
> not produce a trainable pair and the harness had to keep voting to get one.

**The counter lived only on the object.** No frame carried it. So a pick log
could not distinguish these two trajectories:

| | opening as written | clicks actually spent in the opening |
|---|---|---|
| `flat_mid` as designed | 16 | 16 |
| `flat_mid`, `coco_val/baseball glove/seed 0` | 16 | **200** — 184 held, 0 positives ever found |

`flat_mid` exists to be the **length-matched control**: the same 16 opening
clicks as every banded arm, none of them spent mining. An arm compared against
the second row is not length-matched to anything, and the whole "did it win on
depth or just on clicks?" question — the one the control was added for — silently
becomes unanswerable.

**And the analyzer already meant to catch it**, which is the part worth staring
at. `_declared_clicks` computed the overrun as

```python
last = int(rounds["startup_round"].max())
return int((rounds["startup_round"] < last).sum()
           + min((rounds["startup_round"] == last).sum(), len(rounds)))
```

`min(count_of_last_round, len(rounds))` **is** `count_of_last_round` — the count
of one round can never exceed the count of all of them. So the expression is
just `len(rounds)`, `open_overrun` was `len(op) - len(op)` = **identically zero
for every arm on every cell**, and the column read as a clean bill of health.
The state was never derivable from the round indices in the first place; the
reconstruction was reconstructing something the log did not contain.

**The shape.** Not a wrong value — a **defensible-looking derivation of a fact
the data does not hold**, guarded by a check that cannot fail. A zero that is
structurally unable to be non-zero is worse than a missing column, because a
missing column gets noticed. Compare
[the harness seeded from a crop](2026-08-26-the-harness-seeded-from-a-crop.md):
same family, a parameter carrying something other than what its name says, with
every downstream number populated and plausible.

**Prevented.** `StartupState.held_for_quorum` is public and every pick row now
carries `startup_held` and `startup_extended_clicks`. The analyzer reads the
column, splits `open_clicks` into `open_scheduled_clicks` + `open_overrun`, and
flags `open_starved`. A pick log written before the column existed reports
**NaN**, not zero — refusing to answer beats answering confidently from nothing.

**Still only advice:** the general rule that produced this. *If a run's rows
cannot state a fact, no analyzer can recover it — put the state in the frame at
the moment it is known, rather than deriving it later from what survived.* The
same thing bit #3156, which fixed it the same way (`seed_mode` / `seed_query`
replacing `exemplar_id`): "the root cause was not that the seeding was wrong but
that it was **unnameable after the fact**."
