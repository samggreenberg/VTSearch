# 2026-09-02 — one pilot cell cleared a hazard the full wave then hit (#3319)

**Study:** #3319, extending the acquisition-offset grid past `−4`, to half steps,
and to a 400-click horizon. **Cost:** nothing re-run — the confound is one-sided
and the verdict survived it — but the report shipped a wrong sentence that had to
be retracted, and the deep wave can no longer answer half of its own question.

#3319's plan named positive exhaustion as the artefact that would masquerade as
"the offset stops mattering at depth", which is precisely the finding the deep
wave existed to test. So the hazard was pre-registered, correctly, as the thing
most likely to fake the result.

It was then cleared from **one sizing cell**. `backpack`, seed 0, ran 400 steps
and found 57 of ~150 available positives with the harvest rate still
accelerating — 14, 5, 15, 18 per hundred clicks. That reads as comfortable
headroom, and the report said so: *"the tail is real, not flattened for want of
positives."*

On the full 768-cell wave it was wrong by 25 points:

| arm | median positives @400 | median harvest of the sim half | cells >90% harvested |
|---|---:|---:|---:|
| `prod` | 22 | 14.7% | 0.0% |
| `acq_m3` | 123 | **82.0%** | **21.9%** |
| `acq_m4` | 128 | **85.3%** | **29.2%** |

`backpack` is simply a hard category. A sizing cell is chosen to measure **wall
clock and memory**, where one cell is a reasonable sample because the cost of a
cell barely depends on which cell it is. It is not a sample of anything that
varies *across* cells, and how many positives a category yields is the single
most category-dependent quantity in this harness — #2910 measured the whole
effect under study as a function of it.

**The generalisable rule: a pilot answers questions about the MACHINE, never
questions about the DATA.** Timing, memory, does-it-run — one cell. Anything whose
answer differs per category needs the grid, or a check that reads every category.

Two second-order points worth keeping:

- **The confound was one-sided, which is the only reason this was survivable.**
  The aggressive arms hit the ceiling and the control never came near it, so the
  ceiling could only *compress* their measured advantage. They still won by
  −0.033 in cost, so "the sign does not flip at depth" holds with margin. Had the
  deep result been a null, it would have been uninterpretable.
- **It cost the study the other half of its question.** "Does the optimum get
  *deeper* at depth?" cannot be answered by arms that are ceiling-limited over
  the last quarter of their trajectory. Only the falsifier survives: the optimum
  does not get *shallower*.

**The control added:** `preflight.sh` check **16b**. Check 16 already asked
whether the horizon empties the sim set of *media* — the 400-step run passed it
comfortably on 3873 media while harvesting 85% of the 150 positives inside them,
because an aggressive selector does not sample uniformly, which is the entire
point of it. 16b reads `category_counts` from `prepare_info.json`, takes the
**thinnest selected category** (it exhausts first), and notes when `max_steps`
reaches the positives available in the sim half. A note, not a failure: running
to exhaustion is sometimes exactly what a study wants, and it is the *unnoticed*
case this exists to stop. It fires on the 400-step configuration and stays quiet
on the 100-step one.
