# 2026-10-19 — A run cut at 20 votes parted from today's 150-vote session in half its cells (#4800)

**Cost:** ~20 min of doubt over a fidelity check that looked failed, and a reference check that had to be
rewritten; no cells lost.

**What broke.** #4800 ran Region cells to 20 votes (`SOTA_MAX_STEPS=20`), because every vote its rules read falls by
vote 17. Its check against #4278's control cells (the same app code, 150 votes) compared the picks through vote 20
and found 89 of 164 sessions different. Every one parted at vote 14 to 20, after the quota, where the 150-vote
session starts the prompted weak-separation check (#4496, `phase` `prompt`). A check starts only when its opening
bands fit the budget left (EVAL.md), and a 20-vote budget at vote 14 has no room, so the cut run went on with Hard
picks instead. The budget is part of the session from that check on.

The study's own votes were never touched: through the vote that meets the quota all 288 shared sessions were the
same pick and the same row. The analyzer's `reference()` now compares through that vote and names the phase where
the runs part (`checks.json`).

**Prevented?** *Advice only.* A run cut short is the full session only up to the first click a budget-aware step
(the prompted check, the check before Done) could start. Compare it to a full run through the clicks the study reads,
and say which phase the two part in. If the study reads clicks past a possible prompt, run the full budget.
