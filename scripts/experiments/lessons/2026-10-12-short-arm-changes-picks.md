# 2026-10-12 — An arm cut short of the full session picks differently near its end (#4736)

**What happened.** #4736's arm only needed votes 1-30, so it ran the 720 COCO Binary sessions a preset with
`SOTA_MAX_STEPS=30` and compared them with #4786's 150-vote sessions. In 150-200 of the ~400 runs a preset finished by then, the
picks diverged from about vote 16. The voting budget is read inside the loop by the prompted weak check
(#4496): it starts only when its opening rounds fit the votes left (`t + _check_start_cost() <= n_steps`;
15 votes at beta 1/4 and 1, 25 at beta 4). Cut at 30, every check due after vote 15 never started.

**Cost.** Three V100 pack jobs, about 20 minutes, cancelled and re-run cut at 60 votes (`hold_4736.sh`),
after which all 702 runs a preset matched click for click through vote 30.

**The rule.** A run cut at N votes is today's session only through vote N minus the longest prompted check
(25 votes today), not through N. Cut at least that far past the last vote the study reads, and check the
picks against a full session before reading anything off them.

**Status: advice.** Nothing checks it. `handover_4736.py arm` compares picks click for click and reports the
count, which is how it was caught.
