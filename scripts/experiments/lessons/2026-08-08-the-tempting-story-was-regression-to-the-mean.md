# 2026-08-08 — the tempting story was regression to the mean (#2905)

**What happened.** Three environments disagreed about
`ACQUISITION_INCLUSION_OFFSET`, and a clean unifying explanation presented
itself: the offset is a *starvation remedy*, paying where the detector has few
positives and charging its price everywhere. I tested it by binning each cell on
how many positives the **`prod` arm** found in that cell, then reading the
treatment response per bin. The curve appeared in both voting modes, monotone
and beautiful, with a sharp crossover from benefit to harm.

The response was measured **against that same `prod` run**. Cells where `prod`
happened to do unusually well are, by construction, more likely to show a
negative delta. Mean reversion manufactures exactly that curve with no mechanism
at all.

**Re-cut on axes independent of the arm being scored** — the category's
`realized_prevalence`, and a leave-one-out baseline (the mean `prod` positives
of the category's *other* seeds) — the binary-voting curve **survived** at full
strength (AP slope −0.0207 on log prevalence, CI [−0.0259, −0.0159]; −0.0402 on
LOO). The region-voting curve **vanished**: significant on the contaminated axis
(−0.0074, CI excludes 0) and null on both clean ones. Half the finding was real
and half was an artefact, and they looked identical.

**Cost.** ~20 minutes, because the check was run before the report was written
rather than after. Had it not been, the report would have recommended a
supply-based rule on evidence that was half self-fulfilling, and the
voting-mode conclusion would have been backwards — at *matched* prevalence the
modes still differ, which is only visible once the contaminated axis is gone.

**Still advice — never bin on a quantity that also appears in the contrast.**
If cells are grouped by a baseline arm's own outcome and then scored on
`treatment − baseline`, the grouping variable is inside the response and the
slope is partly arithmetic. Two cheap fixes, both used here: bin on something
fixed by the data (prevalence, category, pool size), or leave the cell's own
observation out of the statistic it is binned on. The diagnostic signature is
worth memorising: **an effect that is significant on the contaminated axis and
absent on the clean ones is mean reversion, not mechanism.**
