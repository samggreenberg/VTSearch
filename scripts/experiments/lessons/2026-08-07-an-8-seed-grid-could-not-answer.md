# 2026-08-07 — an 8-seed grid could not answer the question it was run to answer (#2877)

**What happened.** The VG region-voting generalisation check reused #2876's
sizing verbatim — 8 seeds, which on COCO had been comfortable — and drained
clean: 1288/1288 cells, no failures. It reproduced the mechanism perfectly. It
also put a 95% CI of **[−0.014, +0.019]** on the decision endpoint against a
pre-registered tolerance of **+0.01**. That interval contains "the offset is
free, keep it global" *and* "the offset costs something, gate it" — **opposite
shipping decisions**. The run had measured nothing decision-relevant.

**Why the transplanted sizing failed.** Sizing does not travel with an arm
table; it travels with the *endpoint's variance in that environment*. VG
region-voting costs sit near 0.43 where COCO's sit near 0.137, and the paired
per-cell SD is correspondingly larger (0.111). At that SD, a ±0.010 half-width
needs **n≈473**; 8 seeds × 23 categories delivered 180. The *positives* endpoint
was hugely over-powered at the same n in both environments, which is exactly how
this hides — the run looks healthy because the endpoint you can see moving is
the one that was never binding.

**Cost.** ~55 minutes of cpu-partition time to rerun at 24 seeds (n=540), which
put the CI at [+0.003, +0.022] and made the answer unambiguous. Cheap only
because these cells are single-threaded and GPU-free.

**Still only advice (no control).** When porting a study to a new environment,
**re-derive n from a pilot's observed SD on the decision endpoint** before
running the full grid — do not inherit the seed count along with the arm table.
One arm's worth of pilot is enough to compute it: `n = (1.96·SD/half_width)²`.
And report the CI on the decision endpoint even when the ship rule passes, so a
wide null is never mistaken for a tight one. (`analyze_acq.py` already refuses to
read a p-value as a null for this reason; the gap was that nothing checked
whether the *design* could produce a usable interval.)

**Prevented, separately — smoke on a representative cell, not on cell 0.** The
first smoke ran array index 0 and wrote **zero rows**, which looks exactly like
a broken harness. It was not: rows are only emitted from the first positive
onward, and index 0 was `bag`/seed 0, whose first positive arrives at vote 106 —
the worst of 92 cells, where the median is vote 3. Fifteen minutes went to
confirming the harness was fine. Cell 0 is the alphabetically-first category at
seed 0, which is a biased draw, not a neutral one; and "0 rows" is a legitimate
outcome in a starved environment, so it cannot be treated as a failure signal on
its own. Smoke a mid-grid index, and check the row count against a known-good
run of the same environment before concluding anything.
