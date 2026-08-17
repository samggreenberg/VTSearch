# 2026-08-17 — four decimals invented a finding (#3129)

**Cost:** ~1h of report rewrite, and two follow-up items that should never have
been written.

**What broke.** The overview-bench report quoted every arm mean to four decimals
from *unpaired* averages. Three claims were then read straight off those digits:
"the AP margin grows as targets shrink" (0.046 → 0.051 → 0.062), "you lose
~0.026 cost on VG by staying on `siglip`", and "the medium band beats the shipped
cut rule on all three encoders" (+0.017 / +0.010 / +0.004). Re-computed **paired**
on (category, seed) with a standard error, the first is three positive margins
whose ordering survives and whose *growth* is inside ±0.03; the second is
+0.04 ± 0.03, i.e. not measurable at all; the third is +0.010 ± 0.006 pooled —
three same-signed coin flips. One of them had already become a numbered
follow-up recommending a sweep.

**The mechanism is not "rounding".** With 3 seeds the between-category variance
dominates, and an unpaired mean carries it into the difference. Pairing cancels
most of it, which is why the paired SE is the number that decides whether a
difference exists. Four decimals additionally *invite* the error: a reader (me)
compares 0.0462 against 0.0508 and sees a trend, because nothing on the page
says the third decimal is noise.

**Prevented (partly).** `analyze_bench.py` now emits a `PAIRED ARM CONTRASTS`
table with `mean ± SE` for every arm pair, and `analyze_bench_interaction.py`
does the same across two result dirs (the binary-vs-boxes case, which one-dir
analysis could not pair at all). The `grid-experiments` skill's new "Writing the
report" section requires two significant digits, paired differences with an SE,
and the words "not resolvable" when |mean| < 2·SE.

**Still advice:** nothing stops a future report from quoting an unpaired mean.
The check that would catch it is a reader asking "±what?" — so the table now
prints the SE next to every difference, which makes the omission visible rather
than invisible.
