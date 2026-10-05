<!-- _class: full -->

![bg fit](figs/atlas-pvalues.png)

## A Likely Story

<div class="asof">Measured 2026-08-30</div>

<!-- build: figs/atlas-pvalues.build1.png -->

<!-- Far Out was the easy case: a whole collection a long way off. This is
     the slide where the atlas loses. Its answer is a p-value, the share of the
     training data that looks stranger than this, and a p-value owes something
     on data from the *same* place too. -->

<!-- **a** — So test it on data the atlas has never seen but which came from the
     same place, where the answer is known: a p-value should be below 5% about
     5% of the time. The shipped combiner reads **4.3%**. That is the number
     anybody checks, and it passes. -->

<!-- **b** — Up the page is the other thing it owes: how far the whole
     distribution sits from the uniform a calibrated p-value would be. Nothing
     is near the floor. The one that looks best at the number people check is
     not the one closest to calibrated, and taking the **median** across the
     path instead — the obvious repair — trades one failure for the other. -->

<!-- And on a patch embedder it collapses outright — say this one, it does not
     need a page. Those spaces are the least concentrated, and the atlas calls
     **12%** of its own held-out data strange; the verdict fires on 80% of
     in-domain runs against 93% cross-corpus, which is not a test. The route
     refuses patch embedders rather than answer. -->

<!-- Two things keep this honest rather than alarming. The **ranking** is
     unaffected, and ranking is all the diversity walk uses — it reads each
     cell's sorted items and never a p-value, so everything on the last two
     slides stands. And this was invisible for as long as the atlas existed,
     because the one number a spot check reads was the one number that was
     right. Measured properly in the #3329 study; the fix, if it is ever wanted,
     is a per-node model rather than a re-tuned alpha, which was priced and
     does not rescue it. -->
