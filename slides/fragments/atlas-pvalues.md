<!-- _class: full -->

![bg fit](figs/atlas-pvalues.png)

## A Likely Story

<!-- build: figs/atlas-pvalues.build1.png -->

<!-- build: figs/atlas-pvalues.build2.png -->

<!-- The atlas has a second job, and this is the slide where it loses. Having
     built a model of what a collection looks like, you can ask a new item how
     typical it is — which is how a detector trained on one dataset gets checked
     against another before anybody trusts its scores there. It answers with a
     p-value: the share of the training data that looks stranger than this. -->

<!-- **a** — So test it on data the atlas has never seen but which came from the
     same place, where the answer is known: a p-value should be below 5% about
     5% of the time. The shipped combiner reads **4.3%**. That is the number
     anybody checks, and it passes. -->

<!-- **b** — Up the page is the other thing it owes: how far the whole
     distribution sits from the uniform a calibrated p-value would be. Nothing
     is near the floor. The one that looks best at the number people check is
     not the one closest to calibrated, and taking the **median** across the
     path instead — the obvious repair — trades one failure for the other. -->

<!-- **c** — And on a patch embedder it collapses: those spaces are the least
     concentrated, and the atlas calls **12%** of its own held-out data strange.
     The verdict fires on 80% of in-domain runs against 93% cross-corpus, which
     is not a test. The route refuses patch embedders rather than answer. -->

<!-- Two things keep this honest rather than alarming. The **ranking** is
     unaffected, and ranking is all the diversity walk uses — it reads each
     cell's sorted items and never a p-value, so everything on the last two
     slides stands. And this was invisible for as long as the atlas existed,
     because the one number a spot check reads was the one number that was
     right. Measured properly in the #3329 study; the fix, if it is ever wanted,
     is a per-node model rather than a re-tuned alpha, which was priced and
     does not rescue it. -->
