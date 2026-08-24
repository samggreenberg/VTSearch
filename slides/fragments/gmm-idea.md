<!-- _class: full -->

![bg fit](figs/calib-gmm-flow.png)

## A Mixed Blessing

<!-- build: figs/calib-gmm-flow.build1.png -->

<!-- build: figs/calib-gmm-flow.build2.png -->

<!-- build: figs/calib-gmm-flow.build3.png -->

<!-- build: figs/calib-gmm-flow.build4.png -->

<!-- **a** — The figure opens with the last slide's drawing rearranged: the same
     votes, the same model. The new object is the grey bar above — the
     unlabeled corpus the votes were drawn out of. Same height, same left edge,
     far wider, and no Good/Bad hatching, because unlabeled means the classes
     are unknown, not absent. That contrast is the whole argument: the labelled
     sliver is what iteration 1 was starving on, and the grey bar was sitting
     there the entire time. -->

<!-- **b** — So run the loop the other way round. Train M₀ on the votes as
     before. -->

<!-- **c** — And score the *whole corpus* with it: fifty thousand scores instead
     of tens. The shape of those scores is bimodal on its own — a big mound of
     clear rejects, a smaller mound of high scorers. -->

<!-- **d** — Fit a two-component Gaussian mixture to it. **e** — And cut at the
     midpoint between the two component means. That is the whole estimator.
     The figure runs the real shipped code on synthetic scores; the positives
     are drawn richer than a real corpus so both modes are visible from the
     back of the room. -->

<!-- Two asides. The midpoint looks naive and it survived two separate attempts
     to replace it — the epilogue is one of them. And note what this estimator
     costs: nothing in the bottom half of the figure ever looks at a vote.
     That is both its superpower and its ceiling. -->

<!-- If you want to plant the seed for iteration 4, this is the place: colouring
     the low mode rust and the high mode green is an *assumption* the fit cannot
     justify, because it has read no labels. Measured later, it was wrong by a
     factor of four — a fitted high-component weight of 0.35 against a true
     prevalence of 0.09. "High" means confidently scored, not true match. -->
