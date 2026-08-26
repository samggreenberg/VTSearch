<!-- _class: full -->

![bg fit](figs/calib-gmm-flow.png)

## A Mixed Blessing

<!-- build: figs/calib-gmm-flow.build1.png -->

<!-- build: figs/calib-gmm-flow.build2.png -->

<!-- build: figs/calib-gmm-flow.build3.png -->

<!-- build: figs/calib-gmm-flow.build4.png -->

<!-- **a** — The same votes and the same model as the last slide, rearranged.
     The new object is the grey bar above: the unlabeled corpus the votes were
     drawn out of. Far wider, and no Good/Bad hatching, because unlabeled means
     the classes are unknown, not absent. The labelled sliver is what iteration
     1 starved on, and the grey bar was there the whole time. -->

<!-- **b** — So run the loop the other way round. Train M₀ on the votes as
     before. **c** — And score the *whole corpus*: fifty thousand scores instead
     of tens, and their shape is bimodal on its own. -->

<!-- **d** — Fit a two-component Gaussian mixture to it. **e** — And cut at the
     midpoint between the two means. That is the whole estimator, and nothing
     in the bottom half of the figure ever looks at a vote. -->

<!-- The midpoint looks naive and survived two separate attempts to replace it.
     And colouring the low mode rust and the high mode green is an assumption
     the fit cannot justify: measured later, it was wrong by a factor of four —
     a fitted high-component weight of 0.35 against a true prevalence of 0.09. -->
