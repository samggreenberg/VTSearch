<!-- _class: full -->

![bg fit](figs/calib-gmm-flow.png)

## Oops! All<br>Haystack

<!-- build: figs/calib-gmm-flow.build1.png -->

<!-- build: figs/calib-gmm-flow.build2.png -->

<!-- build: figs/calib-gmm-flow.build3.png -->

<!-- build: figs/calib-gmm-flow.build4.png -->

<!-- **a** — The same votes and the same model as the last slide, rearranged.
     The new object is the grey bar above: the unlabeled corpus the votes were
     drawn out of. Far wider, and no Good/Bad hatching, because unlabeled means
     the classes are unknown, not absent. The labelled sliver is what the last
     slide starved on, and the grey bar was there the whole time. -->

<!-- **b** — So run the loop the other way round. Train M₀ on the votes as
     before. **c** — And score the *whole corpus*: fifty thousand scores instead
     of tens, and their shape is bimodal on its own. -->

<!-- **d** — Fit a two-component Gaussian mixture to it. **e** — And cut at the
     midpoint between the two means. That is the whole estimator, and nothing
     in the bottom half of the figure ever looks at a vote. -->

<!-- Say which half lasted. Reading the corpus did: section III's line
     counts its matches on exactly this grey bar. The midpoint did not.
     Scored in F-beta, cutting here instead of cross-calibrating lost 0.38,
     0.40 and 0.33 at β ¼, 1 and 4 (#4582): where under half a percent
     of the corpus matches, a cut between the two means keeps far more than
     there are matches. And the colours are an assumption the fit cannot
     justify: measured, the high component weighed 0.35 against a true
     prevalence of 0.09. -->
