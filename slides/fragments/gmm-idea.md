![bg right:70% fit](figs/calib-gmm-flow.png)

### Iteration 2 — the idea

## A cut with no labels at all

<!-- build: figs/calib-gmm-flow.build1.png -->

<!-- build: figs/calib-gmm-flow.build2.png -->

<!-- build: figs/calib-gmm-flow.build3.png -->

<!-- build: figs/calib-gmm-flow.build4.png -->

<!-- In the audience deck this slide is a five-page build (one page number):
     the figure assembles a step per advance, and this page — the complete
     picture — is where it lands. There are no bullets by design; the figure
     is the slide, and everything below is what you say over it.

     The figure opens with the previous slide's drawing rearranged rather than
     a new one: D₀ —train→ M₀ is exactly what cross-calibration already
     showed, and the new object is D₋₁ above it — the unlabeled haystack the
     votes were drawn out of. Same height as D₀ and the same left edge, far
     wider, and with no Good/Bad hatching, because unlabeled means the classes
     are unknown, not absent. That contrast is the whole argument for the
     iteration: the labelled sliver is what iteration 1 was starving on, and
     the grey bar above it was there the entire time.

     So run the loop the other way round. M₀ scores all of D₋₁ — 50 000 scores
     instead of tens — and the histogram of those scores is bimodal on its own:
     a big mound of clear rejects, a smaller mound of high scorers. Fit a
     two-component Gaussian mixture (fit_score_gmm) and cut at the midpoint
     between the component means (calculate_gmm_threshold), which is what θ_G
     is — ticked exactly between the two marked means. Real vtscore fits on
     synthetic scores; the positives are drawn richer than a real haystack so
     both modes are visible at the back of the room.

     Two asides worth making. The midpoint looks naive and survived two
     attempts to replace it with something smarter — the epilogue will show
     #2836 proving it IS the rate-optimal cut under equal variances. And note
     what this estimator costs: nothing in the bottom half of that figure ever
     looks at a vote, which is both its superpower and, next slide, its
     ceiling.

     If you want to plant the seed for iteration 4 here, this is the place:
     colouring the low mode rust and the high mode green is an *assumption*
     the fit cannot justify, because it has read no labels. #2836 measured how
     wrong it gets — a fitted high-component weight of 0.35 against a true
     prevalence of 0.09. "High" means confidently scored, not true match. -->
