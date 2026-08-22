![bg right:70% fit](figs/calib-gmm-flow.png)

### Iteration 2 — the idea

## A cut with no labels at all

<!-- build: figs/calib-gmm-flow.build1.png -->

<!-- build: figs/calib-gmm-flow.build2.png -->

<!-- build: figs/calib-gmm-flow.build3.png -->

<!-- build: figs/calib-gmm-flow.build4.png -->

<!-- In the audience deck this slide is a five-page build sharing one page
     number: the figure assembles a step per advance, and this page — the
     complete picture — is where it lands. There are no bullets by design; the
     figure is the slide.

     The figure opens with the previous slide's drawing rearranged rather than
     a new one: the votes, and the model trained on them, are exactly what
     cross-calibration already showed. The new object is the grey bar above —
     the unlabeled corpus the votes were drawn out of. Same height, same left
     edge, far wider, and with no Good/Bad hatching, because unlabeled means
     the classes are unknown, not absent. That contrast is the whole argument
     for the iteration: the labelled sliver is what iteration 1 was starving
     on, and the grey bar above it was sitting there the entire time.

     So run the loop the other way round. The model scores the whole corpus —
     fifty thousand scores instead of tens — and the histogram of those scores
     is bimodal on its own: a big mound of clear rejects, a smaller mound of
     high scorers. Fit a two-component Gaussian mixture to it and cut at the
     midpoint between the two component means. That is the whole estimator.
     The figure runs the real shipped code on synthetic scores; the positives
     are drawn richer than a real corpus so that both modes are visible from
     the back of the room.

     Two asides worth making. The midpoint looks naive, and it survived two
     separate attempts to replace it with something smarter — the epilogue
     shows one of them proving the midpoint IS the rate-optimal cut when the
     two components share a variance. And note what this estimator costs:
     nothing in the bottom half of that figure ever looks at a vote. That is
     both its superpower and, on the next slide, its ceiling.

     If you want to plant the seed for iteration 4, this is the place:
     colouring the low mode rust and the high mode green is an *assumption*
     the fit cannot justify, because it has read no labels. It was later
     measured to be wrong by a factor of four — a fitted high-component weight
     of 0.35 against a true prevalence of 0.09. "High" means confidently
     scored, not true match. -->
