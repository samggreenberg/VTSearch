<!-- _class: full -->

![bg fit](figs/calib-xcal-flow.png)

## Cross Examination

<!-- build: figs/calib-xcal-flow.build1.png -->

<!-- build: figs/calib-xcal-flow.build2.png -->

<!-- build: figs/calib-xcal-flow.build3.png -->

<!-- build: figs/calib-xcal-flow.build4.png -->

<!-- build: figs/calib-xcal-flow.build5.png -->

<!-- build: figs/calib-xcal-flow.build6.png -->

<!-- The pre-history of the line: the textbook answer everything else is measured
     against. Walk the mechanism off the figure, top to bottom, and land on the
     property that defines it — this estimator is *consistent*. With enough
     labels it converges on the right answer. -->

<!-- **a** — D₀ is every vote so far, and M₀ is the model trained on all of it.
     M₀ is the model you keep; the problem is that its scores on its own
     training votes are optimistically shifted, so you cannot cut on them. -->

<!-- **b** — So split the votes in half. -->

<!-- **c** — And train a model on each half. -->

<!-- **d** — Now cross them: each fold model scores the half it never trained
     on. Honest scores, at the price of training extra models on half the
     data. That X in the middle of the figure is the "cross" in
     cross-calibration. -->

<!-- **e** — On each half the Bad scores mostly pile up low and the Good scores
     high, and a cut goes between them. **f** — Same on the other half — and
     note the Bad that lands above θ₂. Each cut is a trade-off, not a free
     gap. -->

<!-- **g** — Average the two cuts and hand θ₀ to M₀. Green is Good media, rust
     is Bad, matching the checks and crosses on the score lines. -->

<!-- The shipped code has since refined this — the halves are pooled into one
     score set rather than cut separately, the cut is a quantile the Inclusion
     knob can bias, the splits are redrawn rather than fixed. Later polish, not
     the idea; do not front-load it. The next slide is what happens *before*
     "enough labels". -->
