<!-- _class: full -->

![bg fit](figs/calib-xcal-flow.webp)

## Grade My Own<br>Homework

<!-- build: figs/calib-xcal-flow.build1.webp -->

<!-- build: figs/calib-xcal-flow.build2.webp -->

<!-- build: figs/calib-xcal-flow.build3.webp -->

<!-- build: figs/calib-xcal-flow.build4.webp -->

<!-- build: figs/calib-xcal-flow.build5.webp -->

<!-- build: figs/calib-xcal-flow.build6.webp -->

<!-- **a** — So where should the line go? The textbook answer: train M₀ on
     every vote so far, D₀. But M₀ scoring its own votes is grading its own
     homework, so those scores look better than they are. -->

<!-- **b** — So we split the votes in half. -->

<!-- **c** — And train a model on each half. -->

<!-- **d** — M₁ scores D₂, the half it never saw. Those scores are honest:
     Bads low, Goods high. -->

<!-- **e** — So we cut between them: θ₁. -->

<!-- **f** — Same thing the other way round. This time one Bad lands above
     θ₂, so that cut's a trade-off. -->

<!-- **g** — Average the two cuts and hand θ₀ to M₀. That's cross-calibration.
     With enough votes it gets it right. The catch is that word: enough. -->
