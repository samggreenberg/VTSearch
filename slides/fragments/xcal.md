<!-- _class: full -->

![bg fit](figs/calib-xcal-flow.png)

## Grade My Own<br>Homework

<!-- build: figs/calib-xcal-flow.build1.png -->

<!-- build: figs/calib-xcal-flow.build2.png -->

<!-- build: figs/calib-xcal-flow.build3.png -->

<!-- build: figs/calib-xcal-flow.build4.png -->

<!-- build: figs/calib-xcal-flow.build5.png -->

<!-- build: figs/calib-xcal-flow.build6.png -->

<!-- **a** — So where should the line go? Start with the textbook answer. D₀ is
     every vote so far, and M₀ is the model trained on all of it. That's the
     model we keep. But we can't pick a cut from M₀'s scores on its own votes:
     it's grading its own homework, and it'll look better than it really is. -->

<!-- **b** — So we split the votes in half, D₁ and D₂, and -->

<!-- **c** — train a separate model on each half, M₁ and M₂. -->

<!-- **d** — Now cross them over. M₁ scores D₂, the half it never trained on.
     Those scores are honest: the Bads land low and the Goods land high. -->

<!-- **e** — So there's a clean place to cut between them: θ₁. -->

<!-- **f** — Same thing the other way round: M₂ scores D₁. This time one Bad
     lands above the cut, so θ₂ is a trade-off, not a free gap. -->

<!-- **g** — Average the two cuts and hand θ₀ to M₀. That's called
     cross-calibration. With enough votes it lands in the right place, and
     nothing we tried after it beat it until the line in the next section,
     which is built on these same honest scores. The catch is that word:
     enough. -->
