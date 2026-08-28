<!-- _class: full -->

![bg fit](figs/calib-fold-anchored-flow.png)

## Above<br>Average

<!-- build: figs/calib-fold-anchored-flow.build1.png -->

<!-- build: figs/calib-fold-anchored-flow.build2.png -->

<!-- build: figs/calib-fold-anchored-flow.build3.png -->

<!-- build: figs/calib-fold-anchored-flow.build4.png -->

<!-- build: figs/calib-fold-anchored-flow.build5.png -->

<!-- build: figs/calib-fold-anchored-flow.build6.png -->

<!-- build: figs/calib-fold-anchored-flow.build7.png -->

<!-- The headline is the iteration: the blend averaged two finished answers;
     this fuses the evidence *before* answering. -->

<!-- **a**, **b** — Recapitulation, so move fast: the corpus, the votes, the
     model, the split, a model per half. -->

<!-- **c** — Where the two lines meet. Each fold model scores the whole corpus,
     and that is the panel underneath: bare bars with nothing over them. The
     shape of the data, and all anyone has. -->

<!-- **d** — Fit it, and there is iteration 2 again: one low component, one
     high. But every question mark asks the same thing — *which is the Good
     one?* — and the fit cannot answer, having read no labels. -->

<!-- **e** — So bring the other evidence. The held-out votes arrive on each
     fold's own baseline, the same checks and crosses as iteration 1, and the
     crossed strokes above say where they came from: fold 1 is read by the
     votes fold 1 never trained on. Nothing has been decided yet — the question
     marks are still question marks. -->

<!-- **f** — Read the two together, and the question marks give way to
     hatching. The shape did not change; the components are now *identified*.
     That is the line the iteration turns on — labels identify the components,
     they do not estimate them, and identification is the cheap question. -->

<!-- Worth being exact if anyone asks how the two pages relate to the code:
     there is one estimator, not two. `fit_anchored_score_gmm` reads the corpus
     and the votes in the same EM, each vote weighted κ times over — the deck
     takes it apart into e and f because at the shipped κ the votes move the
     *shape* by almost nothing while settling the identity outright, which is
     the whole claim. The next slide draws that mechanism directly. -->

<!-- **g** — Each fold cuts at the midpoint of its own two means. **h** —
     Undercut the last line in the same breath: averaging θ₁ and θ₂ is drawn
     plainly and production does not do it, because two fold models score on
     scales that need not agree. -->
