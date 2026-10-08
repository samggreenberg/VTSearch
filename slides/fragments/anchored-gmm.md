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

<!-- Appendix. The blend averaged two answers; this fuses the evidence
     first. -->

<!-- **a**, **b** — A recap, so move fast: corpus, votes, model, split,
     folds. -->

<!-- **c** — Where the two lines meet. Each fold model scores the whole corpus,
     and that is the panel underneath: bare bars with nothing over them. The
     shape of the data, and all anyone has. -->

<!-- **d** — Fit it, and there is Oops! All Haystack again: one low component,
     one high. But every question mark asks the same thing — *which is the Good
     one?* — and the fit cannot answer, having read no labels. -->

<!-- **e** — Now the other evidence: the held-out votes, on each fold's own
     baseline as in Grade My Own Homework. The crossed strokes say where they
     came from: fold 1 is read by the votes fold 1 never trained on. Nothing
     is decided yet. -->

<!-- **f** — Read the two together, and the question marks give way to
     hatching. The shape did not change; the components are now *identified*.
     That is the line this slide turns on — labels identify the components,
     they do not estimate them, and identification is the cheap question. -->

<!-- If asked about the code: one estimator, `fit_anchored_score_gmm`, one
     EM over the corpus and the votes, each vote weighted κ times. At the
     shipped κ the votes barely move the shape and settle the identity, which
     is why e and f split it. The next slide draws it. -->

<!-- **g** — Each fold cuts at the midpoint of its own two means. **h** —
     Undercut the last line in the same breath: averaging θ₁ and θ₂ is drawn
     plainly and production did not do it, because two fold models score on
     scales that need not agree. -->

<!-- Measured in F-beta: 0.031, 0.042 and 0.076 over the blend at β ¼, 1
     and 4, and 0.21 to 0.33 under cross-calibration (#4582). Today's line
     reverses this slide's claim: the labels estimate the Good component, and
     the corpus fits only its share and the rest. -->
