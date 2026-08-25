<!-- _class: full -->

![bg fit](figs/calib-fold-anchored-flow.png)

## Above<br>Average

<!-- build: figs/calib-fold-anchored-flow.build1.png -->

<!-- build: figs/calib-fold-anchored-flow.build2.png -->

<!-- build: figs/calib-fold-anchored-flow.build3.png -->

<!-- build: figs/calib-fold-anchored-flow.build4.png -->

<!-- build: figs/calib-fold-anchored-flow.build5.png -->

<!-- build: figs/calib-fold-anchored-flow.build6.png -->

<!-- Open with the headline, because it is the whole iteration. The blend
     averages two finished answers. This one fuses the evidence *before*
     answering — the difference between asking two people and getting one
     opinion out of everything they both saw. -->

<!-- **a**, **b** — Pure recapitulation, so move fast: the corpus, the votes,
     the model, the split, a model per half. The cross from iteration 1,
     unchanged, because held-out votes are about to be the point. -->

<!-- **c** — Where the two lines meet. Each fold model scores the whole corpus,
     and that is the panel underneath it: the same histogram iteration 2
     fitted, one per fold, drawn as bare bars with nothing over them. Say that
     out loud — it is the shape of the data and it is all anyone has. -->

<!-- **d** — Fit it, and there is iteration 2 again: two components, one low
     and one high. But look at what fills them. Every one of those question
     marks is the same question — *which of these is the Good one?* — and the
     fit cannot answer it, because it has read no labels. That colouring is an
     assumption, and iteration 2 shipped it. -->

<!-- **e** — The beat of the talk. The held-out votes arrive — the same checks
     and crosses from iteration 1, on the same baseline, the identical evidence
     — and the question marks give way to the Good and Bad hatching of the
     block those votes came from. Nothing about the shape changed. What changed
     is that the components are now *identified* rather than guessed, and you
     can see who did it. -->

<!-- The line the iteration turns on: the labels' job is to **identify** the
     components, not to estimate them. Fifty thousand scores describe two mounds
     perfectly well; what they cannot say is which mound is Good. Five votes
     settle that. Identification is a far cheaper question than estimation,
     which is exactly why this estimator does not starve. -->

<!-- Two things the figure shows if you point at them. The votes in each panel
     are the ones that fold's model never trained on, so they carry no
     train-set optimism. And one cross in the right-hand panel sits above its
     own fold's cut: a fold model's ranking of votes it never saw is not
     trivially clean, and the fit has to survive that. -->

<!-- Two numbers the figure deliberately does not carry. Each vote enters the
     fit pinned to its component and weighing κ, not one — a vote is a claim
     about identity, not a thirty-thousandth of the shape. And the labels'
     share of the fit is κ times the vote count over that plus the corpus size,
     around two and a half percent at the shipped setting. That ratio is the
     previous slide's hand-tuned ramp arriving *derived*. -->

<!-- **f** — Each fold cuts at the midpoint of its own two fitted means. Same
     rule iteration 2 used, on a fit that has now been told which mound is
     which. -->

<!-- **g** — And the last line, which you should undercut in the same breath.
     Averaging θ₁ and θ₂ is drawn plainly here and production does not do it:
     two fold models score the same corpus on scales that need not agree. The
     fix is the next slide, so spend one sentence on it. -->
