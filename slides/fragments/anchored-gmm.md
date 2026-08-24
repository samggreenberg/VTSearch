<!-- _class: full -->

![bg fit](figs/calib-fold-anchored-flow.png)

## Stop averaging cuts. Fuse the evidence.

<!-- build: figs/calib-fold-anchored-flow.build1.png -->

<!-- build: figs/calib-fold-anchored-flow.build2.png -->

<!-- build: figs/calib-fold-anchored-flow.build3.png -->

<!-- build: figs/calib-fold-anchored-flow.build4.png -->

<!-- build: figs/calib-fold-anchored-flow.build5.png -->

<!-- In the audience deck this slide is a six-page build sharing one page
     number: the figure assembles a step per advance, and this page — the
     complete picture — is where it lands. There are no bullets by design; the
     figure is the slide.

     Open with the sentence in the headline, because it is the whole
     iteration. The blend averages two finished answers. This one fuses the
     evidence before answering, which is the difference between asking two
     people and getting one opinion out of everything they both saw.

     The build says it in two halves. The first two advances are pure
     recapitulation and you can move fast: the corpus, the votes drawn out of
     it, the model trained on them, then the split and a model per half — the
     cross from iteration 1, unchanged, because held-out votes are about to be
     the point.

     Advance three is where the two lines meet. Each fold model scores the
     whole corpus, and that is what the panel underneath it is: the same
     histogram iteration 2 fitted, one per fold, drawn as bare bars with nothing
     over them. Say that out loud — it is the shape of the data and it is all
     anyone has. Iteration
     2 had to look at exactly this and *assume* the high mound was the Good
     one. That assumption is the ceiling this slide lifts, and it was measured
     wrong: the fit put a third of the corpus in the high component when the
     true prevalence was under a tenth. Confidently scored is not the same
     thing as a true match.

     Advance four is the beat of the talk. The held-out votes arrive — the
     same checks and Xs from iteration 1's score lines, on the same baseline,
     the identical evidence — and the humps take the Good and Bad hatching of
     the block they came from. Nothing about the shape changed. What changed
     is that the components are now *identified* rather than guessed, and you
     can see who did it. Deliver the line the iteration turns on here: the
     labels' job is to identify the components, not to estimate them. Fifty
     thousand scores describe two mounds perfectly well; what they cannot say
     is which mound is Good. Five votes settle that. Identification is a far
     cheaper question than estimation, which is exactly why this estimator
     does not starve the way iteration 1 does.

     Two things the figure shows if you point at them. The votes in each panel
     are the ones that fold's model never trained on — that is what the
     crossed strokes above are for — so they carry no train-set optimism.
     And one X in the right-hand panel sits above its own fold's cut: a fold
     model's ranking of votes it never saw is not trivially clean, and the fit
     has to survive that.

     Two numbers the figure deliberately does not carry, so say them. Each vote
     enters the fit pinned to its component and weighing κ, not one — a vote is
     a claim about identity, not a thirty-thousandth of the shape. And the
     labels' share of the fit is then κ times the vote count over that plus the
     corpus size, around two and a half percent at the shipped setting. That
     ratio is iteration 3½'s hand-tuned ramp arriving *derived*: it grows with
     the vote count on its own, with nobody choosing endpoints. The next slide
     but one is about how κ was nearly shipped wrong.

     Land on the last line, and undercut it in the same breath. Averaging θ₁
     and θ₂ is drawn plainly here and production does not do it: two fold
     models score the same corpus on scales that need not agree, so a cut
     crosses between them as a quantile rank rather than as a number, and the
     mean rank is realised on the final model's own distribution. Drawn plain
     because the plain version is what makes the fix obvious — and the fix is
     the next slide, so do not spend more than a sentence on it here. -->
