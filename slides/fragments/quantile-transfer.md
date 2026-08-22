![bg right:70% fit](figs/calib-quantile-flow.png)

### Iteration 4 — the fix

## Cuts don't transfer. Ranks do.

<!-- build: figs/calib-quantile-flow.build1.png -->

<!-- build: figs/calib-quantile-flow.build2.png -->

<!-- build: figs/calib-quantile-flow.build3.png -->

<!-- build: figs/calib-quantile-flow.build4.png -->

<!-- build: figs/calib-quantile-flow.build5.png -->

<!-- build: figs/calib-quantile-flow.build6.png -->

<!-- The seven-page build that finishes the algorithm. There are no bullets by
     design; the figure is the slide, and this page — the complete picture — is
     what actually ships today, end to end. If someone photographs one slide of
     this talk, it should be this one.

     The first three advances are recapitulation and you can move through them
     at speed. The corpus, the votes drawn out of it, the model trained on them;
     the split and a fold model per half; then both fold panels arriving at
     once — fitted, voted, cut. That third advance is the whole of the previous
     slide in one step, and it is deliberately one step: the audience has just
     been walked through it, and this slide needs its advances for the three
     moves that are new.

     Two numbers the figure still does not carry, so say them here if you did
     not on the previous slide: each vote enters the fit weighing κ rather than
     one, and the labels' share of the fit is κ times the vote count over that
     plus the corpus size. The slide after next is about how κ was nearly
     shipped wrong.

     Advance four is where the slide turns, and it is worth slowing down for,
     because the thing it adds looks redundant and is not. M₀ scores the corpus
     too, and its distribution appears on the right — in flat black, with no
     fitted curves over it. Say that out loud: nothing is estimated here. This
     panel is not a third piece of evidence. It is the scale the answer has to
     be spoken in, because M₀ is the model that will actually apply the
     threshold, and a model's scores are its own.

     Advance five is the strawman, and let the room do the arithmetic. The two
     fold cuts are 0.50 and 0.66; average them and you get 0.58; here is 0.58 on
     M₀'s distribution. It is the middle of the Good mound. Not slightly wrong —
     it throws away half of everything the detector was built to find. Nothing
     has gone wrong with any of the three fits. Three models scored the same
     media and none of them agreed what a score of 0.58 means, which is all it
     takes.

     Advance six is the fix, and it is one sentence: stop reading the cut as a
     number and read it as a *share*. Each fold's cut admits some fraction of
     the corpus — that is the bar under each panel — and the two folds, which
     disagreed about the number, agree about the fraction. That is not luck. A
     quantile survives any monotone re-scoring of the same corpus, and two
     models scoring the same haystack are close enough to that for the share to
     carry across where the number cannot. Average the shares.

     Advance seven realises it: find the score on M₀'s own distribution that
     admits that share, and that is θ₀. It lands in the valley, which is where
     the answer was always supposed to be.

     Two honesty notes, both worth volunteering. Production never went through
     the cardinal-averaging stage — quantile transfer was in the fold-anchored
     design from the start, and the strawman is drawn because it makes the fix
     obvious, not because it is history. And the two fold shares are drawn as
     0.79 and 0.81 to make the averaging legible; measured, they agree to within
     a fifth of a percent, which is a stronger version of the same point rather
     than a weaker one.

     The next slide is what this measures at. -->
