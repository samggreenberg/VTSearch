<!-- _class: full -->

![bg fit](figs/calib-quantile-flow.png)

## The Rank<br>& File

<!-- build: figs/calib-quantile-flow.build1.png -->

<!-- build: figs/calib-quantile-flow.build2.png -->

<!-- build: figs/calib-quantile-flow.build3.png -->

<!-- build: figs/calib-quantile-flow.build4.png -->

<!-- build: figs/calib-quantile-flow.build5.png -->

<!-- build: figs/calib-quantile-flow.build6.png -->

<!-- This page is what ships today, end to end. The line to say as it comes up
     is "cuts don't transfer; ranks do". -->

<!-- **a**, **b**, **c** — Recapitulation at speed: corpus, votes, model; the
     split and a fold model per half; then both fold panels arriving at once,
     fitted, voted, cut. -->

<!-- **d** — Where it turns. M₀ scores the corpus too, and its distribution
     appears on the right — bare bars, nothing estimated. This is not a third
     piece of evidence. It is the *scale the answer has to be spoken in*,
     because M₀ is the model that will apply the threshold. -->

<!-- **e** — The strawman; let the room do the arithmetic. The fold cuts are
     0.50 and 0.66, the average is 0.58, and here is 0.58 on M₀. It is the
     middle of the Good mound. Three models scored the same media and none of
     them agreed what 0.58 means. -->

<!-- **f** — The fix, in one sentence: read the cut as a **share**, not a number.
     Each fold's cut admits some fraction of the corpus, and the two folds that
     disagreed about the number agree about the fraction. A quantile survives
     any monotone re-scoring. Average the shares. -->

<!-- **g** — Find the score on M₀'s own distribution that admits that share.
     That is θ₀, and it lands in the valley. -->
