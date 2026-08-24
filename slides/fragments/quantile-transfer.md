<!-- _class: full -->

![bg fit](figs/calib-quantile-flow.png)

## The Rank<br>& File

<!-- The build that finishes the algorithm. This page — the complete picture —
     is what actually ships today, end to end. If someone photographs one slide
     of this talk, it should be this one. The line to say as it comes up:
     "cuts don't transfer; ranks do". -->

<!-- **a**, **b**, **c** — Recapitulation, at speed: the corpus, the votes, the
     model; the split and a fold model per half; then both fold panels arriving
     at once, fitted, voted, cut. That third advance is the whole of the
     previous slide in one step, deliberately — this slide needs its advances
     for the three moves that are new. -->

<!-- **d** — Where the slide turns, and worth slowing down for, because what it
     adds looks redundant and is not. M₀ scores the corpus too, and its
     distribution appears on the right — bare bars, no fitted curves. Say that
     out loud: nothing is estimated here. This is not a third piece of
     evidence. It is the *scale the answer has to be spoken in*, because M₀ is
     the model that will apply the threshold, and a model's scores are its
     own. -->

<!-- **e** — The strawman, and let the room do the arithmetic. The two fold cuts
     are 0.50 and 0.66; average them and you get 0.58; here is 0.58 on M₀'s
     distribution. It is the middle of the Good mound. Not slightly wrong — it
     throws away half of everything the detector was built to find. Nothing has
     gone wrong with any of the three fits. Three models scored the same media
     and none of them agreed what 0.58 means. -->

<!-- **f** — The fix, in one sentence: stop reading the cut as a number and read
     it as a **share**. Each fold's cut admits some fraction of the corpus —
     that is the bar under each panel, sitting directly under the cut it
     re-reads — and the two folds that disagreed about the number agree about
     the fraction. Not luck: a quantile survives any monotone re-scoring of the
     same corpus, and two models scoring the same haystack are close enough to
     that. Average the shares. -->

<!-- **g** — Realise it: find the score on M₀'s own distribution that admits
     that share, and that is θ₀. It lands in the valley, which is where the
     answer was always supposed to be. -->

<!-- Two honesty notes, both worth volunteering. Production never went through
     the cardinal-averaging stage — quantile transfer was in the design from
     the start, and the strawman is drawn because it makes the fix obvious, not
     because it is history. And the two fold shares are drawn as 79% and 81% to
     make the averaging legible; measured, they agree to within a fifth of a
     percent, which is a stronger version of the same point. -->
