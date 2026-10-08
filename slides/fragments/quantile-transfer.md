<!-- _class: full -->

![bg fit](figs/calib-quantile-flow.png)

## No Mean Feat

<!-- build: figs/calib-quantile-flow.build4.png -->

<!-- build: figs/calib-quantile-flow.build5.png -->

<!-- build: figs/calib-quantile-flow.build6.png -->

<!-- Appendix: where the mixture line ended up before the labels line, end
     to end. Open on the problem: the folds' cuts are scalars, each on its
     own model's scale, and the obvious move is to take their mean. The
     argument is "cuts don't transfer; ranks do", and the measurement is not
     on its side: scored in F-beta the raw mean beat rank transfer, by 0.052,
     0.072 and 0.14 at β ¼, 1 and 4 (#4582). Today it draws the line and
     places Autopilot's question only as a fallback, when the votes cannot
     support the labels' model. -->

<!-- **a** — Where it turns. M₀ scores the corpus too, and its distribution
     appears on the right — bare bars, nothing estimated. This is not a third
     piece of evidence. It is the *scale the answer has to be spoken in*,
     because M₀ is the model that will apply the threshold. -->

<!-- **b** — The strawman, which measured better; let the room do the
     arithmetic. The fold cuts are
     0.50 and 0.66, the average is 0.58, and here is 0.58 on M₀. It is the
     middle of the Good mound. Three models scored the same media and none of
     them agreed what 0.58 means. -->

<!-- **c** — The fix, in one sentence: read the cut as a **share**, not a number.
     Each fold's cut admits some fraction of the corpus, and the two folds that
     disagreed about the number agree about the fraction. A quantile survives
     any monotone re-scoring. Average the shares. -->

<!-- **d** — Find the score on M₀'s own distribution that admits that share.
     That is θ₀, and it lands in the valley. -->
