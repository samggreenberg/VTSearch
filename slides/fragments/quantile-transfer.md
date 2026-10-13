<!-- _class: full -->

![bg fit](figs/calib-quantile-flow.png)

## No Mean Feat

<!-- build: figs/calib-quantile-flow.build4.png -->

<!-- build: figs/calib-quantile-flow.build5.png -->

<!-- build: figs/calib-quantile-flow.build6.png -->

<!-- **a** — Here's where the mixture line ended up, right before the labels
     line replaced it. The two halves cut at 0.50 and 0.66. But M₀ is the model
     we keep, and that's its score on the whole pile, over on the right. That's
     the scale the answer has to be in. -->

<!-- **b** — The obvious move: average the cuts, 0.58, and use that on M₀.
     Which lands right in the middle of the Good bump. Three models scored the
     same photos, and none of them agree on what 0.58 means. -->

<!-- **c** — The fix: read each cut as a share, not a number. The first half's
     cut sits 79% of the way up its pile, the second's at 81%. They disagreed
     about the number, but they agree about the share. Average the shares: 80%. -->

<!-- **d** — Then find the score on M₀ that leaves 80% below it, and that's the
     cut, right in the valley where it belongs. Funny thing, though: when we
     measured it, the plain average did better. Today this only draws the line
     as a fallback, for a session whose votes can't support the labels line. -->
