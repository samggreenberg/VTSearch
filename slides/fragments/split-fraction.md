<!-- _class: full -->

![bg fit](figs/calib-split-idea.png)

## Train More,<br>Check Less

<!-- build: figs/calib-split-idea.build1.png -->

<!-- The same picture as the slide before, with every number taken off it: the
     cuts and the shares were that slide's argument, and refit at a different
     split they would be different numbers anyway. One thing changes between
     these two pages, and it is the divider through D₀. -->

<!-- **a** — Every fold in this deck has made the same quiet choice, and no
     slide has said so out loud: half the votes train the fold model, half
     are held out to read its threshold from. Fifty-fifty was never measured.
     It was the obvious split, and it stayed. -->

<!-- **b** — So it was measured, under the mixture line, and it moved.
     Seventy percent into Train — and that is why the divider becomes two.
     Each fold draws its *own* seventy percent out of the same votes, so at
     70/30 the two training halves cannot be halves any more: they overlap,
     and forty percent of the votes train both fold models. Each one bows
     into the other's side; the lens in the middle is in both. -->

<!-- At twenty votes the move is four votes crossing the line: fourteen train
     the model, six place the cut. -->

<!-- The reason given was that the fold model is the starved thing, while a
     threshold is one quantile of one list. Today's line reads the held-out
     scores as a class model, not one quantile, and re-measured under it the
     split makes no difference: within 0.004 of F-beta either way at every
     radio (#4583). Holding out half returns more, 92 more items at β 4, for
     no recall the objective sees. 70/30 ships because nothing beats it. -->
