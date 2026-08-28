<!-- _class: full -->

![bg fit](figs/calib-split-idea.png)

## Train More,<br>Check Less

<!-- build: figs/calib-split-idea.build1.png -->

<!-- The same picture as the slide before, with every number taken off it: the
     cuts and the shares were that slide's argument, and refit at a different
     split they would be different numbers anyway. One thing changes between
     these two pages, and it is the divider through D₀. -->

<!-- **a** — Every fold in this section has made the same quiet choice, and
     none of the last five slides said so out loud: half the votes train the
     fold model, half are held out to read its threshold from. Fifty-fifty was
     never measured. It was the obvious split, and it stayed. -->

<!-- **b** — So we measured it, and it moved. Seventy percent into Train — and
     that is why the divider becomes two. Each fold draws its *own* seventy
     percent out of the same votes, so at 70/30 the two training halves cannot
     be halves any more: they overlap, and forty percent of the votes train
     both fold models. Each one bows into the other's side; the lens in the
     middle is in both. -->

<!-- At twenty votes the move is four votes crossing the line: fourteen train
     the model, six place the cut. -->

<!-- And the reason it goes that way rather than the other. Early on the fold
     model is the starved thing — it is fitting a whole decision boundary out
     of ten examples — while the threshold it needs is one quantile of one
     list, and a quantile does not need many scores to sit in roughly the right
     place. So the scarce votes are worth more in Train. The measured curves
     are in the appendix; ask and I will bring them up. -->
