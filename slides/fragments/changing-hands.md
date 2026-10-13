<!-- _class: full -->

![bg fit](figs/calib-changing-hands.png)

## Changing<br>Hands

<!-- build: figs/calib-changing-hands.build1.png -->

<!-- build: figs/calib-changing-hands.build2.png -->

<!-- **a** — So who draws the line changes as the labels pile up. Before we've
     trained anything, we rank by the text query or the Goods' average, and
     the line is the outlier cut. -->

<!-- **b** — Once we hit the quota we can train the head. But with that few
     Goods it can't place the matches yet, so it keeps the outlier cut, just
     on its own scores. -->

<!-- **c** — Once the folds hold out enough Goods, the labels line takes over.
     These gates count labels, not clicks, so imported labels land in the same
     place. -->
