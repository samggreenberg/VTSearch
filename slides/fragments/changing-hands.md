<!-- _class: full -->

![bg fit](figs/calib-changing-hands.png)

## Changing<br>Hands

<!-- build: figs/calib-changing-hands.build1.png -->

<!-- build: figs/calib-changing-hands.build2.png -->

<!-- **a** — So who draws the line changes as the labels pile up: three hands,
     two rules. Before there's a trained detector, we rank by the typed query
     or the Goods' average, and the line counts what stands out. -->

<!-- **b** — At three Goods and four Bads, we can train the head, the tiny
     model our votes actually train. But with only a few Goods it can't place
     the matches yet, so it keeps counting what stands out, on its own scores. -->

<!-- **c** — Once the folds hold out six Goods, about ten in hand, the labels
     line takes over. These gates count labels, not clicks, so imported labels
     land where a session would, in Train or Find. -->
