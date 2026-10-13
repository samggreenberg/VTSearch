<!-- _class: full -->

![bg fit](figs/calib-changing-hands.png)

## Changing<br>Hands

<!-- build: figs/calib-changing-hands.build1.png -->

<!-- build: figs/calib-changing-hands.build2.png -->

<!-- **a** — So who draws the line changes as the labels pile up: three hands,
     two rules. Before there's a trained detector, we rank by the typed query,
     or, if we started from example photos, by how close things are to the
     Goods' average. Either way, the line counts what stands out. -->

<!-- **b** — At three Goods and four Bads, there's enough to train the head,
     the tiny model on top of the big network that our votes actually train.
     But with only a few Goods, its fit can't tell where the matches are yet
     and can hand back way too much. So it keeps counting what stands out, now
     on its own scores. -->

<!-- **c** — Once the folds hold out six Goods, which takes about ten in hand,
     the labels line takes over: the fit, cut at F-beta's peak. These gates
     count labels, not clicks, so labels you import land in the same place a
     session would, and Find makes the same choice from the labels it's given. -->
