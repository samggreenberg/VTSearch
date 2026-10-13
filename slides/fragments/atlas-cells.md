<!-- _class: full -->

![bg fit](figs/atlas-cells.png)

## Cover Story

<!-- build: figs/atlas-cells.build1.png -->

<!-- build: figs/atlas-cells.build2.png -->

<!-- build: figs/atlas-cells.build3.png -->

<!-- build: figs/atlas-cells.build4.png -->

<!-- build: figs/atlas-cells.build5.png -->

<!-- **a** — Same photos, and forget the detector for a minute. The only
     question is what parts this collection has. -->

<!-- **b** — Split it into three groups of similar photos. That's plain k-means
     on the points we already have; nothing new to train. -->

<!-- **c** — Then split each group into three, and keep going until a cell's
     down to about twenty photos. -->

<!-- **d** — Every vote marks its cell, and every bigger cell around it. Any
     cell with a vote in it counts as visited. -->

<!-- **e** — So "where haven't I been?" is a walk: from the top, a level at a
     time, biggest first, stopping at the first cell with no votes. This one. -->

<!-- **f** — Then show the photo in that cell most likely to prove the detector
     wrong: its lowest scorer if the cell looks Good, its highest if it looks
     Bad. If we flip it, we've found a pocket the detector had backwards. -->
