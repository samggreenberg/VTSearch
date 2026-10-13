<!-- _class: full -->

![bg fit](figs/atlas-cells.png)

## Cover Story

<!-- build: figs/atlas-cells.build1.png -->

<!-- build: figs/atlas-cells.build2.png -->

<!-- build: figs/atlas-cells.build3.png -->

<!-- build: figs/atlas-cells.build4.png -->

<!-- build: figs/atlas-cells.build5.png -->

<!-- **a** — Same photos, forget the detector for a minute. What parts does this
     collection have? -->

<!-- **b** — Split it into three groups of similar photos. Plain k-means,
     nothing to train. -->

<!-- **c** — Then each group into three, and keep going till a cell's down to
     about twenty photos. -->

<!-- **d** — Every vote marks its cell and every bigger cell around it. A cell
     with a vote in it is visited. -->

<!-- **e** — So "where haven't I been?" is a walk down from the top, biggest
     first, till the first cell with no votes. This one. -->

<!-- **f** — Then ask about the photo in it most likely to prove the detector
     wrong: the lowest scorer if the cell looks Good, the highest if it looks
     Bad. Flip it and we've found a pocket it had backwards. -->
