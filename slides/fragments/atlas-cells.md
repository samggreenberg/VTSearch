<!-- _class: full -->

![bg fit](figs/atlas-cells.png)

## Cover Story

<!-- build: figs/atlas-cells.build1.png -->

<!-- build: figs/atlas-cells.build2.png -->

<!-- build: figs/atlas-cells.build3.png -->

<!-- build: figs/atlas-cells.build4.png -->

<!-- build: figs/atlas-cells.build5.png -->

<!-- **a** — Same photos, forget the detector. What parts does this collection
     have? -->

<!-- **b** — Split it into three groups of similar photos. Plain k-means,
     nothing to train. -->

<!-- **c** — Then each group into three, down to cells of about twenty. -->

<!-- **d** — Every vote marks its cell and every bigger cell around it:
     visited. -->

<!-- **e** — So "where haven't I been?" is a walk from the top, biggest first,
     to the first cell with no votes. This one. -->

<!-- **f** — Then ask about the photo there most likely to prove the detector
     wrong: lowest scorer if the cell looks Good, highest if Bad. A flip is a
     pocket it had backwards. -->
