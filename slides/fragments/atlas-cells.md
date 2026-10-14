<!-- _class: full -->

![bg fit](figs/atlas-cells.webp)

## Cover Story

<!-- build: figs/atlas-cells.build1.webp -->

<!-- build: figs/atlas-cells.build2.webp -->

<!-- build: figs/atlas-cells.build3.webp -->

<!-- build: figs/atlas-cells.build4.webp -->

<!-- build: figs/atlas-cells.build5.webp -->

<!-- build: figs/atlas-cells.build6.webp -->

<!-- **a** — Same photos, forget the detector. What parts does this collection
     have? So far, one: the whole thing, the top of a tree. -->

<!-- **b** — Split it into three groups of similar photos. Plain k-means,
     nothing to train. Three branches. -->

<!-- **c** — Then each group into three, down to cells of about twenty. A
     bigger dot is a bigger cell. -->

<!-- **d** — Every vote marks its cell and every bigger cell above it:
     visited. -->

<!-- **e** — So "where haven't I been?" is a walk from the top, a row at a
     time, biggest first, to the first cell with no votes. This one. -->

<!-- **f** — Then which photo in it? The detector calls this whole cell Bad,
     surest farthest from the checks. -->

<!-- **g** — So we ask the one it's least sure of, up by the checks. If that
     flips, it's a pocket the detector had backwards. In a cell it called
     Good, we'd ask its weakest. -->
