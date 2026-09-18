<!-- _class: full -->

![bg fit](figs/atlas-cone.png)

## Cone of Silence

<!-- build: figs/atlas-cone.build1.png -->

<!-- build: figs/atlas-cone.build2.png -->

<!-- build: figs/atlas-cone.build3.png -->

<!-- The detail that makes the last slide work at all, and the one that would
     have quietly broken it. Worth two minutes because it is not obvious and it
     is not optional. -->

<!-- **a** — Every stored embedding is a unit vector, so the collection lives on
     a sphere — and a contrastive embedder does not spread it over that sphere.
     It packs the whole corpus into a cap a few degrees across. Every pair of
     items on screen is at cosine **0.89 or better**: a photograph of a shelf
     and a photograph of a bicycle are, by this measure, almost the same thing. -->

<!-- **b** — Which means the number carrying nearly all of that similarity is
     the same for everybody — the direction the corpus as a whole points in. -->

<!-- **c** — So subtract it. What is left is what each item has that the corpus
     does not, and that is the only part with any information in it. -->

<!-- **d** — Renormalise, and the same pairs now span the **full range** of
     cosine. That is what k-means gets to work with, and skipping this step is
     not a small loss of quality: partitioning the raw cone means clustering on
     0.93 against 0.95, which is noise with a mean in it. -->

<!-- One consequence to have ready, because it looks like a bug. Centring makes
     the vectors sum to nothing, so the **root has no direction at all** — the
     top of the tree is degenerate by construction, not by accident. Everything
     below it is cohesive, and the typicality machinery gates on exactly that:
     a node with no concentrated direction is skipped rather than trusted. -->

<!-- The picture is three-dimensional and the shipped one is 768, which matters
     here more than usual: centring a narrow cap leaves its residues in one
     fewer dimension than you started with. Losing one of 768 costs nothing.
     Drawn on a flat circle it would be fatal, and the figure would show the
     fix destroying the data — which is why it is a sphere. -->
