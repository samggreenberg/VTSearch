<!-- _class: full -->

![bg fit](figs/atlas-cells.png)

## Cover Story

<!-- build: figs/atlas-cells.build1.png -->

<!-- build: figs/atlas-cells.build2.png -->

<!-- build: figs/atlas-cells.build3.png -->

<!-- build: figs/atlas-cells.build4.png -->

<!-- build: figs/atlas-cells.build5.png -->

<!-- **a** — Same collection, and now the only question is where its parts are.
     Nothing here is about the detector. -->

<!-- **b** — Split it in three. Ordinary k-means, k = 3, on the embeddings the
     items already carry — no new model, nothing trained. -->

<!-- **c** — Then split the parts, and keep going until a cell is too small to
     divide — twenty items in the shipped build. A few thousand cells for a
     corpus of tens of thousands, and it is cached in the dataset pickle, so it
     is built once and not once per session. -->

<!-- **d** — Every vote marks the cell it landed in **and every cell above it**,
     which is the whole state the thing keeps: one pair of counters per node,
     good and bad. A cell is *covered* the moment either is non-zero. -->

<!-- **e** — So "where have I not been?" is a walk: breadth-first from the top,
     biggest sibling first, stop at the first cell carrying nothing. Biggest
     first is the only tuning in it, and it buys the most collection per click. -->

<!-- **f** — Then the good bit. Don't show the typical item of that cell — show
     the one that could prove the model wrong about it. The cell's median score
     says what the model presumes: above the threshold it is a good region, so
     serve its **lowest**-scored item; below, serve its highest. -->

<!-- Informative whichever way the answer lands, which is the part worth saying
     slowly. If it flips, the user has found a pocket the model had backwards —
     the best click in the session. If it doesn't, the region's presumption has
     been tested where it was weakest, and nothing else in the cell was more
     likely to break it. -->

<!-- One refinement, if asked: the extremum is taken over the cell's *typical*
     half. An extreme score on an atypical item is usually a lone oddball — a
     corrupt file, a strange crop — and its flip says nothing about the region. -->
