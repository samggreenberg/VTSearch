<!-- _class: full -->

![bg fit](figs/atlas-cells.png)

## Cover Story

<!-- build: figs/atlas-cells.build1.png -->

<!-- build: figs/atlas-cells.build2.png -->

<!-- build: figs/atlas-cells.build3.png -->

<!-- build: figs/atlas-cells.build4.png -->

<!-- build: figs/atlas-cells.build5.png -->

<!-- **a** — Same photos, and forget the detector for a minute. The only
     question now is what parts this collection has. -->

<!-- **b** — Split it into three groups of similar photos. That's plain k-means
     on the points we already have; nothing new to train. -->

<!-- **c** — Then split each of those into three, and keep going until a cell
     is down to about twenty photos. We build this once per dataset, not once
     per session. -->

<!-- **d** — Every vote marks the cell it lands in, and every bigger cell
     around it. Any cell with a vote in it, Good or Bad, counts as visited.
     Those are the shaded ones. -->

<!-- **e** — So "where haven't I been?" is just a walk. Start at the biggest
     cells, go down a level at a time, biggest first, and stop at the first
     cell with no votes. That's this one. Biggest first means each click covers
     as much of the collection as it can. -->

<!-- **f** — Then which photo from that cell? Not a typical one, but the one
     most likely to prove the detector wrong. If it thinks the cell is mostly
     Good, show its lowest-scoring photo; if mostly Bad, its highest. If we
     flip it, we've found a pocket the detector had backwards. If we don't,
     that region just passed its hardest test. -->
