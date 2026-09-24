<!-- _class: full -->

![bg fit](figs/dataset-coco-quarry-cells.png)

## No Variation<br>to Plot

<!-- Every tile on this grid is the same shape: 100 positives against the
     same shared pool of 9,900 images that hold none of the 49 classes. There
     is no variation to plot, and that *is* the design. -->

<!-- Why it has to be that way: if prevalence differed between bands, a
     small-versus-large difference would be part size and part difficulty,
     and nobody could say which. So it is pinned by construction rather than
     checked afterwards. The three dotted tiles are the small fruit — not
     built short, because a thin cell's AP would not be comparable with any
     other. -->

<!-- The negatives are not only the barren pool. An image that is a positive
     for another class, and holds none of this one, is a negative here too —
     a picture of a street is a fair negative for `bird`. Without those, every
     negative would be an empty scene and every positive a busy one, and a
     detector could score by noticing clutter. -->

<!-- The part that surprises people: an image holding a **large** bus is not
     a `bus@small` negative. It is excluded from that cell, because scoring it
     as a negative would mark a detector wrong for finding a real bus. And
     since a class's three bands share one pool, a detector trained on small
     buses can be scored on large ones. -->

<!-- Why "quarry": the 144 cells are one block cut to one spec. All 123,287
     images are embedded, so another prevalence, a per-class pool of ~100,000
     negatives, or a different mix of sizes to train on is a filter over the
     corpus rather than a rebuild — and a cut at the standard spec reproduces
     these cells exactly. It is a bench for comparing **methods**: the
     classes and bands are strata to report across, not the thing under
     study. -->
