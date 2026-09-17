<!-- _class: full -->

![bg fit](figs/dataset-vg-scale-cells.png)

## No Variation<br>to Plot

<!-- Every tile on this grid is the same shape: 100 positives, the same 9,900
     shared negatives, the same 1% prevalence. There is no variation to plot,
     and that *is* the design. -->

<!-- Why it has to be that way: if prevalence differed between bands, a
     small-versus-large difference would be part size and part difficulty,
     and nobody could say which. Two earlier waves of the overview benchmark
     were made non-comparable by exactly that, which is why it is pinned by
     construction rather than checked afterwards. -->

<!-- The part that surprises people: a cell is **designated**, not inferred.
     It is exactly its hundred positives plus the shared pool, and every
     other image in the pickle is *excluded* from it — not counted as a
     negative. An image holding a large bus is not a `bus@small` negative,
     and scoring it as one would mark a detector wrong for finding a real
     bus. -->

<!-- If asked about the numbers moving: the pool went from 3,900 to 9,900
     when the negatives were made provable, and the class list from twelve to
     twenty-five. Both are why this figure is generated from the config the
     pile builds against rather than typed onto a slide. -->
