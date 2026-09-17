<!-- _class: full -->

![bg fit](figs/dataset-card-vg-box.png)

## The Bands That<br>Came First

<!-- The first attempt at "does the size of the thing change how hard it is
     to find?", and it is here because it is instructive rather than because
     it worked. -->

<!-- The good half. It is drawn from the *whole* Visual Genome source — all
     108K images and the full free-text vocabulary — not the demo
     pipeline's hundred curated categories. That matters more than it
     sounds: the curated vocabulary puts **five** categories below one model
     patch, and the full source puts **643** there. A vocabulary chosen for
     being recognisable is not a sample of sizes. -->

<!-- The half that sank it. A category goes in a band by its own *median*
     box, so each category lands in exactly one band, so the three sets carry
     **disjoint vocabularies**. Small is `nose`, `glasses`, `watch`. Large is
     `fence`, `hill`, `lady`. Whatever the difference between them measures,
     it is box size and class identity at once, and nothing separates the
     two. -->

<!-- So these three are perfectly valid for what they measured and are not
     comparable to what comes next. `vg_scale` exists precisely to hold the
     word fixed and vary only the size — which is the whole of the next
     slide. -->

<!-- If asked about the "38 of 40": band membership was assigned on the
     median voted area over all of VG, and the check recomputes it on the
     12K-image sample, so the strays are a measurement difference rather
     than a mis-filing. -->
