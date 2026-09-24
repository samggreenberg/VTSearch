<!-- _class: full -->
<!-- frames: equal -->

![bg fit](figs/data-set-quarry-zoom.webp)

## Data, Set

<!-- build: figs/data-set-quarry-grid.webp -->

<!-- **COCO Quarry**: what we have. All of COCO 2017, 49 classes after the
     merges, and 144 cells — a class at a size. No web address at the foot:
     we built it, so there is nowhere to go and get it. Three are missing: small
     bananas, apples and oranges, where no honest supply reaches 100. -->

<!-- **a** — The grid. Each picture is one positive, and it carries exactly one
     region: the class's largest instance, the box the simulated user drags.
     The label under it is its cell. -->

<!-- **b** — The zoom, larger than the grid had room for. The dog@medium
     frame again. Blue: the quarry's classes in this picture, one region each.
     Grey: classes checked absent — a longer list now there is room for it. The teddy
     bear has no box at all — it is not a quarry class. -->

<!-- Why "quarry": every image is embedded once, so a new prevalence, a
     per-class pool, or a different mix of sizes is a filter over the corpus
     rather than a rebuild. It is a bench for comparing **methods**; the
     classes and bands are strata to report across. -->
