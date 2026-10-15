<!-- _class: full -->
<!-- frames: equal -->

![bg fit](figs/coco-problems-cuts.webp)

## Picture<br>(im)Perfect

<!-- build: figs/coco-problems-blank.webp -->

<!-- build: figs/coco-problems-merge.webp -->

<!-- build: figs/coco-problems-pile.webp -->

<!-- build: figs/coco-problems-largest.webp -->

<!-- **a** — So COCO's complete, but not ready to use. Four problems, in the
     order we fix them. -->

<!-- **b** — One: the same thing under two names. A pickup's a car in one photo
     and a truck in the next, so we merge those pairs. -->

<!-- **c** — Two: a pile isn't large. Both are COCO's biggest book box: one
     book, versus a whole shelf. So we check against a second labeling that
     boxes books one at a time, and a box with a pile in it doesn't count. -->

<!-- **d** — Three: a size needs one box per photo, and a desk holds lots of
     books. The largest is what a user would box, so it sets the size. -->

<!-- **e** — Four: now every photo has its box, we want lots of datasets from
     one pile, books common or rare, small or large. So a dataset is a filter
     over the photos, not a rebuild. -->
