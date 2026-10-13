<!-- _class: full -->
<!-- frames: equal -->

![bg fit](figs/coco-problems-pile.webp)

## Picture<br>(im)Perfect

<!-- build: figs/coco-problems-blank.webp -->

<!-- build: figs/coco-problems-merge.webp -->

<!-- build: figs/coco-problems-cuts.webp -->

<!-- build: figs/coco-problems-largest.webp -->

<!-- **a** — So COCO's complete, but not ready to use. Four problems first. -->

<!-- **b** — One: the same thing under two names. A pickup's a car in one photo
     and a truck in the next, so we merge those pairs. -->

<!-- **c** — Two: we want lots of datasets from one pile, books common or
     rare, small or large. So a dataset is a filter over the photos, not a
     rebuild. -->

<!-- **d** — Three: a size needs one box per photo, and a desk holds lots of
     books. The largest is what a user would box, so it sets the size. -->

<!-- **e** — Four: a pile isn't large. Both are COCO's biggest book box: one
     book, versus a whole shelf. So we check against a second labeling that
     boxes books one at a time. -->
