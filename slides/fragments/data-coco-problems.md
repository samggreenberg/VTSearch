<!-- _class: full -->
<!-- frames: equal -->

![bg fit](figs/coco-problems-pile.webp)

## Picture<br>(im)Perfect

<!-- build: figs/coco-problems-blank.webp -->

<!-- build: figs/coco-problems-merge.webp -->

<!-- build: figs/coco-problems-cuts.webp -->

<!-- build: figs/coco-problems-largest.webp -->

<!-- **a** — So COCO's complete, but it still isn't ready to use. Four problems
     we had to fix first. -->

<!-- **b** — One: the same thing under two names. A pickup is a car in one
     photo and a truck in the next. The labelers don't agree, so we merge each
     pair into one class. -->

<!-- **c** — Two: we want lots of datasets from one set of photos, books common
     or rare, small or large. So a dataset is a rule over the photos, and each
     of these is a filter, not a rebuild. -->

<!-- **d** — Three: a size needs one box per photo, and a desk holds lots of
     books. The one a user would box is the obvious one, the largest, so that
     sets the size. -->

<!-- **e** — Four: a pile is not large. Both are COCO's largest book box. Left,
     one book filling the photo; right, a whole shelf under one box. So we
     check against a second labeling that boxes books one at a time, and a pile
     doesn't count. -->
