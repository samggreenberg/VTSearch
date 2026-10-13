<!-- _class: full -->
<!-- frames: equal -->

![bg fit](figs/coco-problems-pile.webp)

## Picture<br>(im)Perfect

<!-- build: figs/coco-problems-blank.webp -->

<!-- build: figs/coco-problems-merge.webp -->

<!-- build: figs/coco-problems-cuts.webp -->

<!-- build: figs/coco-problems-largest.webp -->

<!-- **a** — So COCO answers for every class on every image, but it still isn't
     ready to use. There are four problems we had to fix first. -->

<!-- **b** — One: the same thing under two names. A pickup is a car in one
     photo and a truck in the next. A beer glass is a cup or a wine glass.
     Flowers in a glass are a vase or a potted plant. The labelers didn't draw
     these lines the same way every time, so we merge each pair into one class. -->

<!-- **c** — Two: we want lots of datasets out of one set of photos. Books
     common or rare, small or large. So a dataset has to be a rule over the
     photos, not a hand-picked list, and then each of these is just a filter,
     not a rebuild. -->

<!-- **d** — Three: a size needs one box per photo, and a desk can hold a lot
     of books. Box them all together and you're measuring the desk. The one a
     user would draw a box around is the obvious one, the largest, so that's
     the one that sets the size. -->

<!-- **e** — Four: a pile is not large. Both of these are COCO's largest book
     box, and both count as large. On the left that's right: one book, its
     cover filling the photo. On the right it's a whole shelf under one box,
     and not one book in it is large. So we check against a second labeling of
     the same photos that boxes books one at a time, and a box around a pile
     doesn't count. -->
