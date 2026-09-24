<!-- _class: full -->
<!-- frames: equal -->

![bg fit](figs/coco-problems-pile.webp)

## Not Yet<br>a Quarry

<!-- build: figs/coco-problems-merge.webp -->

<!-- build: figs/coco-problems-cuts.webp -->

<!-- build: figs/coco-problems-largest.webp -->

<!-- COCO answers every class on every image. Four things it still does not
     hand you, one per click. -->

<!-- **a** — The same object under two names. A pickup is a `car` in one
     photo and a `truck` in the next; a beer glass is a `cup` or a
     `wine glass`; flowers in a glass are a `vase` or a `potted plant`.
     Region voting drags a box, so what matters is the box: 6–11% of these
     classes' boxes carry the other name. Where the line does not hold, the
     two become one class. -->

<!-- **b** — We want many datasets out of one set of pictures: common or
     rare, small or large. So a dataset has to be a rule over the images,
     not a hand-picked list — then any of these is a filter, not a rebuild. -->

<!-- **c** — A size band needs one size per image, and an image holds many
     cars. Boxing all of them measures the street. The one a user would drag
     is the most obvious one, the largest — so that is the size. -->

<!-- **d** — COCO sometimes draws one box round a pile: a stalk of bananas,
     a shelf of books. By area that is "large", but no single banana is. A
     second annotation of the same images, LVIS, boxes one object at a time,
     and a pile box is not admitted as a positive. -->
