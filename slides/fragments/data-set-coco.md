<!-- _class: full -->
<!-- frames: equal -->
<!-- bookmark: Data, Set: Caltech-101 and COCO -->

![bg fit](figs/data-set-coco-zoom.webp)

## Data, Set

<!-- build: figs/data-set-title.webp -->

<!-- build: figs/data-set-caltech.webp -->

<!-- build: figs/data-set-coco-grid.webp -->

<!-- **a** — Every result from here on is measured on somebody's photos, and
     what got labeled decides what the number means. So every dataset in this
     talk shows up on the same card: its name, a few counts, where to get it,
     and what the pictures look like. -->

<!-- **b** — Caltech-101. About nine thousand photos, each one a single object,
     centered, filling the frame, usually on a plain background. A hundred and
     one categories, one per picture, and no boxes at all. It's our control:
     the object is the whole picture, so if a result holds here and on
     cluttered scenes too, it's not just a result about clutter. -->

<!-- **c** — COCO. Rooms and streets and kitchens, lots of things in every
     picture, and every one of them boxed: about 120 thousand photos, eighty
     classes, 900 thousand boxes. And "complete" is the important word: every
     photo answers for all eighty classes. -->

<!-- **d** — Here's one up close: a museum case with Mary Poppins' umbrella in
     it. Blue names are in the picture, each with its box: a book, a bowl, the
     umbrella. Gray names aren't, and since COCO answers for every class, gray
     means somebody checked. There's no cup in this case, and COCO says so. But
     nobody boxed the paperweight, the placard or the glass hen, because they
     aren't among the eighty. Checked means checked for these classes, and only
     these. -->
