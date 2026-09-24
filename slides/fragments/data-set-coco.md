<!-- _class: full -->
<!-- frames: equal -->

![bg fit](figs/data-set-coco-zoom.webp)

## Data, Set

<!-- build: figs/data-set-caltech.webp -->

<!-- build: figs/data-set-coco-grid.webp -->

<!-- One card, and every dataset in the talk arrives on it: its name under
     the headline, counts, where to get it, and what the pictures look like.
     Click to the next. -->

<!-- **a** — Caltech-101. One object, centred, filling the frame, on a
     background that is usually nothing; 101 categories, one per picture, and
     no boxes at all. It is the control: the set where region voting has
     nothing to point at, so a result that holds here and on cluttered scenes
     is not a result about clutter. (8,677 images exactly, leaving out the
     `BACKGROUND_Google` pile the paper shipped as a negative class.) -->

<!-- **b** — COCO. Rooms and streets, many things in each, and every one of them
     boxed. 123,287 images exactly, across train and val, and 80
     classes. -->

<!-- **c** — The zoom, bigger than its cell in the grid. Blue names are in the
     picture, each joined to its box.
     Grey names are not — and that is the property the rest of this section
     leans on: COCO answers for all 80 classes on every image, so a grey name
     is a *checked* absence, not an unmentioned one. There is no mouse on
     this desk, and COCO says so. -->

<!-- Not a gold standard: two prohibition circles and a school-crossing
     paddle labelled `stop sign`, a box on a hedge labelled `umbrella`. Better
     than anything else available, and still made by people. -->
