<!-- _class: full -->

![bg fit](figs/dataset-coco-quarry-complement.png)

## Everything<br>That Isn't

<!-- build: figs/dataset-coco-quarry-complement.build1.png -->

<!-- build: figs/dataset-coco-quarry-complement.build2.png -->

<!-- build: figs/dataset-coco-quarry-complement.build3.png -->

<!-- build: figs/dataset-coco-quarry-complement.build4.png -->

<!-- build: figs/dataset-coco-quarry-complement.build5.png -->

<!-- Why the set moved to COCO, in one slide. Three classes here; it has
     eighty. Everything on this slide is a consequence of one property —
     COCO answers for all eighty on every image it touches. -->

<!-- **a** — A circle is the images holding that class, whatever else they
     also hold. A photograph of a bus with a car in it is inside two of
     them. -->

<!-- **b** — The positives for an A-detector: the whole circle, overlaps
     included. Nothing controversial yet. -->

<!-- **c** — The negatives, the cheap way: the outside. Images holding none
     of the three. It costs nothing, and it is the best a set built by
     *designation* can do — a class was designated present, and everything
     else is silence. -->

<!-- **d** — And it is hollow. Nothing in the white ever reaches the
     detector, so it is never once asked to tell an A from a B. A detector
     that fires on **any** of the three scores exactly as well as one that
     knows what an A is, and the benchmark cannot tell them apart. -->

<!-- **e** — What exhaustive annotation adds: not "an A is here" but "an A
     and a B are here, and nothing else is". Seven cells, each an exact set.
     This is the sentence VG cannot say — its silence is not a no — and the
     reason the migration is worth its cost. -->

<!-- **f** — So the negatives become everything that is not an A: B's and
     C's included, the empty images with them. Same positives, but the line
     now has to fall **between** classes. The A-or-B-or-C detector fails it,
     which was the point. -->
