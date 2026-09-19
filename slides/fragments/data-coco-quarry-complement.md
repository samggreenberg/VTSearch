<!-- _class: full -->
<!-- frames: equal -->

![bg fit](figs/dataset-coco-quarry-complement.png)

## Venn Nothing<br>Isn't Enough

<!-- build: figs/dataset-coco-quarry-complement.build1.png -->

<!-- build: figs/dataset-coco-quarry-complement.build2.png -->

<!-- build: figs/dataset-coco-quarry-complement.build3.png -->

<!-- build: figs/dataset-coco-quarry-complement.build4.png -->

<!-- build: figs/dataset-coco-quarry-complement.build5.png -->

<!-- build: figs/dataset-coco-quarry-complement.build6.png -->

<!-- Why the set moved to COCO, in one slide. Three classes here; COCO has
     eighty of them, and answers for every one of them on every image it
     touches. Everything below is a consequence of that one property. -->

<!-- **a** — A circle is every image holding that class, whatever else it
     also holds: that is the plus. Outside all three is the empty set —
     images with none of them. -->

<!-- **b** — The cheap experiment. Positives are the A circle, negatives are
     the outside, and it costs nothing to build. The numbers come out
     beautiful. -->

<!-- **c** — Same set, rotated for B. -->

<!-- **d** — And for C. Three detectors, three easy wins — and not one of
     them was ever asked to tell an A from a B, because nothing in the
     crescents is shown to anybody. A single detector that fires on **any**
     of the three takes all three of these. -->

<!-- **e** — What exhaustive annotation adds is the equals sign: not "an A is
     here" but "an A and a B are here, and nothing else is". Seven cells,
     each an exact set. This is the sentence Visual Genome cannot say — its
     silence is not a no. -->

<!-- **f** — And it is the exact sets that buy the complement: everything
     without an A is three of those cells plus the empty one. You can only
     name that region if absence is annotated rather than inferred. -->

<!-- **g** — Same positives, honest negatives. The line now has to fall
     **between** classes, and the A-or-B-or-C detector fails it — which is
     the benchmark worth building. -->
