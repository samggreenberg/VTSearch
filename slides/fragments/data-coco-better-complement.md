<!-- _class: full -->
<!-- frames: equal -->

![bg fit](figs/dataset-coco-better-complement.png)

## Venn Nothing<br>Isn't Enough

<!-- build: figs/dataset-coco-better-complement.build1.png -->

<!-- build: figs/dataset-coco-better-complement.build2.png -->

<!-- build: figs/dataset-coco-better-complement.build3.png -->

<!-- build: figs/dataset-coco-better-complement.build4.png -->

<!-- build: figs/dataset-coco-better-complement.build5.png -->

<!-- build: figs/dataset-coco-better-complement.build6.png -->

<!-- build: figs/dataset-coco-better-complement.build7.png -->

<!-- **a** — Here's why "complete" matters. Three classes, A, B and C. Each
     circle is every image showing that class, whatever else is in it; that's
     the plus. Outside all three is the empty set: images with none of them. -->

<!-- **b** — The cheap experiment: positives are the A circle, negatives are
     everything outside. It costs nothing to build, and the numbers come out
     beautiful. -->

<!-- **c** — Same thing for B. -->

<!-- **d** — And for C. Three detectors, three easy wins, and not one of them
     ever had to tell an A from a B; the rest of the circles just sit out. A
     single detector that fires on any of the three would ace all three tests. -->

<!-- **e** — What complete labels add is the equals sign. Not "there's an A
     here," but "there's an A and a B here, and nothing else." Seven cells,
     each one exact. A dataset with spotty labels can't say that: if it doesn't
     mention a B, that's not a no. -->

<!-- **f** — And exact cells get us the complement, not-A: every image without
     an A, which is three of those cells plus the empty set. You can only draw
     that region if absence is actually labeled. -->

<!-- **g** — So here's the hard test: same positives, honest negatives, and no
     image sits out. Now the line has to fall between classes, and the detector
     that fires on A or B or C fails. That's the benchmark worth building. -->

<!-- **h** — And back to the easy test, same drawing. Compare it to the last
     one: the A-or-B-or-C detector aces this one and fails that one. -->
