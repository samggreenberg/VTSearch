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

<!-- **a** — Here's why "complete" matters. Three classes. Each circle is every
     image showing that class, whatever else is in it; outside all three is the
     empty set. -->

<!-- **b** — The cheap experiment: positives are the A circle, negatives are
     everything outside. It costs nothing, and the numbers come out beautiful. -->

<!-- **c** — Same for B. -->

<!-- **d** — And for C. Three easy wins, and none of these detectors ever had
     to tell an A from a B. One detector that fires on any of the three aces
     all three tests. -->

<!-- **e** — Complete labels add the equals sign: not "there's an A here," but
     "there's an A and a B, and nothing else." Seven exact cells. Spotty labels
     can't say that; silence isn't a no. -->

<!-- **f** — And exact cells give us not-A: every image without an A. You can
     only draw that region if absence is actually labeled. -->

<!-- **g** — The hard test: same positives, honest negatives, and nobody sits
     out. Now the line has to fall between classes, and the A-or-B-or-C
     detector fails. That's the benchmark worth building. -->

<!-- **h** — Back to the easy test. The A-or-B-or-C detector aces this one and
     fails the last one. -->
