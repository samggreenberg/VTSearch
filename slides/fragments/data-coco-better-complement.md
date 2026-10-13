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

<!-- **a** — Here's why "complete" matters. Three classes; each circle is every
     image with that class in it, whatever else is there. -->

<!-- **b** — The cheap experiment: positives are the A circle, negatives
     everything outside. Costs nothing, and the numbers come out beautiful. -->

<!-- **c** — Same for B. -->

<!-- **d** — And C. Three easy wins, and none of them ever had to tell an A from
     a B. A detector that fires on ANY of the three aces all three tests. -->

<!-- **e** — Complete labels add the equals sign: not "there's an A here" but
     "an A and a B and nothing else." Seven exact cells. Spotty labels can't say
     that; silence isn't a no. -->

<!-- **f** — And exact cells give us not-A, every image without one. You can
     only draw that if absence is actually labeled. -->

<!-- **g** — The hard test: same positives, honest negatives, nobody sits out.
     Now the line has to fall between classes, and the any-of-the-three detector
     fails. That's the benchmark worth building. -->

<!-- **h** — Back to the easy test: that same detector aces this one and fails
     the hard one. -->
