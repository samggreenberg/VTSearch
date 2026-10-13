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

<!-- **a** — Why "complete" matters. Three classes; each circle is every image
     with that class, whatever else is in it. -->

<!-- **b** — The cheap experiment: positives are A, negatives everything
     outside. Costs nothing, numbers come out beautiful. -->

<!-- **c** — Same for B. -->

<!-- **d** — And C. Three easy wins, and none ever had to tell an A from a B. A
     detector that fires on ANY of the three aces all three. -->

<!-- **e** — Complete labels add the equals sign: "an A and a B and nothing
     else." Seven exact cells. Spotty labels can't say that; silence isn't a
     no. -->

<!-- **f** — And exact cells give us not-A, every image without one. You can
     only draw that if absence is labeled. -->

<!-- **g** — The hard test: same positives, honest negatives, nobody sits out.
     Now the line has to fall between classes, and any-of-the-three fails.
     That's the benchmark. -->

<!-- **h** — Back to the easy test: the same detector aces this one and fails
     the hard one. -->
