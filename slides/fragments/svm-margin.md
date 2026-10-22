<!-- _class: full -->

![bg fit](figs/svm-margin.webp)

## Marginal<br>at Best

<!-- build: figs/svm-margin.build1.webp -->

<!-- build: figs/svm-margin.build2.webp -->

<!-- build: figs/svm-margin.build3.webp -->

<!-- build: figs/svm-margin.build4.webp -->

<!-- **a** — So what is that boundary, really? Some votes again, Goods and
     Bads, flattened onto a plane. -->

<!-- **b** — Every one of these lines gets every vote right, so the votes alone
     can't pick one. -->

<!-- **c** — An SVM, a support vector machine, takes the one with the widest
     street between them. Only the votes on the curb set it: move any other
     vote and nothing changes. In 768 dimensions it's a flat sheet, not a line,
     but it's the same idea. -->

<!-- **d** — Now keep just the line, and bring in the photos nobody voted
     on. -->

<!-- **e** — Each one gets a score: which side of the line it's on, and how
     far, straight across to it. That's one multiply-and-add per number, one
     dot product a photo. Near the line it can't tell, and that's where it'll
     ask. -->
