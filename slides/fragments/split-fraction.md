<!-- _class: full -->

![bg fit](figs/calib-split-idea.png)

## Train More,<br>Check Less

<!-- build: figs/calib-split-idea.build1.png -->

<!-- **a** — Same picture as the last slide, with the numbers off. Every fold
     in this talk has quietly split the votes fifty-fifty: half train the
     model, half are held out to place its cut. Nobody ever measured
     fifty-fifty; it was just the obvious choice, and it stuck. -->

<!-- **b** — So we measured it under the old mixture line, and seventy percent
     into training won. Each half draws its own seventy percent from the same
     votes, so the two training sets overlap now: that lens in the middle,
     forty percent of the votes, trains both models. At twenty votes, that's
     fourteen to train and six to place the cut. Under today's line the split
     hardly matters, within a few thousandths of F-beta either way, so
     seventy-thirty stays because nothing beats it. -->
