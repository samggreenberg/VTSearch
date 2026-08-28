<!-- _class: full -->

![bg fit](figs/calib-split-idea.png)

## Train More,<br>Check Less

<!-- build: figs/calib-split-idea.build1.png -->

<!-- build: figs/calib-split-idea.build2.png -->

<!-- build: figs/calib-split-idea.build3.png -->

<!-- **a** — Every fold in this section has made the same quiet choice, and
     none of the last five slides said so out loud: half the votes train the
     fold model, half are held out to read its threshold from. Fifty-fifty was
     never measured. It was the obvious split, and it stayed. -->

<!-- **b** — So we measured it, and it moved. Seventy percent into Train,
     thirty into Check — the same votes, divided somewhere else. -->

<!-- **c** — At twenty votes that is four votes crossing the line: fourteen
     train the model, six place the cut. -->

<!-- **d** — And the reason it goes that way rather than the other. Early on
     the fold model is the starved thing — it is fitting a whole decision
     boundary out of ten examples — while the threshold it needs is one
     quantile of one list, and a quantile does not need many scores to sit in
     roughly the right place. So the scarce votes are worth more in Train. The
     measured curves are in the appendix; ask and I will bring them up. -->
