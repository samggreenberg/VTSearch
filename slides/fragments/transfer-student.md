<!-- _class: full -->

![bg fit](figs/calib-transfer.png)

## Transfer<br>Student

<!-- build: figs/calib-transfer.build1.png -->

<!-- build: figs/calib-transfer.build2.png -->

<!-- build: figs/calib-transfer.build3.png -->

<!-- Everything so far drew the line in Train. Find is the other half of the
     app: the same detector, pointed at a pile nobody voted on, and the line
     has to be drawn there too. -->

<!-- **a** — Train, as What to Expect left it: the votes, the corpus they came
     from, and the line at the middle radio, keeping 227. Goods and Bads on
     their own rows now, at half the width. -->

<!-- **b** — What goes to Find: the labels, each with where it was found, and
     that is all. Not the Train corpus, not its scores, not its line; the line
     was a fact about that corpus, and the model is retrained from the labels
     rather than carried. -->

<!-- **c** — What Find has on its side: that model's scores on the new corpus.
     Its own distribution, here half again as many items and a third the share
     of matches. -->

<!-- **d** — The same labels and the same fit, drawn on this corpus. The line
     moves, and keeps 116, near the 120 matches really there. Because the
     labels are all that travel, they are all an exported labelset needs: open
     another corpus, even under another embedder, and Find re-fits it from the
     same votes. Train and Find draw the line one way. -->
