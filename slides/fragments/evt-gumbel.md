<!-- _class: full -->

![bg fit](figs/calib-region-max.png)

## Extreme<br>Measures

<!-- build: figs/calib-region-max.build1.png -->

<!-- build: figs/calib-region-max.build2.png -->

<!-- build: figs/calib-region-max.build3.png -->

<!-- DINO Might ended on a maximum. This is what that maximum does to the
     distribution the line is drawn on, and the idea it suggested. -->

<!-- **a** — Two photographs, closer in, each cut into a grid of regions:
     coarser than DINOv3's patches, so a score fits in each. Let the room look
     first: left, the doll's book; right, a dog on a sofa, and no book
     anywhere. -->

<!-- **b** — Region voting again, and the number it implies: an item's score is
     the **maximum** over its regions. Left, the regions over the book score
     high, and the photo's score is the best of them. Right, no book at all —
     and the regions still differ, so this photo has a maximum too. Every item
     gets one, book or not. (The numbers are drawn to make the point, not read
     off a detector.) -->

<!-- **c** — Now do that for every item in the corpus, and look at what the
     threshold is actually applied to: how many items (up) at each similarity
     (across). Every score in this distribution is a maximum of twenty
     draws. -->

<!-- **d** — And a maximum is not a mean. It leans right, and the shape it leans
     toward is not the Gaussian. The one-line version: the Gumbel is to a
     maximum what the Gaussian is to an average. Fit the family the data
     implies and you should beat any rule that assumes a Gaussian. Note that
     even here the Gumbel is not a perfect fit — twenty draws is not "many",
     which is the first hint of how this ends. -->

<!-- The premise is principled and testable, and the sweep was pre-registered
     before any result came back. What it found belongs to the Results
     section, with the runs that are still reporting. -->
