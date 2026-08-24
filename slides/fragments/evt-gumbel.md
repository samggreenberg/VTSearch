<!-- _class: full -->

![bg fit](figs/calib-region-max.png)

## Extreme Measures

<!-- build: figs/calib-region-max.build1.png -->

<!-- build: figs/calib-region-max.build2.png -->

<!-- So the axis had something on it, and the next idea on it was the ambitious
     one. It needs three things said first, because the deck has not needed any
     of them since the tool slides. -->

<!-- **a** — Region voting. Instead of "is this item a match?", the user draws
     a box: *which part of it*. A whole-item Good on a mostly-irrelevant image
     is a mostly-wrong training signal; a box says where the evidence is. The
     consequence is the number at the bottom — an item's score is the
     **maximum** over its regions, not one number from one vector. -->

<!-- **b** — Now do that for every item in the corpus, and look at what the
     threshold is actually being applied to. Every score in this distribution
     is a maximum of a dozen draws. -->

<!-- **c** — And a maximum is not a mean. It leans right, and the shape it
     leans toward is not the Gaussian. The one-line version, if the room needs
     it: the Gumbel is to a maximum what the Gaussian is to an average — the
     shape you converge on when you take enough draws. That is the whole
     extreme-value family, and the Gumbel is its workhorse. Fit the family
     the data actually implies and you should beat any rule that assumes a
     Gaussian. Note that even here the Gumbel is not a perfect fit: a dozen
     draws is not "many", which is the first hint of how this ends. The premise
     is principled and testable, and the sweep was pre-registered before any
     result came back — which is what makes the next slide's answer mean
     something. -->
