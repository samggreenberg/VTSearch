<!-- _class: full -->

![bg fit](figs/logo-balance.png)

## The Fine<br>Print

<!-- build: figs/logo-balance.build1.png -->

<!-- build: figs/logo-balance.build2.png -->

<!-- build: figs/logo-balance.build3.png -->

<!-- **a** — The three radios again, now on documents. On photos the radio just
     moves the cut. Here the line is a count of inliers with rules on it, so
     each radio gets its own rule. This is one detector partway through a
     session. Top line: the votes, Goods above and Bads below, placed by how
     many inliers each gets. Bottom: the pages, real copies of the mark above,
     wrong pages below. See how the faint copies sit as low as the hard wrong
     pages? That's the whole problem. -->

<!-- **b** — The middle radio, beta of 1, is the rule from the last slide: beat
     every Bad. That keeps 11 pages, 1 of them wrong, and the faint copies
     under the top Bad are lost. -->

<!-- **c** — Beta of 4, for when a miss hurts more than a wrong one: also keep
     any page that gets a quarter of the Goods' typical count, at least 10,
     whatever the Bads say. Now it keeps 18, 5 of them wrong. It's always the
     middle set plus more, so turning the radio that way only adds pages. -->

<!-- **d** — Beta of a quarter, for when a wrong one hurts more: take the
     middle set and keep only the pages that beat every Bad by at least 5. Nine
     kept, none wrong. And since that's always a piece of the middle set, the
     radio stays in order both ways. -->
