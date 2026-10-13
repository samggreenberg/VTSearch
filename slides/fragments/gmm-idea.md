<!-- _class: full -->

![bg fit](figs/calib-gmm-flow.png)

## Oops! All<br>Haystack

<!-- build: figs/calib-gmm-flow.build1.png -->

<!-- build: figs/calib-gmm-flow.build2.png -->

<!-- build: figs/calib-gmm-flow.build3.png -->

<!-- build: figs/calib-gmm-flow.build4.png -->

<!-- **a** — What that left out: the haystack. Every photo in the dataset,
     unlabeled. I'll call it the corpus. Unlabeled doesn't mean there are no
     books in it, just that we don't know which ones. -->

<!-- **b** — Our votes are a little sample out of it, and M₀ trains on them,
     same as before. -->

<!-- **c** — Now M₀ scores the whole corpus: thousands of scores instead of a
     few dozen, and they fall into two humps on their own. -->

<!-- **d** — Fit a bell curve to each hump. See the question marks? Calling the
     high one Good is a guess; nothing here checked it. -->

<!-- **e** — Then cut halfway between them, without ever looking at a vote.
     Reading the corpus stuck around; the halfway cut didn't. When under half a
     percent of photos are matches, it keeps way more than there are. -->
