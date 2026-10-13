<!-- _class: full -->

![bg fit](figs/calib-gmm-flow.png)

## Oops! All<br>Haystack

<!-- build: figs/calib-gmm-flow.build1.png -->

<!-- build: figs/calib-gmm-flow.build2.png -->

<!-- build: figs/calib-gmm-flow.build3.png -->

<!-- build: figs/calib-gmm-flow.build4.png -->

<!-- **a** — Here's what that left out: the haystack. Before anybody votes,
     there's every photo in the dataset, unlabeled. I'll call it the corpus.
     It's way bigger than our votes, and unlabeled doesn't mean there are no
     books in it. It means we don't know which ones they are. -->

<!-- **b** — Our votes are a little sample pulled out of it, and we train M₀ on
     them, same as before. -->

<!-- **c** — Now run it the other way: M₀ scores the whole corpus. That's
     thousands of scores instead of a few dozen, and look, they fall into two
     humps all on their own. -->

<!-- **d** — So fit a bell curve to each hump. See the question marks? We're
     calling the low one Bad and the high one Good, but nothing here actually
     checked that. -->

<!-- **e** — Then cut halfway between the two humps. That's the whole method,
     and the cut never looks at a vote. Half of this idea stuck: reading the
     whole corpus is how the app counts matches, at the end of this section.
     The halfway cut didn't. When under half a percent of the photos are
     matches, cutting halfway keeps way more than there are matches. -->
