<!-- _class: full -->

![bg fit](figs/atlas-depth.png)

## Out of Its Depth

<!-- build: figs/atlas-depth.build1.png -->

<!-- build: figs/atlas-depth.build2.png -->

<!-- build: figs/atlas-depth.build3.png -->

<!-- build: figs/atlas-depth.build4.png -->

<!-- The atlas has a second job, and this is what it is for. Point a detector
     at a collection it was not trained on and it does not hesitate, decline,
     or look uncertain. It scores. -->

<!-- **a** — Nothing new yet: the corpus, the votes and the boundary, exactly
     as the last few slides left them. -->

<!-- **b** — One move, and not one new item. That plane is the *floor* of a
     room. The floor is not the space — it is the part of the space this corpus
     happened to occupy. -->

<!-- **c** — And a whole second collection, a long way up a direction nobody
     ever voted along. Not a few odd items: the same count and the same spread
     as the floor, because it is a domain. -->

<!-- **d** — What the detector says about them. The head is a **single linear
     layer** — the score is one number, a projection — so it is *exactly*
     constant along every direction that projection misses. Not an
     approximation of the model. That **is** the model. So the cut has no lid,
     and it **sorts** the new collection, cleanly, into the ones inside and the
     ones outside. Every one of those verdicts is a copy, item for item, of one
     about a different collection. Nothing in the output says the word
     *extrapolating*. -->

<!-- **e** — And the honest alternative: the concept stops. It fits every vote
     on that floor exactly as well as the pillar does, and no amount of further
     voting separates the two, because **every vote is on the floor** — which is
     what makes this unlike the last two slides, where more clicks were the
     answer. -->

<!-- So the fix cannot come from the detector, but from something that models
     where the data *was* — the atlas: it does not need to know the truth up
     there, only to say "this is nowhere near anything I was built on". That is
     the typicality p-value, and the next slide is about how well it says it. -->
