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

<!-- **a** — A room, and every item anybody has ever voted on lying on its
     floor. The floor is not the space — it is the part of the space this
     corpus happened to occupy. -->

<!-- **b** — The detector, cutting the floor, exactly as on the earlier
     slides. Nothing new yet. -->

<!-- **c** — A whole second collection, a long way up a direction nobody ever
     voted along — not a few odd items, a domain. What does the cut say about
     *these*? -->

<!-- **d** — Here is the answer everybody pictures without noticing they have.
     The concept stops. It has bounded extent, it sits where the votes were,
     and whatever is up here is simply outside it. Every vote on that floor is
     consistent with this. -->

<!-- **e** — And here is what actually ships. The head is a **single linear
     layer** — the score is one number, a projection — so it is *exactly*
     constant along every direction that projection does not point in. Not an
     approximation of the model. That **is** the model. The cut has no
     ceiling. It **sorts** the new collection, cleanly, into the ones inside
     and the ones outside. Every one of those verdicts is a
     copy, item for item, of one about a different collection. Ordinary scores,
     an ordinary ranking, an ordinary cut. Nothing in the output says the word
     *extrapolating*. -->

<!-- Nothing here chooses between those last two frames, because **every vote
     is on the floor** — which is what makes this unlike the last two slides,
     where more clicks were the answer. So the fix cannot
     come from the detector, but from something that models where the data
     *was* — the atlas: it does not need to know the truth up there, only to
     say "this is nowhere near anything I was built on". That is the typicality
     p-value, and the next slide is about how well it says it. -->
