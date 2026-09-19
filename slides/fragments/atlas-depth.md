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

<!-- **c** — And here is the turn: that cut has no ceiling. The shipped head is
     a **single linear layer** — the score is one number, a projection — so it
     is *exactly* constant along every direction that projection does not point
     in. Not an approximation of the model. That **is** the model. -->

<!-- **d** — So when a second collection arrives a long way up a direction
     nobody ever voted along, every one of those items comes back Good, with
     ordinary-looking scores and a threshold cut in the ordinary way. Nothing
     in the output says the word *extrapolating*. -->

<!-- **e** — And the honest alternative: the concept stops. The dome fits every
     vote on that floor exactly as well as the tube does. No amount of further
     voting separates them, because **every vote is on the floor** — that is
     what makes this different from the last two slides, where more clicks were
     the answer. -->

<!-- Which is why the fix cannot come from the detector. It has to come from
     something that models where the data *was*, and that is the atlas: it does
     not need to know the truth up there, only to say "this is nowhere near
     anything I was built on". That is the typicality p-value — and the next
     slide is about how well it says it. -->
