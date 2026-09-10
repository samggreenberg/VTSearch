<!-- _class: full -->

![bg fit](figs/ui-train-loop.webp)

## Rock the Vote

<!-- build: figs/ui-train-loop.build1.webp -->

<!-- build: figs/ui-train-loop.build2.webp -->

<!-- build: figs/ui-train-loop.build3.webp -->

<!-- build: figs/ui-train-loop.build4.webp -->

<!-- build: figs/ui-train-loop.build5.webp -->

<!-- **a** — Train, one second after opening it, and the interaction is already
     complete: one item in the middle, **Good** and **Bad** under it, the answers
     so far — none — on the right. Nothing else is asked of anybody. The sliver
     on the far left is autopilot, folded to a rail: it is choosing what to put
     in front of you, so nobody is scrolling a result list deciding what to
     judge. -->

<!-- **b** — First answer, and it lands in the Good pile. Behind it, the head
     retrains and all 228 items re-rank. That is a fraction of a second, because
     it is a small linear model on frozen embeddings — the heavy network ran
     once, at import, and never runs again. -->

<!-- **c** — Second. Watch the middle as much as the right: the item changed,
     and it changed because the model that just retrained went looking for what
     it could least call. Which is why it is now showing you three teddy bears
     in front of a shelf of DVD box sets — **the frame from slide 2**, the one
     the room could not agree about. It did not come up by accident: it is the
     nearest thing this corpus has to the edge of the concept. -->

<!-- **d** — Third, and the first **No**. Box sets are not books to the person
     doing the asking, so the answer is Bad and the piles now have two halves.
     Say it out loud, because it is the whole argument of slide 2 arriving as a
     click: nothing in the data settled that — COCO's own annotators called
     those spines books — and nothing had to. The concept ends where this user
     says it ends, and the only place they get to say so is here. -->

<!-- **e** — Fourth, and back to a Good: someone reading in bed, which nobody
     needs to think about. That is the rhythm — most answers are instant and the
     occasional one is the whole point of the exercise — and there is no second
     mode to learn, no threshold to set, nothing to configure between answers. -->

<!-- **f** — And here it is twenty-eight questions in — twelve Good, sixteen
     Bad, a few minutes, which is the whole budget this task was ever going to
     get. Note what it is asking about *now*: a man on an exercise bike in a
     living room, with a bookcase over his shoulder. The obvious ones are long
     since settled, so what is left is the frames where the books are *present*
     and the picture is not about them — which is exactly the line the rest of
     the talk is about. -->

<!-- If someone asks where the rest of the corpus went: there is a manual mode
     with the whole pile in a grid, sort controls and a threshold slider. That
     slider is a character in the second half of the talk. -->
