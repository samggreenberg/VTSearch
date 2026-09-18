<!-- _class: full -->
<!-- frames: equal -->

![bg fit](figs/ui-train-loop.webp)

## Rock the Vote

<!-- build: figs/ui-train-loop.build1.webp -->

<!-- build: figs/ui-train-loop.build2.webp -->

<!-- build: figs/ui-train-loop.build3.webp -->

<!-- build: figs/ui-train-loop.build4.webp -->

<!-- build: figs/ui-train-loop.build5.webp -->

<!-- **a** — Train, one second after opening it, and the interaction is already
     complete: one item in the middle, **Good** and **Bad** under it, the answers
     so far — none — on the right. The sliver on the far left is autopilot,
     folded to a rail: it picks what to put in front of you, so nobody is
     scrolling a result list deciding what to judge. -->

<!-- **b** — First answer, and it lands in the Good pile. Behind it, the head
     retrains and all 228 items re-rank. That is a fraction of a second, because
     it is a small linear model on frozen embeddings — the heavy network ran
     once, at import, and never runs again. -->

<!-- **c** — Second. Watch the middle as much as the right: the item changed,
     because the model that just retrained went looking for what it could least
     call. Which is why it is showing you three teddy bears in front of a shelf
     of DVD box sets — **the frame from slide 2**, the one the room could not
     agree about. Not an accident: it is the nearest thing this corpus has to
     the edge of the concept. -->

<!-- **d** — Third, and the first **No**. Box sets are not books to the person
     doing the asking. Say it out loud — it is the whole argument of slide 2
     arriving as a click: nothing in the data settled that, COCO's own
     annotators called those spines books, and nothing had to. The concept ends
     where this user says it ends, and here is the only place they get to say
     so. -->

<!-- **e** — Fourth, and back to a Good: someone reading in bed, which nobody
     needs to think about. That is the rhythm — most answers instant, the
     occasional one the whole point — with no second mode to learn and nothing
     to configure between answers. -->

<!-- **f** — Twenty-eight questions in — twelve Good, sixteen Bad, a few
     minutes, the whole budget this task was ever going to get. Note what it
     asks *now*: a man on an exercise bike, a bookcase over his shoulder. The
     obvious ones are settled, so what is left is the frames where the books are
     *present* and the picture is not about them — exactly the line the rest of
     the talk is about. -->

<!-- If someone asks where the rest of the corpus went: there is a manual mode
     with the whole pile in a grid, sort controls and a threshold slider. That
     slider is a character in the second half of the talk. -->
