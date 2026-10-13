<!-- _class: full -->
<!-- frames: equal -->

![bg fit](figs/ui-train-loop.webp)

## Rock the Vote

<!-- build: figs/ui-train-loop.build1.webp -->

<!-- build: figs/ui-train-loop.build2.webp -->

<!-- build: figs/ui-train-loop.build3.webp -->

<!-- build: figs/ui-train-loop.build4.webp -->

<!-- build: figs/ui-train-loop.build5.webp -->

<!-- **a** — This is training. It shows us one picture, and we hit Good or Bad.
     Our votes pile up on the right, and right now there aren't any. That strip
     on the left is Autopilot: it picks what we see next, so we never go
     hunting for something to label. First up, a suitcase full of books. Good. -->

<!-- **b** — Every vote retrains the detector, and it takes a fraction of a
     second, so there's no waiting. A stack of books. Good. -->

<!-- **c** — Now teddy bears in front of a shelf of DVDs, just like the last
     slide. The dataset these came from actually labels this one "book". I
     don't think it is, so -->

<!-- **d** — Bad. That's the last slide's argument as one click. The data says
     book, I say no, and I'm the one asking, so I win. Next, someone reading in
     bed. Good. -->

<!-- **e** — A cat watching TV. Bad. Most votes are instant like that; once in
     a while one makes you think, and those are the ones that matter. -->

<!-- **f** — Skip ahead to about thirty votes: eighteen Good, thirteen Bad, a
     few minutes of clicking. Now it's asking things like this: a desk with a
     laptop, a pile of cables, and one book. The easy ones are done, and what's
     left is what the detector can't call yet. -->
