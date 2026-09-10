<!-- _class: full -->

![bg fit](figs/embed-flow.png)

## Embed-time Stories

<!-- build: figs/embed-flow.build1.png -->

<!-- build: figs/embed-flow.build2.png -->

<!-- build: figs/embed-flow.build3.png -->

<!-- The sentence the deck has been assuming and never said. Slide 3 had the
     user type one word, and nothing since has explained why a *word* is
     allowed to seed a ranking over *pictures*. This slide is the answer, and
     it is the only machine-learning claim the talk actually needs. -->

<!-- **a** — Where it starts, and it is what they have already been shown: a
     pile of photographs. Nothing has been done to them yet. -->

<!-- **b** — Every one of them goes through **SigLIP** once, at import, and
     comes out as a point in a 768-dimensional space. That is the only time the
     heavy network runs — which is why a vote retrains the detector in a
     fraction of a second, something they watched happen four slides ago
     without being told why. The cube is a lie of scale and worth naming as
     one: 768 dimensions do not fit on a slide, so three stand in for them.
     What survives the lie is the part that matters — items a person would call
     similar come out near each other, and nothing but the network decided
     that. -->

<!-- **c** — Now the other input, and the word on the arrow is the same word.
     The phrase the user typed goes through the **same network** — one model,
     two kinds of input — which is the property the whole seed depends on and
     the one thing on this slide worth saying slowly. -->

<!-- **d** — And it lands *in there*, with the pictures. That is the answer to
     "why does typing `book` do anything at all": the phrase is a point in the
     same space as the photographs, so "which items are near it?" is a question
     with an actual answer, and that ranking is what the first question is
     drawn from. Be honest about the drawing, though — the square is at the
     middle of the cube because the middle claims nothing. Where it *really*
     sits, and which items really fall near it, is the next twenty minutes. -->

<!-- If someone asks: no, the detector is not this. This is the seed, used once
     — from the next slide on the votes take over and the phrase stops
     mattering. -->
