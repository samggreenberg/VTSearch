<!-- _class: full -->

![bg fit](figs/vts-loop.png)

## In the Loop

<!-- build: figs/vts-loop.build1.png -->

<!-- build: figs/vts-loop.build2.png -->

<!-- build: figs/vts-loop.build3.png -->

<!-- build: figs/vts-loop.build4.png -->

<!-- build: figs/vts-loop.build5.png -->

<!-- build: figs/vts-loop.build6.png -->

<!-- In the audience deck this slide is a seven-page build sharing one page
     number: the diagram assembles a step per advance, and this page — the
     complete loop — is where it lands. There are no bullets by design; the
     figure is the slide. Keep advancing as you narrate.

     Stage by stage. The grey bar is everything the user has, and it is
     unlabeled — that is the situation from two slides ago, drawn. The box
     under it is the detector: a small linear head trained on whatever votes
     exist so far, sitting on top of frozen embeddings. It scores every item
     in the bar, which gives the number line: every item in the corpus,
     ordered, with no colours on it, because at this point nobody knows which
     of them are matches.

     Now the line. Somewhere on that axis is a cut, and this is the object the
     rest of the talk is about. Above it: match. Below it: not. Nothing about
     the ranking tells you where it goes — a perfect ranking with a bad cut
     still looks broken to the user, because what the user sees is never the
     ranking, it is what came back.

     Then the point of the slide, and the reason it is worth a build rather
     than a bullet. Follow the two arrows out of the cut. The first goes where
     everyone expects: to the answer — what gets kept, exported, counted. The
     second goes back into the loop, because the cut is also what decides
     which item to put in front of the user next: the interesting items to ask
     about are the ones near the line. Get the line wrong and you are not just
     returning the wrong set at the end, you are spending the user's next
     twenty votes on the wrong questions.

     Close on the shape: the vote goes into the pile, the head retrains, and
     the whole thing goes round again — a few times a second, a hundred and
     fifty times a session. The threshold is inside that cycle, not after it.
     That is why a talk about one number is worth your time. -->
