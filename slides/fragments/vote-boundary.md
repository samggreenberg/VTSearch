<!-- _class: full -->

![bg fit](figs/vote-boundary.png)

## Rock the Vote

<!-- build: figs/vote-boundary.build1.png -->

<!-- build: figs/vote-boundary.build2.png -->

<!-- build: figs/vote-boundary.build3.png -->

<!-- build: figs/vote-boundary.build4.png -->

<!-- build: figs/vote-boundary.build5.png -->

<!-- build: figs/vote-boundary.build6.png -->

<!-- build: figs/vote-boundary.build7.png -->

<!-- build: figs/vote-boundary.build8.png -->

<!-- The figure is the slide; the words are yours. Say first that this is a
     drawing, not a plot: the real detector is a linear SVM in a few hundred
     dimensions, where the boundary is a flat plane. Squeeze that onto a page
     and it curves. Every curve here is still a real fit to the votes on
     screen. -->

<!-- **a** — One circle per item: a photo, a clip, a document. Nothing is
     labelled and nothing is known. This is the pile, drawn. -->

<!-- **b** — Five Good, five Bad: a couple of minutes of clicking. Ten of the
     circles are now marks and the other eighty-six are exactly as they were.
     That ratio is the situation, and it is worth a beat before anything is
     fitted to it. -->

<!-- **c** — Already enough to fit a detector. The curve is everything the
     model now calls a match. Note what it enclosed — mostly circles, not
     checks: it has generalised from ten answers to a hundred items. -->

<!-- **d** — Two items no one should be asked about: one deep inside, one far
     outside. The model would bet on either and be right, so a vote spent on
     either buys nothing. This is the trap in "just show them the top of the
     ranking" — the top is exactly where the model is already sure. -->

<!-- **e** — So it asks about this one instead: right on the line, where the
     model genuinely cannot call it. The least comfortable item to be shown and
     the most valuable one to answer. -->

<!-- **f** — The user says Good. Retrain, and the boundary is somewhere else;
     dashed is where it was. Say that the other branch is just as real: a Bad
     there would have pulled the curve *in* on that side. One answer, and the
     model's opinion about a dozen items it has never seen has changed. -->

<!-- **g** — And the loop closes. The boundary moved, so a *different* item is
     now the one on it, and that is the next thing the user sees. -->

<!-- **h** — Now the other job. Same detector, cut looser and cut tighter. Both
     dashed curves still keep every check in and every cross out, so nothing
     the user has said can choose between them — but look inside the strip:
     those items come back under one cut and not the other. Where the line goes
     is the user's call, not the data's. -->

<!-- **i** — And here is the twist. Go back to the moment it chose a question:
     the same three cuts were available *then*. Which one you were on decides
     which item sits nearest the line — so the threshold did not only pick what
     came back at the end, it picked the question that got you here. Land on
     that: get the line wrong and you do not just return the wrong set, you
     spend the user's next twenty votes on the wrong questions. -->
