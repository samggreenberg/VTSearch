<!-- _class: full -->

![bg fit](figs/vote-boundary.png)

## Rock the Vote

<!-- The line to say as the figure comes up is “Every answer moves the boundary” — the
     headline names the slide, this names the move. -->

<!-- build: figs/vote-boundary.build1.png -->

<!-- build: figs/vote-boundary.build2.png -->

<!-- build: figs/vote-boundary.build3.png -->

<!-- build: figs/vote-boundary.build4.png -->

<!-- build: figs/vote-boundary.build5.png -->

<!-- A six-page build sharing one page number, and there are deliberately no
     bullets: the figure is the slide and the words are yours. Advance as you
     narrate. Say up front that this is a drawing, not a plot of anything —
     the real detector is a linear SVM in a few hundred dimensions, where the
     boundary is a flat plane; squeeze that down to a page and it curves. Every
     curve you are about to see is nonetheless a real fit to the votes on
     screen, and every item it picks out is really the one the app would pick.

     Page 1. One circle per item — a photo, a clip, a document. Nothing is
     labelled and nothing is known. This is the pile from two slides ago,
     drawn.

     Page 2. Some votes: five Good, five Bad. That is a couple of minutes of
     someone's time, and it is already enough to fit a detector — the curve is
     everything the model currently calls a match. Note what it enclosed:
     mostly circles, not checks. The model is guessing about all of those, and
     the whole game is which guess to check next.

     Page 3. Here are two of them, and they are the ones you would *not* ask
     about. Sitting well inside, surrounded by Goods, nothing near them to
     suggest otherwise — the model would bet on these and it would be right.
     Ask the user about one and you have spent a vote confirming what you
     already believed. This is the trap in "just show them the top of the
     ranking": the top of the ranking is precisely where the model is already
     sure.

     Page 4. So it asks about this one instead — the item on the line, where
     the model genuinely cannot call it. It is the least comfortable item to be
     shown and the most valuable one to answer.

     Page 5. The user says Good. Retrain, and the boundary is somewhere else:
     dashed is where it used to be. Worth saying out loud that the other branch
     is just as real — a Bad there would have pulled the curve *in* on that
     side, tightening around the votes instead of reaching past them. One
     answer, one item, and the model's opinion about a dozen items it has never
     been shown has changed.

     Page 6. And that is the loop closing. The boundary moved, so a *different*
     item is now the one sitting on it, and that is the next thing the user
     sees. Land here: the model does not just answer the question, it chooses
     the next one — which is the point the following slides are built on. -->
