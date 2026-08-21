<!-- _class: statement -->

# A detector *ranks*. The threshold decides what you see.

## cost = *w*<sub>f</sub>·FPR + *w*<sub>n</sub>·FNR — and at 6 votes you may hold one positive.

<!-- Separate the two jobs cleanly, because the whole deck lives in the gap
     between them. Training produces a ranking of the pool; the audience never
     experiences the ranking directly. What they experience is the threshold:
     which items are shown, exported, auto-labeled. A perfect ranking with a
     bad cut still looks broken to the user.

     Define the cost once, here, and note that the Inclusion knob is exactly a
     reweighting of w_f against w_n — every study in this deck scores this
     same cost, so "better" always means the same thing. Then land the second
     line hard: the threshold must be estimated from the user's own votes, and
     six votes deep you may be holding one positive example. Everything that
     follows is a fight against that scarcity. -->
