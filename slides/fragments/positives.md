![bg right:56% fit](figs/positive-starvation.png)

### Why this is hard

## Positives are the binding constraint

- After **150 votes**: a median of only **4–11** positives
- One trace sat at **3 positives for 120 votes**
- **3.7%** of runs never found a single one

<!-- This is the headline empirical fact of the whole line; give it room, and
     make sure the room understands what is being counted.

     Votes are cheap — a click each, and the system keeps asking. Positives
     are not: the user is hunting a concept that occupies a fraction of a
     percent of the corpus, so most of what comes back, especially early, is
     a Bad. The plot counts only the Good votes. After 150 votes the median
     run is holding between four and eleven of them. Point at the dotted line
     for scale: that is what the count would look like if every vote were a
     positive. Point at the flat trace that sat on three positives for a
     hundred and twenty straight votes, and note that about one run in
     twenty-seven finished having never seen a single one.

     Then draw the general conclusion explicitly, because it is the premise of
     everything that follows: any part of this system that learns from
     labelled data is starved by this, not limited by its own cleverness. An
     estimator whose variance scales with the number of positives will be at
     its wildest at exactly the moment the user is watching most closely — the
     first thirty seconds. Hold that thought; it is the reason four of the
     five iterations in this deck exist. -->
