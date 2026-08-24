<!-- _class: full -->

![bg fit](figs/positive-starvation.png)

## Positives Are the Binding Constraint

<!-- Backup, and the headline empirical fact behind four of the five iterations.
     Votes are cheap — a click each, and the system keeps asking. Positives are
     not: the user is hunting a concept that occupies a fraction of a percent of
     the corpus, so most of what comes back, especially early, is a Bad. -->

<!-- The plot counts only the Good votes. After 150 votes the median run is
     holding between four and eleven of them. The dotted line is what the count
     would look like if every vote were a positive. One trace sat on three
     positives for a hundred and twenty straight votes, and about one run in
     twenty-seven finished having never seen a single one. -->

<!-- The general conclusion, stated explicitly: any part of this system that
     learns from labelled data is starved by this, not limited by its own
     cleverness. An estimator whose variance scales with the number of positives
     will be at its wildest at exactly the moment the user is watching most
     closely — the first thirty seconds. -->
