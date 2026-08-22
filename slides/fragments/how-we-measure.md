![bg right:56% fit](figs/cost-traces.png)

### The yardstick

## Simulated voters, thousands of runs

- Replay the real loop on **labeled** corpora — labels score it, never feed it
- Every step: **cost = *w*<sub>f</sub>·FPR + *w*<sub>n</sub>·FNR**
- **Regret** = that cost, minus the best cut available in hindsight

<!-- Spend a minute here; every number in the rest of the deck is in these
     units, and a room that does not have them will hear "minus 0.074" as
     noise.

     The method: take a corpus that does have ground-truth labels, hide them,
     and run the actual application loop against a simulated voter who answers
     the way the labels say. The labels are the yardstick and never an input —
     nothing in the system sees them. A run is 150 votes, which is roughly
     what a real session is worth; a study is a sweep of runs across datasets,
     embedding models and both voting modes, launched as hundreds of cells on
     a cluster.

     Two quantities. Cost is a weighted sum of the two error rates — false
     positives and false negatives — measured at whatever cut the system chose
     at that step. The weights are exposed to the user, as an Inclusion
     setting: someone who cannot afford to miss anything and someone who
     cannot afford noise want different lines, and the same estimator has to
     serve both. Regret is the honest version: the same cost minus what the
     best possible cut on that same ranking would have scored. Regret isolates
     the threshold from the model — it is zero when the line is perfectly
     placed, however good or bad the ranking underneath happens to be.

     Point at the figure: each thin trace is one run's cost as votes
     accumulate, the heavy line is the median. Two things to note — it falls,
     which is the system working, and the spread is enormous, which is why
     nothing in this talk is argued from a single run. -->
