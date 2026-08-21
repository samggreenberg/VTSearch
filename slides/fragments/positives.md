![bg right:56% fit](figs/positive-starvation.png)

### Finding 2

## Positives are the binding constraint

- After **150 votes**: a median of only **4–11** positives
- One trace sat at **3 positives for 120 votes**
- **3.7%** of runs never found a single one

<!-- This is the headline empirical fact of the whole line; give it room.
     Voting is cheap but positives are rare, so labels accumulate lopsidedly:
     after 150 votes the median run holds only 4 to 11 positives, and the
     tails are brutal — point at the trace that sat on 3 positives for 120
     straight votes, and note that about one run in twenty-seven never found a
     single positive at all.

     Draw the general conclusion explicitly: every labeled-data component
     downstream — calibration, cut rules, even the choice of model head — is
     starved by this, not limited by its own cleverness. Any estimator whose
     variance scales with the number of positives is going to be wild exactly
     when the user is watching most closely, at the start. That is the case
     for bringing in a data source that cannot starve, which is the next
     iteration. -->
