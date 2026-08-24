<!-- _class: full -->

![bg fit](figs/cost-traces.png)

## Simulated Voters, Thousands of Runs

<!-- Backup, and the units of every number in the talk. We replay the real loop
     on labelled corpora, where the labels score the run and never feed it. A
     run is 150 votes, roughly what a real session is worth; a study is a sweep
     across datasets, embedding models and both voting modes, launched as
     hundreds of cells on a cluster. -->

<!-- Two quantities. **Cost** is a weighted sum of the two error rates at
     whatever cut the system chose — the same weights the Inclusion setting
     exposes. **Regret** is that cost minus the best cut available in hindsight;
     it isolates the threshold from the model, and is zero when the line is
     perfectly placed however good the ranking underneath happens to be. -->

<!-- The figure is drawn at equal weights, which is why its axis reads simply
     FPR + FNR. Each thin trace is one run's cost as votes accumulate; the heavy
     line is the median. Two things to note — it falls, which is the system
     working, and the spread is enormous, which is why nothing in this talk is
     argued from a single run. -->
