![bg right:70% fit](figs/calib-xcal-flow.png)

### Iteration 1

## Cross-calibration

- Half-vs-half models score the votes they **never saw**; average the cuts
- Right in the limit — wild when positives are scarce

<!-- Brief-deck cut of xcal + xcal-results; one slide carries the whole
     iteration, so give the mechanism in one breath and the verdict in the
     next. Mechanism (walk the figure top to bottom): train the model you
     keep on every vote; split the votes in half, train a model on each
     half, and score each half with the model that never saw it — honest
     scores — then cut each half in the gap and average the two cuts. A
     consistent estimator that is provably right with enough labels. (The
     shipped code has since polished this — pooled halves, a quantile cut
     the Inclusion knob can bias — but this slide teaches the original
     idea.) Verdict: "enough labels" is doing all the work. The cut hangs
     off tens of positives, half of them hidden from each model, and below
     about 20 votes it degenerates into "admit nothing" spikes. Plant the
     phrase "consistent but starved" — the next slide's GMM is its exact
     mirror image, and the tension between them is the talk. -->
