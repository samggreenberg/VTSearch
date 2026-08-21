![bg right:56% fit](figs/calib-xcal-flow.png)

### Iteration 1

## Cross-calibration

- Pooled held-out folds, one conformal quantile cut
- Right in the limit — wild when positives are scarce

<!-- Brief-deck cut of xcal + xcal-results; one slide carries the whole
     iteration, so give the mechanism in one breath and the verdict in the
     next. Mechanism (walk the figure top to bottom): train the final model
     on every vote; per round, re-draw a stratified split, train a fold
     model, score the votes it never saw; pool the held-out scores and cut
     one conformal quantile — honest scores, and a consistent estimator that
     is provably right with enough labels. Verdict: "enough labels" is doing all
     the work. The cut is a low quantile over tens of positives, folds redraw
     every vote, and below about 20 votes it degenerates into "admit nothing"
     spikes. Plant the phrase "consistent but starved" — the next slide's GMM
     is its exact mirror image, and the tension between them is the talk. -->
