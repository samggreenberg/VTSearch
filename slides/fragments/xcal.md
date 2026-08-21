![bg right:56% fit](figs/calib-xcal-flow.png)

### Iteration 1 — the idea

## Cross-calibration

- Re-split the votes; train fold models; score the **held-out** side
- Pool the folds; take one **conformal quantile** cut
- Inclusion biases the quantile — never the training

<!-- This is the pre-history of the line, the textbook answer everything else
     is measured against. Walk the mechanism off the figure, top to bottom:
     the final model M0 trains on every vote, and its scores on its own
     training votes are optimistically shifted, so you cannot cut on them
     directly. Instead each round re-draws a stratified Train/Calibrate split
     of the whole labelset (not a partition — a vote can be held out twice,
     or never), trains a fold model on the Train side, and scores the
     held-out side with the model that never saw it — honest scores, at the
     price of training extra models on half the data. Green squares are Good
     votes, red are Bad, matching the checks and crosses on the pooled score
     line.

     Two design choices worth a sentence each: the folds are pooled rather
     than averaged, so the Inclusion knob keeps its resolution over the merged
     score set; and Inclusion enters only by biasing which quantile is cut,
     never by reweighting training, so the ranking is identical at every knob
     setting. Close on the property that defines the iteration: this is a
     consistent estimator — with enough labels it converges to the right
     answer. The next slide is about what happens before "enough". -->
