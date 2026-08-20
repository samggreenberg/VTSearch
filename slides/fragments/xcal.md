### Iteration 1 — the idea

## Cross-calibration

- Split the votes; train fold models; score each **held-out** half
- Pool the folds; take one **conformal quantile** cut
- Inclusion biases the quantile — never the training

<!-- The starting point, pre-history of the line. Two folds, stratified,
     pooled rather than averaged so the Inclusion knob keeps resolution.
     Consistent estimator: with enough labels this is the right answer. -->
