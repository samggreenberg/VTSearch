### Iteration 1 — the idea

## Cross-calibration

- Split the votes; train fold models; score each **held-out** half
- Pool the folds; take one **conformal quantile** cut
- Inclusion biases the quantile — never the training

<!-- This is the pre-history of the line, the textbook answer everything else
     is measured against. Walk the mechanism: the final model's scores on its
     own training votes are optimistically shifted, so you cannot cut on them
     directly. Instead split the votes into two stratified folds, train a
     model per fold, and score each vote with the model that never saw it —
     honest scores, at the price of training extra models on half the data.

     Two design choices worth a sentence each: the folds are pooled rather
     than averaged, so the Inclusion knob keeps its resolution over the merged
     score set; and Inclusion enters only by biasing which quantile is cut,
     never by reweighting training, so the ranking is identical at every knob
     setting. Close on the property that defines the iteration: this is a
     consistent estimator — with enough labels it converges to the right
     answer. The next slide is about what happens before "enough". -->
