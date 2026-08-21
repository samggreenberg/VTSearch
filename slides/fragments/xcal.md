![bg right:56% fit](figs/calib-xcal-flow.png)

### Iteration 1 — the idea

## Cross-calibration

- Split the votes in half; train a model on each half
- Each model scores the half it **never saw** — honest scores
- Cut each half; **average** the two cuts

<!-- This is the pre-history of the line, the textbook answer everything else
     is measured against. Walk the mechanism off the figure, top to bottom:
     the model you keep, M0, trains on every vote — but its scores on its own
     training votes are optimistically shifted, so you cannot cut on them
     directly. So split the votes in half, train a model on each half, and
     have each model score the half it never trained on — honest scores, at
     the price of training extra models on half the data. The two scoring
     paths cross on the slide — that X is the "cross" in cross-calibration.
     On each half the Bad scores mostly pile up low and the Good scores high,
     but not perfectly (note the Bad that lands above θ2), so each cut is a
     trade-off, not a free gap; average the two cuts and hand θ0 to M0. Green
     is Good media, red is Bad, matching the checks and crosses on the score
     lines.

     This slide teaches the original, simplest form of the idea; the shipped
     code has since refined it (the halves are pooled into one score set
     rather than cut separately, the cut is a quantile the Inclusion knob can
     bias, the splits are re-drawn rather than a fixed partition) — those are
     later polish, not the idea, so don't front-load them here. Close on the
     property that defines the iteration: this is a consistent estimator —
     with enough labels it converges to the right answer. The next slide is
     about what happens before "enough". -->
