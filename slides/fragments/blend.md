### Iteration 3 — the idea

## Average the rivals

- `threshold = w·xcal + (1−w)·gmm`
- `w` ramps with votes: pure GMM ≤ 6, pure x-cal ≥ 20
- The GMM absorbs the cold start; the labels take over

<!-- The two estimators fail in opposite regimes — x-cal is starved early and
     right late, the GMM is well-fed early and biased forever — so the obvious
     move is a weighted average of the two *thresholds*, with the weight
     ramping along the vote count. Six votes or fewer, trust the mixture
     entirely; twenty or more, hand over to the labels; interpolate linearly
     between. This shipped as "safe thresholds" (#2798, #2799).

     Be honest about how crude the first version was, because it sets up
     iteration 3½: one hard-coded line, w = clip((n−6)/14, 0, 1), with three
     unmeasured choices baked in — the endpoints of the ramp, the shape of the
     ramp, and the statistic driving it (raw click count, when the thing that
     actually matters is positives). Cheap, shippable, and the single biggest
     win in the line, which is the next slide. -->
