### Iteration 3 — the idea

## Average the rivals

- `threshold = w·xcal + (1−w)·gmm`
- `w` ramps with votes: pure GMM ≤ 6, pure x-cal ≥ 20
- The GMM absorbs the cold start; the labels take over

<!-- "Safe thresholds", #2798/#2799. One hard-coded line at first:
     w = clip((n-6)/14, 0, 1). Three unmeasured choices baked in — endpoints,
     shape, statistic — which is what iteration 3½ sweeps. -->
